"""
Build a supervised fine-tuning corpus from PDFs in ./pdfs.

Outputs JSONL rows containing a single `text` field (chat-style), compatible with
the current `scripts/train_lora.py` which loads JSON and trains on `text`.

Pipeline:
  1) Extract PDFs -> text/*.txt
  2) Clean text -> prepared/*.txt
  3) Create instruction/response pairs -> data/train_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _optional_import_transformers():
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer
    except Exception:
        return None


def extract_pdf_to_text(pdf_path: Path) -> str:
    from pypdf import PdfReader  # type: ignore

    try:
        reader = PdfReader(str(pdf_path))

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return ""

        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                pages.append(f"\n[ERROR page {i+1}: {e}]\n")
        return "\n\n".join(pages)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # de-hyphenate line breaks like "architec-\nture"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # join wrapped lines within paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # normalize spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def title_from_filename(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip().title()


def split_into_chunks(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for p in paragraphs:
        if cur_len + len(p) + 2 > max_chars and cur:
            chunk = "\n\n".join(cur).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2

    if cur:
        chunk = "\n\n".join(cur).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    return chunks


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if len(s.strip()) >= 20]


def heuristic_summary_bullets(chunk: str, *, bullets: int = 4) -> str:
    sents = split_sentences(chunk)
    if not sents:
        return "- (no extractable text)"
    picked = sents[: max(1, bullets)]
    picked = picked[:bullets]
    return "\n".join(f"- {s}" for s in picked)


_REQ_RE = re.compile(
    r"\b(must|shall|required|requirement|prohibited|may not|not allowed|need to)\b",
    re.IGNORECASE,
)


def heuristic_requirements(chunk: str, *, max_items: int = 6) -> str:
    sents = split_sentences(chunk)
    req = [s for s in sents if _REQ_RE.search(s)]
    if not req:
        req = sents[: min(len(sents), max(2, max_items // 2))]
    req = req[:max_items]
    return "\n".join(f"- {s}" for s in req)


def make_prefix_suffix_pair(chunk: str, *, min_prefix: int = 250) -> Optional[tuple[str, str]]:
    # Completion-style example that is always "correct":
    # predict the continuation of the excerpt.
    if len(chunk) < min_prefix + 200:
        return None
    cut = max(min_prefix, int(len(chunk) * 0.55))
    # cut on a whitespace boundary
    while cut < len(chunk) and cut > min_prefix and chunk[cut - 1] not in {" ", "\n", "\t"}:
        cut -= 1
    prefix = chunk[:cut].rstrip()
    suffix = chunk[cut:].lstrip()
    if len(prefix) < min_prefix or len(suffix) < 120:
        return None
    return prefix, suffix


@dataclass(frozen=True)
class Pair:
    instruction: str
    context: str
    response: str


def build_pairs_for_chunk(title: str, chunk: str) -> list[Pair]:
    pairs: list[Pair] = []

    pairs.append(
        Pair(
            instruction=f"Summarize the following excerpt from '{title}' in 4 bullet points.",
            context=chunk,
            response=heuristic_summary_bullets(chunk, bullets=4),
        )
    )

    pairs.append(
        Pair(
            instruction=(
                f"Extract the key requirements, prohibitions, or obligations stated in this excerpt from '{title}'. "
                "Return them as a bullet list."
            ),
            context=chunk,
            response=heuristic_requirements(chunk, max_items=6),
        )
    )

    ps = make_prefix_suffix_pair(chunk)
    if ps:
        prefix, suffix = ps
        pairs.append(
            Pair(
                instruction=(
                    f"Continue the excerpt from '{title}' verbatim. "
                    "Do not add commentary; output only the continuation text."
                ),
                context=prefix,
                response=suffix,
            )
        )

    return pairs


def render_chat_text(
    *,
    model_id: Optional[str],
    system_prompt: str,
    instruction: str,
    context: str,
    response: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n\nEXCERPT:\n{context}"},
        {"role": "assistant", "content": response},
    ]

    AutoTokenizer = _optional_import_transformers()
    if model_id and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
        except Exception:
            pass

    # Fallback: readable, model-agnostic chat format
    return (
        f"### System\n{system_prompt}\n\n"
        f"### User\n{instruction}\n\nEXCERPT:\n{context}\n\n"
        f"### Assistant\n{response}\n"
    )


def iter_prepared_texts(prepared_dir: Path) -> Iterable[tuple[Path, str]]:
    for f in sorted(prepared_dir.glob("*.txt")):
        yield f, f.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--text-dir", default="text")
    ap.add_argument("--prepared-dir", default="prepared")
    ap.add_argument("--out", default="data/train_pairs.jsonl")
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
    ap.add_argument(
        "--system",
        default="You are a concise technical assistant. Prefer precise definitions, direct answers, and clear structure.",
    )
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chunks-per-doc", type=int, default=30)
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    text_dir = Path(args.text_dir)
    prepared_dir = Path(args.prepared_dir)
    out_path = Path(args.out)

    text_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Extract PDFs
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    extracted = 0
    for pdf in pdfs:
        raw = extract_pdf_to_text(pdf)
        (text_dir / f"{pdf.stem}.txt").write_text(raw, encoding="utf-8")
        extracted += 1

    # 2) Clean text
    cleaned = 0
    for txt_path, raw in iter_prepared_texts(text_dir):
        out = prepared_dir / txt_path.name
        out.write_text(clean_text(raw), encoding="utf-8")
        cleaned += 1

    # 3) Build pairs
    written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for f, doc in iter_prepared_texts(prepared_dir):
            title = title_from_filename(f.name)
            chunks = split_into_chunks(doc, max_chars=args.max_chars, min_chars=args.min_chars)
            chunks = chunks[: args.max_chunks_per_doc]
            for chunk in chunks:
                pairs = build_pairs_for_chunk(title, chunk)
                for p in pairs:
                    row = {
                        "text": render_chat_text(
                            model_id=args.model_id,
                            system_prompt=args.system,
                            instruction=p.instruction,
                            context=p.context,
                            response=p.response,
                        ),
                        "source_file": f.name,
                        "title": title,
                        "kind": "pair",
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Extracted PDFs: {extracted}")
    print(f"Cleaned texts: {cleaned}")
    print(f"Wrote pairs: {written}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

