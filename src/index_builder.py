#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Delegate SSL verification to the macOS Security framework so the Sigma
# corporate proxy CA (which lacks keyUsage, causing Python 3.14/OpenSSL 3.x
# to reject it) is trusted via the system keychain instead of certifi.
# Must run before any import that touches ssl (sentence_transformers, httpx…).
try:
    import truststore

    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful Tolkien lore assistant. Use the retrieved canon context "
    "carefully, distinguish certainty from inference, and avoid inventing details."
)
START_MARKERS = [
    "START OF THE PROJECT GUTENBERG EBOOK",
    "START OF THIS PROJECT GUTENBERG EBOOK",
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
]
END_MARKERS = [
    "END OF THE PROJECT GUTENBERG EBOOK",
    "END OF THIS PROJECT GUTENBERG EBOOK",
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
]
SECTION_HEADING_RE = re.compile(
    r"^(?P<heading>(?:BOOK|PART|CHAPTER|APPENDIX)\b[\w .,'’:-]*|[IVXLCDM]+\.\s+.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    section_path: List[str]


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines[:500]):
        upper_line = line.upper()
        if any(marker in upper_line for marker in START_MARKERS):
            start_idx = i + 1
            break

    end_idx = len(lines)
    tail_start = max(0, len(lines) - 1000)
    for i, line in enumerate(lines[tail_start:], start=tail_start):
        upper_line = line.upper()
        if any(marker in upper_line for marker in END_MARKERS):
            end_idx = i
            break

    stripped = "\n".join(lines[start_idx:end_idx]).strip()
    return stripped if stripped else text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
            continue

        blank_run = 0
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n[-=_*]{4,}\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_source_text(text: str, strip_gutenberg: bool) -> str:
    if strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    return clean_text(text)


def is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 140:
        return False
    if SECTION_HEADING_RE.match(line):
        return True
    # Catch short all-caps literary headings while avoiding OCR noise.
    letters = [ch for ch in line if ch.isalpha()]
    if 4 <= len(letters) and len(line) <= 80 and line.upper() == line:
        return True
    return False


def section_path_at(text: str, char_pos: int, max_depth: int = 4) -> List[str]:
    section_path: List[str] = []
    for line in text[:char_pos].splitlines():
        heading = line.strip()
        if not is_section_heading(heading):
            continue
        if heading in section_path:
            section_path = section_path[: section_path.index(heading) + 1]
            continue
        section_path.append(heading)
        section_path = section_path[-max_depth:]
    return section_path


def choose_chunk_end(text: str, start: int, target_end: int, min_end: int, max_end: int) -> int:
    """Choose a natural chunk end near target_end without exceeding max_end."""
    target_end = max(min_end, min(target_end, max_end))
    candidates: List[int] = []

    for pattern in (r"\n\s*\n", r'[.!?]["\')\]]?\s+', r"\s+"):
        for match in re.finditer(pattern, text[start:max_end]):
            pos = start + match.end()
            if min_end <= pos <= max_end:
                candidates.append(pos)

    if candidates:
        return min(candidates, key=lambda pos: abs(pos - target_end))
    return target_end


def chunk_text(
    text: str,
    chunk_size_chars: int = 1800,
    overlap_chars: int = 250,
    min_chunk_chars: int = 120,
) -> List[TextChunk]:
    text = clean_text(text).strip()
    if not text:
        return []

    chunks: List[TextChunk] = []
    start = 0
    n = len(text)

    while start < n:
        max_end = min(start + chunk_size_chars, n)
        min_end = min(max(start + min_chunk_chars, start + 1), max_end)
        end = choose_chunk_end(
            text=text,
            start=start,
            target_end=max_end,
            min_end=min_end,
            max_end=max_end,
        )

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(
                TextChunk(
                    text=chunk,
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=end,
                    section_path=section_path_at(text, start),
                )
            )

        if end >= n:
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def extract_title_author(path: Path, text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    author = ""

    for line in lines[:80]:
        low = line.lower()
        if low.startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
        elif low.startswith("author:"):
            author = line.split(":", 1)[1].strip()
        elif not title and len(line) <= 140:
            title = line
        if title and author:
            break

    if title == path.stem.replace("_", " ").replace("-", " ").strip():
        for line in lines[:20]:
            if len(line) <= 140 and not line.lower().startswith(("produced by ", "transcribed by ")):
                title = line
                break

    return {"title": title, "author": author}


def slugify(value: str, fallback: str = "source") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def stable_source_id(relative_path: Path) -> str:
    digest = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(relative_path.stem)}-{digest}"


def chunk_to_metadata(
    *,
    row_id: int,
    chunk: TextChunk,
    file_path: Path,
    input_dir: Path,
    source_id: str,
    title: str,
    author: str,
    system_prompt: str,
) -> Dict[str, Any]:
    relative_path = file_path.relative_to(input_dir)
    chunk_id = f"{source_id}_{chunk.chunk_index:05d}"
    section_path = chunk.section_path

    return {
        # Compatibility with existing chat scripts that expect positional ids and source fields.
        "id": row_id,
        "source": relative_path.name,
        "source_path": str(file_path),
        "source_relpath": str(relative_path),
        # Evidence-oriented metadata for generation -> verification -> filtering pipelines.
        "source_id": source_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk.chunk_index,
        "book": title,
        "title": title,
        "author": author,
        "part": section_path[0] if section_path else "",
        "chapter": section_path[-1] if section_path else "",
        "section_path": section_path,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "system_prompt": system_prompt,
        "text": chunk.text,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a FAISS index and evidence-chunk manifest for RAG-backed "
            "lore Q/A generation and verification."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt books")
    parser.add_argument("--output-dir", required=True, help="Directory for FAISS index + metadata")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunk-size-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--min-chunk-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    parser.add_argument(
        "--no-strip-gutenberg",
        action="store_true",
        help="Keep Project Gutenberg headers/footers instead of stripping them.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Grounding prompt stored with each evidence chunk for downstream pair generation.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_text_files(input_dir))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    print(f"Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    all_meta: List[Dict[str, Any]] = []
    batch_chunks: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    all_embeddings = []

    for file_path in tqdm(files, desc="Reading books"):
        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[skip] {file_path}: {e}")
            continue

        text = prepare_source_text(raw_text, strip_gutenberg=not args.no_strip_gutenberg)
        info = extract_title_author(file_path, text)
        relative_path = file_path.relative_to(input_dir)
        source_id = stable_source_id(relative_path)
        chunks = chunk_text(
            text,
            chunk_size_chars=args.chunk_size_chars,
            overlap_chars=args.overlap_chars,
            min_chunk_chars=args.min_chunk_chars,
        )

        for chunk in chunks:
            meta = chunk_to_metadata(
                row_id=len(all_meta) + len(batch_meta),
                chunk=chunk,
                file_path=file_path,
                input_dir=input_dir,
                source_id=source_id,
                title=info["title"],
                author=info["author"],
                system_prompt=args.system_prompt,
            )
            batch_chunks.append(chunk.text)
            batch_meta.append(meta)

            if len(batch_chunks) >= args.batch_size * 16:
                embs = embed_texts(embedder, batch_chunks, batch_size=args.batch_size)
                all_embeddings.append(embs)
                all_meta.extend(batch_meta)
                batch_chunks = []
                batch_meta = []

    if batch_chunks:
        embs = embed_texts(embedder, batch_chunks, batch_size=args.batch_size)
        all_embeddings.append(embs)
        all_meta.extend(batch_meta)

    if not all_embeddings:
        raise SystemExit("No chunks were embedded.")

    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]

    # IndexHNSWFlat: approximate nearest-neighbour graph index.
    # ~10-50× faster at query time than IndexFlatIP with <1% quality loss.
    # M=32 (edges per node) is a good default; raise to 64 for higher recall.
    # No training step required — safe to add vectors immediately.
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "embed_model": args.embed_model,
        "index_type": "faiss.IndexHNSWFlat(M=32)",
        "normalize_embeddings": True,
        "chunk_size_chars": args.chunk_size_chars,
        "overlap_chars": args.overlap_chars,
        "min_chunk_chars": args.min_chunk_chars,
        "strip_gutenberg": not args.no_strip_gutenberg,
        "system_prompt": args.system_prompt,
        "metadata_schema": "evidence_chunks.v1",
        "num_files": len(files),
        "num_chunks": len(all_meta),
        "dim": dim,
    }
    (output_dir / "index_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved index to: {output_dir / 'index.faiss'}")
    print(f"Saved metadata to: {output_dir / 'chunks.jsonl'}")
    print(f"Indexed files: {len(files)}")
    print(f"Indexed chunks: {len(all_meta)}")


if __name__ == "__main__":
    main()
