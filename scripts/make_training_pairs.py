#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple

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

REPO_ROOT = Path(__file__).resolve().parents[1]
def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


OUT_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build continuation-style training pairs from Gutenberg .txt books."
    )
    p.add_argument("--text-dir", default=OUT_DIR, help="Directory containing .txt files")
    p.add_argument(
        "--output-train",
        default="data/gutenberg_train.jsonl",
        help="Output JSONL for training split",
    )
    p.add_argument(
        "--output-val",
        default="data/gutenberg_val.jsonl",
        help="Output JSONL for validation split",
    )
    p.add_argument(
        "--glob",
        default="*.txt",
        help='Glob pattern under --text-dir (default: "*.txt")',
    )
    p.add_argument(
        "--format",
        choices=["messages", "prompt_completion"],
        default="messages",
        help="Output schema",
    )
    p.add_argument(
        "--system-prompt",
        default="Continue the passage faithfully, preserving its meaning, tone, and style.",
        help="System prompt used when --format messages",
    )
    p.add_argument(
        "--user-template",
        default=(
            "Continue the following passage from the book.\n\n"
            "Title: {title}\n\n"
            "{prompt}"
        ),
        help="User prompt template for messages format",
    )
    p.add_argument(
        "--min-book-chars",
        type=int,
        default=4000,
        help="Skip books shorter than this after cleaning",
    )
    p.add_argument(
        "--target-window-chars",
        type=int,
        default=2600,
        help="Target total chars per training example",
    )
    p.add_argument(
        "--max-window-chars",
        type=int,
        default=3200,
        help="Hard max total chars per example",
    )
    p.add_argument(
        "--prompt-chars",
        type=int,
        default=900,
        help="Target chars for the prompt side",
    )
    p.add_argument(
        "--min-prompt-chars",
        type=int,
        default=500,
        help="Minimum chars allowed for the prompt",
    )
    p.add_argument(
        "--min-response-chars",
        type=int,
        default=350,
        help="Minimum chars allowed for the response",
    )
    p.add_argument(
        "--stride-chars",
        type=int,
        default=1200,
        help="Step size between windows",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="Train split ratio",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return p.parse_args()


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines[:500]):
        u = line.upper()
        if any(marker in u for marker in START_MARKERS):
            start_idx = i + 1
            break

    end_idx = len(lines)
    for i, line in enumerate(lines[max(0, len(lines) - 1000):], start=max(0, len(lines) - 1000)):
        u = line.upper()
        if any(marker in u for marker in END_MARKERS):
            end_idx = i
            break

    stripped = "\n".join(lines[start_idx:end_idx]).strip()
    return stripped if stripped else text


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")

    # Normalize whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
        else:
            blank_run = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    # Remove obvious scan noise / repeated separators
    text = re.sub(r"\n[-=_*]{4,}\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:20]:
        s = line.strip()
        if not s:
            continue
        if len(s) <= 140 and not s.lower().startswith("produced by "):
            return s
    return path.stem.replace("_", " ").replace("-", " ").strip()


def choose_boundary(text: str, target: int, min_pos: int, max_pos: int) -> int:
    """
    Choose a reasonably natural cut point near target.
    Preference:
      1) paragraph breaks
      2) sentence-ish endings
      3) whitespace
      4) hard cut
    """
    n = len(text)
    min_pos = max(1, min(min_pos, n))
    max_pos = max(min_pos, min(max_pos, n))
    target = max(min_pos, min(target, max_pos))

    candidates: List[int] = []

    for m in re.finditer(r"\n\s*\n", text):
        pos = m.end()
        if min_pos <= pos <= max_pos:
            candidates.append(pos)

    for m in re.finditer(r'[.!?]["\')\]]?\s+', text):
        pos = m.end()
        if min_pos <= pos <= max_pos:
            candidates.append(pos)

    for m in re.finditer(r"\s+", text):
        pos = m.end()
        if min_pos <= pos <= max_pos:
            candidates.append(pos)

    if candidates:
        return min(candidates, key=lambda x: abs(x - target))

    return target


def build_segment(text: str, start: int, target_len: int, max_len: int) -> Tuple[str, int]:
    remaining = text[start:]
    if not remaining:
        return "", start

    raw = remaining[:max_len]
    if len(raw) < target_len:
        return raw.strip(), start + len(raw)

    end_rel = choose_boundary(
        raw,
        target=target_len,
        min_pos=max(200, int(target_len * 0.75)),
        max_pos=min(len(raw), max_len),
    )
    seg = raw[:end_rel].strip()
    return seg, start + end_rel


def iter_examples(
    text: str,
    title: str,
    source_name: str,
    target_window_chars: int,
    max_window_chars: int,
    prompt_chars: int,
    min_prompt_chars: int,
    min_response_chars: int,
    stride_chars: int,
) -> Iterable[dict]:
    text_len = len(text)
    start = 0
    example_idx = 0

    while start + min_prompt_chars + min_response_chars <= text_len:
        segment, seg_end = build_segment(
            text=text,
            start=start,
            target_len=target_window_chars,
            max_len=max_window_chars,
        )

        if len(segment) < (min_prompt_chars + min_response_chars):
            break

        split_at = choose_boundary(
            segment,
            target=prompt_chars,
            min_pos=min_prompt_chars,
            max_pos=max(len(segment) - min_response_chars, min_prompt_chars),
        )

        prompt = segment[:split_at].strip()
        response = segment[split_at:].strip()

        if len(prompt) >= min_prompt_chars and len(response) >= min_response_chars:
            yield {
                "title": title,
                "source_file": source_name,
                "example_index": example_idx,
                "prompt": prompt,
                "response": response,
                "prompt_chars": len(prompt),
                "response_chars": len(response),
            }
            example_idx += 1

        start += stride_chars


def to_record(example: dict, fmt: str, system_prompt: str, user_template: str) -> dict:
    if fmt == "prompt_completion":
        return {
            "prompt": example["prompt"],
            "completion": example["response"],
            "meta": {
                "title": example["title"],
                "source_file": example["source_file"],
                "example_index": example["example_index"],
                "prompt_chars": example["prompt_chars"],
                "response_chars": example["response_chars"],
            },
        }

    user_content = user_template.format(
        title=example["title"],
        prompt=example["prompt"],
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["response"]},
        ],
        "meta": {
            "title": example["title"],
            "source_file": example["source_file"],
            "example_index": example["example_index"],
            "prompt_chars": example["prompt_chars"],
            "response_chars": example["response_chars"],
        },
    }


def load_and_clean_book(path: Path) -> Tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = strip_gutenberg_boilerplate(raw)
    cleaned = normalize_text(raw)
    title = detect_title(path, cleaned)
    return title, cleaned


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    text_dir = Path(args.text_dir)
    files = sorted(text_dir.rglob(args.glob))

    if not files:
        raise SystemExit(f"No files matched {args.glob!r} under {text_dir}")

    all_examples: List[dict] = []
    kept_books = 0
    skipped_books = 0

    for path in files:
        try:
            title, text = load_and_clean_book(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            skipped_books += 1
            continue

        if len(text) < args.min_book_chars:
            skipped_books += 1
            continue

        book_examples = list(
            iter_examples(
                text=text,
                title=title,
                source_name=str(path.relative_to(text_dir)),
                target_window_chars=args.target_window_chars,
                max_window_chars=args.max_window_chars,
                prompt_chars=args.prompt_chars,
                min_prompt_chars=args.min_prompt_chars,
                min_response_chars=args.min_response_chars,
                stride_chars=args.stride_chars,
            )
        )

        if not book_examples:
            skipped_books += 1
            continue

        all_examples.extend(book_examples)
        kept_books += 1
        print(f"[OK] {path.name}: {len(book_examples)} examples")

    if not all_examples:
        raise SystemExit("No training examples were created.")

    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    records = [
        to_record(
            example=ex,
            fmt=args.format,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
        )
        for ex in all_examples
    ]

    split_idx = int(len(records) * args.train_ratio)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    output_train = Path(args.output_train)
    output_val = Path(args.output_val)

    write_jsonl(output_train, train_records)
    write_jsonl(output_val, val_records)

    print()
    print(f"Books kept:    {kept_books}")
    print(f"Books skipped: {skipped_books}")
    print(f"Examples:      {len(records)}")
    print(f"Train:         {len(train_records)} -> {output_train}")
    print(f"Val:           {len(val_records)} -> {output_val}")


if __name__ == "__main__":
    main()