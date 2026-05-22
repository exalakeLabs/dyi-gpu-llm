#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
from pathlib import Path

from runtime_env import env_path

PREPARED_DIR = env_path("PREPARED_DIR", "prepared")
RAWTEXT_DIR = env_path("RAWTEXT_DIR", "text")

INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
LINE_BREAK_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
SEPARATOR_LINE_RE = re.compile(r"[-=_*]{4,}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # de-hyphenate line breaks like "architec-\nture"
    text = LINE_BREAK_HYPHEN_RE.sub(r"\1\2", text)

    paragraphs: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
            paragraph_lines.clear()

    for raw_line in text.split("\n"):
        line = INLINE_WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            flush_paragraph()
            continue
        if SEPARATOR_LINE_RE.fullmatch(line):
            flush_paragraph()
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return "\n".join(paragraphs).strip()


def clean_file(txt_file: Path, out_dir: Path) -> Path:
    raw = txt_file.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    out_file = out_dir / txt_file.name
    out_file.write_text(cleaned, encoding="utf-8")
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean text/*.txt into prepared/*.txt.")
    parser.add_argument(
        "--input-dir",
        default=str(RAWTEXT_DIR),
        help="Directory containing raw .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PREPARED_DIR),
        help="Directory for cleaned .txt files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(input_dir.glob("*.txt")):
        clean_file(txt_file, output_dir)
        print(f"Cleaned {txt_file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
