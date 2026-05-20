#!/usr/bin/env python

from __future__ import annotations

import re
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
LINE_BREAK_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
SEPARATOR_LINE_RE = re.compile(r"[-=_*]{4,}")


def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p

IN_DIR = env_dir("LLAMA_TEXT_DIR", "text")
OUT_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(IN_DIR.glob("*.txt")):
        clean_file(txt_file, OUT_DIR)
        print(f"Cleaned {txt_file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
