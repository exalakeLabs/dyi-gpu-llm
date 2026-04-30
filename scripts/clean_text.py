#!/usr/bin/env python

import re
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p

IN_DIR = env_dir("LLAMA_TEXT_DIR", "text")
OUT_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

for txt_file in sorted(IN_DIR.glob("*.txt")):
    raw = txt_file.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    out_file = OUT_DIR / txt_file.name
    out_file.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned {txt_file.name}")