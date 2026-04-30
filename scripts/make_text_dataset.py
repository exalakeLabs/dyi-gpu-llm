#!/usr/bin/env python
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p

IN_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")
DATA_DIR = env_dir("LLAMA_DATA_DIR", "data")
OUT_FILE = DATA_DIR / "train.jsonl"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def chunks(text, max_chars=2500):
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    buf, size = [], 0

    for p in paras:
        if size + len(p) > max_chars and buf:
            yield "\n\n".join(buf)
            buf, size = [], 0
        buf.append(p)
        size += len(p)

    if buf:
        yield "\n\n".join(buf)

with OUT_FILE.open("w", encoding="utf-8") as out:
    for file in sorted(IN_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunks(text):
            out.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

print(f"Wrote {OUT_FILE}")
