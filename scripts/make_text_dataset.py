import json
from pathlib import Path

IN_DIR = Path("prepared")
OUT_FILE = Path("data/train.jsonl")
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
