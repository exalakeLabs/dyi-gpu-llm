import json
import re
from pathlib import Path

IN_DIR = Path("prepared")
OUT_FILE = Path("data/train_auto.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are a concise technical assistant. Prefer precise definitions, direct answers, and clear structure."

def split_into_chunks(text: str, max_chars: int = 1800):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = []
    cur_len = 0

    for p in paragraphs:
        if cur_len + len(p) + 2 > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def title_from_filename(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"[_\-]+", " ", base)
    return base.strip().title()

with OUT_FILE.open("w", encoding="utf-8") as out:
    for file in sorted(IN_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        title = title_from_filename(file.name)
        chunks = split_into_chunks(text)

        for chunk in chunks[:20]:
            if len(chunk) < 300:
                continue

            examples = [
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Summarize this document section from '{title}' in 4 bullet points:\n\n{chunk}"},
                        {"role": "assistant", "content": f"Summary of '{title}':\n- ...\n- ...\n- ...\n- ..."}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Explain the following section from '{title}' in plain English:\n\n{chunk}"},
                        {"role": "assistant", "content": "This section says that ..."}
                    ]
                }
            ]

            for ex in examples:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {OUT_FILE}")