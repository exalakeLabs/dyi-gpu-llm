import re
from pathlib import Path

IN_DIR = Path("text")
OUT_DIR = Path("prepared")
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