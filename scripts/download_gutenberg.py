import argparse
import re
from pathlib import Path
from typing import Optional

import requests

API_BASE = "https://gutendex.com/books"
OUT_DIR = Path("gutenberg_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_TEXT_KEYS = [
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
    "text/plain",
]

def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\s.-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:120] or "book"

def pick_text_url(formats: dict) -> Optional[str]:
    for key in PREFERRED_TEXT_KEYS:
        if key in formats:
            return formats[key]
    for k, v in formats.items():
        if k.startswith("text/plain"):
            return v
    return None

def iter_books(query: str, topic: Optional[str], language: str, max_books: int):
    params = {
        "search": query,
        "languages": language,
        "mime_type": "text/plain",
    }
    if topic:
        params["topic"] = topic

    next_url = API_BASE
    yielded = 0

    while next_url and yielded < max_books:
        resp = requests.get(next_url, params=params if next_url == API_BASE else None, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for book in data.get("results", []):
            yield book
            yielded += 1
            if yielded >= max_books:
                return

        next_url = data.get("next")
        params = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search query, e.g. 'Sherlock Holmes'")
    parser.add_argument("--topic", default=None, help="Optional subject/topic filter")
    parser.add_argument("--language", default="en", help="Language code, default: en")
    parser.add_argument("--max-books", type=int, default=20, help="Maximum number of books to download")
    args = parser.parse_args()

    count = 0
    skipped = 0

    for book in iter_books(args.query, args.topic, args.language, args.max_books):
        title = book.get("title", "unknown_title")
        book_id = book.get("id")
        authors = ", ".join(a.get("name", "Unknown") for a in book.get("authors", []))
        url = pick_text_url(book.get("formats", {}))

        if not url:
            skipped += 1
            print(f"[skip] no plain-text format for #{book_id}: {title}")
            continue

        filename = f"{book_id}_{safe_name(title)}.txt"
        path = OUT_DIR / filename

        if path.exists():
            print(f"[exists] {path.name}")
            continue

        print(f"[download] #{book_id} {title} | {authors}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8", errors="ignore")
        count += 1

    print(f"\nDone. Downloaded {count} books, skipped {skipped}.")

if __name__ == "__main__":
    main()