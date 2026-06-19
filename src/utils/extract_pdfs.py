#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_path

PDF_DIR = env_path("PDF_DIR", "pdfs")
RAWTEXT_DIR = env_path("RAWTEXT_DIR", "text")


from pypdf import PdfReader


def extract_pdf(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))

        # Try decrypt if needed (empty password attempt)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                print(f"[SKIP - encrypted] {pdf_path}")
                return ""

        pages = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as e:
                txt = f"\n[ERROR page {i+1}: {e}]\n"
            pages.append(txt)

        return "\n\n".join(pages)

    except Exception as e:
        print(f"[FAILED] {pdf_path}: {e}")
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from PDFs.")
    parser.add_argument("--pdf-dir", default=str(PDF_DIR))
    parser.add_argument("--text-dir", default=str(RAWTEXT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir).expanduser()
    text_dir = Path(args.text_dir).expanduser()
    text_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        text = extract_pdf(pdf_file)
        out_file = text_dir / f"{pdf_file.stem}.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"Wrote {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
