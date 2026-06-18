#!/usr/bin/env python
"""
Convert PDFs in a directory into plain-text files.

This script extracts text from each `*.pdf` in `--pdf-dir` and writes a
corresponding `*.txt` file into `--text-dir` (default: `text/`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_path

PDF_DIR = env_path("PDF_DIR", "pdfs")
RAWTEXT_DIR = env_path("RAWTEXT_DIR", "text")


def extract_pdf_to_text(pdf_path: Path) -> str:
    from pypdf import PdfReader  # type: ignore

    try:
        reader = PdfReader(str(pdf_path))

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return ""

        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                pages.append(f"\n[ERROR page {i+1}: {e}]\n")
        return "\n\n".join(pages)
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    ap.add_argument("--text-dir", default=str(RAWTEXT_DIR))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    extracted = 0
    for pdf in pdfs:
        raw = extract_pdf_to_text(pdf)
        (text_dir / f"{pdf.stem}.txt").write_text(raw, encoding="utf-8")
        extracted += 1
    print(f"Extracted PDFs: {extracted}")
    print(f"Output directory: {text_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
