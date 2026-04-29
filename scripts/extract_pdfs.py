from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("pdfs")
OUT_DIR = Path("text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
    text = extract_pdf(pdf_file)
    out_file = OUT_DIR / f"{pdf_file.stem}.txt"
    out_file.write_text(text, encoding="utf-8")
    print(f"Wrote {out_file}")

