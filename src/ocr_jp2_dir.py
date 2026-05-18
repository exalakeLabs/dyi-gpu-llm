#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a command and fail loudly if it errors."""
    subprocess.run(cmd, check=True)


def check_dependencies() -> None:
    missing = []

    if shutil.which("tesseract") is None:
        missing.append("tesseract")

    # We prefer ImageMagick because it can stream directly into Tesseract.
    if shutil.which("magick") is None and shutil.which("convert") is None:
        missing.append("imagemagick")

    if missing:
        raise SystemExit(
            "Missing required command(s): "
            + ", ".join(missing)
            + "\n\nInstall on Ubuntu with:\n"
            + "  sudo apt update\n"
            + "  sudo apt install tesseract-ocr imagemagick\n"
        )


def natural_sort_key(path: Path) -> list[object]:
    """
    Natural-ish sort so page_2.jp2 comes before page_10.jp2.
    """
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def ocr_jp2_file(
    jp2_path: Path,
    output_txt_path: Path,
    language: str = "eng",
    resize_percent: int = 200,
) -> None:
    """
    Convert JP2 image to PNG stream, then OCR with Tesseract.
    Produces output_txt_path.
    """
    output_base = output_txt_path.with_suffix("")

    magick_cmd = "magick" if shutil.which("magick") else "convert"

    # ImageMagick streams PNG to stdout.
    convert_cmd = [
        magick_cmd,
        str(jp2_path),
        "-colorspace",
        "Gray",
        "-resize",
        f"{resize_percent}%",
        "png:-",
    ]

    # Tesseract reads image from stdin and writes output_base.txt.
    tesseract_cmd = [
        "tesseract",
        "stdin",
        str(output_base),
        "-l",
        language,
    ]

    print(f"OCR: {jp2_path.name} -> {output_txt_path.name}")

    convert_proc = subprocess.Popen(
        convert_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    tesseract_proc = subprocess.Popen(
        tesseract_cmd,
        stdin=convert_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if convert_proc.stdout:
        convert_proc.stdout.close()

    tesseract_stdout, tesseract_stderr = tesseract_proc.communicate()
    convert_stderr = convert_proc.stderr.read().decode("utf-8", errors="replace")
    convert_return = convert_proc.wait()

    if convert_return != 0:
        raise RuntimeError(
            f"ImageMagick failed for {jp2_path}\n\n{convert_stderr}"
        )

    if tesseract_proc.returncode != 0:
        raise RuntimeError(
            f"Tesseract failed for {jp2_path}\n\n{tesseract_stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR a directory of .jp2 files into text/*.txt"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .jp2 files",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="eng",
        help="Tesseract language code, default: eng",
    )
    parser.add_argument(
        "--resize-percent",
        type=int,
        default=200,
        help="Resize percentage before OCR, default: 200",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also combine all page text files into text/combined.txt",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip OCR for .txt files that already exist",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    check_dependencies()

    output_dir = input_dir / "text"
    output_dir.mkdir(parents=True, exist_ok=True)

    jp2_files = sorted(input_dir.glob("*.jp2"), key=natural_sort_key)

    if not jp2_files:
        raise SystemExit(f"No .jp2 files found in: {input_dir}")

    print(f"Found {len(jp2_files)} JP2 files")
    print(f"Output directory: {output_dir}")

    output_txt_files: list[Path] = []

    for jp2_file in jp2_files:
        output_txt = output_dir / f"{jp2_file.stem}.txt"
        output_txt_files.append(output_txt)

        if args.skip_existing and output_txt.exists():
            print(f"Skipping existing: {output_txt.name}")
            continue

        ocr_jp2_file(
            jp2_path=jp2_file,
            output_txt_path=output_txt,
            language=args.language,
            resize_percent=args.resize_percent,
        )

    if args.combine:
        combined_path = output_dir / "combined.txt"
        print(f"Combining text into: {combined_path}")

        with combined_path.open("w", encoding="utf-8") as combined:
            for txt_file in output_txt_files:
                if txt_file.exists():
                    combined.write(txt_file.read_text(encoding="utf-8", errors="replace"))
                    combined.write("\n\n")

    print("Done.")


if __name__ == "__main__":
    main()