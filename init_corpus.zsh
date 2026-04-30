#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

./scripts/download_gutenberg.py --query "science" --max-books 25000
./scripts/download_gutenberg.py --query "history" --topic "Ancient" --max-books 15
./scripts/build_pdf_training_pairs.py --pdf-dir pdfs --text-dir text