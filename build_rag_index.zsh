#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

python ./src/clean_text.py

python ./src/index_builder.py \
  --input-dir prepared \
  --output-dir rag \
  --embed-model BAAI/bge-base-en-v1.5 \
  --chunk-size-chars 1800 \
  --overlap-chars 250 \
  --batch-size 32
