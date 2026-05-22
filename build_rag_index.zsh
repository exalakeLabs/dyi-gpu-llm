#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

python ./src/clean_text.py

python ./src/index_builder.py \
  --input-dir $PREPARED_DIR \
  --output-dir $RAG_DIR \
  --embed-model $EMBED_MODEL \
  --chunk-size-chars $CHUNK_SIZE_CHARS \
  --overlap-chars $OVERLAP_CHARS \
  --batch-size $BATCH_SIZE
