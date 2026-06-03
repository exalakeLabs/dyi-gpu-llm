#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

INPUT_DIR="${1:-${PREPARED_DIR:-prepared}}"
OUTPUT_DIR="${2:-${RAG_DIR:-rag}}"
EMBED="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
RERANK="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
CHUNK_SIZE="${CHUNK_SIZE_CHARS:-1800}"
OVERLAP="${OVERLAP_CHARS:-250}"
BATCH="${BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON:-python3}"

case "${EMBED:l}" in
  *gpt-oss*)
    print -u2 "error: EMBED_MODEL is set to '$EMBED', but gpt-oss is a generator model."
    print -u2 "Set EMBED_MODEL=BAAI/bge-base-en-v1.5 and rerun."
    exit 2
    ;;
esac

case "${RERANK:l}" in
  *gpt-oss*)
    print -u2 "error: RERANKER_MODEL is set to '$RERANK', but gpt-oss is a generator model."
    print -u2 "Set RERANKER_MODEL=BAAI/bge-reranker-v2-m3 and rerun."
    exit 2
    ;;
esac

"$PYTHON_BIN" ./src/index_builder.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --embed-model "$EMBED" \
  --chunk-size-chars "$CHUNK_SIZE" \
  --overlap-chars "$OVERLAP" \
  --batch-size "$BATCH"
