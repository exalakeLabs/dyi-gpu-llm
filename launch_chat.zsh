#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.runtime" ]]; then
  source "$ROOT/.runtime" >/dev/null
fi

GENERATOR="${GENERATOR_MODEL:-openai/gpt-oss-20b}"
EMBED="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
RERANK="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"

case "${GENERATOR:l}" in
  gpt-oss:*)
    print -u2 "error: GENERATOR_MODEL must be a Hugging Face model id."
    print -u2 "Set GENERATOR_MODEL=openai/gpt-oss-20b and rerun."
    exit 2
    ;;
esac

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

exec "${PYTHON:-python3}" ./src/chat_rag.py \
  --rag-dir "${RAG_DIR:-rag}" \
  --generator-model "$GENERATOR" \
  --embed-model "$EMBED" \
  --top-k "${RETRIEVE_K:-24}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}"
