#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.runtime" ]]; then
  source "$ROOT/.runtime" >/dev/null
fi

exec "${PYTHON:-python}" ./src/chat_rag_qwen_bge.py \
  --index-dir "${RAG_DIR:-rag}" \
  --generator-backend "${GENERATOR_BACKEND:-ollama}" \
  --generator-model "${GENERATOR_MODEL:-gpt-oss:20b}" \
  --retrieve-k "${RETRIEVE_K:-24}" \
  --rerank-top-n "${RERANK_TOP_N:-6}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}"
