#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.runtime" ]]; then
  source "$ROOT/.runtime" >/dev/null
fi

exec "${PYTHON:-python3}" ./src/chat_rag.py \
  --rag-dir "${RAG_DIR:-rag}" \
  --generator-model "${GENERATOR_MODEL:-openai/gpt-oss-20b}" \
  --top-k "${RETRIEVE_K:-24}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}"
