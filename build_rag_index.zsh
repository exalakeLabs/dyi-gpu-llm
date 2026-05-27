#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ "$PYTHON" == */* && ! -x "$PYTHON" ]]; then
  echo "error: Python executable not found: $PYTHON" >&2
  echo "Run ./install.zsh --backend rocm, then source .runtime." >&2
  exit 1
fi

"$PYTHON" ./src/clean_text.py

"$PYTHON" ./src/index_builder.py \
  --input-dir "$PREPARED_DIR" \
  --output-dir "$RAG_DIR" \
  --embed-model "$EMBED_MODEL" \
  --device "${EMBED_DEVICE:-auto}" \
  --chunk-size-chars "$CHUNK_SIZE_CHARS" \
  --overlap-chars "$OVERLAP_CHARS" \
  --batch-size "$BATCH_SIZE"
