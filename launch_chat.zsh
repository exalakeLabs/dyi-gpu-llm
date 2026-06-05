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
EMBED_DEVICE="${RAG_EMBED_DEVICE:-cpu}"
GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-5GiB}"
GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-48GiB}"
GEN_DTYPE="${GENERATOR_DTYPE:-auto}"
GEN_OFFLOAD_DIR="${GENERATOR_OFFLOAD_DIR:-${TMPDIR:-/tmp}/llama32-generator-offload}"

case "${GENERATOR:l}" in
  gpt-oss:20b|gpt-oss:20)
    print -u2 "warning: GENERATOR_MODEL=$GENERATOR is an Ollama-style id; using openai/gpt-oss-20b."
    GENERATOR="openai/gpt-oss-20b"
    ;;
  gpt-oss:120b|gpt-oss:120)
    print -u2 "warning: GENERATOR_MODEL=$GENERATOR is an Ollama-style id; using openai/gpt-oss-120b."
    GENERATOR="openai/gpt-oss-120b"
    ;;
  gpt-oss:*)
    print -u2 "error: GENERATOR_MODEL=$GENERATOR is not a Hugging Face model id."
    print -u2 "Use openai/gpt-oss-20b or openai/gpt-oss-120b."
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

export RAG_EMBED_DEVICE="$EMBED_DEVICE"
export GENERATOR_MODEL="$GENERATOR"
export BASE_MODEL="$GENERATOR"
export GENERATOR_DEVICE_MAP="$GEN_DEVICE_MAP"
export GENERATOR_GPU_MEMORY="$GEN_GPU_MEMORY"
export GENERATOR_CPU_MEMORY="$GEN_CPU_MEMORY"
export GENERATOR_DTYPE="$GEN_DTYPE"
export GENERATOR_OFFLOAD_DIR="$GEN_OFFLOAD_DIR"
mkdir -p "$GEN_OFFLOAD_DIR"

print "Generator: $GENERATOR"
print "Generator device_map: $GENERATOR_DEVICE_MAP"
print "Generator GPU memory cap: $GENERATOR_GPU_MEMORY"
print "Generator CPU memory cap: $GENERATOR_CPU_MEMORY"
print "Generator dtype: $GENERATOR_DTYPE"
print "Generator offload dir: $GENERATOR_OFFLOAD_DIR"
print "RAG embedder: $EMBED on $RAG_EMBED_DEVICE"

exec "${PYTHON:-python3}" ./src/chat_rag.py \
  --rag-dir "${RAG_DIR:-rag}" \
  --generator-model "$GENERATOR" \
  --embed-model "$EMBED" \
  --embed-device "$EMBED_DEVICE" \
  --top-k "${RETRIEVE_K:-24}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-500}"
