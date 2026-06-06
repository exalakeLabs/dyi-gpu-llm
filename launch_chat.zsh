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
LOW_VRAM_NVIDIA=0

if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if [[ "$NVIDIA_TOTAL_MIB" == <-> && "$NVIDIA_TOTAL_MIB" -le 16384 ]]; then
    LOW_VRAM_NVIDIA=1
  fi
fi

EMBED_DEVICE="${RAG_EMBED_DEVICE:-cpu}"
GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
if (( LOW_VRAM_NVIDIA )); then
  GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-3GiB}"
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-64GiB}"
  RETRIEVE_TOP_K="${RETRIEVE_K:-3}"
  NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
  CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-4096}"
  GEN_ATTN="${GENERATOR_ATTN_IMPLEMENTATION:-sdpa}"
  if [[ "$RETRIEVE_TOP_K" == <-> && "$RETRIEVE_TOP_K" -gt 3 ]]; then
    RETRIEVE_TOP_K=3
  fi
  if [[ "$NEW_TOKENS" == <-> && "$NEW_TOKENS" -gt 160 ]]; then
    NEW_TOKENS=160
  fi
  if [[ "$CONTEXT_CHARS" == <-> && "$CONTEXT_CHARS" -gt 4096 ]]; then
    CONTEXT_CHARS=4096
  fi
else
  GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-5GiB}"
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-48GiB}"
  RETRIEVE_TOP_K="${RETRIEVE_K:-24}"
  NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
  CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
  GEN_ATTN="${GENERATOR_ATTN_IMPLEMENTATION:-}"
fi
GEN_DTYPE="${GENERATOR_DTYPE:-auto}"
GEN_OFFLOAD_DIR="${GENERATOR_OFFLOAD_DIR:-${TMPDIR:-/tmp}/llama32-generator-offload}"
CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}}"

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
export GENERATOR_ATTN_IMPLEMENTATION="$GEN_ATTN"
export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
mkdir -p "$GEN_OFFLOAD_DIR"

print "Generator: $GENERATOR"
if (( LOW_VRAM_NVIDIA )); then
  print "Low-VRAM NVIDIA profile: enabled (${NVIDIA_TOTAL_MIB} MiB detected)"
fi
print "Generator device_map: $GENERATOR_DEVICE_MAP"
print "Generator GPU memory cap: $GENERATOR_GPU_MEMORY"
print "Generator CPU memory cap: $GENERATOR_CPU_MEMORY"
print "Generator dtype: $GENERATOR_DTYPE"
print "Generator offload dir: $GENERATOR_OFFLOAD_DIR"
print "Generator attention: ${GENERATOR_ATTN_IMPLEMENTATION:-<default>}"
print "PyTorch CUDA alloc conf: $PYTORCH_CUDA_ALLOC_CONF"
print "RAG embedder: $EMBED on $RAG_EMBED_DEVICE"
print "Retrieve top-k: $RETRIEVE_TOP_K"
print "Context chars: $CONTEXT_CHARS"
print "Max new tokens: $NEW_TOKENS"

exec "${PYTHON:-python3}" ./src/chat_rag.py \
  --rag-dir "${RAG_DIR:-rag}" \
  --generator-model "$GENERATOR" \
  --embed-model "$EMBED" \
  --embed-device "$EMBED_DEVICE" \
  --top-k "$RETRIEVE_TOP_K" \
  --max-context-chars "$CONTEXT_CHARS" \
  --max-new-tokens "$NEW_TOKENS"
