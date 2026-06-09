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
LOW_VRAM_GPU=0
LOW_VRAM_KIND=""
LOW_VRAM_TOTAL_MIB=""
LOW_VRAM_RUNTIME="${LOW_VRAM_RUNTIME:-}"
GPU_VISIBILITY_NOTE=""
GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if [[ "$NVIDIA_TOTAL_MIB" == <-> && "$NVIDIA_TOTAL_MIB" -le 16384 ]]; then
    LOW_VRAM_GPU=1
    LOW_VRAM_KIND="NVIDIA"
    LOW_VRAM_TOTAL_MIB="$NVIDIA_TOTAL_MIB"
    LOW_VRAM_RUNTIME="${LOW_VRAM_RUNTIME:-${LOW_VRAM_NVIDIA_RUNTIME:-cpu}}"
  fi
fi

if (( ! LOW_VRAM_GPU )); then
  TORCH_GPU_INFO="$("${PYTHON:-python3}" - <<'PY' 2>/dev/null || true
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    backend = "ROCm" if torch.version.hip is not None else "CUDA"
    total_mib = props.total_memory // (1024 * 1024)
    print(f"{backend} {total_mib}")
PY
)"
  if [[ -n "$TORCH_GPU_INFO" ]]; then
    TORCH_GPU_KIND="${TORCH_GPU_INFO%% *}"
    TORCH_GPU_TOTAL_MIB="${TORCH_GPU_INFO#* }"
    if [[ "$TORCH_GPU_TOTAL_MIB" == <-> && "$TORCH_GPU_TOTAL_MIB" -le 16384 ]]; then
      LOW_VRAM_GPU=1
      LOW_VRAM_KIND="$TORCH_GPU_KIND"
      LOW_VRAM_TOTAL_MIB="$TORCH_GPU_TOTAL_MIB"
      if [[ -z "$LOW_VRAM_RUNTIME" ]]; then
        if [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
          LOW_VRAM_RUNTIME="${LOW_VRAM_ROCM_RUNTIME:-cpu}"
        else
          LOW_VRAM_RUNTIME="${LOW_VRAM_CUDA_RUNTIME:-cpu}"
        fi
      fi
    fi
  fi
fi

EMBED_DEVICE="${RAG_EMBED_DEVICE:-cpu}"
if (( LOW_VRAM_GPU )); then
  if [[ "${LOW_VRAM_RUNTIME:l}" == "cuda" || "${LOW_VRAM_RUNTIME:l}" == "rocm" || "${LOW_VRAM_RUNTIME:l}" == "gpu" ]]; then
    GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
    GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-3GiB}"
  else
    LOW_VRAM_RUNTIME="cpu"
    GEN_DEVICE_MAP="cpu"
    GEN_GPU_MEMORY=""
    GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-1}"
    if [[ "${LOW_VRAM_HIDE_GPU:-0}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES_VALUE=""
      GPU_VISIBILITY_NOTE="hidden from Python because LOW_VRAM_HIDE_GPU=1"
    else
      GPU_VISIBILITY_NOTE="visible for the RAG embedder; generator is forced to CPU with MXFP4 dequantize"
      if [[ -z "${RAG_EMBED_DEVICE+x}" ]]; then
        EMBED_DEVICE="auto"
      fi
    fi
  fi
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-96GiB}"
  RETRIEVE_TOP_K="${RETRIEVE_K:-3}"
  NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
  CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-4096}"
  GEN_ATTN="${GENERATOR_ATTN_IMPLEMENTATION:-eager}"
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
  GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
  GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-5GiB}"
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-48GiB}"
  RETRIEVE_TOP_K="${RETRIEVE_K:-24}"
  NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
  CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
  GEN_ATTN="${GENERATOR_ATTN_IMPLEMENTATION:-}"
fi
GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-$GEN_MXFP4_DEQUANTIZE}"
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

if [[ "${GENERATOR:l}" == *gpt-oss* && "${GEN_ATTN:l}" == "sdpa" ]]; then
  print -u2 "warning: gpt-oss does not support attn_implementation=sdpa in Transformers yet; using eager."
  GEN_ATTN="eager"
fi

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
export GENERATOR_MXFP4_DEQUANTIZE="$GEN_MXFP4_DEQUANTIZE"
export GENERATOR_OFFLOAD_DIR="$GEN_OFFLOAD_DIR"
export GENERATOR_ATTN_IMPLEMENTATION="$GEN_ATTN"
export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
fi
mkdir -p "$GEN_OFFLOAD_DIR"

print "Generator: $GENERATOR"
if (( LOW_VRAM_GPU )); then
  print "Low-VRAM $LOW_VRAM_KIND profile: enabled (${LOW_VRAM_TOTAL_MIB} MiB detected)"
  print "Low-VRAM runtime: $LOW_VRAM_RUNTIME"
fi
print "Generator device_map: $GENERATOR_DEVICE_MAP"
print "Generator GPU memory cap: ${GENERATOR_GPU_MEMORY:-<none>}"
print "Generator CPU memory cap: $GENERATOR_CPU_MEMORY"
print "Generator dtype: $GENERATOR_DTYPE"
print "Generator MXFP4 dequantize: $GENERATOR_MXFP4_DEQUANTIZE"
print "Generator offload dir: $GENERATOR_OFFLOAD_DIR"
print "Generator attention: ${GENERATOR_ATTN_IMPLEMENTATION:-<default>}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE+x}" ]]; then
  print "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<none>}"
fi
if [[ -n "$GPU_VISIBILITY_NOTE" ]]; then
  print "GPU visibility note: $GPU_VISIBILITY_NOTE"
  if [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
    print "ROCm generator opt-in: LOW_VRAM_ROCM_RUNTIME=rocm RAG_EMBED_DEVICE=rocm ./launch_chat.zsh"
    print "ROCm full CPU isolation: LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh"
  else
    print "CUDA generator opt-in: LOW_VRAM_CUDA_RUNTIME=cuda RAG_EMBED_DEVICE=cuda ./launch_chat.zsh"
    print "CUDA full CPU isolation: LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh"
  fi
fi
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
