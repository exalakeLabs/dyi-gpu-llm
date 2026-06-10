#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

PRESERVE_RUNTIME_ENV=(
  ADAPTER_DIR
  BASE_MODEL
  CHAT_REQUIRE_ACCELERATOR
  CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES_VALUE
  EMBED_MODEL
  GENERATOR_ATTN_IMPLEMENTATION
  GENERATOR_CPU_MEMORY
  GENERATOR_CPU_MEMORY_FALLBACK
  GENERATOR_CPU_RESERVE_GIB
  GENERATOR_DEVICE_MAP
  GENERATOR_DTYPE
  GENERATOR_GPU_MEMORY
  GENERATOR_MODEL
  GENERATOR_MXFP4_DEQUANTIZE
  GENERATOR_OFFLOAD_DIR
  HSA_OVERRIDE_GFX_VERSION
  LAUNCH_CHAT_DRY_RUN
  LOW_VRAM_CUDA_RUNTIME
  LOW_VRAM_HIDE_GPU
  LOW_VRAM_NVIDIA_RUNTIME
  LOW_VRAM_ROCM_RUNTIME
  MAX_CONTEXT_CHARS
  MAX_NEW_TOKENS
  PYTHON
  PYTORCH_ALLOC_CONF
  PYTORCH_CUDA_ALLOC_CONF
  RAG_DIR
  RAG_EMBED_DEVICE
  RERANKER_MODEL
  RETRIEVE_K
)
typeset -A RUNTIME_ENV_OVERRIDES
for name in "${PRESERVE_RUNTIME_ENV[@]}"; do
  if [[ -n "${(P)name+x}" ]]; then
    RUNTIME_ENV_OVERRIDES[$name]="${(P)name}"
  fi
done

if [[ -f "$ROOT/.runtime" ]]; then
  source "$ROOT/.runtime" >/dev/null
fi
for name value in "${(@kv)RUNTIME_ENV_OVERRIDES}"; do
  export "$name=$value"
done

GENERATOR="${GENERATOR_MODEL:-openai/gpt-oss-20b}"
EMBED="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
RERANK="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
LOW_VRAM_GPU=0
LOW_VRAM_KIND=""
LOW_VRAM_NAME=""
LOW_VRAM_TOTAL_MIB=""
LOW_VRAM_RUNTIME="${LOW_VRAM_RUNTIME:-}"
GPU_VISIBILITY_NOTE=""
GPU_CAP_WARNING=""
GENERATOR_RUNTIME_WARNING=""
REQUIRE_ACCELERATOR="${CHAT_REQUIRE_ACCELERATOR:-}"
GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-0}"
HOST_RAM_GIB=""
DEFAULT_GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY_FALLBACK:-24GiB}"
GEN_CPU_MEMORY_OVERRIDDEN=0
if [[ -n "${GENERATOR_CPU_MEMORY:-}" ]]; then
  GEN_CPU_MEMORY_OVERRIDDEN=1
fi

detect_cpu_memory_cap() {
  local total_kib total_gib reserve_gib cap_gib

  if [[ -r /proc/meminfo ]]; then
    total_kib="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true)"
  fi

  if [[ "$total_kib" == <-> ]]; then
    total_gib=$(( total_kib / 1024 / 1024 ))
    HOST_RAM_GIB="$total_gib"
    reserve_gib="${GENERATOR_CPU_RESERVE_GIB:-8}"
    if [[ ! "$reserve_gib" == <-> ]]; then
      reserve_gib=8
    fi
    cap_gib=$(( total_gib - reserve_gib ))
    if (( cap_gib < 8 )); then
      cap_gib=8
    fi
    DEFAULT_GEN_CPU_MEMORY="${cap_gib}GiB"
    return
  fi

  DEFAULT_GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY_FALLBACK:-24GiB}"
}

detect_cpu_memory_cap

if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true)"
  NVIDIA_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if [[ "$NVIDIA_TOTAL_MIB" == <-> && "$NVIDIA_TOTAL_MIB" -le 16384 ]]; then
    LOW_VRAM_GPU=1
    LOW_VRAM_KIND="NVIDIA"
    LOW_VRAM_NAME="$NVIDIA_NAME"
    LOW_VRAM_TOTAL_MIB="$NVIDIA_TOTAL_MIB"
    if [[ -z "$LOW_VRAM_RUNTIME" ]]; then
      if [[ -n "${LOW_VRAM_NVIDIA_RUNTIME:-}" ]]; then
        LOW_VRAM_RUNTIME="$LOW_VRAM_NVIDIA_RUNTIME"
      elif [[ -n "${LOW_VRAM_CUDA_RUNTIME:-}" ]]; then
        LOW_VRAM_RUNTIME="$LOW_VRAM_CUDA_RUNTIME"
      else
        LOW_VRAM_RUNTIME="cuda"
      fi
    fi
  fi
fi

if (( ! LOW_VRAM_GPU )); then
  TORCH_GPU_INFO="$("${PYTHON:-python3}" - <<'PY' 2>/dev/null || true
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    backend = "ROCm" if torch.version.hip is not None else "CUDA"
    total_mib = props.total_memory // (1024 * 1024)
    print(f"{backend}\t{total_mib}\t{props.name}")
PY
)"
  if [[ -n "$TORCH_GPU_INFO" ]]; then
    TORCH_GPU_KIND="${TORCH_GPU_INFO%%$'\t'*}"
    TORCH_GPU_REST="${TORCH_GPU_INFO#*$'\t'}"
    TORCH_GPU_TOTAL_MIB="${TORCH_GPU_REST%%$'\t'*}"
    TORCH_GPU_NAME="${TORCH_GPU_REST#*$'\t'}"
    if [[ "$TORCH_GPU_TOTAL_MIB" == <-> && "$TORCH_GPU_TOTAL_MIB" -le 16384 ]]; then
      LOW_VRAM_GPU=1
      LOW_VRAM_KIND="$TORCH_GPU_KIND"
      LOW_VRAM_NAME="$TORCH_GPU_NAME"
      LOW_VRAM_TOTAL_MIB="$TORCH_GPU_TOTAL_MIB"
      if [[ -z "$LOW_VRAM_RUNTIME" ]]; then
        if [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
          LOW_VRAM_RUNTIME="${LOW_VRAM_ROCM_RUNTIME:-rocm}"
        else
          LOW_VRAM_RUNTIME="${LOW_VRAM_CUDA_RUNTIME:-cuda}"
        fi
      fi
    fi
  fi
fi

EMBED_DEVICE="${RAG_EMBED_DEVICE:-cpu}"
if (( LOW_VRAM_GPU )); then
  if [[ "${LOW_VRAM_RUNTIME:l}" == "cuda" || "${LOW_VRAM_RUNTIME:l}" == "rocm" || "${LOW_VRAM_RUNTIME:l}" == "gpu" ]]; then
    if [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
      GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-single}"
      REQUIRE_ACCELERATOR="${CHAT_REQUIRE_ACCELERATOR:-rocm}"
    else
      GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
      REQUIRE_ACCELERATOR="${CHAT_REQUIRE_ACCELERATOR:-cuda}"
    fi
    if [[ -n "${GENERATOR_GPU_MEMORY:-}" ]]; then
      GEN_GPU_MEMORY="$GENERATOR_GPU_MEMORY"
      GEN_GPU_MEMORY_GIB="${GEN_GPU_MEMORY:l}"
      if [[ "$LOW_VRAM_TOTAL_MIB" == <-> && "$LOW_VRAM_TOTAL_MIB" -le 12288 && "$GEN_GPU_MEMORY_GIB" == <->gib ]]; then
        GEN_GPU_MEMORY_GIB="${GEN_GPU_MEMORY_GIB%gib}"
        if (( GEN_GPU_MEMORY_GIB > 6 )); then
          GPU_CAP_WARNING="GENERATOR_GPU_MEMORY=$GEN_GPU_MEMORY leaves little MXFP4 conversion headroom on a 12 GB NVIDIA GPU; retry with 4GiB if CUDA reports device not ready."
        fi
      fi
    elif [[ "$LOW_VRAM_KIND" == "NVIDIA" || "$LOW_VRAM_KIND" == "CUDA" ]]; then
      if [[ "$LOW_VRAM_TOTAL_MIB" == <-> && "$LOW_VRAM_TOTAL_MIB" -le 12288 ]]; then
        GEN_GPU_MEMORY="4GiB"
      else
        GEN_GPU_MEMORY="14GiB"
      fi
    elif [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
      GEN_GPU_MEMORY="7GiB"
    else
      GEN_GPU_MEMORY="5GiB"
    fi
    GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-0}"
    if [[ "$LOW_VRAM_KIND" == "NVIDIA" || "$LOW_VRAM_KIND" == "CUDA" ]]; then
      GPU_VISIBILITY_NOTE="CUDA stays visible for RTX/NVIDIA; RAG embedder defaults to CPU to preserve VRAM"
    else
      GPU_VISIBILITY_NOTE="ROCm/HIP stays visible; PyTorch exposes the Radeon GPU as cuda:0; generator is required to use ROCm"
    fi
  else
    CPU_RUNTIME="${LOW_VRAM_RUNTIME:l}"
    LOW_VRAM_RUNTIME="cpu"
    CPU_RUNTIME="cpu"
    REQUIRE_ACCELERATOR="${CHAT_REQUIRE_ACCELERATOR:-}"
    GEN_DEVICE_MAP="cpu"
    GEN_GPU_MEMORY=""
    GEN_MXFP4_DEQUANTIZE="${GENERATOR_MXFP4_DEQUANTIZE:-1}"
    if [[ -z "${CUDA_VISIBLE_DEVICES_VALUE+x}" && "${LOW_VRAM_HIDE_GPU:-0}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES_VALUE=""
      GPU_VISIBILITY_NOTE="hidden from Python because LOW_VRAM_HIDE_GPU=1"
    elif [[ -z "$GPU_VISIBILITY_NOTE" ]]; then
      GPU_VISIBILITY_NOTE="visible for the RAG embedder; generator is forced to CPU with MXFP4 dequantize"
      if [[ -z "${RAG_EMBED_DEVICE+x}" ]]; then
        EMBED_DEVICE="auto"
      fi
    fi
  fi
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-$DEFAULT_GEN_CPU_MEMORY}"
  RETRIEVE_TOP_K="${RETRIEVE_K:-3}"
  if [[ "$LOW_VRAM_KIND" == "NVIDIA" || "$LOW_VRAM_KIND" == "CUDA" ]]; then
    NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
    CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-2048}"
    NEW_TOKEN_LIMIT=96
    CONTEXT_CHAR_LIMIT=2048
  else
    NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
    CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-4096}"
    NEW_TOKEN_LIMIT=160
    CONTEXT_CHAR_LIMIT=4096
  fi
  GEN_ATTN="${GENERATOR_ATTN_IMPLEMENTATION:-eager}"
  if [[ "$RETRIEVE_TOP_K" == <-> && "$RETRIEVE_TOP_K" -gt 3 ]]; then
    RETRIEVE_TOP_K=3
  fi
  if [[ "$NEW_TOKENS" == <-> && "$NEW_TOKENS" -gt "$NEW_TOKEN_LIMIT" ]]; then
    NEW_TOKENS="$NEW_TOKEN_LIMIT"
  fi
  if [[ "$CONTEXT_CHARS" == <-> && "$CONTEXT_CHARS" -gt "$CONTEXT_CHAR_LIMIT" ]]; then
    CONTEXT_CHARS="$CONTEXT_CHAR_LIMIT"
  fi
else
  GEN_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-auto}"
  GEN_GPU_MEMORY="${GENERATOR_GPU_MEMORY:-5GiB}"
  GEN_CPU_MEMORY="${GENERATOR_CPU_MEMORY:-$DEFAULT_GEN_CPU_MEMORY}"
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

if (( LOW_VRAM_GPU )) && [[ "$LOW_VRAM_KIND" == "ROCm" && "${LOW_VRAM_RUNTIME:l}" == "rocm" && "${GENERATOR:l}" == *gpt-oss* ]]; then
  print -u2 "error: $GENERATOR is not a workable Radeon RX 7600 chat model in this Transformers path."
  print -u2 "Set GENERATOR_MODEL and BASE_MODEL to a smaller Hugging Face model, for example:"
  print -u2 "  GENERATOR_MODEL=Qwen/Qwen2.5-3B-Instruct BASE_MODEL=Qwen/Qwen2.5-3B-Instruct ./launch_chat.zsh"
  exit 2
fi

if (( LOW_VRAM_GPU )) && [[ "$LOW_VRAM_TOTAL_MIB" == <-> && "$LOW_VRAM_TOTAL_MIB" -le 12288 ]]; then
  if [[ "${LOW_VRAM_RUNTIME:l}" == "cuda" || "${LOW_VRAM_RUNTIME:l}" == "gpu" ]]; then
    if [[ "${GENERATOR:l}" == *gpt-oss* && "${GEN_DEVICE_MAP:l}" == "auto" && "$GEN_MXFP4_DEQUANTIZE" == "0" ]]; then
      GENERATOR_RUNTIME_WARNING="openai/gpt-oss-20b with Transformers MXFP4 is a ~16GB-memory path; a 12GB RTX/NVIDIA card usually needs CPU/disk offload, which this MXFP4 backend may reject."
    fi
  fi
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
export CHAT_REQUIRE_ACCELERATOR="$REQUIRE_ACCELERATOR"
export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
fi
mkdir -p "$GEN_OFFLOAD_DIR"

print "Generator: $GENERATOR"
if (( LOW_VRAM_GPU )); then
  if [[ -n "$LOW_VRAM_NAME" ]]; then
    print "Low-VRAM $LOW_VRAM_KIND profile: enabled ($LOW_VRAM_NAME, ${LOW_VRAM_TOTAL_MIB} MiB detected)"
  else
    print "Low-VRAM $LOW_VRAM_KIND profile: enabled (${LOW_VRAM_TOTAL_MIB} MiB detected)"
  fi
  print "Low-VRAM runtime: $LOW_VRAM_RUNTIME"
fi
print "Generator device_map: $GENERATOR_DEVICE_MAP"
print "Generator GPU memory cap: ${GENERATOR_GPU_MEMORY:-<none>}"
if [[ -n "$GPU_CAP_WARNING" ]]; then
  print "warning: $GPU_CAP_WARNING"
fi
print "Generator CPU memory cap: $GENERATOR_CPU_MEMORY"
if [[ -n "$HOST_RAM_GIB" && "$GEN_CPU_MEMORY_OVERRIDDEN" == "0" ]]; then
  print "Host RAM detected: ${HOST_RAM_GIB}GiB; CPU memory reserve: ${GENERATOR_CPU_RESERVE_GIB:-8}GiB"
fi
print "Generator dtype: $GENERATOR_DTYPE"
print "Generator MXFP4 dequantize: $GENERATOR_MXFP4_DEQUANTIZE"
print "Required accelerator: ${CHAT_REQUIRE_ACCELERATOR:-<none>}"
if [[ -n "$GENERATOR_RUNTIME_WARNING" ]]; then
  print "warning: $GENERATOR_RUNTIME_WARNING"
fi
print "Generator offload dir: $GENERATOR_OFFLOAD_DIR"
print "Generator attention: ${GENERATOR_ATTN_IMPLEMENTATION:-<default>}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE+x}" ]]; then
  print "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<none>}"
fi
if [[ -n "$GPU_VISIBILITY_NOTE" ]]; then
  print "GPU visibility note: $GPU_VISIBILITY_NOTE"
  if [[ "$LOW_VRAM_KIND" == "ROCm" ]]; then
    if [[ "${LOW_VRAM_RUNTIME:l}" == "cpu" ]]; then
      print "ROCm generator opt-in: LOW_VRAM_ROCM_RUNTIME=rocm ./launch_chat.zsh"
      print "ROCm full CPU isolation: LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh"
    else
      print "ROCm CPU fallback: LOW_VRAM_ROCM_RUNTIME=cpu ./launch_chat.zsh"
      print "ROCm embedder opt-in: RAG_EMBED_DEVICE=rocm ./launch_chat.zsh"
    fi
  else
    if [[ "${LOW_VRAM_RUNTIME:l}" == "cpu" ]]; then
      print "CUDA generator opt-in: LOW_VRAM_CUDA_RUNTIME=cuda ./launch_chat.zsh"
      print "CUDA full CPU isolation: LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh"
    else
      print "CUDA CPU fallback: LOW_VRAM_CUDA_RUNTIME=cpu LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh"
      print "CUDA lower VRAM cap: GENERATOR_GPU_MEMORY=4GiB ./launch_chat.zsh"
      print "CUDA diagnostic mode: CUDA_LAUNCH_BLOCKING=1 GENERATOR_GPU_MEMORY=4GiB ./launch_chat.zsh"
    fi
  fi
fi
print "PyTorch CUDA alloc conf: $PYTORCH_CUDA_ALLOC_CONF"
print "RAG embedder: $EMBED on $RAG_EMBED_DEVICE"
print "Retrieve top-k: $RETRIEVE_TOP_K"
print "Context chars: $CONTEXT_CHARS"
print "Max new tokens: $NEW_TOKENS"

if [[ "${LAUNCH_CHAT_DRY_RUN:-0}" == "1" ]]; then
  print "Dry run: not starting chat_rag.py"
  exit 0
fi

CHAT_ARGS=(
  --rag-dir "${RAG_DIR:-rag}"
  --generator-model "$GENERATOR"
  --embed-model "$EMBED"
  --embed-device "$EMBED_DEVICE"
  --top-k "$RETRIEVE_TOP_K"
  --max-context-chars "$CONTEXT_CHARS"
  --max-new-tokens "$NEW_TOKENS"
)
if [[ -n "$REQUIRE_ACCELERATOR" ]]; then
  CHAT_ARGS+=(--require-accelerator "$REQUIRE_ACCELERATOR")
fi

exec "${PYTHON:-python3}" ./src/chat_rag.py "${CHAT_ARGS[@]}"
