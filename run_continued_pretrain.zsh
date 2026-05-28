#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${PYTORCH_ALLOC_CONF:-}" && -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ "$PYTHON" == */* && ! -x "$PYTHON" ]]; then
  echo "error: Python executable not found: $PYTHON" >&2
  echo "Run ./install.zsh --backend rocm, then source .runtime." >&2
  exit 1
fi

EVAL_PROMPTS="${EVAL_PROMPTS:-$ROOT/eval_prompts.txt}"
if [[ ! -f "$EVAL_PROMPTS" ]]; then
  echo "error: eval prompts file not found: $EVAL_PROMPTS" >&2
  echo "Set EVAL_PROMPTS=/path/to/prompts.txt or pass --eval_prompts /path/to/prompts.txt." >&2
  exit 1
fi

has_arg() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$name" || "$arg" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

eval_prompt_args=()
if ! has_arg "--eval_prompts" "$@"; then
  eval_prompt_args=(--eval_prompts "$EVAL_PROMPTS")
fi

rocm_safe_args=()
if ! has_arg "--attention" "$@"; then
  rocm_safe_args+=(--attention "${CONTINUED_PRETRAIN_ATTENTION:-eager}")
fi
if ! has_arg "--max_memory" "$@"; then
  rocm_safe_args+=(--max_memory "${CONTINUED_PRETRAIN_MAX_MEMORY:-4GiB}")
fi
if ! has_arg "--optim" "$@"; then
  rocm_safe_args+=(--optim "${CONTINUED_PRETRAIN_OPTIM:-adamw_torch}")
fi
if ! has_arg "--mxfp4_dequantize" "$@" && ! has_arg "--no-mxfp4_dequantize" "$@"; then
  if [[ "${CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE:-1}" != "0" ]]; then
    rocm_safe_args+=(--mxfp4_dequantize)
  fi
fi

echo "Continued pretrain wrapper:"
echo "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  injected args: ${eval_prompt_args[*]} ${rocm_safe_args[*]}"

exec "$PYTHON" "$ROOT/src/continued_pretrain_partial.py" \
  "${eval_prompt_args[@]}" \
  "${rocm_safe_args[@]}" \
  "$@"
