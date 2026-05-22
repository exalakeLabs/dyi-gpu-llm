#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
SCRIPT_NAME="${0:t}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
ENV_FILE="${ENV_FILE:-.env}"
ENV_DEFAULT_FILE="${ENV_DEFAULT_FILE:-.env.default}"
PROMPT_ENV="${PROMPT_ENV:-1}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BACKEND=""

usage() {
  local usage_status="${1:-1}"
  cat >&2 <<EOF
Usage: ${SCRIPT_NAME} --backend <cuda|rocm|mps> [--cuda-version <ver>] [--rocm-version <ver>]

  --backend       cuda   Install PyTorch with CUDA (Nvidia)  [required]
                  rocm   Install PyTorch with ROCm (Radeon/AMD)
                  mps    Install PyTorch with MPS (Apple Silicon)

  --cuda-version  CUDA wheel suffix, e.g. cu124 (default: cu124)
  --rocm-version  ROCm wheel suffix, e.g. rocm6.2 (default: rocm6.2)

Environment overrides:
  PYTHON             Python executable used to create the venv (default: python3)
  VENV_DIR           Virtual environment directory (default: .venv)
  REQUIREMENTS_FILE  Pip requirements file (default: requirements.txt)
  ENV_FILE           Environment file to update (default: .env)
  ENV_DEFAULT_FILE   Template copied to ENV_FILE when missing (default: .env.default)
  PROMPT_ENV         Set to 0 to skip prompts for literal .env defaults

Options:
  --no-env-prompt    Skip prompts for literal .env defaults
EOF
  exit "$usage_status"
}

CUDA_VERSION="cu124"
ROCM_VERSION="rocm6.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)        BACKEND="$2";       shift 2 ;;
    --cuda-version)   CUDA_VERSION="$2";  shift 2 ;;
    --rocm-version)   ROCM_VERSION="$2";  shift 2 ;;
    --no-env-prompt)  PROMPT_ENV=0;       shift ;;
    -h|--help)        usage 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage 1 ;;
  esac
done

if [[ -z "$BACKEND" ]]; then
  echo "error: --backend is required (cuda, rocm, or mps)" >&2
  usage 1
fi

case "$BACKEND" in
  cuda)
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}"
    BACKEND_LABEL="CUDA (Nvidia) — wheel: ${CUDA_VERSION}"
    ;;
  rocm)
    TORCH_INDEX="https://download.pytorch.org/whl/${ROCM_VERSION}"
    BACKEND_LABEL="ROCm (AMD/Radeon) — wheel: ${ROCM_VERSION}"
    ;;
  mps)
    TORCH_INDEX=""
    BACKEND_LABEL="MPS (Apple Silicon)"
    ;;
  *)
    echo "error: --backend must be 'cuda', 'rocm', or 'mps', got '${BACKEND}'" >&2
    usage 1
    ;;
esac

printf '\n\033[1;34mBackend: %s\033[0m\n\n' "$BACKEND_LABEL"

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
quote_env_value() {
  local value="$1"
  print -r -- "${(qqq)value}"
}

set_env_var() {
  local name="$1"
  local value="$2"
  local tmp_file="${ENV_FILE}.tmp.$$"
  local found=0
  local quoted
  quoted="$(quote_env_value "$value")"

  if [[ ! -f "$ENV_FILE" ]]; then
    : > "$ENV_FILE"
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ "^[[:space:]]*(export[[:space:]]+)?${name}=" ]]; then
      print -r -- "export ${name}=${quoted}" >> "$tmp_file"
      found=1
    else
      print -r -- "$line" >> "$tmp_file"
    fi
  done < "$ENV_FILE"

  if (( ! found )); then
    print -r -- "export ${name}=${quoted}" >> "$tmp_file"
  fi

  mv "$tmp_file" "$ENV_FILE"
}

constant_env_names() {
  local env_source="${ENV_DEFAULT_FILE}"
  local line name value

  if [[ ! -f "$env_source" ]]; then
    env_source="$ENV_FILE"
  fi

  if [[ ! -f "$env_source" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ "^[[:space:]]*export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.*)$" ]]; then
      name="$match[1]"
      value="$match[2]"
    elif [[ "$line" =~ "^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$" ]]; then
      name="$match[1]"
      value="$match[2]"
    else
      continue
    fi

    if [[ -z "$value" || "$value" == *'$'* ]]; then
      continue
    fi

    print -r -- "$name"
  done < "$env_source"
}

literal_env_defaults() {
  local env_source="${ENV_DEFAULT_FILE}"
  local line name value fallback

  if [[ ! -f "$env_source" ]]; then
    env_source="$ENV_FILE"
  fi

  if [[ ! -f "$env_source" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ "^[[:space:]]*export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.*)$" ]]; then
      name="$match[1]"
      value="$match[2]"
    elif [[ "$line" =~ "^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$" ]]; then
      name="$match[1]"
      value="$match[2]"
    else
      continue
    fi

    if [[ -z "$value" || "$value" == *'$'* ]]; then
      continue
    fi

    fallback="$value"
    if [[ "$fallback" == \"*\" ]]; then
      fallback="${fallback#\"}"
      fallback="${fallback%\"}"
    elif [[ "$fallback" == \'*\' ]]; then
      fallback="${fallback#\'}"
      fallback="${fallback%\'}"
    fi

    printf '%s\t%s\n' "$name" "$fallback"
  done < "$env_source"
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return
  fi

  if [[ ! -f "$ENV_DEFAULT_FILE" ]]; then
    echo "error: environment file not found: $ENV_FILE" >&2
    echo "error: default template not found: $ENV_DEFAULT_FILE" >&2
    exit 1
  fi

  cp "$ENV_DEFAULT_FILE" "$ENV_FILE"
  printf 'Created %s from %s.\n' "$ENV_FILE" "$ENV_DEFAULT_FILE"
}

load_env_file() {
  ensure_env_file

  if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
  fi
}

prompt_env_var() {
  local name="$1"
  local fallback="$2"
  local current="${(P)name-}"
  local value

  if [[ -z "$current" ]]; then
    current="$fallback"
  fi

  printf '%s [%s]: ' "$name" "$current"
  read -r value
  if [[ -z "$value" ]]; then
    value="$current"
  fi

  export "${name}=${value}"
  set_env_var "$name" "$value"
}

configure_runtime_env() {
  if [[ "$PROMPT_ENV" == "0" ]]; then
    load_env_file
    return
  fi

  if [[ ! -t 0 ]]; then
    printf '\033[1;33mSkipping environment prompts because stdin is not interactive.\033[0m\n'
    load_env_file
    return
  fi

  load_env_file

  local -a prompt_defaults
  prompt_defaults=("${(@f)$(literal_env_defaults)}")

  if (( ${#prompt_defaults[@]} == 0 )); then
    return
  fi

  printf '\n\033[1;34mEnvironment defaults\033[0m\n'
  printf 'Press Enter to keep the current value.\n\n'

  local entry name fallback
  for entry in "$prompt_defaults[@]"; do
    name="${entry%%	*}"
    fallback="${entry#*	}"
    prompt_env_var "$name" "$fallback"
  done

  load_env_file

  printf '\nSaved environment defaults to %s.\n\n' "$ENV_FILE"
}

configure_runtime_env

# ---------------------------------------------------------------------------
# Python check
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found; install Python 3 first." >&2
  exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "error: requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Virtual environment + dependencies
# ---------------------------------------------------------------------------
PYTHON_PATH="$(command -v "$PYTHON")"
VENV_PATH="${VENV_DIR:A}"
ROOT_PATH="${ROOT:A}"

if [[ "$PYTHON_PATH" == "$VENV_PATH"/* ]]; then
  if [[ -x /usr/bin/python3 ]]; then
    PYTHON_PATH=/usr/bin/python3
  else
    echo "error: PYTHON resolves inside $VENV_DIR and /usr/bin/python3 is unavailable." >&2
    echo "Set PYTHON to a Python executable outside the venv and retry." >&2
    exit 1
  fi
fi

if [[ -z "$VENV_DIR" || "$VENV_PATH" == "/" || "$VENV_PATH" == "$ROOT_PATH" ]]; then
  echo "error: refusing to delete unsafe VENV_DIR: $VENV_DIR" >&2
  exit 1
fi

if [[ "$VENV_PATH" != "$ROOT_PATH"/* ]]; then
  echo "error: refusing to delete VENV_DIR outside project root: $VENV_DIR" >&2
  exit 1
fi

if [[ -e "$VENV_PATH" && ! -d "$VENV_PATH" ]]; then
  echo "error: VENV_DIR exists but is not a directory: $VENV_DIR" >&2
  exit 1
fi

if [[ -d "$VENV_PATH" ]]; then
  printf '\033[1;33mRemoving existing virtual environment: %s\033[0m\n' "$VENV_PATH"
  rm -rf -- "$VENV_PATH"
fi

"$PYTHON_PATH" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install -U pip

pip_args=(-r "$REQUIREMENTS_FILE")
if [[ -n "$TORCH_INDEX" ]]; then
  pip_args+=(--extra-index-url "$TORCH_INDEX")
fi

python -m pip install "${pip_args[@]}"

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
python - <<PY
import torch

print()
print("=== PyTorch / GPU Environment Check ===")
print(f"torch version : {torch.__version__}")
print(f"backend       : ${BACKEND}")

if "${BACKEND}" == "mps":
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available : {mps_available}")
    if not mps_available:
        print("  (run 'python -c \"import torch; print(torch.backends.mps.is_built())\"' to debug)")
else:
    gpu_available = torch.cuda.is_available()
    gpu_count    = torch.cuda.device_count()
    print(f"CUDA version  : {torch.version.cuda}")
    print(f"HIP version   : {torch.version.hip}")
    print(f"GPU available : {gpu_available}")
    print(f"GPU count     : {gpu_count}")
    if gpu_available:
        for i in range(gpu_count):
            print(f"GPU {i}        : {torch.cuda.get_device_name(i)}")

print()
PY

printf '\n'
printf '\033[1;32m✔ Setup complete.\033[0m\n'
printf 'Activate the environment with:\n'
printf '  \033[1;36msource .runtime\033[0m\n'
printf '\n'
printf 'Then verify with:\n'
printf '  \033[1;36mpython --version\033[0m\n'
printf '\n'
