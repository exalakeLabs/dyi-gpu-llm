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
PROMPT_HF_TOKEN="${PROMPT_HF_TOKEN:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-hf_token.txt}"

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

  --cuda-version  CUDA wheel suffix, e.g. cu130 (default: cu130)
  --rocm-version  ROCm wheel suffix, e.g. rocm6.4 (default: rocm6.4)

Environment overrides:
  PYTHON             Python executable used to create the venv (default: python3)
  PIP_VERSION        Pip version installed in the venv, or latest (default: 25.1.1)
  VENV_DIR           Virtual environment directory (default: .venv)
  REQUIREMENTS_FILE  Pip requirements file (default: requirements.txt)
  ENV_FILE           Environment file to update (default: .env)
  ENV_DEFAULT_FILE   Template copied to ENV_FILE when missing (default: .env.default)
  PROMPT_ENV         Set to 0 to skip prompts for literal .env defaults
  PROMPT_HF_TOKEN    Set to 0 to skip Hugging Face token setup (default: 1)
  HF_TOKEN_FILE      Optional file to read a Hugging Face token from (default: hf_token.txt)
  GENERATOR_MODEL    Existing value shown as the default in the model chooser

Options:
  --no-env-prompt    Skip prompts for literal .env defaults
  --no-hf-token      Skip Hugging Face token setup
EOF
  exit "$usage_status"
}

CUDA_VERSION="cu130"
ROCM_VERSION="rocm6.4"
PIP_VERSION="${PIP_VERSION:-25.1.1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)        BACKEND="$2";       shift 2 ;;
    --cuda-version)   CUDA_VERSION="$2";  shift 2 ;;
    --rocm-version)   ROCM_VERSION="$2";  shift 2 ;;
    --no-env-prompt)  PROMPT_ENV=0;       shift ;;
    --no-hf-token)    PROMPT_HF_TOKEN=0;  shift ;;
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

is_path_env_var() {
  local name="$1"

  [[ "$name" == *_DIR || "$name" == *_FILE || "$name" == *_PATH ]] && return 0

  case "$name" in
    RAWTEXT_DIR|RAW_TEXT_DIR|VENV_DIR|GENERATOR_OFFLOAD_DIR|EVAL_PROMPTS|SYSTEM_PROMPT_FILE)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

expand_home_path_value() {
  local value="$1"

  if [[ "$value" == \~ ]]; then
    print -r -- "$HOME"
  elif [[ "$value" == \~/* ]]; then
    print -r -- "$HOME/${value#\~/}"
  else
    print -r -- "$value"
  fi
}

generator_model_options() {
  cat <<'EOF'
Qwen/Qwen2.5-3B-Instruct	Recommended small default; fast local RAG/chat and lightweight fine-tuning.
Qwen/Qwen2.5-7B-Instruct	Stronger Qwen baseline; comfortable on a single A100.
meta-llama/Llama-3.1-8B-Instruct	Strong general instruct model; may require Hugging Face license approval.
openai/gpt-oss-20b	OpenAI open-weight reasoning model; practical on a single A100 and useful for RAG reasoning.
openai/gpt-oss-120b	Larger OpenAI open-weight reasoning model; target this only for an 80GB A100-class node.
custom	Enter another Hugging Face model id.
EOF
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

    if [[ "$name" == "GENERATOR_MODEL" ]]; then
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

  if is_path_env_var "$name"; then
    value="$(expand_home_path_value "$value")"
  fi

  export "${name}=${value}"
  set_env_var "$name" "$value"
}

prompt_generator_model() {
  local current="${GENERATOR_MODEL:-${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}}"
  local -a model_ids model_notes option_lines
  local line model_id note choice selected

  option_lines=("${(@f)$(generator_model_options)}")
  model_ids=()
  model_notes=()

  for line in "$option_lines[@]"; do
    model_id="${line%%	*}"
    note="${line#*	}"
    model_ids+=("$model_id")
    model_notes+=("$note")
  done

  printf '\n\033[1;34mGenerator model\033[0m\n'
  printf 'Choose the model used for chat/RAG generation and base fine-tuning defaults.\n\n'

  local i
  for (( i = 1; i <= ${#model_ids[@]}; i++ )); do
    printf '  %d) %s\n     %s\n' "$i" "$model_ids[$i]" "$model_notes[$i]"
  done

  printf '\nGenerator model [%s]: ' "$current"
  read -r choice

  if [[ -z "$choice" ]]; then
    selected="$current"
  elif [[ "$choice" == <-> && "$choice" -ge 1 && "$choice" -le "${#model_ids[@]}" ]]; then
    selected="$model_ids[$choice]"
  else
    selected="$choice"
  fi

  if [[ "${selected:l}" == "custom" ]]; then
    printf 'Custom Hugging Face model id [%s]: ' "$current"
    read -r selected
    if [[ -z "$selected" ]]; then
      selected="$current"
    fi
  fi

  if [[ "${selected:l}" == gpt-oss:* ]]; then
    print -u2 "error: Generator model must be a Hugging Face model id."
    print -u2 "Use openai/gpt-oss-20b or openai/gpt-oss-120b."
    return 2
  fi

  export GENERATOR_MODEL="$selected"
  export BASE_MODEL="$selected"
  export DEFAULT_MODEL="$selected"
  set_env_var "GENERATOR_MODEL" "$selected"
  set_env_var "BASE_MODEL" "$selected"
  set_env_var "DEFAULT_MODEL" "$selected"
}

configure_hf_token() {
  if [[ "$PROMPT_HF_TOKEN" == "0" ]]; then
    return
  fi

  local token="${HF_TOKEN:-}"

  if [[ -z "$token" && -f "$HF_TOKEN_FILE" ]]; then
    token="$(head -n 1 "$HF_TOKEN_FILE" | tr -d '[:space:]')"
    if [[ -n "$token" ]]; then
      export HF_TOKEN="$token"
      set_env_var "HF_TOKEN" "$token"
      printf 'Saved HF_TOKEN from %s to %s.\n' "$HF_TOKEN_FILE" "$ENV_FILE"
      return
    fi
  fi

  if [[ -n "$token" ]]; then
    set_env_var "HF_TOKEN" "$token"
    printf 'Saved existing HF_TOKEN to %s.\n' "$ENV_FILE"
    return
  fi

  if [[ ! -t 0 ]]; then
    printf '\033[1;33mSkipping Hugging Face token prompt because stdin is not interactive.\033[0m\n'
    return
  fi

  printf '\n\033[1;34mHugging Face token\033[0m\n'
  printf 'Paste a Hugging Face access token, or press Enter to skip: '
  read -rs token
  printf '\n'

  if [[ -z "$token" ]]; then
    printf 'Skipped HF_TOKEN setup.\n'
    return
  fi

  export HF_TOKEN="$token"
  set_env_var "HF_TOKEN" "$token"
  printf 'Saved HF_TOKEN to %s.\n' "$ENV_FILE"
}

configure_runtime_env() {
  if [[ "$PROMPT_ENV" == "0" ]]; then
    load_env_file
    configure_hf_token
    return
  fi

  if [[ ! -t 0 ]]; then
    printf '\033[1;33mSkipping environment prompts because stdin is not interactive.\033[0m\n'
    load_env_file
    configure_hf_token
    return
  fi

  load_env_file

  local -a prompt_defaults
  prompt_defaults=("${(@f)$(literal_env_defaults)}")

  if (( ${#prompt_defaults[@]} == 0 )); then
    configure_hf_token
    return
  fi

  printf '\n\033[1;34mEnvironment defaults\033[0m\n'
  printf 'Press Enter to keep the current value.\n\n'

  prompt_generator_model

  local entry name fallback sep
  sep=$'\t'
  for entry in "$prompt_defaults[@]"; do
    name="${entry%%${sep}*}"
    fallback="${entry#*${sep}}"
    prompt_env_var "$name" "$fallback"
  done

  load_env_file
  configure_hf_token

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
if [[ "$PIP_VERSION" == "latest" ]]; then
  python -m pip install -U pip wheel
else
  python -m pip install -U "pip==${PIP_VERSION}" wheel
fi

print_python_environment() {
  python - <<'PY'
import sys
import sysconfig

print()
print("Python environment:")
print(f"  executable : {sys.executable}")
print(f"  version    : {sys.version.split()[0]}")
print(f"  platform   : {sysconfig.get_platform()}")
print(f"  cache tag  : {sys.implementation.cache_tag}")
PY
}

check_torch_index() {
  local index_url="$1"
  local output

  output="$(python -m pip index versions torch --index-url "$index_url" 2>&1 || true)"
  if [[ "$output" == *"Available versions:"* ]]; then
    return
  fi

  printf '\nerror: no compatible torch wheels were found for this Python/platform/backend.\n' >&2
  printf 'Selected PyTorch index: %s\n\n' "$index_url" >&2
  printf '%s\n\n' "$output" >&2

  if [[ "$BACKEND" == "cuda" ]]; then
    printf 'CUDA wheels are for Nvidia GPUs. On AMD/Radeon, rerun:\n' >&2
    printf '  ./%s --backend rocm --rocm-version %s\n\n' "$SCRIPT_NAME" "$ROCM_VERSION" >&2
    printf 'For Nvidia CUDA on RTX 50-series, use a CUDA 13 wheel such as cu130.\n' >&2
  elif [[ "$BACKEND" == "rocm" ]]; then
    printf 'For AMD/Radeon, try a currently published ROCm wheel suffix such as rocm6.4.\n' >&2
  fi

  exit 1
}

print_python_environment

if [[ -n "$TORCH_INDEX" ]]; then
  printf '\nInstalling PyTorch from backend-specific index: %s\n' "$TORCH_INDEX"
  check_torch_index "$TORCH_INDEX"
  python -m pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio

  REQUIREMENTS_NO_TORCH="$(mktemp)"
  trap 'rm -f "$REQUIREMENTS_NO_TORCH"' EXIT
  grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio)([<>=!~[:space:]]|$)' "$REQUIREMENTS_FILE" > "$REQUIREMENTS_NO_TORCH"
  python -m pip install -r "$REQUIREMENTS_NO_TORCH"
else
  python -m pip install -r "$REQUIREMENTS_FILE"
fi

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
            capability = torch.cuda.get_device_capability(i)
            print(
                f"GPU {i}        : {torch.cuda.get_device_name(i)} "
                f"(sm_{capability[0]}{capability[1]})"
            )

        try:
            probe = torch.ones(1, device="cuda")
            probe = probe + 1
            torch.cuda.synchronize()
            print(f"CUDA smoke    : ok ({probe.cpu().item():.1f})")
        except Exception as exc:
            raise SystemExit(
                "CUDA smoke test failed. PyTorch can see the GPU, but the installed "
                "wheel cannot execute kernels on it. For RTX 50-series / sm_120, "
                "reinstall with a CUDA 13 wheel, for example: "
                "./install.zsh --backend cuda --cuda-version cu130"
            ) from exc

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
