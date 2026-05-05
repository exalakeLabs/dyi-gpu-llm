#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BACKEND=""

usage() {
  cat >&2 <<EOF
Usage: $0 --backend <cuda|rocm> [--cuda-version <ver>] [--rocm-version <ver>]

  --backend       cuda   Install PyTorch with CUDA (Nvidia)  [required]
                  rocm   Install PyTorch with ROCm (Radeon/AMD)

  --cuda-version  CUDA wheel suffix, e.g. cu124 (default: cu124)
  --rocm-version  ROCm wheel suffix, e.g. rocm6.2 (default: rocm6.2)
EOF
  exit 1
}

CUDA_VERSION="cu124"
ROCM_VERSION="rocm6.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)        BACKEND="$2";       shift 2 ;;
    --cuda-version)   CUDA_VERSION="$2";  shift 2 ;;
    --rocm-version)   ROCM_VERSION="$2";  shift 2 ;;
    -h|--help)        usage ;;
    *) echo "error: unknown argument '$1'" >&2; usage ;;
  esac
done

if [[ -z "$BACKEND" ]]; then
  echo "error: --backend is required (cuda or rocm)" >&2
  usage
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
  *)
    echo "error: --backend must be 'cuda' or 'rocm', got '${BACKEND}'" >&2
    usage
    ;;
esac

printf '\n\033[1;34mBackend: %s\033[0m\n\n' "$BACKEND_LABEL"

# ---------------------------------------------------------------------------
# Python check
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found; install Python 3 first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Virtual environment + dependencies
# ---------------------------------------------------------------------------
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# torch: required. torchvision/torchaudio: optional (often no wheels yet for newest Python on cu124/rocm).
python -m pip install torch --index-url "$TORCH_INDEX"
if ! python -m pip install torchvision torchaudio --index-url "$TORCH_INDEX"; then
  printf '\033[1;33m⚠ Skipping torchvision/torchaudio: no matching wheels on this index for your Python/platform.\033[0m\n' >&2
  printf '\033[1;33m  (torch alone is enough for text / LoRA in this repo.)\033[0m\n' >&2
fi
python -m pip install pypdf cryptography datasets transformers trl peft accelerate sentencepiece requests sentence-transformers faiss-cpu fastapi uvicorn

python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
python - <<'PY'
import torch

gpu_available = torch.cuda.is_available()
gpu_count    = torch.cuda.device_count()

print()
print("=== PyTorch / GPU Environment Check ===")
print(f"torch version : {torch.__version__}")
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
printf '  \033[1;36msource .launch_env\033[0m\n'
printf '\n'
printf 'Then verify with:\n'
printf '  \033[1;36mpython --version\033[0m\n'
printf '\n'
