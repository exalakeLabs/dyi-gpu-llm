#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found; install Python 3 first." >&2
  exit 1
fi

"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

python -m ensurepip --upgrade || uv pip install --python .venv/bin/python pip
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
python -m pip install pypdf cryptography datasets transformers trl peft accelerate sentencepiece requests sentence-transformers faiss-cpu fastapi uvicorn

python -m pip install -r requirements.txt

pip install torch transformers sentence-transformers faiss-cpu tqdm

python - <<'PY'
import torch

gpu_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()

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
