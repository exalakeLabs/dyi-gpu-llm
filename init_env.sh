#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "error: '$PYTHON' not found; install Python 3 first." >&2
  exit 1
fi

"$PYTHON" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt

echo
echo "Bootstrap complete. In new shells run:"
echo "  cd \"$ROOT\" && source .venv/bin/activate"
