#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT/src" || exit 1

exec "${PYTHON:-python3}" train_pipeline.py "$@"
