#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT" || exit 1

exec "${PYTHON:-python3}" training/train_pipeline.py "$@"
