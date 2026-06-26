#!/usr/bin/env zsh
set -euo pipefail

APP_HOME="${APP_HOME:-/opt/dyi-gpu-llm}"
cd "$APP_HOME"

export PYTHON="${PYTHON:-/opt/venv/bin/python}"
export PATH="${PYTHON:h}:$PATH"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p /datasets "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [[ ! -f .env && -f .env.default ]]; then
  cp .env.default .env
fi

if [[ $# -eq 0 ]]; then
  exec zsh
fi

case "$1" in
  pipeline|pipeline.zsh)
    shift
    exec ./pipeline.zsh "$@"
    ;;
  chat|chat.zsh)
    shift
    exec ./chat.zsh "$@"
    ;;
  install|install.zsh)
    shift
    exec ./install.zsh "$@"
    ;;
  bash|zsh|sh|python|python3|pip|pip3)
    exec "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
