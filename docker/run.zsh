#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

GPU="${GPU:-cuda}"
IMAGE="${IMAGE:-}"
IMAGE_SET=0
NAME="${NAME:-dyi-gpu-llm}"
DATA_DIR="${DATA_DIR:-$ROOT/.container-data/datasets}"
CACHE_DIR="${CACHE_DIR:-$ROOT/.container-data/huggingface}"
ENV_FILE="${ENV_FILE:-}"
REMOVE=1
INTERACTIVE=1
DOCKER_ARGS=()
COMMAND=()
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./docker/run.zsh [options] [-- command ...]

Options:
  --gpu cuda|rocm|none  GPU exposure mode (default: $GPU)
  --image IMAGE         Container image (default: dyi-gpu-llm:<gpu>, or dyi-gpu-llm:cpu for --gpu none)
  --name NAME           Container name prefix (default: $NAME)
  --data-dir DIR        Host directory mounted at /datasets.
  --cache-dir DIR       Host Hugging Face cache mounted at /cache/huggingface.
  --env-file FILE       Docker env-file to pass into the container.
  --no-rm               Keep the container after it exits.
  --no-tty              Disable interactive tty flags.
  --dry-run             Print the docker run command without running it.
  -h, --help            Show this help.

Examples:
  ./docker/run.zsh --gpu cuda -- chat
  ./docker/run.zsh --gpu cuda -- pipeline rag
  ./docker/run.zsh --gpu rocm -- zsh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:?missing value for --gpu}"
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing value for --image}"
      IMAGE_SET=1
      shift 2
      ;;
    --name)
      NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:?missing value for --cache-dir}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:?missing value for --env-file}"
      shift 2
      ;;
    --no-rm)
      REMOVE=0
      shift
      ;;
    --no-tty)
      INTERACTIVE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      COMMAND=("$@")
      break
      ;;
  esac
done

case "$GPU" in
  cuda|rocm|none) ;;
  *)
    print -u2 "error: --gpu must be cuda, rocm, or none"
    exit 2
    ;;
esac

if (( ! IMAGE_SET )); then
  if [[ "$GPU" == "none" ]]; then
    IMAGE="dyi-gpu-llm:cpu"
  else
    IMAGE="dyi-gpu-llm:${GPU}"
  fi
fi

mkdir -p "$DATA_DIR" "$CACHE_DIR"

cmd=(docker run)
if (( REMOVE )); then
  cmd+=(--rm)
fi
if (( INTERACTIVE )); then
  cmd+=(-it)
fi

cmd+=(--name "${NAME}-$$")
cmd+=(-v "$DATA_DIR:/datasets")
cmd+=(-v "$CACHE_DIR:/cache/huggingface")

if [[ -n "$ENV_FILE" ]]; then
  cmd+=(--env-file "$ENV_FILE")
elif [[ -f "$ROOT/.env.container" ]]; then
  cmd+=(--env-file "$ROOT/.env.container")
fi

case "$GPU" in
  cuda)
    cmd+=(--gpus all)
    ;;
  rocm)
    cmd+=(--device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined)
    ;;
  none)
    ;;
esac

cmd+=("${DOCKER_ARGS[@]}")
cmd+=("$IMAGE")

if [[ "${#COMMAND[@]}" -gt 0 ]]; then
  cmd+=("${COMMAND[@]}")
fi

print "Running $IMAGE"
print "GPU mode: $GPU"
print "Datasets: $DATA_DIR -> /datasets"
print "HF cache: $CACHE_DIR -> /cache/huggingface"
print "Command: ${cmd[*]}"
if (( DRY_RUN )); then
  exit 0
fi
exec "${cmd[@]}"
