#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

BACKEND="${BACKEND:-cuda}"
CUDA_VERSION="${CUDA_VERSION:-cu130}"
ROCM_VERSION="${ROCM_VERSION:-rocm6.4}"
UBUNTU_VERSION="${UBUNTU_VERSION:-24.04}"
TAG="${TAG:-}"
TAG_SET=0
NO_CACHE=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./docker/build.zsh [options]

Options:
  --backend cuda|rocm|cpu   PyTorch backend wheel set (default: $BACKEND)
  --cuda-version VERSION    PyTorch CUDA wheel suffix (default: $CUDA_VERSION)
  --rocm-version VERSION    PyTorch ROCm wheel suffix (default: $ROCM_VERSION)
  --ubuntu-version VERSION  Ubuntu base version (default: $UBUNTU_VERSION)
  --tag IMAGE_TAG           Output image tag (default: dyi-gpu-llm:<backend>)
  --no-cache                Build without Docker layer cache.
  --dry-run                 Print the docker build command without running it.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:?missing value for --backend}"
      shift 2
      ;;
    --cuda-version)
      CUDA_VERSION="${2:?missing value for --cuda-version}"
      shift 2
      ;;
    --rocm-version)
      ROCM_VERSION="${2:?missing value for --rocm-version}"
      shift 2
      ;;
    --ubuntu-version)
      UBUNTU_VERSION="${2:?missing value for --ubuntu-version}"
      shift 2
      ;;
    --tag)
      TAG="${2:?missing value for --tag}"
      TAG_SET=1
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "error: unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

case "$BACKEND" in
  cuda|rocm|cpu) ;;
  *)
    print -u2 "error: --backend must be cuda, rocm, or cpu"
    exit 2
    ;;
esac

if (( ! TAG_SET )); then
  TAG="dyi-gpu-llm:${BACKEND}"
fi

cmd=(
  docker build
  --build-arg "BACKEND=$BACKEND"
  --build-arg "CUDA_VERSION=$CUDA_VERSION"
  --build-arg "ROCM_VERSION=$ROCM_VERSION"
  --build-arg "UBUNTU_VERSION=$UBUNTU_VERSION"
  --tag "$TAG"
  --file Dockerfile
)

if (( NO_CACHE )); then
  cmd+=(--no-cache)
fi

cmd+=(.)

print "Building $TAG"
print "Backend: $BACKEND"
print "Ubuntu: $UBUNTU_VERSION"
print "Command: ${cmd[*]}"
if (( DRY_RUN )); then
  exit 0
fi
exec "${cmd[@]}"
