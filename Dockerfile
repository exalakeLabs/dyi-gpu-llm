# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

ARG BACKEND=cuda
ARG CUDA_VERSION=cu130
ARG ROCM_VERSION=rocm6.4
ARG PIP_VERSION=25.1.1

ENV DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/opt/dyi-gpu-llm \
    VENV_DIR=/opt/venv \
    PYTHON=/opt/venv/bin/python \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_DATASETS_CACHE=/cache/huggingface/datasets

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      imagemagick \
      libglib2.0-0 \
      libgl1 \
      poppler-utils \
      python3 \
      python3-pip \
      python3-venv \
      tesseract-ocr \
      zsh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

COPY requirements.txt ./

RUN python3 -m venv "${VENV_DIR}" \
    && python -m pip install --upgrade "pip==${PIP_VERSION}" wheel setuptools \
    && case "${BACKEND}" in \
      cuda) TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VERSION}" ;; \
      rocm) TORCH_INDEX="https://download.pytorch.org/whl/${ROCM_VERSION}" ;; \
      cpu) TORCH_INDEX="" ;; \
      *) echo "Unsupported BACKEND=${BACKEND}; use cuda, rocm, or cpu" >&2; exit 2 ;; \
    esac \
    && if [ -n "${TORCH_INDEX}" ]; then \
      python -m pip install --index-url "${TORCH_INDEX}" torch torchvision torchaudio; \
      grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio)([<>=!~[:space:]]|$)' requirements.txt > /tmp/requirements-no-torch.txt; \
      python -m pip install -r /tmp/requirements-no-torch.txt; \
      rm -f /tmp/requirements-no-torch.txt; \
    else \
      python -m pip install -r requirements.txt; \
    fi

COPY . .

RUN ln -sfn "${VENV_DIR}" .venv \
    && chmod +x install.zsh pipeline.zsh chat.zsh docker/entrypoint.zsh \
    && mkdir -p /datasets /cache/huggingface

VOLUME ["/datasets", "/cache/huggingface"]

ENTRYPOINT ["/opt/dyi-gpu-llm/docker/entrypoint.zsh"]
CMD ["zsh"]
