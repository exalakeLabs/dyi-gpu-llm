#!/bin/bash
# =============================================================================
# Databricks cluster init script — llama32-local
#
# Attach this script to a GPU cluster (g4dn/g5/A10G/V100/A100) via:
#   Cluster → Advanced Options → Init Scripts → DBFS path to this file
#
# This runs once per node at cluster startup (as root, before Spark launches).
# Heavy packages like torch are pre-installed on Databricks ML Runtime — this
# script only adds the small extras the project needs.
# =============================================================================
set -euo pipefail

echo "[init] Starting llama32-local cluster init..."

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    > /dev/null 2>&1

# ---------------------------------------------------------------------------
# 2. Python packages
# ---------------------------------------------------------------------------
# Databricks ML Runtime already ships torch, transformers, accelerate, peft.
# We only add what is missing or needs a newer version.
pip install --quiet --no-cache-dir \
    trl>=0.8.0 \
    sentence-transformers>=2.6.0 \
    faiss-cpu>=1.7.4 \
    pypdf>=4.0.0 \
    truststore \
    fastapi \
    "uvicorn[standard]"

echo "[init] Python packages installed."

# ---------------------------------------------------------------------------
# 3. DBFS HuggingFace cache
# ---------------------------------------------------------------------------
# Pin HF downloads to DBFS so models survive cluster restarts.
# Notebooks also set these, but setting here ensures they apply to all
# processes (including the Spark executor JVMs spawning Python workers).
LLAMA_DBFS_ROOT="${LLAMA_DBFS_ROOT:-/dbfs/FileStore/llama32}"
HF_CACHE="${LLAMA_DBFS_ROOT}/hf_cache"
mkdir -p "$HF_CACHE"

cat >> /etc/environment <<EOF
HF_HOME=${HF_CACHE}
TRANSFORMERS_CACHE=${HF_CACHE}
HF_DATASETS_CACHE=${HF_CACHE}
LLAMA_DBFS_ROOT=${LLAMA_DBFS_ROOT}
EOF

echo "[init] HuggingFace cache set to: ${HF_CACHE}"

# ---------------------------------------------------------------------------
# 4. ROCm compatibility shim (only needed on AMD GPU clusters)
# ---------------------------------------------------------------------------
if python3 -c "import torch; exit(0 if torch.version.hip else 1)" 2>/dev/null; then
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    echo "[init] ROCm detected, GFX override: ${HSA_OVERRIDE_GFX_VERSION}"
fi

echo "[init] llama32-local cluster init complete."
