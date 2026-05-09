# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# ///
# =============================================================================
# 00 · Cluster Setup & Verification
# Run this first to install dependencies and verify the GPU cluster is ready.
# =============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## 00 · Cluster Setup & Verification
# MAGIC
# MAGIC **Run once after attaching to a GPU cluster.**
# MAGIC Installs project dependencies and checks that PyTorch, CUDA, and the
# MAGIC HuggingFace cache on DBFS are all working correctly.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   trl>=0.8.0 \
# MAGIC   sentence-transformers>=2.6.0 \
# MAGIC   faiss-cpu>=1.7.4 \
# MAGIC   pypdf>=4.0.0 \
# MAGIC   truststore \
# MAGIC   "uvicorn[standard]"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./nb_config

# COMMAND ----------

import os, sys
os.environ["LLAMA_DBFS_ROOT"] = dbfs_root

# Add src/ to path (works whether the repo is in Repos or /Workspace)
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
print(f"Repo root : {_repo_root}")
print(f"src/ path : {_src}")

# COMMAND ----------

# MAGIC %md ### 1 · GPU / CUDA Check

# COMMAND ----------

import sys

# Clear stale typing_extensions cached from runtime startup so the
# notebook-scoped version (which has 'deprecated') is found instead.
for _mod in [k for k in sys.modules if k == 'typing_extensions' or k.startswith('typing_extensions.')]:
    del sys.modules[_mod]

import torch

print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
print(f"CUDA version     : {torch.version.cuda}")
print(f"HIP  version     : {torch.version.hip}")
print(f"GPU count        : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  VRAM={props.total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md ### 2 · DBFS Directory Layout

# COMMAND ----------

from pathlib import Path

dirs = [
    f"{dbfs_root}/text",
    f"{dbfs_root}/prepared",
    f"{dbfs_root}/data",
    f"{dbfs_root}/output/lora",
    f"{dbfs_root}/rag",
    f"{dbfs_root}/hf_cache",
]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✔  {d}")

# COMMAND ----------

# MAGIC %md ### 3 · HuggingFace Cache

# COMMAND ----------

hf_cache = f"{dbfs_root}/hf_cache"
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["HF_DATASETS_CACHE"] = hf_cache

# Quick connectivity check — downloads tokenizer config only (~1 KB)
from transformers import AutoTokenizer
print("Downloading tokenizer config for Qwen2.5-3B-Instruct…")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
print(f"✔  Tokenizer loaded. Vocab size: {tok.vocab_size}")
print(f"   Cache location: {hf_cache}")

# COMMAND ----------

# MAGIC %md ### 4 · project_config smoke-test

# COMMAND ----------

# ---------------------------------------------------------------------------
# Databricks detection
# ---------------------------------------------------------------------------
# DATABRICKS_RUNTIME_VERSION is injected automatically into every Databricks
# cluster.  When it is present we default all paths to DBFS so models, data,
# and checkpoints survive cluster restarts.
IS_DATABRICKS: bool = "DATABRICKS_RUNTIME_VERSION" in os.environ

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
REPO_ROOT = "/Workspace/alex.gauthier@sigmacomputing.com/llama32-local/databricks"

def env_dir(var: str, default_rel: str) -> Path:
    value = os.environ.get(var, "").strip()
    path = Path(value).expanduser() if value else REPO_ROOT / default_rel
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


# ---------------------------------------------------------------------------
# Directory layout — DBFS on Databricks, local env vars elsewhere
# ---------------------------------------------------------------------------
if IS_DATABRICKS:
    # All paths live under a single DBFS root so they are shared across the
    # driver and all worker nodes.  Override with LLAMA_DBFS_ROOT if you use
    # Unity Catalog Volumes or a different mount point.
    _dbfs_root = Path(
        os.environ.get("LLAMA_DBFS_ROOT", "/Users/alex.gauthier@sigmacomputing.com/Lora-slow")
    )

    TEXT_DIR     = _dbfs_root / "text"
    PREPARED_DIR = _dbfs_root / "prepared"
    DATA_DIR     = _dbfs_root / "data"
    OUTPUT_DIR   = _dbfs_root / "output"
    RAG_DIR      = _dbfs_root / "rag"

    # Pin the HuggingFace cache to DBFS so large model weights are downloaded
    # once and reused across cluster restarts and jobs.
    _hf_cache = str(_dbfs_root / "hf_cache")
    os.environ.setdefault("HF_HOME", _hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)
else:
    TEXT_DIR     = env_dir("LLAMA_TEXT_DIR",     "text")
    PREPARED_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")
    DATA_DIR     = env_dir("LLAMA_DATA_DIR",     "data")
    OUTPUT_DIR   = env_dir("LLAMA_OUTPUT_DIR",   "output")
    RAG_DIR      = env_dir("LLAMA_RAG_DIR",      "rag")

# ---------------------------------------------------------------------------
# Derived paths (same structure on both runtimes)
# ---------------------------------------------------------------------------
TRAIN_FILE  = DATA_DIR  / "train.jsonl"
VAL_FILE    = DATA_DIR  / "val.jsonl"
LORA_DIR    = OUTPUT_DIR / "lora"
ADAPTER_DIR = LORA_DIR  / "final"

# ---------------------------------------------------------------------------
# Shared storage for distributed / Spark training
# ---------------------------------------------------------------------------
# On Databricks this defaults to the DBFS output dir so all workers agree on
# a single checkpoint path without extra configuration.
_shared_env = os.environ.get("LLAMA_SHARED_OUTPUT_DIR", "").strip()
if _shared_env:
    SHARED_OUTPUT_DIR: Path | None = Path(_shared_env).expanduser()
elif IS_DATABRICKS:
    SHARED_OUTPUT_DIR = OUTPUT_DIR
else:
    SHARED_OUTPUT_DIR = None

# ---------------------------------------------------------------------------
# MLflow experiment (Databricks managed tracking by default)
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "/Users/alex.gauthier@sigmacomputing.com/Lora-slow/lora-finetune"
)


# COMMAND ----------

import src.project_config as cfg

print(f"IS_DATABRICKS  : {IS_DATABRICKS}")
print(f"BASE_MODEL     : {BASE_MODEL}")
print(f"TEXT_DIR       : {TEXT_DIR}")
print(f"PREPARED_DIR   : {PREPARED_DIR}")
print(f"DATA_DIR       : {DATA_DIR}")
print(f"OUTPUT_DIR     : {OUTPUT_DIR}")
print(f"RAG_DIR        : {RAG_DIR}")
print(f"LORA_DIR       : {LORA_DIR}")
print(f"ADAPTER_DIR    : {ADAPTER_DIR}")
print(f"SHARED_OUTPUT  : {SHARED_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md ### ✅ Setup complete — proceed to notebook 01.
