# Databricks notebook source
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
dbutils.widgets.text("dbfs_root", "/dbfs/FileStore/llama32", "DBFS Root")

# COMMAND ----------
import os, sys
dbfs_root = dbutils.widgets.get("dbfs_root")
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
import project_config as cfg
print(f"IS_DATABRICKS  : {cfg.IS_DATABRICKS}")
print(f"BASE_MODEL     : {cfg.BASE_MODEL}")
print(f"TEXT_DIR       : {cfg.TEXT_DIR}")
print(f"PREPARED_DIR   : {cfg.PREPARED_DIR}")
print(f"DATA_DIR       : {cfg.DATA_DIR}")
print(f"OUTPUT_DIR     : {cfg.OUTPUT_DIR}")
print(f"RAG_DIR        : {cfg.RAG_DIR}")
print(f"LORA_DIR       : {cfg.LORA_DIR}")
print(f"ADAPTER_DIR    : {cfg.ADAPTER_DIR}")
print(f"SHARED_OUTPUT  : {cfg.SHARED_OUTPUT_DIR}")

# COMMAND ----------
# MAGIC %md ### ✅ Setup complete — proceed to notebook 01.
