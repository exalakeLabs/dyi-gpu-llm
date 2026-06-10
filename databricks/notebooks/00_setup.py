# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = next(
    root for root in (Path.cwd(), *Path.cwd().parents) if (root / "src" / "databricks_env.py").exists()
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from databricks_env import (
    ensure_directories,
    set_default_env,
    set_huggingface_token,
    to_local_path,
    widget_bool,
    widget_text,
)

model_root = widget_text("model_root", "/dbfs/FileStore/llama32-local", "Model/data root")
install_requirements = widget_bool("install_requirements", False, "Install requirements")
hf_secret_scope = widget_text("hf_secret_scope", "", "HF secret scope")
hf_secret_key = widget_text("hf_secret_key", "HF_TOKEN", "HF secret key")

set_default_env(model_root)
set_huggingface_token(hf_secret_scope, hf_secret_key)
ensure_directories()

if install_requirements:
    requirements_path = REPO_ROOT / "databricks" / "requirements-databricks.txt"
    if not requirements_path.exists():
        requirements_path = REPO_ROOT / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    dbutils.library.restartPython()

print("Repository:", REPO_ROOT)
print("MODEL_ROOT:", os.environ["MODEL_ROOT"])
print("PREPARED_DIR:", os.environ["PREPARED_DIR"])
print("CORPUS_DIR:", os.environ["CORPUS_DIR"])
print("RAG_DIR:", os.environ["RAG_DIR"])
print("MODEL_DIR:", os.environ["MODEL_DIR"])
print("HF_TOKEN configured:", bool(os.environ.get("HF_TOKEN")))
