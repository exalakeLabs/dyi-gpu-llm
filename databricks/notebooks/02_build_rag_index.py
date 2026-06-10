# Databricks notebook source
# MAGIC %md
# MAGIC # Build RAG Index

# COMMAND ----------

import os
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
    run_cli,
    set_default_env,
    to_local_path,
    widget_text,
)

model_root = widget_text("model_root", "/dbfs/FileStore/llama32-local", "Model/data root")
input_dir = widget_text("input_dir", os.environ.get("PREPARED_DIR", f"{model_root}/prepared"), "Prepared text dir")
output_dir = widget_text("output_dir", os.environ.get("RAG_DIR", f"{model_root}/rag"), "RAG output dir")
embed_model = widget_text("embed_model", os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5"), "Embedding model")
device = widget_text("device", os.environ.get("EMBED_DEVICE", "auto"), "Embedding device")
chunk_size_chars = widget_text("chunk_size_chars", os.environ.get("CHUNK_SIZE_CHARS", "1800"), "Chunk chars")
overlap_chars = widget_text("overlap_chars", os.environ.get("OVERLAP_CHARS", "250"), "Overlap chars")
batch_size = widget_text("batch_size", os.environ.get("BATCH_SIZE", "32"), "Batch size")
max_files = widget_text("max_files", "0", "Max files")

set_default_env(model_root)
os.environ["PREPARED_DIR"] = str(to_local_path(input_dir))
os.environ["RAG_DIR"] = str(to_local_path(output_dir))
os.environ["EMBED_MODEL"] = embed_model
ensure_directories()

import index_builder

args = [
    "--input-dir",
    str(to_local_path(input_dir)),
    "--output-dir",
    str(to_local_path(output_dir)),
    "--embed-model",
    embed_model,
    "--device",
    device,
    "--chunk-size-chars",
    chunk_size_chars,
    "--overlap-chars",
    overlap_chars,
    "--batch-size",
    batch_size,
    "--max-files",
    max_files,
]

result = run_cli(index_builder.main, "index_builder.py", args)
if result:
    raise RuntimeError(f"index_builder failed with exit code {result}")
