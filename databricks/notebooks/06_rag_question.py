# Databricks notebook source
# MAGIC %md
# MAGIC # Ask A RAG Question

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
    widget_bool,
    widget_dropdown,
    widget_text,
)

model_root = widget_text("model_root", "/dbfs/FileStore/llama32-local", "Model/data root")
rag_dir = widget_text("rag_dir", os.environ.get("RAG_DIR", f"{model_root}/rag"), "RAG dir")
question = widget_text("question", "What does the prepared material say?", "Question")
model_name = widget_text("model_name", os.environ.get("GENERATOR_MODEL", "openai/gpt-oss-20b"), "Generator model")
embed_model = widget_text("embed_model", os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5"), "Embedding model")
top_k = widget_text("top_k", os.environ.get("RETRIEVE_K", "8"), "Top K")
max_context_chars = widget_text("max_context_chars", os.environ.get("MAX_CONTEXT_CHARS", "12000"), "Max context chars")
max_new_tokens = widget_text("max_new_tokens", os.environ.get("MAX_NEW_TOKENS", "500"), "Max new tokens")
dtype = widget_dropdown("dtype", "auto", ["auto", "bf16", "fp16", "fp32"], "Dtype")
dry_run = widget_bool("dry_run", True, "Dry run")

set_default_env(model_root)
os.environ["RAG_DIR"] = str(to_local_path(rag_dir))
os.environ["GENERATOR_MODEL"] = model_name
os.environ["EMBED_MODEL"] = embed_model
ensure_directories()

import teach_gpt_oss_rag

args = [
    "--rag-dir",
    str(to_local_path(rag_dir)),
    "--model",
    model_name,
    "--embed-model",
    embed_model,
    "--top-k",
    top_k,
    "--max-context-chars",
    max_context_chars,
    "--max-new-tokens",
    max_new_tokens,
    "--dtype",
    dtype,
    "--question",
    question,
]
if dry_run:
    args.append("--dry-run")

result = run_cli(teach_gpt_oss_rag.main, "teach_gpt_oss_rag.py", args)
if result:
    raise RuntimeError(f"teach_gpt_oss_rag failed with exit code {result}")
