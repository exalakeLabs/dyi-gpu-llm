# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Continued-Pretraining Corpus

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
text_dir = widget_text("text_dir", os.environ.get("PREPARED_DIR", f"{model_root}/prepared"), "Prepared text dir")
corpus_dir = widget_text("corpus_dir", os.environ.get("CORPUS_DIR", f"{model_root}/corpus"), "Corpus dir")
model_name = widget_text("model_name", os.environ.get("DEFAULT_MODEL", "openai/gpt-oss-20b"), "Tokenizer model")
seq_len = widget_text("seq_len", os.environ.get("DEFAULT_SEQ_LEN", "2048"), "Sequence length")
eval_ratio = widget_text("eval_ratio", os.environ.get("DEFAULT_EVAL_RATIO", "0.01"), "Eval ratio")
min_chars = widget_text("min_chars", os.environ.get("DEFAULT_MIN_CHARS", "1000"), "Minimum chars")
num_proc = widget_text("num_proc", os.environ.get("DEFAULT_DATASET_NUM_PROC", "1"), "Dataset num proc")
tokenize_batch_size = widget_text(
    "tokenize_batch_size",
    os.environ.get("DEFAULT_TOKENIZE_BATCH_SIZE", "128"),
    "Tokenize batch size",
)

set_default_env(model_root)
os.environ["PREPARED_DIR"] = str(to_local_path(text_dir))
os.environ["CORPUS_DIR"] = str(to_local_path(corpus_dir))
os.environ["DEFAULT_TEXT_DIR"] = str(to_local_path(text_dir))
os.environ["DEFAULT_CORPUS_DIR"] = str(to_local_path(corpus_dir))
os.environ["DEFAULT_MODEL"] = model_name
os.environ["TRAIN_FILE"] = str(to_local_path(corpus_dir) / "train.jsonl")
ensure_directories()

import generate_pretrain_corpus

args = [
    "--model_name",
    model_name,
    "--text_dir",
    str(to_local_path(text_dir)),
    "--corpus_dir",
    str(to_local_path(corpus_dir)),
    "--seq_len",
    seq_len,
    "--eval_ratio",
    eval_ratio,
    "--min_chars",
    min_chars,
    "--num_proc",
    num_proc,
    "--tokenize_batch_size",
    tokenize_batch_size,
]

result = run_cli(generate_pretrain_corpus.main, "generate_pretrain_corpus.py", args)
if result:
    raise RuntimeError(f"generate_pretrain_corpus failed with exit code {result}")
