# Databricks notebook source
# MAGIC %md
# MAGIC # Continued Pretraining

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
corpus_dir = widget_text("corpus_dir", os.environ.get("CORPUS_DIR", f"{model_root}/corpus"), "Corpus dir")
output_dir = widget_text(
    "output_dir",
    os.environ.get("DEFAULT_OUTPUT_DIR", f"{model_root}/model/output_partial"),
    "Output dir",
)
eval_prompts = widget_text("eval_prompts", str(REPO_ROOT / "eval_prompts.txt"), "Eval prompts")
model_name = widget_text("model_name", os.environ.get("DEFAULT_MODEL", "openai/gpt-oss-20b"), "Model")
dtype = widget_dropdown("dtype", os.environ.get("DEFAULT_DTYPE", "auto"), ["auto", "bf16", "fp16", "fp32"], "Dtype")
attention = widget_dropdown(
    "attention",
    os.environ.get("DEFAULT_ATTENTION", "auto"),
    ["auto", "flash_attention_2", "sdpa", "eager", "default"],
    "Attention",
)
device_map = widget_dropdown("device_map", os.environ.get("DEFAULT_DEVICE_MAP", "auto"), ["auto", "single", "none"], "Device map")
max_memory = widget_text("max_memory", os.environ.get("DEFAULT_MAX_MEMORY", ""), "GPU max memory")
cpu_memory = widget_text("cpu_memory", os.environ.get("DEFAULT_CPU_MEMORY", "64GiB"), "CPU max memory")
train_last_n_layers = widget_text(
    "train_last_n_layers",
    os.environ.get("DEFAULT_TRAIN_LAST_N_LAYERS", "8"),
    "Train last N layers",
)
per_device_train_batch_size = widget_text(
    "per_device_train_batch_size",
    os.environ.get("DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE", "1"),
    "Train batch size",
)
gradient_accumulation_steps = widget_text(
    "gradient_accumulation_steps",
    os.environ.get("DEFAULT_GRADIENT_ACCUMULATION_STEPS", "8"),
    "Gradient accumulation",
)
num_train_epochs = widget_text("num_train_epochs", os.environ.get("DEFAULT_NUM_TRAIN_EPOCHS", "1.0"), "Epochs")
learning_rate = widget_text("learning_rate", os.environ.get("DEFAULT_LEARNING_RATE", "2e-4"), "Learning rate")
save_steps = widget_text("save_steps", os.environ.get("DEFAULT_SAVE_STEPS", "100"), "Save steps")
logging_steps = widget_text("logging_steps", os.environ.get("DEFAULT_LOGGING_STEPS", "1"), "Logging steps")
dataloader_num_workers = widget_text("dataloader_num_workers", "2", "Dataloader workers")
mxfp4_dequantize = widget_bool("mxfp4_dequantize", False, "MXFP4 dequantize")
compile_model = widget_bool("compile_model", False, "Compile model")

set_default_env(model_root)
os.environ["CORPUS_DIR"] = str(to_local_path(corpus_dir))
os.environ["DEFAULT_CORPUS_DIR"] = str(to_local_path(corpus_dir))
os.environ["DEFAULT_OUTPUT_DIR"] = str(to_local_path(output_dir))
os.environ["DEFAULT_MODEL"] = model_name
os.environ["BASE_MODEL"] = model_name
os.environ["GENERATOR_MODEL"] = model_name
ensure_directories()

import continued_pretrain_partial

args = [
    "--model_name",
    model_name,
    "--corpus_dir",
    str(to_local_path(corpus_dir)),
    "--output_dir",
    str(to_local_path(output_dir)),
    "--eval_prompts",
    str(to_local_path(eval_prompts)),
    "--dtype",
    dtype,
    "--attention",
    attention,
    "--device_map",
    device_map,
    "--cpu_memory",
    cpu_memory,
    "--train_last_n_layers",
    train_last_n_layers,
    "--per_device_train_batch_size",
    per_device_train_batch_size,
    "--gradient_accumulation_steps",
    gradient_accumulation_steps,
    "--num_train_epochs",
    num_train_epochs,
    "--learning_rate",
    learning_rate,
    "--save_steps",
    save_steps,
    "--logging_steps",
    logging_steps,
    "--dataloader_num_workers",
    dataloader_num_workers,
]
if max_memory:
    args.extend(["--max_memory", max_memory])
args.append("--mxfp4_dequantize" if mxfp4_dequantize else "--no-mxfp4_dequantize")
args.append("--compile_model" if compile_model else "--no-compile_model")

result = run_cli(continued_pretrain_partial.main, "continued_pretrain_partial.py", args)
if result:
    raise RuntimeError(f"continued_pretrain_partial failed with exit code {result}")
