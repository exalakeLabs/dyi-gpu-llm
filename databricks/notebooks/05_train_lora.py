# Databricks notebook source
# MAGIC %md
# MAGIC # Train LoRA Adapter

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
    widget_text,
)

model_root = widget_text("model_root", "/dbfs/FileStore/llama32-local", "Model/data root")
model_name = widget_text("model_name", os.environ.get("BASE_MODEL", "openai/gpt-oss-20b"), "Base model")
train_file = widget_text("train_file", os.environ.get("TRAIN_FILE", f"{model_root}/corpus/train.jsonl"), "Train JSONL")
output_dir = widget_text("output_dir", os.environ.get("LORA_DIR", f"{model_root}/model/lora"), "LoRA output dir")
max_length = widget_text("max_length", os.environ.get("DEFAULT_MAX_LENGTH", "512"), "Max length")
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
lora_rank = widget_text("lora_rank", os.environ.get("DEFAULT_LORA_RANK", "16"), "LoRA rank")
save_steps = widget_text("save_steps", os.environ.get("DEFAULT_SAVE_STEPS", "100"), "Save steps")
logging_steps = widget_text("logging_steps", os.environ.get("DEFAULT_LOGGING_STEPS", "1"), "Logging steps")
dataloader_num_workers = widget_text("dataloader_num_workers", "2", "Dataloader workers")
gradient_checkpointing = widget_bool("gradient_checkpointing", False, "Gradient checkpointing")

set_default_env(model_root)
os.environ["BASE_MODEL"] = model_name
os.environ["GENERATOR_MODEL"] = model_name
os.environ["TRAIN_FILE"] = str(to_local_path(train_file))
os.environ["LORA_DIR"] = str(to_local_path(output_dir))
os.environ["ADAPTER_DIR"] = str(to_local_path(output_dir))
ensure_directories()

import train_lora_gpu

args = [
    "--model-name",
    model_name,
    "--train-file",
    str(to_local_path(train_file)),
    "--output-dir",
    str(to_local_path(output_dir)),
    "--max-length",
    max_length,
    "--per-device-train-batch-size",
    per_device_train_batch_size,
    "--gradient-accumulation-steps",
    gradient_accumulation_steps,
    "--num-train-epochs",
    num_train_epochs,
    "--learning-rate",
    learning_rate,
    "--lora-rank",
    lora_rank,
    "--save-steps",
    save_steps,
    "--logging-steps",
    logging_steps,
    "--dataloader-num-workers",
    dataloader_num_workers,
]
args.append("--gradient-checkpointing" if gradient_checkpointing else "--no-gradient-checkpointing")

result = run_cli(train_lora_gpu.main, "train_lora_gpu.py", args)
if result:
    raise RuntimeError(f"train_lora_gpu failed with exit code {result}")
