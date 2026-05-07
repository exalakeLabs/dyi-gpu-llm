# Databricks notebook source
# =============================================================================
# 03 · Build Training Pairs & Fine-tune LoRA
# Generates continuation-style training pairs from the prepared corpus, then
# runs LoRA fine-tuning tracked in MLflow.
# Requires a GPU cluster (single-node multi-GPU supported via TorchDistributor).
# =============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03 · Build Training Pairs & Fine-tune LoRA
# MAGIC
# MAGIC **Stage A — Training pair generation**
# MAGIC Reads `{dbfs_root}/prepared/`, builds continuation-style JSONL pairs,
# MAGIC and writes them to `{dbfs_root}/data/train.jsonl`.
# MAGIC
# MAGIC **Stage B — LoRA fine-tuning**
# MAGIC Fine-tunes `Qwen/Qwen2.5-3B-Instruct` with LoRA (r=8) using
# MAGIC [TorchDistributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/torch-distributor.html)
# MAGIC for single-node multi-GPU training.  All metrics and the final adapter
# MAGIC are logged to the managed MLflow experiment.
# MAGIC
# MAGIC **Cluster**: GPU required (`g4dn.xlarge` / `g5.xlarge` minimum).
# MAGIC **Estimated time**: 2–8 h depending on corpus size and cluster type.

# COMMAND ----------

# MAGIC %pip install --quiet peft accelerate trl>=0.8.0 truststore

# COMMAND ----------

# Widget parameters — edit before running
dbutils.widgets.text(    "dbfs_root",         "/Volumes/customer_success/exalabs_writeback/llrun",   "DBFS Root")
dbutils.widgets.text(    "base_model",        "Qwen/Qwen2.5-3B-Instruct",  "Base Model")
dbutils.widgets.text(    "num_gpus",          "8",                          "GPUs (TorchDistributor num_processes)")
dbutils.widgets.dropdown("local_mode",        "false",  ["true", "false"],   "local_mode (false = multi-node)")
dbutils.widgets.text(    "num_epochs",        "1",                          "Training Epochs")
dbutils.widgets.text(    "batch_size",        "2",                          "Per-device Batch Size")
dbutils.widgets.text(    "grad_accum",        "8",                          "Gradient Accumulation Steps")
dbutils.widgets.text(    "learning_rate",     "3e-4",                       "Learning Rate")
dbutils.widgets.text(    "max_length",        "512",                        "Max Sequence Length")
dbutils.widgets.text(    "lora_r",            "16",                         "LoRA rank (r)")
dbutils.widgets.text(    "max_pair_files",    "0",                          "Max files for pair gen (0=all)")

# COMMAND ----------

import os, sys, math
from pathlib import Path

dbfs_root     = dbutils.widgets.get("dbfs_root")
base_model    = dbutils.widgets.get("base_model")
num_gpus      = int(dbutils.widgets.get("num_gpus"))
local_mode    = dbutils.widgets.get("local_mode").lower() == "true"
num_epochs    = float(dbutils.widgets.get("num_epochs"))
batch_size    = int(dbutils.widgets.get("batch_size"))
grad_accum    = int(dbutils.widgets.get("grad_accum"))
learning_rate = float(dbutils.widgets.get("learning_rate"))
max_length    = int(dbutils.widgets.get("max_length"))
lora_r        = int(dbutils.widgets.get("lora_r"))
max_pair_files = int(dbutils.widgets.get("max_pair_files"))

os.environ["LLAMA_DBFS_ROOT"]     = dbfs_root
os.environ["LLAMA_PREPARED_DIR"]  = f"{dbfs_root}/prepared"
os.environ["LLAMA_DATA_DIR"]      = f"{dbfs_root}/data"
os.environ["LLAMA_OUTPUT_DIR"]    = f"{dbfs_root}/output"
os.environ["MODEL_NAME"]          = base_model

_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

prepared_dir = Path(f"{dbfs_root}/prepared")
data_dir     = Path(f"{dbfs_root}/data")
output_dir   = Path(f"{dbfs_root}/output")
lora_dir     = output_dir / "lora"
adapter_dir  = lora_dir / "final"

data_dir.mkdir(parents=True, exist_ok=True)
lora_dir.mkdir(parents=True, exist_ok=True)

train_file = data_dir / "train.jsonl"
val_file   = data_dir / "val.jsonl"

print(f"DBFS root      : {dbfs_root}")
print(f"Base model     : {base_model}")
print(f"GPUs           : {num_gpus}  (local_mode={local_mode})")
print(f"LoRA r / alpha : {lora_r} / {lora_r * 2}")
print(f"Epochs         : {num_epochs}")
print(f"Effective batch: {batch_size} × {num_gpus} GPU(s) × accum {grad_accum} = {batch_size * num_gpus * grad_accum}")

# COMMAND ----------

# MAGIC %md ### Stage A · Generate training pairs

# COMMAND ----------

import random, json

# Re-inject src/ — %pip install restarts the kernel and wipes sys.path.
_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from make_training_pairs import (
    iter_examples, load_and_clean_book, to_record, write_jsonl
)

prep_files = sorted(prepared_dir.glob("*.txt"))
if max_pair_files > 0:
    prep_files = prep_files[:max_pair_files]

print(f"Processing {len(prep_files)} prepared files…")

all_examples = []
kept = skipped = 0

for path in prep_files:
    try:
        title, text = load_and_clean_book(path)
    except Exception as e:
        print(f"[WARN] {path.name}: {e}")
        skipped += 1
        continue

    if len(text) < 4000:
        skipped += 1
        continue

    examples = list(iter_examples(
        text=text,
        title=title,
        source_name=path.name,
        target_window_chars=2600,
        max_window_chars=3200,
        prompt_chars=900,
        min_prompt_chars=500,
        min_response_chars=350,
        stride_chars=1200,
    ))
    if not examples:
        skipped += 1
        continue

    all_examples.extend(examples)
    kept += 1

if not all_examples:
    raise RuntimeError("No training examples created — check prepared_dir.")

rng = random.Random(42)
rng.shuffle(all_examples)

records    = [to_record(ex, "messages",
                        "Continue the passage faithfully, preserving its meaning, tone, and style.",
                        "Continue the following passage from the book.\n\nTitle: {title}\n\n{prompt}")
              for ex in all_examples]

split_idx  = int(len(records) * 0.98)
train_recs = records[:split_idx]
val_recs   = records[split_idx:]

write_jsonl(train_file, train_recs)
write_jsonl(val_file,   val_recs)

print(f"\nBooks kept   : {kept}")
print(f"Books skipped: {skipped}")
print(f"Train records: {len(train_recs)}  →  {train_file}")
print(f"Val records  : {len(val_recs)}   →  {val_file}")

# COMMAND ----------

# MAGIC %md ### Stage B · LoRA fine-tuning (TorchDistributor)

# COMMAND ----------

import mlflow

# mlflow.set_experiment() auto-creates the experiment on Databricks —
# os.makedirs won't work here since /Users/... is a Workspace path, not a filesystem path.
MLFLOW_EXPERIMENT = "/Users/alex.gauthier@sigmacomputing.com/Lora-slow/Lora-MLFLOW"

mlflow.set_experiment(MLFLOW_EXPERIMENT)


# All serialisable hyperparams are captured up-front so MLflow records them
# even if the distributed job crashes partway through training.
hparams = {
    "base_model":          base_model,
    "num_gpus":            num_gpus,
    "max_length":          max_length,
    "per_device_batch":    batch_size,
    "gradient_accumulation": grad_accum,
    "effective_batch":     batch_size * num_gpus * grad_accum,
    "epochs":              num_epochs,
    "learning_rate":       learning_rate,
    "lora_r":              lora_r,
    "lora_alpha":          lora_r * 2,
    "num_train_examples":  len(train_recs),
}

# ---------------------------------------------------------------------------
# Worker function — runs inside each GPU process.
# Must be fully self-contained: no driver-side closures or module-level state.
# ---------------------------------------------------------------------------
def _train_worker(
    base_model: str,
    train_file: str,
    lora_dir: str,
    adapter_dir: str,
    num_epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_length: int,
    lora_r: int,
) -> None:
    import math, os
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[rank {rank}/{world_size}] local_rank={local_rank}  GPU={torch.cuda.get_device_name(local_rank)}")

    use_bf16 = torch.cuda.is_bf16_supported()
    precision = {"bf16": use_bf16, "fp16": not use_bf16}

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Use device_map={"": local_rank} (not "auto") to prevent model parallelism
    # from conflicting with DDP across ranks.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    if rank == 0:
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=train_file, split="train")

    # Estimate warmup steps
    eff_batch     = max(1, batch_size * grad_accum * world_size)
    steps_per_ep  = max(1, math.ceil(len(dataset) / eff_batch))
    total_steps   = max(1, math.ceil(steps_per_ep * num_epochs))
    warmup_steps  = max(1, int(total_steps * 0.03))

    sft_args = SFTConfig(
        output_dir=lora_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_seq_length=max_length,
        packing=False,
        logging_steps=1,
        logging_first_step=True,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        report_to="none",          # MLflow logged from driver only
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        disable_tqdm=True,
        **precision,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    last_ckpt = get_last_checkpoint(lora_dir)
    if last_ckpt:
        print(f"[rank {rank}] Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt or False)

    if rank == 0:
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[rank 0] Saved LoRA adapter → {adapter_dir}")


# ---------------------------------------------------------------------------
# Launch via TorchDistributor
# ---------------------------------------------------------------------------
from pyspark.ml.torch.distributor import TorchDistributor

with mlflow.start_run() as run:
    mlflow.log_params(hparams)
    print(f"MLflow run: {run.info.run_id}")

    distributor = TorchDistributor(
        num_processes=num_gpus,
        local_mode=local_mode,   # True = single-node multi-GPU; False = multi-node
        use_gpu=True,
    )
    distributor.run(
        _train_worker,
        base_model,
        str(train_file),
        str(lora_dir),
        str(adapter_dir),
        num_epochs,
        batch_size,
        grad_accum,
        learning_rate,
        max_length,
        lora_r,
    )

    # Log the adapter as an MLflow artifact so it can be retrieved without
    # knowing the DBFS path.
    if adapter_dir.exists():
        mlflow.log_artifacts(str(adapter_dir), artifact_path="lora_adapter")
        print(f"MLflow artifact logged: lora_adapter")

print(f"\nLoRA adapter saved to : {adapter_dir}")
print(f"MLflow experiment     : {MLFLOW_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md ### LoRA adapter size

# COMMAND ----------

if adapter_dir.exists():
    adapter_files = list(adapter_dir.iterdir())
    display(spark.createDataFrame(
        [(f.name, f"{f.stat().st_size / 1024:.1f} KB") for f in adapter_files],
        ["filename", "size"]
    ))
else:
    print("Adapter directory not yet present — training may still be running.")

# COMMAND ----------

# MAGIC %md ### ✅ Training complete — proceed to notebook 04.
