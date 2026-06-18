#!/usr/bin/env python
"""
Distributed LoRA fine-tuning via PySpark TorchDistributor.

IMPORTANT: checkpoint and adapter output paths must be on shared storage
visible to all workers (DBFS, NFS, S3, etc.). Pass --shared-output-dir so the
driver and workers agree on a single checkpoint location.

Single-node multi-GPU (default, no Spark cluster needed):
    python src/train_lora_spark.py --shared-output-dir /mnt/shared

Multi-node (e.g. 2 nodes × 4 GPUs = 8 processes):
    python src/train_lora_spark.py \\
        --num-processes 8 \\
        --no-local-mode \\
        --shared-output-dir /dbfs/mnt/shared

Scaling notes:
  - per_device_train_batch_size stays fixed per GPU; effective batch grows with
    num_processes (8 GPUs × bs 8 × accum 2 = effective 128).
  - Scale the learning rate linearly if you increase effective batch size beyond
    the single-GPU baseline of 16 (lr × num_processes).
  - average_tokens_across_devices=True is required for correct loss normalisation
    under DDP when sequences have variable length.
"""

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401


# ---------------------------------------------------------------------------
# Training function — runs inside each distributed worker process.
# Must be fully self-contained: no driver-side objects, no module-level state.
# TorchDistributor initialises torch.distributed before calling this; SFTTrainer
# / accelerate detect the active process group and enable DDP automatically.
# ---------------------------------------------------------------------------

def train_fn(
    base_model: str,
    train_file: str,
    lora_dir: str,
    adapter_dir: str,
) -> None:
    import os
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.runtime_env import env_float, env_int

    default_dataloader_num_workers = env_int("DEFAULT_DATALOADER_NUM_WORKERS", 4)
    default_gradient_accumulation_steps = env_int("DEFAULT_GRADIENT_ACCUMULATION_STEPS", 8)
    default_learning_rate = env_float("DEFAULT_LEARNING_RATE", 2e-4)
    default_logging_steps = env_int("DEFAULT_LOGGING_STEPS", 1)
    default_lora_rank = env_int("DEFAULT_LORA_RANK", 16)
    default_max_length = env_int("DEFAULT_MAX_LENGTH", 512)
    default_num_train_epochs = env_float("DEFAULT_NUM_TRAIN_EPOCHS", 1.0)
    default_per_device_train_batch_size = env_int("DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE", 1)
    default_save_steps = env_int("DEFAULT_SAVE_STEPS", 100)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[rank {rank}/{world_size}] local_rank={local_rank}")

    # Precision: bf16 on Ampere+ (sm80+), fp16 on older NVIDIA hardware.
    if torch.cuda.is_bf16_supported():
        precision = {"fp16": False, "bf16": True}
    else:
        precision = {"fp16": True, "bf16": False}

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map={"": local_rank} loads each worker's copy directly onto its
    # assigned GPU.  Do NOT use device_map="auto" here — that activates model
    # parallelism and conflicts with DDP.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": local_rank},
    )

    peft_config = LoraConfig(
        r=default_lora_rank,
        lora_alpha=default_lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.config.use_cache = False
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    if rank == 0:
        model.print_trainable_parameters()

    # HuggingFace Trainer applies DistributedSampler automatically when
    # torch.distributed is active, so each worker sees a disjoint shard.
    dataset = load_dataset("json", data_files=train_file, split="train")

    args = SFTConfig(
        output_dir=lora_dir,
        learning_rate=default_learning_rate,
        per_device_train_batch_size=default_per_device_train_batch_size,
        gradient_accumulation_steps=default_gradient_accumulation_steps,
        num_train_epochs=default_num_train_epochs,
        logging_steps=default_logging_steps,
        save_steps=default_save_steps,
        max_length=default_max_length,
        packing=False,
        **precision,
        max_grad_norm=0.3,
        # Correct loss normalisation when sequence lengths vary across workers.
        average_tokens_across_devices=True,
        # LoRA only trains adapter params; unused base params would trigger DDP
        # warnings without this flag.
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_num_workers=default_dataloader_num_workers,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # resume_from_checkpoint=True is a no-op when no checkpoint exists yet.
    trainer.train(resume_from_checkpoint=True)

    # Trainer.save_model() already guards internally, but be explicit for the
    # tokenizer to avoid N copies written to the same path.
    if rank == 0:
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[rank 0] Saved LoRA adapter to {adapter_dir}")


# ---------------------------------------------------------------------------
# Driver — sets up TorchDistributor and launches train_fn on each worker.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    from utils.runtime_env import env_path, env_str

    adapter_dir = env_path("ADAPTER_DIR", "output/lora/final")
    base_model = env_str("BASE_MODEL")
    lora_dir = env_path("LORA_DIR", "output/lora")
    train_file = env_path("TRAIN_FILE", "corpus/train.jsonl")

    parser = argparse.ArgumentParser(description="Distributed LoRA fine-tuning via PySpark.")
    parser.add_argument("--base-model", default=base_model)
    parser.add_argument("--train-file", default=str(train_file))
    parser.add_argument("--lora-dir", default=str(lora_dir))
    parser.add_argument("--adapter-dir", default=str(adapter_dir))
    parser.add_argument(
        "--shared-output-dir",
        default=None,
        help="Shared storage root for lora/ checkpoints and final adapter.",
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument(
        "--local-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run TorchDistributor in Spark local mode.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
    except ImportError:
        raise SystemExit(
            "pyspark is not installed. Run: pip install pyspark"
        )

    args = parse_args()

    train_file = Path(args.train_file).expanduser()
    lora_dir = Path(args.lora_dir).expanduser()
    adapter_dir = Path(args.adapter_dir).expanduser()

    if args.shared_output_dir:
        shared = Path(args.shared_output_dir).expanduser()
        lora_dir = shared / "lora"
        adapter_dir = lora_dir / "final"

    lora_dir.mkdir(parents=True, exist_ok=True)

    print(f"TorchDistributor: num_processes={args.num_processes}, local_mode={args.local_mode}")
    print(f"Train file : {train_file}")
    print(f"Output dir : {lora_dir}")

    distributor = TorchDistributor(
        num_processes=args.num_processes,
        local_mode=args.local_mode,
        use_gpu=True,
    )
    distributor.run(
        train_fn,
        args.base_model,
        str(train_file),
        str(lora_dir),
        str(adapter_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
