#!/usr/bin/env python
"""
Distributed LoRA fine-tuning via PySpark TorchDistributor.

IMPORTANT: checkpoint and adapter output paths must be on shared storage
visible to all workers (DBFS, NFS, S3, etc.).  Set LLAMA_SHARED_OUTPUT_DIR
to point there before running, or the per-worker saves will be silently
inconsistent.

Single-node multi-GPU (default, no Spark cluster needed):
    LLAMA_SHARED_OUTPUT_DIR=/mnt/shared python src/train_lora_spark.py

Multi-node (e.g. 2 nodes × 4 GPUs = 8 processes):
    LLAMA_SPARK_NUM_PROCESSES=8 \\
    LLAMA_SPARK_LOCAL_MODE=false \\
    LLAMA_SHARED_OUTPUT_DIR=/dbfs/mnt/shared \\
    python src/train_lora_spark.py

Scaling notes:
  - per_device_train_batch_size stays fixed per GPU; effective batch grows with
    num_processes (8 GPUs × bs 8 × accum 2 = effective 128).
  - Scale the learning rate linearly if you increase effective batch size beyond
    the single-GPU baseline of 16 (lr × num_processes).
  - average_tokens_across_devices=True is required for correct loss normalisation
    under DDP when sequences have variable length.
"""

import os
import sys
from pathlib import Path


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
        r=16,
        lora_alpha=32,
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
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        max_length=384,
        packing=False,
        **precision,
        max_grad_norm=0.3,
        # Correct loss normalisation when sequence lengths vary across workers.
        average_tokens_across_devices=True,
        # LoRA only trains adapter params; unused base params would trigger DDP
        # warnings without this flag.
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_num_workers=4,
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

def main() -> int:
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
    except ImportError:
        raise SystemExit(
            "pyspark is not installed. Run: pip install pyspark"
        )

    # project_config is driver-side only; pass resolved strings into train_fn
    # so workers don't need to re-evaluate env vars against the driver's cwd.
    from project_config import ADAPTER_DIR, LORA_DIR, TRAIN_FILE, BASE_MODEL

    num_processes = int(os.environ.get("LLAMA_SPARK_NUM_PROCESSES", "1"))
    local_mode = os.environ.get("LLAMA_SPARK_LOCAL_MODE", "true").lower() not in (
        "false", "0", "no"
    )

    lora_dir = LORA_DIR
    adapter_dir = ADAPTER_DIR

    # Override output paths with shared storage when set, so all workers and
    # the driver agree on a single checkpoint location.
    shared_root = os.environ.get("LLAMA_SHARED_OUTPUT_DIR", "").strip()
    if shared_root:
        shared = Path(shared_root).expanduser()
        lora_dir = shared / "lora"
        adapter_dir = lora_dir / "final"

    lora_dir.mkdir(parents=True, exist_ok=True)

    print(f"TorchDistributor: num_processes={num_processes}, local_mode={local_mode}")
    print(f"Train file : {TRAIN_FILE}")
    print(f"Output dir : {lora_dir}")

    distributor = TorchDistributor(
        num_processes=num_processes,
        local_mode=local_mode,
        use_gpu=True,
    )
    distributor.run(
        train_fn,
        BASE_MODEL,
        str(TRAIN_FILE),
        str(lora_dir),
        str(adapter_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
