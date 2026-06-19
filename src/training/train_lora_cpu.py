#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from inference.model_runtime import load_base_model, load_tokenizer
from utils.runtime_env import env_float, env_int, env_path, env_str

ADAPTER_DIR = env_path("ADAPTER_DIR", "output/lora/final")
BASE_MODEL = env_str("BASE_MODEL")
DEFAULT_DATALOADER_NUM_WORKERS = env_int("DEFAULT_DATALOADER_NUM_WORKERS", 4)
DEFAULT_GRADIENT_ACCUMULATION_STEPS = env_int("DEFAULT_GRADIENT_ACCUMULATION_STEPS", 8)
DEFAULT_LEARNING_RATE = env_float("DEFAULT_LEARNING_RATE", 2e-4)
DEFAULT_LOGGING_STEPS = env_int("DEFAULT_LOGGING_STEPS", 1)
DEFAULT_LORA_RANK = env_int("DEFAULT_LORA_RANK", 16)
DEFAULT_MAX_LENGTH = env_int("DEFAULT_MAX_LENGTH", 512)
DEFAULT_NUM_TRAIN_EPOCHS = env_float("DEFAULT_NUM_TRAIN_EPOCHS", 1.0)
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = env_int("DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE", 1)
DEFAULT_SAVE_STEPS = env_int("DEFAULT_SAVE_STEPS", 100)
LORA_DIR = env_path("LORA_DIR", "output/lora")
TRAIN_FILE = env_path("TRAIN_FILE", "corpus/train.jsonl")


def print_device_info(model) -> None:
    print("Model first param device:", next(model.parameters()).device)
    print("Torch version:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Inter-op threads:", torch.get_num_interop_threads())
    if torch.version.hip:
        print("HIP version:", torch.version.hip)
    if torch.version.cuda:
        print("CUDA version:", torch.version.cuda)
    print("GPU available:", torch.cuda.is_available())


def prepare_lora_model(model):
    peft_config = LoraConfig(
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # On CPU this can still help memory, though it may slow training.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    # Keep trainable params in float32 for CPU training stability.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, dataset, lora_dir: Path):
    args = SFTConfig(
        output_dir=str(lora_dir),
        learning_rate=DEFAULT_LEARNING_RATE,
        per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        logging_steps=DEFAULT_LOGGING_STEPS,
        save_steps=DEFAULT_SAVE_STEPS,
        max_length=DEFAULT_MAX_LENGTH,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        average_tokens_across_devices=False,
        report_to="none",
        dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=False,
        # Force CPU training
        use_cpu=True,
    )

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on CPU.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--lora-dir", default=str(LORA_DIR))
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=os.cpu_count() or 4,
        help="Torch intra-op CPU threads.",
    )
    parser.add_argument(
        "--cpu-interop-threads",
        type=int,
        default=1,
        help="Torch inter-op CPU threads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_file = Path(args.train_file).expanduser()
    lora_dir = Path(args.lora_dir).expanduser()
    adapter_dir = Path(args.adapter_dir).expanduser()
    lora_dir.mkdir(parents=True, exist_ok=True)

    # Optional: keep CPU thread usage under control.
    # Tune these if needed for your machine.
    torch.set_num_threads(args.cpu_threads)

    # Inter-op threads can be kept lower to avoid oversubscription.
    torch.set_num_interop_threads(args.cpu_interop_threads)

    tokenizer = load_tokenizer(args.base_model)

    # Load model without any GPU/ROCm-specific attention assumptions.
    model = load_base_model(args.base_model, attn_implementation="eager")
    model = model.to("cpu")

    print_device_info(model)

    dataset = load_dataset("json", data_files=str(train_file), split="train")
    model = prepare_lora_model(model)

    trainer = build_trainer(model, tokenizer, dataset, lora_dir)
    trainer.train()

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
