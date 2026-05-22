#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from model_runtime import load_base_model, load_tokenizer
from project_config import ADAPTER_DIR, BASE_MODEL, LORA_DIR, TRAIN_FILE


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
        r=8,
        lora_alpha=16,
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
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        max_length=192,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        average_tokens_across_devices=False,
        report_to="none",
        dataloader_num_workers=0,
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
