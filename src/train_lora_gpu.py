#!/usr/bin/env python

import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from model_runtime import load_base_model, load_tokenizer
from project_config import ADAPTER_DIR, LORA_DIR, TRAIN_FILE


def require_nvidia_gpu() -> torch.device:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.version.hip:", torch.version.hip)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run because train_lora_gpu.py must use an NVIDIA GPU."
        )

    if torch.version.cuda is None:
        raise RuntimeError(
            "This PyTorch build does not appear to have CUDA support. Refusing to run."
        )

    device_name = torch.cuda.get_device_name(0)
    print("GPU 0:", device_name)

    if "NVIDIA" not in device_name.upper():
        print(
            f"Warning: detected CUDA device name does not contain 'NVIDIA': {device_name}",
            file=sys.stderr,
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def _precision_flags() -> dict:
    """
    NVIDIA-only precision selection:
    - prefer bf16 when supported
    - otherwise fall back to fp16
    """
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}

    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}

    return {"fp16": True, "bf16": False}


def _attn_implementation() -> str:
    """
    For NVIDIA/CUDA, SDPA is the default fast path.
    """
    return "sdpa"


def print_device_info(model) -> None:
    first_param_device = next(model.parameters()).device
    print("Model first param device:", first_param_device)
    print("Backend: CUDA")
    print("CUDA version:", torch.version.cuda)
    print("GPU available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print(
            "CUDA memory allocated:",
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
        )
        print(
            "CUDA memory reserved:",
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB",
        )


def assert_model_on_cuda(model) -> None:
    first_param_device = next(model.parameters()).device
    print("Verifying model device:", first_param_device)

    if first_param_device.type != "cuda":
        raise RuntimeError(
            f"Model is on {first_param_device}, not CUDA. Refusing to continue."
        )


def prepare_lora_model(model):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Disabled to trade VRAM for speed/throughput on the A10.
    # Re-enable if you later increase batch/sequence length and hit OOM.
    # model.gradient_checkpointing_enable()

    model.config.use_cache = False
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, dataset):
    args = SFTConfig(
        output_dir=str(LORA_DIR),
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        max_length=384,
        packing=False,
        **_precision_flags(),
        max_grad_norm=0.3,
        average_tokens_across_devices=False,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def main() -> int:
    device = require_nvidia_gpu()
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_base_model(attn_implementation=_attn_implementation())

    model = model.to(device)
    torch.cuda.empty_cache()
    print_device_info(model)
    assert_model_on_cuda(model)

    dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")

    model = prepare_lora_model(model)
    model = model.to(device)
    assert_model_on_cuda(model)
    print_device_info(model)

    trainer = build_trainer(model, tokenizer, dataset)

    assert_model_on_cuda(trainer.model)
    print(
        "Starting training on:",
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
