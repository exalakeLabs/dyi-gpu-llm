#!/usr/bin/env python

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from model_runtime import is_rocm, patch_rocm_isin
from project_config import BASE_MODEL, LORA_DIR, TRAIN_FILE

transformers.logging.set_verbosity_info()

# "cuda" | "rocm" — set by configure_gpu_backend() in main()
GPU_BACKEND: str = "none"

DEFAULT_MAX_LENGTH = 512
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_NUM_TRAIN_EPOCHS = 1.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_RANK = 16
DEFAULT_LOGGING_STEPS = 1
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 3
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_DATALOADER_NUM_WORKERS = 4


def _rocm_supports_bf16() -> bool:
    """Native bf16 on CDNA2/CDNA3 and RDNA3 (mirrors model_runtime)."""
    if not is_rocm():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return any(
            arch.startswith(p)
            for p in ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101", "gfx1102")
        )
    except Exception:
        return False


def detect_gpu_backend() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU accelerator available. Install PyTorch with CUDA or ROCm and ensure a GPU is visible."
        )
    return "rocm" if is_rocm() else "cuda"


def configure_gpu_backend() -> str:
    """Apply backend-specific torch settings. Returns 'cuda' or 'rocm'."""
    global GPU_BACKEND
    GPU_BACKEND = detect_gpu_backend()
    if GPU_BACKEND == "cuda":
        # A100+ — TF32 speeds matmul/conv on NVIDIA when enabled.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        patch_rocm_isin()
    return GPU_BACKEND


def gpu_bf16_supported() -> bool:
    if GPU_BACKEND == "rocm":
        return _rocm_supports_bf16()
    return torch.cuda.is_bf16_supported()


def training_optimizer() -> str:
    # Fused AdamW is well-tested on CUDA; standard AdamW is safer on ROCm builds.
    return "adamw_torch" if GPU_BACKEND == "rocm" else "adamw_torch_fused"


def print_gpu_debug() -> None:
    print(f"torch: {torch.__version__}")
    print(f"GPU backend: {GPU_BACKEND}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def pick_dtype() -> torch.dtype:
    if gpu_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, tokenizer):
    dtype = pick_dtype()
    common = dict(pretrained_model_name_or_path=model_name, dtype=dtype, device_map="auto")

    if GPU_BACKEND == "cuda":
        # Flash Attention 2 on Ampere+ reduces memory and latency when installed.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                **common,
                attn_implementation="flash_attention_2",
            )
            print("Flash Attention 2 enabled.")
        except (ValueError, ImportError):
            print("Flash Attention 2 not available; falling back to default attention.")
            model = AutoModelForCausalLM.from_pretrained(**common)
    else:
        # ROCm: prefer SDPA (well supported); fall back to default attention.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                **common,
                attn_implementation="sdpa",
            )
            print("SDPA attention enabled.")
        except (ValueError, ImportError):
            print("SDPA not available; falling back to default attention.")
            model = AutoModelForCausalLM.from_pretrained(**common)

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    first_param = next(model.parameters())
    print(f"Model first param device: {first_param.device}")
    print(f"Backend: {GPU_BACKEND.upper()}")
    if torch.version.cuda:
        print(f"CUDA version: {torch.version.cuda}")
    if torch.version.hip:
        print(f"HIP version: {torch.version.hip}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")

    return model


def attach_lora(model, model_name: str, lora_rank: int):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        base_model_name_or_path=model_name,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_train_dataset(train_file: str):
    dataset = load_dataset("json", data_files=train_file, split="train")
    print(f"Generating train split: {len(dataset)} examples")
    return dataset


def format_messages_with_template(messages, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def tokenize_dataset(dataset, tokenizer, max_length: int, train_file: Path):
    if "text" in dataset.column_names:

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    elif "messages" in dataset.column_names:

        def tokenize_fn(batch):
            texts = [
                format_messages_with_template(messages, tokenizer)
                for messages in batch["messages"]
            ]
            return tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    else:
        raise ValueError(
            f"Expected either 'text' or 'messages' column in {train_file}, "
            f"but found: {dataset.column_names}"
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )
    return tokenized


def estimate_warmup_steps(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    return max(1, int(total_steps * warmup_ratio))


def make_training_arguments(
    *,
    output_dir: Path,
    use_bf16: bool,
    use_fp16: bool,
    warmup_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    learning_rate: float,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    dataloader_num_workers: int,
):
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim=training_optimizer(),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="no",
        report_to="none",
        log_level="info",
        logging_first_step=True,
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=dataloader_num_workers,
        disable_tqdm=True,
    )

    # Include optional kwargs only when this transformers version supports them.
    valid = inspect.signature(TrainingArguments.__init__).parameters
    optional = {
        "include_num_input_tokens_seen": True,
        "group_by_length": True,
    }
    for key, val in optional.items():
        if key in valid:
            kwargs[key] = val
        else:
            print(f"Note: transformers {transformers.__version__} does not support {key!r}; skipping.")

    return TrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on GPU.")
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--output-dir", default=str(LORA_DIR))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_gpu_backend()
    print_gpu_debug()

    train_path = Path(args.train_file).expanduser()
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, tokenizer)
    model = attach_lora(model, args.model_name, args.lora_rank)

    first_param = next(model.parameters())
    print(f"Verifying model device: {first_param.device}")

    dataset = load_train_dataset(str(train_path))
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_file=train_path,
    )

    first_param = next(model.parameters())
    print(f"Verifying model device: {first_param.device}")

    print(f"Starting training on: {torch.cuda.get_device_name(0)}")

    use_fp16 = False
    use_bf16 = gpu_bf16_supported()
    warmup_steps = estimate_warmup_steps(
        num_examples=len(tokenized_dataset),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    training_args = make_training_arguments(
        output_dir=output_dir,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No valid checkpoint found. Starting fresh.")
        trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
