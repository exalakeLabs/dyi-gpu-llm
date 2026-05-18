#!/usr/bin/env python

from __future__ import annotations

import inspect
import math
import os
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

transformers.logging.set_verbosity_info()

# A100 supports TF32 for matmul/conv — ensure it's active regardless of torch defaults.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise ValueError("MODEL_ROOT is not set")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
TRAIN_FILE = os.environ.get("TRAIN_FILE", f"{MODEL_ROOT}/data/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODEL_ROOT}/output/lora")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

# A100-40GB has ample VRAM for Mistral-7B (~14 GB bf16). Batch 8 + accum 4 keeps
# effective batch at 32 while maximising GPU occupancy.
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# LoRA rank — bumped to 16 (alpha=32) to make better use of available VRAM.
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))

# Verbose logging / checkpoint defaults
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "1"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "100"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "3"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))


def print_cuda_debug() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is for NVIDIA GPU training.")
    if torch.cuda.is_bf16_supported():
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

    # Flash Attention 2 is natively supported on A100 and reduces both memory
    # and latency for attention. Fall back silently if flash-attn isn't installed.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("Flash Attention 2 enabled.")
    except (ValueError, ImportError):
        print("Flash Attention 2 not available; falling back to eager attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
        )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    first_param = next(model.parameters())
    print(f"Model first param device: {first_param.device}")
    print("Backend: CUDA")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")

    return model


def attach_lora(model):
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
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        base_model_name_or_path=MODEL_NAME,
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


def tokenize_dataset(dataset, tokenizer):
    if "text" in dataset.column_names:

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
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
                max_length=MAX_LENGTH,
                padding=False,
            )

    else:
        raise ValueError(
            f"Expected either 'text' or 'messages' column in {TRAIN_FILE}, "
            f"but found: {dataset.column_names}"
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )
    return tokenized


def estimate_warmup_steps(num_examples: int) -> int:
    effective_batch = max(1, PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * NUM_TRAIN_EPOCHS))
    return max(1, int(total_steps * WARMUP_RATIO))


def make_training_arguments(output_dir: Path, use_bf16: bool, use_fp16: bool, warmup_steps: int):
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch_fused",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        report_to="none",
        log_level="info",
        logging_first_step=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
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


def main() -> int:
    print_cuda_debug()

    train_path = Path(TRAIN_FILE)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, tokenizer)
    model = attach_lora(model)

    first_param = next(model.parameters())
    print(f"Verifying model device: {first_param.device}")

    dataset = load_train_dataset(TRAIN_FILE)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    first_param = next(model.parameters())
    print(f"Verifying model device: {first_param.device}")

    print(f"Starting training on: {torch.cuda.get_device_name(0)}")

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    warmup_steps = estimate_warmup_steps(len(tokenized_dataset))

    training_args = make_training_arguments(
        output_dir=output_dir,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        warmup_steps=warmup_steps,
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
