#!/usr/bin/env python3

"""
continued_pretrain_partial_optimized.py

Optimized continued pretraining script for:
- A100 40GB
- BF16 tensor-core training
- Partial-layer continued pretraining
- Scientific / technical corpora
- Proper sequence packing
- Validation perplexity tracking
- FlashAttention2
- Fused AdamW
- Tensor-core friendly settings

Recommended install:

pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  flash-attn \
  sentencepiece \
  evaluate

Example:

python continued_pretrain_partial_optimized.py \
    --dataset_dir ./science_texts \
    --output_dir ./output_science \
    --eval_prompts ./eval_prompts.txt
"""

import os
import math
import argparse
from pathlib import Path

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ============================================================
# Tensor Core / A100 optimizations
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ============================================================
# Recommended models
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"

# Good alternatives:
# "mistralai/Mistral-7B-v0.1"
# "meta-llama/Llama-3.2-3B"

# ============================================================
# Dataset loader
# ============================================================

def load_text_files(dataset_dir):

    texts = []

    paths = list(Path(dataset_dir).rglob("*.txt"))

    print(f"\nFound {len(paths)} text files\n")

    for path in paths:

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            # Skip tiny docs
            if len(content) < 1000:
                continue

            texts.append({"text": content})

        except Exception as e:
            print(f"Skipping {path}: {e}")

    return Dataset.from_list(texts)

# ============================================================
# Tokenization + sequence packing
# ============================================================

def tokenize_dataset(dataset, tokenizer, seq_len=2048):

    print("\nTokenizing dataset...\n")

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=max(os.cpu_count() // 2, 1),
    )

    print("\nPacking sequences...\n")

    def group_texts(examples):

        concatenated = {}

        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])

        total_length = len(concatenated["input_ids"])

        total_length = (total_length // seq_len) * seq_len

        result = {}

        for k, t in concatenated.items():
            result[k] = [
                t[i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]

        result["labels"] = result["input_ids"].copy()

        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=max(os.cpu_count() // 2, 1),
    )

    return lm_dataset

# ============================================================
# Freeze lower layers
# ============================================================

def freeze_lower_layers(model, train_last_n_layers=8):

    layers = model.model.layers

    total_layers = len(layers)

    freeze_until = total_layers - train_last_n_layers

    print("\n======================================")
    print(f"Total transformer layers: {total_layers}")
    print(f"Training last {train_last_n_layers} layers")
    print(f"Freezing first {freeze_until} layers")
    print("======================================\n")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Train upper transformer blocks
    for idx in range(freeze_until, total_layers):

        for param in layers[idx].parameters():
            param.requires_grad = True

    # Train final norm
    if hasattr(model.model, "norm"):

        for param in model.model.norm.parameters():
            param.requires_grad = True

    # Train LM head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Explicitly keep embeddings frozen
    if hasattr(model.model, "embed_tokens"):

        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {(100 * trainable / total):.2f}%\n")

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def run_eval(model, tokenizer, prompts):

    model.eval()

    print("\n================ EVAL ================\n")

    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        print("=" * 80)
        print(f"PROMPT:\n{prompt}\n")
        print(f"OUTPUT:\n{text}\n")

# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):

    loss = eval_pred.loss

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "perplexity": perplexity,
    }

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--eval_prompts",
        required=True,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--train_last_n_layers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    # ========================================================
    # Load tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # Load model
    # ========================================================

    print("\nLoading model...\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )

    # Required with gradient checkpointing
    model.config.use_cache = False

    # PyTorch 2.x compile
    print("\nCompiling model...\n")

    model = torch.compile(model)

    # ========================================================
    # Freeze lower layers
    # ========================================================

    freeze_lower_layers(
        model,
        train_last_n_layers=args.train_last_n_layers,
    )

    # ========================================================
    # Load dataset
    # ========================================================

    print("\nLoading dataset...\n")

    dataset = load_text_files(args.dataset_dir)

    print(dataset)

    # ========================================================
    # Train/validation split
    # ========================================================

    dataset = dataset.train_test_split(
        test_size=0.01,
        seed=42,
    )

    train_dataset = tokenize_dataset(
        dataset["train"],
        tokenizer,
        seq_len=args.seq_len,
    )

    eval_dataset = tokenize_dataset(
        dataset["test"],
        tokenizer,
        seq_len=args.seq_len,
    )

    # ========================================================
    # Eval prompts
    # ========================================================

    prompts = Path(args.eval_prompts).read_text().splitlines()

    prompts = [
        p.strip()
        for p in prompts
        if p.strip()
    ]

    # ========================================================
    # Baseline evaluation
    # ========================================================

    print("\nRUNNING BASELINE EVALUATION\n")

    run_eval(model, tokenizer, prompts)

    # ========================================================
    # Training arguments
    # ========================================================

    training_args = TrainingArguments(

        output_dir=args.output_dir,

        overwrite_output_dir=True,

        num_train_epochs=1,

        learning_rate=2e-6,

        weight_decay=0.01,

        warmup_ratio=0.03,

        lr_scheduler_type="cosine",

        per_device_train_batch_size=2,

        gradient_accumulation_steps=16,

        bf16=True,

        tf32=True,

        logging_steps=1,

        save_steps=250,

        eval_steps=250,

        evaluation_strategy="steps",

        save_total_limit=2,

        report_to="none",

        gradient_checkpointing=True,

        optim="adamw_torch_fused",

        dataloader_num_workers=8,

        dataloader_pin_memory=True,

        dataloader_persistent_workers=True,

        max_grad_norm=1.0,

        remove_unused_columns=False,
    )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # ========================================================
    # Train
    # ========================================================

    print("\nSTARTING CONTINUED PRETRAINING\n")

    trainer.train()

    # ========================================================
    # Final evaluation
    # ========================================================

    print("\nRUNNING FINAL EVALUATION\n")

    run_eval(model, tokenizer, prompts)

    # ========================================================
    # Save model
    # ========================================================

    print("\nSaving model...\n")

    trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)

    print("\nDONE\n")

if __name__ == "__main__":
    main()