#!/usr/bin/env python3

"""
continued_pretrain_partial.py

Optimized continued pretraining script for:
- NVIDIA GPUs ranging from smaller RTX cards to A100-class accelerators
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
  sentencepiece \
  evaluate

Optional CUDA acceleration:

pip install flash-attn --no-build-isolation

Example:

python src/continued_pretrain_partial.py \
    --dataset_dir ./science_texts \
    --output_dir ./output_science \
    --eval_prompts ./eval_prompts.txt

Smaller GPU starting point:

python src/continued_pretrain_partial.py \
    --dataset_dir ./science_texts \
    --eval_prompts ./eval_prompts.txt \
    --seq_len 1024 \
    --train_last_n_layers 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --attention sdpa \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
"""

import argparse
import inspect
import math
import os
from pathlib import Path

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    __version__ as TRANSFORMERS_VERSION,
)
from transformers.trainer_utils import get_last_checkpoint

# ============================================================
# Runtime defaults
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_OUTPUT_DIR = "~/pretrain/output_partial"
DEFAULT_SAVE_STEPS = 250
DEFAULT_NUM_TRAIN_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-6
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_LR_SCHEDULER_TYPE = "cosine"
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 128
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 2048
DEFAULT_TRAIN_LAST_N_LAYERS = 8
DEFAULT_DTYPE = "auto"
DEFAULT_ATTENTION = "auto"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_OPTIM = "adamw_torch_fused"
DEFAULT_DATALOADER_NUM_WORKERS = 8
DEFAULT_FLOAT32_MATMUL_PRECISION = "high"

# Good alternatives:
# "mistralai/Mistral-7B-v0.1"
# "meta-llama/Llama-3.2-3B"


def transformers_dtype_kwarg():
    try:
        version = tuple(int(part) for part in TRANSFORMERS_VERSION.split(".")[:2])
    except ValueError:
        return "dtype"

    return "dtype" if version >= (4, 51) else "torch_dtype"


def resolve_torch_dtype(dtype_name):
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtypes[dtype_name]


def configure_torch_backend(tf32, float32_matmul_precision):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision(float32_matmul_precision)
    print(f"TF32 enabled: {tf32}")
    print(f"float32 matmul precision: {float32_matmul_precision}")


def resolve_device_map(device_map):
    if device_map == "none":
        return None
    if device_map == "single":
        return {"": 0}
    return device_map


def resolve_num_proc(dataset_num_proc):
    if dataset_num_proc > 0:
        return dataset_num_proc
    return max((os.cpu_count() or 2) // 2, 1)


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

def tokenize_dataset(dataset, tokenizer, seq_len=DEFAULT_SEQ_LEN, dataset_num_proc=0):

    print("\nTokenizing dataset...\n")
    num_proc = resolve_num_proc(dataset_num_proc)

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc,
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
        num_proc=num_proc,
    )

    return lm_dataset

# ============================================================
# Model loading
# ============================================================

def load_causal_lm(
    model_name,
    dtype,
    attention=DEFAULT_ATTENTION,
    device_map=DEFAULT_DEVICE_MAP,
    low_cpu_mem_usage=True,
):

    common = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    common[transformers_dtype_kwarg()] = dtype

    resolved_device_map = resolve_device_map(device_map)
    if resolved_device_map is not None:
        common["device_map"] = resolved_device_map

    if attention == "default":
        print("Using default attention implementation.")
        return AutoModelForCausalLM.from_pretrained(**common)

    if attention != "auto":
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            attn_implementation=attention,
        )
        print(f"{attention} attention enabled.")
        return model

    try:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            attn_implementation="flash_attention_2",
        )
        print("Flash Attention 2 enabled.")
        return model
    except (ImportError, TypeError, ValueError) as e:
        print(f"Flash Attention 2 not available; falling back to SDPA. ({e})")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            attn_implementation="sdpa",
        )
        print("SDPA attention enabled.")
        return model
    except (ImportError, TypeError, ValueError) as e:
        print(f"SDPA not available; falling back to default attention. ({e})")

    return AutoModelForCausalLM.from_pretrained(**common)


# ============================================================
# Freeze lower layers
# ============================================================

def freeze_lower_layers(model, train_last_n_layers=8):

    layers = model.model.layers

    total_layers = len(layers)

    train_last_n_layers = min(max(train_last_n_layers, 0), total_layers)
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
def run_eval(model, tokenizer, prompts, max_new_tokens=128):

    model.eval()

    print("\n================ EVAL ================\n")

    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
# Training arguments
# ============================================================

def estimate_training_steps(
    num_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
):
    if num_examples <= 0:
        return 0

    effective_batch_size = max(
        1,
        per_device_train_batch_size * gradient_accumulation_steps,
    )
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch_size))
    return max(1, math.ceil(steps_per_epoch * num_train_epochs))


def estimate_warmup_steps(
    num_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
):
    if num_examples <= 0 or warmup_ratio <= 0:
        return 0

    total_steps = estimate_training_steps(
        num_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return max(1, int(total_steps * warmup_ratio))


def make_training_arguments(
    output_dir,
    train_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    lr_scheduler_type=DEFAULT_LR_SCHEDULER_TYPE,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    requested_save_steps=DEFAULT_SAVE_STEPS,
    eval_steps=None,
    save_total_limit=2,
    logging_steps=1,
    dtype=torch.bfloat16,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_use_reentrant=False,
    optim=DEFAULT_OPTIM,
    dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    max_grad_norm=1.0,
):

    valid = inspect.signature(TrainingArguments.__init__).parameters

    training_steps = estimate_training_steps(
        train_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    warmup_steps = estimate_warmup_steps(
        train_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
    )
    save_steps = max(1, min(requested_save_steps, training_steps))
    eval_steps = save_steps if eval_steps is None else max(1, min(eval_steps, training_steps))
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16
    dataloader_persistent_workers = (
        dataloader_persistent_workers and dataloader_num_workers > 0
    )

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": gradient_checkpointing_use_reentrant,
        },
        optim=optim,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        max_grad_norm=max_grad_norm,
        remove_unused_columns=False,
    )

    if "overwrite_output_dir" in valid:
        kwargs["overwrite_output_dir"] = False

    if "eval_strategy" in valid:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = "steps"
    else:
        print("Note: this transformers version does not expose an eval strategy argument.")

    if "save_strategy" in valid:
        kwargs["save_strategy"] = "steps"

    unsupported = [
        key
        for key in sorted(kwargs)
        if key not in valid
    ]
    for key in unsupported:
        print(f"Note: this transformers version does not support {key!r}; skipping.")
        del kwargs[key]

    return TrainingArguments(**kwargs)

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
        default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint and final model output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )

    parser.add_argument(
        "--eval_prompts",
        required=True,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
    )

    parser.add_argument(
        "--train_last_n_layers",
        type=int,
        default=DEFAULT_TRAIN_LAST_N_LAYERS,
    )

    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=DEFAULT_DTYPE,
        help="Model and training dtype. Use fp16 if bf16 is unsupported.",
    )

    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "default"),
        default=DEFAULT_ATTENTION,
        help="Attention backend. Smaller systems often work well with sdpa.",
    )

    parser.add_argument(
        "--device_map",
        choices=("auto", "single", "none"),
        default=DEFAULT_DEVICE_MAP,
        help="'auto' lets Transformers place layers, 'single' forces GPU 0.",
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--float32_matmul_precision",
        choices=("highest", "high", "medium"),
        default=DEFAULT_FLOAT32_MATMUL_PRECISION,
    )

    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        default=DEFAULT_LR_SCHEDULER_TYPE,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="Checkpoint interval in optimizer steps. Small runs are capped to save at least one checkpoint.",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation interval in optimizer steps. Defaults to save_steps.",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--optim",
        default=DEFAULT_OPTIM,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
    )

    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--dataloader_persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=0,
        help="Tokenization/packing process count. 0 means half of available CPUs.",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=128,
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    eval_prompts_path = Path(args.eval_prompts).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_torch_dtype(args.dtype)

    configure_torch_backend(
        tf32=args.tf32,
        float32_matmul_precision=args.float32_matmul_precision,
    )

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

    print(f"Requested dtype: {args.dtype}")
    print(f"Resolved dtype: {dtype}")
    print(f"Attention: {args.attention}")
    print(f"Device map: {args.device_map}")

    model = load_causal_lm(
        args.model_name,
        dtype=dtype,
        attention=args.attention,
        device_map=args.device_map,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if args.gradient_checkpointing:
        # Required with gradient checkpointing
        model.config.use_cache = False

    if args.compile_model:
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

    dataset = load_text_files(dataset_dir)

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
        dataset_num_proc=args.dataset_num_proc,
    )

    eval_dataset = tokenize_dataset(
        dataset["test"],
        tokenizer,
        seq_len=args.seq_len,
        dataset_num_proc=args.dataset_num_proc,
    )

    # ========================================================
    # Eval prompts
    # ========================================================

    prompts = eval_prompts_path.read_text().splitlines()

    prompts = [
        p.strip()
        for p in prompts
        if p.strip()
    ]

    # ========================================================
    # Baseline evaluation
    # ========================================================

    print("\nRUNNING BASELINE EVALUATION\n")

    run_eval(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.eval_max_new_tokens,
    )

    # ========================================================
    # Training arguments
    # ========================================================

    training_args = make_training_arguments(
        output_dir,
        train_examples=len(train_dataset),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        requested_save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        dtype=dtype,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_use_reentrant=args.gradient_checkpointing_use_reentrant,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        max_grad_norm=args.max_grad_norm,
    )

    print(f"\nCheckpoint/output directory: {output_dir}\n")
    print(f"Sequence length: {args.seq_len}")
    print(f"Train batch size per device: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Trainable upper layers: {args.train_last_n_layers}")

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

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No valid checkpoint found. Starting fresh.")
        trainer.train()

    # ========================================================
    # Final evaluation
    # ========================================================

    print("\nRUNNING FINAL EVALUATION\n")

    run_eval(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.eval_max_new_tokens,
    )

    # ========================================================
    # Save model
    # ========================================================

    print("\nSaving model...\n")

    trainer.save_model(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))

    print("\nDONE\n")

if __name__ == "__main__":
    main()
