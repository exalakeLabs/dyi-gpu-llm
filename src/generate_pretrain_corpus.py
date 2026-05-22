#!/usr/bin/env python3

"""
generate_pretrain_corpus.py

Build a packed token corpus for continued pretraining.

Reads raw .txt files from prepared/ by default, tokenizes them with the target
model tokenizer, packs the token stream into fixed-length examples, and writes:

- corpus/train.jsonl
- corpus/eval.jsonl

Each JSONL row contains an input/label pair:

{"input_ids": [...], "labels": [...]}

Example:

python src/generate_pretrain_corpus.py \
    --text_dir ./prepared \
    --corpus_dir ./corpus \
    --model_name Qwen/Qwen2.5-3B \
    --seq_len 2048 \
    --num_proc 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_TEXT_DIR = "./prepared"
DEFAULT_CORPUS_DIR = "./corpus"
DEFAULT_SEQ_LEN = 2048
DEFAULT_DATASET_NUM_PROC = 1
DEFAULT_TOKENIZE_BATCH_SIZE = 128
DEFAULT_EVAL_RATIO = 0.01
DEFAULT_SEED = 42
DEFAULT_MIN_CHARS = 1000


def resolve_num_proc(dataset_num_proc: int) -> int:
    if dataset_num_proc > 0:
        return dataset_num_proc
    return max((os.cpu_count() or 2) // 2, 1)


def load_text_files(text_dir: Path, glob_pattern: str, min_chars: int) -> Dataset:
    texts = []
    paths = sorted(text_dir.rglob(glob_pattern))

    print(f"\nFound {len(paths)} text files under {text_dir}\n")

    for path in paths:
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if len(content) < min_chars:
                continue

            texts.append({"text": content})
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not texts:
        raise SystemExit(f"No usable text files found under {text_dir}")

    return Dataset.from_list(texts)


def tokenize_and_pack_dataset(
    dataset: Dataset,
    tokenizer,
    seq_len: int,
    dataset_num_proc: int,
    tokenize_batch_size: int,
) -> Dataset:
    print("\nTokenizing dataset...\n")
    num_proc = resolve_num_proc(dataset_num_proc)
    print(f"Using num_proc={num_proc}, tokenize_batch_size={tokenize_batch_size}\n")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=["text"],
        num_proc=num_proc,
    )

    print("\nPacking sequences...\n")

    def group_texts(examples):
        input_ids = sum(examples["input_ids"], [])
        total_length = len(input_ids)
        total_length = (total_length // seq_len) * seq_len

        packed_input_ids = [
            input_ids[i : i + seq_len]
            for i in range(0, total_length, seq_len)
        ]

        return {
            "input_ids": packed_input_ids,
            "labels": [tokens.copy() for tokens in packed_input_ids],
        }

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
    )


def split_documents(dataset: Dataset, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return dataset, None

    if len(dataset) < 2:
        print(
            "Only one source document available; validation will be split from packed tokens."
        )
        return dataset, None

    split = dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
    )
    return split["train"], split["test"]


def split_packed_tokens(dataset: Dataset, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return dataset, None

    if len(dataset) < 2:
        raise SystemExit(
            "Need at least two packed sequences to create train/eval JSONL files. "
            "Add more text or reduce --seq_len."
        )

    split = dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
    )
    return split["train"], split["test"]


def split_empty_eval_from_train(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    eval_ratio: float,
    seed: int,
):
    if len(eval_dataset) > 0:
        return train_dataset, eval_dataset

    print(
        "Eval document split produced no packed tokens; validation will be split from train tokens."
    )
    return split_packed_tokens(
        train_dataset,
        eval_ratio=eval_ratio,
        seed=seed,
    )


def write_jsonl(path: Path, dataset: Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate packed token JSONL files for continued pretraining."
    )

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--text_dir",
        default=DEFAULT_TEXT_DIR,
        help=f"Directory containing prepared .txt files. Defaults to {DEFAULT_TEXT_DIR}.",
    )

    parser.add_argument(
        "--corpus_dir",
        default=DEFAULT_CORPUS_DIR,
        help=f"Output directory for token JSONL files. Defaults to {DEFAULT_CORPUS_DIR}.",
    )

    parser.add_argument(
        "--train_file",
        default="train.jsonl",
        help="Train JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--eval_file",
        default="eval.jsonl",
        help="Eval JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--glob",
        default="*.txt",
        help='Glob pattern under --text_dir. Defaults to "*.txt".',
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
    )

    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help="Validation split ratio. Defaults to 0.01.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip text files shorter than this many characters.",
    )

    parser.add_argument(
        "--dataset_num_proc",
        "--num_proc",
        dest="dataset_num_proc",
        type=int,
        default=DEFAULT_DATASET_NUM_PROC,
        help="Tokenization/packing process count. 0 means half of available CPUs.",
    )

    parser.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=DEFAULT_TOKENIZE_BATCH_SIZE,
        help="Documents per tokenizer batch. Lower this if tokenization is killed for memory.",
    )

    return parser.parse_args()


def resolve_output_file(corpus_dir: Path, file_arg: str) -> Path:
    path = Path(file_arg).expanduser()
    if not path.is_absolute():
        path = corpus_dir / path
    return path


def main() -> int:
    args = parse_args()

    text_dir = Path(args.text_dir).expanduser()
    corpus_dir = Path(args.corpus_dir).expanduser()
    train_file = resolve_output_file(corpus_dir, args.train_file)
    eval_file = resolve_output_file(corpus_dir, args.eval_file)

    if args.seq_len <= 0:
        raise SystemExit("--seq_len must be greater than 0")
    if args.eval_ratio <= 0 or args.eval_ratio >= 1:
        raise SystemExit("--eval_ratio must be > 0 and < 1")
    if args.tokenize_batch_size <= 0:
        raise SystemExit("--tokenize_batch_size must be greater than 0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_text_files(
        text_dir=text_dir,
        glob_pattern=args.glob,
        min_chars=args.min_chars,
    )

    train_docs, eval_docs = split_documents(
        raw_dataset,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    train_dataset = tokenize_and_pack_dataset(
        train_docs,
        tokenizer,
        seq_len=args.seq_len,
        dataset_num_proc=args.dataset_num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    if eval_docs is None:
        train_dataset, eval_dataset = split_packed_tokens(
            train_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )
    else:
        eval_dataset = tokenize_and_pack_dataset(
            eval_docs,
            tokenizer,
            seq_len=args.seq_len,
            dataset_num_proc=args.dataset_num_proc,
            tokenize_batch_size=args.tokenize_batch_size,
        )
        train_dataset, eval_dataset = split_empty_eval_from_train(
            train_dataset,
            eval_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )

    if eval_dataset is None:
        raise SystemExit("An eval split is required for continued_pretrain_partial.py.")

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise SystemExit(
            "Token corpus is empty after packing. Add more text or reduce --seq_len."
        )

    write_jsonl(train_file, train_dataset)
    write_jsonl(eval_file, eval_dataset)

    print()
    print(f"Train examples: {len(train_dataset)} -> {train_file}")
    print(f"Eval examples:  {len(eval_dataset)} -> {eval_file}")
    print("\nDONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
