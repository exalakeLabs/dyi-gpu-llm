#!/usr/bin/env python3

"""
CreateCorpusToken

Tokenize cleaned text files with the target model tokenizer, concatenate the
token stream with EOS separators between documents, pack it into fixed-length
training blocks, and save the result as a Hugging Face Dataset directory.

This is an independent preprocessing step. It does not replace
generate_pretrain_corpus.py, which writes train/eval JSONL files for the
existing continued-pretraining flow.

Example:

python src/create_corpus_token.py \
    --text_dir ./prepared \
    --output_dir ./corpus/tokenized-qwen-4096 \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --block_size 4096 \
    --num_proc 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from datasets import Dataset
from transformers import AutoTokenizer

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_int, env_str

DEFAULT_CORPUS_TOKEN_DIR = env_str("DEFAULT_CORPUS_TOKEN_DIR", "corpus/tokenized")
DEFAULT_DATASET_NUM_PROC = env_int("DEFAULT_DATASET_NUM_PROC", 1)
DEFAULT_MIN_CHARS = env_int("DEFAULT_MIN_CHARS", 1000)
DEFAULT_MODEL = env_str("DEFAULT_MODEL")
DEFAULT_SEQ_LEN = env_int("DEFAULT_SEQ_LEN", 2048)
DEFAULT_TEXT_DIR = env_str("DEFAULT_TEXT_DIR", "prepared")
DEFAULT_TOKENIZE_BATCH_SIZE = env_int("DEFAULT_TOKENIZE_BATCH_SIZE", 128)
DEFAULT_PACK_BATCH_SIZE = env_int("DEFAULT_PACK_BATCH_SIZE", 1000)


def resolve_num_proc(dataset_num_proc: int) -> int:
    if dataset_num_proc > 0:
        return dataset_num_proc
    return max((os.cpu_count() or 2) // 2, 1)


def iter_text_records(
    paths: Iterable[Path],
    text_dir: Path,
    min_chars: int,
) -> Iterable[dict]:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            print(f"Skipping {path}: {exc}")
            continue

        if len(text) < min_chars:
            continue

        try:
            source = str(path.relative_to(text_dir))
        except ValueError:
            source = str(path)

        yield {
            "text": text,
            "source": source,
        }


def load_text_dataset(text_dir: Path, glob_pattern: str, min_chars: int) -> Dataset:
    paths = sorted(path for path in text_dir.rglob(glob_pattern) if path.is_file())
    print(f"\nFound {len(paths)} text files under {text_dir}\n")

    dataset = Dataset.from_generator(
        lambda: iter_text_records(
            paths=paths,
            text_dir=text_dir,
            min_chars=min_chars,
        )
    )

    if len(dataset) == 0:
        raise SystemExit(f"No usable text files found under {text_dir}")

    print(f"Usable documents: {len(dataset)}")
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    dataset_num_proc: int,
    tokenize_batch_size: int,
) -> Dataset:
    num_proc = resolve_num_proc(dataset_num_proc)
    print(f"\nTokenizing with num_proc={num_proc}, batch_size={tokenize_batch_size}\n")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing text",
    )


def pack_token_dataset(
    tokenized: Dataset,
    eos_token_id: int,
    block_size: int,
    dataset_num_proc: int,
    pack_batch_size: int,
    add_eos: bool,
) -> Dataset:
    num_proc = resolve_num_proc(dataset_num_proc)
    print(f"\nPacking fixed-length blocks of {block_size} tokens\n")

    def pack_sequences(batch):
        concatenated = []

        for document_tokens in batch["input_ids"]:
            concatenated.extend(document_tokens)
            if add_eos:
                concatenated.append(eos_token_id)

        usable_length = (len(concatenated) // block_size) * block_size
        blocks = [
            concatenated[i : i + block_size]
            for i in range(0, usable_length, block_size)
        ]

        return {
            "input_ids": blocks,
            "attention_mask": [[1] * block_size for _ in blocks],
            "labels": [block.copy() for block in blocks],
        }

    return tokenized.map(
        pack_sequences,
        batched=True,
        batch_size=pack_batch_size,
        remove_columns=tokenized.column_names,
        num_proc=num_proc,
        desc="Packing token blocks",
    )


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer_name_or_path: str,
    document_count: int,
    block_count: int,
) -> None:
    manifest = {
        "tool": "CreateCorpusToken",
        "model_name": args.model_name,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "text_dir": str(Path(args.text_dir).expanduser()),
        "glob": args.glob,
        "min_chars": args.min_chars,
        "block_size": args.block_size,
        "add_eos": args.add_eos,
        "documents": document_count,
        "blocks": block_count,
        "dataset_format": "huggingface_save_to_disk",
        "columns": ["input_ids", "attention_mask", "labels"],
    }
    (output_dir / "create_corpus_token_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CreateCorpusToken",
        description=(
            "Tokenize .txt files, pack them into fixed-length causal-LM blocks, "
            "and save a Hugging Face Dataset to disk."
        ),
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--text_dir",
        default=DEFAULT_TEXT_DIR,
        help=f"Directory containing cleaned .txt files. Default: {DEFAULT_TEXT_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_CORPUS_TOKEN_DIR,
        help=f"Output Hugging Face Dataset directory. Default: {DEFAULT_CORPUS_TOKEN_DIR}",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help='Glob pattern under --text_dir. Default: "*.txt"',
    )
    parser.add_argument(
        "--block_size",
        "--seq_len",
        dest="block_size",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Fixed token block size. Default: {DEFAULT_SEQ_LEN}",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Skip text files shorter than this many characters. Default: {DEFAULT_MIN_CHARS}",
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
        help=f"Documents per tokenizer batch. Default: {DEFAULT_TOKENIZE_BATCH_SIZE}",
    )
    parser.add_argument(
        "--pack_batch_size",
        type=int,
        default=DEFAULT_PACK_BATCH_SIZE,
        help=f"Documents per packing batch. Default: {DEFAULT_PACK_BATCH_SIZE}",
    )
    parser.add_argument(
        "--add_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append tokenizer.eos_token_id between documents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output_dir first if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    text_dir = Path(args.text_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not args.model_name:
        raise SystemExit("--model_name is required")
    if args.block_size <= 0:
        raise SystemExit("--block_size must be greater than 0")
    if args.min_chars < 0:
        raise SystemExit("--min_chars must be 0 or greater")
    if args.tokenize_batch_size <= 0:
        raise SystemExit("--tokenize_batch_size must be greater than 0")
    if args.pack_batch_size <= 0:
        raise SystemExit("--pack_batch_size must be greater than 0")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None and args.add_eos:
        raise SystemExit("Tokenizer has no eos_token_id; rerun with --no-add_eos.")

    raw_dataset = load_text_dataset(
        text_dir=text_dir,
        glob_pattern=args.glob,
        min_chars=args.min_chars,
    )
    tokenized = tokenize_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        dataset_num_proc=args.dataset_num_proc,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    packed = pack_token_dataset(
        tokenized,
        eos_token_id=tokenizer.eos_token_id,
        block_size=args.block_size,
        dataset_num_proc=args.dataset_num_proc,
        pack_batch_size=args.pack_batch_size,
        add_eos=args.add_eos,
    )

    if len(packed) == 0:
        raise SystemExit(
            "Packed dataset is empty. Add more text, lower --min_chars, or lower --block_size."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    packed.save_to_disk(str(output_dir))
    write_manifest(
        output_dir=output_dir,
        args=args,
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", args.model_name),
        document_count=len(raw_dataset),
        block_count=len(packed),
    )

    print()
    print(f"Documents: {len(raw_dataset)}")
    print(f"Packed blocks: {len(packed)}")
    print(f"Block size: {args.block_size}")
    print(f"Saved token dataset: {output_dir}")
    print("\nDONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
