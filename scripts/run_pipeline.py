#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{suffix}] ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    print(prompt)
    for key, label in choices.items():
        marker = " (default)" if key == default else ""
        print(f"  {key}. {label}{marker}")
    while True:
        value = input("> ").strip().lower() or default
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def run(cmd: list[str], *, dry_run: bool) -> int:
    print()
    print("==>", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def python_cmd() -> str:
    return os.environ.get("PYTHON") or sys.executable or "python3"


def chat_command(mode: str) -> list[str]:
    py = python_cmd()
    cmd = [py, str(ROOT / "inference" / "chat_rag.py")]
    if mode in {"base", "adapter"}:
        cmd.append("--no-rag")
    if mode in {"base", "rag"}:
        cmd.append("--no-adapter")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided local data/RAG/training/inference pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected commands without running them.")
    args = parser.parse_args()

    print("Guided pipeline")
    print(f"Project root: {ROOT}")

    steps: list[list[str]] = []
    py = python_cmd()

    if ask_yes_no("Prepare/download raw data?", default=False):
        jobs = input("Parallel download jobs [4]: ").strip() or "4"
        steps.append([str(ROOT / "get_raw_text.zsh"), "--jobs", jobs])

    if ask_yes_no("Build a RAG index?", default=False):
        steps.append([str(ROOT / "build_rag_index.zsh")])

    if ask_yes_no("Run LoRA training?", default=False):
        steps.append([py, str(ROOT / "training" / "train_pipeline.py")])

    if ask_yes_no("Run chat/inference afterward?", default=True):
        mode = ask_choice(
            "Choose inference mode:",
            {
                "base": "base model only",
                "adapter": "base model + LoRA adapter",
                "rag": "base model + RAG",
                "both": "base model + LoRA adapter + RAG",
            },
            default="both",
        )
        steps.append(chat_command(mode))

    if not steps:
        print("No steps selected.")
        return 0

    for cmd in steps:
        rc = run(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"Step failed with exit code {rc}: {' '.join(cmd)}", file=sys.stderr)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
