#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parent
MAKE_PAIRS = ROOT / "make_training_pairs.py"
TRAIN_GPU = ROOT / "train_lora_gpu.py"
TRAIN_CPU = ROOT / "train_lora_cpu.py"
CLEAN_TEXT = ROOT / "clean_text.py"


def run_step(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print()
    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    print()

    result = subprocess.run(cmd, env=env)
    return result.returncode


def python_executable() -> str:
    return sys.executable or "python3"


def cuda_available(py: str) -> bool:
    probe = """
import torch
print("1" if torch.cuda.is_available() else "0")
"""
    try:
        result = subprocess.run(
            [py, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() == "1"
    except Exception:
        return False


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> int:
    py = python_executable()

    ensure_exists(MAKE_PAIRS)
    ensure_exists(TRAIN_GPU)
    ensure_exists(TRAIN_CPU)
    ensure_exists(CLEAN_TEXT)

    env = os.environ.copy()

    print("Project root:", ROOT)
    print("Python:", py)

    rc = run_step([py, str(MAKE_PAIRS)], env=env)
    if rc != 0:
        print(f"[ERROR] make_training_pairs.py failed with exit code {rc}")
        return rc

    rc = run_step([py, str(CLEAN_TEXT)], env=env)
    if rc != 0:
        print(f"[ERROR] clean_text.py failed with exit code {rc}")
        return rc

    if cuda_available(py):
        print("[INFO] CUDA detected. Using GPU trainer.")
        trainer = TRAIN_GPU
    else:
        print("[INFO] CUDA not detected. Using CPU trainer.")
        trainer = TRAIN_CPU

    rc = run_step([py, str(trainer)], env=env)
    if rc != 0:
        print(f"[ERROR] {trainer.name} failed with exit code {rc}")
        return rc

    print()
    print("[OK] Training pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
