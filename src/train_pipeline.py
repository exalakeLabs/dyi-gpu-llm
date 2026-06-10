#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from runtime_env import is_databricks


ROOT = Path(__file__).resolve().parent
TRAIN_GPU = ROOT / "train_lora_gpu.py"
TRAIN_CPU = ROOT / "train_lora_cpu.py"


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
    if is_databricks():
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

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


def run_databricks_trainer(trainer: Path) -> int:
    old_argv = sys.argv[:]
    sys.argv = [trainer.name]
    try:
        if trainer == TRAIN_GPU:
            import train_lora_gpu

            return int(train_lora_gpu.main())
        import train_lora_cpu

        return int(train_lora_cpu.main())
    finally:
        sys.argv = old_argv


def main() -> int:
    py = python_executable()

    #ensure_exists(MAKE_PAIRS)
    ensure_exists(TRAIN_GPU)
    ensure_exists(TRAIN_CPU)

    print("Project root:", ROOT)
    print("Python:", py)

    #rc = run_step([py, str(MAKE_PAIRS)])
    #if rc != 0:
    #    print(f"[ERROR] make_training_pairs.py failed with exit code {rc}")
    #    return rc

    if cuda_available(py):
        print("[INFO] CUDA detected. Using GPU trainer.")
        trainer = TRAIN_GPU
    else:
        print("[INFO] CUDA not detected. Using CPU trainer.")
        trainer = TRAIN_CPU

    if is_databricks():
        print("[INFO] Databricks detected. Running trainer in the notebook process.")
        rc = run_databricks_trainer(trainer)
    else:
        rc = run_step([py, str(trainer)])
    if rc != 0:
        print(f"[ERROR] {trainer.name} failed with exit code {rc}")
        return rc

    print()
    print("[OK] Training pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
