#!/usr/bin/env python
from make_training_pairs import build_dataset
from project_config import TRAIN_FILE
from train_lora_cpu import main as train_lora_cpu_main
from train_lora_gpu import main as train_lora_gpu_main


def _prompt_trainer() -> int:
    print(
        "How do you want to train LoRA?\n"
        "  [g] GPU — train_lora_gpu.py (uses GPU when available)\n"
        "  [c] CPU — train_lora_cpu.py"
    )
    while True:
        choice = (input("Enter g or c (default: g): ").strip().lower() or "g")
        if choice in ("g", "gpu"):
            return train_lora_gpu_main()
        if choice in ("c", "cpu"):
            return train_lora_cpu_main()
        print("Invalid choice; enter 'g' for GPU or 'c' for CPU.")


def main() -> int:
    rows = build_dataset()
    if rows == 0:
        raise RuntimeError(f"No training rows were written to {TRAIN_FILE}")
    return _prompt_trainer()


if __name__ == "__main__":
    raise SystemExit(main())
