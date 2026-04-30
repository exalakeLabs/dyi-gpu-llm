#!/usr/bin/env python
from make_text_dataset import build_dataset
from project_config import TRAIN_FILE
from train_lora import main as train_lora_main


def main() -> int:
    rows = build_dataset()
    if rows == 0:
        raise RuntimeError(f"No training rows were written to {TRAIN_FILE}")
    return train_lora_main()


if __name__ == "__main__":
    raise SystemExit(main())
