import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    value = os.environ.get(var, "").strip()
    path = Path(value).expanduser() if value else REPO_ROOT / default_rel
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


DATA_DIR = env_dir("LLAMA_DATA_DIR", "data")
OUTPUT_DIR = env_dir("LLAMA_OUTPUT_DIR", "output")
PREPARED_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")

TRAIN_FILE = DATA_DIR / "train.jsonl"
LORA_DIR = OUTPUT_DIR / "lora"
ADAPTER_DIR = LORA_DIR / "final"
