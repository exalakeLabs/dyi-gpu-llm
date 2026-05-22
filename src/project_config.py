from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


DATA_DIR = repo_path("data")
OUTPUT_DIR = repo_path("output")
PREPARED_DIR = repo_path("prepared")

TRAIN_FILE = DATA_DIR / "train.jsonl"
LORA_DIR = OUTPUT_DIR / "lora"
ADAPTER_DIR = LORA_DIR / "final"
