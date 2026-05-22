from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

_ASSIGNMENT_RE = re.compile(
    r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)
_ENV_REF_RE = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)")


def _bracket_delta(value: str) -> int:
    return value.count("(") + value.count("[") - value.count(")") - value.count("]")


def _expand_env_refs(value: str, env_values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        return str(os.environ.get(name, env_values.get(name, "")))

    return _ENV_REF_RE.sub(replace, value)


def _parse_env_value(raw_value: str, env_values: dict[str, Any]) -> Any:
    value = _expand_env_refs(raw_value.strip(), env_values)
    if not value:
        return ""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

    if isinstance(parsed, tuple) and all(isinstance(item, str) for item in parsed):
        return "".join(parsed)
    return parsed


def load_env_file(path: Path = ENV_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}

    values: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        i += 1
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue

        name = match.group("name")
        value = match.group("value").strip()
        depth = _bracket_delta(value)

        while depth > 0 and i < len(lines):
            next_line = lines[i]
            i += 1
            value = f"{value}\n{next_line}"
            depth += _bracket_delta(next_line)

        parsed = _parse_env_value(value, values)
        values[name] = parsed
        if not isinstance(parsed, (list, tuple, dict)):
            os.environ.setdefault(name, str(parsed))

    return values


ENV = load_env_file()


def _env_value(name: str, default: Any) -> Any:
    return os.environ.get(name, ENV.get(name, default))


def env_str(name: str, default: str = "") -> str:
    value = _env_value(name, default)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def env_int(name: str, default: int) -> int:
    return int(_env_value(name, default))


def env_float(name: str, default: float) -> float:
    return float(_env_value(name, default))


def env_list(name: str, default: list[str]) -> list[str]:
    value = _env_value(name, default)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [part.strip() for part in str(value).split("|") if part.strip()]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def env_path(name: str, default: str | Path) -> Path:
    return repo_path(_env_value(name, default))


DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "You are a careful Tolkien lore assistant. Use the retrieved canon context "
    "carefully, distinguish certainty from inference, and avoid inventing details."
)

DEFAULT_START_MARKERS = [
    "START OF THE PROJECT GUTENBERG EBOOK",
    "START OF THIS PROJECT GUTENBERG EBOOK",
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
]
DEFAULT_END_MARKERS = [
    "END OF THE PROJECT GUTENBERG EBOOK",
    "END OF THIS PROJECT GUTENBERG EBOOK",
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
]

MODEL_ROOT = env_path("MODEL_ROOT", REPO_ROOT)
RAWTEXT_DIR = env_path("RAWTEXT_DIR", MODEL_ROOT / "text")
TEXT_DIR = RAWTEXT_DIR
PREPARED_DIR = env_path("PREPARED_DIR", MODEL_ROOT / "prepared")
RAG_DIR = env_path("RAG_DIR", MODEL_ROOT / "rag")
MODEL_DIR = env_path("MODEL_DIR", MODEL_ROOT / "model")
PDF_DIR = env_path("PDF_DIR", MODEL_ROOT / "pdfs")
CORPUS_DIR = env_path("CORPUS_DIR", MODEL_ROOT / "corpus")
OUTPUT_DIR = env_path("OUTPUT_DIR", MODEL_DIR)

GENERATOR_MODEL = env_str("GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BASE_MODEL = env_str("BASE_MODEL", GENERATOR_MODEL)
EMBED_MODEL = env_str("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = env_str("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

CHUNK_SIZE_CHARS = env_int("CHUNK_SIZE_CHARS", 1800)
OVERLAP_CHARS = env_int("OVERLAP_CHARS", 250)
BATCH_SIZE = env_int("BATCH_SIZE", 32)
RETRIEVE_K = env_int("RETRIEVE_K", 24)
RERANK_TOP_N = env_int("RERANK_TOP_N", 6)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)

DEFAULT_SYSTEM_PROMPT = env_str("DEFAULT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT_FALLBACK)
SYSTEM_PROMPT = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
START_MARKERS = env_list("START_MARKERS", DEFAULT_START_MARKERS)
END_MARKERS = env_list("END_MARKERS", DEFAULT_END_MARKERS)

TRAIN_FILE = env_path("TRAIN_FILE", CORPUS_DIR / "train.jsonl")
LORA_DIR = env_path("LORA_DIR", OUTPUT_DIR / "lora")
ADAPTER_DIR = env_path("ADAPTER_DIR", LORA_DIR / "final")

DEFAULT_MODEL = env_str("DEFAULT_MODEL", BASE_MODEL)
DEFAULT_MODEL_PATH = str(env_path("DEFAULT_MODEL_PATH", MODEL_DIR))
DEFAULT_BACKEND = env_str("DEFAULT_BACKEND", "cuda:0")
DEFAULT_TEXT_DIR = str(env_path("DEFAULT_TEXT_DIR", PREPARED_DIR))
DEFAULT_CORPUS_DIR = str(env_path("DEFAULT_CORPUS_DIR", CORPUS_DIR))
DEFAULT_TRAIN_FILE = env_str("DEFAULT_TRAIN_FILE", "train.jsonl")
DEFAULT_EVAL_FILE = env_str("DEFAULT_EVAL_FILE", "eval.jsonl")
DEFAULT_OUTPUT_DIR = str(env_path("DEFAULT_OUTPUT_DIR", MODEL_DIR / "output_partial"))

DEFAULT_MAX_LENGTH = env_int("DEFAULT_MAX_LENGTH", 512)
DEFAULT_SEQ_LEN = env_int("DEFAULT_SEQ_LEN", 2048)
DEFAULT_DATASET_NUM_PROC = env_int("DEFAULT_DATASET_NUM_PROC", 1)
DEFAULT_TOKENIZE_BATCH_SIZE = env_int("DEFAULT_TOKENIZE_BATCH_SIZE", 128)
DEFAULT_EVAL_RATIO = env_float("DEFAULT_EVAL_RATIO", 0.01)
DEFAULT_SEED = env_int("DEFAULT_SEED", 42)
DEFAULT_MIN_CHARS = env_int("DEFAULT_MIN_CHARS", 1000)

DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = env_int("DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE", 1)
DEFAULT_GRADIENT_ACCUMULATION_STEPS = env_int("DEFAULT_GRADIENT_ACCUMULATION_STEPS", 8)
DEFAULT_NUM_TRAIN_EPOCHS = env_float("DEFAULT_NUM_TRAIN_EPOCHS", 1.0)
DEFAULT_LEARNING_RATE = env_float("DEFAULT_LEARNING_RATE", 2e-4)
DEFAULT_LORA_RANK = env_int("DEFAULT_LORA_RANK", 16)
DEFAULT_LOGGING_STEPS = env_int("DEFAULT_LOGGING_STEPS", 1)
DEFAULT_SAVE_STEPS = env_int("DEFAULT_SAVE_STEPS", 100)
DEFAULT_SAVE_TOTAL_LIMIT = env_int("DEFAULT_SAVE_TOTAL_LIMIT", 3)
DEFAULT_WARMUP_RATIO = env_float("DEFAULT_WARMUP_RATIO", 0.03)
DEFAULT_DATALOADER_NUM_WORKERS = env_int("DEFAULT_DATALOADER_NUM_WORKERS", 4)

DEFAULT_WEIGHT_DECAY = env_float("DEFAULT_WEIGHT_DECAY", 0.01)
DEFAULT_LR_SCHEDULER_TYPE = env_str("DEFAULT_LR_SCHEDULER_TYPE", "cosine")
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = env_int("DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE", 1)
DEFAULT_TRAIN_LAST_N_LAYERS = env_int("DEFAULT_TRAIN_LAST_N_LAYERS", 8)
DEFAULT_DTYPE = env_str("DEFAULT_DTYPE", "auto")
DEFAULT_ATTENTION = env_str("DEFAULT_ATTENTION", "auto")
DEFAULT_DEVICE_MAP = env_str("DEFAULT_DEVICE_MAP", "auto")
DEFAULT_OPTIM = env_str("DEFAULT_OPTIM", "adamw_torch_fused")
DEFAULT_FLOAT32_MATMUL_PRECISION = env_str("DEFAULT_FLOAT32_MATMUL_PRECISION", "high")
