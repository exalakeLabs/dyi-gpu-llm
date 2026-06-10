from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
ENV_DEFAULT_PATH = REPO_ROOT / ".env.default"
_DATABRICKS_ENV_DEFAULTS = {
    "MODEL_ROOT": "/dbfs/FileStore/llama32-local",
    "RAWTEXT_DIR": "/dbfs/FileStore/llama32-local/raw-text",
    "RAW_TEXT_DIR": "/dbfs/FileStore/llama32-local/raw-text",
    "TEXT_DIR": "/dbfs/FileStore/llama32-local/raw-text",
    "PDF_DIR": "/dbfs/FileStore/llama32-local/pdfs",
    "PREPARED_DIR": "/dbfs/FileStore/llama32-local/prepared",
    "RAG_DIR": "/dbfs/FileStore/llama32-local/rag",
    "MODEL_DIR": "/dbfs/FileStore/llama32-local/model",
    "CORPUS_DIR": "/dbfs/FileStore/llama32-local/corpus",
    "OUTPUT_DIR": "/dbfs/FileStore/llama32-local/model",
    "TRAIN_FILE": "/dbfs/FileStore/llama32-local/corpus/train.jsonl",
    "LORA_DIR": "/dbfs/FileStore/llama32-local/model/lora",
    "ADAPTER_DIR": "/dbfs/FileStore/llama32-local/model/lora/final",
    "DEFAULT_MODEL_PATH": "/dbfs/FileStore/llama32-local/model",
    "DEFAULT_TEXT_DIR": "/dbfs/FileStore/llama32-local/prepared",
    "DEFAULT_CORPUS_DIR": "/dbfs/FileStore/llama32-local/corpus",
    "DEFAULT_OUTPUT_DIR": "/dbfs/FileStore/llama32-local/model/output_partial",
    "DEFAULT_BACKEND": "cuda:0",
    "DEFAULT_DATALOADER_NUM_WORKERS": "2",
}

_ENV_REF_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


def _expand_env_refs(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        return os.environ.get(name, "")

    return _ENV_REF_RE.sub(replace, value)


def is_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def _normalize_databricks_path(value: str) -> str:
    if value.startswith("dbfs:/"):
        return "/dbfs/" + value.removeprefix("dbfs:/").lstrip("/")
    return value


def load_dotenv(path: Path = ENV_PATH, *, override: bool = False) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue

        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]

        for part in parts:
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            if not name:
                continue
            expanded = _normalize_databricks_path(_expand_env_refs(value))
            if override:
                os.environ[name] = expanded
            else:
                os.environ.setdefault(name, expanded)


def load_environment() -> None:
    if is_databricks():
        for name, value in _DATABRICKS_ENV_DEFAULTS.items():
            os.environ.setdefault(name, value)
    load_dotenv(ENV_DEFAULT_PATH)
    load_dotenv(ENV_PATH, override=True)


load_environment()


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    return [part.strip() for part in value.split("|") if part.strip()]


def repo_path(path: str | Path) -> Path:
    path = Path(_normalize_databricks_path(str(path))).expanduser()
    if not path.is_absolute():
        base = Path(env_str("MODEL_ROOT", "")) if is_databricks() else REPO_ROOT
        if not str(base):
            base = REPO_ROOT
        path = base / path
    return path


def env_path(name: str, default: str | Path) -> Path:
    return repo_path(env_str(name, str(default)))
