from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_DATABRICKS_MODEL_ROOT = "/dbfs/FileStore/llama32-local"


def is_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def dbutils_or_none():
    try:
        import IPython  # type: ignore

        shell = IPython.get_ipython()
        if shell is not None:
            return shell.user_ns.get("dbutils")
    except Exception:
        return None
    return globals().get("dbutils")


def to_local_path(path: str | Path) -> Path:
    value = str(path).strip()
    if value.startswith("dbfs:/"):
        value = "/dbfs/" + value.removeprefix("dbfs:/").lstrip("/")
    return Path(value).expanduser()


def to_dbfs_uri(path: str | Path) -> str:
    value = str(path)
    if value.startswith("dbfs:/"):
        return value
    if value.startswith("/dbfs/"):
        return "dbfs:/" + value.removeprefix("/dbfs/").lstrip("/")
    return value


def ensure_src_on_path() -> Path:
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def set_default_env(model_root: str = DEFAULT_DATABRICKS_MODEL_ROOT) -> None:
    root = str(to_local_path(model_root))
    defaults = {
        "MODEL_ROOT": root,
        "RAWTEXT_DIR": f"{root}/raw-text",
        "RAW_TEXT_DIR": f"{root}/raw-text",
        "TEXT_DIR": f"{root}/raw-text",
        "PDF_DIR": f"{root}/pdfs",
        "PREPARED_DIR": f"{root}/prepared",
        "RAG_DIR": f"{root}/rag",
        "MODEL_DIR": f"{root}/model",
        "CORPUS_DIR": f"{root}/corpus",
        "OUTPUT_DIR": f"{root}/model",
        "TRAIN_FILE": f"{root}/corpus/train.jsonl",
        "LORA_DIR": f"{root}/model/lora",
        "ADAPTER_DIR": f"{root}/model/lora/final",
        "DEFAULT_MODEL_PATH": f"{root}/model",
        "DEFAULT_TEXT_DIR": f"{root}/prepared",
        "DEFAULT_CORPUS_DIR": f"{root}/corpus",
        "DEFAULT_OUTPUT_DIR": f"{root}/model/output_partial",
        "DEFAULT_BACKEND": "cuda:0",
        "DEFAULT_MODEL": "openai/gpt-oss-20b",
        "BASE_MODEL": "openai/gpt-oss-20b",
        "GENERATOR_MODEL": "openai/gpt-oss-20b",
        "EMBED_MODEL": "BAAI/bge-base-en-v1.5",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "DEFAULT_DATALOADER_NUM_WORKERS": "2",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def ensure_directories(names: Iterable[str] = ()) -> None:
    default_names = (
        "MODEL_ROOT",
        "RAWTEXT_DIR",
        "PDF_DIR",
        "PREPARED_DIR",
        "RAG_DIR",
        "MODEL_DIR",
        "CORPUS_DIR",
        "OUTPUT_DIR",
        "LORA_DIR",
        "ADAPTER_DIR",
        "DEFAULT_OUTPUT_DIR",
    )
    for name in tuple(default_names) + tuple(names):
        value = os.environ.get(name)
        if value:
            to_local_path(value).mkdir(parents=True, exist_ok=True)


def set_huggingface_token(secret_scope: str = "", secret_key: str = "HF_TOKEN") -> None:
    if os.environ.get("HF_TOKEN"):
        return
    dbutils = dbutils_or_none()
    if not dbutils or not secret_scope:
        return
    try:
        token = dbutils.secrets.get(scope=secret_scope, key=secret_key)
    except Exception as exc:
        print(f"Could not read Hugging Face token from scope={secret_scope!r}: {exc}")
        return
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def widget_text(name: str, default: str, label: str | None = None) -> str:
    dbutils = dbutils_or_none()
    if not dbutils:
        return os.environ.get(name.upper(), default)
    try:
        dbutils.widgets.text(name, default, label or name)
    except Exception:
        pass
    return dbutils.widgets.get(name)


def widget_dropdown(name: str, default: str, choices: list[str], label: str | None = None) -> str:
    dbutils = dbutils_or_none()
    if not dbutils:
        return os.environ.get(name.upper(), default)
    try:
        dbutils.widgets.dropdown(name, default, choices, label or name)
    except Exception:
        pass
    return dbutils.widgets.get(name)


def widget_bool(name: str, default: bool, label: str | None = None) -> bool:
    value = widget_dropdown(name, "true" if default else "false", ["true", "false"], label)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def cli_args(script_name: str, args: list[str]):
    old_argv = sys.argv[:]
    sys.argv = [script_name, *args]
    try:
        yield
    finally:
        sys.argv = old_argv


def run_cli(main_func, script_name: str, args: list[str]) -> int:
    with cli_args(script_name, args):
        result = main_func()
    return 0 if result is None else int(result)
