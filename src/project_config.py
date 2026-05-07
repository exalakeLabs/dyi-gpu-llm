import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Databricks detection
# ---------------------------------------------------------------------------
# DATABRICKS_RUNTIME_VERSION is injected automatically into every Databricks
# cluster.  When it is present we default all paths to DBFS so models, data,
# and checkpoints survive cluster restarts.
IS_DATABRICKS: bool = "DATABRICKS_RUNTIME_VERSION" in os.environ

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    value = os.environ.get(var, "").strip()
    path = Path(value).expanduser() if value else REPO_ROOT / default_rel
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


# ---------------------------------------------------------------------------
# Directory layout — DBFS on Databricks, local env vars elsewhere
# ---------------------------------------------------------------------------
if IS_DATABRICKS:
    # All paths live under a single DBFS root so they are shared across the
    # driver and all worker nodes.  Override with LLAMA_DBFS_ROOT if you use
    # Unity Catalog Volumes or a different mount point.
    _dbfs_root = Path(
        os.environ.get("LLAMA_DBFS_ROOT", "/Volumes/customer_success/exalabs_writeback/fileupload")
    )

    TEXT_DIR     = _dbfs_root / "text"
    PREPARED_DIR = _dbfs_root / "prepared"
    DATA_DIR     = _dbfs_root / "data"
    OUTPUT_DIR   = _dbfs_root / "output"
    RAG_DIR      = _dbfs_root / "rag"

    # Pin the HuggingFace cache to DBFS so large model weights are downloaded
    # once and reused across cluster restarts and jobs.
    _hf_cache = str(_dbfs_root / "hf_cache")
    os.environ.setdefault("HF_HOME", _hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)
else:
    TEXT_DIR     = env_dir("LLAMA_TEXT_DIR",     "text")
    PREPARED_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")
    DATA_DIR     = env_dir("LLAMA_DATA_DIR",     "data")
    OUTPUT_DIR   = env_dir("LLAMA_OUTPUT_DIR",   "output")
    RAG_DIR      = env_dir("LLAMA_RAG_DIR",      "rag")

# ---------------------------------------------------------------------------
# Derived paths (same structure on both runtimes)
# ---------------------------------------------------------------------------
TRAIN_FILE  = DATA_DIR  / "train.jsonl"
VAL_FILE    = DATA_DIR  / "val.jsonl"
LORA_DIR    = OUTPUT_DIR / "lora"
ADAPTER_DIR = LORA_DIR  / "final"

# ---------------------------------------------------------------------------
# Shared storage for distributed / Spark training
# ---------------------------------------------------------------------------
# On Databricks this defaults to the DBFS output dir so all workers agree on
# a single checkpoint path without extra configuration.
_shared_env = os.environ.get("LLAMA_SHARED_OUTPUT_DIR", "").strip()
if _shared_env:
    SHARED_OUTPUT_DIR: Path | None = Path(_shared_env).expanduser()
elif IS_DATABRICKS:
    SHARED_OUTPUT_DIR = OUTPUT_DIR
else:
    SHARED_OUTPUT_DIR = None

# ---------------------------------------------------------------------------
# MLflow experiment (Databricks managed tracking by default)
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "llama32-lora-finetune"
)
