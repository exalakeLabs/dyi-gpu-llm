# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Text

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = next(
    root for root in (Path.cwd(), *Path.cwd().parents) if (root / "src" / "databricks_env.py").exists()
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from databricks_env import (
    ensure_directories,
    run_cli,
    set_default_env,
    to_local_path,
    widget_dropdown,
    widget_text,
)

model_root = widget_text("model_root", "/dbfs/FileStore/llama32-local", "Model/data root")
mode = widget_dropdown("mode", "clean_text", ["clean_text", "extract_pdfs"], "Prepare mode")
input_dir = widget_text("input_dir", os.environ.get("RAWTEXT_DIR", f"{model_root}/raw-text"), "Input dir")
output_dir = widget_text("output_dir", os.environ.get("PREPARED_DIR", f"{model_root}/prepared"), "Output dir")

set_default_env(model_root)
os.environ["RAWTEXT_DIR"] = str(to_local_path(input_dir))
os.environ["PREPARED_DIR"] = str(to_local_path(output_dir))
ensure_directories()

if mode == "clean_text":
    import clean_text

    args = [
        "--input-dir",
        str(to_local_path(input_dir)),
        "--output-dir",
        str(to_local_path(output_dir)),
    ]
    result = run_cli(clean_text.main, "clean_text.py", args)
    if result:
        raise RuntimeError(f"clean_text failed with exit code {result}")

if mode == "extract_pdfs":
    import extract_pdfs

    args = [
        "--pdf-dir",
        str(to_local_path(input_dir)),
        "--text-dir",
        str(to_local_path(output_dir)),
    ]
    result = run_cli(extract_pdfs.main, "extract_pdfs.py", args)
    if result:
        raise RuntimeError(f"extract_pdfs failed with exit code {result}")

raise ValueError(f"Unsupported mode: {mode}")
