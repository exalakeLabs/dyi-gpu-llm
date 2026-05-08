# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# ///
# =============================================================================
# 02 · Clean Text & Build FAISS Index → DBFS
# Cleans the raw Gutenberg corpus, then embeds chunks and builds the RAG index.
# Run on a GPU cluster (GPU accelerates sentence-transformers embedding).
# =============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Clean Text & Build FAISS Index
# MAGIC
# MAGIC **Stage A — Text cleaning** (`{dbfs_root}/text/` → `{dbfs_root}/prepared/`)
# MAGIC Uses Spark to parallelize the cleaning step across all workers.
# MAGIC
# MAGIC **Stage B — Embedding & indexing** (`{dbfs_root}/prepared/` → `{dbfs_root}/rag/`)
# MAGIC Embeds text chunks with `BAAI/bge-base-en-v1.5` and stores them in a
# MAGIC `faiss.IndexHNSWFlat` (approximate NN, ~10–50× faster than IndexFlatIP at
# MAGIC query time with negligible recall loss).
# MAGIC
# MAGIC **Cluster**: GPU recommended for the embedding step (Stage B).
# MAGIC **Estimated time**: 20–60 min depending on corpus size.

# COMMAND ----------

# MAGIC %pip install --quiet sentence-transformers faiss-cpu truststore faiss-cpu sentence-transformers truststore 

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %run ./nb_config

# COMMAND ----------

import os, sys
batch_size = embed_batch_size

os.environ["LLAMA_DBFS_ROOT"]    = dbfs_root
os.environ["LLAMA_TEXT_DIR"]     = f"{dbfs_root}/text"
os.environ["LLAMA_PREPARED_DIR"] = f"{dbfs_root}/prepared"
os.environ["LLAMA_RAG_DIR"]      = f"{dbfs_root}/rag"
# Reduce allocator fragmentation — helps when GPU memory is partially occupied
# by Spark executors or previous notebook sessions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from pathlib import Path
text_dir     = Path(f"{dbfs_root}/text")
prepared_dir = Path(f"{dbfs_root}/prepared")
rag_dir      = Path(f"{dbfs_root}/rag")
#prepared_dir.mkdir(parents=True, exist_ok=True)
#rag_dir.mkdir(parents=True, exist_ok=True)

print(f"DBFS root    : {dbfs_root}")
print(f"Text dir     : {text_dir}")
print(f"Prepared dir : {prepared_dir}")
print(f"RAG dir      : {rag_dir}")
print(f"Embed model  : {embed_model}")
print(f"Batch size   : {batch_size}")

# COMMAND ----------

# MAGIC %md ### Stage A · Clean text with Spark

# COMMAND ----------

import re

def clean_text(text: str) -> str:
    """Gutenberg plain-text cleaning (mirrors src/clean_text.py logic)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # join wrapped lines
    text = re.sub(r"[ \t]{2,}", " ", text)              # normalise spaces
    return text.strip()


txt_files = sorted(text_dir.glob("*.txt"))
if max_files > 0:
    txt_files = txt_files[:max_files]

print(f"Files to clean: {len(txt_files)}")

# Distribute cleaning across Spark workers (map over DBFS paths).
#files_rdd = spark.parallelize(
#    [str(p) for p in txt_files], numSlices=min(len(txt_files), 200)
#)

def clean_file(path_str: str):
    import re
    from pathlib import Path

    def _clean(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    p = Path(path_str)
    out = Path(path_str.replace("/text/", "/prepared/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return ("skipped", p.name)
    try:
        raw     = p.read_text(encoding="utf-8", errors="ignore")
        cleaned = _clean(raw)
        out.write_text(cleaned, encoding="utf-8")
        return ("ok", p.name)
    except Exception as e:
        return ("error", f"{p.name}: {e}")

results = [clean_file(str(p)) for p in txt_files]

ok      = sum(1 for s, _ in results if s == "ok")
skipped = sum(1 for s, _ in results if s == "skipped")
errors  = [(n) for s, n in results if s == "error"]

print(f"Cleaned : {ok}")
print(f"Skipped : {skipped}  (already existed)")
print(f"Errors  : {len(errors)}")
for e in errors[:10]:
    print(f"  [error] {e}")

total_prepared = len(list(prepared_dir.glob("*.txt")))
print(f"\nTotal files in prepared_dir: {total_prepared}")

# COMMAND ----------

# MAGIC %md ### Stage B · Embed chunks & build FAISS index

# COMMAND ----------

import json
import math
import os
import sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

prepared_dir = Path(f"{dbfs_root}/prepared")


# Re-inject src/ into sys.path — %pip install restarts the kernel and wipes
# any sys.path changes made in earlier cells.
_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)  # Ensure 'index_builder.py' is in this directory.

from src.index_builder import chunk_text, extract_title_author

# Truststore is needed on the driver when downloading the HF model on Mac/corp proxy.
try:
    import truststore
    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

prep_files = sorted(prepared_dir.glob("*.txt"))
if max_files > 0:
    prep_files = prep_files[:max_files]

# ---------------------------------------------------------------------------
# GPU memory hygiene — Spark executors (Stage A) and previous notebooks may
# have left allocations on the GPU.  Release them before loading the embedder.
# ---------------------------------------------------------------------------
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # mem_get_info() reports true free memory across ALL processes on the GPU,
    # unlike memory_reserved() which only covers this Python process.
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb  = free_bytes  / 1e9
    total_gb = total_bytes / 1e9
    alloc_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Free (global) : {free_gb:.1f} GB / {total_gb:.1f} GB")
    print(f"Allocated here: {alloc_gb:.2f} GB")
    if free_gb < 2.0:
        print("⚠️  Less than 2 GB free — detach other notebooks from this cluster "
              "before running Stage B, or use a dedicated indexing cluster.")
else:
    print("No CUDA device — running on CPU")

print(f"\nFiles to index: {len(prep_files)}")
print(f"Loading embedding model: {embed_model} …")

# Load in FP16 to halve model memory (~450 MB FP32 → ~225 MB FP16).
# bge-base-en-v1.5 produces identical results in FP16 for retrieval tasks.
# Use "dtype" (not the deprecated "torch_dtype") for transformers ≥ 4.45.
embedder = SentenceTransformer(
    embed_model,
    model_kwargs={"dtype": torch.float16},
)
if torch.cuda.is_available():
    alloc_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"Model loaded — GPU allocated: {alloc_gb:.2f} GB")

all_meta        = []
batch_chunks    = []
batch_meta_buf  = []
all_embeddings  = []
chunk_id        = 0

for file_path in tqdm(prep_files, desc="Chunking & embedding"):
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[skip] {file_path.name}: {e}")
        continue

    info   = extract_title_author(file_path, text)
    chunks = chunk_text(text, chunk_size_chars=chunk_size, overlap_chars=overlap)

    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_id":    chunk_id,
            "source_path": str(file_path),
            "title":       info["title"],
            "author":      info["author"],
            "chunk_index": i,
            "text":        chunk,
        }
        batch_chunks.append(chunk)
        batch_meta_buf.append(meta)
        chunk_id += 1

        # flush every 512 chunks to bound memory
        if len(batch_chunks) >= batch_size * 8:
            embs = embedder.encode(
                batch_chunks,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            all_embeddings.append(embs)
            all_meta.extend(batch_meta_buf)
            batch_chunks    = []
            batch_meta_buf  = []

# flush remainder
if batch_chunks:
    embs = embedder.encode(
        batch_chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    all_embeddings.append(embs)
    all_meta.extend(batch_meta_buf)

if not all_embeddings:
    raise RuntimeError("No chunks were embedded — check prepared_dir.")

embeddings = np.vstack(all_embeddings).astype("float32")
dim        = embeddings.shape[1]
print(f"\nTotal chunks: {len(all_meta)}, dim: {dim}")

# COMMAND ----------

# MAGIC %md #### Build IndexHNSWFlat and save

# COMMAND ----------

# IndexHNSWFlat: approximate NN graph index.
# M=32 edges per node — good recall/speed trade-off for this corpus size.
# No training step required; vectors can be added immediately.
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.add(embeddings)
print(f"HNSW index built: {index.ntotal} vectors")

faiss.write_index(index, str(rag_dir / "index.faiss"))

with (rag_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
    for row in all_meta:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

config = {
    "embed_model":         embed_model,
    "index_type":          "faiss.IndexHNSWFlat(M=32)",
    "normalize_embeddings": True,
    "chunk_size_chars":    chunk_size,
    "overlap_chars":       overlap,
    "num_files":           len(prep_files),
    "num_chunks":          len(all_meta),
    "dim":                 dim,
}
(rag_dir / "index_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print(f"Saved index      : {rag_dir / 'index.faiss'}")
print(f"Saved chunks     : {rag_dir / 'chunks.jsonl'}")
print(f"Saved config     : {rag_dir / 'index_config.json'}")

# COMMAND ----------

# MAGIC %md ### Preview index stats

# COMMAND ----------

import json
cfg = json.loads((rag_dir / "index_config.json").read_text())
display(spark.createDataFrame([(k, str(v)) for k, v in cfg.items()], ["key", "value"]))

# COMMAND ----------

# MAGIC %md ### ✅ Index built — proceed to notebook 03.
