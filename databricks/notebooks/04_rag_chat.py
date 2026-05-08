# Databricks notebook source
# =============================================================================
# 04 · RAG Chat
# Interactive question-answering over the Gutenberg corpus using the FAISS
# index built in notebook 02 and (optionally) the LoRA adapter from notebook 03.
# Requires a GPU cluster.
# =============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04 · RAG Chat
# MAGIC
# MAGIC Retrieves relevant passages from the FAISS index, reranks with a
# MAGIC cross-encoder, then generates an answer with `Qwen2.5-3B-Instruct`
# MAGIC (base model or LoRA-adapted).
# MAGIC
# MAGIC **Usage**: edit `query` (and other parameters) in `nb_config.py`, then
# MAGIC run cells top to bottom.  Re-run only the **Query** cell to test a new query.
# MAGIC
# MAGIC **Cluster**: GPU recommended for fast generation.

# COMMAND ----------

# MAGIC %pip install --quiet sentence-transformers faiss-cpu truststore

# COMMAND ----------

# MAGIC %run ./nb_config

# COMMAND ----------

import os, sys
from pathlib import Path

os.environ["LLAMA_DBFS_ROOT"] = dbfs_root

_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

rag_dir     = Path(f"{dbfs_root}/rag")
adapter_dir = Path(f"{dbfs_root}/output/lora/final")

try:
    import truststore
    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

print(f"RAG dir      : {rag_dir}")
print(f"Adapter dir  : {adapter_dir}")
print(f"Embed model  : {embed_model}")
print(f"Rerank model : {rerank_model}")

# COMMAND ----------

# MAGIC %md ### Load FAISS index and chunks

# COMMAND ----------

import json
import faiss
import numpy as np

# Re-inject src/ — %pip install restarts the kernel and wipes sys.path.
_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

index_file  = rag_dir / "index.faiss"
chunks_file = rag_dir / "chunks.jsonl"

if not index_file.exists():
    raise FileNotFoundError(f"FAISS index not found: {index_file}\nRun notebook 02 first.")
if not chunks_file.exists():
    raise FileNotFoundError(f"Chunks file not found: {chunks_file}\nRun notebook 02 first.")

index  = faiss.read_index(str(index_file))
chunks = []
with chunks_file.open("r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"FAISS index  : {index.ntotal:,} vectors")
print(f"Chunks loaded: {len(chunks):,}")

# COMMAND ----------

# MAGIC %md ### Load embedding model

# COMMAND ----------

from sentence_transformers import SentenceTransformer, CrossEncoder

print(f"Loading embedder: {embed_model} …")
embedder = SentenceTransformer(embed_model, model_kwargs={"use_safetensors": True})

print(f"Loading reranker: {rerank_model} …")
reranker = CrossEncoder(rerank_model, model_kwargs={"use_safetensors": True})

print("Models ready.")

# COMMAND ----------

# MAGIC %md ### Load generation model

# COMMAND ----------

import os, sys

# Redirect HF cache to local SSD — DBFS FUSE does not support syscalls used
# by the HF XET downloader (copy_file_range, sendfile).
os.environ["HF_HOME"] = "/local_disk0/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
_LOCAL_CACHE = "/local_disk0/hf_cache/hub"

import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_CACHE = _LOCAL_CACHE

# Fix src/ path — upstream cells use [1:4] which skips 'Users' and goes one
# level too deep.  Correct slice is [0:3] → Users/<user>/<repo>.
_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[0:3])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from project_config import BASE_MODEL

_should_use_adapter = (
    adapter_dir.exists() if use_adapter == "auto"
    else (use_adapter == "yes")
)

print(f"Loading base model: {BASE_MODEL} …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, cache_dir=_LOCAL_CACHE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=dtype,
    device_map="auto",
    attn_implementation="sdpa",
    cache_dir=_LOCAL_CACHE,
)

if _should_use_adapter:
    print(f"Loading LoRA adapter: {adapter_dir} …")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    print("Adapter merged into base model.")
else:
    print("Using base model (no adapter).")

model.eval()
print("Generation model ready.")

# COMMAND ----------

# MAGIC %md ### RAG helper functions

# COMMAND ----------

def retrieve(query: str, k: int) -> list:
    """FAISS approximate nearest-neighbour retrieval."""
    q_emb = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, ids = index.search(q_emb, k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(chunks[int(idx)])
        row["faiss_score"] = float(score)
        hits.append(row)
    return hits


def rerank(query: str, hits: list, top_n: int) -> list:
    """Cross-encoder reranking of FAISS candidates."""
    pairs  = [(query, h["text"]) for h in hits]
    scores = reranker.predict(pairs)
    for hit, s in zip(hits, scores):
        hit["rerank_score"] = float(s)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_n]


def generate(query: str, context_hits: list) -> str:
    """Generate an answer grounded in the retrieved context."""
    context_parts = []
    for i, hit in enumerate(context_hits, start=1):
        context_parts.append(
            f"[{i}] {hit.get('title', 'Unknown')} (by {hit.get('author', '?')})\n"
            f"{hit['text']}"
        )
    context_str = "\n\n".join(context_parts)

    system_prompt = (
        "You are a grounded assistant. Answer using only the context passages provided. "
        "If the answer is not clearly supported by the context, say: "
        "\"I don't know based on the provided documents.\""
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context_str}\n\n"
        "Answer concisely, citing the passage numbers above."
    )

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tok,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("RAG functions defined.")

# COMMAND ----------

# MAGIC %md ### Query

# COMMAND ----------

print(f"Query: {query!r}\n")

# Step 1 — retrieve
candidates = retrieve(query, top_k_faiss)
print(f"FAISS candidates: {len(candidates)}")

# Step 2 — rerank
top_hits = rerank(query, candidates, top_k_rerank)
print(f"After reranking (top {top_k_rerank}):")
for i, h in enumerate(top_hits, 1):
    print(f"  [{i}] rerank={h['rerank_score']:.3f}  faiss={h['faiss_score']:.3f}  "
          f"{h.get('title','?')[:60]}")

# Step 3 — generate
answer = generate(query, top_hits)

print(f"\n{'='*60}")
print(f"ANSWER:\n{answer}")
print('='*60)

# COMMAND ----------

# MAGIC %md ### Display retrieved passages

# COMMAND ----------

rows = [
    (
        i + 1,
        h.get("title", "?")[:60],
        h.get("author", "?")[:40],
        round(h["rerank_score"], 4),
        h["text"][:200] + "…",
    )
    for i, h in enumerate(top_hits)
]
display(spark.createDataFrame(rows, ["rank", "title", "author", "rerank_score", "excerpt"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ RAG pipeline complete.
# MAGIC
# MAGIC - Edit `query` in `nb_config.py` and re-run the **Query** cell to ask different questions.
# MAGIC - Adjust `top_k_faiss` / `top_k_rerank` in `nb_config.py` to trade recall vs. latency.
# MAGIC - Set `use_adapter = "yes"` in `nb_config.py` to use the fine-tuned LoRA adapter from notebook 03.
