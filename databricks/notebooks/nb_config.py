# Databricks notebook source
# =============================================================================
# nb_config · Centralized notebook configuration
#
# Edit variables here instead of using widget prompts.
# Every pipeline notebook starts with:  %run ./nb_config
# =============================================================================

# COMMAND ----------

# ---------------------------------------------------------------------------
# Shared — used by all notebooks
# ---------------------------------------------------------------------------
dbfs_root: str = "/Volumes/customer_success/exalabs_writeback/llrun"

# ---------------------------------------------------------------------------
# 01_download_corpus
# ---------------------------------------------------------------------------
max_books: int  = 100               # max books per Gutendex query
profile: str    = "smart_assistant" # smart_assistant | tech_biz | bookish

# ---------------------------------------------------------------------------
# 02_clean_and_index
# ---------------------------------------------------------------------------
embed_model: str      = "BAAI/bge-base-en-v1.5"
embed_batch_size: int = 128   # batch size for sentence-transformers encoding
chunk_size: int       = 1800  # chunk size in characters
overlap: int          = 250   # overlap in characters between consecutive chunks
max_files: int        = 0     # 0 = process all files

# ---------------------------------------------------------------------------
# 03_train_lora
# ---------------------------------------------------------------------------
base_model: str       = "Qwen/Qwen2.5-3B-Instruct"
num_gpus: int         = 4
# True = single-node multi-GPU (TorchDistributor local_mode=True)
# False = multi-node distributed
local_mode: bool      = True
num_epochs: float     = 1.0
train_batch_size: int = 2     # per-device batch size during LoRA training
grad_accum: int       = 8     # gradient accumulation steps
learning_rate: float  = 3e-4
max_length: int       = 512   # max sequence length (tokens)
lora_r: int           = 16    # LoRA rank
max_pair_files: int   = 0     # 0 = use all prepared files for pair generation

# ---------------------------------------------------------------------------
# 04_rag_chat
# ---------------------------------------------------------------------------
rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k_faiss: int  = 24        # number of FAISS candidates to retrieve
top_k_rerank: int = 6         # number of results to keep after reranking
max_new_tok: int  = 300       # max new tokens for answer generation
use_adapter: str  = "auto"    # auto | yes | no
query: str        = "What is the nature of light?"
