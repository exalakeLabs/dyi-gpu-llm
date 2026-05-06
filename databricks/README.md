# llama32-local on Databricks

Run the full RAG + LoRA fine-tuning pipeline on a Databricks GPU cluster.

## Architecture

```
Project Gutenberg           DBFS (persists across cluster restarts)
─────────────────           ───────────────────────────────────────
Gutendex API ──► 01_download_corpus.py ──► {dbfs_root}/text/
                 02_clean_and_index.py ──► {dbfs_root}/prepared/
                                      ──► {dbfs_root}/rag/          (FAISS index)
                 03_train_lora.py     ──► {dbfs_root}/output/lora/
                 04_rag_chat.py       ◄── all of the above
```

All DBFS paths default to `/dbfs/FileStore/llama32` (configurable via the
`dbfs_root` widget on each notebook).

## Prerequisites

| Requirement | Details |
|---|---|
| Databricks ML Runtime | 14.x or later (pre-installs torch, transformers, peft, accelerate) |
| Cluster type | GPU for notebooks 02–04 (`g4dn.xlarge` / `g5.xlarge` minimum) |
| HuggingFace token | Required only if the base model is gated |

## Quick Start

### 1 · Import the repository into Databricks Repos

1. In the Databricks sidebar go to **Repos → Add Repo**.
2. Paste your git URL and click **Create Repo**.

### 2 · Create a GPU cluster with the init script

1. Create a new cluster (or edit an existing one).
2. Under **Advanced Options → Init Scripts**, add the DBFS path to
   `databricks/cluster_init.sh` (upload it first with
   `dbutils.fs.cp("file:/path/to/cluster_init.sh", "dbfs:/FileStore/llama32/cluster_init.sh")`).
3. Alternatively, install dependencies manually via the **Libraries** tab using
   `databricks/requirements_databricks.txt`.

### 3 · Run the notebooks in order

| Notebook | Cluster | Purpose |
|---|---|---|
| `00_cluster_setup.py` | GPU | Verify GPU, create DBFS dirs, smoke-test config |
| `01_download_corpus.py` | CPU | Download Gutenberg books → DBFS |
| `02_clean_and_index.py` | GPU | Clean text + build FAISS index → DBFS |
| `03_train_lora.py` | GPU | Generate training pairs + LoRA fine-tune → MLflow |
| `04_rag_chat.py` | GPU | Interactive RAG Q&A |

Each notebook exposes **widgets** at the top — adjust them before running.
The defaults work end-to-end with no changes required.

## DBFS Directory Layout

```
/dbfs/FileStore/llama32/
├── text/           raw Gutenberg .txt files          (notebook 01)
├── prepared/       cleaned .txt files                (notebook 02)
├── data/
│   ├── train.jsonl training pairs                   (notebook 03)
│   └── val.jsonl
├── rag/
│   ├── index.faiss HNSW FAISS index                 (notebook 02)
│   ├── chunks.jsonl chunk metadata                  (notebook 02)
│   └── index_config.json
├── output/
│   └── lora/
│       ├── checkpoint-*/  mid-training checkpoints  (notebook 03)
│       └── final/         merged LoRA adapter       (notebook 03)
└── hf_cache/       HuggingFace model cache
```

## MLflow Tracking

Notebook 03 logs every run to the MLflow experiment
`/llama32-local/lora-finetune` (configurable via the `MLFLOW_EXPERIMENT_NAME`
environment variable).

Each run records:
- Hyperparameters (model, LoRA rank, batch size, learning rate, …)
- Training metrics (loss per step via `report_to="mlflow"`)
- The final LoRA adapter directory as an artifact (`lora_adapter/`)

View runs in the Databricks sidebar under **Experiments**.

## Multi-GPU Training

Notebook 03 uses
[TorchDistributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/torch-distributor.html)
for single-node multi-GPU training.

Set the **GPUs** widget to the number of GPUs on your cluster
(e.g. `4` for a `g5.12xlarge`). The effective batch size scales linearly:

```
effective_batch = per_device_batch × num_gpus × grad_accum
```

For multi-node training set `local_mode=False` in `_train_worker` and ensure
`LLAMA_SHARED_OUTPUT_DIR` points to shared storage (DBFS works).

## Source Module Path

Each notebook auto-detects the repo root from `notebookPath` and appends
`src/` to `sys.path`, so you can import `project_config`, `index_builder`,
`make_training_pairs`, etc. without installing them as a package.

`project_config.py` detects whether it is running on Databricks by checking
`DATABRICKS_RUNTIME_VERSION` and switches all path constants between local
(`~/llrun/`) and DBFS (`/dbfs/FileStore/llama32/`) accordingly.

## Corpus Profiles (notebook 01)

| Profile | Books | Focus |
|---|---|---|
| `smart_assistant` | ~2 700 | Science, maths, philosophy, history |
| `tech_biz` | ~1 750 | Engineering, economics, law, statistics |
| `bookish` | ~1 350 | History, philosophy, travel, biography |
