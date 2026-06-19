# llama32-local

Local data preparation, RAG indexing, LoRA training, and chat/inference tooling
for Hugging Face causal language models.

The project is split into functional modules so each workflow can run
independently:

```text
project-root/
  install.zsh
  .env.example
  data_prep/      # downloads, cleanup, corpus creation, training-pair generation
  rag/            # chunking, embedding, FAISS index creation, index metadata
  training/       # LoRA/SFT and continued-pretraining entrypoints
  inference/      # base, adapter, RAG, and adapter+RAG chat/inference
  utils/          # shared env/http helpers plus PDF/OCR/text conversion utilities
  scripts/        # guided orchestration
  src/            # compatibility wrappers for old src/... commands
```

The old `src/*.py` entrypoints remain as thin wrappers, so existing commands and
notebooks can keep working while new workflows use the module folders directly.

## Configuration

Copy the example env and edit paths/models:

```bash
cp .env.example .env
```

Existing environment variable names are preserved. Important model variables:

```bash
EMBED_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
GENERATOR_MODEL=openai/gpt-oss-20b
BASE_MODEL=${GENERATOR_MODEL}
```

Keep `EMBED_MODEL` and `RERANKER_MODEL` on embedding/reranking models. Do not
point them at `gpt-oss`; use `GENERATOR_MODEL` / `BASE_MODEL` for the chat
model.

## Install

Use the existing installer; its behavior is unchanged:

```bash
./install.zsh
```

If you install manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Data Preparation

Run the complete corpus stage: download raw source text, clean it, create
packed continued-pretrain corpus files, and write training-pair JSONL files:

```bash
./pipeline.zsh corpus --jobs 8
```

Run individual corpus steps when iterating:

```bash
./pipeline.zsh raw-text --jobs 8
./pipeline.zsh clean-text
./pipeline.zsh pretrain-corpus
./pipeline.zsh pairs
```

Utility helpers live in `utils/`, for example:

```bash
python3 utils/extract_pdfs.py --pdf-dir "$PDF_DIR" --text-dir "$RAWTEXT_DIR"
python3 utils/pdf_to_txt.py --pdf-dir "$PDF_DIR" --text-dir "$RAWTEXT_DIR"
```

## RAG Index Building

Build a FAISS RAG index independently of any LoRA training:

```bash
./pipeline.zsh rag
```

Equivalent module entrypoint:

```bash
python3 rag/index_builder.py \
  --input-dir "$PREPARED_DIR" \
  --output-dir "$RAG_DIR" \
  --embed-model "$EMBED_MODEL"
```

The index builder writes:

- `index.faiss`
- `chunks.jsonl`
- `index_config.json`

The chat runtime reads `index_config.json` and uses the recorded embedding model
when it differs from the current environment.

## LoRA Training

Run the existing GPU/CPU-selecting training pipeline:

```bash
./pipeline.zsh lora
```

Or call the module directly:

```bash
python3 training/train_pipeline.py
```

Run a specific trainer:

```bash
python3 training/train_lora_gpu.py
python3 training/train_lora_cpu.py
```

Continued pretraining remains a separate workflow:

```bash
./pipeline.zsh pretrain
```

Pass trainer-specific arguments after `--`:

```bash
./pipeline.zsh pretrain -- --corpus_dir "$CORPUS_DIR"
```

LoRA training consumes prepared datasets from `data_prep/`; it does not require
a RAG index.

## Chat And Inference

The main chat entrypoint supports all runtime combinations:

```bash
# Base model only
python3 inference/chat_rag.py --no-rag --no-adapter

# Base model + LoRA adapter
python3 inference/chat_rag.py --no-rag

# Base model + RAG
python3 inference/chat_rag.py --no-adapter

# Base model + LoRA adapter + RAG
python3 inference/chat_rag.py
```

The top-level launcher still works and applies GPU/runtime defaults:

```bash
./chat.zsh
```

For gpt-oss teaching-style RAG inspection:

```bash
python3 inference/teach_gpt_oss_rag.py --question "What does the prepared material say?"
python3 inference/teach_gpt_oss_rag.py --dry-run --question "What should I know?"
python3 inference/teach_gpt_oss_rag.py --print-teaching-prompt
```

## Guided Pipeline

Use the guided orchestrator to choose which stages to run:

```bash
python3 scripts/run_pipeline.py
```

Preview selected commands without running them:

```bash
python3 scripts/run_pipeline.py --dry-run
```

The orchestrator only coordinates. It calls `pipeline.zsh corpus`,
`pipeline.zsh rag`, `training/train_pipeline.py`, and
`inference/chat_rag.py` rather than duplicating their implementation logic.

For the full top-level training flow without prompts:

```bash
./pipeline.zsh --build-corpus --build-rag --pretrain
```

## GPU Runtime Notes

On high-VRAM NVIDIA GPUs such as an A100, `chat.zsh` defaults to a
single-GPU generator placement and avoids the old 5 GiB memory cap. On low-VRAM
NVIDIA/ROCm cards, it keeps conservative defaults and prints the selected
device map, memory cap, dtype, and attention settings.

Useful overrides:

```bash
GENERATOR_DEVICE_MAP=auto ./chat.zsh
GENERATOR_GPU_MEMORY=36GiB ./chat.zsh
GENERATOR_COMPILE=1 ./chat.zsh
RAG_EMBED_DEVICE=cuda ./chat.zsh
```

## Compatibility Notes

- `install.zsh`, `.runtime`, and `.env` loading behavior are preserved.
- `src/*.py` files are compatibility wrappers around the new modules.
- Existing environment variable names remain valid.
- RAG indexes are reusable as long as the embedding model matches the index
  metadata.
- LoRA adapters are reusable only with the same base model they were trained
  against.
