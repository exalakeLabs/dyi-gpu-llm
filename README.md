# Training And RAG Guide

This repository is already set up to fine-tune **Qwen/Qwen2.5-3B-Instruct** with LoRA.

## RAG With Prepared Text And gpt-oss

Put cleaned source `.txt` files under `prepared/`, or set `PREPARED_DIR` in `.env`.
Then build the FAISS index:

```bash
./build_rag_index.zsh
```

That writes:

- `rag/index.faiss`
- `rag/chunks.jsonl`
- `rag/index_config.json`

Ask gpt-oss questions through the index:

```bash
python3 src/teach_gpt_oss_rag.py --question "What does the prepared material say about this topic?"
```

The script defaults to `openai/gpt-oss-20b`. Override it with:

```bash
python3 src/teach_gpt_oss_rag.py --model openai/gpt-oss-120b
```

To inspect the exact instruction prompt and retrieved passages without loading
the model:

```bash
python3 src/teach_gpt_oss_rag.py --dry-run --question "What should I know?"
python3 src/teach_gpt_oss_rag.py --print-teaching-prompt
```

This does not fine-tune gpt-oss. It "teaches" the model at inference time by
retrieving relevant passages and wrapping them in strict grounding instructions.

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers datasets peft trl accelerate safetensors
```

## 2) Prepare raw text files

Place your source `.txt` files in:

- `prepared/`

If your source is PDF files, run the PDF helpers first (`src/extract_pdfs.py`, then optional cleaning).

## 3) Run the training pipeline

Build `data/train.jsonl` from `prepared/*.txt` and train the LoRA adapter:

```bash
python src/train_pipeline.py
```

The pipeline writes one JSON object per line with a `text` field, then starts LoRA training.

If you only need to rebuild the dataset:

```bash
python src/make_text_dataset.py
```

If you already have `data/train.jsonl` and only need to train:

```bash
python src/train_lora_gpu.py
```

Default training settings in `src/train_lora_gpu.py`:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Data file: `data/train.jsonl`
- Output dir: `output/lora`
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Batch size: `1` (with grad accumulation `8`)
- Epochs: `1`

Final adapter output is saved at:

- `output/lora/final/`

## Continued Pretraining

The partial continued-pretraining path is split into two stages. First build the
packed token corpus from `prepared/`:

```bash
python3 src/generate_pretrain_corpus.py --text_dir prepared --corpus_dir corpus --num_proc 1
```

Then train from the token JSONL files in `corpus/`:

```bash
python3 src/continued_pretrain_partial.py --corpus_dir corpus --eval_prompts eval_prompts.txt
```

The corpus generator writes `corpus/train.jsonl` and `corpus/eval.jsonl`, with
each row containing fixed-length `input_ids` and matching `labels`.

## 4) Test the tuned model

```bash
python src/test_tuned.py
```

## 5) Optional: serve locally

```bash
uvicorn --app-dir src serve_tuned:app --host 127.0.0.1 --port 8000
```

Serving and chat entrypoints share the runtime model loader in `src/model_runtime.py`.

## Practical tuning tips

- If you hit GPU memory errors, reduce `max_length` or increase gradient accumulation while keeping per-device batch size small.
- For better quality, run more epochs and/or increase dataset diversity.
- If your data is instruction/response formatted, adapt preprocessing to produce chat-style examples instead of plain `text` chunks.
