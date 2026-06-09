# Training And RAG Guide

This repository is set up to run RAG/runtime generation through
**openai/gpt-oss-20b** with Transformers, while keeping BGE models for
embedding and reranking.

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

Default training settings come from `.env.default`:

- Base model: `openai/gpt-oss-20b`
- Data file: `${CORPUS_DIR}/train.jsonl`
- Output dir: `${MODEL_DIR}/output_partial`
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Batch size: `1` (with grad accumulation `8`)
- Epochs: `1`

Final adapter output is saved at:

- `${ADAPTER_DIR}`

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

## GPT-OSS Runtime

The default runtime model settings live in `.env.default`:

```bash
EMBED_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
GENERATOR_MODEL=openai/gpt-oss-20b
BASE_MODEL=openai/gpt-oss-20b
```

`EMBED_MODEL` and `RERANKER_MODEL` must stay on embedding/reranking models.
Do not point them at gpt-oss; that will try to load the generator as an
embedder and can exhaust GPU memory during index builds.

Build the RAG index with the embedding model, then launch chat with
Transformers:

```bash
./build_rag_index.zsh
./launch_chat.zsh
```

On low-VRAM NVIDIA cards such as a 12 GB RTX 5070, the launcher keeps CUDA
visible and uses native MXFP4 with a conservative GPU memory cap. The BGE
embedder stays on CPU by default so the generator gets the VRAM:

```bash
RAG_EMBED_DEVICE=cpu
GENERATOR_DEVICE_MAP=auto
GENERATOR_GPU_MEMORY=6GiB
GENERATOR_MXFP4_DEQUANTIZE=0
MAX_CONTEXT_CHARS=2048
MAX_NEW_TOKENS=96
```

On bf16-capable CUDA cards, the runtime uses bf16 automatically unless
`GENERATOR_DTYPE` is set. If CUDA still reports `device not ready` during
generation, reboot the host to reset the driver, then lower the cap further:

```bash
GENERATOR_GPU_MEMORY=4GiB ./launch_chat.zsh
```

On 8 GB Radeon cards such as the RX 7600, Transformers may dequantize the
gpt-oss MXFP4 checkpoint instead of running it in-place as 4-bit weights. The
launcher defaults ROCm low-VRAM cards to CPU with explicit MXFP4 dequantization,
while leaving the GPU visible for the RAG embedder:

```bash
RAG_EMBED_DEVICE=auto
GENERATOR_DEVICE_MAP=cpu
GENERATOR_MXFP4_DEQUANTIZE=1
```

By default, `GENERATOR_CPU_MEMORY` is computed from host RAM and reserves 8 GiB
for the OS and other processes. On a 32 GB server, Linux usually reports about
31 GiB usable RAM, so the cap is about `23GiB`. Override it only when you know
the machine has more headroom:

```bash
GENERATOR_CPU_MEMORY=24GiB ./launch_chat.zsh
```

For a hybrid ROCm run, opt in explicitly and leave conversion headroom:

```bash
LOW_VRAM_ROCM_RUNTIME=rocm
RAG_EMBED_DEVICE=rocm
GENERATOR_GPU_MEMORY=3GiB
./launch_chat.zsh
```

If the explicit dequantized CPU path still touches the GPU on your stack, fully
hide the GPU from the Python process:

```bash
LOW_VRAM_HIDE_GPU=1 ./launch_chat.zsh
```

Increase `GENERATOR_GPU_MEMORY` only if there is free VRAM after the model
loads. This can run, but it will be much slower than native MXFP4 execution on
supported hardware.

## Practical tuning tips

- If you hit GPU memory errors, reduce `max_length` or increase gradient accumulation while keeping per-device batch size small.
- For better quality, run more epochs and/or increase dataset diversity.
- If your data is instruction/response formatted, adapt preprocessing to produce chat-style examples instead of plain `text` chunks.
