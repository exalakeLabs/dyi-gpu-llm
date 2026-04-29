# Training Guide (LoRA)

This repository is already set up to fine-tune **Qwen/Qwen2.5-3B-Instruct** with LoRA.

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

If your source is PDF files, use extraction scripts first (`scripts/extract_pdfs.py`, then optional cleaning).

## 3) Build the training dataset

Create `data/train.jsonl` expected by the trainer:

```bash
python scripts/make_text_dataset.py
```

This writes one JSON object per line with a `text` field.

## 4) Start LoRA training

```bash
python scripts/train_lora.py
```

Default training settings in `scripts/train_lora.py`:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Data file: `data/train.jsonl`
- Output dir: `output/lora`
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Batch size: `1` (with grad accumulation `8`)
- Epochs: `1`

Final adapter output is saved at:

- `output/lora/final/`

## 5) Test the tuned model

```bash
python scripts/test_tuned.py
```

## 6) Optional: serve locally

```bash
python scripts/serve_tuned.py
```

## Practical tuning tips

- If you hit GPU memory errors, reduce `max_length` or increase gradient accumulation while keeping per-device batch size small.
- For better quality, run more epochs and/or increase dataset diversity.
- If your data is instruction/response formatted, adapt preprocessing to produce chat-style examples instead of plain `text` chunks.
