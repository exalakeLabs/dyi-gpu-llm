import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

RAG_DIR = Path("rag")
CHUNKS_FILE = RAG_DIR / "chunks.jsonl"
INDEX_FILE = RAG_DIR / "index.faiss"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = Path("output/lora/final")  # optional
USE_ADAPTER = ADAPTER_PATH.exists()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

# ROCm workaround for tiny torch.isin GPU calls
_orig_isin = torch.isin
def _safe_isin(elements, test_elements, *args, **kwargs):
    elems_dev = getattr(elements, "device", None)
    test_dev = getattr(test_elements, "device", None)

    if (elems_dev is not None and elems_dev.type == "cuda") or (
        test_dev is not None and test_dev.type == "cuda"
    ):
        out = _orig_isin(elements.cpu(), test_elements.cpu(), *args, **kwargs)
        if elems_dev is not None and elems_dev.type == "cuda":
            return out.to(elems_dev)
        if test_dev is not None and test_dev.type == "cuda":
            return out.to(test_dev)
    return _orig_isin(elements, test_elements, *args, **kwargs)

torch.isin = _safe_isin

def load_chunks():
    rows = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def retrieve(query, embedder, index, rows, top_k=TOP_K):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, top_k)

    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = rows[int(idx)]
        row = dict(row)
        row["score"] = float(score)
        hits.append(row)
    return hits

def format_context(hits):
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[Context {i} | source={hit['source']} | chunk={hit['chunk_id']} | score={hit['score']:.4f}]\n"
            f"{hit['text']}"
        )
    return "\n\n".join(parts)

def main():
    rows = load_chunks()
    index = faiss.read_index(str(INDEX_FILE))
    embedder = SentenceTransformer(EMBED_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if USE_ADAPTER:
        model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
            autocast_adapter_dtype=False,
        )
    else:
        model = base_model

    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    system_prompt = (
        "You are a grounded assistant. Answer only using the provided context from the local document corpus. "
        "If the answer is not clearly supported by the context, say: "
        "'I don't know based on the provided documents.' "
        "Do not use outside knowledge."
    )

    while True:
        query = input("\nPrompt> ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break

        hits = retrieve(query, embedder, index, rows, top_k=TOP_K)
        context = format_context(hits)

        print("\n--- RETRIEVED CONTEXT ---\n")
        for hit in hits:
            print(f"{hit['source']} [chunk {hit['chunk_id']}] score={hit['score']:.4f}")

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer using only the context above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        print("\n--- ANSWER ---\n")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()