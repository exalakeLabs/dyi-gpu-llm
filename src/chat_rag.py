#!/usr/bin/env python
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from model_runtime import generate_text, load_generation_model
from project_config import ADAPTER_DIR, env_dir


RAG_DIR = env_dir("LLAMA_RAG_DIR", "rag")

CHUNKS_FILE = RAG_DIR / "chunks.jsonl"
INDEX_FILE = RAG_DIR / "index.faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4


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


def main() -> int:
    rows = load_chunks()
    index = faiss.read_index(str(INDEX_FILE))
    embedder = SentenceTransformer(EMBED_MODEL)

    tokenizer, model = load_generation_model(use_adapter=ADAPTER_DIR.exists())

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

        answer = generate_text(
            tokenizer,
            model,
            messages,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

        print("\n--- ANSWER ---\n")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
