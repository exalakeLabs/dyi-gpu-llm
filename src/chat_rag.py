#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from model_runtime import generate_text, load_generation_model
from rag_model_config import validate_embedding_model, validate_generator_model
from runtime_env import env_int, env_path, env_str

ADAPTER_DIR = env_path("ADAPTER_DIR", "output/lora/final")
BASE_MODEL = env_str("BASE_MODEL")
EMBED_MODEL = env_str("EMBED_MODEL")
EMBED_DEVICE = env_str("RAG_EMBED_DEVICE", "auto")
GENERATOR_MODEL = env_str("GENERATOR_MODEL", BASE_MODEL)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 0)
RAG_DIR = env_path("RAG_DIR", "rag")
RETRIEVE_K = env_int("RETRIEVE_K", 24)
SYSTEM_PROMPT = env_str("SYSTEM_PROMPT")


def load_chunks(chunks_file: Path):
    rows = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def retrieve(query, embedder, index, rows, top_k=RETRIEVE_K):
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


def format_context(hits, max_chars: int = MAX_CONTEXT_CHARS):
    parts = []
    remaining = max_chars
    for i, hit in enumerate(hits, start=1):
        text = hit["text"]
        if max_chars > 0:
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "\n[truncated]"
            remaining -= len(text)
        parts.append(
            f"[Context {i} | source={hit['source']} | chunk={hit['chunk_id']} | score={hit['score']:.4f}]\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a local FAISS RAG index.")
    parser.add_argument("--rag-dir", default=str(RAG_DIR))
    parser.add_argument("--base-model", "--generator-model", dest="generator_model", default=GENERATOR_MODEL)
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--embed-device", default=EMBED_DEVICE)
    parser.add_argument("--top-k", type=int, default=RETRIEVE_K)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir).expanduser()
    chunks_file = rag_dir / "chunks.jsonl"
    index_file = rag_dir / "index.faiss"
    adapter_dir = Path(args.adapter_dir).expanduser()

    rows = load_chunks(chunks_file)
    index = faiss.read_index(str(index_file))
    validate_embedding_model(args.embed_model)
    validate_generator_model(args.generator_model)
    embed_device = None if args.embed_device == "auto" else args.embed_device
    print(f"Loading embedder: {args.embed_model} on {embed_device or 'auto'}")
    embedder = SentenceTransformer(args.embed_model, device=embed_device)

    tokenizer, model = load_generation_model(
        base_model=args.generator_model,
        adapter_path=adapter_dir,
        use_adapter=not args.no_adapter and adapter_dir.exists(),
    )

    while True:
        query = input("\nPrompt> ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break

        hits = retrieve(query, embedder, index, rows, top_k=args.top_k)
        context = format_context(hits, max_chars=args.max_context_chars)

        print("\n--- RETRIEVED CONTEXT ---\n")
        for hit in hits:
            print(f"{hit['source']} [chunk {hit['chunk_id']}] score={hit['score']:.4f}")
        if args.max_context_chars > 0:
            print(f"Context character budget: {args.max_context_chars}")

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer using only the context above."
        )

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = generate_text(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

        print("\n--- ANSWER ---\n")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
