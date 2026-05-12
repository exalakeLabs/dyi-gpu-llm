#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEFAULT_GENERATOR = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def retrieve(
    query: str,
    index,
    embedder: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    top_k: int,
):
    q = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q, top_k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(chunks[idx])
        row["retrieval_score"] = float(score)
        out.append(row)
    return out

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    rerank_tokenizer,
    rerank_model,
    device: str,
    top_n: int,
):
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        logits = rerank_model(**inputs).logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits.squeeze(1)
        elif logits.ndim == 1:
            scores = logits
        else:
            scores = logits[:, 0]

        scores = torch.sigmoid(scores).detach().cpu().numpy()

    rescored = []
    for cand, score in zip(candidates, scores):
        row = dict(cand)
        row["rerank_score"] = float(score)
        rescored.append(row)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[:top_n]

def build_context(passages: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        header = f"[{i}] Title: {p.get('title','')} | Author: {p.get('author','')} | Chunk: {p.get('chunk_index','')}"
        body = p["text"].strip()
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)

def generate_answer(
    query: str,
    context: str,
    gen_tokenizer,
    gen_model,
    device: str,
    max_new_tokens: int = 500,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful literary research assistant. "
                "Answer using only the retrieved passages when possible. "
                "If the passages are insufficient, say so clearly. "
                "Cite passage numbers like [1], [2]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                "Give a concise answer grounded in the passages."
            ),
        },
    ]

    text = gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = gen_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=gen_tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return gen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--retrieve-k", type=int, default=24)
    parser.add_argument("--rerank-top-n", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    chunks_path = index_dir / "chunks.jsonl"
    index_path = index_dir / "index.faiss"
    config_path = index_dir / "index_config.json"

    # If --embed-model was not explicitly supplied, use the model recorded at
    # index-build time so the query dimension always matches the index dimension.
    if args.embed_model == DEFAULT_EMBED_MODEL and config_path.exists():
        saved_model = json.loads(config_path.read_text())["embed_model"]
        if saved_model != args.embed_model:
            print(f"Note: using embed model from index config: {saved_model}")
            args.embed_model = saved_model

    print("Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("Loading chunk metadata...")
    chunks = load_chunks(chunks_path)

    print(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading reranker: {args.reranker_model}")
    rerank_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_model
    ).to(device)
    rerank_model.eval()

    print(f"Loading generator: {args.generator_model}")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    gen_model.eval()

    print("\nRAG chat ready. Type 'exit' to quit.\n")

    while True:
        query = input("You> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(
            query=query,
            index=index,
            embedder=embedder,
            chunks=chunks,
            top_k=args.retrieve_k,
        )
        reranked = rerank(
            query=query,
            candidates=retrieved,
            rerank_tokenizer=rerank_tokenizer,
            rerank_model=rerank_model,
            device=device,
            top_n=args.rerank_top_n,
        )

        context = build_context(reranked)
        answer = generate_answer(
            query=query,
            context=context,
            gen_tokenizer=gen_tokenizer,
            gen_model=gen_model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        print("\nAssistant>\n")
        print(answer)
        print("\nTop passages:")
        for i, p in enumerate(reranked, 1):
            print(
                f"  [{i}] {p.get('title','')} — {p.get('author','')} "
                f"(chunk {p.get('chunk_index')}, rerank={p.get('rerank_score',0):.4f})"
            )
        print("")

if __name__ == "__main__":
    main()
