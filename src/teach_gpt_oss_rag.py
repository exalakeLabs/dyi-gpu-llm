#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import truststore

    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

from runtime_env import env_int, env_path, env_str

DEFAULT_MODEL = env_str("GENERATOR_MODEL", "openai/gpt-oss-20b")
DEFAULT_EMBED_MODEL = env_str("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)
RAG_DIR = env_path("RAG_DIR", "rag")
RETRIEVE_K = env_int("RETRIEVE_K", 8)

RAG_TEACHING_PROMPT = """You are a retrieval-grounded assistant.
Reasoning: medium

You will receive user questions plus numbered evidence passages retrieved from a local FAISS RAG index.

Rules:
- Answer only from the supplied evidence passages.
- Cite the passage numbers you used, for example [1] or [2][4].
- If the evidence does not answer the question, say what is missing instead of guessing.
- Keep the answer concise unless the user asks for depth.
- Separate facts from inference when the evidence requires interpretation.
- Never mention internal embedding scores unless the user asks for retrieval diagnostics."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_embed_model(index_dir: Path, requested: str) -> str:
    config_path = index_dir / "index_config.json"
    if config_path.exists() and requested == DEFAULT_EMBED_MODEL:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return config.get("embed_model", requested)
    return requested


def retrieve(
    query: str,
    *,
    embedder: Any,
    index,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    encoded = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vector = np.asarray(encoded, dtype="float32")
    scores, ids = index.search(query_vector, top_k)

    hits: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(chunks[int(idx)])
        row["score"] = float(score)
        hits.append(row)
    return hits


def format_evidence(hits: list[dict[str, Any]], max_context_chars: int) -> str:
    blocks: list[str] = []
    used_chars = 0

    for i, hit in enumerate(hits, start=1):
        title = hit.get("title") or hit.get("book") or hit.get("source", "")
        source = hit.get("source_relpath") or hit.get("source", "")
        chunk = hit.get("chunk_index", hit.get("chunk_id", ""))
        text = hit.get("text", "").strip()
        header = f"[{i}] title={title} source={source} chunk={chunk}"
        block = f"{header}\n{text}"

        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()

        blocks.append(block)
        used_chars += len(block) + 2

    return "\n\n".join(blocks)


def build_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RAG_TEACHING_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved evidence:\n{evidence}\n\n"
                "Answer with citations to the evidence passage numbers."
            ),
        },
    ]


def choose_dtype(dtype: str) -> Any:
    import torch

    if dtype == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_generator(model_name: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": choose_dtype(dtype),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_answer(
    messages: list[dict[str, str]],
    *,
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        tokenize=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teach gpt-oss to answer through a prepared FAISS RAG index."
    )
    parser.add_argument("--rag-dir", default=str(RAG_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=RETRIEVE_K)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--question", help="Ask one question and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print retrieved evidence and the exact chat messages without loading gpt-oss.",
    )
    parser.add_argument(
        "--print-teaching-prompt",
        action="store_true",
        help="Print only the reusable gpt-oss RAG instruction prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.print_teaching_prompt:
        print(RAG_TEACHING_PROMPT)
        return 0

    index_dir = Path(args.rag_dir).expanduser().resolve()
    chunks_path = index_dir / "chunks.jsonl"
    index_path = index_dir / "index.faiss"
    if not chunks_path.exists() or not index_path.exists():
        raise SystemExit(
            f"Missing RAG index files in {index_dir}. Run ./build_rag_index.zsh first."
        )

    embed_model = resolve_embed_model(index_dir, args.embed_model)
    print(f"Loading RAG index: {index_dir}")
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(index_path))
    chunks = load_jsonl(chunks_path)
    print(f"Loading embedder: {embed_model}")
    embedder = SentenceTransformer(embed_model)

    tokenizer = None
    model = None
    if not args.dry_run:
        print(f"Loading gpt-oss model: {args.model}")
        tokenizer, model = load_generator(args.model, args.dtype)

    def answer(question: str) -> None:
        hits = retrieve(
            question,
            embedder=embedder,
            index=index,
            chunks=chunks,
            top_k=args.top_k,
        )
        evidence = format_evidence(hits, args.max_context_chars)
        messages = build_messages(question, evidence)

        if args.dry_run:
            print("\n--- CHAT MESSAGES SENT TO GPT-OSS ---\n")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("\n--- TOP PASSAGES ---")
            for i, hit in enumerate(hits, start=1):
                source = hit.get("source_relpath") or hit.get("source", "")
                print(f"[{i}] {source} score={hit['score']:.4f}")
            return

        print("\nAssistant>\n")
        answer_text = generate_answer(
            messages,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
        )
        print(answer_text)
        print("\nSources:")
        for i, hit in enumerate(hits, start=1):
            source = hit.get("source_relpath") or hit.get("source", "")
            print(f"  [{i}] {source} score={hit['score']:.4f}")

    if args.question:
        answer(args.question)
        return 0

    print("\nRAG teaching chat ready. Type 'exit' to quit.\n")
    while True:
        question = input("You> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer(question)
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
