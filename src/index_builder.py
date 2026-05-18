#!/usr/bin/env python

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any

# Delegate SSL verification to the macOS Security framework so the Sigma
# corporate proxy CA (which lacks keyUsage, causing Python 3.14/OpenSSL 3.x
# to reject it) is trusted via the system keychain instead of certifi.
# Must run before any import that touches ssl (sentence_transformers, httpx…).
try:
    import truststore
    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def iter_text_files(root: Path):
    for path in root.rglob("*.txt"):
        if path.is_file():
            yield path


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def chunk_text(
    text: str,
    chunk_size_chars: int = 1800,
    overlap_chars: int = 250,
) -> List[str]:
    text = clean_text(text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size_chars, n)

        if end < n:
            newline_break = text.rfind("\n\n", start, end)
            if newline_break > start + 400:
                end = newline_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def extract_title_author(path: Path, text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = path.stem
    author = ""

    for line in lines[:80]:
        low = line.lower()
        if low.startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
        elif low.startswith("author:"):
            author = line.split(":", 1)[1].strip()
        if title and author:
            break

    return {"title": title, "author": author}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt books")
    parser.add_argument("--output-dir", required=True, help="Directory for FAISS index + metadata")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunk-size-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_text_files(input_dir))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    print(f"Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    all_meta: List[Dict[str, Any]] = []
    batch_chunks: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    all_embeddings = []

    chunk_id = 0

    for file_path in tqdm(files, desc="Reading books"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[skip] {file_path}: {e}")
            continue

        info = extract_title_author(file_path, text)
        chunks = chunk_text(
            text,
            chunk_size_chars=args.chunk_size_chars,
            overlap_chars=args.overlap_chars,
        )

        for i, chunk in enumerate(chunks):
            meta = {
                "chunk_id": chunk_id,
                "source_path": str(file_path),
                "title": info["title"],
                "author": info["author"],
                "chunk_index": i,
                "text": chunk,
            }
            batch_chunks.append(chunk)
            batch_meta.append(meta)
            chunk_id += 1

            if len(batch_chunks) >= args.batch_size * 16:
                embs = embed_texts(embedder, batch_chunks, batch_size=args.batch_size)
                all_embeddings.append(embs)
                all_meta.extend(batch_meta)
                batch_chunks = []
                batch_meta = []

    if batch_chunks:
        embs = embed_texts(embedder, batch_chunks, batch_size=args.batch_size)
        all_embeddings.append(embs)
        all_meta.extend(batch_meta)

    if not all_embeddings:
        raise SystemExit("No chunks were embedded.")

    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]

    # IndexHNSWFlat: approximate nearest-neighbour graph index.
    # ~10-50× faster at query time than IndexFlatIP with <1% quality loss.
    # M=32 (edges per node) is a good default; raise to 64 for higher recall.
    # No training step required — safe to add vectors immediately.
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "embed_model": args.embed_model,
        "index_type": "faiss.IndexHNSWFlat(M=32)",
        "normalize_embeddings": True,
        "chunk_size_chars": args.chunk_size_chars,
        "overlap_chars": args.overlap_chars,
        "num_files": len(files),
        "num_chunks": len(all_meta),
        "dim": dim,
    }
    (output_dir / "index_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved index to: {output_dir / 'index.faiss'}")
    print(f"Saved metadata to: {output_dir / 'chunks.jsonl'}")
    print(f"Indexed files: {len(files)}")
    print(f"Indexed chunks: {len(all_meta)}")


if __name__ == "__main__":
    main()
