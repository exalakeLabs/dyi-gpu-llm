#!/usr/bin/env python
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_dir(var: str, default_rel: str) -> Path:
    v = os.environ.get(var, "").strip()
    p = Path(v).expanduser() if v else (REPO_ROOT / default_rel)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p

PREPARED_DIR = env_dir("LLAMA_PREPARED_DIR", "prepared")
RAG_DIR = env_dir("LLAMA_RAG_DIR", "rag")
RAG_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_FILE = RAG_DIR / "chunks.jsonl"
INDEX_FILE = RAG_DIR / "index.faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150):
    text = text.strip()
    if not text:
        return

    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if len(chunk) > 120:
            yield chunk
        if end == n:
            break
        start = max(0, end - overlap)

def load_chunks():
    rows = []
    for txt_file in sorted(PREPARED_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            rows.append(
                {
                    "id": len(rows),
                    "source": txt_file.name,
                    "chunk_id": i,
                    "text": chunk,
                }
            )
    return rows

def main():
    rows = load_chunks()
    if not rows:
        raise RuntimeError("No chunks found in prepared/*.txt")

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    texts = [r["text"] for r in rows]

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Wrote {len(rows)} chunks to {CHUNKS_FILE}")
    print(f"Wrote index to {INDEX_FILE}")
    print(f"Embedding dim: {dim}")

if __name__ == "__main__":
    main()