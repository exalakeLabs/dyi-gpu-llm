#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path

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

def load_chunks(prepared_dir: Path, max_chars: int, overlap: int):
    rows = []
    for txt_file in sorted(prepared_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
            rows.append(
                {
                    "id": len(rows),
                    "source": txt_file.name,
                    "chunk_id": i,
                    "text": chunk,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS RAG index from prepared text files.")
    parser.add_argument("--prepared-dir", default=str(repo_path("prepared")))
    parser.add_argument("--rag-dir", default=str(repo_path("rag")))
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--chunk-max-chars", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    prepared_dir = Path(args.prepared_dir).expanduser()
    rag_dir = Path(args.rag_dir).expanduser()
    rag_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = rag_dir / "chunks.jsonl"
    index_file = rag_dir / "index.faiss"

    rows = load_chunks(
        prepared_dir=prepared_dir,
        max_chars=args.chunk_max_chars,
        overlap=args.chunk_overlap,
    )
    if not rows:
        raise RuntimeError(f"No chunks found in {prepared_dir}/*.txt")

    with chunks_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    texts = [r["text"] for r in rows]

    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
    index.add(embeddings)
    faiss.write_index(index, str(index_file))

    print(f"Wrote {len(rows)} chunks to {chunks_file}")
    print(f"Wrote index to {index_file}")
    print(f"Embedding dim: {dim}")

if __name__ == "__main__":
    main()
