"""Explicit offline embedding-index build: storage/chunks.json -> storage/embeddings.npy + manifest.

This is the ONLY normal workflow that runs Granite over the whole chunk
collection. Runtime retrieval (EmbeddingRetriever.load_index()) must only
ever load this output, never rebuild it automatically.

Run from the project root:
    venv/bin/python -m knowledge_system.build_embeddings
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge_system.embeddings import chunk_fingerprint, save_embeddings
from knowledge_system.retrieval.embedding import _DEFAULT_MODEL_NAME, searchable_text
from knowledge_system.storage import load_chunks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHUNKS_PATH = _PROJECT_ROOT / "storage" / "chunks.json"
_DEFAULT_EMBEDDINGS_PATH = _PROJECT_ROOT / "storage" / "embeddings.npy"
_DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / "storage" / "embeddings_manifest.json"


def build_embeddings(
    chunks_path: Path = _DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = _DEFAULT_EMBEDDINGS_PATH,
    manifest_path: Path = _DEFAULT_MANIFEST_PATH,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> None:
    """Load persisted chunks, encode them with Granite, and save the resulting index. No hidden runtime fallback calls this."""
    chunks = load_chunks(chunks_path)  # persisted corpus only - never re-parses raw documents

    model = SentenceTransformer(model_name)
    texts = [searchable_text(chunk) for chunk in chunks]  # exact same construction EmbeddingRetriever uses
    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True))

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"Unexpected embedding matrix shape {embeddings.shape} for {len(chunks)} chunks")

    save_embeddings(embeddings, chunks, model_name, embeddings_path, manifest_path)

    print(f"Model: {model_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Chunk fingerprint: {chunk_fingerprint(chunks)}")
    print(f"Saved: {embeddings_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    build_embeddings()
