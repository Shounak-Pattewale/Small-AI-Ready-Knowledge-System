"""Persistence for precomputed document embeddings: matrix + manifest on disk.

Separates offline embedding-index building (slow: Granite encodes every
chunk) from online retrieval (fast: only the query gets encoded).
EmbeddingRetriever.load_index() loads what this module writes; it never
recomputes document embeddings once a valid persisted index exists.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from knowledge_system.models import DocumentChunk

SCHEMA_VERSION = 1


def chunk_fingerprint(chunks: list[DocumentChunk]) -> str:
    """Deterministic SHA-256 fingerprint of a chunk collection's retrieval-relevant fields.

    Changes if text, source, section, page, or chunk_index changes for any
    chunk, or if the chunks are reordered - anything that would change what
    a persisted embedding matrix actually represents. The persisted chunks
    (storage/chunks.json) are the source of truth here, not the raw
    PDFs/Markdown documents.
    """
    hasher = hashlib.sha256()
    for chunk in chunks:
        record = {
            "text": chunk.text,
            "source": chunk.source,
            "section": chunk.section,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
        }
        hasher.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\x00")  # separates records so adjacent chunks' fields can't blur together
    return hasher.hexdigest()


def _atomic_write(path: Path, write) -> None:
    """Write via `write(file_obj)` to a temp file, then atomically replace `path`.

    Same pattern as knowledge_system.storage.save_chunks(): a crash
    mid-write leaves the temp file behind but never a half-written
    embeddings.npy/manifest - readers only ever see the old complete file
    or the new complete file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def save_embeddings(
    embeddings: np.ndarray,
    chunks: list[DocumentChunk],
    model_name: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> None:
    """Persist a document embedding matrix plus a manifest describing what it corresponds to.

    Manifest paths are stored as plain filenames only (no machine-specific
    absolute paths) - it just names model/shape/fingerprint, not locations.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"embeddings has shape {embeddings.shape}, expected ({len(chunks)}, embedding_dim)")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "normalized": True,
        "chunk_fingerprint": chunk_fingerprint(chunks),
    }

    _atomic_write(Path(embeddings_path), lambda f: np.save(f, embeddings))
    _atomic_write(Path(manifest_path), lambda f: f.write(json.dumps(manifest, indent=2).encode("utf-8")))


def load_embeddings(
    chunks: list[DocumentChunk],
    model_name: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> np.ndarray:
    """Load and validate a persisted embedding matrix against the given chunks/model.

    Raises a clear exception rather than silently rebuilding or silently
    continuing with stale/mismatched embeddings.
    """
    embeddings_path = Path(embeddings_path)
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No embedding manifest at {manifest_path}. Build the index first: "
            f"venv/bin/python -m knowledge_system.build_embeddings"
        )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Manifest exists but embeddings file is missing: {embeddings_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported embeddings schema_version {manifest.get('schema_version')!r} (expected {SCHEMA_VERSION})"
        )
    if manifest.get("model_id") != model_name:
        raise ValueError(
            f"Persisted embeddings were built with model {manifest.get('model_id')!r}, "
            f"but this retriever expects {model_name!r}"
        )
    if manifest.get("chunk_count") != len(chunks):
        raise ValueError(
            f"Persisted embeddings have {manifest.get('chunk_count')} chunks, "
            f"current corpus has {len(chunks)} - rebuild the embedding index"
        )
    if not manifest.get("normalized"):
        raise ValueError("Persisted embeddings are not normalized, but retrieval requires normalized vectors")

    expected_fingerprint = chunk_fingerprint(chunks)
    if manifest.get("chunk_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Persisted embeddings' chunk fingerprint does not match the current chunk collection "
            "(storage/chunks.json has changed since the embedding index was built). Rebuild it: "
            "venv/bin/python -m knowledge_system.build_embeddings"
        )

    embeddings = np.load(embeddings_path)

    expected_dim = manifest.get("embedding_dimension")
    if embeddings.shape != (len(chunks), expected_dim):
        raise ValueError(
            f"Embedding matrix shape {embeddings.shape} does not match expected ({len(chunks)}, {expected_dim})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Embedding matrix at {embeddings_path} contains non-finite values (NaN/Inf)")

    return embeddings
