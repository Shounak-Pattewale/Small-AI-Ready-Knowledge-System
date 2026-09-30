"""Persistence for DocumentChunk objects: a small, inspectable JSON store.

This is the boundary between offline ingestion (loaders + Docling +
chunking - slow, run explicitly via ingest.py) and online retrieval/
evaluation (fast, must never touch raw documents or Docling). Retrieval
code loads chunks from here; it never calls load_directory() itself.
"""

import json
import os
import tempfile
from pathlib import Path

from knowledge_system.models import DocumentChunk

_SCHEMA_VERSION = 1
_REQUIRED_CHUNK_FIELDS = {"text", "source", "section", "page", "chunk_index"}


def save_chunks(chunks: list[DocumentChunk], path: Path) -> None:
    """Write `chunks` to `path` as JSON, atomically.

    Atomicity: write to a temporary file in the same directory, then
    os.replace() it onto the destination. os.replace() is a single
    filesystem rename, so a crash mid-write leaves the temp file behind
    but never a half-written chunks.json - readers only ever see the old
    complete file or the new complete file, never something in between.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "chunks": [
            {
                "text": chunk.text,
                "source": chunk.source,
                "section": chunk.section,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
    }

    # dir=path.parent keeps the temp file on the same filesystem as the
    # destination, which os.replace() requires in order to be atomic.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)  # clean up the temp file if writing failed partway through
        raise


def load_chunks(path: Path) -> list[DocumentChunk]:
    """Read `path` and reconstruct the persisted DocumentChunk list.

    Raises a clear, specific exception (never silently returns an empty
    list) for: missing file, malformed JSON, wrong top-level shape,
    unsupported schema version, or a chunk missing a required field.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted chunk corpus at {path}. Run ingestion first: "
            f"venv/bin/python -m knowledge_system.ingest"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a top-level JSON object, got {type(payload).__name__}")

    schema_version = payload.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {schema_version!r} (expected {_SCHEMA_VERSION})")

    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError(f"{path}: 'chunks' must be a list, got {type(raw_chunks).__name__}")

    chunks: list[DocumentChunk] = []
    for i, raw in enumerate(raw_chunks):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: chunks[{i}] must be an object, got {type(raw).__name__}")
        missing = _REQUIRED_CHUNK_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"{path}: chunks[{i}] is missing required field(s): {sorted(missing)}")
        if not isinstance(raw["text"], str) or not isinstance(raw["source"], str):
            raise ValueError(f"{path}: chunks[{i}] 'text'/'source' must be strings")
        if not isinstance(raw["chunk_index"], int):
            raise ValueError(f"{path}: chunks[{i}] 'chunk_index' must be an int")
        chunks.append(
            DocumentChunk(
                text=raw["text"],
                source=raw["source"],
                section=raw["section"],
                page=raw["page"],
                chunk_index=raw["chunk_index"],
            )
        )

    return chunks
