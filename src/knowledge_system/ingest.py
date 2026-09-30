"""Explicit offline ingestion: raw documents -> DocumentChunk[] -> storage/chunks.json.

This is the ONLY normal workflow that parses raw PDFs/Markdown (Docling
runs here). Retrieval, evaluation, and any future web app must load
chunks via knowledge_system.storage.load_chunks() instead of calling
load_directory() themselves.

Run from the project root:
    venv/bin/python -m knowledge_system.ingest
"""

from pathlib import Path

from knowledge_system.chunking import chunk_sections
from knowledge_system.loaders import load_directory
from knowledge_system.storage import save_chunks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_CHUNKS_PATH = _PROJECT_ROOT / "storage" / "chunks.json"


def run_ingestion(data_dir: Path = _DEFAULT_DATA_DIR, chunks_path: Path = _DEFAULT_CHUNKS_PATH) -> None:
    """Load, chunk, and persist the corpus; print a summary. No hidden retrieval-side fallback calls this."""
    sections = load_directory(data_dir)
    chunks = chunk_sections(sections)  # existing chunker, default chunk_size/chunk_overlap
    save_chunks(chunks, chunks_path)

    print(f"Loaded: {len(sections)} sections")
    print(f"Created: {len(chunks)} chunks")
    print(f"Saved: {chunks_path}")

    per_source: dict[str, int] = {}
    for chunk in chunks:
        per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
    print("\nChunks per source:")
    for source in sorted(per_source):
        print(f"  {source:<30} {per_source[source]}")


if __name__ == "__main__":
    run_ingestion()
