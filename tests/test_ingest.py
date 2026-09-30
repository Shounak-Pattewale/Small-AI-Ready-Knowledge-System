"""Tests for src/knowledge_system/ingest.py.

Mocks load_directory()/chunk_sections()/save_chunks() at the module
boundary so this never invokes real Docling parsing - it only verifies
that run_ingestion() wires load -> chunk -> save together correctly.
"""

from pathlib import Path

from knowledge_system import ingest
from knowledge_system.models import DocumentChunk, DocumentSection


def test_run_ingestion_wires_load_chunk_save_together(monkeypatch, kb_dir: Path, capsys):
    fake_sections = [DocumentSection(text="body", source="a.md", section=None, page=None)]
    fake_chunks = [
        DocumentChunk(text="body", source="a.md", section=None, page=None, chunk_index=0),
        DocumentChunk(text="more", source="b.pdf", section="Topic", page=1, chunk_index=1),
    ]

    calls = {}

    def fake_load_directory(data_dir):
        calls["load_directory_arg"] = data_dir
        return fake_sections

    def fake_chunk_sections(sections):
        calls["chunk_sections_arg"] = sections
        return fake_chunks

    def fake_save_chunks(chunks, path):
        calls["save_chunks_args"] = (chunks, path)

    monkeypatch.setattr(ingest, "load_directory", fake_load_directory)
    monkeypatch.setattr(ingest, "chunk_sections", fake_chunk_sections)
    monkeypatch.setattr(ingest, "save_chunks", fake_save_chunks)

    data_dir = kb_dir / "data"
    chunks_path = kb_dir / "storage" / "chunks.json"
    ingest.run_ingestion(data_dir=data_dir, chunks_path=chunks_path)

    # load_directory -> chunk_sections -> save_chunks, each fed the previous step's output
    assert calls["load_directory_arg"] == data_dir
    assert calls["chunk_sections_arg"] == fake_sections
    assert calls["save_chunks_args"] == (fake_chunks, chunks_path)


def test_run_ingestion_prints_summary(monkeypatch, kb_dir: Path, capsys):
    fake_sections = [DocumentSection(text="body", source="a.md", section=None, page=None)]
    fake_chunks = [
        DocumentChunk(text="body one", source="a.md", section=None, page=None, chunk_index=0),
        DocumentChunk(text="body two", source="a.md", section=None, page=None, chunk_index=1),
        DocumentChunk(text="body three", source="b.pdf", section="Topic", page=1, chunk_index=2),
    ]

    monkeypatch.setattr(ingest, "load_directory", lambda data_dir: fake_sections)
    monkeypatch.setattr(ingest, "chunk_sections", lambda sections: fake_chunks)
    monkeypatch.setattr(ingest, "save_chunks", lambda chunks, path: None)

    chunks_path = kb_dir / "storage" / "chunks.json"
    ingest.run_ingestion(data_dir=kb_dir / "data", chunks_path=chunks_path)

    output = capsys.readouterr().out
    assert "Loaded: 1 sections" in output
    assert "Created: 3 chunks" in output
    assert str(chunks_path) in output
    assert "a.md" in output and "2" in output  # per-source count for a.md
    assert "b.pdf" in output


def test_run_ingestion_default_paths_point_at_project_data_and_storage():
    assert ingest._DEFAULT_DATA_DIR.name == "data"
    assert ingest._DEFAULT_CHUNKS_PATH == ingest._PROJECT_ROOT / "storage" / "chunks.json"
