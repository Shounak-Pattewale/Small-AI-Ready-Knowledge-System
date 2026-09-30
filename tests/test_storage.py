"""Tests for src/knowledge_system/storage.py."""

import json
from pathlib import Path

import pytest
from knowledge_system.models import DocumentChunk
from knowledge_system.storage import load_chunks, save_chunks


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(text="First chunk.", source="a.md", section="Intro", page=None, chunk_index=0),
        DocumentChunk(text="Second chunk.", source="b.pdf", section=None, page=3, chunk_index=1),
        DocumentChunk(text="Third chunk.", source="b.pdf", section="Topic > Sub", page=3, chunk_index=2),
    ]


def test_save_and_load_round_trip(kb_dir: Path):
    chunks = make_chunks()
    path = kb_dir / "chunks.json"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert loaded == chunks


def test_all_fields_survive_correctly(kb_dir: Path):
    chunks = make_chunks()
    path = kb_dir / "chunks.json"
    save_chunks(chunks, path)

    loaded = load_chunks(path)

    for original, reloaded in zip(chunks, loaded):
        assert reloaded.text == original.text
        assert reloaded.source == original.source
        assert reloaded.section == original.section
        assert reloaded.page == original.page
        assert reloaded.chunk_index == original.chunk_index


def test_none_section_and_page_survive(kb_dir: Path):
    chunk = DocumentChunk(text="No metadata.", source="a.md", section=None, page=None, chunk_index=0)
    path = kb_dir / "chunks.json"

    save_chunks([chunk], path)
    loaded = load_chunks(path)

    assert loaded[0].section is None
    assert loaded[0].page is None


def test_deterministic_order_survives(kb_dir: Path):
    chunks = [DocumentChunk(text=f"chunk {i}", source="a.md", section=None, page=None, chunk_index=i) for i in range(10)]
    path = kb_dir / "chunks.json"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert [c.chunk_index for c in loaded] == list(range(10))


def test_missing_file_fails_clearly(kb_dir: Path):
    with pytest.raises(FileNotFoundError, match="ingestion"):
        load_chunks(kb_dir / "does_not_exist.json")


def test_malformed_json_fails_clearly(kb_dir: Path):
    path = kb_dir / "chunks.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_chunks(path)


def test_unsupported_schema_version_fails_clearly(kb_dir: Path):
    path = kb_dir / "chunks.json"
    path.write_text(json.dumps({"schema_version": 99, "chunks": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_chunks(path)


def test_invalid_top_level_structure_fails_clearly(kb_dir: Path):
    path = kb_dir / "chunks.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level"):
        load_chunks(path)


def test_chunks_not_a_list_fails_clearly(kb_dir: Path):
    path = kb_dir / "chunks.json"
    path.write_text(json.dumps({"schema_version": 1, "chunks": "not a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="'chunks' must be a list"):
        load_chunks(path)


def test_missing_required_chunk_field_fails_clearly(kb_dir: Path):
    path = kb_dir / "chunks.json"
    incomplete = {"text": "hi", "source": "a.md", "chunk_index": 0}  # missing section, page
    path.write_text(json.dumps({"schema_version": 1, "chunks": [incomplete]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        load_chunks(path)


def test_atomic_save_replaces_destination(kb_dir: Path):
    path = kb_dir / "chunks.json"
    save_chunks(make_chunks(), path)
    assert len(load_chunks(path)) == 3

    # Save again with different content - the destination must end up
    # fully replaced, and no leftover temp files should remain beside it.
    save_chunks([DocumentChunk(text="Replaced.", source="c.md", section=None, page=None, chunk_index=0)], path)

    loaded = load_chunks(path)
    assert len(loaded) == 1
    assert loaded[0].text == "Replaced."
    leftover_tmp_files = list(kb_dir.glob(".*.tmp"))
    assert leftover_tmp_files == []
