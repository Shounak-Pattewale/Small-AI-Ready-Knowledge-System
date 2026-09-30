"""Tests for src/knowledge_system/embeddings.py and EmbeddingRetriever.load_index().

Uses the FakeEmbeddingModel fixture (tests/conftest.py) - never
downloads/loads the real Granite model.
"""

from pathlib import Path

import numpy as np
import pytest
from knowledge_system.embeddings import chunk_fingerprint, load_embeddings, save_embeddings
from knowledge_system.models import DocumentChunk
from knowledge_system.retrieval.embedding import EmbeddingRetriever


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(text="alpha content", source="a.md", section="Intro", page=None, chunk_index=0),
        DocumentChunk(text="beta content", source="b.pdf", section=None, page=3, chunk_index=1),
    ]


def make_embeddings(n_rows: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(n_rows, dim))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)  # unit-normalized, like real output


# --- chunk_fingerprint ---


def test_fingerprint_is_deterministic():
    chunks = make_chunks()
    assert chunk_fingerprint(chunks) == chunk_fingerprint(make_chunks())


def test_fingerprint_changes_when_text_changes():
    chunks = make_chunks()
    original = chunk_fingerprint(chunks)
    chunks[0].text = "changed content"
    assert chunk_fingerprint(chunks) != original


def test_fingerprint_changes_when_metadata_changes():
    chunks = make_chunks()
    original = chunk_fingerprint(chunks)
    chunks[1].page = 4  # page changed, text/source unchanged
    assert chunk_fingerprint(chunks) != original


def test_fingerprint_changes_when_order_changes():
    chunks = make_chunks()
    reversed_chunks = list(reversed(chunks))
    assert chunk_fingerprint(chunks) != chunk_fingerprint(reversed_chunks)


# --- save_embeddings / load_embeddings ---


def test_save_and_load_round_trip(kb_dir: Path):
    chunks = make_chunks()
    embeddings = make_embeddings(len(chunks))
    embeddings_path = kb_dir / "embeddings.npy"
    manifest_path = kb_dir / "manifest.json"

    save_embeddings(embeddings, chunks, "fake-model", embeddings_path, manifest_path)
    loaded = load_embeddings(chunks, "fake-model", embeddings_path, manifest_path)

    assert np.allclose(loaded, embeddings)


def test_save_writes_manifest_with_expected_fields(kb_dir: Path):
    chunks = make_chunks()
    embeddings = make_embeddings(len(chunks), dim=4)
    embeddings_path = kb_dir / "embeddings.npy"
    manifest_path = kb_dir / "manifest.json"

    save_embeddings(embeddings, chunks, "fake-model", embeddings_path, manifest_path)

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_id"] == "fake-model"
    assert manifest["embedding_dimension"] == 4
    assert manifest["chunk_count"] == 2
    assert manifest["normalized"] is True
    assert manifest["chunk_fingerprint"] == chunk_fingerprint(chunks)


def test_matrix_shape_mismatch_rejected_at_save(kb_dir: Path):
    chunks = make_chunks()
    wrong_shape_embeddings = make_embeddings(len(chunks) + 1)  # 3 rows for 2 chunks

    with pytest.raises(ValueError, match="shape"):
        save_embeddings(wrong_shape_embeddings, chunks, "fake-model", kb_dir / "e.npy", kb_dir / "m.json")


def test_chunk_count_mismatch_rejected_at_load(kb_dir: Path):
    chunks = make_chunks()
    save_embeddings(make_embeddings(len(chunks)), chunks, "fake-model", kb_dir / "e.npy", kb_dir / "m.json")

    fewer_chunks = chunks[:1]
    with pytest.raises(ValueError, match="chunk"):
        load_embeddings(fewer_chunks, "fake-model", kb_dir / "e.npy", kb_dir / "m.json")


def test_fingerprint_mismatch_rejected_at_load(kb_dir: Path):
    chunks = make_chunks()
    save_embeddings(make_embeddings(len(chunks)), chunks, "fake-model", kb_dir / "e.npy", kb_dir / "m.json")

    changed_chunks = make_chunks()
    changed_chunks[0].text = "different content now"
    with pytest.raises(ValueError, match="fingerprint"):
        load_embeddings(changed_chunks, "fake-model", kb_dir / "e.npy", kb_dir / "m.json")


def test_model_mismatch_rejected_at_load(kb_dir: Path):
    chunks = make_chunks()
    save_embeddings(make_embeddings(len(chunks)), chunks, "model-a", kb_dir / "e.npy", kb_dir / "m.json")

    with pytest.raises(ValueError, match="model"):
        load_embeddings(chunks, "model-b", kb_dir / "e.npy", kb_dir / "m.json")


def test_embedding_dimension_mismatch_rejected_at_load(kb_dir: Path):
    chunks = make_chunks()
    embeddings_path = kb_dir / "e.npy"
    manifest_path = kb_dir / "m.json"
    save_embeddings(make_embeddings(len(chunks), dim=4), chunks, "fake-model", embeddings_path, manifest_path)

    # Overwrite the .npy with a different (wrong) dimension while leaving
    # the manifest's declared embedding_dimension as-is (still 4).
    np.save(embeddings_path, make_embeddings(len(chunks), dim=8))

    with pytest.raises(ValueError, match="shape"):
        load_embeddings(chunks, "fake-model", embeddings_path, manifest_path)


def test_unsupported_schema_version_rejected(kb_dir: Path):
    chunks = make_chunks()
    embeddings_path = kb_dir / "e.npy"
    manifest_path = kb_dir / "m.json"
    save_embeddings(make_embeddings(len(chunks)), chunks, "fake-model", embeddings_path, manifest_path)

    import json

    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="schema_version"):
        load_embeddings(chunks, "fake-model", embeddings_path, manifest_path)


def test_non_finite_embeddings_rejected(kb_dir: Path):
    chunks = make_chunks()
    embeddings_path = kb_dir / "e.npy"
    manifest_path = kb_dir / "m.json"
    embeddings = make_embeddings(len(chunks))
    save_embeddings(embeddings, chunks, "fake-model", embeddings_path, manifest_path)

    # Corrupt the saved matrix with a NaN after the manifest was already written correctly.
    corrupted = embeddings.copy()
    corrupted[0][0] = float("nan")
    np.save(embeddings_path, corrupted)

    with pytest.raises(ValueError, match="non-finite"):
        load_embeddings(chunks, "fake-model", embeddings_path, manifest_path)


def test_missing_manifest_fails_clearly(kb_dir: Path):
    with pytest.raises(FileNotFoundError, match="Build the index"):
        load_embeddings(make_chunks(), "fake-model", kb_dir / "e.npy", kb_dir / "m.json")


def test_persisted_embeddings_preserve_chunk_ordering(kb_dir: Path):
    chunks = [DocumentChunk(text=f"chunk {i}", source="a.md", section=None, page=None, chunk_index=i) for i in range(5)]
    embeddings = np.arange(5 * 4, dtype=float).reshape(5, 4)
    embeddings_path = kb_dir / "e.npy"
    manifest_path = kb_dir / "m.json"

    save_embeddings(embeddings, chunks, "fake-model", embeddings_path, manifest_path)
    loaded = load_embeddings(chunks, "fake-model", embeddings_path, manifest_path)

    assert np.array_equal(loaded, embeddings)  # row i still corresponds to chunks[i]


# --- EmbeddingRetriever.load_index() ---


def test_search_after_loading_persisted_embeddings_works(mock_embedding_model, kb_dir: Path):
    chunks = [
        DocumentChunk(text="alpha content", source="a.md", section=None, page=None, chunk_index=0),
        DocumentChunk(text="beta content", source="b.md", section=None, page=None, chunk_index=1),
    ]
    # Build a real (fake-model) index first, exactly as the offline builder would.
    builder = EmbeddingRetriever()
    builder.fit(chunks)
    from knowledge_system.embeddings import save_embeddings as save

    save(builder._embeddings, chunks, builder._model_name, kb_dir / "e.npy", kb_dir / "m.json")

    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, kb_dir / "e.npy", kb_dir / "m.json")
    results = retriever.search("alpha", top_k=1)

    assert results[0].chunk.text == "alpha content"


def test_document_encoding_not_invoked_after_load_index(mock_embedding_model, kb_dir: Path):
    chunks = [DocumentChunk(text="alpha content", source="a.md", section=None, page=None, chunk_index=0)]
    builder = EmbeddingRetriever()
    builder.fit(chunks)
    from knowledge_system.embeddings import save_embeddings as save

    save(builder._embeddings, chunks, builder._model_name, kb_dir / "e.npy", kb_dir / "m.json")

    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, kb_dir / "e.npy", kb_dir / "m.json")

    # load_index() itself must not have called encode() at all - it reads
    # the persisted matrix, it doesn't re-embed the chunk text.
    assert retriever._model.encode_calls == []


def test_query_encoding_still_occurs_during_search(mock_embedding_model, kb_dir: Path):
    chunks = [DocumentChunk(text="alpha content", source="a.md", section=None, page=None, chunk_index=0)]
    builder = EmbeddingRetriever()
    builder.fit(chunks)
    from knowledge_system.embeddings import save_embeddings as save

    save(builder._embeddings, chunks, builder._model_name, kb_dir / "e.npy", kb_dir / "m.json")

    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, kb_dir / "e.npy", kb_dir / "m.json")
    retriever.search("alpha", top_k=1)

    assert retriever._model.encode_calls == [["alpha"]]  # exactly one call, for the query only
