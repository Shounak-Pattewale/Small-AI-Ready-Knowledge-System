"""Tests for src/knowledge_system/retrieval/embedding.py.

Uses the FakeEmbeddingModel fixture (tests/conftest.py) so these never
download/load the real Granite model - deterministic bag-of-keywords
vectors over "alpha"/"beta"/"gamma" stand in for real semantics.
"""

import pytest
from knowledge_system.models import DocumentChunk
from knowledge_system.retrieval.embedding import EmbeddingRetriever


def make_chunk(
    text: str,
    source: str = "doc.md",
    section: str | None = None,
    page: int | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, section=section, page=page, chunk_index=chunk_index)


def test_fit_stores_embeddings_corresponding_to_chunks(mock_embedding_model):
    chunks = [
        make_chunk("alpha content", chunk_index=0),
        make_chunk("beta content", chunk_index=1),
    ]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    assert retriever._embeddings.shape[0] == 2
    assert retriever._chunks == chunks


def test_section_and_body_used_as_embedding_input(mock_embedding_model):
    # The body alone shares no keyword with the query; only the section does.
    chunks = [
        make_chunk("unrelated body text", section="alpha topic", chunk_index=0),
        make_chunk("unrelated body text", section="beta topic", chunk_index=1),
    ]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    results = retriever.search("alpha", top_k=1)

    assert results[0].chunk.section == "alpha topic"


def test_body_only_used_when_section_is_none(mock_embedding_model):
    chunks = [
        make_chunk("alpha content here", section=None, chunk_index=0),
        make_chunk("beta content here", section=None, chunk_index=1),
    ]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    results = retriever.search("alpha", top_k=1)

    assert results[0].chunk.text == "alpha content here"


def test_search_before_fit_raises(mock_embedding_model):
    retriever = EmbeddingRetriever()

    with pytest.raises(RuntimeError):
        retriever.search("alpha", top_k=3)


def test_empty_question_raises_value_error(mock_embedding_model):
    retriever = EmbeddingRetriever()
    retriever.fit([make_chunk("alpha content", chunk_index=0)])

    with pytest.raises(ValueError):
        retriever.search("   ", top_k=3)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_value_error(mock_embedding_model, top_k):
    retriever = EmbeddingRetriever()
    retriever.fit([make_chunk("alpha content", chunk_index=0)])

    with pytest.raises(ValueError):
        retriever.search("alpha", top_k=top_k)


def test_top_k_larger_than_corpus_returns_all_chunks(mock_embedding_model):
    chunks = [make_chunk(f"alpha content {i}", chunk_index=i) for i in range(2)]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    results = retriever.search("alpha", top_k=10)

    assert len(results) == 2


def test_ranking_follows_similarity(mock_embedding_model):
    chunks = [
        make_chunk("alpha alpha alpha", chunk_index=0),  # strongest alpha match
        make_chunk("alpha beta", chunk_index=1),  # partial alpha match
        make_chunk("gamma gamma gamma", chunk_index=2),  # no alpha at all
    ]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    results = retriever.search("alpha", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.chunk_index == 0
    assert results[-1].chunk.chunk_index == 2


def test_deterministic_tie_breaking_uses_chunk_index(mock_embedding_model):
    # Identical text -> identical fake vectors -> identical scores for every chunk.
    chunks = [make_chunk("alpha beta gamma", chunk_index=i) for i in [3, 1, 2, 0]]
    retriever = EmbeddingRetriever()
    retriever.fit(chunks)

    results = retriever.search("alpha", top_k=4)

    assert [r.chunk.chunk_index for r in results] == [0, 1, 2, 3]


def test_result_contains_original_chunk(mock_embedding_model):
    chunk = make_chunk("alpha content", source="policy.pdf", section="Alpha", page=3, chunk_index=5)
    retriever = EmbeddingRetriever()
    retriever.fit([chunk])

    result = retriever.search("alpha", top_k=1)[0]

    assert result.chunk is chunk


def test_score_is_raw_similarity_not_percentage(mock_embedding_model):
    retriever = EmbeddingRetriever()
    retriever.fit([make_chunk("alpha content", chunk_index=0)])

    result = retriever.search("alpha", top_k=1)[0]

    # Cosine similarity of normalized vectors is bounded in [-1, 1], never
    # a 0-100 percentage scale.
    assert -1.0 <= result.score <= 1.0
