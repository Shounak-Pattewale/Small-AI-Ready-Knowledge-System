"""Tests for research/evaluation/_hybrid.py (archived - RRF hybrid retrieval was rejected;
moved out of src/knowledge_system/retrieval/ during the research/production cleanup).

Uses two kinds of doubles:
- `_StubRetriever`: a fixed, hand-ranked fake retriever (defined below),
  for tests that need exact control over rank positions to verify RRF
  arithmetic precisely.
- The real TfidfRetriever + EmbeddingRetriever (with the FakeEmbeddingModel
  fixture from conftest.py), for tests that verify HybridRetriever's wiring
  against the real retriever classes without downloading Granite.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from knowledge_system.models import DocumentChunk, SearchResult
from knowledge_system.retrieval.embedding import EmbeddingRetriever

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
sys.path.insert(0, str(_EVAL_DIR))  # so _hybrid.py's own `from _tfidf import TfidfRetriever` resolves


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _EVAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TfidfRetriever = _load("_tfidf").TfidfRetriever
HybridRetriever = _load("_hybrid").HybridRetriever


def make_chunk(chunk_index: int, source: str = "doc.md", text: str = "content", section=None, page=None) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, section=section, page=page, chunk_index=chunk_index)


class _StubRetriever:
    """A fake retriever returning a fixed, pre-ranked SearchResult list for every query.

    `_chunks` is set (like a real fitted retriever) so HybridRetriever's
    constructor validation passes. Scores are deliberately huge/arbitrary
    to prove RRF fusion ignores raw score magnitude - only rank order.
    """

    def __init__(self, chunks: list[DocumentChunk], ranked_chunk_indexes: list[int]) -> None:
        self._chunks = list(chunks)
        self._by_index = {c.chunk_index: c for c in chunks}
        self._ranked_chunk_indexes = ranked_chunk_indexes

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        results = [SearchResult(chunk=self._by_index[idx], score=999.0) for idx in self._ranked_chunk_indexes]
        return results[:top_k]


def make_hybrid(chunks, lexical_order, semantic_order, k=60) -> HybridRetriever:
    lexical = _StubRetriever(chunks, lexical_order)
    semantic = _StubRetriever(chunks, semantic_order)
    return HybridRetriever(lexical, semantic, chunks, k=k)


def test_rrf_formula_is_correct():
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    # chunk 0: lexical rank 1, semantic rank 2. chunk 1: lexical rank 2, semantic rank 1.
    # chunk 2: lexical rank 3, semantic rank 3.
    hybrid = make_hybrid(chunks, lexical_order=[0, 1, 2], semantic_order=[1, 0, 2], k=60)

    results = hybrid.search("query", top_k=3)
    scores = {r.chunk.chunk_index: r.score for r in results}

    assert scores[0] == pytest.approx(1 / (60 + 1) + 1 / (60 + 2))
    assert scores[1] == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))
    assert scores[2] == pytest.approx(1 / (60 + 3) + 1 / (60 + 3))


def test_one_based_ranks_used():
    chunks = [make_chunk(0), make_chunk(1)]
    hybrid = make_hybrid(chunks, lexical_order=[0, 1], semantic_order=[0, 1], k=60)

    results = hybrid.search("query", top_k=2)
    top_score = results[0].score

    # If ranks were 0-based, the top chunk's score would be 2/60 exactly;
    # with correct 1-based ranks it's 2/61.
    assert top_score == pytest.approx(2 / 61)
    assert top_score != pytest.approx(2 / 60)


def test_both_retrievers_contribute_equally():
    chunks = [make_chunk(0), make_chunk(1)]
    # chunk 0 ranks #1 lexically but dead last semantically; chunk 1 is the reverse.
    # With equal weighting, a symmetric swap should give both chunks the identical total score.
    hybrid = make_hybrid(chunks, lexical_order=[0, 1], semantic_order=[1, 0], k=60)

    results = hybrid.search("query", top_k=2)
    scores = {r.chunk.chunk_index: r.score for r in results}

    assert scores[0] == pytest.approx(scores[1])


def test_raw_score_magnitude_does_not_affect_fusion():
    # _StubRetriever always reports score=999.0 regardless of rank - if RRF
    # used raw scores instead of rank position, both chunks would tie.
    chunks = [make_chunk(0), make_chunk(1)]
    hybrid = make_hybrid(chunks, lexical_order=[0, 1], semantic_order=[0, 1], k=60)

    results = hybrid.search("query", top_k=2)

    assert results[0].chunk.chunk_index == 0
    assert results[0].score != pytest.approx(results[1].score)  # rank-based scores differ despite equal raw scores


def test_final_ranking_follows_rrf_score():
    chunks = [make_chunk(i) for i in range(4)]
    # chunk 3 is best on both sides; chunk 0 is worst on both sides.
    hybrid = make_hybrid(chunks, lexical_order=[3, 2, 1, 0], semantic_order=[3, 2, 1, 0])

    results = hybrid.search("query", top_k=4)

    assert [r.chunk.chunk_index for r in results] == [3, 2, 1, 0]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_deterministic_chunk_index_tie_break():
    chunks = [make_chunk(i) for i in range(4)]
    # chunk 0 and chunk 1 swap positions between lexical/semantic -> equal RRF score.
    hybrid = make_hybrid(chunks, lexical_order=[0, 1, 2, 3], semantic_order=[1, 0, 2, 3])

    results = hybrid.search("query", top_k=4)

    assert results[0].chunk.chunk_index == 0  # tie broken by chunk_index ascending
    assert results[1].chunk.chunk_index == 1


def test_configurable_k_changes_scores():
    chunks = [make_chunk(0), make_chunk(1)]
    hybrid_k60 = make_hybrid(chunks, [0, 1], [0, 1], k=60)
    hybrid_k1 = make_hybrid(chunks, [0, 1], [0, 1], k=1)

    score_k60 = hybrid_k60.search("query", top_k=1)[0].score
    score_k1 = hybrid_k1.search("query", top_k=1)[0].score

    assert score_k60 == pytest.approx(2 / 61)
    assert score_k1 == pytest.approx(2 / 2)
    assert score_k60 != score_k1


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_rejected(k):
    chunks = [make_chunk(0)]
    with pytest.raises(ValueError):
        make_hybrid(chunks, [0], [0], k=k)


def test_duplicate_chunk_index_rejected():
    chunks = [make_chunk(0), make_chunk(0)]  # duplicate chunk_index
    with pytest.raises(ValueError, match="duplicate"):
        make_hybrid(chunks, [0], [0])


def test_unfitted_lexical_retriever_rejected():
    chunks = [make_chunk(0)]
    with pytest.raises(RuntimeError):
        HybridRetriever(TfidfRetriever(), _StubRetriever(chunks, [0]), chunks)


def test_empty_question_rejected():
    chunks = [make_chunk(0)]
    hybrid = make_hybrid(chunks, [0], [0])
    with pytest.raises(ValueError):
        hybrid.search("   ", top_k=1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_rejected(top_k):
    chunks = [make_chunk(0)]
    hybrid = make_hybrid(chunks, [0], [0])
    with pytest.raises(ValueError):
        hybrid.search("query", top_k=top_k)


def test_top_k_greater_than_corpus_returns_available_results():
    chunks = [make_chunk(0), make_chunk(1)]
    hybrid = make_hybrid(chunks, [0, 1], [0, 1])

    results = hybrid.search("query", top_k=100)

    assert len(results) == 2


def test_result_contains_original_chunk():
    chunk = make_chunk(0, source="policy.pdf", section="Topic")
    hybrid = make_hybrid([chunk], [0], [0])

    result = hybrid.search("query", top_k=1)[0]

    assert result.chunk is chunk


def test_score_is_the_calculated_rrf_score():
    chunks = [make_chunk(0)]
    hybrid = make_hybrid(chunks, [0], [0], k=60)

    result = hybrid.search("query", top_k=1)[0]

    assert result.score == pytest.approx(1 / 61 + 1 / 61)


# --- wiring against real TfidfRetriever + EmbeddingRetriever(+fake model) ---


def test_semantic_side_does_not_re_encode_documents(mock_embedding_model):
    chunks = [
        make_chunk(0, text="alpha content"),
        make_chunk(1, text="beta content"),
    ]
    lexical = TfidfRetriever()
    lexical.fit(chunks)

    builder = EmbeddingRetriever()
    builder.fit(chunks)  # this "offline build" call is allowed to encode
    builder._model.encode_calls.clear()  # reset so we only observe HybridRetriever's own calls

    hybrid = HybridRetriever(lexical, builder, chunks, k=60)
    hybrid.search("alpha", top_k=1)

    # Only the query should have been encoded by the semantic retriever
    # during search() - never the document chunks again.
    assert builder._model.encode_calls == [["alpha"]]


def test_query_gets_encoded_by_semantic_retriever(mock_embedding_model):
    chunks = [make_chunk(0, text="alpha content")]
    lexical = TfidfRetriever()
    lexical.fit(chunks)

    semantic = EmbeddingRetriever()
    semantic.fit(chunks)
    semantic._model.encode_calls.clear()

    hybrid = HybridRetriever(lexical, semantic, chunks, k=60)
    hybrid.search("alpha", top_k=1)

    assert semantic._model.encode_calls == [["alpha"]]


def test_mismatched_chunk_collections_rejected(mock_embedding_model):
    chunks_a = [make_chunk(0, text="alpha")]
    chunks_b = [make_chunk(1, text="beta")]  # different chunk_index set

    lexical = TfidfRetriever()
    lexical.fit(chunks_a)
    semantic = EmbeddingRetriever()
    semantic.fit(chunks_b)

    with pytest.raises(ValueError, match="different chunk"):
        HybridRetriever(lexical, semantic, chunks_a, k=60)
