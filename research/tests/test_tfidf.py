"""Tests for research/evaluation/_tfidf.py (archived - TF-IDF was rejected as primary retrieval;
moved out of src/knowledge_system/retrieval/ during the research/production cleanup)."""

import importlib.util
from pathlib import Path

import pytest
from knowledge_system.models import DocumentChunk

_MODULE_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "_tfidf.py"
_spec = importlib.util.spec_from_file_location("tfidf_under_test", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TfidfRetriever = _module.TfidfRetriever


def make_chunk(
    text: str,
    source: str = "doc.md",
    section: str | None = None,
    page: int | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, section=section, page=page, chunk_index=chunk_index)


def test_fit_and_retrieve_relevant_content():
    chunks = [
        make_chunk("Employees can claim rail travel expenses via the portal.", chunk_index=0),
        make_chunk("Annual leave carries over subject to manager approval.", chunk_index=1),
        make_chunk("Phishing emails should be reported to the security team.", chunk_index=2),
    ]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("How do I claim rail travel expenses?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.text == chunks[0].text


def test_results_sorted_by_descending_score():
    chunks = [
        make_chunk("Rail travel rail travel rail travel expenses.", chunk_index=0),
        make_chunk("Rail travel is mentioned once here.", chunk_index=1),
        make_chunk("This chunk is about something unrelated entirely.", chunk_index=2),
    ]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("rail travel", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_result_count():
    chunks = [make_chunk(f"content about topic {i}", chunk_index=i) for i in range(5)]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("topic", top_k=2)

    assert len(results) == 2


def test_top_k_larger_than_corpus_returns_all_chunks():
    chunks = [make_chunk(f"content about topic {i}", chunk_index=i) for i in range(2)]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("topic", top_k=10)

    assert len(results) == 2


def test_metadata_survives_retrieval_unchanged():
    chunk = make_chunk("Book rail travel via the portal.", source="expenses.md", section="Rail Travel", page=None, chunk_index=7)
    retriever = TfidfRetriever()
    retriever.fit([chunk])

    result = retriever.search("rail travel", top_k=1)[0]

    assert result.chunk.source == "expenses.md"
    assert result.chunk.section == "Rail Travel"
    assert result.chunk.page is None
    assert result.chunk.chunk_index == 7
    assert result.chunk.text == "Book rail travel via the portal."


def test_section_heading_contributes_to_retrieval():
    # The body text alone shares no vocabulary with the query; only the
    # section heading does, so a hit here proves heading context is indexed.
    chunks = [
        make_chunk("Employees should follow the designated internal process.", section="Rail Travel", chunk_index=0),
        make_chunk("Employees should follow the designated internal process.", section="Office Supplies", chunk_index=1),
    ]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("rail travel", top_k=1)

    assert results[0].chunk.section == "Rail Travel"


def test_original_chunk_text_is_not_mutated():
    chunk = make_chunk("Book rail travel via the portal.", section="Rail Travel", chunk_index=0)
    original_text = chunk.text

    retriever = TfidfRetriever()
    retriever.fit([chunk])
    retriever.search("rail travel", top_k=1)

    assert chunk.text == original_text  # fit()/search() must not have rewritten it in place


def test_empty_question_raises_value_error():
    retriever = TfidfRetriever()
    retriever.fit([make_chunk("some content", chunk_index=0)])

    with pytest.raises(ValueError):
        retriever.search("   ", top_k=3)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_value_error(top_k):
    retriever = TfidfRetriever()
    retriever.fit([make_chunk("some content", chunk_index=0)])

    with pytest.raises(ValueError):
        retriever.search("some content", top_k=top_k)


def test_search_before_fit_raises():
    retriever = TfidfRetriever()

    with pytest.raises(RuntimeError):
        retriever.search("some content", top_k=3)


def test_deterministic_tie_breaking():
    # Identical text -> identical TF-IDF vectors -> identical scores for
    # every chunk. Order must still be reproducible (by chunk_index).
    chunks = [make_chunk("identical content here", chunk_index=i) for i in range(4)]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results_a = retriever.search("identical content", top_k=4)
    results_b = retriever.search("identical content", top_k=4)

    order_a = [r.chunk.chunk_index for r in results_a]
    order_b = [r.chunk.chunk_index for r in results_b]
    assert order_a == order_b == sorted(order_a)


def test_query_with_no_vocabulary_overlap_still_returns_ranked_results():
    chunks = [
        make_chunk("Rail travel expenses and booking process.", chunk_index=0),
        make_chunk("Annual leave and carry-over rules.", chunk_index=1),
    ]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    results = retriever.search("zzz qqq nonexistent gibberish", top_k=2)

    assert len(results) == 2
    assert all(r.score == 0.0 for r in results)
