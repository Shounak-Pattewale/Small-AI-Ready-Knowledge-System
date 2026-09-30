"""Tests for src/knowledge_system/service.py's LLMKnowledgeService.

No real model or network call - fake retriever/LLM-client components are
injected via the constructor, same pattern as tests/test_service.py's
KnowledgeService tests.

Top-3 evidence-block contract (promoted from the Top-3 research
experiment, docs/EVALUATION_HISTORY.md Section 29): the service now
retrieves Top-3 in one call and maps `GroundedLLMResult.evidence_id`
(1/2/3) back to the corresponding SearchResult for provenance.
"""

import pytest
from knowledge_system.llm import GroundedLLMResult, LLMProviderError
from knowledge_system.models import DocumentChunk, SearchResult
from knowledge_system.service import LLMKnowledgeService


class _FakeRetriever:
    """Returns up to 3 distinct SearchResults (one per chunk given), in the given order."""

    def __init__(self, chunks: list[DocumentChunk], scores: list[float] | None = None) -> None:
        self._chunks = chunks
        self._scores = scores or [0.9 - 0.01 * i for i in range(len(chunks))]
        self.search_calls: list[tuple[str, int]] = []

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        self.search_calls.append((question, top_k))
        return [SearchResult(chunk=c, score=s) for c, s in zip(self._chunks, self._scores)][:top_k]


class _FakeLLMClient:
    def __init__(self, result: GroundedLLMResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.answer_calls: list[tuple[str, list[str]]] = []

    def answer(self, question: str, evidences: list[str]) -> GroundedLLMResult:
        self.answer_calls.append((question, evidences))
        if self._exc is not None:
            raise self._exc
        return self._result


def make_chunk(**overrides) -> DocumentChunk:
    defaults = dict(text="Employees may carry over up to five days.", source="holiday_leave_policy.pdf", section="4. Carry-over", page=3, chunk_index=0)
    defaults.update(overrides)
    return DocumentChunk(**defaults)


def make_three_chunks() -> list[DocumentChunk]:
    return [
        make_chunk(text="First chunk text.", source="doc_a.md", section="Section A", page=1, chunk_index=0),
        make_chunk(text="Second chunk text.", source="doc_b.pdf", section="Section B", page=2, chunk_index=1),
        make_chunk(text="Third chunk text.", source="doc_c.md", section="Section C", page=None, chunk_index=2),
    ]


# --- input validation ---


def test_empty_question_rejected():
    service = LLMKnowledgeService(_FakeRetriever(make_three_chunks()), _FakeLLMClient(GroundedLLMResult(True, "x", 1)))
    with pytest.raises(ValueError):
        service.ask("")


def test_whitespace_question_rejected():
    service = LLMKnowledgeService(_FakeRetriever(make_three_chunks()), _FakeLLMClient(GroundedLLMResult(True, "x", 1)))
    with pytest.raises(ValueError):
        service.ask("   \n\t  ")


# --- retrieval: Top-3, exactly once, exact chunk texts supplied ---


def test_service_requests_top3_from_retriever():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 1))
    service = LLMKnowledgeService(retriever, client)

    service.ask("a question")

    assert retriever.search_calls == [("a question", 3)]


def test_retrieval_called_once_per_question():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 1))
    service = LLMKnowledgeService(retriever, client)

    service.ask("question one")
    service.ask("question two")

    assert len(retriever.search_calls) == 2


def test_exact_retrieved_chunk_texts_supplied_to_llm_in_order():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 1))
    service = LLMKnowledgeService(retriever, client)

    service.ask("my question")

    assert client.answer_calls == [("my question", ["First chunk text.", "Second chunk text.", "Third chunk text."])]


# --- evidence_id -> SearchResult mapping ---


def test_evidence_id_1_maps_to_first_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 1))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.source == "doc_a.md"
    assert answer.section == "Section A"


def test_evidence_id_2_maps_to_second_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 2))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.source == "doc_b.pdf"
    assert answer.section == "Section B"


def test_evidence_id_3_maps_to_third_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 3))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.source == "doc_c.md"
    assert answer.section == "Section C"


# --- answerable=true -> answered=True, provenance from the SELECTED SearchResult ---


def test_answerable_result_produces_answered_true():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(True, "up to five days", 1))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("Can I carry over leave?")

    assert answer.answered is True


def test_answer_text_comes_from_llm_result():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(True, "a distinctive grounded answer", 2))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.answer == "a distinctive grounded answer"


def test_source_comes_from_selected_search_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "some answer", 2))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.source == "doc_b.pdf"


def test_section_comes_from_selected_search_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "some answer", 3))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.section == "Section C"


def test_page_comes_from_selected_search_result():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "some answer", 2))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.page == 2


def test_page_none_when_selected_result_has_no_page():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "some answer", 3))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.page is None


def test_llm_cannot_override_provenance():
    # GroundedLLMResult has no source/section/page fields at all - structurally
    # impossible for the LLM's output to supply provenance. This test guards
    # against someone adding such fields later and wiring them in by mistake.
    result = GroundedLLMResult(True, "x", 1)
    assert not hasattr(result, "source")
    assert not hasattr(result, "section")
    assert not hasattr(result, "page")


def test_no_similarity_or_confidence_in_knowledge_answer():
    chunks = make_three_chunks()
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "some answer", 1))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    for forbidden in ("similarity", "confidence", "score", "evidence_id"):
        assert not hasattr(answer, forbidden)


# --- answerable=false -> answered=False, no provenance exposed ---


def test_abstention_produces_answered_false():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(False, None, None))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.answered is False
    assert answer.answer is None


def test_abstention_exposes_no_provenance():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(GroundedLLMResult(False, None, None))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert answer.source is None
    assert answer.section is None
    assert answer.page is None


# --- provider failure propagates, never disguised as abstention ---


def test_provider_failure_propagates_not_disguised_as_abstention():
    retriever = _FakeRetriever(make_three_chunks())
    client = _FakeLLMClient(exc=LLMProviderError("Ollama Cloud unavailable"))
    service = LLMKnowledgeService(retriever, client)

    with pytest.raises(LLMProviderError):
        service.ask("q")


# --- edge case: retriever returns fewer than 3 results ---


def test_fewer_than_three_results_handled_safely():
    chunks = make_three_chunks()[:2]  # retriever can only return 2
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 2))
    service = LLMKnowledgeService(retriever, client)

    answer = service.ask("q")

    assert client.answer_calls[0][1] == ["First chunk text.", "Second chunk text."]
    assert answer.source == "doc_b.pdf"


def test_evidence_id_for_nonexistent_block_raises_indexerror_not_silently_wrong_provenance():
    # If the client ever returned an evidence_id beyond what was actually retrieved (it shouldn't -
    # OllamaCloudClient validates this itself - see tests/test_llm.py), the service must fail loudly
    # rather than silently return wrong/mismatched provenance.
    chunks = make_three_chunks()[:2]
    retriever = _FakeRetriever(chunks)
    client = _FakeLLMClient(GroundedLLMResult(True, "answer", 3))  # out of range for only 2 results
    service = LLMKnowledgeService(retriever, client)

    with pytest.raises(IndexError):
        service.ask("q")
