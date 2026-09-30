"""Tests for src/knowledge_system/service.py (production KnowledgeService).

No real model is loaded - fake retriever/QA components are injected via
the constructor, proving KnowledgeService's own logic (threshold
application, provenance, validation, call wiring) without needing Granite
or MiniLM.
"""

import pytest
from knowledge_system.models import DocumentChunk, SearchResult
from knowledge_system.qa import QAAnswer
from knowledge_system.service import QA_THRESHOLD, KnowledgeService


class _FakeRetriever:
    """Records every search() call; always returns the single chunk it was built with."""

    def __init__(self, chunk: DocumentChunk, score: float = 0.5) -> None:
        self._chunk = chunk
        self._score = score
        self.search_calls: list[tuple[str, int]] = []

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        self.search_calls.append((question, top_k))
        return [SearchResult(chunk=self._chunk, score=self._score)]


class _FakeQA:
    """Records every answer() call; always returns the fixed QAAnswer it was built with."""

    def __init__(self, text: str, signal: float) -> None:
        self._result = QAAnswer(text=text, signal=signal)
        self.answer_calls: list[tuple[str, str]] = []

    def answer(self, question: str, context: str) -> QAAnswer:
        self.answer_calls.append((question, context))
        return self._result


def make_chunk(**overrides) -> DocumentChunk:
    defaults = dict(text="Employees may carry over up to five days.", source="holiday_leave_policy.pdf", section="4. Carry-over", page=3, chunk_index=0)
    defaults.update(overrides)
    return DocumentChunk(**defaults)


def test_valid_answered_question():
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="up to five days", signal=QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("How many days can I carry over?")

    assert answer.answered is True
    assert answer.answer == "up to five days"


def test_insufficient_information_result():
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="some extracted text", signal=QA_THRESHOLD - 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("What guaranteed bonus do I get?")

    assert answer.answered is False
    assert answer.answer is None


def test_exact_threshold_boundary_is_answered():
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="boundary answer", signal=QA_THRESHOLD)  # exactly at the frozen threshold
    service = KnowledgeService(retriever, qa)

    answer = service.ask("boundary question")

    assert answer.answered is True  # >= is inclusive


def test_just_below_threshold_is_insufficient():
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="almost", signal=QA_THRESHOLD - 0.0001)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("almost question")

    assert answer.answered is False


def test_source_section_page_preserved_when_answered():
    chunk = make_chunk(source="flexible_remote_working.pdf", section="8. Working from another UK location", page=5)
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="short periods may be possible", signal=QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("Can I work from another UK city?")

    assert answer.source == "flexible_remote_working.pdf"
    assert answer.section == "8. Working from another UK location"
    assert answer.page == 5


def test_provenance_is_none_when_insufficient():
    # The retrieved chunk has real source/section/page, but since the QA
    # signal didn't clear the threshold, none of it should be surfaced -
    # citing it would misleadingly imply it answered the question.
    chunk = make_chunk(source="employee_handbook.pdf", section="9. Benefits", page=10)
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="loosely related span", signal=QA_THRESHOLD - 5)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("What gym membership is covered?")

    assert answer.source is None
    assert answer.section is None
    assert answer.page is None


def test_empty_question_rejected():
    service = KnowledgeService(_FakeRetriever(make_chunk()), _FakeQA("x", 0.0))
    with pytest.raises(ValueError):
        service.ask("")


def test_whitespace_question_rejected():
    service = KnowledgeService(_FakeRetriever(make_chunk()), _FakeQA("x", 0.0))
    with pytest.raises(ValueError):
        service.ask("   \n\t  ")


def test_retriever_called_once_per_valid_question():
    retriever = _FakeRetriever(make_chunk())
    qa = _FakeQA("x", QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    service.ask("question one")
    service.ask("question two")

    assert len(retriever.search_calls) == 2


def test_qa_called_with_question_and_top1_chunk_text():
    chunk = make_chunk(text="Specific chunk body text.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA("x", QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    service.ask("my question")

    assert qa.answer_calls == [("my question", "Specific chunk body text.")]


def test_only_top1_retrieval_used():
    retriever = _FakeRetriever(make_chunk())
    qa = _FakeQA("x", QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    service.ask("a question")

    assert retriever.search_calls == [("a question", 1)]


def test_whitespace_only_answer_above_threshold_is_treated_as_insufficient():
    # Deterministic safeguard: a signal clearing the threshold but an
    # extracted span that's only whitespace isn't a real answer - must not
    # be surfaced as answered=True with a blank answer.
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="   ", signal=QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("some question")

    assert answer.answered is False
    assert answer.answer is None
    assert answer.source is None


def test_empty_string_answer_above_threshold_is_treated_as_insufficient():
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="", signal=QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("some question")

    assert answer.answered is False


def test_granite_similarity_not_used_for_answerability():
    # Deliberately low retrieval similarity but a QA signal above threshold
    # - the answer must still be accepted, proving similarity plays no
    # role in the answered/insufficient decision.
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk, score=0.01)
    qa = _FakeQA(text="answer despite low similarity", signal=QA_THRESHOLD + 1)
    service = KnowledgeService(retriever, qa)

    answer = service.ask("question")

    assert answer.answered is True
    assert answer.answer == "answer despite low similarity"
