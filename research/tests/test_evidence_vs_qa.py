"""Tests for evaluation/evaluate_evidence_vs_qa.py's direct-evidence-vs-QA comparison.

No real Hugging Face model is loaded - fake retriever/QA components stand
in (same pattern as tests/test_service.py), proving the comparison logic
itself: same chunk fed to both branches, evidence text is verbatim
(never generated/paraphrased), metadata preserved, and nothing here
mutates chunks or embeddings.
"""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from knowledge_system.models import DocumentChunk, SearchResult
from knowledge_system.qa import QAAnswer

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_evidence_vs_qa.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evidence_vs_qa_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evq():
    return _load_module()


def make_chunk(**overrides) -> DocumentChunk:
    defaults = dict(
        text="Up to five days of unused annual leave may be carried into the next holiday year.",
        source="holiday_leave_policy.pdf",
        section="3. Carry-over",
        page=1,
        chunk_index=0,
    )
    defaults.update(overrides)
    return DocumentChunk(**defaults)


class _FakeRetriever:
    """No `fit()` method at all - if any code path tried to re-encode/mutate the index, this raises AttributeError."""

    def __init__(self, chunk: DocumentChunk, score: float = 0.9) -> None:
        self._chunk = chunk
        self._score = score
        self.search_calls: list[tuple[str, int]] = []

    def search(self, question: str, top_k: int = 1) -> list[SearchResult]:
        self.search_calls.append((question, top_k))
        return [SearchResult(chunk=self._chunk, score=self._score)]


class _FakeQA:
    def __init__(self, text: str, signal: float) -> None:
        self._result = QAAnswer(text=text, signal=signal)
        self.answer_calls: list[tuple[str, str]] = []

    def answer(self, question: str, context: str) -> QAAnswer:
        self.answer_calls.append((question, context))
        return self._result


# --- 1. QA and direct evidence receive the SAME SearchResult/chunk ---


def test_qa_and_evidence_branches_use_the_same_chunk(evq):
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)
    qa_model = _FakeQA(text="up to five days", signal=evq.QA_THRESHOLD + 1)

    result = evq.retrieve_once("Can I carry over leave?", retriever)
    qa_branch = evq.run_qa_branch("Can I carry over leave?", result.chunk.text, qa_model)
    evidence_branch = evq.run_evidence_branch(result)

    assert len(retriever.search_calls) == 1  # exactly one retrieval call
    assert qa_model.answer_calls[0][1] == evidence_branch["evidence_text"]  # QA saw the same text shown as evidence
    assert result.chunk is chunk  # both branches trace back to the one retrieved chunk


# --- 2/3. Direct evidence is exactly the stored chunk text - no generation/paraphrasing ---


def test_direct_evidence_is_exactly_the_chunk_text(evq):
    chunk = make_chunk(text="Employees may carry over up to five days into the next holiday year.")
    retriever = _FakeRetriever(chunk)
    result = evq.retrieve_once("q", retriever)

    evidence_branch = evq.run_evidence_branch(result)

    assert evidence_branch["evidence_text"] == chunk.text
    assert evidence_branch["evidence_text"] is chunk.text  # same string object, not a rebuilt/rewritten copy


# --- 4/5/6. Source/section/page metadata preserved ---


def test_evidence_branch_preserves_source_section_page(evq):
    chunk = make_chunk(source="it_support_security.md", section="3. Lost or stolen devices", page=None)
    retriever = _FakeRetriever(chunk)
    result = evq.retrieve_once("q", retriever)

    evidence_branch = evq.run_evidence_branch(result)

    assert evidence_branch["source"] == "it_support_security.md"
    assert evidence_branch["section"] == "3. Lost or stolen devices"
    assert evidence_branch["page"] is None


def test_evidence_branch_preserves_page_when_present(evq):
    chunk = make_chunk(page=7)
    retriever = _FakeRetriever(chunk)
    result = evq.retrieve_once("q", retriever)

    assert evq.run_evidence_branch(result)["page"] == 7


# --- 7. Unsupported diagnostic labels never invent an expected answer ---


@pytest.mark.parametrize("qid", ["q12", "q13", "q14"])
def test_unsupported_questions_have_no_invented_expected_source_or_topic(evq, qid):
    question_def = next(q for q in evq.DIAGNOSTIC_QUESTIONS if q["id"] == qid)
    assert question_def["expected_answerability"] == "unsupported"
    assert question_def["expected_source"] is None
    assert question_def["expected_topic"] is None


# --- 8. Complete 14-question diagnostic set is present ---


def test_diagnostic_set_has_exactly_fourteen_unique_questions(evq):
    assert len(evq.DIAGNOSTIC_QUESTIONS) == 14
    ids = [q["id"] for q in evq.DIAGNOSTIC_QUESTIONS]
    assert len(set(ids)) == 14
    texts = [q["question"] for q in evq.DIAGNOSTIC_QUESTIONS]
    assert len(set(texts)) == 14


def test_every_diagnostic_question_declares_an_expected_answerability(evq):
    allowed = {"supported", "unsupported", "no_exact_value_in_corpus"}
    for q in evq.DIAGNOSTIC_QUESTIONS:
        assert q["expected_answerability"] in allowed


# --- 9. Does not mutate chunks ---


def test_evidence_and_qa_branches_do_not_mutate_the_chunk(evq):
    chunk = make_chunk()
    chunk_before = deepcopy(chunk)
    retriever = _FakeRetriever(chunk)
    qa_model = _FakeQA(text="x", signal=0.0)

    result = evq.retrieve_once("q", retriever)
    evq.run_qa_branch("q", result.chunk.text, qa_model)
    evq.run_evidence_branch(result)

    assert chunk == chunk_before


# --- 10. Does not mutate/re-encode embeddings (no fit()/write path exists on the fake retriever) ---


def test_retrieval_only_calls_search_never_fit_or_encode(evq):
    chunk = make_chunk()
    retriever = _FakeRetriever(chunk)  # deliberately has no fit()/encode() method
    qa_model = _FakeQA(text="x", signal=0.0)

    # If retrieve_once/run_qa_branch/run_evidence_branch ever tried to
    # re-encode or rebuild the index, this fake would raise AttributeError.
    result = evq.retrieve_once("q", retriever)
    evq.run_qa_branch("q", result.chunk.text, qa_model)
    evq.run_evidence_branch(result)

    assert retriever.search_calls == [("q", 1)]


# --- QA branch structural diagnostics ---


def test_qa_branch_reports_structural_facts(evq):
    qa_model = _FakeQA(text="", signal=evq.QA_THRESHOLD - 1)
    branch = evq.run_qa_branch("q", "context", qa_model)
    assert branch["answered"] is False
    assert branch["qa_span_empty"] is True
    assert branch["qa_passed_threshold"] is False


def test_qa_branch_uses_production_threshold_constant(evq):
    from knowledge_system.service import QA_THRESHOLD as production_threshold

    assert evq.QA_THRESHOLD == production_threshold
