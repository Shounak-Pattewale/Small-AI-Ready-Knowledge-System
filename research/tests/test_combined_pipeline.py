"""Tests for evaluation/evaluate_combined_pipeline.py's QA-gate + Granite-sentence pipeline.

No real Hugging Face model is loaded - fake retriever/QA/Granite-encoder
components stand in (same pattern as tests/test_service.py and
tests/test_sentence_answers.py), proving: one retrieval call feeds every
component, the gate uses exactly QA_THRESHOLD with the whitespace-answer
safeguard, sentence selection only runs its OUTPUT is exposed after a
gate pass, and no raw QA signal/Granite similarity leaks into the final
user-facing result.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_combined_pipeline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("combined_pipeline_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_module()


class _FakeChunk:
    def __init__(self, text, source="holiday_leave_policy.pdf", section="3. Carry-over", page=1):
        self.text = text
        self.source = source
        self.section = section
        self.page = page


class _FakeResult:
    def __init__(self, chunk, score=0.9):
        self.chunk = chunk
        self.score = score


class _FakeRetriever:
    def __init__(self, chunk):
        self._chunk = chunk
        self.search_calls = []

    def search(self, question, top_k=1):
        self.search_calls.append((question, top_k))
        return [_FakeResult(self._chunk)]


class _FakeQAResult:
    def __init__(self, text, signal):
        self.text = text
        self.signal = signal


class _FakeQA:
    def __init__(self, text, signal):
        self._result = _FakeQAResult(text, signal)
        self.answer_calls = []

    def answer(self, question, context):
        self.answer_calls.append((question, context))
        return self._result


class _FakeGraniteModel:
    """Deterministic: embeds by text length, so ranking is predictable without a real model."""

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[float(len(t))] for t in texts])


# --- 1/2. One retrieval call, same chunk feeds QA and sentence selection ---


def test_retrieval_happens_exactly_once_per_question(pipeline):
    chunk = _FakeChunk("Short one. This is a considerably longer second sentence with more words in it.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="the carry-over", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    pipeline.evaluate_one("Can I carry over leave?", retriever, qa, granite)

    assert len(retriever.search_calls) == 1


def test_qa_and_sentence_selection_use_the_same_chunk_text(pipeline):
    chunk = _FakeChunk("Short one. This is a considerably longer second sentence with more words in it.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="carry-over", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert qa.answer_calls[0][1] == chunk.text
    for candidate in outcome.ranked:
        assert candidate.text in chunk.text


# --- 3. Gate uses exactly QA_THRESHOLD = -5.7906 ---


def test_gate_threshold_is_the_frozen_production_constant(pipeline):
    from knowledge_system.service import QA_THRESHOLD as production_threshold

    assert pipeline.QA_THRESHOLD == production_threshold == -5.7906


# --- 4/5. Gate pass returns sentence answer; gate fail returns answered=False ---


def test_gate_pass_returns_sentence_answer(pipeline):
    chunk = _FakeChunk("First sentence here. Second sentence follows after it.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="some span", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.gate_passed is True
    assert outcome.combined_answered is True
    assert outcome.combined_answer == outcome.ranked[0].text


def test_gate_fail_returns_answered_false(pipeline):
    chunk = _FakeChunk("First sentence here. Second sentence follows after it.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="some span", signal=pipeline.QA_THRESHOLD - 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.gate_passed is False
    assert outcome.combined_answered is False
    assert outcome.combined_answer is None


# --- 6. Gate fail does not expose the sentence as the employee-facing answer ---


def test_gate_fail_never_exposes_sentence_as_final_answer(pipeline):
    chunk = _FakeChunk("A clearly relevant complete sentence about the topic.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="x", signal=pipeline.QA_THRESHOLD - 0.01)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.combined_answer is None
    # the sentence WAS ranked (available as a diagnostic) but never promoted to combined_answer
    assert outcome.ranked  # sentences were still computed for diagnostic purposes
    assert outcome.combined_answered is False


# --- 7/8/9. Gate edge cases ---


def test_empty_qa_span_fails_gate_even_if_signal_passes_threshold(pipeline):
    chunk = _FakeChunk("A sentence.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="   ", signal=pipeline.QA_THRESHOLD + 5)  # signal clears threshold, span is blank
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.gate_passed is False


def test_signal_below_threshold_fails_gate(pipeline):
    chunk = _FakeChunk("A sentence.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="real span", signal=pipeline.QA_THRESHOLD - 0.0001)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.gate_passed is False


def test_signal_exactly_at_threshold_passes_if_span_non_empty(pipeline):
    chunk = _FakeChunk("A sentence.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="real span", signal=pipeline.QA_THRESHOLD)  # exactly at threshold, inclusive
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.gate_passed is True


# --- 10/11/12/13. Verbatim answer + metadata preservation ---


def test_sentence_answer_is_verbatim_source_text(pipeline):
    chunk = _FakeChunk("Employees may carry over up to five days into the next holiday year.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="up to five days", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.combined_answer in chunk.text


def test_source_section_page_preserved(pipeline):
    chunk = _FakeChunk("A sentence.", source="it_support_security.md", section="3. Lost or stolen devices", page=None)
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="x", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    assert outcome.result.chunk.source == "it_support_security.md"
    assert outcome.result.chunk.section == "3. Lost or stolen devices"
    assert outcome.result.chunk.page is None


# --- 14/15. No raw scores in the final user-facing result ---


def test_no_raw_qa_signal_or_similarity_in_combined_answer_field(pipeline):
    chunk = _FakeChunk("A sentence about the topic.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="x", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    outcome = pipeline.evaluate_one("q", retriever, qa, granite)

    # The "final combined result" is (combined_answered, combined_answer, source/section/page) -
    # none of these fields is a number derived from qa_result.signal or a Granite score.
    assert isinstance(outcome.combined_answered, bool)
    assert outcome.combined_answer is None or isinstance(outcome.combined_answer, str)


# --- 16. Sentence selection is computed for every question (diagnostic), but only
#         PROMOTED to the final answer after a gate pass - verified via combined_answered ---


def test_sentence_ranking_always_computed_but_only_promoted_after_gate_pass(pipeline):
    chunk = _FakeChunk("A clearly relevant complete sentence about the topic.")
    retriever = _FakeRetriever(chunk)
    granite = _FakeGraniteModel()

    failing_qa = _FakeQA(text="x", signal=pipeline.QA_THRESHOLD - 1)
    fail_outcome = pipeline.evaluate_one("q", retriever, failing_qa, granite)
    assert fail_outcome.ranked  # sentence work still happened (available as diagnostic)
    assert fail_outcome.combined_answer is None  # but never promoted

    passing_qa = _FakeQA(text="x", signal=pipeline.QA_THRESHOLD + 1)
    pass_outcome = pipeline.evaluate_one("q", retriever, passing_qa, granite)
    assert pass_outcome.combined_answer == pass_outcome.ranked[0].text


# --- 17. QA-only vs combined answerability decisions match ---


def test_qa_gate_passed_matches_manual_threshold_and_span_check(pipeline):
    import _answerability as ans

    qa_result = _FakeQAResult(text="a real span", signal=pipeline.QA_THRESHOLD + 2)
    combined_decision = pipeline.qa_gate_passed(qa_result)
    baseline_decision = ans.classify(qa_result.signal, pipeline.QA_THRESHOLD) and qa_result.text.strip() != ""
    assert combined_decision == baseline_decision is True

    qa_result_fail = _FakeQAResult(text="a real span", signal=pipeline.QA_THRESHOLD - 2)
    combined_decision_fail = pipeline.qa_gate_passed(qa_result_fail)
    baseline_decision_fail = ans.classify(qa_result_fail.signal, pipeline.QA_THRESHOLD) and qa_result_fail.text.strip() != ""
    assert combined_decision_fail == baseline_decision_fail is False


# --- 18/19/20/21. Diagnostic-set and formal-data integrity (no mutation, no relabeling) ---


def test_diagnostic_set_has_fourteen_questions(pipeline):
    assert len(pipeline.DIAGNOSTIC_QUESTIONS) == 14
    ids = [q["id"] for q in pipeline.DIAGNOSTIC_QUESTIONS]
    assert len(set(ids)) == 14


@pytest.mark.parametrize("qid", ["q12", "q13", "q14"])
def test_unsupported_diagnostics_have_no_invented_expected_answer(pipeline, qid):
    qdef = next(q for q in pipeline.DIAGNOSTIC_QUESTIONS if q["id"] == qid)
    assert qdef["expected_answerability"] == "unsupported"
    assert qdef["expected_source"] is None
    assert qdef["expected_topic"] is None


def test_formal_split_filtering_reuses_shared_helpers_without_mutation(pipeline):
    import _answerability as ans

    questions = [
        {"id": "a", "split": "development", "answerable": True},
        {"id": "b", "split": "heldout", "answerable": False},
    ]
    original = [dict(q) for q in questions]
    dev = ans.filter_by_split(questions, "development")
    heldout = ans.filter_by_split(questions, "heldout")
    assert [q["id"] for q in dev] == ["a"]
    assert [q["id"] for q in heldout] == ["b"]
    assert questions == original  # filtering never mutates the input labels


# --- 24. Deterministic repeated result ---


def test_evaluate_one_is_deterministic_across_repeated_calls(pipeline):
    chunk = _FakeChunk("First sentence here. Second sentence follows after it.")
    retriever = _FakeRetriever(chunk)
    qa = _FakeQA(text="some span", signal=pipeline.QA_THRESHOLD + 1)
    granite = _FakeGraniteModel()

    first = pipeline.evaluate_one("q", retriever, qa, granite)
    second = pipeline.evaluate_one("q", retriever, qa, granite)

    assert first.combined_answer == second.combined_answer
    assert first.gate_passed == second.gate_passed
    assert [c.text for c in first.ranked] == [c.text for c in second.ranked]
