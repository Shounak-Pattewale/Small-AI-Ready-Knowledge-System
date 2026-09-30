"""Tests for evaluation/_sentence_selection.py and evaluate_sentence_answers.py's
sentence-level semantic answer-selection experiment.

No real Hugging Face model is loaded - fake embeddings (plain numpy arrays)
stand in for Granite, proving the splitting/ranking/context logic itself:
same chunk feeds both QA and sentence selection, sentences are verbatim,
ranking is deterministic, and local context follows the fixed positional
rule (never a semantic one).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SENT_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_sentence_selection.py"
_SCRIPT_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_sentence_answers.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sents():
    return _load(_SENT_MODULE_PATH, "sentence_selection_under_test")


@pytest.fixture(scope="module")
def script():
    return _load(_SCRIPT_MODULE_PATH, "evaluate_sentence_answers_under_test")


# --- sentence splitting: verbatim, no generation/paraphrasing ---


def test_split_sentences_returns_complete_verbatim_sentences(sents):
    text = "First sentence here. Second sentence follows! Is this a third?"
    result = sents.split_sentences(text)
    assert result == ["First sentence here.", "Second sentence follows!", "Is this a third?"]


def test_split_sentences_preserves_negation_words_unchanged(sents):
    text = "There is no single fixed allowance that applies to everyone."
    result = sents.split_sentences(text)
    assert result == [text]
    assert "no" in result[0]


def test_split_sentences_handles_paragraph_breaks(sents):
    text = "First paragraph sentence.\n\nSecond paragraph sentence."
    result = sents.split_sentences(text)
    assert result == ["First paragraph sentence.", "Second paragraph sentence."]


def test_split_sentences_handles_bullet_list_paragraph(sents):
    text = "- First item\n- Second item\n- Third item"
    result = sents.split_sentences(text)
    assert result == ["- First item", "- Second item", "- Third item"]


def test_split_sentences_drops_empty_segments(sents):
    text = "One sentence.   \n\n   "
    assert sents.split_sentences(text) == ["One sentence."]


def test_split_sentences_on_heading_like_line_is_its_own_segment(sents):
    # Documented limitation: a heading with no terminal punctuation becomes
    # its own short candidate rather than merging with the next paragraph.
    text = "3.1 Becoming a mentor\n\nAny employee can volunteer."
    result = sents.split_sentences(text)
    assert result == ["3.1 Becoming a mentor", "Any employee can volunteer."]


# --- ranking: descending similarity, deterministic tie-break by original order ---


def test_rank_sentences_sorts_descending_by_similarity(sents):
    sentences = ["low", "high", "mid"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    question = np.array([0.0, 1.0])
    ranked = sents.rank_sentences(sentences, embeddings, question)
    assert [r.text for r in ranked] == ["high", "mid", "low"]


def test_rank_sentences_breaks_ties_by_original_order(sents):
    sentences = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # all tied
    question = np.array([1.0, 0.0])
    ranked = sents.rank_sentences(sentences, embeddings, question)
    assert [r.index for r in ranked] == [0, 1, 2]


def test_rank_sentences_preserves_verbatim_text_and_index(sents):
    sentences = ["Employees may claim reasonable costs.", "First-class travel requires approval."]
    embeddings = np.array([[0.1, 0.9], [0.9, 0.1]])
    question = np.array([1.0, 0.0])
    ranked = sents.rank_sentences(sentences, embeddings, question)
    assert ranked[0].text == "First-class travel requires approval."
    assert ranked[0].index == 1


# --- local context: fixed positional rule, never semantic ---


def test_local_context_prefers_following_sentence(sents):
    sentences = ["S0", "S1", "S2", "S3"]
    assert sents.local_context(sentences, top1_index=1) == ["S1", "S2"]


def test_local_context_falls_back_to_preceding_when_top1_is_last(sents):
    sentences = ["S0", "S1", "S2"]
    assert sents.local_context(sentences, top1_index=2) == ["S1", "S2"]


def test_local_context_handles_single_sentence_chunk(sents):
    sentences = ["Only sentence."]
    assert sents.local_context(sentences, top1_index=0) == ["Only sentence."]


def test_local_context_ignores_similarity_and_uses_position_only(sents):
    # Even if the "following" sentence would score terribly, the rule must
    # still pick it - local_context() never sees similarity scores at all.
    sentences = ["Top pick.", "Totally unrelated filler sentence."]
    assert sents.local_context(sentences, top1_index=0) == ["Top pick.", "Totally unrelated filler sentence."]


# --- same-chunk requirement + verbatim/no-generation guarantee (script-level) ---


def test_diagnostic_set_has_fourteen_questions(script):
    assert len(script.DIAGNOSTIC_QUESTIONS) == 14
    ids = [q["id"] for q in script.DIAGNOSTIC_QUESTIONS]
    assert len(set(ids)) == 14


@pytest.mark.parametrize("qid", ["q12", "q13", "q14"])
def test_unsupported_questions_have_no_invented_expected_metadata(script, qid):
    qdef = next(q for q in script.DIAGNOSTIC_QUESTIONS if q["id"] == qid)
    assert qdef["expected_answerability"] == "unsupported"
    assert qdef["expected_source"] is None
    assert qdef["expected_topic"] is None


def test_source_matched_is_not_applicable_when_no_expected_source(script):
    assert script.source_matched(None, "any_source.pdf") == "not-applicable"


def test_topic_matched_is_not_applicable_when_no_expected_topic(script):
    assert script.topic_matched(None, "any section") == "not-applicable"


def test_topic_matched_does_not_influence_ranking(sents):
    # Ranking (rank_sentences) takes no expected_topic/expected_source argument
    # at all - structurally impossible for expected metadata to affect it.
    import inspect

    params = inspect.signature(sents.rank_sentences).parameters
    assert "expected_topic" not in params
    assert "expected_source" not in params


# --- Top-N and edge-case sentence counts ---


def test_top_three_available_when_at_least_three_sentences(sents):
    sentences = ["a", "b", "c", "d"]
    embeddings = np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]])
    question = np.array([1.0, 0.0])
    ranked = sents.rank_sentences(sentences, embeddings, question)
    assert len(ranked) >= 3
    assert ranked[:3][0].text == "d"


def test_fewer_than_three_sentences_handled_without_error(sents):
    sentences = ["Only one sentence here."]
    embeddings = np.array([[1.0, 0.0]])
    question = np.array([1.0, 0.0])
    ranked = sents.rank_sentences(sentences, embeddings, question)
    assert len(ranked) == 1
    top_three = ranked[:3]
    assert len(top_three) == 1


def test_empty_sentence_list_ranks_to_empty_without_error(sents):
    embeddings = np.zeros((0, 2))
    question = np.array([1.0, 0.0])
    ranked = sents.rank_sentences([], embeddings, question)
    assert ranked == []


# --- process_question(): same-chunk requirement + metadata preservation, via fakes ---


class _FakeChunk:
    def __init__(self, text, source, section, page):
        self.text = text
        self.source = source
        self.section = section
        self.page = page


class _FakeResult:
    def __init__(self, chunk, score=0.9):
        self.chunk = chunk
        self.score = score


class _FakeRetriever:
    def __init__(self, result):
        self._result = result
        self.search_calls = []

    def search(self, question, top_k=1):
        self.search_calls.append((question, top_k))
        return [self._result]


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
    """Deterministic fake standing in for the reused Granite SentenceTransformer."""

    def encode(self, texts, normalize_embeddings=True):
        # 1-D embedding: length of text (arbitrary but deterministic) - enough to rank distinctly.
        return np.array([[float(len(t))] for t in texts])


def test_process_question_qa_and_sentences_come_from_the_same_chunk(script):
    chunk = _FakeChunk(
        text="Short sentence. This is a considerably longer second sentence with more words.",
        source="holiday_leave_policy.pdf", section="3. Carry-over", page=1,
    )
    retriever = _FakeRetriever(_FakeResult(chunk))
    qa_model = _FakeQA(text="carried into the next holiday year", signal=1.0)
    granite = _FakeGraniteModel()
    qdef = {"question": "Can I carry over leave?", "expected_source": "holiday_leave_policy.pdf",
            "expected_topic": "Carry-over", "expected_answerability": "supported"}

    outcome = script.process_question(qdef, retriever, qa_model, granite)

    assert len(retriever.search_calls) == 1  # exactly one retrieval call
    assert qa_model.answer_calls[0][1] == chunk.text  # QA saw the exact retrieved chunk text
    # every sentence candidate must be a substring of the SAME chunk text QA received
    for candidate in outcome["ranked"]:
        assert candidate.text in chunk.text
    assert outcome["result"].chunk is chunk


def test_process_question_preserves_source_section_page(script):
    chunk = _FakeChunk(text="One sentence here.", source="it_support_security.md", section="3. Lost or stolen devices", page=None)
    retriever = _FakeRetriever(_FakeResult(chunk))
    qa_model = _FakeQA(text="the police", signal=1.0)
    granite = _FakeGraniteModel()
    qdef = {"question": "q", "expected_source": None, "expected_topic": None, "expected_answerability": "supported"}

    outcome = script.process_question(qdef, retriever, qa_model, granite)

    assert outcome["result"].chunk.source == "it_support_security.md"
    assert outcome["result"].chunk.section == "3. Lost or stolen devices"
    assert outcome["result"].chunk.page is None


def test_process_question_produces_no_text_outside_the_chunk(script):
    chunk = _FakeChunk(text="First sentence. Second sentence.", source="s.md", section="sec", page=2)
    retriever = _FakeRetriever(_FakeResult(chunk))
    qa_model = _FakeQA(text="Second sentence", signal=1.0)
    granite = _FakeGraniteModel()
    qdef = {"question": "q", "expected_source": None, "expected_topic": None, "expected_answerability": "supported"}

    outcome = script.process_question(qdef, retriever, qa_model, granite)

    for sentence in outcome["sentences"]:
        assert sentence in chunk.text
    assert " ".join(outcome["context_sentences"]) in chunk.text or all(c in chunk.text for c in outcome["context_sentences"])
