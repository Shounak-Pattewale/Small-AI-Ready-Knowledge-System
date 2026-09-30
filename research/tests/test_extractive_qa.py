"""Tests for evaluation/_extractive_qa.py and the development-only loading
helper in evaluation/evaluate_extractive_qa.py.

No real Hugging Face model is downloaded here - pure-function tests feed
crafted logits directly, and the ExtractiveQAModel integration test uses a
fake tokenizer/model pair standing in for transformers.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_QA_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_extractive_qa.py"
_SCRIPT_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_extractive_qa.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qa():
    return _load_module(_QA_MODULE_PATH, "extractive_qa_under_test")


NEG_INF = float("-inf")


# --- pure logit-level functions ---


def test_null_span_score_is_cls_start_plus_end(qa):
    start_logits = [1.5, 0.1, 0.2]
    end_logits = [0.5, 0.3, 0.1]
    assert qa.null_span_score(start_logits, end_logits) == pytest.approx(2.0)


def test_best_non_null_span_finds_highest_scoring_pair(qa):
    # index 0 reserved for null; best real span should be (2, 3): 0.9 + 0.9 = 1.8
    start_logits = [0.0, 0.1, 0.9, 0.2]
    end_logits = [0.0, 0.2, 0.1, 0.9]
    score, start, end = qa.best_non_null_span(start_logits, end_logits)
    assert score == pytest.approx(1.8)
    assert (start, end) == (2, 3)


def test_best_non_null_span_respects_max_answer_length(qa):
    # Without a length cap, (1, 4) would win (0.1+0.95=1.05); with max_answer_length=2
    # only spans of length <= 2 are allowed, so (1,2) (0.1+0.5=0.6) should win
    # over both the too-long (1,4) span and the shorter (3,4) span (0.0+0.3=0.3).
    start_logits = [0.0, 0.1, 0.0, 0.0, 0.0]
    end_logits = [0.0, 0.0, 0.5, 0.0, 0.3]
    score, start, end = qa.best_non_null_span(start_logits, end_logits, max_answer_length=2)
    assert (start, end) == (1, 2)
    assert score == pytest.approx(0.6)


def test_best_non_null_span_skips_masked_positions(qa):
    # Position 1 is masked out (-inf) - the winner must come from position 2 onward.
    start_logits = [0.0, NEG_INF, 0.05]
    end_logits = [0.0, NEG_INF, 0.05]
    score, start, end = qa.best_non_null_span(start_logits, end_logits)
    assert (start, end) == (2, 2)
    assert score == pytest.approx(0.1)


def test_best_non_null_span_empty_context_returns_negative_infinity(qa):
    # Only the CLS position exists (n=1) - no valid non-null span at all.
    score, start, end = qa.best_non_null_span([0.0], [0.0])
    assert score == float("-inf")
    assert (start, end) == (0, 0)


def test_compute_qa_signal_positive_when_span_beats_null(qa):
    start_logits = [0.0, 2.0]
    end_logits = [0.0, 2.0]
    signal, best_score, null_score, start, end = qa.compute_qa_signal(start_logits, end_logits)
    assert null_score == pytest.approx(0.0)
    assert best_score == pytest.approx(4.0)
    assert signal == pytest.approx(4.0)  # strongly favors "answer exists"


def test_compute_qa_signal_negative_when_null_beats_span(qa):
    start_logits = [3.0, 0.1]
    end_logits = [3.0, 0.1]
    signal, best_score, null_score, start, end = qa.compute_qa_signal(start_logits, end_logits)
    assert signal < 0  # strongly favors "no answer"


def test_compute_qa_signal_malformed_all_masked_gives_negative_infinity_signal(qa):
    # Every non-null position masked out - the model effectively can't answer at all.
    start_logits = [1.0, NEG_INF, NEG_INF]
    end_logits = [1.0, NEG_INF, NEG_INF]
    signal, best_score, null_score, start, end = qa.compute_qa_signal(start_logits, end_logits)
    assert best_score == float("-inf")
    assert signal == float("-inf")


# --- aggregation ---


def test_aggregate_top_k_signal_single_value_top1(qa):
    assert qa.aggregate_top_k_signal([0.42]) == pytest.approx(0.42)


def test_aggregate_top_k_signal_top3_takes_maximum(qa):
    assert qa.aggregate_top_k_signal([0.1, 0.9, 0.3]) == pytest.approx(0.9)


# --- ExtractiveQAModel integration, using a fake tokenizer/model (no download) ---


class _FakeArray(list):
    def tolist(self):
        return list(self)


class _FakeEncoding(dict):
    def __init__(self, seq_ids, offsets):
        super().__init__(input_ids=_FakeArray([0] * len(seq_ids)))
        self._seq_ids = seq_ids
        self._offset_mapping = [_FakeArray(offsets)]

    def pop(self, key, *default):
        if key == "offset_mapping":
            return self._offset_mapping
        return super().pop(key, *default)

    def sequence_ids(self, batch_index=0):
        return self._seq_ids


class _FakeTokenizer:
    def __init__(self, seq_ids, offsets):
        self._seq_ids = seq_ids
        self._offsets = offsets

    @classmethod
    def from_pretrained(cls, model_name):
        raise AssertionError("from_pretrained should not be reached in this test - instance is injected directly")

    def __call__(self, question, context, **kwargs):
        return _FakeEncoding(self._seq_ids, self._offsets)


class _FakeQAOutput:
    def __init__(self, start_logits, end_logits):
        self.start_logits = [_FakeArray(start_logits)]
        self.end_logits = [_FakeArray(end_logits)]


class _FakeQAModelInner:
    def __init__(self, start_logits, end_logits):
        self._start_logits = start_logits
        self._end_logits = end_logits

    def eval(self):
        pass

    def __call__(self, **kwargs):
        return _FakeQAOutput(self._start_logits, self._end_logits)


def _make_extractive_qa_model(qa, seq_ids, offsets, start_logits, end_logits):
    """Build an ExtractiveQAModel with its tokenizer/model swapped for fakes, no download."""
    model = qa.ExtractiveQAModel.__new__(qa.ExtractiveQAModel)  # bypass __init__ (which calls from_pretrained)
    model._tokenizer = _FakeTokenizer(seq_ids, offsets)
    model._model = _FakeQAModelInner(start_logits, end_logits)
    return model


def test_extractive_qa_model_extracts_answer_from_context_tokens_only(qa):
    # Layout: [CLS] [Q1] [Q2] [SEP] [C1] [C2] [C3] [SEP]
    #  index:    0    1    2    3    4    5    6    7
    # sequence_ids: None for specials, 0 for question, 1 for context.
    seq_ids = [None, 0, 0, None, 1, 1, 1, None]
    # offsets are (char_start, char_end) into `context` for context tokens;
    # question/special tokens' offsets are irrelevant since they get masked.
    context = "five days paid leave"
    offsets = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]  # "five" "days" "paid"

    # Strongly favor a context span (indices 4-5 = "five days") over null and
    # over question-token positions (which must be masked out even though
    # their raw logits here are deliberately high, to prove masking works).
    start_logits = [0.0, 9.0, 9.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    end_logits = [0.0, 9.0, 9.0, 0.0, 0.0, 3.0, 0.0, 0.0]

    model = _make_extractive_qa_model(qa, seq_ids, offsets, start_logits, end_logits)
    result = model.answer("How many days?", context)

    assert result.answer_text == "five days"
    assert result.signal > 0  # span (3.0+3.0=6.0) beats null (0.0)


def test_extractive_qa_model_prefers_null_when_null_score_is_higher(qa):
    seq_ids = [None, 0, None, 1, 1, None]
    offsets = [(0, 0), (0, 0), (0, 0), (0, 5), (6, 10), (0, 0)]
    context = "salary raise"

    # Null (index 0) strongly outscores any context span.
    start_logits = [5.0, 0.0, 0.0, 0.1, 0.1, 0.0]
    end_logits = [5.0, 0.0, 0.0, 0.1, 0.1, 0.0]

    model = _make_extractive_qa_model(qa, seq_ids, offsets, start_logits, end_logits)
    result = model.answer("What guaranteed raise applies?", context)

    assert result.signal < 0  # null wins
    # answer_text is still populated (best non-null span found), but the
    # negative signal is what a threshold rule should act on - not the
    # mere presence of an answer_text string.
    assert result.best_span_score < result.null_score


# --- development-only loading (evaluate_extractive_qa.py's single choke point) ---


@pytest.fixture(scope="module")
def script():
    return _load_module(_SCRIPT_MODULE_PATH, "evaluate_extractive_qa_under_test")


def test_load_development_questions_excludes_heldout(script, kb_dir: Path):
    path = kb_dir / "questions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "questions": [
                    {"id": "d1", "split": "development", "answerable": True, "question": "q1"},
                    {"id": "h1", "split": "heldout", "answerable": True, "question": "q2"},
                    {"id": "d2", "split": "development", "answerable": False, "question": "q3"},
                ],
            }
        )
    )

    result = script.load_development_questions(path)

    assert [q["id"] for q in result] == ["d1", "d2"]
    assert "h1" not in [q["id"] for q in result]


def test_load_development_questions_returns_only_development_split_field(script, kb_dir: Path):
    path = kb_dir / "questions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "questions": [
                    {"id": "d1", "split": "development"},
                    {"id": "h1", "split": "heldout"},
                ],
            }
        )
    )

    result = script.load_development_questions(path)

    assert all(q["split"] == "development" for q in result)
