"""Tests for src/knowledge_system/qa.py (production extractive QA component).

No real model is downloaded - a fake tokenizer/model pair stands in for
transformers, same pattern as tests/test_extractive_qa.py (the equivalent
experimental-code tests), proving the production port behaves identically.
"""

import pytest
from knowledge_system.qa import ExtractiveQA, _best_non_null_span, _null_span_score

NEG_INF = float("-inf")


def test_null_span_score_is_cls_start_plus_end():
    assert _null_span_score([1.5, 0.1], [0.5, 0.3]) == pytest.approx(2.0)


def test_best_non_null_span_finds_highest_scoring_pair():
    start_logits = [0.0, 0.1, 0.9, 0.2]
    end_logits = [0.0, 0.2, 0.1, 0.9]
    score, start, end = _best_non_null_span(start_logits, end_logits, max_answer_length=30)
    assert score == pytest.approx(1.8)
    assert (start, end) == (2, 3)


def test_best_non_null_span_respects_max_answer_length():
    start_logits = [0.0, 0.1, 0.0, 0.0, 0.0]
    end_logits = [0.0, 0.0, 0.5, 0.0, 0.3]
    score, start, end = _best_non_null_span(start_logits, end_logits, max_answer_length=2)
    assert (start, end) == (1, 2)
    assert score == pytest.approx(0.6)


def test_best_non_null_span_skips_masked_positions():
    start_logits = [0.0, NEG_INF, 0.05]
    end_logits = [0.0, NEG_INF, 0.05]
    score, start, end = _best_non_null_span(start_logits, end_logits, max_answer_length=30)
    assert (start, end) == (2, 2)


def test_best_non_null_span_empty_context_returns_negative_infinity():
    score, start, end = _best_non_null_span([0.0], [0.0], max_answer_length=30)
    assert score == float("-inf")
    assert (start, end) == (0, 0)


# --- ExtractiveQA integration, using a fake tokenizer/model (no download) ---


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

    def __call__(self, question, context, **kwargs):
        return _FakeEncoding(self._seq_ids, self._offsets)


class _FakeQAOutput:
    def __init__(self, start_logits, end_logits):
        self.start_logits = [_FakeArray(start_logits)]
        self.end_logits = [_FakeArray(end_logits)]


class _FakeModel:
    def __init__(self, start_logits, end_logits):
        self._start_logits = start_logits
        self._end_logits = end_logits

    def eval(self):
        pass  # PyTorch nn.Module.eval() - switches to inference mode, not the eval() builtin

    def __call__(self, **kwargs):
        return _FakeQAOutput(self._start_logits, self._end_logits)


def _make_qa(seq_ids, offsets, start_logits, end_logits) -> ExtractiveQA:
    qa = ExtractiveQA.__new__(ExtractiveQA)  # bypass __init__ (which calls from_pretrained)
    qa._tokenizer = _FakeTokenizer(seq_ids, offsets)
    qa._model = _FakeModel(start_logits, end_logits)
    return qa


def test_extractive_qa_extracts_answer_from_context_tokens_only():
    # [CLS] [Q1] [Q2] [SEP] [C1] [C2] [C3] [SEP]
    seq_ids = [None, 0, 0, None, 1, 1, 1, None]
    context = "five days paid leave"
    offsets = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]

    # Question-token logits are deliberately huge to prove masking excludes them.
    start_logits = [0.0, 9.0, 9.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    end_logits = [0.0, 9.0, 9.0, 0.0, 0.0, 3.0, 0.0, 0.0]

    qa = _make_qa(seq_ids, offsets, start_logits, end_logits)
    result = qa.answer("How many days?", context)

    assert result.text == "five days"
    assert result.signal > 0


def test_extractive_qa_prefers_null_when_null_score_is_higher():
    seq_ids = [None, 0, None, 1, 1, None]
    offsets = [(0, 0), (0, 0), (0, 0), (0, 5), (6, 10), (0, 0)]
    context = "salary raise"

    start_logits = [5.0, 0.0, 0.0, 0.1, 0.1, 0.0]
    end_logits = [5.0, 0.0, 0.0, 0.1, 0.1, 0.0]

    qa = _make_qa(seq_ids, offsets, start_logits, end_logits)
    result = qa.answer("What guaranteed raise applies?", context)

    assert result.signal < 0
    assert result.signal == pytest.approx((0.1 + 0.1) - (5.0 + 5.0))  # best_span_score - null_score, no other adjustment


def test_extractive_qa_empty_context_gives_no_answer_and_negative_signal():
    # Only [CLS] [Q1] [SEP] - no context tokens at all.
    seq_ids = [None, 0, None]
    offsets = [(0, 0), (0, 0), (0, 0)]
    start_logits = [1.0, 0.5, 0.5]
    end_logits = [1.0, 0.5, 0.5]

    qa = _make_qa(seq_ids, offsets, start_logits, end_logits)
    result = qa.answer("Anything?", "")

    assert result.text == ""
    assert result.signal == float("-inf")
