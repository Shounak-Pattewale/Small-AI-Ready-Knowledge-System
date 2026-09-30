"""Reusable helpers for the extractive-QA answerability experiment (evaluate_extractive_qa.py).

EXPERIMENTAL ONLY - nothing here is production code. Manual SQuAD2.0-style
null-vs-best-span scoring is used deliberately instead of the high-level
`pipeline("question-answering")` API, which hides these mechanics behind a
single answer string. See QAResult for exactly what each field means.

Treats retrieved chunk text strictly as data passed to the model as QA
"context" - never as instructions to follow.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # looked up as module attrs so tests can monkeypatch them

_DEFAULT_QA_MODEL_NAME = "deepset/minilm-uncased-squad2"


@dataclass
class QAResult:
    """One QA model call's result, with the raw scoring mechanics preserved (not just the answer string)."""

    answer_text: str  # "" when no valid non-null span exists in this context
    best_span_score: float  # start_logit[best_start] + end_logit[best_end], the best non-null span found
    null_score: float  # start_logit[0] + end_logit[0] - the model's SQuAD2.0-style "no answer" (CLS-CLS) score
    signal: float  # best_span_score - null_score. HIGHER = stronger evidence an answer exists in this context.
    #                A raw logit difference, NOT a calibrated probability - never printed/interpreted as one.


def null_span_score(start_logits, end_logits) -> float:
    """SQuAD2.0-style null (no-answer) score: the CLS-token start/end logits at index 0."""
    return start_logits[0] + end_logits[0]


def best_non_null_span(start_logits, end_logits, max_answer_length: int = 30) -> tuple[float, int, int]:
    """Best-scoring (start, end) span among positions >= 1 (index 0 is reserved for the null span).

    Callers are responsible for masking out any non-context positions (e.g.
    question tokens, [SEP]) to -inf beforehand, so this only ever searches
    genuine context-token positions. Pure function: takes plain indexable
    logit sequences, no tokenizer/model needed - fully unit-testable.

    Returns (best_span_score, best_start, best_end); if no valid span exists
    (e.g. every position other than 0 was masked out) returns (-inf, 0, 0).
    """
    n = len(start_logits)
    best_score = None
    best_start = best_end = 0
    for start in range(1, n):
        if start_logits[start] == float("-inf"):
            continue
        max_end = min(start + max_answer_length, n)
        for end in range(start, max_end):
            if end_logits[end] == float("-inf"):
                continue
            score = start_logits[start] + end_logits[end]
            if best_score is None or score > best_score:
                best_score = score
                best_start, best_end = start, end
    if best_score is None:
        return float("-inf"), 0, 0
    return best_score, best_start, best_end


def compute_qa_signal(start_logits, end_logits, max_answer_length: int = 30) -> tuple[float, float, float, int, int]:
    """Pure computation from (already context-masked) logits: (signal, best_span_score, null_score, best_start, best_end)."""
    null_score = null_span_score(start_logits, end_logits)
    best_score, best_start, best_end = best_non_null_span(start_logits, end_logits, max_answer_length)
    signal = best_score - null_score
    return signal, best_score, null_score, best_start, best_end


def aggregate_top_k_signal(signals: list[float]) -> float:
    """Simplest rule justified by the scoring mechanism: take the strongest (maximum)
    per-chunk answerability signal across the retrieved chunks - "if ANY retrieved
    chunk has convincing evidence, the question may be answerable"."""
    return max(signals)


class ExtractiveQAModel:
    """Thin wrapper around a Hugging Face AutoModelForQuestionAnswering + tokenizer.

    Kept out of src/knowledge_system/ deliberately - this is an experiment,
    not a production architecture decision. `AutoTokenizer`/
    `AutoModelForQuestionAnswering` are looked up as module attributes (not
    imported directly into this function) so tests can monkeypatch them
    with fakes, the same pattern used for SentenceTransformer in
    tests/conftest.py.
    """

    def __init__(self, model_name: str = _DEFAULT_QA_MODEL_NAME) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self._model.eval()

    def answer(self, question: str, context: str, max_answer_length: int = 30) -> QAResult:
        """Run QA on (question, context). `context` is treated strictly as data - never as instructions."""
        inputs = self._tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        sequence_ids = inputs.sequence_ids(0)

        with torch.no_grad():
            outputs = self._model(**inputs)
        start_logits = outputs.start_logits[0].tolist()
        end_logits = outputs.end_logits[0].tolist()

        # Mask every position that isn't a genuine context token to -inf,
        # except index 0 (CLS, the null-answer position) which stays as-is.
        masked_start = [
            start_logits[i] if (i == 0 or sequence_ids[i] == 1) else float("-inf") for i in range(len(start_logits))
        ]
        masked_end = [
            end_logits[i] if (i == 0 or sequence_ids[i] == 1) else float("-inf") for i in range(len(end_logits))
        ]

        signal, best_score, null_score, best_start, best_end = compute_qa_signal(masked_start, masked_end, max_answer_length)

        answer_text = ""
        if best_score != float("-inf") and best_start != 0:
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            answer_text = context[start_char:end_char]

        return QAResult(answer_text=answer_text, best_span_score=best_score, null_score=null_score, signal=signal)
