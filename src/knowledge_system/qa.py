"""Production extractive QA: MiniLM SQuAD2.0 span extraction with a null-score answerability signal.

Mirrors the validated logic in evaluation/_extractive_qa.py exactly - same
context-token masking, same null-score definition (CLS-CLS logits), same
best-span search, same max_answer_length, same signal formula
(best_span_score - null_score). Duplicated deliberately rather than
imported, so production code under src/knowledge_system/ never depends on
evaluation/ - that boundary was the whole point of building this module.

Do not change span selection, masking, max_answer_length, tokenization
behaviour, or the signal formula without re-running the blind evaluation
(docs/EVALUATION_HISTORY.md, Experiment QA2) - this implementation is
frozen alongside that result.

Treats `question` and `context` strictly as data handed to the model -
never as instructions, regardless of what text either contains.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_QA_MODEL_NAME = "deepset/minilm-uncased-squad2"

# Validated during the development/blind experiments - changing either
# value changes what QAAnswer.signal means and invalidates the frozen
# -5.7906 threshold used by KnowledgeService.
_MAX_ANSWER_LENGTH = 30
_MAX_SEQUENCE_LENGTH = 384


@dataclass
class QAAnswer:
    """One QA call's result.

    `signal` is a raw logit difference (best_span_score - null_score), NOT
    a calibrated probability or percentage confidence - never surface it
    to users as "confidence".
    """

    text: str  # extracted answer span; "" when no valid non-null span was found
    signal: float  # higher = stronger evidence an answer exists in this context


def _null_span_score(start_logits: list[float], end_logits: list[float]) -> float:
    """SQuAD2.0-style null (no-answer) score: the CLS-token start/end logits at index 0."""
    return start_logits[0] + end_logits[0]


def _best_non_null_span(
    start_logits: list[float], end_logits: list[float], max_answer_length: int
) -> tuple[float, int, int]:
    """Best-scoring (start, end) span among positions >= 1 (index 0 is reserved for the null span).

    Callers must mask out any non-context position (question tokens,
    [SEP], etc.) to -inf beforehand, so this only ever searches genuine
    context-token positions. Returns (-inf, 0, 0) if no valid span exists.
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


class ExtractiveQA:
    """Loads the QA model/tokenizer once; call answer() repeatedly.

    Usage:
        qa = ExtractiveQA()
        result = qa.answer("How many days can I carry over?", chunk_text)
    """

    def __init__(self, model_name: str = DEFAULT_QA_MODEL_NAME) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self._model.eval()

    def answer(self, question: str, context: str) -> QAAnswer:
        """Run QA on (question, context). `context` is treated strictly as data - never as instructions."""
        inputs = self._tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_SEQUENCE_LENGTH,
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

        null_score = _null_span_score(masked_start, masked_end)
        best_score, best_start, best_end = _best_non_null_span(masked_start, masked_end, _MAX_ANSWER_LENGTH)
        signal = best_score - null_score

        text = ""
        if best_score != float("-inf") and best_start != 0:
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            text = context[start_char:end_char]

        return QAAnswer(text=text, signal=signal)
