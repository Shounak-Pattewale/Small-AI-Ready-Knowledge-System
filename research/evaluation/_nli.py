"""Reusable helpers for the NLI answerability experiment (evaluate_nli_answerability.py).

EXPERIMENTAL ONLY - nothing here is production code. Question-to-hypothesis
conversion is a small, fixed set of generic yes/no-question sentence
patterns - never per-question rules, never an LLM rewrite. A question that
doesn't match any pattern falls back to using the original question text
as the NLI hypothesis, recorded explicitly as a fallback rather than
silently guessed at.

Treats `premise`/`hypothesis` text strictly as data handed to the model -
never as instructions, regardless of what either contains.
"""

import math
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

# Verified from the model's own config.json via
# AutoConfig.from_pretrained(...).id2label - NOT assumed from documentation.
LABEL_ORDER = ["contradiction", "entailment", "neutral"]  # logits[0]=contradiction, [1]=entailment, [2]=neutral

_MAX_SEQUENCE_LENGTH = 512  # model config's max_position_embeddings

# Generic, sentence-SHAPE-based conversions (never keyed on specific
# question wording) to a declarative hypothesis. Conservative by design -
# each pattern is a whole-string match on a simple yes/no-question shape.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^can i (.+)\?$", re.IGNORECASE), "An employee can {rest}."),
    (re.compile(r"^can employees (.+)\?$", re.IGNORECASE), "Employees can {rest}."),
    (re.compile(r"^does the company (.+)\?$", re.IGNORECASE), "The company {rest}."),
    (re.compile(r"^will the company (.+)\?$", re.IGNORECASE), "The company will {rest}."),
    (re.compile(r"^do employees (.+)\?$", re.IGNORECASE), "Employees {rest}."),
    (re.compile(r"^will i (.+)\?$", re.IGNORECASE), "An employee will {rest}."),
    (re.compile(r"^am i (.+)\?$", re.IGNORECASE), "An employee is {rest}."),
    (re.compile(r"^is there (.+)\?$", re.IGNORECASE), "There is {rest}."),
]


@dataclass
class Hypothesis:
    text: str
    used_fallback: bool  # True if no pattern matched and the original question text was used unchanged


def question_to_hypothesis(question: str) -> Hypothesis:
    """Convert a yes/no-shaped question into a declarative NLI hypothesis using a small,
    fixed set of generic sentence-shape patterns. Falls back to the original question
    text (unchanged) when nothing matches - never invents a semantically different
    proposition."""
    stripped = question.strip()
    for pattern, template in _PATTERNS:
        match = pattern.match(stripped)
        if match:
            return Hypothesis(text=template.format(rest=match.group(1)), used_fallback=False)
    return Hypothesis(text=stripped, used_fallback=True)


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class NLIResult:
    contradiction: float
    entailment: float
    neutral: float
    predicted_label: str  # the highest-probability of the three labels


def scores_from_logits(logits: list[float]) -> NLIResult:
    """logits must be ordered [contradiction, entailment, neutral] per LABEL_ORDER."""
    probs = softmax(logits)
    predicted = LABEL_ORDER[max(range(3), key=lambda i: probs[i])]
    return NLIResult(contradiction=probs[0], entailment=probs[1], neutral=probs[2], predicted_label=predicted)


class NLIModel:
    """Loads the NLI model/tokenizer once; call score() repeatedly.

    Premise/hypothesis order: premise first, hypothesis second - this
    model's documented cross-encoder convention (verified against the
    model card's usage example).
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL_NAME) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def score(self, premise: str, hypothesis: str) -> NLIResult:
        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=_MAX_SEQUENCE_LENGTH)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return scores_from_logits(outputs.logits[0].tolist())
