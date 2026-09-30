"""Tests for evaluation/_nli.py's deterministic logic: question-to-hypothesis
conversion, label-mapping/softmax handling, decision-rule application. No real
NLI model is downloaded - scores_from_logits is pure, and NLIModel is
exercised via a fake tokenizer/model pair.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_nli.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("nli_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nli():
    return _load_module()


# --- question-to-hypothesis conversion ---


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Can I carry over unused annual leave?", "An employee can carry over unused annual leave."),
        ("Can employees claim rail travel costs?", "Employees can claim rail travel costs."),
        ("Does the company support professional certifications?", "The company support professional certifications."),
        ("Will the company reimburse first-class travel?", "The company will reimburse first-class travel."),
        ("Do employees get a laptop?", "Employees get a laptop."),
        ("Will I automatically be promoted?", "An employee will automatically be promoted."),
        ("Am I allowed to work from home?", "An employee is allowed to work from home."),
        ("Is there a mentoring scheme?", "There is a mentoring scheme."),
    ],
)
def test_question_to_hypothesis_matches_generic_patterns(nli, question, expected):
    result = nli.question_to_hypothesis(question)
    assert result.text == expected
    assert result.used_fallback is False


def test_question_to_hypothesis_is_case_insensitive(nli):
    result = nli.question_to_hypothesis("CAN I work remotely?")
    assert result.used_fallback is False
    assert result.text == "An employee can work remotely."


@pytest.mark.parametrize(
    "question",
    [
        "What should I do if my laptop is lost or stolen?",
        "How do I submit an expense claim?",
        "My work laptop has disappeared. What should I do?",
        "If standard-class tickets are sold out, will the company reimburse first-class rail travel?",
    ],
)
def test_question_to_hypothesis_falls_back_when_no_pattern_matches(nli, question):
    result = nli.question_to_hypothesis(question)
    assert result.used_fallback is True
    assert result.text == question.strip()  # original question preserved verbatim, never guessed


def test_fallback_never_mutates_original_question_text(nli):
    question = "  What is the policy for rail travel?  "
    result = nli.question_to_hypothesis(question)
    assert result.text == question.strip()


# --- label mapping / softmax ---


def test_label_order_matches_verified_model_config(nli):
    # Verified via AutoConfig.from_pretrained("cross-encoder/nli-deberta-v3-xsmall").id2label
    # == {0: 'contradiction', 1: 'entailment', 2: 'neutral'} - not assumed from docs.
    assert nli.LABEL_ORDER == ["contradiction", "entailment", "neutral"]


def test_softmax_sums_to_one(nli):
    probs = nli.softmax([1.0, 2.0, 0.5])
    assert sum(probs) == pytest.approx(1.0)


def test_softmax_is_numerically_stable_for_large_logits(nli):
    probs = nli.softmax([1000.0, 1001.0, 999.0])
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_scores_from_logits_picks_highest_probability_as_predicted_label(nli):
    # logits ordered [contradiction, entailment, neutral]
    result = nli.scores_from_logits([0.0, 5.0, 0.0])
    assert result.predicted_label == "entailment"
    assert result.entailment > result.contradiction
    assert result.entailment > result.neutral


def test_scores_from_logits_contradiction_wins(nli):
    result = nli.scores_from_logits([5.0, 0.0, 0.0])
    assert result.predicted_label == "contradiction"


def test_scores_from_logits_probabilities_sum_to_one(nli):
    result = nli.scores_from_logits([1.0, 2.0, 3.0])
    assert result.contradiction + result.entailment + result.neutral == pytest.approx(1.0)


# --- NLIModel integration via fake tokenizer/model (no download) ---


class _FakeArray(list):
    def tolist(self):
        return list(self)


class _FakeTokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        self.last_call = (premise, hypothesis)
        return {"input_ids": _FakeArray([0])}


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = [_FakeArray(logits)]


class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def eval(self):
        pass  # PyTorch nn.Module.eval() - inference mode switch, not the eval() builtin

    def __call__(self, **kwargs):
        return _FakeOutputs(self._logits)


def _make_nli(nli, logits) -> "nli.NLIModel":
    model = nli.NLIModel.__new__(nli.NLIModel)  # bypass __init__ (which calls from_pretrained)
    model._tokenizer = _FakeTokenizer()
    model._model = _FakeModel(logits)
    return model


def test_nli_model_score_uses_premise_then_hypothesis_order(nli):
    model = _make_nli(nli, [0.0, 3.0, 0.0])
    model.score("policy text here", "An employee can do X.")
    assert model._tokenizer.last_call == ("policy text here", "An employee can do X.")


def test_nli_model_score_returns_expected_result(nli):
    model = _make_nli(nli, [0.0, 3.0, 0.0])
    result = model.score("premise", "hypothesis")
    assert result.predicted_label == "entailment"


# --- decision-rule sanity (reusing _answerability's generic threshold machinery) ---


def test_decision_rule_via_classify_is_deterministic(nli):
    import importlib.util

    ans_spec = importlib.util.spec_from_file_location(
        "answerability_for_nli_test", Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_answerability.py"
    )
    ans = importlib.util.module_from_spec(ans_spec)
    ans_spec.loader.exec_module(ans)

    assert ans.classify(0.8, threshold=0.5) is True
    assert ans.classify(0.8, threshold=0.5) == ans.classify(0.8, threshold=0.5)
