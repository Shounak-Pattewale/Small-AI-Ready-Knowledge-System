"""Tests for evaluation/_claim.py's deterministic QA-aware claim construction.

No model is downloaded - build_claim() is pure string logic. Confirms
generic WHO/WHAT/WHERE/WHEN/HOW/HOW MUCH/HOW MANY/WHICH/yes-no handling,
operator preservation, unusable-claim behaviour, and that no rule is
keyed on individual question IDs or corpus topics.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_claim.py"
sys.path.insert(0, str(_MODULE_PATH.parent))  # _claim.py does `import _nli as nli` as a sibling import


def _load_module():
    spec = importlib.util.spec_from_file_location("claim_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def claim():
    return _load_module()


# --- yes/no questions reuse Experiment 1's conversion, ignoring the candidate ---


def test_yesno_question_uses_direct_conversion_strategy(claim):
    result = claim.build_claim("Can I carry over unused leave?", "irrelevant candidate")
    assert result.strategy == "yesno_direct"
    assert result.usable is True
    assert result.claim == "An employee can carry over unused leave."


# --- WHO/WHERE/WHEN: append mode ---


def test_who_question_appends_candidate(claim):
    result = claim.build_claim("Who should I report a stolen laptop to?", "the Service Desk")
    assert result.strategy == "append"
    assert result.usable is True
    assert result.claim.endswith("the Service Desk.")
    assert "report a stolen laptop to" in result.claim


def test_where_question_appends_candidate(claim):
    result = claim.build_claim("Where should I report a phishing email?", "the security mailbox")
    assert result.strategy == "append"
    assert result.claim.endswith("the security mailbox.")


def test_when_question_appends_candidate(claim):
    result = claim.build_claim("When should I report a lost device?", "immediately")
    assert result.strategy == "append"
    assert result.claim.endswith("immediately.")


# --- WHAT/WHICH/HOW/HOW MUCH/HOW MANY: colon mode ---


def test_what_question_uses_colon_join(claim):
    result = claim.build_claim("What should I do if I receive a phishing message?", "report it")
    assert result.strategy == "colon"
    assert result.usable is True
    assert result.claim.endswith(": report it.")


def test_how_many_question_uses_colon_join(claim):
    result = claim.build_claim("How many days can I carry over?", "five days")
    assert result.strategy == "colon"
    assert result.claim.endswith(": five days.")


def test_how_much_question_uses_colon_join(claim):
    result = claim.build_claim("How much will the company pay toward a course?", "fees")
    assert result.strategy == "colon"
    assert result.claim.endswith(": fees.")


def test_which_question_uses_colon_join(claim):
    result = claim.build_claim("Which policy covers overseas work?", "the Overseas remote work section")
    assert result.strategy == "colon"
    assert result.claim.endswith(": the Overseas remote work section.")


def test_how_much_tried_before_bare_how(claim):
    # "how much"/"how many" must be matched before the generic "how" pattern,
    # otherwise "much"/"many" would leak into the declarativized remainder.
    result = claim.build_claim("How much notice do I need to give?", "two weeks")
    assert "much" not in result.claim.split(":")[0]


# --- semantic operator preservation (critical rule) ---


@pytest.mark.parametrize("operator", ["automatically", "guaranteed", "always", "must", "exact"])
def test_operator_words_survive_unchanged_in_who_where_when_claims(claim, operator):
    question = f"Who is {operator} responsible for lost equipment?"
    result = claim.build_claim(question, "the manager")
    assert operator in result.claim


def test_will_i_automatically_question_preserves_automatically(claim):
    # Explicit regression guard for the exact example in the task instructions:
    # must NOT collapse "Will I automatically receive X?" into "X exists."
    result = claim.build_claim("Will I automatically receive a bonus?", "irrelevant")
    assert "automatically" in result.claim
    assert result.claim == "An employee will automatically receive a bonus."


def test_how_much_question_does_not_become_generic_statement(claim):
    # Must NOT collapse "How much will the company reimburse?" into
    # "The company reimburses expenses." - the candidate must appear verbatim.
    result = claim.build_claim("How much will the company reimburse?", "up to five days' worth")
    assert "up to five days' worth" in result.claim


def test_can_i_question_does_not_become_generic_mention_statement(claim):
    # Must NOT collapse "Can I use first-class rail?" into "First-class rail is mentioned."
    result = claim.build_claim("Can I use first-class rail?", "irrelevant")
    assert result.claim == "An employee can use first-class rail."
    assert "mentioned" not in result.claim


# --- unusable claims: never forced ---


def test_unmatched_shape_is_unusable(claim):
    # Multi-sentence lead-in before the actual question - matches no shape pattern.
    result = claim.build_claim("My laptop disappeared yesterday. What should I do about it now?", "report it")
    # Note: this one DOES match "what ... ?" at the end only if the whole
    # string starts with the WH-word, which it doesn't here (starts with "My"),
    # so it must fall back to unusable.
    assert result.usable is False
    assert result.strategy == "no_shape_match"


def test_empty_qa_candidate_is_unusable(claim):
    result = claim.build_claim("Who should I contact about this?", "")
    assert result.usable is False
    assert result.strategy == "append_empty_candidate"


def test_whitespace_only_candidate_is_unusable(claim):
    result = claim.build_claim("What should I do?", "   ")
    assert result.usable is False


def test_unusable_claim_has_empty_claim_text(claim):
    result = claim.build_claim("My laptop disappeared. What now?", "report it")
    assert result.claim == ""


# --- genericity: no question-ID or topic special-casing exists in the module source ---


def test_no_hardcoded_question_ids_in_module_source(claim):
    import inspect

    source = inspect.getsource(claim)
    forbidden_substrings = ["unanswerable_0", "unanswerable_1", "direct_0", "paraphrase_0", "semantic_0", "blind_a", "blind_u", "heldout_a", "heldout_u"]
    for forbidden in forbidden_substrings:
        assert forbidden not in source, f"found question-ID-like string {forbidden!r} in _claim.py"


def test_no_corpus_topic_keywords_in_module_source(claim):
    import inspect

    source = inspect.getsource(claim).lower()
    forbidden_topics = ["laptop", "rail travel", "holiday", "mentoring", "certification", "phishing", "carry-over", "carry over"]
    for topic in forbidden_topics:
        assert topic not in source, f"found corpus-topic-specific string {topic!r} in _claim.py"


# --- determinism ---


def test_build_claim_is_deterministic(claim):
    a = claim.build_claim("Who should I report this to?", "the Service Desk")
    b = claim.build_claim("Who should I report this to?", "the Service Desk")
    assert a == b
