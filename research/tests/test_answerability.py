"""Tests for evaluation/_answerability.py's pure statistics/threshold helpers.

Loaded via importlib since evaluation/ is a script directory, not a package
(same pattern as test_evaluate_no_docling.py). No model, no retrieval.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_answerability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("answerability_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


def test_score_gap(m):
    assert m.score_gap(0.9, 0.6) == pytest.approx(0.3)
    assert m.score_gap(0.5, 0.8) == pytest.approx(-0.3)  # negative gap allowed, caller decides meaning


def test_summarize(m):
    result = m.summarize([1.0, 2.0, 3.0, 4.0])
    assert result == {"min": 1.0, "max": 4.0, "mean": 2.5, "median": 2.5}


def test_overlap_range_when_ranges_overlap(m):
    assert m.overlap_range([0.1, 0.5], [0.3, 0.8]) == (0.3, 0.5)


def test_overlap_range_when_no_overlap(m):
    assert m.overlap_range([0.1, 0.2], [0.5, 0.6]) is None


def test_threshold_candidates_are_midpoints_of_unique_sorted_values(m):
    candidates = m.threshold_candidates([0.5, 0.1, 0.5, 0.3])
    assert candidates == [0.2, 0.4]  # unique sorted: 0.1, 0.3, 0.5 -> midpoints 0.2, 0.4


def test_threshold_candidates_single_value_has_no_candidates(m):
    assert m.threshold_candidates([0.5, 0.5]) == []


def test_classify_direction_higher_value_accepts(m):
    assert m.classify(0.9, threshold=0.5) is True
    assert m.classify(0.4, threshold=0.5) is False
    assert m.classify(0.5, threshold=0.5) is True  # boundary: >= accepts


def test_confusion_counts(m):
    labels = [True, True, False, False]  # answerable, answerable, unanswerable, unanswerable
    predictions = [True, False, False, True]  # accept, abstain, abstain, accept
    confusion = m.confusion_counts(labels, predictions)

    assert confusion.true_accept == 1
    assert confusion.false_abstain == 1
    assert confusion.true_abstain == 1
    assert confusion.false_accept == 1


def test_confusion_derived_rates(m):
    confusion = m.Confusion(true_accept=3, false_abstain=1, true_abstain=4, false_accept=0)
    assert confusion.answerable_recall == pytest.approx(3 / 4)
    assert confusion.unanswerable_rejection_rate == pytest.approx(1.0)
    assert confusion.balanced_accuracy == pytest.approx((3 / 4 + 1.0) / 2)


def test_best_single_threshold_finds_perfectly_separating_boundary(m):
    # answerable scores clearly higher than unanswerable scores
    values = [0.9, 0.8, 0.3, 0.2]
    labels = [True, True, False, False]

    threshold, confusion = m.best_single_threshold(values, labels)

    assert confusion.balanced_accuracy == pytest.approx(1.0)
    assert 0.3 < threshold < 0.8  # sits somewhere in the separating gap


def test_best_two_signal_rule_can_improve_on_a_single_signal_that_cannot_separate_cleanly(m):
    # Signal A alone can't perfectly separate: one unanswerable row (A=0.87)
    # sits between the two answerable rows' A values (0.85, 0.9), forcing
    # any single A threshold into a false-accept or a false-abstain.
    # Signal B cleanly separates that same troublesome row, so AND-ing it
    # in reaches perfect balanced accuracy where A alone tops out at 0.75.
    values_a = [0.9, 0.85, 0.87, 0.2]
    values_b = [0.9, 0.85, 0.1, 0.1]
    labels = [True, True, False, False]

    _, single_confusion = m.best_single_threshold(values_a, labels)
    _, _, combined_confusion = m.best_two_signal_rule(values_a, values_b, labels)

    assert single_confusion.balanced_accuracy == pytest.approx(0.75)
    assert combined_confusion.balanced_accuracy == pytest.approx(1.0)
    assert combined_confusion.balanced_accuracy > single_confusion.balanced_accuracy


# --- split filtering ---


def test_filter_by_split_returns_only_matching_rows(m):
    questions = [
        {"id": "a", "split": "development"},
        {"id": "b", "split": "heldout"},
        {"id": "c", "split": "development"},
    ]

    dev = m.filter_by_split(questions, "development")
    heldout = m.filter_by_split(questions, "heldout")

    assert [q["id"] for q in dev] == ["a", "c"]
    assert [q["id"] for q in heldout] == ["b"]


def test_filter_by_split_missing_field_fails_clearly(m):
    questions = [{"id": "a", "split": "development"}, {"id": "b"}]  # "b" has no split field

    with pytest.raises(ValueError, match="split"):
        m.filter_by_split(questions, "development")


def test_development_selection_cannot_be_influenced_by_heldout_rows(m):
    # A threshold fit on development-only values must be identical whether
    # or not heldout rows exist elsewhere in the full dataset - proving
    # heldout data structurally never reaches the fitting step.
    questions = [
        {"id": "d1", "split": "development"},
        {"id": "d2", "split": "development"},
        {"id": "h1", "split": "heldout"},
    ]
    dev_values = {"d1": 0.9, "d2": 0.2}
    dev_labels = {"d1": True, "d2": False}

    dev_questions = m.filter_by_split(questions, "development")
    values = [dev_values[q["id"]] for q in dev_questions]
    labels = [dev_labels[q["id"]] for q in dev_questions]

    threshold, confusion = m.best_single_threshold(values, labels)

    # Only 2 development rows were ever seen by the fitter - "h1" (which
    # has no entry in dev_values/dev_labels at all) could not have
    # contributed even if its score were wildly different.
    assert len(values) == 2
    assert confusion.balanced_accuracy == pytest.approx(1.0)
    assert 0.2 < threshold < 0.9


# --- frozen-rule application (no search) ---


def test_apply_frozen_single_threshold_does_not_search(m):
    values = [0.9, 0.1, 0.5]
    labels = [True, False, True]

    # A deliberately "wrong"/arbitrary threshold - apply_frozen_* must use
    # exactly it, never silently pick a better one from these values.
    confusion = m.apply_frozen_single_threshold(values, threshold=0.95, labels=labels)

    assert confusion.true_accept == 0  # nothing clears 0.95
    assert confusion.false_abstain == 2
    assert confusion.true_abstain == 1
    assert confusion.false_accept == 0


def test_apply_frozen_two_signal_rule_does_not_search(m):
    values_a = [0.9, 0.9]
    values_b = [0.9, 0.1]
    labels = [True, True]

    confusion = m.apply_frozen_two_signal_rule(values_a, values_b, threshold_a=0.5, threshold_b=0.5, labels=labels)

    assert confusion.true_accept == 1  # only the row with both a>=0.5 and b>=0.5
    assert confusion.false_abstain == 1


def test_frozen_rule_application_is_deterministic(m):
    values = [0.9, 0.1, 0.5, 0.5, 0.3]
    labels = [True, False, True, False, True]

    first = m.apply_frozen_single_threshold(values, threshold=0.4, labels=labels)
    second = m.apply_frozen_single_threshold(values, threshold=0.4, labels=labels)

    assert first == second


# --- negative-type breakdown ---


def test_negative_type_breakdown_groups_and_computes_stats(m):
    entries = [
        ("absent_topic", 0.80, False),  # rejected
        ("absent_topic", 0.90, True),  # falsely accepted
        ("missing_amount", 0.70, False),  # rejected
    ]

    breakdown = m.negative_type_breakdown(entries)

    assert breakdown["absent_topic"]["count"] == 2
    assert breakdown["absent_topic"]["rejected"] == 1
    assert breakdown["absent_topic"]["falsely_accepted"] == 1
    assert breakdown["absent_topic"]["rejection_rate"] == pytest.approx(0.5)
    assert breakdown["absent_topic"]["avg_top1"] == pytest.approx(0.85)

    assert breakdown["missing_amount"]["count"] == 1
    assert breakdown["missing_amount"]["rejection_rate"] == pytest.approx(1.0)


def test_negative_type_breakdown_is_deterministic_regardless_of_input_order(m):
    entries_a = [("type_x", 0.5, False), ("type_y", 0.6, True)]
    entries_b = [("type_y", 0.6, True), ("type_x", 0.5, False)]

    assert m.negative_type_breakdown(entries_a) == m.negative_type_breakdown(entries_b)


# --- _shared.load_questions() new wrapped-schema compatibility ---

_SHARED_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_shared.py"


def _load_shared_module():
    spec = importlib.util.spec_from_file_location("shared_under_test", _SHARED_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_questions_unwraps_new_schema_with_questions_key(kb_dir: Path):
    import json

    shared = _load_shared_module()
    path = kb_dir / "questions.json"
    path.write_text(json.dumps({"schema_version": 2, "notes": {}, "questions": [{"id": "a"}, {"id": "b"}]}))

    result = shared.load_questions(path)

    assert result == [{"id": "a"}, {"id": "b"}]


def test_load_questions_still_accepts_bare_list(kb_dir: Path):
    import json

    shared = _load_shared_module()
    path = kb_dir / "questions.json"
    path.write_text(json.dumps([{"id": "a"}]))

    result = shared.load_questions(path)

    assert result == [{"id": "a"}]
