"""Tests for evaluation/evaluate_blind_answerability.py.

No real model is downloaded - only the pure/loading functions are tested
directly. Confirms the frozen thresholds are exactly what was specified,
that the script has no threshold-search capability, and that loading
validates the blind file's shape strictly.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_blind_answerability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blind_answerability_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


def _make_blind_payload(n_answerable=15, n_unanswerable=15, split="blind_test"):
    questions = [
        {"id": f"a{i:02d}", "split": split, "answerable": True, "question": f"q{i}"} for i in range(n_answerable)
    ] + [
        {"id": f"u{i:02d}", "split": split, "answerable": False, "question": f"q{i}"} for i in range(n_unanswerable)
    ]
    return {"schema_version": 1, "questions": questions}


# --- loading / validation ---


def test_load_blind_questions_accepts_valid_30_15_15(m, kb_dir: Path):
    path = kb_dir / "blind.json"
    path.write_text(json.dumps(_make_blind_payload()))

    result = m.load_blind_questions(path)

    assert len(result) == 30
    assert sum(1 for q in result if q["answerable"]) == 15
    assert sum(1 for q in result if not q["answerable"]) == 15


def test_load_blind_questions_rejects_wrong_total_count(m, kb_dir: Path):
    path = kb_dir / "blind.json"
    path.write_text(json.dumps(_make_blind_payload(n_answerable=14, n_unanswerable=15)))

    with pytest.raises(ValueError, match="30"):
        m.load_blind_questions(path)


def test_load_blind_questions_rejects_wrong_answerable_split(m, kb_dir: Path):
    path = kb_dir / "blind.json"
    path.write_text(json.dumps(_make_blind_payload(n_answerable=16, n_unanswerable=14)))

    with pytest.raises(ValueError, match="15"):
        m.load_blind_questions(path)


def test_load_blind_questions_rejects_wrong_split_field(m, kb_dir: Path):
    path = kb_dir / "blind.json"
    path.write_text(json.dumps(_make_blind_payload(split="development")))

    with pytest.raises(ValueError, match="blind_test"):
        m.load_blind_questions(path)


# --- frozen rules use exactly the specified constants ---


def test_frozen_cosine_thresholds_are_exact(m):
    assert m._COSINE_TOP1_MIN == 0.8591
    assert m._COSINE_GAP12_MIN == 0.0049


def test_frozen_qa_threshold_is_exact(m):
    assert m._QA_THRESHOLD == -5.7906


def test_cosine_predict_boundary(m):
    # Exactly at both thresholds -> accept (classify() is >=, inclusive).
    assert m.cosine_predict(top1_score=0.8591, top2_score=0.8591 - 0.0049) is True
    # Just below the gap threshold -> abstain.
    assert m.cosine_predict(top1_score=0.9, top2_score=0.9 - 0.0048) is False
    # Just below Top-1 threshold -> abstain even with a huge gap.
    assert m.cosine_predict(top1_score=0.8590, top2_score=0.0) is False


def test_qa_predict_boundary(m):
    assert m.qa_predict(-5.7906) is True  # exactly at threshold -> accept
    assert m.qa_predict(-5.7907) is False
    assert m.qa_predict(100.0) is True


def test_predictions_are_deterministic(m):
    assert m.cosine_predict(0.9, 0.85) == m.cosine_predict(0.9, 0.85)
    assert m.qa_predict(3.0) == m.qa_predict(3.0)


# --- no threshold-search capability ---


def test_module_imports_no_threshold_search_functions(m):
    # The blind evaluator must be structurally incapable of tuning - it
    # never binds a threshold-search function into its own namespace,
    # even though it imports the _answerability module that defines them
    # (used elsewhere, e.g. by analyze_answerability.py, but never called here).
    assert not hasattr(m, "best_single_threshold")
    assert not hasattr(m, "best_two_signal_rule")
    import inspect

    source = inspect.getsource(m)
    assert "best_single_threshold(" not in source
    assert "best_two_signal_rule(" not in source


# --- confusion matrix / head-to-head ---


def test_confusion_matrix_from_frozen_predictions(m):
    labels = [True, True, False, False]
    cosine_predictions = [m.cosine_predict(0.9, 0.86), m.cosine_predict(0.8, 0.1), m.cosine_predict(0.9, 0.86), m.cosine_predict(0.1, 0.05)]
    confusion = m.ans.confusion_counts(labels, cosine_predictions)

    assert confusion.true_accept == 1  # first answerable question, strong signal
    assert confusion.false_abstain == 1  # second answerable question, weak signal
    assert confusion.true_abstain == 1  # fourth unanswerable, weak signal
    assert confusion.false_accept == 1  # third unanswerable, strong signal (fooled)


def test_head_to_head_classification(m):
    labels = [True, True, False, False]
    cosine_predictions = [True, False, False, True]  # correct, wrong, correct, wrong
    qa_predictions = [True, True, False, False]  # correct, correct, correct, correct... wait check below
    ids = ["q1", "q2", "q3", "q4"]

    both_correct, cosine_only, qa_only, both_wrong = m.head_to_head(labels, cosine_predictions, qa_predictions, ids)

    # q1: cosine=True(correct) qa=True(correct) -> both_correct
    # q2: label=True cosine=False(wrong) qa=True(correct) -> qa_only
    # q3: label=False cosine=False(correct) qa=False(correct) -> both_correct
    # q4: label=False cosine=True(wrong) qa=False(correct) -> qa_only
    assert both_correct == ["q1", "q3"]
    assert cosine_only == []
    assert qa_only == ["q2", "q4"]
    assert both_wrong == []


def test_head_to_head_all_four_buckets_reachable(m):
    labels = [True, True, True, True]
    cosine_predictions = [True, True, False, False]
    qa_predictions = [True, False, True, False]
    ids = ["both_ok", "cosine_only", "qa_only", "both_wrong"]

    both_correct, cosine_only, qa_only, both_wrong = m.head_to_head(labels, cosine_predictions, qa_predictions, ids)

    assert both_correct == ["both_ok"]
    assert cosine_only == ["cosine_only"]
    assert qa_only == ["qa_only"]
    assert both_wrong == ["both_wrong"]


# --- negative-type grouping (reused from _answerability, integration check) ---


def test_negative_type_grouping_via_reused_helper(m):
    entries = [
        ("missing_amount", 0.9, True),  # falsely accepted
        ("missing_amount", 0.5, False),  # correctly rejected
        ("absent_topic", 0.2, False),
    ]
    breakdown = m.ans.negative_type_breakdown(entries)

    assert breakdown["missing_amount"]["count"] == 2
    assert breakdown["missing_amount"]["falsely_accepted"] == 1
    assert breakdown["absent_topic"]["rejection_rate"] == pytest.approx(1.0)
