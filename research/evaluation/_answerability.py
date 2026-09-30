"""Reusable, pure helper functions for the answerability diagnostic (analyze_answerability.py).

Everything here is generic statistics/threshold-sweep logic over plain
numbers - no retrieval, no model access, no evaluation labels baked in.
Kept separate from _shared.py because this is diagnostic-only analysis,
not the source/topic scoring the three retrieval evaluators share.
"""

import statistics
from dataclasses import dataclass


def score_gap(higher: float, lower: float) -> float:
    """Signed gap between two scores (e.g. Top-1 minus Top-2). Larger = more separation."""
    return higher - lower


def summarize(values: list[float]) -> dict:
    """min/max/mean/median of a list of numbers."""
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def overlap_range(group_a: list[float], group_b: list[float]) -> tuple[float, float] | None:
    """The [lo, hi] interval where group_a's and group_b's value ranges overlap, or None if they don't."""
    lo = max(min(group_a), min(group_b))
    hi = min(max(group_a), max(group_b))
    if lo > hi:
        return None
    return (lo, hi)


def threshold_candidates(values: list[float]) -> list[float]:
    """Midpoints between consecutive sorted unique values - candidate thresholds derived
    only from observed data, never a hand-picked round number."""
    unique_sorted = sorted(set(values))
    return [(unique_sorted[i] + unique_sorted[i + 1]) / 2 for i in range(len(unique_sorted) - 1)]


def classify(value: float, threshold: float) -> bool:
    """True = accept (treat as answerable), False = abstain. Direction: higher value = stronger evidence -> accept."""
    return value >= threshold


@dataclass
class Confusion:
    """Confusion counts for an accept/abstain classifier against ground-truth answerable/unanswerable labels."""

    true_accept: int  # answerable, correctly accepted
    false_abstain: int  # answerable, incorrectly abstained
    true_abstain: int  # unanswerable, correctly abstained
    false_accept: int  # unanswerable, incorrectly accepted

    @property
    def answerable_recall(self) -> float:
        denom = self.true_accept + self.false_abstain
        return self.true_accept / denom if denom else 0.0

    @property
    def unanswerable_rejection_rate(self) -> float:
        denom = self.true_abstain + self.false_accept
        return self.true_abstain / denom if denom else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.answerable_recall + self.unanswerable_rejection_rate) / 2


def confusion_counts(labels: list[bool], predictions: list[bool]) -> Confusion:
    """labels: True=answerable (ground truth). predictions: True=accept, False=abstain."""
    true_accept = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    false_abstain = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    true_abstain = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    false_accept = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    return Confusion(true_accept, false_abstain, true_abstain, false_accept)


def best_single_threshold(values: list[float], labels: list[bool]) -> tuple[float, Confusion]:
    """Sweep every observed-data threshold candidate; return the one with highest balanced accuracy.

    Ties broken by the candidate that appears first in threshold_candidates()
    (i.e. the lowest threshold achieving the best balanced accuracy) - deterministic, not cherry-picked.
    """
    best_threshold = None
    best_confusion = None
    for threshold in threshold_candidates(values):
        predictions = [classify(v, threshold) for v in values]
        confusion = confusion_counts(labels, predictions)
        if best_confusion is None or confusion.balanced_accuracy > best_confusion.balanced_accuracy:
            best_threshold, best_confusion = threshold, confusion
    return best_threshold, best_confusion


def best_two_signal_rule(
    values_a: list[float],
    values_b: list[float],
    labels: list[bool],
) -> tuple[float, float, Confusion]:
    """Grid-search over observed-data threshold candidates for two signals combined with AND.

    Accept requires value_a >= threshold_a AND value_b >= threshold_b. Returns
    the (threshold_a, threshold_b) pair with the highest balanced accuracy.
    """
    best = None
    for threshold_a in threshold_candidates(values_a):
        for threshold_b in threshold_candidates(values_b):
            predictions = [
                classify(a, threshold_a) and classify(b, threshold_b)
                for a, b in zip(values_a, values_b)
            ]
            confusion = confusion_counts(labels, predictions)
            if best is None or confusion.balanced_accuracy > best[2].balanced_accuracy:
                best = (threshold_a, threshold_b, confusion)
    return best


def filter_by_split(questions: list[dict], split: str) -> list[dict]:
    """Return only the questions whose 'split' field equals `split`.

    Fails clearly (rather than silently dropping rows) if any question is
    missing a 'split' field - a malformed dataset should be visible, not swallowed.
    """
    for question in questions:
        if "split" not in question:
            raise ValueError(f"question {question.get('id', '?')!r} has no 'split' field")
    return [q for q in questions if q["split"] == split]


def apply_frozen_single_threshold(values: list[float], threshold: float, labels: list[bool]) -> Confusion:
    """Classify with an already-decided threshold - no search happens here.

    Used to evaluate a development-selected rule on held-out data without
    any risk of accidentally re-fitting against it.
    """
    predictions = [classify(v, threshold) for v in values]
    return confusion_counts(labels, predictions)


def apply_frozen_two_signal_rule(
    values_a: list[float],
    values_b: list[float],
    threshold_a: float,
    threshold_b: float,
    labels: list[bool],
) -> Confusion:
    """Classify with already-decided thresholds - no search happens here."""
    predictions = [classify(a, threshold_a) and classify(b, threshold_b) for a, b in zip(values_a, values_b)]
    return confusion_counts(labels, predictions)


def negative_type_breakdown(entries: list[tuple[str, float, bool]]) -> dict[str, dict]:
    """Diagnostic grouping of unanswerable questions by negative_type.

    `entries` is (negative_type, top1_score, was_accepted) for unanswerable
    questions only. Returns {negative_type: {count, rejected, falsely_accepted,
    rejection_rate, avg_top1}}. Purely descriptive - never used to select
    per-type thresholds.
    """
    by_type: dict[str, list[tuple[float, bool]]] = {}
    for negative_type, top1, accepted in entries:
        by_type.setdefault(negative_type, []).append((top1, accepted))

    result = {}
    for negative_type, items in by_type.items():
        count = len(items)
        falsely_accepted = sum(1 for _, accepted in items if accepted)
        rejected = count - falsely_accepted
        result[negative_type] = {
            "count": count,
            "rejected": rejected,
            "falsely_accepted": falsely_accepted,
            "rejection_rate": rejected / count,
            "avg_top1": statistics.mean(top1 for top1, _ in items),
        }
    return result
