"""Diagnostic-only: can simple retrieval signals distinguish answerable from unanswerable questions?

Loads the persisted Granite semantic index (storage/embeddings.npy +
manifest via EmbeddingRetriever.load_index()) - does NOT recompute
document embeddings, does NOT run Docling/raw-document ingestion, and
does NOT modify any retriever, chunk, or embedding artifact.

Experimental discipline: evaluation/questions.json now has "development"
(45 questions) and "heldout" (20 questions) splits. ALL threshold/rule
selection happens on split=="development" only. The winning rule is then
applied to split=="heldout" exactly as selected - no search, no retuning
- and reported honestly even if it performs worse there.

Only retrieval-time signals are used (scores, gaps, source diversity) -
never expected_source/expected_topic, which are evaluation labels a real
user query would never have.

Run from the project root:
    venv/bin/python research/evaluation/analyze_answerability.py
"""

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _answerability`/`_shared` work either way

import _answerability as ans
import _shared
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"
_TOP_K = 5


@dataclass
class QuestionRecord:
    id: str
    split: str
    category: str
    answerable: bool
    negative_type: str | None  # only set for unanswerable questions
    scores: list[float]  # Top-5 cosine scores, descending
    top1_source: str
    top1_section: str | None
    distinct_sources_top3: int
    distinct_sources_top5: int

    @property
    def top1(self) -> float:
        return self.scores[0]

    @property
    def gap12(self) -> float:
        return ans.score_gap(self.scores[0], self.scores[1])

    @property
    def gap13(self) -> float:
        return ans.score_gap(self.scores[0], self.scores[2])

    @property
    def top3_mean(self) -> float:
        return statistics.mean(self.scores[:3])

    @property
    def top5_std(self) -> float:
        return statistics.pstdev(self.scores[:5])  # population stdev over exactly these 5 observed values


def collect_records(retriever: EmbeddingRetriever, questions: list[dict]) -> list[QuestionRecord]:
    records = []
    for question in questions:
        results = retriever.search(question["question"], top_k=_TOP_K)
        scores = [r.score for r in results]
        records.append(
            QuestionRecord(
                id=question["id"],
                split=question["split"],
                category=question["category"],
                answerable=question["answerable"],
                negative_type=question.get("negative_type"),
                scores=scores,
                top1_source=results[0].chunk.source,
                top1_section=results[0].chunk.section,
                distinct_sources_top3=len({r.chunk.source for r in results[:3]}),
                distinct_sources_top5=len({r.chunk.source for r in results[:5]}),
            )
        )
    return records


def print_group_summary(name: str, records: list[QuestionRecord]) -> None:
    print(f"\n--- {name} (n={len(records)}) ---")
    for label, extractor in [
        ("Top-1", lambda r: r.top1),
        ("Top1-Top2 gap", lambda r: r.gap12),
        ("Top1-Top3 gap", lambda r: r.gap13),
        ("Top-3 mean", lambda r: r.top3_mean),
        ("Top-5 stdev", lambda r: r.top5_std),
    ]:
        values = [extractor(r) for r in records]
        s = ans.summarize(values)
        print(f"  {label:<16} min={s['min']:.4f}  max={s['max']:.4f}  mean={s['mean']:.4f}  median={s['median']:.4f}")


def print_overlap(label: str, answerable_values: list[float], unanswerable_values: list[float]) -> None:
    overlap = ans.overlap_range(answerable_values, unanswerable_values)
    if overlap is None:
        print(f"  {label:<16} NO OVERLAP (clean separation)")
    else:
        print(f"  {label:<16} overlap = [{overlap[0]:.4f}, {overlap[1]:.4f}]")


def print_confusion(confusion: ans.Confusion) -> None:
    print(
        f"  true_accept={confusion.true_accept}  false_abstain={confusion.false_abstain}  "
        f"true_abstain={confusion.true_abstain}  false_accept={confusion.false_accept}"
    )
    print(
        f"  answerable_recall={confusion.answerable_recall:.2%}  "
        f"unanswerable_rejection_rate={confusion.unanswerable_rejection_rate:.2%}  "
        f"balanced_accuracy={confusion.balanced_accuracy:.2%}"
    )


def print_single_signal_result(label: str, values: list[float], labels: list[bool]) -> tuple[float, ans.Confusion]:
    threshold, confusion = ans.best_single_threshold(values, labels)
    print(f"\n--- [development] Best single-signal threshold: {label} ---")
    print(f"  threshold >= {threshold:.4f}")
    print_confusion(confusion)
    return threshold, confusion


def print_two_signal_result(
    label: str, values_a: list[float], values_b: list[float], labels: list[bool]
) -> tuple[float, float, ans.Confusion]:
    threshold_a, threshold_b, confusion = ans.best_two_signal_rule(values_a, values_b, labels)
    print(f"\n--- [development] Best two-signal rule: {label} ---")
    print(f"  threshold_A >= {threshold_a:.4f}  AND  threshold_B >= {threshold_b:.4f}")
    print_confusion(confusion)
    return threshold_a, threshold_b, confusion


def leave_one_out_single_signal(values: list[float], labels: list[bool]) -> ans.Confusion:
    predictions = []
    for i in range(len(values)):
        train_values = values[:i] + values[i + 1 :]
        train_labels = labels[:i] + labels[i + 1 :]
        threshold, _ = ans.best_single_threshold(train_values, train_labels)
        predictions.append(ans.classify(values[i], threshold))
    return ans.confusion_counts(labels, predictions)


def leave_one_out_two_signal(values_a: list[float], values_b: list[float], labels: list[bool]) -> ans.Confusion:
    predictions = []
    for i in range(len(values_a)):
        train_a = values_a[:i] + values_a[i + 1 :]
        train_b = values_b[:i] + values_b[i + 1 :]
        train_labels = labels[:i] + labels[i + 1 :]
        threshold_a, threshold_b, _ = ans.best_two_signal_rule(train_a, train_b, train_labels)
        predictions.append(ans.classify(values_a[i], threshold_a) and ans.classify(values_b[i], threshold_b))
    return ans.confusion_counts(labels, predictions)


def print_negative_type_breakdown(name: str, records: list[QuestionRecord], predicted_accept: dict[str, bool]) -> None:
    """records: unanswerable questions only. predicted_accept: id -> whether the frozen rule accepted it."""
    entries = [(r.negative_type or "(none)", r.top1, predicted_accept[r.id]) for r in records]
    breakdown = ans.negative_type_breakdown(entries)
    print(f"\n--- Negative-type breakdown: {name} ---")
    if not breakdown:
        print("  (no unanswerable questions in this split)")
        return
    for negative_type in sorted(breakdown):
        stats = breakdown[negative_type]
        print(
            f"  {negative_type:<26} n={stats['count']:<3} rejected={stats['rejected']:<3} "
            f"falsely_accepted={stats['falsely_accepted']:<3} "
            f"rejection_rate={stats['rejection_rate']:.2%}  avg_top1={stats['avg_top1']:.4f}"
        )


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    retriever = EmbeddingRetriever()  # default model: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded

    questions = _shared.load_questions(_QUESTIONS_PATH)
    # Split the raw question list FIRST (before any retrieval/analysis), so
    # it's structurally impossible for development-set analysis below to
    # ever touch a heldout row.
    dev_questions = ans.filter_by_split(questions, "development")
    heldout_questions = ans.filter_by_split(questions, "heldout")

    dev_records = collect_records(retriever, dev_questions)
    heldout_records = collect_records(retriever, heldout_questions)
    records = dev_records + heldout_records

    dev_answerable = [r for r in dev_records if r.answerable]
    dev_unanswerable = [r for r in dev_records if not r.answerable]
    heldout_answerable = [r for r in heldout_records if r.answerable]
    heldout_unanswerable = [r for r in heldout_records if not r.answerable]

    print(
        f"Collected Top-{_TOP_K} retrieval signals for {len(records)} questions\n"
        f"  development: {len(dev_records)} ({len(dev_answerable)} answerable, {len(dev_unanswerable)} unanswerable)\n"
        f"  heldout:     {len(heldout_records)} ({len(heldout_answerable)} answerable, {len(heldout_unanswerable)} unanswerable)\n"
    )

    # ============================================================
    # DEVELOPMENT: all analysis, threshold search, LOO happens ONLY here
    # ============================================================
    print("=" * 70)
    print("DEVELOPMENT SET ANALYSIS (threshold selection happens here only)")
    print("=" * 70)

    print_group_summary("Development answerable", dev_answerable)
    print_group_summary("Development unanswerable", dev_unanswerable)

    print("\n=== Development overlap ranges (answerable vs unanswerable) ===")
    print_overlap("Top-1", [r.top1 for r in dev_answerable], [r.top1 for r in dev_unanswerable])
    print_overlap("Top1-Top2 gap", [r.gap12 for r in dev_answerable], [r.gap12 for r in dev_unanswerable])
    print_overlap("Top1-Top3 gap", [r.gap13 for r in dev_answerable], [r.gap13 for r in dev_unanswerable])
    print_overlap("Top-3 mean", [r.top3_mean for r in dev_answerable], [r.top3_mean for r in dev_unanswerable])
    print_overlap("Top-5 stdev", [r.top5_std for r in dev_answerable], [r.top5_std for r in dev_unanswerable])

    dev_labels = [r.answerable for r in dev_records]
    dev_top1 = [r.top1 for r in dev_records]
    dev_gap12 = [r.gap12 for r in dev_records]
    dev_gap13 = [r.gap13 for r in dev_records]

    print("\n=== Development single-signal threshold sweeps ===")
    top1_threshold, top1_confusion = print_single_signal_result("Top-1 cosine similarity", dev_top1, dev_labels)
    gap12_threshold, gap12_confusion = print_single_signal_result("Top1-Top2 gap", dev_gap12, dev_labels)
    gap13_threshold, gap13_confusion = print_single_signal_result("Top1-Top3 gap", dev_gap13, dev_labels)

    print("\n=== Development two-signal AND rules ===")
    two_a1, two_b1, two_confusion_gap12 = print_two_signal_result(
        "Top-1 AND Top1-Top2 gap", dev_top1, dev_gap12, dev_labels
    )
    two_a2, two_b2, two_confusion_gap13 = print_two_signal_result(
        "Top-1 AND Top1-Top3 gap", dev_top1, dev_gap13, dev_labels
    )

    print("\n=== Development leave-one-out robustness ===")
    loo_top1 = leave_one_out_single_signal(dev_top1, dev_labels)
    print("Top-1-only LOO:")
    print_confusion(loo_top1)
    loo_two_signal = leave_one_out_two_signal(dev_top1, dev_gap12, dev_labels)
    print("Top-1 AND Top1-Top2-gap LOO:")
    print_confusion(loo_two_signal)

    # Freeze the best-performing rule from development only, by balanced accuracy.
    candidates = [
        ("Top-1 only", top1_confusion.balanced_accuracy, ("single", top1_threshold)),
        ("Top1-Top2 gap only", gap12_confusion.balanced_accuracy, ("single_gap12", gap12_threshold)),
        ("Top1-Top3 gap only", gap13_confusion.balanced_accuracy, ("single_gap13", gap13_threshold)),
        ("Top-1 AND Top1-Top2 gap", two_confusion_gap12.balanced_accuracy, ("two_gap12", two_a1, two_b1)),
        ("Top-1 AND Top1-Top3 gap", two_confusion_gap13.balanced_accuracy, ("two_gap13", two_a2, two_b2)),
    ]
    best_name, best_balanced_accuracy, best_rule = max(candidates, key=lambda c: c[1])

    print("\n" + "=" * 70)
    print(f"FROZEN RULE (selected on development only, balanced_accuracy={best_balanced_accuracy:.2%}):")
    if best_rule[0] == "single":
        print(f"  Top-1 >= {best_rule[1]:.4f}")
    elif best_rule[0] == "single_gap12":
        print(f"  Top1-Top2 gap >= {best_rule[1]:.4f}")
    elif best_rule[0] == "single_gap13":
        print(f"  Top1-Top3 gap >= {best_rule[1]:.4f}")
    elif best_rule[0] == "two_gap12":
        print(f"  Top-1 >= {best_rule[1]:.4f}  AND  Top1-Top2 gap >= {best_rule[2]:.4f}")
    elif best_rule[0] == "two_gap13":
        print(f"  Top-1 >= {best_rule[1]:.4f}  AND  Top1-Top3 gap >= {best_rule[2]:.4f}")
    print("=" * 70)

    def predict(record: QuestionRecord) -> bool:
        """Apply the FROZEN rule (no fitting) to one record."""
        if best_rule[0] == "single":
            return ans.classify(record.top1, best_rule[1])
        if best_rule[0] == "single_gap12":
            return ans.classify(record.gap12, best_rule[1])
        if best_rule[0] == "single_gap13":
            return ans.classify(record.gap13, best_rule[1])
        if best_rule[0] == "two_gap12":
            return ans.classify(record.top1, best_rule[1]) and ans.classify(record.gap12, best_rule[2])
        return ans.classify(record.top1, best_rule[1]) and ans.classify(record.gap13, best_rule[2])

    # ============================================================
    # HELDOUT: apply the frozen rule exactly once, no search, no retuning
    # ============================================================
    print("\n" + "=" * 70)
    print("HELD-OUT EVALUATION (frozen rule applied once, no threshold search)")
    print("=" * 70)
    print(f"Held-out sample count: {len(heldout_records)} "
          f"({len(heldout_answerable)} answerable, {len(heldout_unanswerable)} unanswerable)")

    heldout_labels = [r.answerable for r in heldout_records]
    heldout_predictions = [predict(r) for r in heldout_records]
    heldout_confusion = ans.confusion_counts(heldout_labels, heldout_predictions)
    print("\nHeld-out confusion matrix:")
    print_confusion(heldout_confusion)

    predicted_accept_by_id = {r.id: p for r, p in zip(heldout_records, heldout_predictions)}
    dev_predicted_accept_by_id = {r.id: predict(r) for r in dev_records}

    print("\nHELD-OUT FALSE ACCEPTS (unanswerable questions that would get an answer):")
    heldout_by_id = {r.id: r for r in heldout_records}
    questions_by_id = {q["id"]: q for q in questions}
    for r in heldout_records:
        if not r.answerable and predicted_accept_by_id[r.id]:
            q = questions_by_id[r.id]
            print(f"  {r.id}: {q['question']}")
            print(f"    top1={r.top1:.4f}  predicted=ACCEPT  expected_answerable=False")
            print(f"    top1_source={r.top1_source}  top1_section={r.top1_section}")

    print("\nHELD-OUT FALSE ABSTAINS (answerable questions that would incorrectly abstain):")
    for r in heldout_records:
        if r.answerable and not predicted_accept_by_id[r.id]:
            q = questions_by_id[r.id]
            print(f"  {r.id}: {q['question']}")
            print(f"    top1={r.top1:.4f}  predicted=ABSTAIN  expected_answerable=True")

    print_negative_type_breakdown("development", dev_unanswerable, dev_predicted_accept_by_id)
    print_negative_type_breakdown("heldout", heldout_unanswerable, predicted_accept_by_id)


if __name__ == "__main__":
    main()
