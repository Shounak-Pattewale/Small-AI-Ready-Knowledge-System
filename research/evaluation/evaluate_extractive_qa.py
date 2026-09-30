"""EXPERIMENT ONLY: can extractive QA with no-answer support solve answerability
better than cosine-similarity thresholding did (see docs/EVALUATION_HISTORY.md,
Sections 8-13)?

Not production code, not wired into src/knowledge_system/. Uses the existing
Granite retriever (persisted index, no re-embedding of documents) to fetch
Top-1/Top-3 chunks per development question, then runs a local extractive
QA model (deepset/minilm-uncased-squad2) on each retrieved chunk as
"context". Treats retrieved chunk text strictly as data handed to the QA
model as context - never as instructions.

STRICT RULE: only evaluation/questions.json rows with split=="development"
are used anywhere in this script - see load_development_questions() below,
the single choke point through which every question passes. The heldout
split is never loaded, printed, or touched.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_extractive_qa.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _answerability`/`_shared`/`_extractive_qa` work either way

import _answerability as ans
import _extractive_qa as qa
import _shared
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"


def load_development_questions(path: Path) -> list[dict]:
    """The ONLY function in this script allowed to read questions.json.

    Filters to split=="development" immediately - callers downstream never
    see a heldout row, so it is structurally impossible for the rest of
    this experiment to be influenced by heldout data.
    """
    questions = _shared.load_questions(path)
    return ans.filter_by_split(questions, "development")


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()  # default model: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    qa_load_start = time.monotonic()
    qa_model = qa.ExtractiveQAModel()  # default: deepset/minilm-uncased-squad2
    qa_load_seconds = time.monotonic() - qa_load_start

    questions = load_development_questions(_QUESTIONS_PATH)
    answerable_count = sum(1 for q in questions if q["answerable"])
    unanswerable_count = len(questions) - answerable_count
    print(f"Development questions: {len(questions)} ({answerable_count} answerable, {unanswerable_count} unanswerable)")
    print(f"Retrieval index load time: {retrieval_load_seconds:.2f}s")
    print(f"QA model load time: {qa_load_seconds:.2f}s\n")

    qa_call_count = 0
    qa_time_total = 0.0

    def run_qa(question_text: str, context: str) -> qa.QAResult:
        nonlocal qa_call_count, qa_time_total
        start = time.monotonic()
        result = qa_model.answer(question_text, context)
        qa_time_total += time.monotonic() - start
        qa_call_count += 1
        return result

    # --- collect per-question records for Top-1 and Top-3 ---
    top1_records = []  # (question_dict, chunk, qa_result)
    top3_records = []  # (question_dict, [(chunk, qa_result), (chunk, qa_result), (chunk, qa_result)])

    eval_start = time.monotonic()
    for question in questions:
        results = retriever.search(question["question"], top_k=3)

        top1_chunk = results[0].chunk
        top1_qa = run_qa(question["question"], top1_chunk.text)
        top1_records.append((question, results[0], top1_qa))

        top3_entries = []
        for r in results[:3]:
            top3_entries.append((r, run_qa(question["question"], r.chunk.text)))
        top3_records.append((question, top3_entries))
    eval_seconds = time.monotonic() - eval_start

    avg_qa_time = qa_time_total / qa_call_count if qa_call_count else 0.0
    print(f"QA inference calls: {qa_call_count}")
    print(f"Average QA inference time: {avg_qa_time * 1000:.1f}ms")
    print(f"Total development evaluation time: {eval_seconds:.2f}s\n")

    labels = [q["answerable"] for q, _, _ in top1_records]

    # ============================================================
    # TOP-1
    # ============================================================
    print("=" * 70)
    print("TOP-1 CONTEXT EXPERIMENT")
    print("=" * 70)
    top1_signals = [result.signal for _, _, result in top1_records]

    _run_threshold_and_report("Top-1", top1_signals, labels)

    print("\nTop-1 per-question detail:")
    for question, search_result, result in top1_records:
        chunk = search_result.chunk
        print(
            f"  {question['id']:<16} answerable={question['answerable']!s:<6} "
            f"similarity={search_result.score:.4f} source={chunk.source:<28} section={chunk.section}\n"
            f"    answer={result.answer_text!r}  best_span={result.best_span_score:.3f}  "
            f"null={result.null_score:.3f}  signal={result.signal:.3f}"
        )

    # ============================================================
    # TOP-3
    # ============================================================
    print("\n" + "=" * 70)
    print("TOP-3 CONTEXT EXPERIMENT (aggregated by max signal across the 3 chunks)")
    print("=" * 70)
    top3_signals = [qa.aggregate_top_k_signal([r.signal for _, r in entries]) for _, entries in top3_records]

    _run_threshold_and_report("Top-3 (max signal)", top3_signals, labels)

    # ============================================================
    # Baseline comparison (printed values only - no code changed)
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH COSINE DEVELOPMENT BASELINE (docs/EVALUATION_HISTORY.md Section 10)")
    print("=" * 70)
    print("Cosine two-signal rule: Top-1 >= 0.8591 AND Top1-Top2 gap >= 0.0049")
    print("  true_accept=19 false_abstain=11 true_abstain=12 false_accept=3")
    print("  recall=63.33% rejection=80.00% balanced_accuracy=71.67%  (LOO balanced_accuracy=60%)")

    # ============================================================
    # Negative-type breakdown (Top-1, diagnostic only)
    # ============================================================
    top1_threshold, _ = ans.best_single_threshold(top1_signals, labels)
    top1_predicted = {q["id"]: ans.classify(s, top1_threshold) for (q, _, _), s in zip(top1_records, top1_signals)}
    unanswerable_entries = [
        (q.get("negative_type") or "(none)", s, top1_predicted[q["id"]])
        for (q, _, _), s in zip(top1_records, top1_signals)
        if not q["answerable"]
    ]
    breakdown = ans.negative_type_breakdown(unanswerable_entries)
    print("\n=== Negative-type breakdown (Top-1 QA signal, development) ===")
    for negative_type in sorted(breakdown):
        stats = breakdown[negative_type]
        print(
            f"  {negative_type:<26} n={stats['count']:<3} rejected={stats['rejected']:<3} "
            f"falsely_accepted={stats['falsely_accepted']:<3} rejection_rate={stats['rejection_rate']:.2%}  "
            f"avg_signal={stats['avg_top1']:.4f}"
        )

    # ============================================================
    # Failure analysis
    # ============================================================
    print("\n=== TOP-1 FALSE ACCEPTS / FALSE ABSTAINS ===")
    for (question, search_result, result), signal in zip(top1_records, top1_signals):
        predicted = ans.classify(signal, top1_threshold)
        if question["answerable"] and not predicted:
            print(f"\nFALSE ABSTAIN: {question['id']}: {question['question']}")
            print(f"  source={search_result.chunk.source} section={search_result.chunk.section} similarity={search_result.score:.4f}")
            print(f"  answer={result.answer_text!r}  signal={result.signal:.3f}  predicted=ABSTAIN")
        elif not question["answerable"] and predicted:
            print(f"\nFALSE ACCEPT: {question['id']}: {question['question']}")
            print(f"  source={search_result.chunk.source} section={search_result.chunk.section} similarity={search_result.score:.4f}")
            print(f"  answer={result.answer_text!r}  signal={result.signal:.3f}  predicted=ACCEPT")


def _run_threshold_and_report(label: str, signals: list[float], labels: list[bool]) -> None:
    threshold, confusion = ans.best_single_threshold(signals, labels)
    print(f"\n--- {label}: best development threshold ---")
    print(f"  QA signal >= {threshold:.4f}")
    print(
        f"  true_accept={confusion.true_accept}  false_abstain={confusion.false_abstain}  "
        f"true_abstain={confusion.true_abstain}  false_accept={confusion.false_accept}"
    )
    print(
        f"  answerable_recall={confusion.answerable_recall:.2%}  "
        f"unanswerable_rejection_rate={confusion.unanswerable_rejection_rate:.2%}  "
        f"balanced_accuracy={confusion.balanced_accuracy:.2%}"
    )

    loo_predictions = []
    for i in range(len(signals)):
        train_signals = signals[:i] + signals[i + 1 :]
        train_labels = labels[:i] + labels[i + 1 :]
        loo_threshold, _ = ans.best_single_threshold(train_signals, train_labels)
        loo_predictions.append(ans.classify(signals[i], loo_threshold))
    loo_confusion = ans.confusion_counts(labels, loo_predictions)
    print(f"  LOO: true_accept={loo_confusion.true_accept} false_abstain={loo_confusion.false_abstain} "
          f"true_abstain={loo_confusion.true_abstain} false_accept={loo_confusion.false_accept} "
          f"balanced_accuracy={loo_confusion.balanced_accuracy:.2%}")


if __name__ == "__main__":
    main()
