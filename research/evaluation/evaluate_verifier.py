"""EXPERIMENTAL ONLY: Question-Native Evidence Verifier - Corpus V2.

Pipeline under test:

    question -> Granite Top-1 chunk -> VerifierModel (flan-t5-small) -> SUPPORTED / NOT_SUPPORTED

No question-to-hypothesis/claim conversion (that was NLI Experiments 1/2,
both rejected - see docs/EVALUATION_HISTORY.md Sections 19-20). The
verifier receives the original question and the raw retrieved chunk
text directly. QA is NOT run before the verifier in the standalone
experiment (Steps 8-17); it is only combined in Step 18.

Malformed-output policy (frozen before seeing any split's results):
a malformed decision counts as NOT_SUPPORTED (abstain) in every
confusion count, and is also reported separately as its own rate.
Abstaining on an unparseable response is the safer default for a
guardrail role.

Sequence: load frozen config -> development -> freeze -> heldout
regression -> blind-origin regression -> manual laptop diagnostics ->
QA+verifier combination -> report. No threshold search exists anywhere
in this file; VerifierModel returns a discrete decision.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_verifier.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _answerability`/`_shared`/`_extractive_qa`/`_verifier` work either way

import _answerability as ans
import _extractive_qa as qa
import _shared
import _verifier
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"
_BLIND_PATH = Path(__file__).resolve().parent.parent / "datasets" / "fresh_blind_test_questions.json"

_QA_THRESHOLD = -5.7906  # frozen production threshold, reused unmodified for Step 18 only

_TARGETED_FALSE_ACCEPT_IDS = [
    "unanswerable_01", "unanswerable_07", "unanswerable_08", "unanswerable_09",
    "unanswerable_11", "unanswerable_12", "unanswerable_14",
]

_LAPTOP_QUESTIONS = [
    "I lost my work laptop on the train, what should I do?",
    "My work laptop was stolen. Who should I report it to?",
]


@dataclass
class Record:
    q: dict
    top1: object
    verifier_result: _verifier.VerifierResult
    supported: bool  # decision == "SUPPORTED" (malformed -> False, i.e. abstain)
    qa_result: object | None = None  # only populated for Step 18


def verifier_predict(result: _verifier.VerifierResult) -> bool:
    """True = accept (answered). Malformed output abstains (frozen policy, see module docstring)."""
    return result.decision == "SUPPORTED"


def build_records(questions: list[dict], retriever: EmbeddingRetriever, verifier: _verifier.VerifierModel) -> list[Record]:
    records = []
    for question in questions:
        results = retriever.search(question["question"], top_k=1)
        top1 = results[0]
        verifier_result = verifier.verify(question["question"], top1.chunk.text)
        records.append(Record(q=question, top1=top1, verifier_result=verifier_result, supported=verifier_predict(verifier_result)))
    return records


def print_confusion(label: str, confusion: ans.Confusion, n: int) -> None:
    accuracy = (confusion.true_accept + confusion.true_abstain) / n
    print(f"\n--- {label} ---")
    print(
        f"  true_accept={confusion.true_accept}  false_abstain={confusion.false_abstain}  "
        f"true_abstain={confusion.true_abstain}  false_accept={confusion.false_accept}"
    )
    print(
        f"  answerable_recall={confusion.answerable_recall:.2%}  "
        f"unanswerable_rejection_rate={confusion.unanswerable_rejection_rate:.2%}  "
        f"balanced_accuracy={confusion.balanced_accuracy:.2%}  plain_accuracy={accuracy:.2%}"
    )


def malformed_rate(records: list[Record]) -> tuple[int, int, float]:
    malformed = sum(1 for r in records if r.verifier_result.malformed)
    return malformed, len(records), (malformed / len(records) if records else 0.0)


def report_split(label: str, records: list[Record]) -> ans.Confusion:
    labels = [r.q["answerable"] for r in records]
    predictions = [r.supported for r in records]
    confusion = ans.confusion_counts(labels, predictions)
    malformed, n, rate = malformed_rate(records)
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print_confusion("Verifier", confusion, n)
    print(f"  malformed_outputs={malformed}/{n} = {rate:.2%}")
    return confusion


def print_targeted_false_accepts(records: list[Record]) -> None:
    by_id = {r.q["id"]: r for r in records}
    print(f"\n{'=' * 70}\nTARGETED FALSE-ACCEPT ANALYSIS (7 previous frozen-QA development false accepts)\n{'=' * 70}")
    for qid in _TARGETED_FALSE_ACCEPT_IDS:
        r = by_id.get(qid)
        if r is None:
            print(f"{qid}: not present in this split")
            continue
        outcome = "STILL FALSE ACCEPT" if r.supported else "FIXED (correctly rejected)"
        excerpt = r.top1.chunk.text[:220].replace("\n", " ")
        print(
            f"\n{qid}: {r.q['question']}\n"
            f"  source={r.top1.chunk.source}  section={r.top1.chunk.section}\n"
            f"  evidence_excerpt={excerpt!r}\n"
            f"  verifier_decision={r.verifier_result.decision}  ({outcome})"
        )


def print_answerable_false_abstains(records: list[Record]) -> None:
    print(f"\n{'=' * 70}\nDEVELOPMENT FALSE ABSTAINS (answerable questions the verifier rejected)\n{'=' * 70}")
    retrieval_failures, verifier_failures = 0, 0
    for r in records:
        if not r.q["answerable"] or r.supported:
            continue
        acceptable = _shared.acceptable_sources(r.q)
        retrieval_ok = r.top1.chunk.source in acceptable
        kind = "RETRIEVAL FAILURE (wrong passage retrieved)" if not retrieval_ok else "VERIFIER FAILURE (correct passage, verifier rejected it)"
        if retrieval_ok:
            verifier_failures += 1
        else:
            retrieval_failures += 1
        excerpt = r.top1.chunk.text[:220].replace("\n", " ")
        print(
            f"\n{r.q['id']}: {r.q['question']}\n"
            f"  expected_source={_shared.expected_source_label(r.q)}  actual_source={r.top1.chunk.source}\n"
            f"  evidence_excerpt={excerpt!r}\n"
            f"  verifier_decision={r.verifier_result.decision}  ({kind})"
        )
    print(f"\nBreakdown: {retrieval_failures} retrieval failures, {verifier_failures} genuine verifier failures.")


def retrieval_conditional_metrics(records: list[Record]) -> None:
    """Verifier metrics restricted to the subset where retrieval is independently known to be correct.

    Only answerable questions carry an expected_source label, so this is
    computed on that subset for recall, and reported alongside (not
    combined into) full-set rejection - unanswerable questions have no
    expected-source label to condition on, so manufacturing a combined
    number there would misrepresent what's actually verified.
    """
    answerable = [r for r in records if r.q["answerable"]]
    retrieval_correct = [r for r in answerable if r.top1.chunk.source in _shared.acceptable_sources(r.q)]
    retrieval_wrong = [r for r in answerable if r not in retrieval_correct]

    print(f"\n{'=' * 70}\nRETRIEVAL-CONDITIONAL VERIFIER METRICS (answerable subset only)\n{'=' * 70}")
    print(f"Answerable questions with correct Top-1 retrieval: {len(retrieval_correct)}/{len(answerable)}")
    if retrieval_correct:
        recall_correct = sum(r.supported for r in retrieval_correct) / len(retrieval_correct)
        print(f"  Verifier recall when retrieval is correct: {recall_correct:.2%} ({sum(r.supported for r in retrieval_correct)}/{len(retrieval_correct)})")
    if retrieval_wrong:
        recall_wrong = sum(r.supported for r in retrieval_wrong) / len(retrieval_wrong)
        print(f"  Verifier recall when retrieval is WRONG (verifier cannot fix bad retrieval): {recall_wrong:.2%} ({sum(r.supported for r in retrieval_wrong)}/{len(retrieval_wrong)})")
    print("(Unanswerable questions have no expected-source label, so rejection rate isn't conditionable this way - full-set rejection above is the only available number.)")


def print_negative_answer_sanity_check(records: list[Record]) -> None:
    """Unanswerable questions with negative_type indicating the evidence DOES state a negative
    answer (unsupported_exception/unsupported_consequence/unsupported_entitlement patterns often
    overlap with a real negative statement) are diagnostic only - see module docstring policy.
    This just prints every unanswerable record's decision + excerpt for manual inspection,
    since "the evidence states a negative answer" isn't itself a label in questions.json.
    """
    print(f"\n{'=' * 70}\nNEGATIVE-ANSWER SANITY CHECK (unanswerable questions, manual inspection)\n{'=' * 70}")
    print("Looking for: verifier confusing SUPPORTED with 'affirmative answer' rather than 'evidence resolves the question'.")
    for r in records:
        if r.q["answerable"]:
            continue
        excerpt = r.top1.chunk.text[:160].replace("\n", " ")
        print(f"{r.q['id']:<18} negative_type={r.q.get('negative_type', '(none)'):<24} decision={r.verifier_result.decision:<14} evidence={excerpt!r}")


def print_missing_detail_sanity_check(records: list[Record]) -> None:
    detail_types = {"missing_amount", "missing_deadline", "missing_entitlement", "missing_limit"}
    print(f"\n{'=' * 70}\nMISSING-DETAIL SANITY CHECK (amount/deadline/entitlement/limit negative_types)\n{'=' * 70}")
    relevant = [r for r in records if not r.q["answerable"] and r.q.get("negative_type") in detail_types]
    if not relevant:
        print("(no missing-detail-type questions in this split)")
        return
    correct = sum(1 for r in relevant if not r.supported)
    print(f"Correctly rejected: {correct}/{len(relevant)} = {correct / len(relevant):.2%}")
    for r in relevant:
        excerpt = r.top1.chunk.text[:160].replace("\n", " ")
        mark = "OK" if not r.supported else "FAILED (accepted, but detail is missing)"
        print(f"{r.q['id']:<18} negative_type={r.q['negative_type']:<20} decision={r.verifier_result.decision:<14} {mark}\n  evidence={excerpt!r}")


def laptop_diagnostic(question: str, retriever: EmbeddingRetriever, verifier: _verifier.VerifierModel, qa_model: qa.ExtractiveQAModel) -> None:
    print(f"\n--- Manual diagnostic: {question!r} ---")
    results = retriever.search(question, top_k=3)
    for rank, result in enumerate(results, start=1):
        verifier_result = verifier.verify(question, result.chunk.text)
        excerpt = result.chunk.text[:220].replace("\n", " ")
        print(
            f"  rank={rank} similarity={result.score:.4f} source={result.chunk.source} section={result.chunk.section}\n"
            f"    evidence_excerpt={excerpt!r}\n"
            f"    verifier_decision={verifier_result.decision}"
        )
        if verifier_result.decision == "SUPPORTED":
            qa_result = qa_model.answer(question, result.chunk.text)
            print(f"    QA answer on this SUPPORTED evidence (diagnostic only): {qa_result.answer_text!r}")


def qa_plus_verifier(records: list[Record], qa_model: qa.ExtractiveQAModel) -> ans.Confusion:
    """Step 18: answer only if QA passes its frozen threshold AND verifier == SUPPORTED. Neither retuned."""
    for r in records:
        if r.qa_result is None:
            r.qa_result = qa_model.answer(r.q["question"], r.top1.chunk.text)
    labels = [r.q["answerable"] for r in records]
    predictions = [ans.classify(r.qa_result.signal, _QA_THRESHOLD) and r.supported for r in records]
    return ans.confusion_counts(labels, predictions)


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    verifier_load_start = time.monotonic()
    verifier = _verifier.VerifierModel()  # frozen: google/flan-t5-small
    verifier_load_seconds = time.monotonic() - verifier_load_start

    qa_load_start = time.monotonic()
    qa_model = qa.ExtractiveQAModel()  # frozen: deepset/minilm-uncased-squad2, only used in Step 18/manual diagnostics
    qa_load_seconds = time.monotonic() - qa_load_start

    print(f"Retrieval index load time: {retrieval_load_seconds:.2f}s")
    print(f"Verifier model load time: {verifier_load_seconds:.2f}s")
    print(f"QA model load time: {qa_load_seconds:.2f}s")

    all_questions = _shared.load_questions(_QUESTIONS_PATH)
    dev_questions = ans.filter_by_split(all_questions, "development")
    heldout_questions = ans.filter_by_split(all_questions, "heldout")

    # ---------- Step 8-11: development ----------
    dev_start = time.monotonic()
    dev_records = build_records(dev_questions, retriever, verifier)
    dev_seconds = time.monotonic() - dev_start
    dev_confusion = report_split("DEVELOPMENT (verifier-only, Granite Top-1 -> verifier)", dev_records)
    print(f"Development eval time: {dev_seconds:.2f}s over {len(dev_records)} questions "
          f"({dev_seconds / len(dev_records) * 1000:.1f}ms/question avg, verifier calls only)")

    print_targeted_false_accepts(dev_records)
    print_answerable_false_abstains(dev_records)
    retrieval_conditional_metrics(dev_records)

    # ---------- Step 12: freeze (nothing to tune - decision rule is already discrete) ----------
    print(f"\n{'=' * 70}\nFROZEN CONFIGURATION (unchanged for all splits below)\n{'=' * 70}")
    print("model=google/flan-t5-small  decoding=do_sample=False,num_beams=1,max_new_tokens=6")
    print("decision rule=parse_decision() text match, malformed->NOT_SUPPORTED (abstain)")
    print(f"instruction=\n{_verifier.VERIFIER_INSTRUCTION}")

    # ---------- Step 13: heldout regression ----------
    heldout_records = build_records(heldout_questions, retriever, verifier)
    heldout_confusion = report_split("HELDOUT REGRESSION (frozen verifier, unchanged)", heldout_records)
    print("Compare: QA-only recall=100% rejection=60% balanced=80.00%; "
          "NLI-v1 recall=90% rejection=90% balanced=90.00%")

    # ---------- Step 14: blind-origin regression ----------
    blind_questions = _shared.load_questions(_BLIND_PATH)
    blind_records = build_records(blind_questions, retriever, verifier)
    blind_confusion = report_split("BLIND-ORIGIN REGRESSION (fixed regression set, frozen verifier)", blind_records)
    print("Compare: QA-only recall=100% rejection=46.67% balanced=73.33%; "
          "NLI-v1 recall=86.67% rejection=86.67% balanced=86.67%")

    # ---------- Step 16-17: sanity checks (development split) ----------
    print_negative_answer_sanity_check(dev_records)
    print_missing_detail_sanity_check(dev_records)

    # ---------- Step 15: manual laptop diagnostics ----------
    print(f"\n{'=' * 70}\nMANUAL LAPTOP DIAGNOSTICS (Top-3, diagnostic only, no reranking)\n{'=' * 70}")
    for question in _LAPTOP_QUESTIONS:
        laptop_diagnostic(question, retriever, verifier, qa_model)

    # ---------- Step 18: QA + verifier combination ----------
    dev_combo = qa_plus_verifier(dev_records, qa_model)
    heldout_combo = qa_plus_verifier(heldout_records, qa_model)
    blind_combo = qa_plus_verifier(blind_records, qa_model)
    print(f"\n{'=' * 70}\nQA + VERIFIER COMBINATION (answer only if QA passes {_QA_THRESHOLD} AND verifier==SUPPORTED)\n{'=' * 70}")
    print_confusion("Development", dev_combo, len(dev_records))
    print_confusion("Heldout", heldout_combo, len(heldout_records))
    print_confusion("Blind-origin", blind_combo, len(blind_records))

    # ---------- Step 19: comparison table ----------
    print(f"\n{'=' * 70}\nARCHITECTURE COMPARISON\n{'=' * 70}")
    print(f"{'split':<14} {'system':<16} {'recall':>8} {'rejection':>10} {'balanced':>10}")
    rows = [
        ("development", "QA-only", 0.9667, 0.5333, 0.7500),
        ("development", "NLI-v1", 0.8333, 0.8667, 0.8500),
        ("development", "Verifier-only", dev_confusion.answerable_recall, dev_confusion.unanswerable_rejection_rate, dev_confusion.balanced_accuracy),
        ("development", "QA+Verifier", dev_combo.answerable_recall, dev_combo.unanswerable_rejection_rate, dev_combo.balanced_accuracy),
        ("heldout", "QA-only", 1.0, 0.60, 0.80),
        ("heldout", "NLI-v1", 0.90, 0.90, 0.90),
        ("heldout", "Verifier-only", heldout_confusion.answerable_recall, heldout_confusion.unanswerable_rejection_rate, heldout_confusion.balanced_accuracy),
        ("heldout", "QA+Verifier", heldout_combo.answerable_recall, heldout_combo.unanswerable_rejection_rate, heldout_combo.balanced_accuracy),
        ("blind-origin", "QA-only", 1.0, 0.4667, 0.7333),
        ("blind-origin", "NLI-v1", 0.8667, 0.8667, 0.8667),
        ("blind-origin", "Verifier-only", blind_confusion.answerable_recall, blind_confusion.unanswerable_rejection_rate, blind_confusion.balanced_accuracy),
        ("blind-origin", "QA+Verifier", blind_combo.answerable_recall, blind_combo.unanswerable_rejection_rate, blind_combo.balanced_accuracy),
    ]
    for split, system, recall, rejection, balanced in rows:
        print(f"{split:<14} {system:<16} {recall:>8.2%} {rejection:>10.2%} {balanced:>10.2%}")

    # ---------- Step 21: performance ----------
    all_records = dev_records + heldout_records + blind_records
    n_total = len(all_records)
    total_verifier_seconds = dev_seconds  # dev split already timed end-to-end; used as the representative per-question figure
    per_question_ms = (total_verifier_seconds / len(dev_records)) * 1000
    print(f"\n{'=' * 70}\nPERFORMANCE\n{'=' * 70}")
    print(f"Verifier model load time: {verifier_load_seconds:.2f}s")
    print(f"Average verifier inference latency (development split): {per_question_ms:.1f}ms/question")
    print(f"Total verifier calls across all splits: {n_total}")
    malformed_total, n_all, rate_all = malformed_rate(all_records)
    print(f"Malformed-output rate (all splits combined): {malformed_total}/{n_all} = {rate_all:.2%}")

    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")
    except ImportError:
        print("Peak RSS: unavailable on this platform")


if __name__ == "__main__":
    main()
