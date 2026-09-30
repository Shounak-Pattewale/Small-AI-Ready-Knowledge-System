"""EXPERIMENT 2 (final NLI experiment): does QA-aware claim construction fix
Experiment 1's WH-question recall loss while keeping NLI's rejection gains?

Not production code, not wired into src/knowledge_system/. Architecture tested:

    question -> Granite Top-1 -> ExtractiveQA candidate answer
             -> build_claim(question, candidate) -> declarative claim
             -> NLI(premise=chunk text, hypothesis=claim)

The frozen production QA system (threshold -5.7906) is recomputed here
only as a same-run comparison baseline - never modified. Experiment 1
(evaluate_nli_answerability.py) is left completely untouched for
historical reproducibility; this script reuses its NLI model class and
verified label mapping rather than duplicating them.

Experimental discipline: NLI-v2 threshold/rule selection happens on
split=="development" only. Heldout and the blind-origin set are scored
once with the frozen rule.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text either contains.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_nli_v2_answerability.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so sibling `_*` imports work either way

import _answerability as ans
import _claim
import _nli as nli
import _shared
from knowledge_system.qa import ExtractiveQA
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"
_BLIND_PATH = Path(__file__).resolve().parent.parent / "datasets" / "fresh_blind_test_questions.json"
_QA_THRESHOLD = -5.7906  # frozen; never touched by this script


@dataclass
class Record:
    q: dict
    top1_score: float
    top1_source: str
    top1_section: str | None
    qa_text: str
    qa_signal: float
    qa_would_pass: bool
    claim: _claim.ClaimResult
    nli_result: nli.NLIResult | None  # None when claim.usable is False


def collect_records(retriever, qa, nli_model, questions: list[dict], timing: dict) -> list[Record]:
    records = []
    for q in questions:
        t0 = time.monotonic()
        result = retriever.search(q["question"], top_k=1)[0]
        timing["retrieval_seconds"] += time.monotonic() - t0
        timing["retrieval_calls"] += 1
        chunk = result.chunk

        t0 = time.monotonic()
        qa_result = qa.answer(q["question"], chunk.text)
        timing["qa_seconds"] += time.monotonic() - t0
        timing["qa_calls"] += 1

        claim = _claim.build_claim(q["question"], qa_result.text)

        nli_result = None
        if claim.usable:
            t0 = time.monotonic()
            nli_result = nli_model.score(chunk.text, claim.claim)
            timing["nli_seconds"] += time.monotonic() - t0
            timing["nli_calls"] += 1

        records.append(
            Record(
                q=q,
                top1_score=result.score,
                top1_source=chunk.source,
                top1_section=chunk.section,
                qa_text=qa_result.text,
                qa_signal=qa_result.signal,
                qa_would_pass=qa_result.signal >= _QA_THRESHOLD,
                claim=claim,
                nli_result=nli_result,
            )
        )
    return records


def print_confusion(label: str, confusion: ans.Confusion) -> None:
    print(f"  {label}: true_accept={confusion.true_accept} false_abstain={confusion.false_abstain} "
          f"true_abstain={confusion.true_abstain} false_accept={confusion.false_accept}")
    print(f"    recall={confusion.answerable_recall:.2%}  rejection={confusion.unanswerable_rejection_rate:.2%}  "
          f"balanced_accuracy={confusion.balanced_accuracy:.2%}")


def coverage_report(name: str, records: list[Record]) -> None:
    total = len(records)
    usable = sum(1 for r in records if r.claim.usable)
    answerable_total = sum(1 for r in records if r.q["answerable"])
    answerable_usable = sum(1 for r in records if r.q["answerable"] and r.claim.usable)
    unanswerable_total = total - answerable_total
    unanswerable_usable = usable - answerable_usable
    print(f"--- Claim coverage: {name} ---")
    print(f"  total={total}  usable={usable}  unusable={total - usable}  coverage={usable / total:.2%}")
    print(f"  answerable: usable {answerable_usable}/{answerable_total} ({answerable_usable / answerable_total:.2%})")
    print(f"  unanswerable: usable {unanswerable_usable}/{unanswerable_total} ({unanswerable_usable / unanswerable_total:.2%})")
    strategies: dict[str, int] = {}
    for r in records:
        strategies[r.claim.strategy] = strategies.get(r.claim.strategy, 0) + 1
    for strat, count in sorted(strategies.items()):
        print(f"    strategy={strat:<28} n={count}")
    print()


def predictions_predicted_label(records: list[Record]) -> list[bool]:
    # unusable claims: predicted_label rule has no basis to accept -> treated as reject (predict False)
    return [bool(r.nli_result) and r.nli_result.predicted_label == "entailment" for r in records]


def predictions_threshold(records: list[Record], threshold: float) -> list[bool]:
    return [bool(r.nli_result) and ans.classify(r.nli_result.entailment, threshold) for r in records]


def predictions_margin(records: list[Record], threshold: float) -> list[bool]:
    def margin(r: Record) -> float:
        if not r.nli_result:
            return float("-inf")
        return r.nli_result.entailment - max(r.nli_result.neutral, r.nli_result.contradiction)
    return [ans.classify(margin(r), threshold) for r in records]


def apply_unusable_policy(base_predictions: list[bool], records: list[Record], policy: str) -> list[bool]:
    """policy: 'abstain' (unusable -> False, already the base default) or 'fallback_qa' (unusable -> frozen QA decision)."""
    if policy == "abstain":
        return base_predictions
    return [
        pred if r.claim.usable else r.qa_would_pass
        for pred, r in zip(base_predictions, records)
    ]


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted Corpus V2 corpus only - never re-parses raw documents

    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded

    qa_load_start = time.monotonic()
    qa = ExtractiveQA()
    qa_load_seconds = time.monotonic() - qa_load_start

    nli_load_start = time.monotonic()
    nli_model = nli.NLIModel()  # SAME model as Experiment 1: cross-encoder/nli-deberta-v3-xsmall
    nli_load_seconds = time.monotonic() - nli_load_start

    print(f"QA model load: {qa_load_seconds:.2f}s")
    print(f"NLI model load: {nli_load_seconds:.2f}s\n")

    questions = _shared.load_questions(_QUESTIONS_PATH)
    dev_questions = ans.filter_by_split(questions, "development")
    heldout_questions = ans.filter_by_split(questions, "heldout")
    blind_questions = _shared.load_questions(_BLIND_PATH)

    timing = {"retrieval_seconds": 0.0, "retrieval_calls": 0, "qa_seconds": 0.0, "qa_calls": 0, "nli_seconds": 0.0, "nli_calls": 0}
    dev_records = collect_records(retriever, qa, nli_model, dev_questions, timing)
    heldout_records = collect_records(retriever, qa, nli_model, heldout_questions, timing)
    blind_records = collect_records(retriever, qa, nli_model, blind_questions, timing)

    # ============================================================
    # Claim coverage
    # ============================================================
    print("=" * 70)
    print("CLAIM COVERAGE ANALYSIS")
    print("=" * 70)
    coverage_report("development", dev_records)
    coverage_report("heldout", heldout_records)
    coverage_report("blind-origin", blind_records)

    # ============================================================
    # Manual claim sanity audit (development)
    # ============================================================
    print("=" * 70)
    print("CLAIM SANITY AUDIT (development, every record)")
    print("=" * 70)
    for r in dev_records:
        print(f"{r.q['id']:<18} answerable={r.q['answerable']!s:<6}")
        print(f"  question: {r.q['question']}")
        print(f"  QA candidate: {r.qa_text!r}  (qa_signal={r.qa_signal:.3f}, would_pass_frozen_threshold={r.qa_would_pass})")
        print(f"  claim [{r.claim.strategy}] usable={r.claim.usable}: {r.claim.claim!r}")
        print()

    dev_labels = [r.q["answerable"] for r in dev_records]

    # ============================================================
    # Development decision-rule analysis
    # ============================================================
    print("=" * 70)
    print("DEVELOPMENT: NLI-v2 decision rule analysis (unusable claims -> abstain by default)")
    print("=" * 70)

    pred_label_predictions = predictions_predicted_label(dev_records)
    pred_label_confusion = ans.confusion_counts(dev_labels, pred_label_predictions)
    print("\nRule A: predicted_label == entailment")
    print_confusion("dev", pred_label_confusion)

    entailment_values = [r.nli_result.entailment if r.nli_result else -1.0 for r in dev_records]
    thr_b, thr_b_confusion = ans.best_single_threshold(entailment_values, dev_labels)
    thr_b_predictions = predictions_threshold(dev_records, thr_b)
    print(f"\nRule B: entailment probability >= {thr_b:.4f} (development-selected)")
    print_confusion("dev", thr_b_confusion)

    margin_values = [
        (r.nli_result.entailment - max(r.nli_result.neutral, r.nli_result.contradiction)) if r.nli_result else -2.0
        for r in dev_records
    ]
    thr_c, thr_c_confusion = ans.best_single_threshold(margin_values, dev_labels)
    thr_c_predictions = predictions_margin(dev_records, thr_c)
    print(f"\nRule C: entailment - max(neutral, contradiction) >= {thr_c:.4f} (development-selected)")
    print_confusion("dev", thr_c_confusion)

    candidates = [
        ("predicted_label", pred_label_confusion, pred_label_predictions, None),
        ("threshold", thr_b_confusion, thr_b_predictions, thr_b),
        ("margin", thr_c_confusion, thr_c_predictions, thr_c),
    ]
    best_name, best_confusion, best_predictions_abstain, best_param = max(candidates, key=lambda c: c[1].balanced_accuracy)
    print(f"\nBest rule on development (abstain policy): {best_name} (balanced_accuracy={best_confusion.balanced_accuracy:.2%})")

    # ============================================================
    # Unusable-claim policy comparison: abstain vs fall back to frozen QA
    # ============================================================
    print("\n" + "=" * 70)
    print("UNUSABLE-CLAIM POLICY COMPARISON (best rule: " + best_name + ")")
    print("=" * 70)
    abstain_predictions = apply_unusable_policy(best_predictions_abstain, dev_records, "abstain")
    abstain_confusion = ans.confusion_counts(dev_labels, abstain_predictions)
    print("\nPolicy 1: unusable claim -> abstain")
    print_confusion("dev", abstain_confusion)

    fallback_predictions = apply_unusable_policy(best_predictions_abstain, dev_records, "fallback_qa")
    fallback_confusion = ans.confusion_counts(dev_labels, fallback_predictions)
    print("\nPolicy 2: unusable claim -> fall back to frozen QA decision")
    print_confusion("dev", fallback_confusion)

    if fallback_confusion.balanced_accuracy >= abstain_confusion.balanced_accuracy:
        selected_policy = "fallback_qa"
        selected_confusion = fallback_confusion
        selected_predictions_dev = fallback_predictions
    else:
        selected_policy = "abstain"
        selected_confusion = abstain_confusion
        selected_predictions_dev = abstain_predictions
    print(f"\nSELECTED UNUSABLE-CLAIM POLICY: {selected_policy}")
    print(f"FROZEN NLI-v2 RULE: {best_name}" + (f" (param={best_param:.4f})" if best_param is not None else "") + f", unusable->{selected_policy}")

    # ============================================================
    # Frozen QA on the SAME development records
    # ============================================================
    qa_predictions_dev = [r.qa_would_pass for r in dev_records]
    qa_confusion_dev = ans.confusion_counts(dev_labels, qa_predictions_dev)
    print("\n--- Frozen QA on the SAME development records (comparison baseline, unmodified) ---")
    print_confusion("QA dev", qa_confusion_dev)

    # ============================================================
    # Targeted false-accept analysis: all 7 known QA false accepts
    # ============================================================
    print("\n" + "=" * 70)
    print("TARGETED FALSE-ACCEPT ANALYSIS: all 7 frozen-QA development false accepts")
    print("=" * 70)
    known_ids = {"unanswerable_01", "unanswerable_07", "unanswerable_08", "unanswerable_09",
                 "unanswerable_11", "unanswerable_12", "unanswerable_14"}
    for r, nli_v2_pred in zip(dev_records, selected_predictions_dev):
        if r.q["id"] in known_ids:
            print(f"\n{r.q['id']}: {r.q['question']}")
            print(f"  QA candidate={r.qa_text!r}  section={r.top1_section}")
            print(f"  claim [{r.claim.strategy}] usable={r.claim.usable}: {r.claim.claim!r}")
            if r.nli_result:
                print(f"  NLI: entail={r.nli_result.entailment:.4f} neutral={r.nli_result.neutral:.4f} "
                      f"contra={r.nli_result.contradiction:.4f} predicted={r.nli_result.predicted_label}")
            print(f"  NLI-v2 decision: {'ACCEPT (wrong)' if nli_v2_pred else 'REJECT (correct)'}")

    # ============================================================
    # Answerable recall analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("EVERY ANSWERABLE DEVELOPMENT QUESTION NLI-v2 REJECTS")
    print("=" * 70)
    for r, nli_v2_pred in zip(dev_records, selected_predictions_dev):
        if r.q["answerable"] and not nli_v2_pred:
            print(f"\n{r.q['id']}: {r.q['question']}")
            print(f"  QA candidate={r.qa_text!r}  source={r.top1_source}  section={r.top1_section}")
            print(f"  claim [{r.claim.strategy}] usable={r.claim.usable}: {r.claim.claim!r}")
            if r.nli_result:
                print(f"  NLI: entail={r.nli_result.entailment:.4f} neutral={r.nli_result.neutral:.4f} "
                      f"contra={r.nli_result.contradiction:.4f} predicted={r.nli_result.predicted_label}")
            else:
                print("  (claim unusable - no NLI score)")

    # ============================================================
    # Heldout / blind-origin regression (frozen rule, no tuning)
    # ============================================================
    def apply_frozen(records):
        if best_name == "predicted_label":
            base = predictions_predicted_label(records)
        elif best_name == "threshold":
            base = predictions_threshold(records, best_param)
        else:
            base = predictions_margin(records, best_param)
        return apply_unusable_policy(base, records, selected_policy)

    print("\n" + "=" * 70)
    print("PREVIOUSLY OBSERVED HELDOUT - REGRESSION COMPARISON ONLY (frozen NLI-v2 rule, no tuning)")
    print("=" * 70)
    heldout_labels = [r.q["answerable"] for r in heldout_records]
    heldout_predictions = apply_frozen(heldout_records)
    heldout_confusion = ans.confusion_counts(heldout_labels, heldout_predictions)
    print_confusion("NLI-v2 heldout", heldout_confusion)

    print("\n" + "=" * 70)
    print("FIXED BLIND-ORIGIN REGRESSION SET (frozen NLI-v2 rule, no tuning)")
    print("=" * 70)
    blind_labels = [r.q["answerable"] for r in blind_records]
    blind_predictions = apply_frozen(blind_records)
    blind_confusion = ans.confusion_counts(blind_labels, blind_predictions)
    print_confusion("NLI-v2 blind-origin", blind_confusion)

    # ============================================================
    # Laptop Q1/Q2 Top-3 diagnostic (inspection only)
    # ============================================================
    print("\n" + "=" * 70)
    print("LAPTOP TOP-3 DIAGNOSTIC (QA candidate -> claim -> NLI, inspection only, no reranking)")
    print("=" * 70)
    for label, question in [
        ("Q1", "I lost my work laptop on the train, what should I do?"),
        ("Q2", "My work laptop was stolen. Who should I report it to?"),
    ]:
        print(f"\n{label}: {question}")
        for rank, result in enumerate(retriever.search(question, top_k=3), start=1):
            c = result.chunk
            qa_result = qa.answer(question, c.text)
            claim = _claim.build_claim(question, qa_result.text)
            print(f"  Rank {rank}  sim={result.score:.4f}  source={c.source}  section={c.section!r}")
            print(f"    QA candidate={qa_result.text!r}  qa_signal={qa_result.signal:.3f}")
            print(f"    claim [{claim.strategy}] usable={claim.usable}: {claim.claim!r}")
            if claim.usable:
                nli_result = nli_model.score(c.text, claim.claim)
                print(f"    NLI: entail={nli_result.entailment:.4f} neutral={nli_result.neutral:.4f} "
                      f"contra={nli_result.contradiction:.4f} predicted={nli_result.predicted_label}")

    # ============================================================
    # Performance
    # ============================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    n_retrieval = timing["retrieval_calls"] or 1
    n_qa = timing["qa_calls"] or 1
    n_nli = timing["nli_calls"] or 1
    print(f"Avg Granite query time: {timing['retrieval_seconds'] / n_retrieval * 1000:.1f}ms  ({timing['retrieval_calls']} calls)")
    print(f"Avg QA inference time: {timing['qa_seconds'] / n_qa * 1000:.1f}ms  ({timing['qa_calls']} calls)")
    print(f"Avg NLI inference time: {timing['nli_seconds'] / n_nli * 1000:.1f}ms  ({timing['nli_calls']} calls, usable claims only)")
    combined = (timing["retrieval_seconds"] / n_retrieval + timing["qa_seconds"] / n_qa + timing["nli_seconds"] / n_nli) * 1000
    print(f"Approx combined per-question time (retrieval+QA+NLI): {combined:.1f}ms")
    print(f"QA model load: {qa_load_seconds:.2f}s")
    print(f"NLI model load: {nli_load_seconds:.2f}s")


if __name__ == "__main__":
    main()
