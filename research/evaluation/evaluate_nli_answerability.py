"""EXPERIMENT ONLY: does NLI provide a better evidence-support signal than the frozen QA signal?

Not production code, not wired into src/knowledge_system/. Uses the
existing persisted Corpus V2 Granite index (no document re-embedding) to
fetch each question's Top-1 chunk, then scores (premise=chunk text,
hypothesis=question_to_hypothesis(question)) with a small NLI model.

The existing frozen QA system (threshold -5.7906) is NOT modified and is
recomputed here only as a same-run, same-retrieval comparison baseline -
never as something this script tunes.

Experimental discipline: NLI threshold/rule selection happens on
split=="development" only. Heldout and the blind-origin set are scored
once with the frozen rule and reported honestly, never used to adjust it.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text either contains.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_nli_answerability.py
"""

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so sibling `_*` imports work either way

import _answerability as ans
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
    hypothesis: nli.Hypothesis
    nli_result: nli.NLIResult
    qa_text: str
    qa_signal: float


def collect_records(retriever: EmbeddingRetriever, qa: ExtractiveQA, nli_model: nli.NLIModel, questions: list[dict], timing: dict) -> list[Record]:
    records = []
    for q in questions:
        result = retriever.search(q["question"], top_k=1)[0]
        chunk = result.chunk

        hyp = nli.question_to_hypothesis(q["question"])

        start = time.monotonic()
        nli_result = nli_model.score(chunk.text, hyp.text)
        timing["nli_calls"] += 1
        timing["nli_seconds"] += time.monotonic() - start

        qa_result = qa.answer(q["question"], chunk.text)

        records.append(
            Record(
                q=q,
                top1_score=result.score,
                top1_source=chunk.source,
                top1_section=chunk.section,
                hypothesis=hyp,
                nli_result=nli_result,
                qa_text=qa_result.text,
                qa_signal=qa_result.signal,
            )
        )
    return records


def print_confusion(label: str, confusion: ans.Confusion) -> None:
    print(f"  {label}: true_accept={confusion.true_accept} false_abstain={confusion.false_abstain} "
          f"true_abstain={confusion.true_abstain} false_accept={confusion.false_accept}")
    print(f"    recall={confusion.answerable_recall:.2%}  rejection={confusion.unanswerable_rejection_rate:.2%}  "
          f"balanced_accuracy={confusion.balanced_accuracy:.2%}")


def apply_predicted_label_rule(records: list[Record]) -> list[bool]:
    return [r.nli_result.predicted_label == "entailment" for r in records]


def apply_threshold_rule(records: list[Record], threshold: float) -> list[bool]:
    return [ans.classify(r.nli_result.entailment, threshold) for r in records]


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted Corpus V2 corpus only - never re-parses raw documents

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    qa_load_start = time.monotonic()
    qa = ExtractiveQA()  # frozen production QA component, unmodified, recomputed here only for comparison
    qa_load_seconds = time.monotonic() - qa_load_start

    nli_load_start = time.monotonic()
    nli_model = nli.NLIModel()  # cross-encoder/nli-deberta-v3-xsmall
    nli_load_seconds = time.monotonic() - nli_load_start

    print(f"Retrieval index load: {retrieval_load_seconds:.2f}s")
    print(f"QA model load: {qa_load_seconds:.2f}s")
    print(f"NLI model load: {nli_load_seconds:.2f}s\n")

    questions = _shared.load_questions(_QUESTIONS_PATH)
    dev_questions = ans.filter_by_split(questions, "development")
    heldout_questions = ans.filter_by_split(questions, "heldout")
    blind_questions = _shared.load_questions(_BLIND_PATH)

    timing = {"nli_calls": 0, "nli_seconds": 0.0}
    dev_records = collect_records(retriever, qa, nli_model, dev_questions, timing)
    heldout_records = collect_records(retriever, qa, nli_model, heldout_questions, timing)
    blind_records = collect_records(retriever, qa, nli_model, blind_questions, timing)

    fallback_count = sum(1 for r in dev_records + heldout_records + blind_records if r.hypothesis.used_fallback)
    total_questions = len(dev_records) + len(heldout_records) + len(blind_records)
    print(f"Questions falling back to original-question hypothesis: {fallback_count}/{total_questions}\n")

    dev_labels = [r.q["answerable"] for r in dev_records]

    # ============================================================
    # DEVELOPMENT: predicted-label-only rule vs threshold rule
    # ============================================================
    print("=" * 70)
    print("DEVELOPMENT: NLI decision rule analysis")
    print("=" * 70)

    predicted_label_predictions = apply_predicted_label_rule(dev_records)
    predicted_label_confusion = ans.confusion_counts(dev_labels, predicted_label_predictions)
    print("\nRule A: predicted_label == entailment")
    print_confusion("dev", predicted_label_confusion)

    entailment_scores = [r.nli_result.entailment for r in dev_records]
    threshold, threshold_confusion = ans.best_single_threshold(entailment_scores, dev_labels)
    print(f"\nRule B: entailment probability >= {threshold:.4f} (development-selected)")
    print_confusion("dev", threshold_confusion)

    # Select whichever rule has higher development balanced accuracy - selection happens ONLY here, ONLY on development.
    if threshold_confusion.balanced_accuracy >= predicted_label_confusion.balanced_accuracy:
        selected_rule = ("threshold", threshold)
        selected_confusion = threshold_confusion
        selected_predictions_dev = apply_threshold_rule(dev_records, threshold)
        print(f"\nSELECTED RULE: entailment >= {threshold:.4f} (higher development balanced accuracy)")
    else:
        selected_rule = ("predicted_label", None)
        selected_confusion = predicted_label_confusion
        selected_predictions_dev = predicted_label_predictions
        print("\nSELECTED RULE: predicted_label == entailment (higher development balanced accuracy)")

    # Leave-one-out stability check on the selected rule family, mirroring analyze_answerability.py's method.
    if selected_rule[0] == "threshold":
        loo_predictions = []
        for i in range(len(entailment_scores)):
            train_values = entailment_scores[:i] + entailment_scores[i + 1:]
            train_labels = dev_labels[:i] + dev_labels[i + 1:]
            loo_threshold, _ = ans.best_single_threshold(train_values, train_labels)
            loo_predictions.append(ans.classify(entailment_scores[i], loo_threshold))
        loo_confusion = ans.confusion_counts(dev_labels, loo_predictions)
        print("\nLeave-one-out (threshold rule, refit each fold):")
        print_confusion("dev LOO", loo_confusion)

    # ============================================================
    # Frozen QA comparison, SAME retrieval/premise, same run
    # ============================================================
    qa_predictions_dev = [ans.classify(r.qa_signal, _QA_THRESHOLD) for r in dev_records]
    qa_confusion_dev = ans.confusion_counts(dev_labels, qa_predictions_dev)
    print("\n--- Frozen QA on the SAME development records (comparison baseline, unmodified) ---")
    print_confusion("QA dev", qa_confusion_dev)

    # ============================================================
    # False-accept focus: every QA false accept, what does NLI say?
    # ============================================================
    print("\n" + "=" * 70)
    print("EVERY FROZEN-QA DEVELOPMENT FALSE ACCEPT - NLI VERDICT")
    print("=" * 70)
    for r, qa_pred, nli_pred in zip(dev_records, qa_predictions_dev, selected_predictions_dev):
        if not r.q["answerable"] and qa_pred:
            nli_correct = not nli_pred  # correct if NLI rejects (predicts NOT answerable)
            print(f"\n{r.q['id']} [{r.q.get('negative_type')}]: {r.q['question']}")
            print(f"  source={r.top1_source}  section={r.top1_section}")
            print(f"  hypothesis={r.hypothesis.text!r}  (fallback={r.hypothesis.used_fallback})")
            print(f"  NLI: entail={r.nli_result.entailment:.4f} neutral={r.nli_result.neutral:.4f} "
                  f"contra={r.nli_result.contradiction:.4f} predicted={r.nli_result.predicted_label}")
            print(f"  NLI decision: {'ACCEPT' if nli_pred else 'REJECT'}  ({'CORRECTLY REJECTS' if nli_correct else 'still WRONG'})")

    # ============================================================
    # Answerable-recall check: every answerable dev question NLI rejects
    # ============================================================
    print("\n" + "=" * 70)
    print("EVERY ANSWERABLE DEVELOPMENT QUESTION THE SELECTED NLI RULE REJECTS")
    print("=" * 70)
    for r, nli_pred in zip(dev_records, selected_predictions_dev):
        if r.q["answerable"] and not nli_pred:
            print(f"\n{r.q['id']}: {r.q['question']}")
            print(f"  source={r.top1_source}  section={r.top1_section}")
            print(f"  hypothesis={r.hypothesis.text!r}  (fallback={r.hypothesis.used_fallback})")
            print(f"  NLI: entail={r.nli_result.entailment:.4f} neutral={r.nli_result.neutral:.4f} "
                  f"contra={r.nli_result.contradiction:.4f} predicted={r.nli_result.predicted_label}")

    # ============================================================
    # Head-to-head, development
    # ============================================================
    both_correct, qa_only, nli_only, both_wrong = [], [], [], []
    for r, qa_pred, nli_pred in zip(dev_records, qa_predictions_dev, selected_predictions_dev):
        actual = r.q["answerable"]
        qa_correct = qa_pred == actual
        nli_correct = nli_pred == actual
        if qa_correct and nli_correct:
            both_correct.append(r.q["id"])
        elif qa_correct and not nli_correct:
            qa_only.append(r.q["id"])
        elif nli_correct and not qa_correct:
            nli_only.append(r.q["id"])
        else:
            both_wrong.append(r.q["id"])

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: FROZEN QA vs SELECTED NLI RULE (development)")
    print("=" * 70)
    print(f"Both correct ({len(both_correct)}): {both_correct}")
    print(f"QA correct / NLI wrong ({len(qa_only)}): {qa_only}")
    print(f"NLI correct / QA wrong ({len(nli_only)}): {nli_only}")
    print(f"Both wrong ({len(both_wrong)}): {both_wrong}")

    # ============================================================
    # Heldout regression (frozen rule, no tuning)
    # ============================================================
    def apply_selected(records):
        if selected_rule[0] == "threshold":
            return apply_threshold_rule(records, selected_rule[1])
        return apply_predicted_label_rule(records)

    print("\n" + "=" * 70)
    print("PREVIOUSLY OBSERVED HELDOUT - REGRESSION COMPARISON ONLY (frozen NLI rule, no tuning)")
    print("=" * 70)
    heldout_labels = [r.q["answerable"] for r in heldout_records]
    heldout_predictions = apply_selected(heldout_records)
    heldout_confusion = ans.confusion_counts(heldout_labels, heldout_predictions)
    print_confusion("NLI heldout", heldout_confusion)
    qa_predictions_heldout = [ans.classify(r.qa_signal, _QA_THRESHOLD) for r in heldout_records]
    qa_confusion_heldout = ans.confusion_counts(heldout_labels, qa_predictions_heldout)
    print_confusion("QA heldout (same run, comparison only)", qa_confusion_heldout)

    # ============================================================
    # Fixed blind-origin regression (frozen rule, no tuning)
    # ============================================================
    print("\n" + "=" * 70)
    print("FIXED BLIND-ORIGIN REGRESSION SET (frozen NLI rule, no tuning)")
    print("=" * 70)
    blind_labels = [r.q["answerable"] for r in blind_records]
    blind_predictions = apply_selected(blind_records)
    blind_confusion = ans.confusion_counts(blind_labels, blind_predictions)
    print_confusion("NLI blind-origin", blind_confusion)
    qa_predictions_blind = [ans.classify(r.qa_signal, _QA_THRESHOLD) for r in blind_records]
    qa_confusion_blind = ans.confusion_counts(blind_labels, qa_predictions_blind)
    print_confusion("QA blind-origin (same run, comparison only)", qa_confusion_blind)

    # ============================================================
    # Laptop Q1/Q2 Top-3 diagnostic (NO Top-K selection logic - inspection only)
    # ============================================================
    print("\n" + "=" * 70)
    print("OPTIONAL TOP-3 DIAGNOSTIC: laptop lost/stolen questions (inspection only, no selection logic)")
    print("=" * 70)
    for label, question in [
        ("Q1", "I lost my work laptop on the train, what should I do?"),
        ("Q2", "My work laptop was stolen. Who should I report it to?"),
    ]:
        print(f"\n{label}: {question}")
        hyp = nli.question_to_hypothesis(question)
        print(f"  hypothesis={hyp.text!r} (fallback={hyp.used_fallback})")
        for rank, r in enumerate(retriever.search(question, top_k=3), start=1):
            c = r.chunk
            result = nli_model.score(c.text, hyp.text)
            timing["nli_calls"] += 1
            print(f"  Rank {rank}  sim={r.score:.4f}  source={c.source}  section={c.section!r}")
            print(f"    NLI: entail={result.entailment:.4f} neutral={result.neutral:.4f} "
                  f"contra={result.contradiction:.4f} predicted={result.predicted_label}")

    # ============================================================
    # Performance
    # ============================================================
    avg_ms = (timing["nli_seconds"] / timing["nli_calls"]) * 1000 if timing["nli_calls"] else 0.0
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    print(f"NLI model load time: {nli_load_seconds:.2f}s")
    print(f"NLI inference calls: {timing['nli_calls']}")
    print(f"Average NLI inference time: {avg_ms:.1f}ms")


if __name__ == "__main__":
    main()
