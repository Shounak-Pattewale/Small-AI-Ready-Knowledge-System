"""FINAL BLIND EVALUATION of the two already-frozen answerability approaches.

This script is deliberately incapable of threshold search: it imports only
`ans.classify` (a static >= comparison) and `ans.confusion_counts` from
_answerability.py - never `best_single_threshold` or `best_two_signal_rule`.
The two rules below are hardcoded exactly as frozen during development
(docs/EVALUATION_HISTORY.md Sections 10 and 16) and are not derived from
anything in this file or from evaluation/fresh_blind_test_questions.json.

Sequence: load frozen config -> load blind questions -> generate
predictions -> compare against labels -> report metrics -> STOP. No
tuning loop exists here by construction.

Reuses, unmodified: EmbeddingRetriever.load_index() (persisted Granite
index, no document re-encoding), evaluation._extractive_qa.ExtractiveQAModel
(same span extraction/masking/max_answer_length as the development
experiment), evaluation._shared for source/topic diagnostics.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_blind_answerability.py
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
_BLIND_PATH = Path(__file__).resolve().parent.parent / "datasets" / "fresh_blind_test_questions.json"

# Frozen exactly as selected on the development set. Not derived from this
# file, not searched for here, not adjustable by any function this script imports.
_COSINE_TOP1_MIN = 0.8591
_COSINE_GAP12_MIN = 0.0049
_QA_THRESHOLD = -5.7906


def load_blind_questions(path: Path) -> list[dict]:
    """Load and validate the blind test set. Fails loudly on any shape it doesn't expect."""
    payload = _shared.load_questions(path)
    if len(payload) != 30:
        raise ValueError(f"expected exactly 30 blind questions, got {len(payload)}")
    for q in payload:
        if q.get("split") != "blind_test":
            raise ValueError(f"question {q.get('id', '?')!r} has split {q.get('split')!r}, expected 'blind_test'")
    answerable_count = sum(1 for q in payload if q["answerable"])
    unanswerable_count = len(payload) - answerable_count
    if answerable_count != 15 or unanswerable_count != 15:
        raise ValueError(f"expected 15 answerable / 15 unanswerable, got {answerable_count} / {unanswerable_count}")
    return payload


def cosine_predict(top1_score: float, top2_score: float) -> bool:
    """Frozen cosine rule: Top-1 >= 0.8591 AND Top1-Top2 gap >= 0.0049. No search, no tuning."""
    gap12 = ans.score_gap(top1_score, top2_score)
    return ans.classify(top1_score, _COSINE_TOP1_MIN) and ans.classify(gap12, _COSINE_GAP12_MIN)


def qa_predict(signal: float) -> bool:
    """Frozen QA rule: signal >= -5.7906. No search, no tuning."""
    return ans.classify(signal, _QA_THRESHOLD)


def head_to_head(
    labels: list[bool], cosine_predictions: list[bool], qa_predictions: list[bool], ids: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Classify each question into (both_correct, cosine_only_correct, qa_only_correct, both_wrong) ID lists."""
    both_correct, cosine_only, qa_only, both_wrong = [], [], [], []
    for actual, cosine_pred, qa_pred, qid in zip(labels, cosine_predictions, qa_predictions, ids):
        cosine_correct = cosine_pred == actual
        qa_correct = qa_pred == actual
        if cosine_correct and qa_correct:
            both_correct.append(qid)
        elif cosine_correct and not qa_correct:
            cosine_only.append(qid)
        elif qa_correct and not cosine_correct:
            qa_only.append(qid)
        else:
            both_wrong.append(qid)
    return both_correct, cosine_only, qa_only, both_wrong


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


def print_negative_type_breakdown(label: str, entries: list[tuple[str, float, bool]]) -> None:
    breakdown = ans.negative_type_breakdown(entries)
    print(f"\n--- Negative-type breakdown: {label} ---")
    for negative_type in sorted(breakdown):
        stats = breakdown[negative_type]
        print(
            f"  {negative_type:<26} n={stats['count']:<3} rejected={stats['rejected']:<3} "
            f"falsely_accepted={stats['falsely_accepted']:<3} rejection_rate={stats['rejection_rate']:.2%}  "
            f"avg_score={stats['avg_top1']:.4f}"
        )


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()  # frozen: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    qa_load_start = time.monotonic()
    qa_model = qa.ExtractiveQAModel()  # frozen: deepset/minilm-uncased-squad2, unmodified
    qa_load_seconds = time.monotonic() - qa_load_start

    questions = load_blind_questions(_BLIND_PATH)
    print(f"Loaded {len(questions)} blind questions (15 answerable, 15 unanswerable) from {_BLIND_PATH}")
    print(f"Retrieval index load time: {retrieval_load_seconds:.2f}s")
    print(f"QA model load time: {qa_load_seconds:.2f}s\n")

    qa_call_count = 0
    eval_start = time.monotonic()

    records = []  # one dict per question with everything needed for every downstream report
    for question in questions:
        results = retriever.search(question["question"], top_k=2)  # Top-2 needed only for the cosine gap; QA uses Top-1 only
        top1, top2 = results[0], results[1]
        gap12 = ans.score_gap(top1.score, top2.score)

        qa_result = qa_model.answer(question["question"], top1.chunk.text)  # Granite Top-1 chunk only
        qa_call_count += 1

        records.append(
            {
                "q": question,
                "top1": top1,
                "top2": top2,
                "gap12": gap12,
                "cosine_predicted": cosine_predict(top1.score, top2.score),
                "qa_result": qa_result,
                "qa_predicted": qa_predict(qa_result.signal),
            }
        )
    eval_seconds = time.monotonic() - eval_start

    print(f"QA inference calls: {qa_call_count} (expected ~1 per question, Top-1 only)")
    print(f"Total blind evaluation time: {eval_seconds:.2f}s")

    labels = [r["q"]["answerable"] for r in records]
    cosine_predictions = [r["cosine_predicted"] for r in records]
    qa_predictions = [r["qa_predicted"] for r in records]

    cosine_confusion = ans.confusion_counts(labels, cosine_predictions)
    qa_confusion = ans.confusion_counts(labels, qa_predictions)

    print("\n" + "=" * 70)
    print("BLIND RESULTS - FROZEN RULES, NO TUNING")
    print("=" * 70)
    print(f"Cosine rule: Top-1 >= {_COSINE_TOP1_MIN}  AND  Top1-Top2 gap >= {_COSINE_GAP12_MIN}")
    print_confusion("Cosine", cosine_confusion, len(records))
    print(f"\nQA rule: signal >= {_QA_THRESHOLD}")
    print_confusion("QA", qa_confusion, len(records))
    print(f"\nAbsolute balanced-accuracy difference (QA - cosine): "
          f"{(qa_confusion.balanced_accuracy - cosine_confusion.balanced_accuracy) * 100:.2f} points")

    # --- Retrieval diagnostic (answerable questions only) ---
    answerable_records = [r for r in records if r["q"]["answerable"]]
    source_hits = sum(1 for r in answerable_records if r["top1"].chunk.source in _shared.acceptable_sources(r["q"]))
    topic_hits = sum(1 for r in answerable_records if _shared.topic_matches(r["q"].get("expected_topic"), r["top1"].chunk.section))
    print("\n" + "=" * 70)
    print("ANSWERABLE RETRIEVAL DIAGNOSTIC (does not affect answerability predictions)")
    print("=" * 70)
    print(f"Expected-source Top-1 accuracy: {source_hits}/{len(answerable_records)} = {source_hits / len(answerable_records):.2%}")
    print(f"Expected-topic Top-1 diagnostic accuracy: {topic_hits}/{len(answerable_records)} = {topic_hits / len(answerable_records):.2%}")

    # --- Negative-type breakdowns ---
    unanswerable_records = [r for r in records if not r["q"]["answerable"]]
    cosine_entries = [(r["q"].get("negative_type") or "(none)", r["top1"].score, r["cosine_predicted"]) for r in unanswerable_records]
    qa_entries = [(r["q"].get("negative_type") or "(none)", r["qa_result"].signal, r["qa_predicted"]) for r in unanswerable_records]
    print("\n" + "=" * 70)
    print("NEGATIVE-TYPE ANALYSIS (unanswerable questions, diagnostic only)")
    print("=" * 70)
    print_negative_type_breakdown("Cosine", cosine_entries)
    print_negative_type_breakdown("QA", qa_entries)

    # --- False accept / false abstain listings ---
    print("\n" + "=" * 70)
    print("COSINE FALSE ACCEPTS / FALSE ABSTAINS")
    print("=" * 70)
    for r in records:
        q, top1 = r["q"], r["top1"]
        if not q["answerable"] and r["cosine_predicted"]:
            print(f"\nFALSE ACCEPT: {q['id']}: {q['question']}")
            print(f"  source={top1.chunk.source}  section={top1.chunk.section}  top1={top1.score:.4f}  gap12={r['gap12']:.4f}")
        elif q["answerable"] and not r["cosine_predicted"]:
            print(f"\nFALSE ABSTAIN: {q['id']}: {q['question']}")
            print(f"  source={top1.chunk.source}  section={top1.chunk.section}  top1={top1.score:.4f}  gap12={r['gap12']:.4f}")

    print("\n" + "=" * 70)
    print("QA FALSE ACCEPTS / FALSE ABSTAINS")
    print("=" * 70)
    for r in records:
        q, top1, qa_result = r["q"], r["top1"], r["qa_result"]
        if not q["answerable"] and r["qa_predicted"]:
            print(f"\nFALSE ACCEPT: {q['id']}: {q['question']}")
            print(f"  source={top1.chunk.source}  section={top1.chunk.section}  similarity={top1.score:.4f}")
            print(f"  answer={qa_result.answer_text!r}  signal={qa_result.signal:.3f}")
        elif q["answerable"] and not r["qa_predicted"]:
            print(f"\nFALSE ABSTAIN: {q['id']}: {q['question']}")
            print(f"  source={top1.chunk.source}  section={top1.chunk.section}  similarity={top1.score:.4f}")
            print(f"  answer={qa_result.answer_text!r}  signal={qa_result.signal:.3f}")

    # --- Head-to-head ---
    ids = [r["q"]["id"] for r in records]
    both_correct, cosine_only, qa_only, both_wrong = head_to_head(labels, cosine_predictions, qa_predictions, ids)

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"Both correct ({len(both_correct)}): {both_correct}")
    print(f"Cosine correct / QA wrong ({len(cosine_only)}): {cosine_only}")
    print(f"QA correct / cosine wrong ({len(qa_only)}): {qa_only}")
    print(f"Both wrong ({len(both_wrong)}): {both_wrong}")


if __name__ == "__main__":
    main()
