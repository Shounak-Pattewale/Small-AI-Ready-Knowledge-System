"""EXPERIMENTAL ONLY: Combined QA Gate + Granite Sentence Answer - Corpus V2.

Architecture under test:

    question -> Granite Top-1 chunk -> MiniLM QA (ANSWERABILITY GATE ONLY)
             -> gate FAIL -> abstain
             -> gate PASS -> split the SAME chunk into sentences
                          -> rank with the SAME already-loaded Granite model
                          -> Top-1 complete sentence = employee-facing answer

MiniLM's extracted SPAN is never shown to the user here - only its signal
is used, exactly as production's frozen gate does. Sentence selection
reuses evaluation/_sentence_selection.py unmodified (already validated in
the prior sentence-selection experiment). QA_THRESHOLD is read-only
imported from knowledge_system.service - never retuned, never searched.

Same-evidence rule: ONE Granite Top-1 retrieval per question feeds the
gate, the sentence split, and the source metadata - never re-retrieved
per component.

This script evaluates an experimental architecture; it does not run
inside KnowledgeService, and nothing in src/knowledge_system/ is modified.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_combined_pipeline.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _answerability`/`_shared`/`_sentence_selection` work either way

import numpy as np

import _answerability as ans
import _sentence_selection as sents
import _shared
from knowledge_system.qa import ExtractiveQA
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.service import QA_THRESHOLD  # read-only import of the frozen production constant
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"
_BLIND_PATH = Path(__file__).resolve().parent.parent / "datasets" / "fresh_blind_test_questions.json"

# Same fixed 14-question diagnostic set as the two prior experiments.
DIAGNOSTIC_QUESTIONS = [
    {"id": "q01", "question": "Can I carry unused annual leave into next year?",
     "expected_source": "holiday_leave_policy.pdf", "expected_topic": "Carry-over", "expected_answerability": "supported"},
    {"id": "q02", "question": "I think someone stole my work laptop. What should I do immediately?",
     "expected_source": "it_support_security.md", "expected_topic": "Lost or stolen devices", "expected_answerability": "supported"},
    {"id": "q03", "question": "Who do I contact if my laptop goes missing?",
     "expected_source": "it_support_security.md", "expected_topic": "Lost or stolen devices", "expected_answerability": "supported"},
    {"id": "q04", "question": "I received a suspicious email asking me to reset my password. What should I do?",
     "expected_source": "it_support_security.md", "expected_topic": "Phishing", "expected_answerability": "supported"},
    {"id": "q05", "question": "Can I work remotely from another country?",
     "expected_source": "flexible_remote_working.pdf", "expected_topic": "Overseas remote work", "expected_answerability": "supported"},
    {"id": "q06", "question": "How do I apply for an internal job vacancy?",
     "expected_source": "career_development.pdf", "expected_topic": "Internal vacancies", "expected_answerability": "supported"},
    {"id": "q07", "question": "Will mentoring guarantee me a promotion?",
     "expected_source": "career_development.pdf", "expected_topic": "Mentoring", "expected_answerability": "supported"},
    {"id": "q08", "question": "Can the company pay for a professional certification?",
     "expected_source": "learning_training.md", "expected_topic": "Professional certifications", "expected_answerability": "supported"},
    {"id": "q09", "question": "Exactly how much money can I claim for a professional certification?",
     "expected_source": "learning_training.md", "expected_topic": "Professional certifications", "expected_answerability": "no_exact_value_in_corpus"},
    {"id": "q10", "question": "Can I book first-class train travel for a business trip?",
     "expected_source": "expenses_business_travel.md", "expected_topic": "Rail travel", "expected_answerability": "supported"},
    {"id": "q11", "question": "What is the deadline for submitting a travel expense?",
     "expected_source": "expenses_business_travel.md", "expected_topic": "Submitting claims", "expected_answerability": "no_exact_value_in_corpus"},
    {"id": "q12", "question": "Does the company provide private health insurance?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
    {"id": "q13", "question": "How much is the employee referral cash bonus?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
    {"id": "q14", "question": "Do employees get free gym memberships?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
]


def qa_gate_passed(qa_result) -> bool:
    """Exact mirror of KnowledgeService.ask()'s frozen gate: signal >= QA_THRESHOLD AND non-blank
    extracted text (the same deterministic whitespace-answer safeguard production already has)."""
    return qa_result.signal >= QA_THRESHOLD and bool(qa_result.text.strip())


@dataclass
class CombinedResult:
    question: str
    result: object  # SearchResult
    qa_result: object  # QAAnswer
    gate_passed: bool
    sentences: list[str]
    ranked: list  # list[RankedSentence]
    context_sentences: list[str]
    combined_answered: bool
    combined_answer: str | None
    retrieval_seconds: float
    qa_seconds: float
    split_seconds: float
    rank_seconds: float


def evaluate_one(question_text: str, retriever: EmbeddingRetriever, qa_model: ExtractiveQA, granite_model) -> CombinedResult:
    """The full combined pipeline for ONE question - ONE retrieval call feeds every component below."""
    t0 = time.monotonic()
    result = retriever.search(question_text, top_k=1)[0]
    retrieval_seconds = time.monotonic() - t0

    t1 = time.monotonic()
    qa_result = qa_model.answer(question_text, result.chunk.text)  # SAME chunk text as the gate input
    qa_seconds = time.monotonic() - t1
    gate_passed = qa_gate_passed(qa_result)

    t2 = time.monotonic()
    sentences = sents.split_sentences(result.chunk.text)  # SAME chunk text again - sentence splitting
    split_seconds = time.monotonic() - t2

    t3 = time.monotonic()
    question_embedding = np.asarray(granite_model.encode([question_text], normalize_embeddings=True))[0]
    if sentences:
        sentence_embeddings = np.asarray(granite_model.encode(sentences, normalize_embeddings=True))
        ranked = sents.rank_sentences(sentences, sentence_embeddings, question_embedding)
        context_sentences = sents.local_context(sentences, ranked[0].index)
    else:
        ranked, context_sentences = [], []
    rank_seconds = time.monotonic() - t3

    combined_answered = gate_passed and bool(ranked)
    combined_answer = ranked[0].text if combined_answered else None

    return CombinedResult(
        question=question_text, result=result, qa_result=qa_result, gate_passed=gate_passed,
        sentences=sentences, ranked=ranked, context_sentences=context_sentences,
        combined_answered=combined_answered, combined_answer=combined_answer,
        retrieval_seconds=retrieval_seconds, qa_seconds=qa_seconds,
        split_seconds=split_seconds, rank_seconds=rank_seconds,
    )


# ---------- Part A: formal answerability evaluation ----------


def run_formal_split(label: str, questions: list[dict], retriever, qa_model, granite_model) -> tuple[list[dict], ans.Confusion, int]:
    """Runs the combined pipeline over one formal split. Returns (per-question records, confusion, mismatches)."""
    records = []
    mismatches = 0
    for q in questions:
        outcome = evaluate_one(q["question"], retriever, qa_model, granite_model)
        # Independent re-derivation of the QA-only baseline decision (separate code path,
        # not a call to qa_gate_passed()) - genuinely verifies equivalence, not just asserts it.
        baseline_decision = ans.classify(outcome.qa_result.signal, QA_THRESHOLD) and outcome.qa_result.text.strip() != ""
        if baseline_decision != outcome.gate_passed:
            mismatches += 1
        records.append({"q": q, "outcome": outcome, "baseline_decision": baseline_decision})

    labels = [r["q"]["answerable"] for r in records]
    predictions = [r["outcome"].gate_passed for r in records]
    confusion = ans.confusion_counts(labels, predictions)

    print(f"\n{'=' * 70}\n{label} (n={len(records)})\n{'=' * 70}")
    accuracy = (confusion.true_accept + confusion.true_abstain) / len(records)
    print(f"True Answer={confusion.true_accept}  False Abstain={confusion.false_abstain}  "
          f"True Abstain={confusion.true_abstain}  False Accept={confusion.false_accept}")
    print(f"Answerable Recall={confusion.answerable_recall:.2%}  Unanswerable Rejection={confusion.unanswerable_rejection_rate:.2%}  "
          f"Balanced Accuracy={confusion.balanced_accuracy:.2%}  Plain Accuracy={accuracy:.2%}")
    print(f"Combined vs QA-only-baseline decision mismatches: {mismatches} (expected 0)")

    return records, confusion, mismatches


def print_false_accepts(label: str, records: list[dict]) -> None:
    print(f"\n{'-' * 70}\n{label}: FALSE ACCEPTS (unanswerable questions the gate passed)\n{'-' * 70}")
    false_accepts = [r for r in records if not r["q"]["answerable"] and r["outcome"].gate_passed]
    if not false_accepts:
        print("(none)")
        return
    for r in false_accepts:
        q, o = r["q"], r["outcome"]
        print(
            f"\nQuestion: {q['question']}\n"
            f"Expected answerability: unsupported (negative_type={q.get('negative_type', '(none)')})\n"
            f"Retrieved source: {o.result.chunk.source}\n"
            f"Retrieved section: {o.result.chunk.section}\n"
            f"QA span: {o.qa_result.text!r}\n"
            f"QA signal: {o.qa_result.signal:.4f}\n"
            f"Sentence answer: {o.ranked[0].text!r}\n"
            f"Sentence similarity: {o.ranked[0].similarity:.4f}\n"
            f"Top-1 + local context: {' '.join(o.context_sentences)!r}"
        )


def print_false_abstains(label: str, records: list[dict]) -> None:
    print(f"\n{'-' * 70}\n{label}: FALSE ABSTAINS (answerable questions the gate blocked)\n{'-' * 70}")
    false_abstains = [r for r in records if r["q"]["answerable"] and not r["outcome"].gate_passed]
    if not false_abstains:
        print("(none)")
        return
    for r in false_abstains:
        q, o = r["q"], r["outcome"]
        top_sentence = o.ranked[0].text if o.ranked else "(no sentences in chunk)"
        print(
            f"\nQuestion: {q['question']}\n"
            f"Expected source/topic: {_shared.expected_source_label(q)} / {q.get('expected_topic')}\n"
            f"Retrieved source/topic: {o.result.chunk.source} / {o.result.chunk.section}\n"
            f"QA span: {o.qa_result.text!r}\n"
            f"QA signal: {o.qa_result.signal:.4f}\n"
            f"Top sentence that WOULD have been returned: {top_sentence!r}\n"
            f"Top-1 + local context: {' '.join(o.context_sentences)!r}"
        )


# ---------- Part B: answer presentation (accepted answerable questions) ----------


def print_presentation_comparison(label: str, records: list[dict]) -> None:
    print(f"\n{'-' * 70}\n{label}: PRESENTATION - QA SPAN vs COMBINED SENTENCE (accepted answerable questions)\n{'-' * 70}")
    accepted = [r for r in records if r["q"]["answerable"] and r["outcome"].gate_passed]
    for r in accepted:
        q, o = r["q"], r["outcome"]
        print(
            f"\n{q['id']}: {q['question']}\n"
            f"  source={o.result.chunk.source}  section={o.result.chunk.section}\n"
            f"  QA span={o.qa_result.text!r}  (signal={o.qa_result.signal:.4f})\n"
            f"  sentence answer={o.ranked[0].text!r}  (similarity={o.ranked[0].similarity:.4f})"
        )


# ---------- Part E: 14-question diagnostic ----------


def topic_matched(expected_topic, section) -> str:
    if expected_topic is None:
        return "not-applicable"
    return "yes" if section and expected_topic.lower() in section.lower() else "no"


def source_matched(expected_source, source) -> str:
    if expected_source is None:
        return "not-applicable"
    return "yes" if source == expected_source else "no"


def print_diagnostic_block(index: int, qdef: dict, outcome: CombinedResult) -> None:
    print("=" * 60)
    print(f"QUESTION {qdef['id']}")
    print("=" * 60)
    print(f"\nQuestion:\n{qdef['question']}")
    print(f"\nExpected:\n{qdef['expected_answerability']}")

    print("\nRETRIEVAL\n")
    print(f"Source:\n{outcome.result.chunk.source}")
    print(f"\nSection:\n{outcome.result.chunk.section}")
    print(f"\nPage:\n{outcome.result.chunk.page}")
    print(f"\nGranite section similarity (diagnostic only):\n{outcome.result.score:.4f}")

    print("\n" + "-" * 60)
    print("QA GATE")
    print("-" * 60)
    print(f"\nQA span:\n{outcome.qa_result.text if outcome.qa_result.text.strip() else 'NONE'}")
    print(f"\nQA signal (diagnostic only):\n{outcome.qa_result.signal:.4f}")
    print(f"\nFrozen threshold:\n{QA_THRESHOLD}")
    print(f"\nGate passed:\n{'YES' if outcome.gate_passed else 'NO'}")

    print("\n" + "-" * 60)
    print("SENTENCE SELECTION (diagnostic - shown regardless of gate outcome)")
    print("-" * 60)
    if outcome.ranked:
        print(f"\nTop sentence:\n{outcome.ranked[0].text}")
        print(f"\nSentence similarity (diagnostic only):\n{outcome.ranked[0].similarity:.4f}")
        print(f"\nTop-1 + local context:\n{' '.join(outcome.context_sentences)}")
    else:
        print("\n(no sentences found in retrieved chunk)")

    print("\n" + "-" * 60)
    print("FINAL COMBINED RESULT")
    print("-" * 60)
    print(f"\nAnswered:\n{'YES' if outcome.combined_answered else 'NO'}")
    print(f"\nEmployee-facing answer:\n{outcome.combined_answer if outcome.combined_answered else 'NONE'}")
    print(f"\nSource shown to user:\n{outcome.result.chunk.source if outcome.combined_answered else 'NONE'}")
    print(f"\nSection shown to user:\n{outcome.result.chunk.section if outcome.combined_answered else 'NONE'}")
    print(f"\nPage shown to user:\n{outcome.result.chunk.page if outcome.combined_answered else 'NONE'}")
    print()


def print_diagnostic_summary_table(rows: list[dict]) -> None:
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'ID':<5} {'Expected':<22} {'Source':<26} {'Section':<22} {'Gate':<6} {'QA Span':<20} {'Combined Answer':<28} {'Answered?'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        qa_short = (row["qa_span"][:17] + "...") if len(row["qa_span"]) > 20 else (row["qa_span"] or "NONE")
        ans_short = (row["combined_answer"][:25] + "...") if row["combined_answer"] and len(row["combined_answer"]) > 28 else (row["combined_answer"] or "NONE")
        print(
            f"{row['id']:<5} {row['expected']:<22} {str(row['source']):<26} {str(row['section']):<22} "
            f"{('PASS' if row['gate'] else 'FAIL'):<6} {qa_short:<20} {ans_short:<28} {('YES' if row['answered'] else 'NO')}"
        )


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)

    load_start = time.monotonic()
    retriever = EmbeddingRetriever()  # frozen: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)
    qa_model = ExtractiveQA()  # frozen: deepset/minilm-uncased-squad2, exact production class
    load_seconds = time.monotonic() - load_start

    # Reuse the SAME already-loaded Granite SentenceTransformer the retriever holds - no second model.
    granite_model = retriever._model

    print(f"Model/index load time: {load_seconds:.2f}s")
    print("Granite model reused for both retrieval and sentence ranking - no second instance loaded.\n")

    all_questions = _shared.load_questions(_QUESTIONS_PATH)
    dev_questions = ans.filter_by_split(all_questions, "development")
    heldout_questions = ans.filter_by_split(all_questions, "heldout")
    blind_questions = _shared.load_questions(_BLIND_PATH)

    print(f"Development: {len(dev_questions)}  Heldout: {len(heldout_questions)}  Blind-origin: {len(blind_questions)}")

    # ---------- Part A: formal answerability + decision-equivalence sanity check ----------
    dev_records, dev_confusion, dev_mismatches = run_formal_split("DEVELOPMENT", dev_questions, retriever, qa_model, granite_model)
    heldout_records, heldout_confusion, heldout_mismatches = run_formal_split("HELDOUT", heldout_questions, retriever, qa_model, granite_model)
    blind_records, blind_confusion, blind_mismatches = run_formal_split("BLIND-ORIGIN", blind_questions, retriever, qa_model, granite_model)

    total_mismatches = dev_mismatches + heldout_mismatches + blind_mismatches
    print(f"\n{'=' * 70}\nTOTAL COMBINED vs QA-ONLY-BASELINE DECISION MISMATCHES: {total_mismatches} (expected 0)\n{'=' * 70}")
    if total_mismatches != 0:
        print("STOP: non-zero mismatch count - implementation issue, results below should not be trusted.")

    # ---------- Part C / D: false accepts / false abstains per split ----------
    for label, records in [("DEVELOPMENT", dev_records), ("HELDOUT", heldout_records), ("BLIND-ORIGIN", blind_records)]:
        print_false_accepts(label, records)
        print_false_abstains(label, records)

    # ---------- Part B: presentation comparison per split ----------
    for label, records in [("DEVELOPMENT", dev_records), ("HELDOUT", heldout_records), ("BLIND-ORIGIN", blind_records)]:
        print_presentation_comparison(label, records)

    # ---------- Part E: 14-question diagnostic ----------
    print(f"\n{'=' * 70}\n14-QUESTION NATURAL DIAGNOSTIC SET\n{'=' * 70}")
    diagnostic_start = time.monotonic()
    summary_rows = []
    for index, qdef in enumerate(DIAGNOSTIC_QUESTIONS, start=1):
        outcome = evaluate_one(qdef["question"], retriever, qa_model, granite_model)
        print_diagnostic_block(index, qdef, outcome)
        summary_rows.append(
            {
                "id": qdef["id"], "expected": qdef["expected_answerability"],
                "source": outcome.result.chunk.source, "section": outcome.result.chunk.section,
                "gate": outcome.gate_passed, "qa_span": outcome.qa_result.text,
                "combined_answer": outcome.combined_answer, "answered": outcome.combined_answered,
            }
        )
    diagnostic_seconds = time.monotonic() - diagnostic_start
    print_diagnostic_summary_table(summary_rows)

    # ---------- Part I: performance ----------
    all_records = dev_records + heldout_records + blind_records
    all_outcomes = [r["outcome"] for r in all_records]
    n_formal = len(all_outcomes)

    retrieval_total = sum(o.retrieval_seconds for o in all_outcomes)
    qa_total = sum(o.qa_seconds for o in all_outcomes)
    split_total = sum(o.split_seconds for o in all_outcomes)
    rank_total = sum(o.rank_seconds for o in all_outcomes)

    accepted_outcomes = [o for o in all_outcomes if o.combined_answered]
    abstained_outcomes = [o for o in all_outcomes if not o.combined_answered]
    accepted_total = sum(o.retrieval_seconds + o.qa_seconds + o.split_seconds + o.rank_seconds for o in accepted_outcomes)
    abstained_total = sum(o.retrieval_seconds + o.qa_seconds + o.split_seconds + o.rank_seconds for o in abstained_outcomes)

    print(f"\n{'=' * 70}\nPERFORMANCE (formal splits, n={n_formal})\n{'=' * 70}")
    print(f"Model/index load time: {load_seconds:.2f}s")
    print(f"Granite section retrieval latency: {retrieval_total:.2f}s total, {retrieval_total / n_formal * 1000:.1f}ms/question")
    print(f"QA gate latency: {qa_total:.2f}s total, {qa_total / n_formal * 1000:.1f}ms/question")
    print(f"Sentence splitting latency: {split_total:.3f}s total, {split_total / n_formal * 1000:.2f}ms/question")
    print(f"Sentence embedding+ranking latency: {rank_total:.2f}s total, {rank_total / n_formal * 1000:.1f}ms/question")
    if accepted_outcomes:
        print(f"Total accepted-question latency: {accepted_total:.2f}s total, {accepted_total / len(accepted_outcomes) * 1000:.1f}ms/question ({len(accepted_outcomes)} questions)")
    if abstained_outcomes:
        print(f"Total abstained-question latency: {abstained_total:.2f}s total, {abstained_total / len(abstained_outcomes) * 1000:.1f}ms/question ({len(abstained_outcomes)} questions)")
    print(f"14-question diagnostic total time: {diagnostic_seconds:.2f}s ({diagnostic_seconds / len(DIAGNOSTIC_QUESTIONS) * 1000:.1f}ms/question)")
    accepted_count = sum(1 for r in summary_rows if r["answered"])
    abstained_count = len(summary_rows) - accepted_count
    print(f"14-question set: {accepted_count} accepted, {abstained_count} abstained")
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")
    except ImportError:
        print("Peak RSS: unavailable on this platform")


if __name__ == "__main__":
    main()
