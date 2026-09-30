"""EXPERIMENTAL ONLY: Ollama Cloud Grounded Answerability - Corpus V2.

Architecture under test:

    question -> EXISTING Granite Top-1 retrieval -> gemma4:cloud
             -> structured {"answerable": bool, "answer": str|null}
             -> source metadata from the SAME retrieved chunk

Granite retrieval is unchanged from production/the combined-pipeline
candidate. The LLM receives ONLY the question and the retrieved chunk's
text - never the MiniLM span, never sentence-selection output, never the
expected label, never the split name. MiniLM is run alongside (same
exact frozen gate formula as evaluate_combined_pipeline.py) purely as a
same-question baseline for head-to-head comparison, not as an input to
the LLM.

The system prompt and generation parameters (evaluation/_ollama_grounded_qa.py)
are frozen before this script's formal evaluation loop runs and are never
edited based on development/heldout/blind results - enforced by
discipline, not by code, exactly as every other frozen-configuration
experiment in this project.

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them. The system
prompt itself instructs the model to do the same for EVIDENCE.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_ollama_cloud.py
"""

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _answerability`/`_shared`/`_ollama_grounded_qa` work either way

import _answerability as ans
import _ollama_grounded_qa as ollama
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

# Published MiniLM-only baseline (Sections 21/26) - used only to sanity-check
# that this script's live MiniLM recomputation matches, never to shortcut it.
_PUBLISHED_BASELINE = {
    "development": ans.Confusion(true_accept=29, false_abstain=1, true_abstain=8, false_accept=7),
    "heldout": ans.Confusion(true_accept=10, false_abstain=0, true_abstain=6, false_accept=4),
    "blind-origin": ans.Confusion(true_accept=15, false_abstain=0, true_abstain=7, false_accept=8),
}

DIAGNOSTIC_QUESTIONS = [
    {"id": "q01", "question": "Can I carry unused annual leave into next year?", "expected_answerability": "supported"},
    {"id": "q02", "question": "I think someone stole my work laptop. What should I do immediately?", "expected_answerability": "supported"},
    {"id": "q03", "question": "Who do I contact if my laptop goes missing?", "expected_answerability": "supported"},
    {"id": "q04", "question": "I received a suspicious email asking me to reset my password. What should I do?", "expected_answerability": "supported"},
    {"id": "q05", "question": "Can I work remotely from another country?", "expected_answerability": "supported"},
    {"id": "q06", "question": "How do I apply for an internal job vacancy?", "expected_answerability": "supported"},
    {"id": "q07", "question": "Will mentoring guarantee me a promotion?", "expected_answerability": "supported"},
    {"id": "q08", "question": "Can the company pay for a professional certification?", "expected_answerability": "supported"},
    {"id": "q09", "question": "Exactly how much money can I claim for a professional certification?", "expected_answerability": "no_exact_value_in_corpus"},
    {"id": "q10", "question": "Can I book first-class train travel for a business trip?", "expected_answerability": "supported"},
    {"id": "q11", "question": "What is the deadline for submitting a travel expense?", "expected_answerability": "no_exact_value_in_corpus"},
    {"id": "q12", "question": "Does the company provide private health insurance?", "expected_answerability": "unsupported"},
    {"id": "q13", "question": "How much is the employee referral cash bonus?", "expected_answerability": "unsupported"},
    {"id": "q14", "question": "Do employees get free gym memberships?", "expected_answerability": "unsupported"},
]


def qa_gate_passed(qa_result) -> bool:
    """Exact mirror of KnowledgeService.ask()'s frozen gate (same formula as evaluate_combined_pipeline.py)."""
    return qa_result.signal >= QA_THRESHOLD and bool(qa_result.text.strip())


@dataclass
class LLMOutcome:
    structured: ollama.StructuredResult | None  # None on API failure
    api_error: str | None
    latency_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None

    @property
    def llm_decision(self) -> str:
        """'ANSWER' / 'ABSTAIN' / 'ERROR' (API failure or malformed output - never silently an answer)."""
        if self.api_error is not None or self.structured is None or not self.structured.valid:
            return "ERROR"
        return "ANSWER" if self.structured.answerable else "ABSTAIN"


def call_llm(question: str, evidence: str) -> LLMOutcome:
    """One grounded-QA call. API failures and malformed structured output are both captured here,
    never raised past this point and never silently treated as an answer or an abstention."""
    t0 = time.monotonic()
    try:
        body = ollama.call_ollama_chat_full(question, evidence)
    except ollama.OllamaAPIError as exc:
        latency = time.monotonic() - t0
        return LLMOutcome(structured=None, api_error=str(exc), latency_seconds=latency, prompt_tokens=None, output_tokens=None)
    latency = time.monotonic() - t0

    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    prompt_tokens = body.get("prompt_eval_count") if isinstance(body, dict) else None
    output_tokens = body.get("eval_count") if isinstance(body, dict) else None

    if content is None:
        return LLMOutcome(structured=None, api_error="response missing message.content", latency_seconds=latency,
                           prompt_tokens=prompt_tokens, output_tokens=output_tokens)

    structured = ollama.parse_structured_output(content)
    return LLMOutcome(structured=structured, api_error=None, latency_seconds=latency,
                       prompt_tokens=prompt_tokens, output_tokens=output_tokens)


@dataclass
class QuestionRecord:
    q: dict
    result: object  # SearchResult - the ONE retrieval both MiniLM and the LLM see
    qa_result: object  # QAAnswer (MiniLM baseline, never shown to the LLM)
    minilm_decision: bool  # True = ANSWER
    llm_outcome: LLMOutcome


def evaluate_question(question_text: str, retriever: EmbeddingRetriever, qa_model: ExtractiveQA) -> QuestionRecord:
    result = retriever.search(question_text, top_k=1)[0]  # ONE retrieval - feeds both MiniLM and the LLM
    qa_result = qa_model.answer(question_text, result.chunk.text)  # baseline only, never sent to the LLM
    minilm_decision = qa_gate_passed(qa_result)
    llm_outcome = call_llm(question_text, result.chunk.text)  # ONLY question + retrieved chunk text
    return QuestionRecord(q={"question": question_text}, result=result, qa_result=qa_result,
                           minilm_decision=minilm_decision, llm_outcome=llm_outcome)


# ---------- Smoke test (Part: model availability) ----------


def run_smoke_test() -> bool:
    print("=" * 70)
    print(f"SMOKE TEST - model availability check ({ollama.MODEL_NAME})")
    print("=" * 70)
    question = "How many days of annual leave do employees receive?"
    evidence = "Employees receive 25 days of annual leave."
    try:
        body = ollama.call_ollama_chat_full(question, evidence)
    except ollama.OllamaAPIError as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        return False
    message = body.get("message", {})
    content = message.get("content")
    print(f"Raw response content: {content!r}")
    if content is None:
        print("SMOKE TEST FAILED: no message.content in response")
        return False
    structured = ollama.parse_structured_output(content)
    print(f"Parsed: valid={structured.valid} answerable={structured.answerable} answer={structured.answer!r} error={structured.error}")
    if not structured.valid:
        print("SMOKE TEST FAILED: structured output invalid")
        return False
    if structured.answerable is not True:
        print("SMOKE TEST FAILED: expected answerable=true for a trivially answerable evidence/question pair")
        return False
    print("SMOKE TEST PASSED\n")
    return True


# ---------- Part A: formal split evaluation ----------


def run_formal_split(label: str, questions: list[dict], retriever, qa_model) -> tuple[list[dict], ans.Confusion, ans.Confusion, int]:
    records = []
    errors = 0
    for q in questions:
        record = evaluate_question(q["question"], retriever, qa_model)
        record.q = q  # replace with the full labeled question dict
        records.append(record)
        if record.llm_outcome.llm_decision == "ERROR":
            errors += 1

    labels = [r.q["answerable"] for r in records]
    minilm_predictions = [r.minilm_decision for r in records]
    minilm_confusion = ans.confusion_counts(labels, minilm_predictions)

    scored = [r for r in records if r.llm_outcome.llm_decision != "ERROR"]
    llm_labels = [r.q["answerable"] for r in scored]
    llm_predictions = [r.llm_outcome.llm_decision == "ANSWER" for r in scored]
    llm_confusion = ans.confusion_counts(llm_labels, llm_predictions)

    print(f"\n{'=' * 70}\n{label} (n={len(records)}, LLM scored={len(scored)}, LLM errors={errors})\n{'=' * 70}")
    print(f"MiniLM baseline (live):  TA={minilm_confusion.true_accept} FAb={minilm_confusion.false_abstain} "
          f"TAb={minilm_confusion.true_abstain} FAc={minilm_confusion.false_accept}  "
          f"recall={minilm_confusion.answerable_recall:.2%} rejection={minilm_confusion.unanswerable_rejection_rate:.2%} "
          f"balanced={minilm_confusion.balanced_accuracy:.2%}")
    published = _PUBLISHED_BASELINE[label.lower().replace(" ", "-")]
    matches_published = (
        minilm_confusion.true_accept == published.true_accept and minilm_confusion.false_abstain == published.false_abstain
        and minilm_confusion.true_abstain == published.true_abstain and minilm_confusion.false_accept == published.false_accept
    )
    print(f"Matches published MiniLM baseline: {'YES' if matches_published else 'NO - INVESTIGATE'}")
    print(f"\nOllama LLM (excluding {errors} error(s)):  TA={llm_confusion.true_accept} FAb={llm_confusion.false_abstain} "
          f"TAb={llm_confusion.true_abstain} FAc={llm_confusion.false_accept}  "
          f"recall={llm_confusion.answerable_recall:.2%} rejection={llm_confusion.unanswerable_rejection_rate:.2%} "
          f"balanced={llm_confusion.balanced_accuracy:.2%}")

    return records, minilm_confusion, llm_confusion, errors


def print_head_to_head(label: str, records: list[dict]) -> None:
    scored = [r for r in records if r.llm_outcome.llm_decision != "ERROR"]
    both_correct, minilm_only, llm_only, both_wrong = [], [], [], []
    minilm_false_accept_llm_correct, minilm_false_abstain_llm_correct, minilm_correct_llm_regression = [], [], []

    for r in scored:
        label_true = r.q["answerable"]
        minilm_correct = r.minilm_decision == label_true
        llm_decision_bool = r.llm_outcome.llm_decision == "ANSWER"
        llm_correct = llm_decision_bool == label_true
        qid = r.q.get("id", r.q["question"][:40])

        if minilm_correct and llm_correct:
            both_correct.append(qid)
        elif minilm_correct and not llm_correct:
            minilm_only.append(qid)
            minilm_correct_llm_regression.append(qid)
        elif llm_correct and not minilm_correct:
            llm_only.append(qid)
            if not label_true and r.minilm_decision:  # MiniLM falsely accepted an unanswerable question
                minilm_false_accept_llm_correct.append(qid)
            elif label_true and not r.minilm_decision:  # MiniLM falsely abstained on an answerable question
                minilm_false_abstain_llm_correct.append(qid)
        else:
            both_wrong.append(qid)

    print(f"\n{'-' * 70}\n{label}: HEAD-TO-HEAD (n={len(scored)})\n{'-' * 70}")
    print(f"Both correct: {len(both_correct)}  MiniLM only: {len(minilm_only)}  LLM only: {len(llm_only)}  Both wrong: {len(both_wrong)}")
    print(f"MiniLM false accept -> LLM correct abstain: {minilm_false_accept_llm_correct}")
    print(f"MiniLM false abstain -> LLM correct answer: {minilm_false_abstain_llm_correct}")
    print(f"MiniLM correct -> LLM regression: {minilm_correct_llm_regression}")


def print_errors(label: str, records: list[dict]) -> None:
    errors = [r for r in records if r.llm_outcome.llm_decision == "ERROR"]
    print(f"\n{'-' * 70}\n{label}: LLM ERRORS ({len(errors)})\n{'-' * 70}")
    if not errors:
        print("(none)")
        return
    for r in errors:
        outcome = r.llm_outcome
        if outcome.api_error is not None:
            print(f"{r.q.get('id', '?')}: API ERROR - {outcome.api_error}")
        else:
            print(f"{r.q.get('id', '?')}: MALFORMED OUTPUT - {outcome.structured.error} (raw={outcome.structured.raw!r})")


# ---------- Part: 14-question diagnostic ----------


def print_diagnostic_block(index: int, qdef: dict, record: QuestionRecord) -> None:
    print("=" * 60)
    print(f"QUESTION {qdef['id']}")
    print("=" * 60)
    print(f"\nQuestion:\n{qdef['question']}")
    print(f"\nExpected:\n{qdef['expected_answerability']}")

    print("\nRETRIEVAL\n")
    print(f"Source:\n{record.result.chunk.source}")
    print(f"\nSection:\n{record.result.chunk.section}")
    print(f"\nPage:\n{record.result.chunk.page}")
    print(f"\nGranite similarity (diagnostic only):\n{record.result.score:.4f}")

    print("\n" + "-" * 60)
    print("MINILM BASELINE")
    print("-" * 60)
    print(f"\nQA span:\n{record.qa_result.text if record.qa_result.text.strip() else 'NONE'}")
    print(f"\nQA signal (diagnostic only):\n{record.qa_result.signal:.4f}")
    print(f"\nGate:\n{'PASS' if record.minilm_decision else 'FAIL'}")

    print("\n" + "-" * 60)
    print("OLLAMA CLOUD")
    print("-" * 60)
    outcome = record.llm_outcome
    print(f"\nModel:\n{ollama.MODEL_NAME}")
    if outcome.llm_decision == "ERROR":
        print("\nAnswerable:\nERROR")
        print(f"\nAnswer:\nN/A ({outcome.api_error or (outcome.structured.error if outcome.structured else 'unknown error')})")
        print("\nStructured output valid:\nNO")
    else:
        print(f"\nAnswerable:\n{'TRUE' if outcome.structured.answerable else 'FALSE'}")
        print(f"\nAnswer:\n{outcome.structured.answer if outcome.structured.answer else 'NONE'}")
        print("\nStructured output valid:\nYES")
    print(f"\nLatency:\n{outcome.latency_seconds:.2f}s")

    print("\n" + "-" * 60)
    print("COMPARISON")
    print("-" * 60)
    minilm_decision_label = "ANSWER" if record.minilm_decision else "ABSTAIN"
    llm_decision_label = {"ANSWER": "ANSWER", "ABSTAIN": "ABSTAIN", "ERROR": "ERROR"}[outcome.llm_decision]
    print(f"\nMiniLM decision:\n{minilm_decision_label}")
    print(f"\nLLM decision:\n{llm_decision_label}")
    print(f"\nExpected:\n{qdef['expected_answerability']}")
    expected_answer_bool = qdef["expected_answerability"] != "unsupported"  # supported + no_exact_value_in_corpus both "should attempt"
    minilm_correct = (record.minilm_decision == expected_answer_bool) if qdef["expected_answerability"] != "no_exact_value_in_corpus" else "N/A (nuanced label, see qualitative analysis)"
    print(f"\nCorrectness:\nMiniLM={'n/a (nuanced)' if qdef['expected_answerability'] == 'no_exact_value_in_corpus' else ('correct' if record.minilm_decision == expected_answer_bool else 'incorrect')}  "
          f"LLM={'n/a (nuanced)' if qdef['expected_answerability'] == 'no_exact_value_in_corpus' or outcome.llm_decision == 'ERROR' else ('correct' if (outcome.llm_decision == 'ANSWER') == expected_answer_bool else 'incorrect')}")
    print()


def print_diagnostic_summary_table(rows: list[dict]) -> None:
    print("=" * 60)
    print("14-QUESTION SUMMARY TABLE")
    print("=" * 60)
    header = f"{'ID':<5} {'Expected':<22} {'MiniLM':<8} {'LLM':<8} {'LLM Answer (abridged)':<40}"
    print(header)
    print("-" * len(header))
    for row in rows:
        llm_answer = row["llm_answer"] or "NONE"
        short = (llm_answer[:37] + "...") if len(llm_answer) > 40 else llm_answer
        print(f"{row['id']:<5} {row['expected']:<22} {row['minilm']:<8} {row['llm']:<8} {short:<40}")


def main() -> None:
    if not run_smoke_test():
        print("STOPPING: smoke test failed. Formal evaluation NOT run.")
        return

    chunks = load_chunks(_CHUNKS_PATH)
    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)
    qa_model = ExtractiveQA()

    all_questions = _shared.load_questions(_QUESTIONS_PATH)
    dev_questions = ans.filter_by_split(all_questions, "development")
    heldout_questions = ans.filter_by_split(all_questions, "heldout")
    blind_questions = _shared.load_questions(_BLIND_PATH)

    print(f"Development: {len(dev_questions)}  Heldout: {len(heldout_questions)}  Blind-origin: {len(blind_questions)}")

    latencies: list[float] = []
    prompt_token_total = 0
    output_token_total = 0
    token_data_available = True

    eval_start = time.monotonic()
    all_records = []
    per_split = {}
    for label, questions in [("development", dev_questions), ("heldout", heldout_questions), ("blind-origin", blind_questions)]:
        records, minilm_confusion, llm_confusion, errors = run_formal_split(label.upper(), questions, retriever, qa_model)
        print_head_to_head(label.upper(), records)
        print_errors(label.upper(), records)
        per_split[label] = (records, minilm_confusion, llm_confusion, errors)
        all_records.extend(records)
    eval_seconds = time.monotonic() - eval_start

    for r in all_records:
        latencies.append(r.llm_outcome.latency_seconds)
        if r.llm_outcome.prompt_tokens is None or r.llm_outcome.output_tokens is None:
            token_data_available = False
        else:
            prompt_token_total += r.llm_outcome.prompt_tokens
            output_token_total += r.llm_outcome.output_tokens

    # ---------- 14-question diagnostic ----------
    print(f"\n{'=' * 70}\n14-QUESTION NATURAL DIAGNOSTIC SET\n{'=' * 70}")
    diagnostic_summary_rows = []
    diagnostic_start = time.monotonic()
    for index, qdef in enumerate(DIAGNOSTIC_QUESTIONS, start=1):
        record = evaluate_question(qdef["question"], retriever, qa_model)
        record.q = qdef
        print_diagnostic_block(index, qdef, record)
        latencies.append(record.llm_outcome.latency_seconds)
        if record.llm_outcome.prompt_tokens is not None and record.llm_outcome.output_tokens is not None:
            prompt_token_total += record.llm_outcome.prompt_tokens
            output_token_total += record.llm_outcome.output_tokens
        else:
            token_data_available = False
        llm_answer = None
        if record.llm_outcome.llm_decision == "ANSWER":
            llm_answer = record.llm_outcome.structured.answer
        diagnostic_summary_rows.append({
            "id": qdef["id"], "expected": qdef["expected_answerability"],
            "minilm": "PASS" if record.minilm_decision else "FAIL",
            "llm": record.llm_outcome.llm_decision, "llm_answer": llm_answer,
        })
    diagnostic_seconds = time.monotonic() - diagnostic_start
    print_diagnostic_summary_table(diagnostic_summary_rows)

    # ---------- Latency / usage ----------
    print(f"\n{'=' * 70}\nLATENCY / USAGE\n{'=' * 70}")
    if latencies:
        sorted_lat = sorted(latencies)
        p95_index = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
        print(f"n requests: {len(latencies)}")
        print(f"Mean latency: {statistics.mean(latencies):.2f}s")
        print(f"Median latency: {statistics.median(latencies):.2f}s")
        print(f"P95 latency: {sorted_lat[p95_index]:.2f}s")
    print(f"Formal-split evaluation runtime: {eval_seconds:.2f}s")
    print(f"14-question diagnostic runtime: {diagnostic_seconds:.2f}s")
    print(f"Total runtime (formal + diagnostic): {eval_seconds + diagnostic_seconds:.2f}s")
    if token_data_available:
        print(f"Total prompt/input tokens: {prompt_token_total}")
        print(f"Total output tokens: {output_token_total}")
    else:
        print("Token usage: unavailable for one or more requests")


if __name__ == "__main__":
    main()
