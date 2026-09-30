"""EXPERIMENTAL ONLY: Direct Evidence vs Extractive QA - Corpus V2.

Compares two ways of presenting the SAME retrieved chunk to the user:

    A. CURRENT QA:    Granite Top-1 -> ExtractiveQA -> QA_THRESHOLD gate -> answer/abstain
    B. DIRECT EVIDENCE: Granite Top-1 -> the retrieved chunk's text, verbatim

Both branches consume the exact same SearchResult (one retrieval call per
question) - QA never influences which evidence is shown, and this script
never generates, paraphrases, or summarizes anything. Every "answer" B
produces is text already present in storage/chunks.json.

Reuses ExtractiveQA and QA_THRESHOLD from the real production module
(read-only import - nothing here modifies service.py/qa.py) so the QA
branch is exactly what production does today, not a re-implementation.

Fixed 14-question diagnostic set - not a new benchmark, not used for
tuning. expected_source/expected_topic are diagnostic labels only, filled
in only where the corpus is already known (verified against docs/
CORPUS_HISTORY.md and the persisted chunks) - never invented for the
deliberately unsupported questions (12, 13, 14) or the deadline question
(11, where the corpus gives a vague guideline, not an exact deadline).

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_evidence_vs_qa.py
"""

from pathlib import Path

from knowledge_system.qa import ExtractiveQA
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.service import QA_THRESHOLD  # read-only import of the frozen production constant
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"

# Diagnostic only - not a new benchmark, not scored, not used for tuning.
# expected_answerability: "supported" (corpus has a real answer), "unsupported"
# (deliberately not covered by the corpus), or "no_exact_value_in_corpus"
# (the topic exists but the corpus gives only a vague guideline, not the
# exact figure/date the question asks for - verified against data/*.md,*.pdf,
# not assumed).
DIAGNOSTIC_QUESTIONS = [
    {"id": "q01", "question": "Can I carry unused annual leave into next year?",
     "expected_source": "holiday_leave_policy.pdf", "expected_topic": "Carry-over", "expected_answerability": "supported"},
    {"id": "q02", "question": "I think someone stole my work laptop. What should I do immediately?",
     "expected_source": "it_support_security.md", "expected_topic": "Lost or stolen devices", "expected_answerability": "supported"},
    {"id": "q03", "question": "Who do I contact if my laptop goes missing?",
     "expected_source": "it_support_security.md", "expected_topic": "Lost or stolen devices", "expected_answerability": "supported"},
    {"id": "q04", "question": "I received a suspicious email asking me to reset my password. What should I do?",
     "expected_source": "it_support_security.md", "expected_topic": "Phishing and suspicious messages", "expected_answerability": "supported"},
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
     "expected_source": "expenses_business_travel.md", "expected_topic": "Submitting claims",
     # Verified against data/expenses_business_travel.md Section 1: "as soon as practical ... ideally
     # within the same month" - a guideline, not an exact deadline. No exact deadline exists to extract.
     "expected_answerability": "no_exact_value_in_corpus"},
    {"id": "q12", "question": "Does the company provide private health insurance?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
    {"id": "q13", "question": "How much is the employee referral cash bonus?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
    {"id": "q14", "question": "Do employees get free gym memberships?",
     "expected_source": None, "expected_topic": None, "expected_answerability": "unsupported"},
]


def retrieve_once(question: str, retriever: EmbeddingRetriever):
    """Single Granite Top-1 retrieval call - the one SearchResult both branches below consume."""
    return retriever.search(question, top_k=1)[0]


def run_qa_branch(question: str, chunk_text: str, qa_model: ExtractiveQA) -> dict:
    """Branch A: the exact production QA logic (ExtractiveQA + QA_THRESHOLD gate), diagnostic-only wrapper."""
    qa_result = qa_model.answer(question, chunk_text)
    passed = qa_result.signal >= QA_THRESHOLD
    span_present = bool(qa_result.text.strip())
    return {
        "answered": passed and span_present,
        "qa_answer": qa_result.text,
        "qa_signal": qa_result.signal,
        "qa_span_empty": not span_present,
        "qa_passed_threshold": passed,
    }


def run_evidence_branch(result) -> dict:
    """Branch B: the retrieved chunk's text verbatim - no extraction, no generation, no paraphrasing."""
    return {
        "evidence_text": result.chunk.text,
        "source": result.chunk.source,
        "section": result.chunk.section,
        "page": result.chunk.page,
    }


def print_question_block(index: int, question_def: dict, result, qa_branch: dict, evidence_branch: dict) -> None:
    print("=" * 60)
    print(f"QUESTION {index}")
    print("=" * 60)
    print(f"\nQuestion:\n{question_def['question']}")

    print("\nGRANITE RETRIEVAL\n")
    print(f"Source:\n{result.chunk.source}")
    print(f"\nSection:\n{result.chunk.section}")
    print(f"\nPage:\n{result.chunk.page}")
    print(f"\nGranite score (diagnostic only, not a confidence):\n{result.score:.4f}")

    print("\nCURRENT QA\n")
    print(f"Answered:\n{qa_branch['answered']}")
    print(f"\nQA answer:\n{qa_branch['qa_answer'] if qa_branch['qa_answer'] else 'NONE'}")
    print(f"\nQA signal (diagnostic only, not a confidence):\n{qa_branch['qa_signal']:.4f}")
    print(f"\nQA threshold:\n{QA_THRESHOLD}")

    print("\nDIRECT EVIDENCE\n")
    print(evidence_branch["evidence_text"])

    print("\nSTRUCTURAL DIAGNOSTICS\n")
    print(f"QA returned span: {'yes' if qa_branch['qa_answer'].strip() else 'no'}")
    print(f"QA passed threshold: {'yes' if qa_branch['qa_passed_threshold'] else 'no'}")
    print(f"QA span empty: {'yes' if qa_branch['qa_span_empty'] else 'no'}")
    print(f"Retrieved source: {evidence_branch['source']}")
    print(f"Retrieved section: {evidence_branch['section']}")
    print(f"Expected source: {question_def['expected_source']}")
    print(f"Expected topic: {question_def['expected_topic']}")
    print(f"Expected answerability: {question_def['expected_answerability']}")
    print()


def print_summary_table(rows: list[dict]) -> None:
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'ID':<5} {'Retrieved Source':<28} {'Retrieved Section':<32} {'QA Result':<24} {'Passed?':<8} {'Expected Answerability'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        qa_short = (row["qa_answer"][:20] + "...") if len(row["qa_answer"]) > 23 else (row["qa_answer"] or "NONE")
        print(
            f"{row['id']:<5} {str(row['source']):<28} {str(row['section']):<32} "
            f"{qa_short:<24} {('yes' if row['qa_passed_threshold'] else 'no'):<8} {row['expected_answerability']}"
        )


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only, never re-parsed

    retriever = EmbeddingRetriever()  # frozen: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded

    qa_model = ExtractiveQA()  # frozen: deepset/minilm-uncased-squad2, exact production class

    summary_rows = []
    for index, question_def in enumerate(DIAGNOSTIC_QUESTIONS, start=1):
        result = retrieve_once(question_def["question"], retriever)  # ONE retrieval call, shared by both branches
        qa_branch = run_qa_branch(question_def["question"], result.chunk.text, qa_model)
        evidence_branch = run_evidence_branch(result)

        print_question_block(index, question_def, result, qa_branch, evidence_branch)

        summary_rows.append(
            {
                "id": question_def["id"],
                "source": evidence_branch["source"],
                "section": evidence_branch["section"],
                "qa_answer": qa_branch["qa_answer"],
                "qa_passed_threshold": qa_branch["qa_passed_threshold"],
                "expected_answerability": question_def["expected_answerability"],
            }
        )

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
