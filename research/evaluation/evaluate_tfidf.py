"""Fixed TF-IDF baseline evaluation runner.

Loads the already-persisted chunk corpus (storage/chunks.json) and runs
retrieval against evaluation/questions.json. Deliberately does NOT call
load_directory() or touch Docling - offline ingestion (knowledge_system.ingest)
is a separate, explicit step. Run that first if storage/chunks.json is missing.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_tfidf.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _shared` works whether run as a script or loaded by path

import _shared
from _tfidf import TfidfRetriever  # moved out of src/knowledge_system/retrieval/ - archived, not production
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    retriever = TfidfRetriever()
    retriever.fit(chunks)

    questions = _shared.load_questions(_QUESTIONS_PATH)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print(f"Loaded {len(chunks)} chunks from {_CHUNKS_PATH}")
    print(f"Evaluating {len(answerable)} answerable + {len(unanswerable)} unanswerable questions\n")

    print("--- Answerable questions ---")
    answerable_rows = []
    for question in answerable:
        results = retriever.search(question["question"], top_k=_shared.TOP_K)
        row = _shared.evaluate_answerable(question, results)
        answerable_rows.append(row)
        _shared.print_answerable_row(row)

    print("\n--- Unanswerable questions (no source hit/miss judgement) ---")
    unanswerable_scores = []
    for question in unanswerable:
        results = retriever.search(question["question"], top_k=_shared.TOP_K)
        unanswerable_scores.append(_shared.print_unanswerable_report(question, results))

    print("\n=== Aggregate metrics ===")
    n = len(answerable_rows)
    top1_accuracy = sum(r["top1_source_hit"] for r in answerable_rows) / n
    top3_accuracy = sum(r["top3_source_hit"] for r in answerable_rows) / n
    avg_top1_score = statistics.mean(r["top1_score"] for r in answerable_rows)
    print(f"Answerable questions: {n}")
    print(f"Top-1 source accuracy: {top1_accuracy:.2%}")
    print(f"Top-3 source accuracy: {top3_accuracy:.2%}")
    print(f"Average Top-1 similarity score: {avg_top1_score:.4f}")

    _shared.print_category_breakdown(answerable_rows)
    _shared.print_topic_diagnostic(answerable_rows)

    scores = [s["score"] for s in unanswerable_scores]
    print("\n--- Unanswerable Top-1 score statistics ---")
    print(f"min={min(scores):.4f}  max={max(scores):.4f}  avg={statistics.mean(scores):.4f}")


if __name__ == "__main__":
    main()
