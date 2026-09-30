"""Granite embedding-retrieval evaluation runner - directly comparable with evaluate_tfidf.py.

Loads the already-persisted chunk corpus (storage/chunks.json) AND the
already-persisted document embedding index (storage/embeddings.npy +
manifest) and runs EmbeddingRetriever against evaluation/questions.json.
Deliberately does NOT call load_directory() (Docling) and does NOT
re-encode document chunks - only each question gets encoded here. Build
the index first if missing: venv/bin/python -m knowledge_system.build_embeddings

Run from the project root:
    venv/bin/python research/evaluation/evaluate_embeddings.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _shared` works whether run as a script or loaded by path

import _shared
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    load_start = time.monotonic()
    retriever = EmbeddingRetriever()  # default model: ibm-granite/granite-embedding-small-english-r2
    model_load_seconds = time.monotonic() - load_start

    index_start = time.monotonic()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    index_load_seconds = time.monotonic() - index_start

    questions = _shared.load_questions(_QUESTIONS_PATH)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print(f"Loaded {len(chunks)} chunks from {_CHUNKS_PATH}")
    print(f"Model load time: {model_load_seconds:.2f}s")
    print(f"Persisted index load time ({len(chunks)} chunks, no re-encoding): {index_load_seconds:.2f}s")
    print(f"Evaluating {len(answerable)} answerable + {len(unanswerable)} unanswerable questions\n")

    eval_start = time.monotonic()

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

    eval_seconds = time.monotonic() - eval_start

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

    print(f"\nAll {len(questions)} question searches took: {eval_seconds:.2f}s")
    print(f"Total (model load + index load + all searches): {model_load_seconds + index_load_seconds + eval_seconds:.2f}s")


if __name__ == "__main__":
    main()
