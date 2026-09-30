"""Hybrid (TF-IDF + Granite via RRF) evaluation runner - directly comparable with the other two evaluators.

Loads storage/chunks.json, fits TfidfRetriever against it, loads the
persisted Granite embedding index (storage/embeddings.npy +
embeddings_manifest.json - no document re-encoding), composes both into
a HybridRetriever (k=60), and evaluates the same 30 frozen questions.
Deliberately does NOT call load_directory() (Docling) and does NOT
rebuild the persisted embedding index.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_hybrid.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _shared` works whether run as a script or loaded by path

import _shared
from _hybrid import HybridRetriever  # moved out of src/knowledge_system/retrieval/ - archived, not production
from _tfidf import TfidfRetriever  # moved out of src/knowledge_system/retrieval/ - archived, not production
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"
_K = 60  # fixed default, not tuned in this experiment


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only - never re-parses raw documents

    tfidf_start = time.monotonic()
    lexical = TfidfRetriever()
    lexical.fit(chunks)
    tfidf_fit_seconds = time.monotonic() - tfidf_start

    model_load_start = time.monotonic()
    semantic = EmbeddingRetriever()  # default model: ibm-granite/granite-embedding-small-english-r2
    model_load_seconds = time.monotonic() - model_load_start

    index_load_start = time.monotonic()
    semantic.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    index_load_seconds = time.monotonic() - index_load_start

    hybrid = HybridRetriever(lexical, semantic, chunks, k=_K)

    questions = _shared.load_questions(_QUESTIONS_PATH)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print(f"Loaded {len(chunks)} chunks from {_CHUNKS_PATH}")
    print(f"TF-IDF fit time: {tfidf_fit_seconds:.2f}s")
    print(f"Granite model load time: {model_load_seconds:.2f}s")
    print(f"Persisted index load time ({len(chunks)} chunks, no re-encoding): {index_load_seconds:.2f}s")
    print(f"RRF k = {_K}")
    print(f"Evaluating {len(answerable)} answerable + {len(unanswerable)} unanswerable questions\n")

    eval_start = time.monotonic()

    print("--- Answerable questions ---")
    answerable_rows = []
    for question in answerable:
        results = hybrid.search(question["question"], top_k=_shared.TOP_K)
        row = _shared.evaluate_answerable(question, results)
        answerable_rows.append(row)
        _shared.print_answerable_row(row)

    print("\n--- Unanswerable questions (no source hit/miss judgement) ---")
    unanswerable_scores = []
    for question in unanswerable:
        results = hybrid.search(question["question"], top_k=_shared.TOP_K)
        unanswerable_scores.append(_shared.print_unanswerable_report(question, results))

    eval_seconds = time.monotonic() - eval_start

    print("\n=== Aggregate metrics (RRF score - a ranking score only, not comparable to raw TF-IDF/cosine) ===")
    n = len(answerable_rows)
    top1_accuracy = sum(r["top1_source_hit"] for r in answerable_rows) / n
    top3_accuracy = sum(r["top3_source_hit"] for r in answerable_rows) / n
    avg_top1_score = statistics.mean(r["top1_score"] for r in answerable_rows)
    print(f"Answerable questions: {n}")
    print(f"Top-1 source accuracy: {top1_accuracy:.2%}")
    print(f"Top-3 source accuracy: {top3_accuracy:.2%}")
    print(f"Average Top-1 RRF score: {avg_top1_score:.6f}")

    _shared.print_category_breakdown(answerable_rows)
    _shared.print_topic_diagnostic(answerable_rows)

    scores = [s["score"] for s in unanswerable_scores]
    print("\n--- Unanswerable Top-1 RRF score statistics ---")
    print(f"min={min(scores):.6f}  max={max(scores):.6f}  avg={statistics.mean(scores):.6f}")

    print(f"\nAll {len(questions)} question searches took: {eval_seconds:.2f}s")
    total = tfidf_fit_seconds + model_load_seconds + index_load_seconds + eval_seconds
    print(f"Total (TF-IDF fit + model load + index load + all searches): {total:.2f}s")


if __name__ == "__main__":
    main()
