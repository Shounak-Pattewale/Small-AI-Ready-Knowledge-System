"""EXPERIMENTAL ONLY: Cross-Encoder Reranking - Corpus V2.

Pipeline under test:

    question -> Granite Top-K -> cross-encoder (ms-marco-MiniLM-L6-v2) reranks -> Top-K (reordered)

Pure retrieval-quality experiment - no QA, no answerability logic, in the
primary metrics. QA is only run post-hoc (Step 17) on Q1/Q2 after every
retrieval result is frozen, and never feeds back into reranking.

Reuses the same 40 answerable questions and expected_source/expected_topic
labels evaluate_embeddings.py already scores Granite against (Corpus V2
fixed evaluation, docs/EVALUATION_HISTORY.md Section 18) - no new
questions, no relabeling.

Deliberately does NOT reuse _shared.evaluate_answerable() for the "TopN
hit" fields - that helper bakes in a fixed TOP_K=3, which would silently
mislabel the Top-5 sensitivity pass (Step 14). evaluate_candidate_set()
below is depth-agnostic: it always checks the full candidate list handed
to it, whatever length that is.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_reranker.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _shared`/`_reranker`/`_extractive_qa` work either way

import _extractive_qa as qa
import _reranker
import _shared
from knowledge_system.models import SearchResult
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "questions.json"

_PRIMARY_TOP_K = 3
_SENSITIVITY_TOP_K = 5

_LAPTOP_Q1 = "I lost my work laptop on the train, what should I do?"
_LAPTOP_Q2 = "My work laptop was stolen. Who should I report it to?"

# Diagnostic only - not a benchmark, not used for tuning, not scored against labels.
_PARAPHRASE_DIAGNOSTICS = [
    "Can I get reimbursed for lunch when I'm travelling for work?",
    "How do I get my manager to approve a training course?",
    "What happens if I click a dodgy link in an email?",
    "Am I allowed to work from a coffee shop for a few days?",
    "Do I lose my unused holiday at the end of the year?",
]


def evaluate_candidate_set(question: dict, results: list) -> dict:
    """Depth-agnostic hit evaluation: 'topN' fields check the WHOLE candidate list handed in,
    whatever its length - unlike _shared.evaluate_answerable()'s fixed TOP_K=3."""
    acceptable = set(_shared.acceptable_sources(question))
    expected_topic = question.get("expected_topic")
    top1 = results[0]
    return {
        "id": question["id"],
        "expected_source": _shared.expected_source_label(question),
        "expected_topic": expected_topic,
        "actual_top1_source": top1.chunk.source,
        "actual_top1_section": top1.chunk.section,
        "top1_score": top1.score,
        "top1_source_hit": top1.chunk.source in acceptable,
        "topN_source_hit": any(r.chunk.source in acceptable for r in results),
        "top1_topic_hit": _shared.topic_matches(expected_topic, top1.chunk.section),
        "topN_topic_hit": any(_shared.topic_matches(expected_topic, r.chunk.section) for r in results),
    }


def reranked_results(question: str, granite_results: list, reranker: _reranker.RerankerModel) -> list[SearchResult]:
    """Rerank Granite's candidates and wrap them back into SearchResult (chunk, score=reranker_score)."""
    candidates = _reranker.rerank(question, granite_results, reranker)
    return [SearchResult(chunk=c.chunk, score=c.reranker_score) for c in candidates]


def run_pass(questions: list[dict], retriever: EmbeddingRetriever, reranker: _reranker.RerankerModel, top_k: int) -> tuple[list[dict], list[dict], list[list], list[list], float, float]:
    """One full pass over `questions` at candidate depth `top_k`. Returns
    (original_rows, reranked_rows, granite_results_list, reranked_candidates_list, granite_seconds, rerank_seconds)."""
    original_rows, reranked_rows = [], []
    granite_results_list, reranked_candidates_list = [], []
    granite_seconds = 0.0
    rerank_seconds = 0.0
    for question in questions:
        t0 = time.monotonic()
        granite_results = retriever.search(question["question"], top_k=top_k)
        granite_seconds += time.monotonic() - t0

        t1 = time.monotonic()
        reranked = reranked_results(question["question"], granite_results, reranker)
        rerank_seconds += time.monotonic() - t1

        original_rows.append(evaluate_candidate_set(question, granite_results))
        reranked_rows.append(evaluate_candidate_set(question, reranked))
        granite_results_list.append(granite_results)
        reranked_candidates_list.append(reranked)
    return original_rows, reranked_rows, granite_results_list, reranked_candidates_list, granite_seconds, rerank_seconds


def print_metrics(label: str, rows: list[dict], depth: int) -> None:
    n = len(rows)
    top1_source = sum(r["top1_source_hit"] for r in rows) / n
    topN_source = sum(r["topN_source_hit"] for r in rows) / n
    top1_topic = sum(r["top1_topic_hit"] for r in rows) / n
    topN_topic = sum(r["topN_topic_hit"] for r in rows) / n
    print(f"\n--- {label} (n={n}, depth={depth}) ---")
    print(f"  Source Top-1: {top1_source:.2%}   Source Top-{depth}: {topN_source:.2%}")
    print(f"  Topic  Top-1: {top1_topic:.2%}   Topic  Top-{depth}: {topN_topic:.2%}")


def rank_movement(original_rows: list[dict], reranked_rows: list[dict], field: str) -> dict:
    """Count improved/unchanged/worsened for one Top-1 hit field ('top1_source_hit' or 'top1_topic_hit')."""
    counts = {"improved": 0, "unchanged": 0, "worsened": 0}
    for orig, rer in zip(original_rows, reranked_rows):
        counts[_reranker.classify_change(orig[field], rer[field])] += 1
    return counts


def print_rank_movement(label: str, counts: dict) -> None:
    net = counts["improved"] - counts["worsened"]
    print(f"{label}: improved={counts['improved']}  unchanged={counts['unchanged']}  worsened={counts['worsened']}  net={net:+d}")


def print_changed_top1_report(questions: list[dict], original_rows: list[dict], reranked_rows: list[dict]) -> int:
    print(f"\n{'=' * 70}\nCHANGED TOP-1 RANKINGS (Granite Top-1 chunk != reranked Top-1 chunk)\n{'=' * 70}")
    changed = 0
    for question, orig, rer in zip(questions, original_rows, reranked_rows):
        if (orig["actual_top1_source"], orig["actual_top1_section"]) == (rer["actual_top1_source"], rer["actual_top1_section"]):
            continue
        changed += 1
        source_change = _reranker.classify_change(orig["top1_source_hit"], rer["top1_source_hit"])
        topic_change = _reranker.classify_change(orig["top1_topic_hit"], rer["top1_topic_hit"])
        print(
            f"\n{question['id']}: {question['question']}\n"
            f"  expected_source={orig['expected_source']}  expected_topic={orig['expected_topic']}\n"
            f"  ORIGINAL: source={orig['actual_top1_source']}  section={orig['actual_top1_section']}  granite_score={orig['top1_score']:.4f}\n"
            f"  RERANKED: source={rer['actual_top1_source']}  section={rer['actual_top1_section']}  reranker_score={rer['top1_score']:.4f}\n"
            f"  source_change={source_change}  topic_change={topic_change}"
        )
    print(f"\nTotal Top-1 rankings changed: {changed}/{len(questions)}")
    return changed


def granite_failure_analysis(questions: list[dict], original_rows: list[dict], reranked_rows: list[dict]) -> tuple[int, int, int]:
    print(f"\n{'=' * 70}\nGRANITE TOPIC FAILURES FIXABLE FROM TOP-3\n{'=' * 70}")
    fixable, fixed, unfixable = 0, 0, 0
    for question, orig, rer in zip(questions, original_rows, reranked_rows):
        if orig["top1_topic_hit"]:
            continue  # not a Granite Top-1 topic failure
        if orig["topN_topic_hit"]:
            fixable += 1
            outcome = "FIXED" if rer["top1_topic_hit"] else "NOT FIXED"
            if rer["top1_topic_hit"]:
                fixed += 1
            print(
                f"{question['id']:<18} expected_topic={orig['expected_topic']!r}\n"
                f"  granite_top1_section={orig['actual_top1_section']}  granite_top1_score={orig['top1_score']:.4f}\n"
                f"  reranked_top1_section={rer['actual_top1_section']}  reranked_top1_score={rer['top1_score']:.4f}  -> {outcome}"
            )
        else:
            unfixable += 1
    print(f"\nFixable from Top-3 (expected topic present but not ranked 1st): {fixable}")
    print(f"Actually fixed by reranker: {fixed}")
    print(f"Unfixable - expected topic NOT in Granite Top-3 (candidate-retrieval failure, not a reranker failure): {unfixable}")
    return fixable, fixed, unfixable


def manual_diagnostic(label: str, question: str, retriever: EmbeddingRetriever, reranker: _reranker.RerankerModel, top_k: int = 3) -> tuple[list, list]:
    print(f"\n--- {label}: {question!r} ---")
    granite_results = retriever.search(question, top_k=top_k)
    print("BEFORE reranking (Granite order):")
    for rank, r in enumerate(granite_results, start=1):
        excerpt = r.chunk.text[:180].replace("\n", " ")
        print(f"  rank={rank} granite_score={r.score:.4f} source={r.chunk.source} section={r.chunk.section}\n    evidence={excerpt!r}")

    candidates = _reranker.rerank(question, granite_results, reranker)
    print("AFTER reranking:")
    for new_rank, c in enumerate(candidates, start=1):
        print(f"  new_rank={new_rank} reranker_score={c.reranker_score:.4f} (was granite_rank={c.granite_rank}, granite_score={c.granite_score:.4f})"
              f" source={c.chunk.source} section={c.chunk.section}")
    return granite_results, candidates


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    reranker_load_start = time.monotonic()
    reranker = _reranker.RerankerModel()  # frozen: cross-encoder/ms-marco-MiniLM-L6-v2
    reranker_load_seconds = time.monotonic() - reranker_load_start

    print(f"Retrieval index load time: {retrieval_load_seconds:.2f}s")
    print(f"Reranker model load time: {reranker_load_seconds:.2f}s")

    all_questions = _shared.load_questions(_QUESTIONS_PATH)
    answerable = [q for q in all_questions if q["answerable"]]
    print(f"Evaluating {len(answerable)} answerable questions (same set/labels as evaluate_embeddings.py)")

    # ---------- Step 6: primary Top-3 experiment ----------
    original_rows, reranked_rows, granite_results_list, reranked_candidates_list, granite_seconds, rerank_seconds = run_pass(
        answerable, retriever, reranker, _PRIMARY_TOP_K
    )

    print(f"\n{'=' * 70}\nPRIMARY RESULTS - Granite Top-{_PRIMARY_TOP_K} vs reranked Top-{_PRIMARY_TOP_K}\n{'=' * 70}")
    print_metrics("Granite (original)", original_rows, _PRIMARY_TOP_K)
    print_metrics("Reranked", reranked_rows, _PRIMARY_TOP_K)

    # ---------- Step 7: rank-movement analysis ----------
    print(f"\n{'=' * 70}\nRANK-MOVEMENT ANALYSIS\n{'=' * 70}")
    source_counts = rank_movement(original_rows, reranked_rows, "top1_source_hit")
    topic_counts = rank_movement(original_rows, reranked_rows, "top1_topic_hit")
    print_rank_movement("SOURCE", source_counts)
    print_rank_movement("TOPIC ", topic_counts)

    # ---------- Step 8: changed-ranking report ----------
    changed_count = print_changed_top1_report(answerable, original_rows, reranked_rows)

    # ---------- Step 9: Granite failure analysis ----------
    fixable, fixed, unfixable = granite_failure_analysis(answerable, original_rows, reranked_rows)

    # ---------- Step 10-11: manual laptop diagnostics ----------
    print(f"\n{'=' * 70}\nMANUAL LAPTOP DIAGNOSTICS (Top-3, no QA yet)\n{'=' * 70}")
    q1_granite, q1_reranked = manual_diagnostic("Q1", _LAPTOP_Q1, retriever, reranker)
    q1_top1_is_lost_stolen = q1_reranked[0].chunk.section is not None and "lost or stolen" in q1_reranked[0].chunk.section.lower()
    print(f"  Did Lost/Stolen Devices reach Q1 rank 1 after reranking? {'YES' if q1_top1_is_lost_stolen else 'NO'}")

    q2_granite, q2_reranked = manual_diagnostic("Q2", _LAPTOP_Q2, retriever, reranker)
    q2_original_top1_is_lost_stolen = q2_granite[0].chunk.section is not None and "lost or stolen" in q2_granite[0].chunk.section.lower()
    q2_reranked_top1_is_lost_stolen = q2_reranked[0].chunk.section is not None and "lost or stolen" in q2_reranked[0].chunk.section.lower()
    print(f"  Q2 Granite Top-1 was already Lost/Stolen Devices: {'YES' if q2_original_top1_is_lost_stolen else 'NO'}")
    print(f"  Q2 reranked Top-1 still Lost/Stolen Devices (preserved)? {'YES' if q2_reranked_top1_is_lost_stolen else 'NO'}")

    # ---------- Step 12: natural paraphrase diagnostics ----------
    print(f"\n{'=' * 70}\nNATURAL PARAPHRASE DIAGNOSTICS (diagnostic only, not scored/tuned)\n{'=' * 70}")
    for question in _PARAPHRASE_DIAGNOSTICS:
        manual_diagnostic("Paraphrase", question, retriever, reranker)

    # ---------- Step 14: Top-5 sensitivity check ----------
    print(f"\n{'=' * 70}\nTOP-{_SENSITIVITY_TOP_K} SENSITIVITY CHECK (diagnostic only)\n{'=' * 70}")
    original_rows_5, reranked_rows_5, _, _, granite_seconds_5, rerank_seconds_5 = run_pass(
        answerable, retriever, reranker, _SENSITIVITY_TOP_K
    )
    print_metrics(f"Granite Top-{_SENSITIVITY_TOP_K} (original)", original_rows_5, _SENSITIVITY_TOP_K)
    print_metrics(f"Reranked from Top-{_SENSITIVITY_TOP_K}", reranked_rows_5, _SENSITIVITY_TOP_K)
    unfixable_top3 = sum(1 for r in original_rows if not r["topN_topic_hit"])
    unfixable_top5 = sum(1 for r in original_rows_5 if not r["topN_topic_hit"])
    top5_regressions = rank_movement(original_rows_5, reranked_rows_5, "top1_topic_hit")
    print(f"Candidate-retrieval misses (expected topic outside candidate set): Top-{_PRIMARY_TOP_K}={unfixable_top3}  Top-{_SENSITIVITY_TOP_K}={unfixable_top5}")
    print_rank_movement(f"TOPIC (Top-{_SENSITIVITY_TOP_K})", top5_regressions)
    print(f"Top-{_SENSITIVITY_TOP_K} pass total time: {granite_seconds_5 + rerank_seconds_5:.2f}s (Granite: {granite_seconds_5:.2f}s, reranking: {rerank_seconds_5:.2f}s)")

    # ---------- Step 15: performance ----------
    n = len(answerable)
    print(f"\n{'=' * 70}\nPERFORMANCE\n{'=' * 70}")
    print(f"Reranker model load time: {reranker_load_seconds:.2f}s")
    print(f"Granite retrieval latency (Top-{_PRIMARY_TOP_K} pass, {n} questions): {granite_seconds:.2f}s total, {granite_seconds / n * 1000:.1f}ms/question")
    print(f"Reranking latency (Top-{_PRIMARY_TOP_K}, {n} questions, {n * _PRIMARY_TOP_K} candidate scores): {rerank_seconds:.2f}s total, "
          f"{rerank_seconds / n * 1000:.1f}ms/question, {rerank_seconds / (n * _PRIMARY_TOP_K) * 1000:.1f}ms/candidate")
    print(f"Total retrieval+reranking (Top-{_PRIMARY_TOP_K}): {(granite_seconds + rerank_seconds) / n * 1000:.1f}ms/question")
    print(f"Reranking latency (Top-{_SENSITIVITY_TOP_K}, {n} questions, {n * _SENSITIVITY_TOP_K} candidate scores): {rerank_seconds_5:.2f}s total, {rerank_seconds_5 / n * 1000:.1f}ms/question")
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")
    except ImportError:
        print("Peak RSS: unavailable on this platform")

    # ---------- Step 17: post-hoc QA diagnostic (Q1/Q2 only, after retrieval frozen) ----------
    print(f"\n{'=' * 70}\nPOST-HOC QA DIAGNOSTIC (Q1/Q2 only, diagnostic - does not affect retrieval metrics above)\n{'=' * 70}")
    qa_model = qa.ExtractiveQAModel()  # frozen: deepset/minilm-uncased-squad2, unmodified
    for label, question, granite_results, reranked_candidates in [
        ("Q1", _LAPTOP_Q1, q1_granite, q1_reranked),
        ("Q2", _LAPTOP_Q2, q2_granite, q2_reranked),
    ]:
        print(f"\n--- {label}: {question!r} ---")
        original_qa = qa_model.answer(question, granite_results[0].chunk.text)
        reranked_qa = qa_model.answer(question, reranked_candidates[0].chunk.text)
        print(f"  QA on ORIGINAL Granite Top-1 (section={granite_results[0].chunk.section}): answer={original_qa.answer_text!r}  signal={original_qa.signal:.3f}")
        print(f"  QA on RERANKED Top-1        (section={reranked_candidates[0].chunk.section}): answer={reranked_qa.answer_text!r}  signal={reranked_qa.signal:.3f}")


if __name__ == "__main__":
    main()
