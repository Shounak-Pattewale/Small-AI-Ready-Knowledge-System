"""Shared scoring/reporting helpers for evaluate_tfidf.py and evaluate_embeddings.py.

Each evaluator still owns its own retriever setup and main() loop - this
module only removes duplicated question-scoring and printing logic so
both evaluators report results in a directly comparable format.
"""

import json
import re
from pathlib import Path

CATEGORIES = ["direct", "paraphrase", "semantic", "ambiguous"]  # unanswerable is reported separately
TOP_K = 3

# Strips a leading section-numbering prefix like "4." or "4.1" before
# comparing a retrieved section heading against an expected_topic value.
_LEADING_NUMBERING_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s*")


def load_questions(path: Path) -> list[dict]:
    """Load evaluation/questions.json's question list.

    Accepts both the current schema ({"schema_version": 2, "questions": [...]})
    and a bare list, so a future schema bump that keeps a top-level "questions"
    array doesn't require touching every caller.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["questions"]
    return payload


def acceptable_sources(question: dict) -> list[str]:
    """A question's acceptable_sources list, or a single-item list built from expected_source."""
    if "acceptable_sources" in question:
        return question["acceptable_sources"]
    return [question["expected_source"]]


def expected_source_label(question: dict) -> str:
    """Human-readable acceptable-source label for the per-question output row."""
    return " | ".join(acceptable_sources(question))


def normalize_topic(text: str) -> str:
    """Strip leading numbering and normalize case/whitespace for topic comparison.

    Deliberately conservative: exact match after normalization only, no
    fuzzy/substring matching - a diagnostic is only trustworthy if it
    doesn't manufacture correctness.
    """
    return _LEADING_NUMBERING_RE.sub("", text).strip().lower()


def topic_matches(expected_topic: str | None, section: str | None) -> bool:
    """Whether a retrieved chunk's section heading matches the question's expected_topic.

    Only the top-level heading (before the first " > ") is compared,
    since expected_topic values in questions.json name top-level policy
    topics, not subsections (verified against the actual persisted
    corpus before implementing this). Returns False - never a guessed
    match - when either side is missing.
    """
    if not expected_topic or not section:
        return False
    top_level_section = section.split(" > ")[0]
    return normalize_topic(top_level_section) == normalize_topic(expected_topic)


def evaluate_answerable(question: dict, results: list) -> dict:
    """Score one answerable question's retrieval results against its acceptable source(s) and expected topic."""
    acceptable = set(acceptable_sources(question))
    expected_topic = question.get("expected_topic")
    top1 = results[0] if results else None

    return {
        "id": question["id"],
        "category": question["category"],
        "expected_source": expected_source_label(question),
        "expected_topic": expected_topic,
        "actual_top1_source": top1.chunk.source if top1 else None,
        "actual_top1_section": top1.chunk.section if top1 else None,
        "top1_score": top1.score if top1 else 0.0,
        "top1_source_hit": bool(top1) and top1.chunk.source in acceptable,
        "top3_source_hit": any(r.chunk.source in acceptable for r in results[:TOP_K]),
        "top1_topic_hit": bool(top1) and topic_matches(expected_topic, top1.chunk.section),
        "top3_topic_hit": any(topic_matches(expected_topic, r.chunk.section) for r in results[:TOP_K]),
    }


def print_answerable_row(row: dict) -> None:
    hit_mark = "HIT " if row["top1_source_hit"] else "MISS"
    top3_mark = "y" if row["top3_source_hit"] else "n"
    topic_mark = "y" if row["top1_topic_hit"] else "n"
    print(
        f"{row['id']:<14} {row['category']:<10} "
        f"expected={row['expected_source']:<45} actual={str(row['actual_top1_source']):<28} "
        f"score={row['top1_score']:.3f} top1={hit_mark} top3={top3_mark} topic_top1={topic_mark}"
    )


def print_category_breakdown(rows: list[dict]) -> None:
    print("\n--- Top-1 / Top-3 source accuracy by category ---")
    for category in CATEGORIES:
        category_rows = [r for r in rows if r["category"] == category]
        if not category_rows:
            continue
        top1_acc = sum(r["top1_source_hit"] for r in category_rows) / len(category_rows)
        top3_acc = sum(r["top3_source_hit"] for r in category_rows) / len(category_rows)
        print(f"{category:<12} n={len(category_rows):<3} top1={top1_acc:.2%}  top3={top3_acc:.2%}")


def print_topic_diagnostic(rows: list[dict]) -> None:
    """Diagnostic-only: does the retrieved section actually match the expected topic, not just the file."""
    with_topic = [r for r in rows if r["expected_topic"]]
    n = len(with_topic)
    missing_section = sum(1 for r in with_topic if r["actual_top1_section"] is None)

    print("\n--- Topic-level diagnostic (retrieved section vs expected_topic; source-level metric unaffected) ---")
    if n == 0:
        print("(no questions had expected_topic)")
        return
    top1_acc = sum(r["top1_topic_hit"] for r in with_topic) / n
    top3_acc = sum(r["top3_topic_hit"] for r in with_topic) / n
    print(f"Top-1 topic accuracy: {top1_acc:.2%}  ({sum(r['top1_topic_hit'] for r in with_topic)}/{n})")
    print(f"Top-3 topic accuracy: {top3_acc:.2%}  ({sum(r['top3_topic_hit'] for r in with_topic)}/{n})")
    if missing_section:
        print(
            f"LIMITATION: {missing_section}/{n} Top-1 results had no section metadata "
            f"(e.g. front-matter chunks) - counted as no topic match, not manufactured correctness."
        )


def print_unanswerable_report(question: dict, results: list) -> dict:
    top1 = results[0] if results else None
    score = top1.score if top1 else 0.0
    source = top1.chunk.source if top1 else None
    section_or_page = (top1.chunk.section or (f"page {top1.chunk.page}" if top1.chunk.page else None)) if top1 else None
    print(
        f"{question['id']:<16} score={score:.3f} source={source} section/page={section_or_page}\n"
        f"    Q: {question['question']}"
    )
    return {"id": question["id"], "score": score}
