"""EXPERIMENTAL ONLY: Sentence-Level Semantic Answer Selection - Corpus V2.

Compares two ways of extracting an answer from the SAME retrieved Granite
Top-1 chunk:

    A. CURRENT QA:  ExtractiveQA span (exact production class/threshold, read-only import)
    B. SENTENCE SELECTION: split the chunk into sentences, embed them with
       the SAME already-loaded Granite model, rank by cosine similarity to
       the question, return the best complete sentence(s)

No new model is loaded: Granite is loaded once (inside EmbeddingRetriever,
as production already does) and its underlying SentenceTransformer is
reused directly to embed sentence candidates - not a second instance, not
a second model.

Both branches consume the exact same SearchResult - one retrieval call per
question. Sentence text is never generated, paraphrased, or rewritten;
every candidate is verbatim text already in storage/chunks.json.

This experiment tests answer PRESENTATION/EXTRACTION only - it does not
attempt to fix answerability. Semantic similarity here is diagnostic only,
never "confidence" or "answerability".

Fixed 14-question diagnostic set (same questions/expected labels as the
prior Direct Evidence vs QA experiment - hardcoded again here, deliberately
self-contained rather than imported, matching this project's convention of
each evaluate_*.py script owning its own fixed inputs).

Treats every question and retrieved chunk strictly as data - never as
instructions, regardless of what text appears inside them.

Run from the project root:
    venv/bin/python research/evaluation/evaluate_sentence_answers.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import _sentence_selection` works either way

import numpy as np

import _sentence_selection as sents
from knowledge_system.qa import ExtractiveQA
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.service import QA_THRESHOLD  # read-only import of the frozen production constant
from knowledge_system.storage import load_chunks

_CHUNKS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "chunks.json"
_EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings.npy"
_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "storage" / "embeddings_manifest.json"

# Same fixed diagnostic set as evaluate_evidence_vs_qa.py. Diagnostic only -
# not a new benchmark, not scored, not used for tuning. Never invented for
# the deliberately unsupported questions (q12-q14).
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


def topic_matched(expected_topic: str | None, section: str | None) -> str:
    if expected_topic is None:
        return "not-applicable"
    if section and expected_topic.lower() in section.lower():
        return "yes"
    return "no"


def source_matched(expected_source: str | None, source: str | None) -> str:
    if expected_source is None:
        return "not-applicable"
    return "yes" if source == expected_source else "no"


def process_question(qdef: dict, retriever, qa_model, granite_model) -> dict:
    """One question end-to-end: ONE retrieval call shared by both the QA branch and the sentence-selection
    branch. Returns everything needed for printing/summary - factored out so tests can exercise this
    exact same-chunk / no-generation / metadata-preservation logic with fakes, without loading real models."""
    result = retriever.search(qdef["question"], top_k=1)[0]  # the ONE SearchResult both branches use
    qa_result = qa_model.answer(qdef["question"], result.chunk.text)
    sentences = sents.split_sentences(result.chunk.text)  # SAME chunk text QA just saw

    question_embedding = np.asarray(granite_model.encode([qdef["question"]], normalize_embeddings=True))[0]
    sentence_embeddings = np.asarray(granite_model.encode(sentences, normalize_embeddings=True))
    ranked = sents.rank_sentences(sentences, sentence_embeddings, question_embedding)
    context_sentences = sents.local_context(sentences, ranked[0].index)

    return {
        "result": result,
        "qa_result": qa_result,
        "sentences": sentences,
        "ranked": ranked,
        "context_sentences": context_sentences,
    }


def print_top_sentences(ranked: list) -> None:
    for rank, candidate in enumerate(ranked[:3], start=1):
        print(f"\nTOP {rank}:\n{candidate.text}\n\nSimilarity:\n{candidate.similarity:.4f}")


def print_question_block(index: int, qdef: dict, result, qa_result, ranked: list, context_sentences: list[str]) -> None:
    print("=" * 60)
    print(f"QUESTION {index}")
    print("=" * 60)
    print(f"\nQuestion:\n{qdef['question']}")
    print(f"\nExpected answerability:\n{qdef['expected_answerability']}")

    print("\nGRANITE SECTION RETRIEVAL\n")
    print(f"Source:\n{result.chunk.source}")
    print(f"\nSection:\n{result.chunk.section}")
    print(f"\nPage:\n{result.chunk.page}")
    print(f"\nGranite section similarity (diagnostic only):\n{result.score:.4f}")

    print("\n" + "-" * 60)
    print("CURRENT EXTRACTIVE QA")
    print("-" * 60)
    qa_passed = qa_result.signal >= QA_THRESHOLD
    print(f"\nQA answer:\n{qa_result.text if qa_result.text.strip() else 'NONE'}")
    print(f"\nQA signal (diagnostic only):\n{qa_result.signal:.4f}")
    print(f"\nQA threshold:\n{QA_THRESHOLD}")
    print(f"\nQA passed:\n{'yes' if qa_passed else 'no'}")

    print("\n" + "-" * 60)
    print("SENTENCE-LEVEL SEMANTIC SELECTION")
    print("-" * 60)
    print(f"\nSentence count:\n{len(ranked)}")
    print_top_sentences(ranked)

    print("\n" + "-" * 60)
    print("TOP-1 + LOCAL CONTEXT")
    print("-" * 60)
    print("\n" + " ".join(context_sentences))

    print("\n" + "-" * 60)
    print("STRUCTURAL DIAGNOSTICS")
    print("-" * 60)
    print(f"\nExpected source matched:\n{source_matched(qdef['expected_source'], result.chunk.source)}")
    print(f"\nExpected topic matched:\n{topic_matched(qdef['expected_topic'], result.chunk.section)}")
    print(f"\nQA returned span:\n{'yes' if qa_result.text.strip() else 'no'}")
    print(f"\nQA passed threshold:\n{'yes' if qa_passed else 'no'}")
    print()


def print_summary_table(rows: list[dict]) -> None:
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'ID':<5} {'Expected':<22} {'Source':<26} {'Section':<24} {'QA Output':<22} {'Passed':<7} {'Top Sentence':<30}"
    print(header)
    print("-" * len(header))
    for row in rows:
        qa_short = (row["qa_answer"][:19] + "...") if len(row["qa_answer"]) > 22 else (row["qa_answer"] or "NONE")
        top_short = (row["top_sentence"][:27] + "...") if len(row["top_sentence"]) > 30 else row["top_sentence"]
        print(
            f"{row['id']:<5} {row['expected_answerability']:<22} {str(row['source']):<26} {str(row['section']):<24} "
            f"{qa_short:<22} {('yes' if row['qa_passed'] else 'no'):<7} {top_short:<30}"
        )


def main() -> None:
    chunks = load_chunks(_CHUNKS_PATH)  # persisted corpus only, never re-parsed

    retrieval_load_start = time.monotonic()
    retriever = EmbeddingRetriever()  # frozen: ibm-granite/granite-embedding-small-english-r2
    retriever.load_index(chunks, _EMBEDDINGS_PATH, _MANIFEST_PATH)  # precomputed - chunks are NOT re-encoded
    retrieval_load_seconds = time.monotonic() - retrieval_load_start

    qa_load_start = time.monotonic()
    qa_model = ExtractiveQA()  # frozen: deepset/minilm-uncased-squad2, exact production class
    qa_load_seconds = time.monotonic() - qa_load_start

    # Reuse the SAME already-loaded Granite SentenceTransformer the retriever
    # holds - no second model instance, no new model type. EmbeddingRetriever
    # has no public "embed arbitrary text" method, so this reaches into its
    # already-constructed model object rather than re-instantiating one.
    granite_model = retriever._model

    print(f"Retrieval index load time: {retrieval_load_seconds:.2f}s")
    print(f"QA model load time: {qa_load_seconds:.2f}s")
    print("Granite model reused from EmbeddingRetriever - no second model instance loaded.\n")

    summary_rows = []
    granite_seconds = split_seconds = embed_rank_seconds = qa_seconds = 0.0

    for index, qdef in enumerate(DIAGNOSTIC_QUESTIONS, start=1):
        t0 = time.monotonic()
        result = retriever.search(qdef["question"], top_k=1)[0]  # ONE retrieval call, shared by both branches
        granite_seconds += time.monotonic() - t0

        t1 = time.monotonic()
        qa_result = qa_model.answer(qdef["question"], result.chunk.text)
        qa_seconds += time.monotonic() - t1

        t2 = time.monotonic()
        sentences = sents.split_sentences(result.chunk.text)  # SAME chunk text QA just saw
        split_seconds += time.monotonic() - t2

        t3 = time.monotonic()
        question_embedding = np.asarray(granite_model.encode([qdef["question"]], normalize_embeddings=True))[0]
        sentence_embeddings = np.asarray(granite_model.encode(sentences, normalize_embeddings=True))
        ranked = sents.rank_sentences(sentences, sentence_embeddings, question_embedding)
        embed_rank_seconds += time.monotonic() - t3

        context_sentences = sents.local_context(sentences, ranked[0].index)

        print_question_block(index, qdef, result, qa_result, ranked, context_sentences)

        summary_rows.append(
            {
                "id": qdef["id"],
                "expected_answerability": qdef["expected_answerability"],
                "source": result.chunk.source,
                "section": result.chunk.section,
                "qa_answer": qa_result.text,
                "qa_passed": qa_result.signal >= QA_THRESHOLD,
                "top_sentence": ranked[0].text,
            }
        )

    print_summary_table(summary_rows)

    n = len(DIAGNOSTIC_QUESTIONS)
    print(f"\n{'=' * 60}\nPERFORMANCE\n{'=' * 60}")
    print(f"Granite section retrieval latency: {granite_seconds:.2f}s total, {granite_seconds / n * 1000:.1f}ms/question")
    print(f"Sentence splitting latency: {split_seconds:.3f}s total, {split_seconds / n * 1000:.2f}ms/question")
    print(f"Sentence embedding+ranking latency: {embed_rank_seconds:.2f}s total, {embed_rank_seconds / n * 1000:.1f}ms/question")
    print(f"Current QA latency: {qa_seconds:.2f}s total, {qa_seconds / n * 1000:.1f}ms/question")
    total_sentence_approach = granite_seconds + split_seconds + embed_rank_seconds
    print(f"Total sentence-approach latency (retrieval+split+embed+rank): {total_sentence_approach:.2f}s total, {total_sentence_approach / n * 1000:.1f}ms/question")
    print(f"Total current-QA-approach latency (retrieval+QA): {(granite_seconds + qa_seconds):.2f}s total, {(granite_seconds + qa_seconds) / n * 1000:.1f}ms/question")
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")
    except ImportError:
        print("Peak RSS: unavailable on this platform")


if __name__ == "__main__":
    main()
