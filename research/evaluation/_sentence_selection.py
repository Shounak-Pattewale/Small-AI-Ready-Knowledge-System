"""EXPERIMENTAL ONLY: deterministic sentence splitting + semantic sentence ranking.

Pure logic, no model access - `rank_sentences()` takes already-computed
embeddings so it's fully unit-testable without loading Granite. The actual
embedding calls live in evaluate_sentence_answers.py, which reuses the
EmbeddingRetriever's already-loaded Granite model (no second model load).

Splitting is deliberately simple (paragraph break, then sentence-ending
punctuation followed by whitespace, with an obvious-bullet-list special
case) - no NLP library. Known limitations (documented, not fixed here):
does not special-case abbreviations (e.g. "U.K.", "e.g.") that could be
mistaken for sentence-final punctuation followed by whitespace, and a
heading-like line with no terminal punctuation (e.g. "3.1 Becoming a
mentor") becomes its own short, low-signal candidate sentence rather than
being merged into the paragraph that follows it.
"""

import re
from dataclasses import dataclass

import numpy as np

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` into complete, verbatim sentences/segments - no paraphrasing, no rewriting.

    Paragraphs are split first, then each paragraph is split on
    whitespace immediately after '.', '!', or '?'. A paragraph whose every
    non-blank line looks like a bullet/numbered list item is instead split
    one segment per line (list items rarely end in sentence punctuation).
    """
    segments: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = [line for line in paragraph.split("\n") if line.strip()]
        if len(lines) > 1 and all(_BULLET_RE.match(line) for line in lines):
            segments.extend(line.strip() for line in lines)
            continue
        flattened = " ".join(line.strip() for line in paragraph.split("\n"))
        segments.extend(s.strip() for s in _SENTENCE_RE.split(flattened) if s.strip())
    return segments


@dataclass
class RankedSentence:
    text: str
    index: int  # original position in the sentence list (document order)
    similarity: float


def rank_sentences(sentences: list[str], sentence_embeddings: np.ndarray, question_embedding: np.ndarray) -> list[RankedSentence]:
    """Rank sentences by cosine similarity to the question (both embeddings must already be normalized).

    Descending similarity; ties broken by original sentence order (ascending
    index) - deterministic, never by any secondary heuristic.
    """
    scores = sentence_embeddings @ question_embedding
    order = sorted(range(len(sentences)), key=lambda i: (-float(scores[i]), i))
    return [RankedSentence(text=sentences[i], index=i, similarity=float(scores[i])) for i in order]


def local_context(sentences: list[str], top1_index: int) -> list[str]:
    """Fixed, non-semantic rule: Top-1 + following sentence if one exists, else preceding + Top-1,
    else Top-1 alone. Always returned in original document order - the adjacent sentence is chosen
    by position only, never by similarity score."""
    if top1_index + 1 < len(sentences):
        return [sentences[top1_index], sentences[top1_index + 1]]
    if top1_index - 1 >= 0:
        return [sentences[top1_index - 1], sentences[top1_index]]
    return [sentences[top1_index]]
