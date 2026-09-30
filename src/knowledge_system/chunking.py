"""Recursive, structure-aware chunker: DocumentSection[] -> DocumentChunk[].

Splits each section's text into word-count-bounded chunks while preferring
natural boundaries (paragraphs, then sentences, then raw words as a last
resort), and adds configurable word-overlap between chunks that came from
the same section. Knows nothing about Markdown or PDF parsing - it only
operates on the already-extracted `DocumentSection.text`.
"""

import re  # small paragraph/sentence-boundary regexes, no NLP library needed

from knowledge_system.models import DocumentChunk, DocumentSection

# A paragraph boundary is one or more blank lines. \s within a line is
# collapsed by \n\s*\n so "line\n \nline" (blank line with stray spaces)
# still counts as a paragraph break.
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")

# A sentence boundary is whitespace immediately after ., !, or ?. This is a
# deliberately simple heuristic (it will misfire on abbreviations like
# "Mr." or decimal numbers) - good enough for a chunking prototype, and we
# were explicitly told not to pull in NLTK/spaCy just for this.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_sections(
    sections: list[DocumentSection],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[DocumentChunk]:
    """Convert DocumentSection objects into DocumentChunk objects.

    `chunk_size` and `chunk_overlap` are word counts, not characters or
    model tokens. Chunk indices are assigned as one monotonically
    increasing counter across the whole returned list (not reset per
    section), so a chunk_index alone identifies its position in the output.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    next_index = 0

    for section in sections:
        text = section.text.strip()
        if not text:
            continue  # empty/whitespace-only sections produce no chunks

        # Break the section down into pieces that individually fit within
        # chunk_size (paragraph -> sentence -> hard word split), then
        # greedily re-combine adjacent pieces up to chunk_size, then layer
        # overlap across the resulting chunk boundaries.
        atoms = _split_to_atoms(text, chunk_size)
        piece_texts = _combine_atoms(atoms, chunk_size)
        piece_texts = _add_overlap(piece_texts, chunk_overlap)

        for piece in piece_texts:
            chunks.append(
                DocumentChunk(
                    text=piece,
                    source=section.source,
                    section=section.section,
                    page=section.page,
                    chunk_index=next_index,
                )
            )
            next_index += 1

    return chunks


def _split_to_atoms(text: str, chunk_size: int) -> list[str]:
    """Recursively split `text` into pieces that each fit within chunk_size words.

    Tries, in order: keep as-is if already small enough, paragraph split,
    sentence split, hard word split. Recursion always strictly shrinks the
    text being processed (a split only happens when it yields more than
    one non-empty piece), so this terminates - it never re-splits the same
    text into itself.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        atoms: list[str] = []
        for paragraph in paragraphs:
            atoms.extend(_split_to_atoms(paragraph, chunk_size))
        return atoms

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        atoms = []
        for sentence in sentences:
            atoms.extend(_split_to_atoms(sentence, chunk_size))
        return atoms

    # Single paragraph, single sentence, still too big: nothing left to
    # split on semantically, so fall back to a hard word-count split.
    return _hard_split_words(text, chunk_size)


def _combine_atoms(atoms: list[str], chunk_size: int) -> list[str]:
    """Greedily pack adjacent atoms into chunks, staying within chunk_size words."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for atom in atoms:
        atom_words = len(atom.split())
        if current and current_words + atom_words > chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(atom)
        current_words += atom_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def _add_overlap(chunk_texts: list[str], chunk_overlap: int) -> list[str]:
    """Prepend up to chunk_overlap words of trailing context from the previous chunk.

    Only looks at chunk_texts from a single section (callers never mix
    sections into one list), so overlap never crosses a section boundary.
    The first chunk is never given a predecessor, so it's returned as-is.
    """
    if chunk_overlap <= 0 or len(chunk_texts) <= 1:
        return list(chunk_texts)

    result = [chunk_texts[0]]
    for i in range(1, len(chunk_texts)):
        overlap = _trailing_overlap(chunk_texts[i - 1], chunk_overlap)
        result.append(f"{overlap} {chunk_texts[i]}" if overlap else chunk_texts[i])
    return result


def _trailing_overlap(prev_chunk_text: str, chunk_overlap: int) -> str:
    """Return up to chunk_overlap words of trailing context from `prev_chunk_text`.

    Prefers whole trailing sentences, so overlap doesn't start mid-sentence.
    Falls back to a plain trailing word slice when sentence-based overlap
    would overshoot the target by a lot (e.g. one very long sentence) -
    that keeps chunk sizes from ballooning unpredictably.
    """
    words = prev_chunk_text.split()
    if len(words) <= chunk_overlap:
        return prev_chunk_text  # previous chunk is already small enough to reuse whole

    sentences = _split_sentences(prev_chunk_text)
    picked: list[str] = []
    picked_word_count = 0
    for sentence in reversed(sentences):
        picked.insert(0, sentence)
        picked_word_count += len(sentence.split())
        if picked_word_count >= chunk_overlap:
            break

    if picked_word_count <= chunk_overlap * 2:
        return " ".join(picked)

    return " ".join(words[-chunk_overlap:])


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _hard_split_words(text: str, chunk_size: int) -> list[str]:
    """Last-resort split: cut a single oversized paragraph/sentence into fixed-size word runs."""
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
