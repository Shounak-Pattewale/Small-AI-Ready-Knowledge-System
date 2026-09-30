"""Tests for src/knowledge_system/chunking.py."""

import pytest
from knowledge_system.chunking import chunk_sections
from knowledge_system.models import DocumentSection


def make_section(text: str, source: str = "doc.md", section: str | None = None, page: int | None = None) -> DocumentSection:
    return DocumentSection(text=text, source=source, section=section, page=page)


def words(n: int, prefix: str = "word") -> str:
    """Build a string of n space-separated words (no sentence/paragraph structure)."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentence(n_words: int, tag: str) -> str:
    """Build one 'sentence' of n_words words ending in a period."""
    return words(n_words, prefix=f"{tag}w") + "."


def test_small_section_stays_one_chunk():
    section = make_section("Just a short section.", source="a.md")
    chunks = chunk_sections([section], chunk_size=400, chunk_overlap=60)

    assert len(chunks) == 1
    assert chunks[0].text == "Just a short section."
    assert chunks[0].chunk_index == 0


def test_large_multi_paragraph_section_is_split():
    paragraphs = [words(50, prefix=f"p{p}w") for p in range(6)]  # 6 * 50 = 300 words
    text = "\n\n".join(paragraphs)
    section = make_section(text)

    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.split()) <= 100


def test_paragraph_boundaries_preferred():
    # Two paragraphs, each under chunk_size on its own, but combined over it.
    p1 = words(60, prefix="ap")
    p2 = words(60, prefix="bp")
    text = f"{p1}\n\n{p2}"
    section = make_section(text)

    chunks = chunk_sections([section], chunk_size=60, chunk_overlap=0)

    # Each paragraph should land in its own chunk rather than being cut mid-paragraph.
    assert len(chunks) == 2
    assert chunks[0].text == p1
    assert chunks[1].text == p2


def test_oversized_paragraph_falls_back_to_sentences():
    # One paragraph (no blank lines) made of several sentences, too big to keep whole.
    sentences = [sentence(30, tag=f"s{i}") for i in range(4)]  # 4 * 30 = 120 words, one paragraph
    text = " ".join(sentences)
    section = make_section(text)

    chunks = chunk_sections([section], chunk_size=40, chunk_overlap=0)

    assert len(chunks) > 1
    # Every chunk boundary should land on a sentence boundary (each chunk ends in '.').
    for chunk in chunks:
        assert chunk.text.strip().endswith(".")


def test_oversized_sentence_falls_back_to_word_split():
    # A single "sentence" (no . ! ?) far larger than chunk_size, and no paragraph breaks.
    text = words(250, prefix="runon")
    section = make_section(text)

    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 3  # 250 words -> 100 + 100 + 50
    assert len(chunks[0].text.split()) == 100
    assert len(chunks[1].text.split()) == 100
    assert len(chunks[2].text.split()) == 50


def test_overlap_appears_between_consecutive_chunks():
    text = words(250, prefix="runon")
    section = make_section(text)

    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=20)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # The tail of chunk 0 should reappear at the head of chunk 1.
    assert first_words[-20:] == second_words[:20]


def test_overlap_does_not_cross_section_boundaries():
    section_a = make_section(words(150, prefix="a"), source="a.md")
    section_b = make_section(words(150, prefix="b"), source="b.md")

    chunks = chunk_sections([section_a, section_b], chunk_size=100, chunk_overlap=20)

    a_chunks = [c for c in chunks if c.source == "a.md"]
    b_chunks = [c for c in chunks if c.source == "b.md"]
    assert len(a_chunks) == 2
    assert len(b_chunks) == 2
    # b's first chunk must not start with any trailing words from a's last chunk.
    a_tail = a_chunks[-1].text.split()[-20:]
    b_head = b_chunks[0].text.split()[:20]
    assert a_tail != b_head
    assert "a" not in b_chunks[0].text


def test_metadata_is_preserved():
    section = make_section(words(500), source="policy.pdf", section="Intro", page=3)
    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "policy.pdf"
        assert chunk.section == "Intro"
        assert chunk.page == 3


def test_chunk_indexes_are_deterministic_and_increasing():
    section_a = make_section(words(250, prefix="a"), source="a.md")
    section_b = make_section(words(250, prefix="b"), source="b.md")

    chunks = chunk_sections([section_a, section_b], chunk_size=100, chunk_overlap=0)

    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))  # 0, 1, 2, ... with no resets per section


def test_empty_and_whitespace_sections_produce_no_chunks():
    sections = [
        make_section(""),
        make_section("   \n\n  \n"),
        make_section("Real content here."),
    ]

    chunks = chunk_sections(sections, chunk_size=400, chunk_overlap=60)

    assert len(chunks) == 1
    assert chunks[0].text == "Real content here."


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (0, 10),  # chunk_size must be positive
        (-5, 10),  # chunk_size must be positive
        (100, -1),  # chunk_overlap must be non-negative
        (100, 100),  # chunk_overlap must be smaller than chunk_size
        (100, 150),  # chunk_overlap must be smaller than chunk_size
    ],
)
def test_invalid_configuration_raises_value_error(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_sections([make_section("some text")], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
