"""Typed data models for the knowledge base ingestion pipeline.

These classes are pure data containers (no behavior) that describe the
shape of information as it flows through the system: raw documents get
loaded into `DocumentSection` objects, which will later (in a future step,
not this one) get split into `DocumentChunk` objects for retrieval.
"""

from dataclasses import dataclass  # typed struct with auto __init__/__repr__/__eq__, no Pydantic needed


@dataclass
class DocumentSection:
    """One logical piece of a source document (a heading's content, or a PDF page).

    Produced directly by the loaders in loaders.py, then split into
    DocumentChunk objects by chunking.py.
    """

    text: str  # the extracted section/page text, already stripped of surrounding whitespace
    source: str  # filename the text came from, e.g. "handbook.pdf" (plain string, not a full path)
    section: str | None = None  # Markdown heading path, e.g. "Rail Travel > Standard process"; None for PDFs (no heading structure)
    page: int | None = None  # 1-indexed page number for PDF sections; None for Markdown (no "pages")


@dataclass
class DocumentChunk:
    """A retrieval-sized slice of a DocumentSection, produced by chunking.py."""

    text: str  # the chunk's text content (a sub-piece of a DocumentSection's text)
    source: str  # same meaning as DocumentSection.source, carried through so we can cite it later
    section: str | None = None  # carried through from the parent DocumentSection (may be a heading path)
    page: int | None = None  # carried through from the parent DocumentSection
    chunk_index: int = 0  # monotonically increasing index across the whole returned chunk list, not reset per section


@dataclass
class SearchResult:
    """One ranked retrieval result: a chunk plus the score that ranked it, produced by retrieval/embedding.py."""

    chunk: DocumentChunk
    score: float  # raw retriever score (e.g. cosine similarity); not a probability or percentage confidence


@dataclass
class KnowledgeAnswer:
    """Production response from KnowledgeService.ask() - the shape both the terminal CLI and a future
    Flask UI/API consume. Deliberately excludes internal scores (Granite similarity, QA signal): those
    are implementation/evaluation details, never a user-facing "confidence"."""

    answered: bool  # whether the QA signal cleared the frozen answerability threshold
    answer: str | None  # the QA model's extracted span verbatim; None when answered=False
    source: str | None  # source filename; None when answered=False (see service.py for why)
    section: str | None  # section heading, if any; None when answered=False or the chunk had none
    page: int | None  # page number, if any; None when answered=False or the chunk had none
