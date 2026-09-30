"""Shared test fixtures: builds tiny PDFs without extra dependencies.

We hand-build real, minimal PDFs here (rather than shipping binary .pdf
fixture files in the repo, or adding a PDF-writing dependency like
reportlab) so the PDF loader tests exercise actual pypdf parsing
end-to-end, with the content fully visible/reviewable as Python code.
"""

from pathlib import Path

import pytest  # needed for the @pytest.fixture decorator below
from docling_core.types.doc.labels import DocItemLabel  # real label enum load_pdf() switches on
from pypdf import PdfWriter  # builds a PDF from scratch, object by object
from pypdf.generic import (  # low-level PDF object types for hand-building a content stream
    ContentStream,
    DictionaryObject,
    FloatObject,
    NameObject,
    TextStringObject,
)


def _add_text_page(writer: PdfWriter, text: str) -> None:
    """Append one page containing `text` to `writer`."""
    page = writer.add_blank_page(width=200, height=200)

    # PDF text drawing requires a font resource on the page. Helvetica is
    # one of the 14 "standard" fonts every PDF reader supports without
    # embedding font data.
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    # Objects referenced elsewhere in the PDF must be registered with the
    # writer so they get a proper indirect reference in the output file.
    font_ref = writer._add_object(font)

    # /Resources maps the short name "/F1" (used in the content stream
    # below) to the actual font object.
    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
    page[NameObject("/Resources")] = resources

    # A content stream is a tiny stack-based program describing what to
    # draw on the page. Each tuple is (operands, operator), matching PDF's
    # postfix syntax - e.g. "/F1 12 Tf" sets the font to F1 at size 12.
    stream = ContentStream(None, writer)
    stream.operations = [
        ([], b"q"),  # save graphics state
        ([NameObject("/F1"), FloatObject(12)], b"Tf"),  # set font and size
        ([FloatObject(20), FloatObject(100)], b"Td"),  # move text cursor
        ([TextStringObject(text)], b"Tj"),  # draw the string
        ([], b"Q"),  # restore graphics state
    ]
    page[NameObject("/Contents")] = writer._add_object(stream)


def make_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a PDF to `path` with one page per string in `page_texts`."""
    writer = PdfWriter()
    for text in page_texts:
        _add_text_page(writer, text)
    with path.open("wb") as f:  # PdfWriter.write() wants a binary file handle
        writer.write(f)


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """A fresh empty directory, unique per test, for writing fixture files into."""
    return tmp_path


# --- Fakes for mocking Docling's structured output, so PDF loader unit
# tests don't need to run the real (slow, layout-model-dependent) Docling
# conversion. Only src/knowledge_system/loaders.py's load_pdf() reads
# item.label / item.text / item.level / item.prov[0].page_no, and
# result.document.iterate_items() - these fakes implement exactly that.


class _FakeProv:
    def __init__(self, page_no: int | None) -> None:
        self.page_no = page_no


class _FakeItem:
    def __init__(self, label: DocItemLabel, text: str, level: int | None = None, page_no: int | None = None) -> None:
        self.label = label
        self.text = text
        self.level = level
        self.prov = [_FakeProv(page_no)] if page_no is not None else []


class _FakeDocument:
    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = items

    def iterate_items(self):
        return [(item, 0) for item in self._items]


class _FakeConversionResult:
    def __init__(self, items: list[_FakeItem]) -> None:
        self.document = _FakeDocument(items)
        self.errors: list = []


class FakePdfConverter:
    """Stand-in for docling.document_converter.DocumentConverter in tests."""

    def __init__(self, items: list[_FakeItem]) -> None:
        self._items = items

    def convert(self, path) -> _FakeConversionResult:
        return _FakeConversionResult(self._items)


def fake_title(text: str, page_no: int | None = None) -> _FakeItem:
    return _FakeItem(DocItemLabel.TITLE, text, page_no=page_no)


def fake_heading(text: str, level: int, page_no: int | None = None) -> _FakeItem:
    return _FakeItem(DocItemLabel.SECTION_HEADER, text, level=level, page_no=page_no)


def fake_text(text: str, page_no: int | None = None) -> _FakeItem:
    return _FakeItem(DocItemLabel.TEXT, text, page_no=page_no)


def fake_furniture(text: str, page_no: int | None = None) -> _FakeItem:
    """A page_header item - load_pdf() never sees these in production either, since
    doc.iterate_items() excludes the FURNITURE content layer by default. Included
    here only so a test can confirm load_pdf() doesn't rely on that exclusion
    happening to be true - it just never receives these items at all."""
    return _FakeItem(DocItemLabel.PAGE_HEADER, text, page_no=page_no)


@pytest.fixture
def mock_pdf(monkeypatch):
    """Patch load_pdf()'s Docling converter to return the given fake items.

    Usage: mock_pdf([fake_heading("1. Topic", level=1, page_no=1), fake_text("Body.", page_no=1)])
    """
    from knowledge_system import loaders

    def _set(items: list[_FakeItem]) -> None:
        monkeypatch.setattr(loaders, "_get_pdf_converter", lambda: FakePdfConverter(items))

    return _set


# --- Fake for mocking SentenceTransformer, so EmbeddingRetriever unit
# tests never download/load the real Granite model. Encodes text as a
# small deterministic bag-of-keywords vector: tests craft chunk/query text
# containing these keywords to get predictable similarity ordering.

_FAKE_EMBEDDING_KEYWORDS = {
    "alpha": (1.0, 0.0, 0.0),
    "beta": (0.0, 1.0, 0.0),
    "gamma": (0.0, 0.0, 1.0),
}


class FakeEmbeddingModel:
    """Stand-in for sentence_transformers.SentenceTransformer in tests.

    Records every encode() call's input in `encode_calls`, so tests can
    assert document chunks were (or were NOT) re-encoded - e.g. after
    EmbeddingRetriever.load_index(), only the query should ever pass
    through encode() again.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], normalize_embeddings: bool = False):
        import numpy as np

        self.encode_calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(3)
            for keyword, coords in _FAKE_EMBEDDING_KEYWORDS.items():
                if keyword in lowered:
                    vector += np.array(coords)
            if not vector.any():
                vector = np.array([1.0, 1.0, 1.0])  # arbitrary nonzero fallback, never all-zero
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.array(vectors)


@pytest.fixture
def mock_embedding_model(monkeypatch):
    """Patch EmbeddingRetriever's SentenceTransformer with the deterministic fake above."""
    from knowledge_system.retrieval import embedding

    monkeypatch.setattr(embedding, "SentenceTransformer", FakeEmbeddingModel)
