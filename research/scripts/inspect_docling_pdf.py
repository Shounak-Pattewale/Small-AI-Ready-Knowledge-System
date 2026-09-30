"""Diagnostic only: inspect what Docling recovers from our PDFs.

Does NOT import or modify src/knowledge_system/loaders.py. This script
exists purely to gather evidence on whether Docling's structured-document
output (headings, levels, page provenance) is good enough to justify
replacing the pypdf-based PDF loader later - that decision is NOT made here.

Second pass: the PDFs were replaced with realistic "continuous-flow"
documents (multiple topics per page, headings that don't reset per page,
content that continues across a page break). This version adds page-header/
footer inspection, same-page multi-heading detection, and cross-page body
continuation checks on top of the original heading/level/provenance dump -
all still read-only diagnostics over Docling's output.

Run from the project root:
    venv/bin/python scripts/inspect_docling_pdf.py
"""

import re
import time
from collections import defaultdict
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PDFS = [
    "employee_handbook.pdf",
    "holiday_leave_policy.pdf",
    "flexible_remote_working.pdf",
    "career_development.pdf",
]

_BODY_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM}
_HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
_BOILERPLATE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

# Diagnostic-only heuristic to distinguish a "major numbered topic" (e.g.
# "4. Carry-over") from a nested sub-heading (e.g. "4.1 Requirements") or a
# generic non-numbered heading (e.g. "Purpose"). Purely for this report -
# never fed into production parsing.
_MAJOR_TOPIC_RE = re.compile(r"^\s*\d{1,2}\.\s+\S")
_NESTED_TOPIC_RE = re.compile(r"^\s*\d{1,2}\.\d+\s*\S")


def _build_converter() -> DocumentConverter:
    """Configure the PDF pipeline for this experiment.

    heading_hierarchy_options.enabled=True turns on Docling's optional
    heading-level inference (from PDF bookmarks/numbering/font style).
    do_ocr=False because these PDFs already contain digitally extractable
    text, confirmed by the existing pypdf loader - OCR would only add time.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.heading_hierarchy_options.enabled = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _page_number(item) -> int | None:
    """First provenance page number for a doc item, or None if it has no provenance."""
    if item.prov:
        return item.prov[0].page_no
    return None


def _heading_level(item) -> int | None:
    """SectionHeaderItem has a `.level`; TitleItem (the document title) does not."""
    return getattr(item, "level", None)


def _classify_heading(text: str) -> str:
    if _NESTED_TOPIC_RE.match(text):
        return "nested"
    if _MAJOR_TOPIC_RE.match(text):
        return "major"
    return "generic"


def _body_sample_after(items: list, heading_index: int, max_chars: int = 160) -> tuple[str, list[int]]:
    """Body text + the set of pages it was found on, until the next heading.

    Returns (sample_text, pages_seen) so callers can tell whether the body
    belonging to a heading spills onto a later page than the heading itself.
    """
    sample = "(no body text found before next heading)"
    pages_seen: list[int] = []
    got_sample = False
    for item, _level in items[heading_index + 1 :]:
        if item.label in _HEADING_LABELS:
            break
        if item.label in _BODY_LABELS and item.text.strip():
            page = _page_number(item)
            if page is not None and page not in pages_seen:
                pages_seen.append(page)
            if not got_sample:
                text = item.text.strip()
                sample = text[:max_chars] + ("..." if len(text) > max_chars else "")
                got_sample = True
    return sample, pages_seen


def inspect_pdf(converter: DocumentConverter, pdf_path: Path) -> None:
    print("=" * 60)
    print(pdf_path.name)
    print("=" * 60)

    start = time.monotonic()
    result = converter.convert(pdf_path)
    elapsed = time.monotonic() - start

    doc = result.document
    items = list(doc.iterate_items())  # [(NodeItem, level), ...] in reading order

    headings = []  # (index, item, page_no, level, kind)
    boilerplate_examples = []  # (label, page_no, text) - first few page headers/footers seen
    for index, (item, _tree_level) in enumerate(items):
        if item.label in _HEADING_LABELS:
            page_no = _page_number(item)
            level = _heading_level(item)
            kind = "title" if item.label == DocItemLabel.TITLE else _classify_heading(item.text)
            headings.append((index, item, page_no, level, kind))
    # Repeated headers/footers/page numbers are NOT in `items` above at all:
    # doc.iterate_items() only walks the BODY content layer by default, and
    # Docling routes this boilerplate to a separate FURNITURE content layer
    # with its own page_header/page_footer labels. Query doc.texts directly
    # to see it (this is the correct place to look for requirement D).
    boilerplate_examples = [
        (t.label, _page_number(t), t.text.strip())
        for t in doc.texts
        if t.content_layer.value == "furniture" and t.label in _BOILERPLATE_LABELS
    ]
    furniture_count = len(boilerplate_examples)
    boilerplate_examples = boilerplate_examples[:6]

    # --- A: headings + levels + a body sample/continuation check for majors ---
    major_count = 0
    for index, item, page_no, level, kind in headings:
        level_label = f"Level {level}" if level is not None else "Level n/a (TITLE)"
        print(f"\nPage {page_no}  [{kind}]")
        print(f"{level_label}: {item.text}")
        if kind == "major":
            major_count += 1
            if major_count <= 8:
                sample, pages_seen = _body_sample_after(items, index)
                print(f"  body sample: {sample}")
                if pages_seen and page_no is not None and any(p != page_no for p in pages_seen):
                    print(f"  body provenance pages: {pages_seen} (heading is on page {page_no})")

    # --- B: multiple major topics sharing one physical page ---
    majors_by_page: dict[int, list[str]] = defaultdict(list)
    for _index, item, page_no, _level, kind in headings:
        if kind == "major" and page_no is not None:
            majors_by_page[page_no].append(item.text)
    shared_pages = {p: texts for p, texts in majors_by_page.items() if len(texts) > 1}
    print("\n--- Pages with 2+ major numbered topics ---")
    if shared_pages:
        for page_no in sorted(shared_pages):
            print(f"Page {page_no}:")
            for text in shared_pages[page_no]:
                print(f"  SECTION_HEADER: {text}")
    else:
        print("(none found)")

    # --- C: repeated header/footer handling ---
    print(f"\n--- FURNITURE-layer items (page_header/page_footer), excluded from body reading order: {furniture_count} total ---")
    if boilerplate_examples:
        for label, page_no, text in boilerplate_examples:
            print(f"page {page_no} [{label.value}]: {text[:80]!r}")
    else:
        print("(none found)")

    total_headings = len(headings)
    major_headings = sum(1 for *_r, kind in headings if kind == "major")
    nested_headings = sum(1 for *_r, kind in headings if kind == "nested")
    generic_headings = sum(1 for *_r, kind in headings if kind == "generic")
    print(f"\nDetected headings: {total_headings} (major={major_headings}, nested={nested_headings}, generic/title={generic_headings})")
    print(f"Conversion time: {elapsed:.1f}s")
    if result.errors:
        print(f"Errors: {result.errors}")
    print()


def main() -> None:
    converter = _build_converter()
    for filename in _PDFS:
        inspect_pdf(converter, _DATA_DIR / filename)


if __name__ == "__main__":
    main()
