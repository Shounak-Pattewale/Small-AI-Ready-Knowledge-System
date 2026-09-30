"""Document loaders that turn source files into DocumentSection objects.

This is the ingestion layer: it knows how to read .md and .pdf files off
disk and turn them into the DocumentSection objects defined in models.py.
It deliberately does NOT know anything about chunking, search, or
retrieval - those are separate concerns for a later step.
"""

import re  # matches Markdown headings, no full Markdown parser needed
from pathlib import Path  # all filesystem work goes through pathlib, not os.path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption  # structure-aware PDF parser
from docling_core.types.doc.labels import DocItemLabel

from knowledge_system.models import DocumentSection

# Matches Markdown ATX headings: 1-6 '#' characters, whitespace, then heading text.
# Capture group 1 = the '#' run (unused), group 2 = the heading text itself.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def load_markdown(path: Path) -> list[DocumentSection]:
    """Split a Markdown file into sections using headings as boundaries.

    Why headings and not fixed-size splitting? Headings are the author's
    own semantic boundaries - splitting on them keeps related content
    together instead of cutting an explanation in half at an arbitrary
    character count. Fixed-size chunking is a separate, later step that
    operates on top of this section-level structure (see chunking.py).

    `section` is a "Parent > Child" path built from the active heading
    hierarchy (e.g. "Rail Travel > Standard process"), not just the
    nearest heading - a lone "Standard process" is ambiguous when many
    unrelated headings share that name. The top-level (level-1) heading
    is tracked for hierarchy purposes but left out of the path itself,
    since the source filename already identifies the document.
    """
    source = path.name  # just the filename (e.g. "handbook.md"), not the full path
    lines = path.read_text(encoding="utf-8").splitlines()

    sections: list[DocumentSection] = []
    # Stack of (level, heading_text) for every heading currently "open" -
    # e.g. while inside "# Policy" > "## Rail Travel" > "### Standard
    # process", the stack holds all three, innermost last.
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []  # body lines accumulated under the current heading so far

    def current_section_path() -> str | None:
        """Join the active hierarchy (excluding the level-1 title) into a "Parent > Child" string."""
        parts = [text for level, text in heading_stack if level > 1]
        return " > ".join(parts) if parts else None

    def flush() -> None:
        """Turn whatever body text we've accumulated into a DocumentSection.

        Called once when we hit a new heading (to close out the previous
        section) and once more at the very end (to close out the last one).
        """
        text = "\n".join(current_lines).strip()
        if text:  # skip sections with no real content (e.g. heading immediately followed by another heading)
            sections.append(DocumentSection(text=text, source=source, section=current_section_path()))

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            # We've hit a new heading line, so the section we were building is complete.
            flush()
            level = len(match.group(1))  # number of '#' marks = heading depth
            heading_text = match.group(2).strip()
            # A heading at level N closes out any open heading at level N
            # or deeper (a sibling or a return to a shallower level both
            # replace what came before them in the hierarchy).
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_lines = []
        else:
            # An ordinary content line - accumulate it under the current heading
            # (or under "no heading yet" if we're still before the first '#' line).
            current_lines.append(line)

    flush()  # the loop never flushes the *last* section on its own
    return sections


_PDF_BODY_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM}

_pdf_converter: DocumentConverter | None = None  # lazily built, reused across load_pdf() calls


def _get_pdf_converter() -> DocumentConverter:
    """Build (once) the Docling converter configured for this project.

    do_ocr=False: these PDFs already contain digitally extractable text.
    heading_hierarchy_options.enabled=True: without it, Docling leaves
    every SECTION_HEADER at level 1, which is useless for hierarchy - this
    configuration was proven against the real corpus in
    scripts/inspect_docling_pdf.py before being used here.
    """
    global _pdf_converter
    if _pdf_converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.heading_hierarchy_options.enabled = True
        _pdf_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    return _pdf_converter


def _page_number(item) -> int | None:
    """First provenance page number for a Docling item, or None if it has no provenance."""
    if item.prov:
        return item.prov[0].page_no
    return None


_PRIMARY_HEADING_LEVEL = 1  # a level-1 SECTION_HEADER opens a new DocumentSection; anything deeper nests inside it


def load_pdf(path: Path) -> list[DocumentSection]:
    """Recover semantic sections from a PDF using Docling, one DocumentSection per Level-1 topic.

    Document structure and retrieval-chunk boundaries are different
    concerns: every heading Docling detects should not automatically
    become its own tiny DocumentSection (that produced 7-30 word
    fragments in an earlier version of this loader, which hurt lexical
    retrieval). Instead, a level-1 heading defines the DocumentSection
    boundary; any deeper heading (level 2, 3, ...) stays inside that
    section - its heading text is preserved as a line in `text` (useful
    retrieval context), but it does not start a new section. Splitting an
    oversized section back down to retrieval-sized chunks is the existing
    chunker's job (chunking.py), not this loader's.

    A DocItemLabel.TITLE item is document-level metadata (the filename
    already identifies the document) and is skipped entirely, never
    treated as a heading of any level.

    doc.iterate_items() only walks the BODY content layer by default, so
    repeated page headers/footers/page numbers (Docling's FURNITURE
    layer) never reach `text` here without any extra filtering.

    `page` is fixed to the page the level-1 heading appears on and is
    never updated as nested headings/body content flow onto later pages -
    a section that crosses a page boundary stays one section.
    """
    source = path.name
    result = _get_pdf_converter().convert(str(path))
    doc = result.document

    sections: list[DocumentSection] = []
    current_heading: str | None = None  # active level-1 heading text, or None before the first one
    current_lines: list[str] = []  # nested heading text + body paragraphs, in reading order
    current_page: int | None = None  # page the current level-1 section started on

    def flush() -> None:
        text = "\n\n".join(current_lines).strip()
        if text:  # skip sections with no real content (e.g. a level-1 heading immediately followed by another)
            sections.append(DocumentSection(text=text, source=source, section=current_heading, page=current_page))

    for item, _tree_level in doc.iterate_items():
        if item.label == DocItemLabel.TITLE:
            continue  # document title is metadata, not part of any section
        if item.label == DocItemLabel.SECTION_HEADER:
            level = getattr(item, "level", None) or 1
            if level <= _PRIMARY_HEADING_LEVEL:
                flush()
                current_heading = item.text.strip()
                current_page = _page_number(item)
                current_lines = []
            else:
                # A nested heading (level 2+) does not open a new section -
                # its text is preserved as retrieval context inside the
                # currently active level-1 section's body.
                text = item.text.strip()
                if text:
                    current_lines.append(text)
        elif item.label in _PDF_BODY_LABELS:
            text = item.text.strip()
            if text:
                current_lines.append(text)
                if current_page is None:  # front matter before any level-1 heading: use its own first page
                    current_page = _page_number(item)

    flush()  # the loop never flushes the *last* section on its own
    return sections


def load_directory(path: Path) -> list[DocumentSection]:
    """Load all supported documents (.md, .pdf) from a directory, sorted by name.

    Single entry point the rest of the system calls - hides the
    "which loader for which file extension" decision from callers.
    """
    path = Path(path)  # accept a plain string too, not just a Path

    sections: list[DocumentSection] = []
    # sorted by name so results are deterministic - iterdir() order can
    # vary between runs/filesystems otherwise.
    for file_path in sorted(path.iterdir(), key=lambda p: p.name):
        if not file_path.is_file():
            continue  # skip subdirectories
        suffix = file_path.suffix.lower()  # ".MD" and ".md" should both work
        if suffix == ".md":
            sections.extend(load_markdown(file_path))
        elif suffix == ".pdf":
            sections.extend(load_pdf(file_path))
        # any other extension is silently ignored

    return sections
