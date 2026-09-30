"""Tests for the ingestion loaders in src/knowledge_system/loaders.py."""

from pathlib import Path

from conftest import fake_furniture, fake_heading, fake_text, fake_title, make_pdf  # tests/conftest.py, no package import needed
from knowledge_system.loaders import load_directory, load_markdown, load_pdf


def write_md(path: Path, name: str, content: str) -> Path:
    """Write `content` to `path/name` and return the resulting file path."""
    file_path = path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_markdown_sections_and_heading_metadata(kb_dir: Path):
    """Headings should become section boundaries, and heading text should survive into `section`."""
    content = (
        "# Title\n"
        "Intro text.\n\n"
        "## Section One\n"
        "Body one.\n\n"
        "## Section Two\n"
        "Body two.\n"
    )
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    # The level-1 title is omitted from `section` (the filename already
    # identifies the document), so intro text under it gets section=None.
    assert [s.section for s in sections] == [None, "Section One", "Section Two"]
    assert sections[0].text == "Intro text."
    assert sections[1].text == "Body one."
    assert sections[2].text == "Body two."
    assert all(s.source == "doc.md" for s in sections)


def test_markdown_nested_headings_produce_a_path(kb_dir: Path):
    """A heading nested under a parent should get a 'Parent > Child' section path."""
    content = "# Policy\n## Rail Travel\n### Standard process\nBook via the portal.\n"
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    assert len(sections) == 1
    assert sections[0].section == "Rail Travel > Standard process"


def test_markdown_content_directly_under_level_two_heading(kb_dir: Path):
    """Content under a level-2 heading with no deeper heading should be just that heading's text."""
    content = "# Policy\n## Rail Travel\nBook via the portal.\n"
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    assert len(sections) == 1
    assert sections[0].section == "Rail Travel"


def test_markdown_sibling_headings_replace_deepest_level(kb_dir: Path):
    """A sibling heading at the same depth should replace the previous one, not nest under it."""
    content = "# Policy\n## Rail Travel\n### Standard process\nBody A.\n### Exceptions\nBody B.\n"
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    assert [s.section for s in sections] == [
        "Rail Travel > Standard process",
        "Rail Travel > Exceptions",
    ]


def test_markdown_returning_to_shallower_heading_drops_deeper_ones(kb_dir: Path):
    """A new level-2 heading should close out any level-3 (or deeper) heading that was open."""
    content = (
        "# Policy\n"
        "## Rail Travel\n"
        "### Standard process\n"
        "Body A.\n"
        "### Exceptions\n"
        "Body B.\n"
        "## Hotels\n"
        "### Booking\n"
        "Body C.\n"
    )
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    assert [s.section for s in sections] == [
        "Rail Travel > Standard process",
        "Rail Travel > Exceptions",
        "Hotels > Booking",
    ]


def test_markdown_duplicate_subsection_names_under_different_parents(kb_dir: Path):
    """Same subsection heading text under different parents should produce different, unambiguous paths."""
    content = "# Policy\n## Rail Travel\n### Booking\nBody A.\n## Hotels\n### Booking\nBody B.\n"
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    paths = [s.section for s in sections]
    assert paths == ["Rail Travel > Booking", "Hotels > Booking"]
    assert len(set(paths)) == 2  # the two "Booking" sections must be distinguishable


def test_markdown_page_is_none(kb_dir: Path):
    """Markdown files have no concept of pages, so `page` must always be None."""
    file_path = write_md(kb_dir, "doc.md", "# Heading\nSome text.\n")
    sections = load_markdown(file_path)
    assert all(s.page is None for s in sections)


def test_markdown_ignores_empty_sections(kb_dir: Path):
    """A heading immediately followed by another heading has no body text and should be dropped."""
    content = "# Heading With No Body\n\n## Next\nActual content.\n"
    file_path = write_md(kb_dir, "doc.md", content)

    sections = load_markdown(file_path)

    assert len(sections) == 1
    assert sections[0].section == "Next"


def test_pdf_multiple_level1_headings_on_same_page_stay_separate(mock_pdf, kb_dir: Path):
    """Two level-1 headings sharing one physical page must produce two distinct sections."""
    mock_pdf([
        fake_heading("Topic One", level=1, page_no=3),
        fake_text("Body of topic one.", page_no=3),
        fake_heading("Topic Two", level=1, page_no=3),
        fake_text("Body of topic two.", page_no=3),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert [s.section for s in sections] == ["Topic One", "Topic Two"]
    assert sections[0].text == "Body of topic one."
    assert sections[1].text == "Body of topic two."


def test_pdf_level2_heading_does_not_start_a_new_section(mock_pdf, kb_dir: Path):
    """A level-2 heading nested under a level-1 heading stays inside that one section."""
    mock_pdf([
        fake_heading("Carry-over", level=1, page_no=3),
        fake_heading("Requirements", level=2, page_no=3),
        fake_text("Employees should review balances.", page_no=3),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert len(sections) == 1
    assert sections[0].section == "Carry-over"
    assert "Requirements" in sections[0].text
    assert "Employees should review balances." in sections[0].text


def test_pdf_level3_heading_also_stays_in_active_primary_section(mock_pdf, kb_dir: Path):
    """A level-3 (or deeper) heading must not start a new section either - only level 1 does."""
    mock_pdf([
        fake_heading("Security", level=1, page_no=1),
        fake_heading("Passwords", level=2, page_no=1),
        fake_text("body", page_no=1),
        fake_heading("Password resets", level=3, page_no=1),
        fake_text("body", page_no=1),
        fake_heading("Device security", level=2, page_no=1),
        fake_text("body", page_no=1),
        fake_heading("Expenses", level=1, page_no=2),
        fake_text("expenses body", page_no=2),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert [s.section for s in sections] == ["Security", "Expenses"]
    assert sections[0].text == (
        "Passwords\n\nbody\n\nPassword resets\n\nbody\n\nDevice security\n\nbody"
    )
    assert sections[1].text == "expenses body"


def test_pdf_nested_body_preserves_reading_order(mock_pdf, kb_dir: Path):
    """Nested headings and their body text must appear in `text` in the original reading order."""
    mock_pdf([
        fake_heading("Carry-over", level=1, page_no=3),
        fake_heading("Requirements", level=2, page_no=3),
        fake_text("Requirements body.", page_no=3),
        fake_heading("Exceptions", level=2, page_no=3),
        fake_text("Exceptions body.", page_no=3),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert len(sections) == 1
    assert sections[0].text == (
        "Requirements\n\nRequirements body.\n\nExceptions\n\nExceptions body."
    )


def test_pdf_section_spanning_multiple_pages_stays_one_section(mock_pdf, kb_dir: Path):
    """A level-1 section whose nested content spans several physical pages remains one DocumentSection."""
    mock_pdf([
        fake_heading("Carry-over", level=1, page_no=3),
        fake_heading("Requirements", level=2, page_no=3),
        fake_text("First paragraph, on page 3.", page_no=3),
        fake_text("Second paragraph, spills onto page 4.", page_no=4),
        fake_heading("Exceptions", level=2, page_no=4),
        fake_text("Third paragraph, on page 4.", page_no=4),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert len(sections) == 1
    assert sections[0].section == "Carry-over"
    assert sections[0].page == 3  # the level-1 heading's page, not where its body ends
    assert "First paragraph" in sections[0].text
    assert "Second paragraph" in sections[0].text
    assert "Third paragraph" in sections[0].text


def test_pdf_new_level1_heading_closes_previous_section(mock_pdf, kb_dir: Path):
    """A new level-1 heading must close out the previous level-1 topic's nested content."""
    mock_pdf([
        fake_heading("Topic One", level=1, page_no=1),
        fake_heading("Sub", level=2, page_no=1),
        fake_text("Sub body.", page_no=1),
        fake_heading("Topic Two", level=1, page_no=2),
        fake_text("Topic two body.", page_no=2),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert [s.section for s in sections] == ["Topic One", "Topic Two"]
    assert "Sub" not in sections[1].text  # nesting from Topic One must not leak into Topic Two


def test_pdf_furniture_labeled_items_never_enter_body_text(mock_pdf, kb_dir: Path):
    """Even if a page-header/footer item were handed to load_pdf(), it must not end up in section text."""
    mock_pdf([
        fake_furniture("Repeated Document Title", page_no=1),
        fake_heading("Topic", level=1, page_no=1),
        fake_furniture("Page 1", page_no=1),
        fake_text("Real body content.", page_no=1),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert len(sections) == 1
    assert sections[0].text == "Real body content."
    assert "Repeated Document Title" not in sections[0].text
    assert "Page 1" not in sections[0].text


def test_pdf_title_item_excluded_from_section_metadata_and_body(mock_pdf, kb_dir: Path):
    """A DocItemLabel.TITLE item is document-level metadata: not a section, not repeated in any section."""
    mock_pdf([
        fake_title("Some Policy Document", page_no=1),
        fake_heading("Topic One", level=1, page_no=1),
        fake_text("Body one.", page_no=1),
        fake_heading("Topic Two", level=1, page_no=1),
        fake_text("Body two.", page_no=1),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert [s.section for s in sections] == ["Topic One", "Topic Two"]
    for s in sections:
        assert "Some Policy Document" not in (s.section or "")
        assert "Some Policy Document" not in s.text


def test_pdf_front_matter_before_first_heading_is_preserved(mock_pdf, kb_dir: Path):
    """Meaningful body content before any level-1 heading should become its own section, not be dropped."""
    mock_pdf([
        fake_text("Introductory blurb before any heading.", page_no=1),
        fake_heading("Topic", level=1, page_no=1),
        fake_text("Topic body.", page_no=1),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    assert len(sections) == 2
    assert sections[0].section is None
    assert sections[0].text == "Introductory blurb before any heading."
    assert sections[0].page == 1


def test_pdf_empty_primary_section_not_emitted_until_it_has_content(mock_pdf, kb_dir: Path):
    """A level-1 heading with no body/nested content of its own must not produce an empty section."""
    mock_pdf([
        fake_heading("Carry-over", level=1, page_no=1),  # no body directly under this heading
        fake_heading("Requirements", level=2, page_no=1),
        fake_text("Real content.", page_no=1),
    ])

    sections = load_pdf(kb_dir / "doc.pdf")

    # The level-2 heading's text and body still land inside the ONE
    # section "Carry-over" ends up producing once it has real content.
    assert len(sections) == 1
    assert sections[0].section == "Carry-over"
    assert sections[0].text == "Requirements\n\nReal content."


def test_pdf_deterministic_ordering(mock_pdf, kb_dir: Path):
    """Sections must come back in the same reading order the structured items were given in."""
    items = [
        fake_heading("A", level=1, page_no=1),
        fake_text("A body.", page_no=1),
        fake_heading("B", level=1, page_no=1),
        fake_text("B body.", page_no=1),
        fake_heading("C", level=1, page_no=2),
        fake_text("C body.", page_no=2),
    ]

    mock_pdf(items)
    first_run = [s.section for s in load_pdf(kb_dir / "doc.pdf")]
    mock_pdf(items)
    second_run = [s.section for s in load_pdf(kb_dir / "doc.pdf")]

    assert first_run == second_run == ["A", "B", "C"]


def test_pdf_smoke_real_docling_conversion(kb_dir: Path):
    """One real (unmocked) end-to-end run: load_pdf() must not crash on an actual PDF and must
    return well-formed DocumentSection objects. Does not assert specific heading detection -
    a tiny synthetic PDF has no reliable layout structure for Docling's model to key off."""
    file_path = kb_dir / "doc.pdf"
    make_pdf(file_path, ["First page content.", "Second page content."])

    sections = load_pdf(file_path)

    assert isinstance(sections, list)
    for section in sections:
        assert section.source == "doc.pdf"
        assert section.text.strip() != ""
        assert section.page is None or isinstance(section.page, int)


def test_load_directory_mixed_formats_and_unsupported_files(mock_pdf, kb_dir: Path):
    """load_directory should dispatch .md/.pdf to the right loader and silently skip anything else."""
    mock_pdf([fake_text("PDF content.", page_no=1)])
    write_md(kb_dir, "a.md", "# H\nMarkdown content.\n")
    (kb_dir / "b.pdf").touch()  # load_pdf() is mocked, so the file just needs to exist for iterdir()/suffix checks
    (kb_dir / "c.txt").write_text("should be ignored", encoding="utf-8")

    sections = load_directory(kb_dir)

    sources = {s.source for s in sections}
    assert sources == {"a.md", "b.pdf"}
    assert len(sections) == 2


def test_load_directory_deterministic_order(kb_dir: Path):
    """Results should be filename-sorted, not filesystem/iteration-order dependent."""
    write_md(kb_dir, "z.md", "# H\nZ content.\n")  # written out of order on purpose
    write_md(kb_dir, "a.md", "# H\nA content.\n")

    sections = load_directory(kb_dir)

    assert [s.source for s in sections] == ["a.md", "z.md"]
