"""Shared pytest fixtures for the QuaLLM test suite."""

from pathlib import Path

import pytest

from src.core.models import AIQuote, AtlasQuote, CodedRow, DisplayColumn, Document, Row, Tag


@pytest.fixture
def sample_tags() -> list[Tag]:
    """Return a small set of tags for testing."""
    return [
        Tag(name="Monitoring", color="#ff0000", group="Metacognition"),
        Tag(name="Non Verbalised Thinking", color="#00ff00", group="Cognitive"),
        Tag(name="Decision", color="#0000ff", group="Action"),
    ]


@pytest.fixture
def mock_atlas_quote(sample_tags) -> AtlasQuote:
    """Return a single ATLAS quote with known position."""
    return AtlasQuote(
        text="The patient shows clear signs",
        tags=[sample_tags[0], sample_tags[1]],
        start=10,
        end=39,
    )


@pytest.fixture
def mock_ai_quote(sample_tags) -> AIQuote:
    """Return a single AI quote tied to row 0."""
    return AIQuote(
        text="clear signs",
        tags=[sample_tags[0]],
        row=0,
    )


@pytest.fixture
def mock_paragraphs() -> list[Row]:
    """Return two rows (paragraphs) with known character offsets."""
    return [
        Row(
            index=0,
            text="The patient shows clear signs of improvement.",
            start_position=0,
            end_position=45,
        ),
        Row(
            index=1,
            text="However, further tests are needed.",
            start_position=46,
            end_position=80,
        ),
    ]


@pytest.fixture
def coded_paragraph(mock_paragraphs, sample_tags) -> CodedRow:
    """Return a CodedRow with two coders' quotes."""
    para = mock_paragraphs[0]
    q1 = AtlasQuote(
        text="patient shows",
        tags=[sample_tags[0]],
        start=4,
        end=17,
    )
    q2 = AIQuote(
        text="patient shows",
        tags=[sample_tags[0]],
        row=0,
    )
    # Resolve AIQuote position so metrics can run
    q2.find_position(para.text, para.start_position)

    return CodedRow(
        index=para.index,
        text=para.text,
        start_position=para.start_position,
        end_position=para.end_position,
        quotes_1=[q1],
        quotes_2=[q2],
    )


@pytest.fixture
def mock_document(mock_paragraphs) -> Document:
    """Return a Document with two plain rows."""
    return Document(
        source_name="45_no_advice_cycle_63",
        rows=mock_paragraphs,
        display_column=DisplayColumn(
            source_column="position",
            label="Paragraph",
            values={row.index: str(row.index + 1) for row in mock_paragraphs},
        ),
    )


@pytest.fixture
def tmp_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file with two paragraphs for loading tests."""
    txt = tmp_path / "test_doc.txt"
    txt.write_text(
        "The patient shows clear signs of improvement.\n"
        "However, further tests are needed.\n",
        encoding="utf-8",
    )
    return txt
