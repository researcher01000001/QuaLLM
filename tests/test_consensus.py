import json
from pathlib import Path

import pandas as pd

from src.coding.consensus import (
    build_consensus_progression_quote_sets,
    build_consensus_quotes,
    build_consensus_records,
    build_superconsensus_quotes,
    write_consensus_json,
)
from src.core import process_log
from src.core.models import AIQuote, ConsensusQuote, DisplayColumn, Document, Row, Tag

_EMPTY_DISPLAY_COLUMN = DisplayColumn(source_column="", label="", values={})


def test_build_consensus_quotes_does_not_log_double_coding_conflicts(tmp_path):
    """build_consensus_quotes() is reused for intermediate progression
    snapshots (see build_consensus_progression_quote_sets()), not just the
    final decided consensus, so it must never itself flag double-coding
    conflicts -- that check belongs where the final consensus is actually
    decided (run.py::export_consensus_json()), not in this reused vote
    function. A conflicting pair reaching consensus here should not produce
    any process-log rows."""
    tag_a = Tag("A", can_co_occur=False)
    tag_b = Tag("B", can_co_occur=False)
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    iteration_quotes = [
        [AIQuote(text="Te", tags=[tag_a, tag_b], row=0, start=0, end=2)],
        [AIQuote(text="Te", tags=[tag_a, tag_b], row=0, start=0, end=2)],
    ]

    log_path = tmp_path / "process_log.xlsx"
    log = process_log.ProcessLog(log_path, step="run_main")
    with process_log.active(log):
        consensus = build_consensus_quotes(document, iteration_quotes, threshold=0.5)
    log.flush()

    assert len(consensus) == 1
    assert {tag.name for tag in consensus[0].tags} == {"A", "B"}

    df = pd.read_excel(log_path, sheet_name=process_log.DOUBLE_CODING_CONFLICTS)
    assert len(df) == 0


def test_consensus_keeps_only_tags_above_threshold_for_character():
    tag_a = Tag("A", "#111111")
    tag_b = Tag("B", "#222222")
    tag_c = Tag("C", "#333333")
    tag_d = Tag("D", "#444444")
    tag_e = Tag("E", "#555555")
    # "Ta" (not "Test"): the quote text "T" must be unambiguous in the row --
    # AIQuote.find_position()'s uniqueness-first search marks an ambiguous
    # match "position_unsure" (see test_models.py's find_position tests),
    # which isn't what this test is exercising.
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Ta", start_position=0, end_position=2)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )

    iteration_quotes = [
        [AIQuote(text="T", tags=[tag_a, tag_b], row=0)],
        [AIQuote(text="T", tags=[tag_b, tag_c], row=0)],
        [AIQuote(text="T", tags=[tag_d, tag_e], row=0)],
    ]

    consensus = build_consensus_quotes(document, iteration_quotes, threshold=0.5)

    assert len(consensus) == 1
    assert consensus[0].text == "T"
    assert consensus[0].start == 0
    assert consensus[0].end == 1
    assert [tag.name for tag in consensus[0].tags] == ["B"]


def test_consensus_includes_tags_at_threshold():
    tag = Tag("A", "#111111")
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    iteration_quotes = [
        [AIQuote(text="T", tags=[tag], row=0)],
        [],
    ]

    consensus = build_consensus_quotes(document, iteration_quotes, threshold=0.5)

    assert len(consensus) == 1
    assert consensus[0].text == "T"


def test_consensus_progression_builds_cumulative_quote_sets():
    tag = Tag("A", "#111111")
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    ai_quote_runs = {
        ("config", 0): {
            "test_doc": [AIQuote(text="T", tags=[tag], row=0)]
        },
        ("config", 1): {"test_doc": []},
    }

    progression = build_consensus_progression_quote_sets(
        documents={"test_doc": document},
        ai_quote_runs=ai_quote_runs,
        configuration_names=["config"],
        iterations=2,
        threshold=0.5,
    )

    assert progression[0]["config"]["test_doc"][0].text == "T"
    assert progression[1]["config"]["test_doc"][0].text == "T"


def test_consensus_compresses_adjacent_characters_with_same_tags():
    tag = Tag("A", "#111111")
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    iteration_quotes = [
        [AIQuote(text="Te", tags=[tag], row=0)],
        [AIQuote(text="Te", tags=[tag], row=0)],
        [],
    ]

    consensus = build_consensus_quotes(document, iteration_quotes, threshold=0.5)

    assert len(consensus) == 1
    assert consensus[0].text == "Te"
    assert consensus[0].start == 0
    assert consensus[0].end == 2
    assert consensus[0].source_iterations == 3


def test_superconsensus_votes_across_configuration_consensuses():
    tag_a = Tag("A", "#111111")
    tag_b = Tag("B", "#222222")
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    consensus_quotes = {
        "config-a": {
            "test_doc": [
                ConsensusQuote(text="Te", tags=[tag_a], start=0, end=2),
            ]
        },
        "config-b": {
            "test_doc": [
                ConsensusQuote(text="Te", tags=[tag_a], start=0, end=2),
                ConsensusQuote(text="s", tags=[tag_b], start=2, end=3),
            ]
        },
        "config-c": {"test_doc": []},
    }

    superconsensus = build_superconsensus_quotes(
        documents={"test_doc": document},
        consensus_quotes=consensus_quotes,
        configuration_names=["config-a", "config-b", "config-c"],
        threshold=0.5,
    )

    assert len(superconsensus["test_doc"]) == 1
    quote = superconsensus["test_doc"][0]
    assert quote.text == "Te"
    assert quote.start == 0
    assert quote.end == 2
    assert [tag.name for tag in quote.tags] == ["A"]
    assert quote.source_iterations == 3


def test_single_configuration_superconsensus_is_that_configurations_own_consensus():
    """With only one configuration there's nothing to vote *across*, but the
    single "vote" trivially clears any threshold <= 1.0, so the
    superconsensus is just a character-level copy of that configuration's
    own consensus -- not empty."""
    tag = Tag("A", "#111111")
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )

    superconsensus = build_superconsensus_quotes(
        documents={"test_doc": document},
        consensus_quotes={
            "config-a": {
                "test_doc": [ConsensusQuote(text="Te", tags=[tag], start=0, end=2)]
            }
        },
        configuration_names=["config-a"],
        threshold=0.5,
    )

    assert len(superconsensus["test_doc"]) == 1
    quote = superconsensus["test_doc"][0]
    assert quote.text == "Te"
    assert quote.start == 0
    assert quote.end == 2
    assert [tag.name for tag in quote.tags] == ["A"]


def test_zero_configurations_superconsensus_is_empty():
    document = Document(
        source_name="test_doc",
        rows=[Row(index=0, text="Test", start_position=0, end_position=4)],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )

    superconsensus = build_superconsensus_quotes(
        documents={"test_doc": document},
        consensus_quotes={},
        configuration_names=[],
        threshold=0.5,
    )

    assert superconsensus == {}


def test_build_consensus_records_resolves_row_for_consensus_quote():
    tag = Tag("A", "#111111")
    document = Document(
        source_name="test_doc",
        rows=[
            Row(index=0, text="Hello", start_position=0, end_position=5),
            Row(index=1, text="World", start_position=6, end_position=11),
        ],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    quotes_by_doc = {
        "test_doc": [ConsensusQuote(text="World", tags=[tag], start=6, end=11)]
    }

    records = build_consensus_records({"test_doc": document}, quotes_by_doc)

    assert len(records) == 1
    record = records[0]
    assert record["phrase"] == "World"
    assert record["tags"] == ["A"]
    assert record["source"] == {"source_name": "test_doc", "row": 1}
    assert record["start"] == 6
    assert record["end"] == 11
    assert record["source_iterations"] is None or record["source_iterations"] == 1


def test_build_consensus_records_uses_ai_quote_row_directly():
    tag = Tag("A", "#111111")
    quote = AIQuote(text="Hi", tags=[tag], row=3, start=10, end=12)
    quotes_by_doc = {"test_doc": [quote]}

    records = build_consensus_records({}, quotes_by_doc)

    assert records[0]["source"] == {"source_name": "test_doc", "row": 3}
    assert records[0]["source_iterations"] == 1


def test_build_consensus_records_uses_source_iterations_from_consensus_quote():
    tag = Tag("A", "#111111")
    quote = ConsensusQuote(text="Hi", tags=[tag], start=0, end=2, source_iterations=3)

    records = build_consensus_records({}, {"test_doc": [quote]})

    assert records[0]["source_iterations"] == 3


def test_write_consensus_json_round_trips(tmp_path: Path):
    output_file = tmp_path / "nested" / "consensus.json"
    records = [{"phrase": "Hi", "tags": ["A"], "start": 0, "end": 2}]

    write_consensus_json(records, output_file)

    assert output_file.exists()
    with open(output_file, encoding="utf-8") as f:
        assert json.load(f) == records
