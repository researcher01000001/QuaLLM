from pathlib import Path

import pandas as pd

from src.core import process_log
from src.core.models import DisplayColumn, Document, Row, Tag
from src.post_processing.tag_export import (
    append_ordered_tags_to_xlsx,
    build_ordered_tags,
    collapse_to_consensus_schema,
    find_still_undefined_tag_names_in_rows,
)


_SINGLE_ROW_DISPLAY = DisplayColumn(source_column="position", label="Paragraph", values={0: "1"})
_TWO_ROW_DISPLAY = DisplayColumn(
    source_column="position", label="Paragraph", values={0: "1", 1: "2"}
)


def test_collapse_to_consensus_schema_regroups_by_current_position():
    records = [
        {
            "id": "1",
            "group_id": "g1",
            "source_name": "doc",
            "row": 0,
            "phrase": "span",
            "tag": "TagA",
            "start": 10,
            "end": 20,
            "status": "original",
        },
        {
            "id": "2",
            "group_id": "g1",
            "source_name": "doc",
            "row": 0,
            "phrase": "span",
            "tag": "TagB",
            "start": 10,
            "end": 20,
            "status": "original",
        },
    ]

    result = collapse_to_consensus_schema(records)

    assert len(result) == 1
    assert result[0]["tags"] == ["TagA", "TagB"]
    assert result[0]["start"] == 10
    assert result[0]["end"] == 20


def test_collapse_to_consensus_schema_splits_diverged_positions():
    # Two rows that started on the same span (shared group_id), but one was
    # since moved to a different position -- they must export separately.
    records = [
        {
            "id": "1",
            "group_id": "g1",
            "source_name": "doc",
            "row": 0,
            "phrase": "span",
            "tag": "TagA",
            "start": 10,
            "end": 20,
            "status": "original",
        },
        {
            "id": "2",
            "group_id": "g1",
            "source_name": "doc",
            "row": 0,
            "phrase": "moved span",
            "tag": "TagB",
            "start": 50,
            "end": 60,
            "status": "changed_position",
        },
    ]

    result = collapse_to_consensus_schema(records)

    assert len(result) == 2
    positions = {(r["start"], r["end"]): r["tags"] for r in result}
    assert positions[(10, 20)] == ["TagA"]
    assert positions[(50, 60)] == ["TagB"]


def test_collapse_to_consensus_schema_drops_deleted_rows():
    records = [
        {
            "id": "1",
            "group_id": "g1",
            "source_name": "doc",
            "row": 0,
            "phrase": "span",
            "tag": "TagA",
            "start": 10,
            "end": 20,
            "status": "deleted",
        },
    ]

    assert collapse_to_consensus_schema(records) == []


def _consensus_row(source_name, row, phrase, start, end, tags) -> dict:
    return {
        "phrase": phrase,
        "tags": tags,
        "source": {"source_name": source_name, "row": row},
        "start": start,
        "end": end,
    }


def _patch_get_document_key(monkeypatch) -> None:
    """Stand-in for document_identity.get_document_key() in tests: extracts
    cycle_id from names like '45_no advice_cycle_63'."""

    def get_document_key(doc_name: str) -> dict | None:
        try:
            return {"cycle_id": int(doc_name.rsplit("_", 1)[-1])}
        except ValueError:
            return None

    monkeypatch.setattr("src.dataset.document_identity.get_document_key", get_document_key)


def test_build_ordered_tags_sorts_by_start_then_span_length(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="x" * 50, start_position=0, end_position=50)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    rows = [
        _consensus_row("45_no advice_cycle_63", 0, "b", 20, 25, ["Fixed"]),
        _consensus_row("45_no advice_cycle_63", 0, "a", 10, 30, ["Long"]),  # same start, longer
        _consensus_row("45_no advice_cycle_63", 0, "c", 10, 15, ["Short"]),  # same start, shorter
    ]
    tags_by_name = {
        "Fixed": Tag(name="Fixed", can_co_occur=False),
        "Long": Tag(name="Long", can_co_occur=False),
        "Short": Tag(name="Short", can_co_occur=False),
    }
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(documents, rows, tags_by_name)

    assert len(result) == 1
    assert result[0]["tags"] == ["Short", "Long", "Fixed"]
    assert result[0]["identity"] == {"cycle_id": 63}
    assert result[0]["row"] == 0


def test_build_ordered_tags_puts_non_double_codable_tags_first(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="x" * 50, start_position=0, end_position=50)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    rows = [_consensus_row("45_no advice_cycle_63", 0, "span", 10, 20, ["Flexible", "Fixed"])]
    tags_by_name = {
        "Flexible": Tag(name="Flexible", can_co_occur=True),
        "Fixed": Tag(name="Fixed", can_co_occur=False),
    }
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(documents, rows, tags_by_name)

    assert result[0]["tags"][0] == "Fixed"
    assert result[0]["tags"][1] == "Flexible"


def test_build_ordered_tags_treats_undefined_tags_as_flexible(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="x" * 50, start_position=0, end_position=50)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    rows = [_consensus_row("45_no advice_cycle_63", 0, "span", 10, 20, ["Fixed", "NotInCodebook"])]
    tags_by_name = {"Fixed": Tag(name="Fixed", can_co_occur=False)}
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(documents, rows, tags_by_name)

    # "NotInCodebook" isn't subject to the double-coding rule -> flexible group -> after Fixed
    assert result[0]["tags"][0] == "Fixed"
    assert result[0]["tags"][1] == "NotInCodebook"


def test_build_ordered_tags_is_deterministic_across_runs(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="some paragraph text here", start_position=0, end_position=24)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    rows = [_consensus_row("45_no advice_cycle_63", 0, "a phrase", 5, 15, ["TagA", "TagB", "TagC"])]
    tags_by_name = {name: Tag(name=name, can_co_occur=True) for name in ("TagA", "TagB", "TagC")}
    _patch_get_document_key(monkeypatch)

    first = build_ordered_tags(documents, rows, tags_by_name)
    second = build_ordered_tags(documents, rows, tags_by_name)

    assert first == second
    assert sorted(first[0]["tags"]) == ["TagA", "TagB", "TagC"]


def test_build_ordered_tags_includes_paragraphs_with_no_tags(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[
                Row(index=0, text="x" * 50, start_position=0, end_position=50),
                Row(index=1, text="y" * 50, start_position=51, end_position=101),
            ],
            display_column=_TWO_ROW_DISPLAY,
        )
    }
    rows = [_consensus_row("45_no advice_cycle_63", 0, "span", 10, 20, ["TagA"])]
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(documents, rows, {"TagA": Tag(name="TagA")})

    assert len(result) == 2
    para1 = next(r for r in result if r["row"] == 1)
    assert para1["tags"] == []


def test_build_ordered_tags_skips_documents_with_unresolvable_identity(monkeypatch):
    documents = {
        "not-a-valid-name": Document(
            source_name="not-a-valid-name",
            rows=[Row(index=0, text="x" * 10, start_position=0, end_position=10)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    _patch_get_document_key(monkeypatch)

    assert build_ordered_tags(documents, [], {}) == []


def test_build_ordered_tags_marks_excluded_paragraphs(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[
                Row(index=0, text="x" * 50, start_position=0, end_position=50),
                Row(index=1, text="y" * 50, start_position=51, end_position=101),
            ],
            display_column=_TWO_ROW_DISPLAY,
        )
    }
    rows = [_consensus_row("45_no advice_cycle_63", 0, "span", 10, 20, ["TagA"])]
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(
        documents, rows, {"TagA": Tag(name="TagA")}, frozenset({("45_no advice_cycle_63", 1)})
    )

    by_row = {r["row"]: r for r in result}
    assert by_row[0]["excluded"] is False
    assert by_row[1]["excluded"] is True


def test_build_ordered_tags_defaults_excluded_false(monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="x" * 50, start_position=0, end_position=50)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    _patch_get_document_key(monkeypatch)

    result = build_ordered_tags(documents, [], {"TagA": Tag(name="TagA")})

    assert result[0]["excluded"] is False


def test_build_ordered_tags_logs_excluded_paragraph(tmp_path: Path, monkeypatch):
    documents = {
        "45_no advice_cycle_63": Document(
            source_name="45_no advice_cycle_63",
            rows=[Row(index=0, text="x" * 50, start_position=0, end_position=50)],
            display_column=_SINGLE_ROW_DISPLAY,
        )
    }
    _patch_get_document_key(monkeypatch)

    log_file = tmp_path / "process_log.xlsx"
    log = process_log.ProcessLog(
        log_file,
        step=process_log.STEP_POST_PROCESS_STEP3,
        owned_steps={process_log.STEP_POST_PROCESS_STEP3},
    )
    with process_log.active(log):
        build_ordered_tags(
            documents, [], {}, frozenset({("45_no advice_cycle_63", 0)})
        )
    log.flush()

    df = pd.read_excel(log_file, sheet_name=process_log.PARAGRAPH_EXCLUSIONS)
    assert len(df) == 1
    assert df.iloc[0]["source_name"] == "45_no advice_cycle_63"
    assert df.iloc[0]["row"] == 0


def test_find_still_undefined_tag_names_in_rows_skips_excluded():
    rows = [
        {
            "source": {"source_name": "d", "row": 0},
            "tags": ["NotInCodebook"],
            "start": 0,
            "end": 5,
        },
        {
            "source": {"source_name": "d", "row": 1},
            "tags": ["AlsoNotInCodebook"],
            "start": 0,
            "end": 5,
        },
    ]

    result = find_still_undefined_tag_names_in_rows(rows, {}, frozenset({("d", 0)}))

    assert [r["row"] for r in result] == [1]


def test_append_ordered_tags_to_xlsx(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.project.DOCUMENT_KEY_COLUMNS", ["cycle_id"])
    monkeypatch.setattr("src.config.project.DOCUMENT_ORDER_COLUMN", "position")
    source_xlsx = tmp_path / "output.xlsx"
    pd.DataFrame(
        {
            "cycle_id": [63, 63, 99],
            "position": [1, 2, 1],
            "other_column": ["a", "b", "c"],
        }
    ).to_excel(source_xlsx, index=False)

    ordered_tags = [
        {
            "source_name": "d",
            "identity": {"cycle_id": 63},
            "row": 0,
            "display_value": "1",
            "tags": ["TagA", "TagB"],
        },
        {
            "source_name": "d",
            "identity": {"cycle_id": 63},
            "row": 1,
            "display_value": "2",
            "tags": [],
        },
    ]
    output_xlsx = tmp_path / "output_with_tags.xlsx"

    matched = append_ordered_tags_to_xlsx(
        ordered_tags,
        source_xlsx,
        output_xlsx,
        "ordered_tags",
    )

    assert matched == 2
    result = pd.read_excel(output_xlsx)
    row63_1 = result[(result["cycle_id"] == 63) & (result["position"] == 1)].iloc[0]
    row63_2 = result[(result["cycle_id"] == 63) & (result["position"] == 2)].iloc[0]
    row99_1 = result[(result["cycle_id"] == 99) & (result["position"] == 1)].iloc[0]
    assert row63_1["ordered_tags"] == "TagA, TagB"
    # An empty string round-trips through Excel as a blank cell (read back as
    # NaN by pandas) -- both "matched but no tags" and "no matching entry at
    # all" look the same in the saved file, which is the desired behavior.
    assert pd.isna(row63_2["ordered_tags"])
    assert pd.isna(row99_1["ordered_tags"])


def test_append_ordered_tags_to_xlsx_drops_excluded_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.project.DOCUMENT_KEY_COLUMNS", ["cycle_id"])
    monkeypatch.setattr("src.config.project.DOCUMENT_ORDER_COLUMN", "position")
    source_xlsx = tmp_path / "output.xlsx"
    pd.DataFrame(
        {
            "cycle_id": [63, 63, 99],
            "position": [1, 2, 1],
            "other_column": ["a", "b", "c"],
        }
    ).to_excel(source_xlsx, index=False)

    ordered_tags = [
        {
            "source_name": "d",
            "identity": {"cycle_id": 63},
            "row": 0,
            "display_value": "1",
            "tags": ["TagA"],
            "excluded": False,
        },
        {
            "source_name": "d",
            "identity": {"cycle_id": 63},
            "row": 1,
            "display_value": "2",
            "tags": ["TagB"],
            "excluded": True,
        },
        {
            "source_name": "e",
            "identity": {"cycle_id": 99},
            "row": 0,
            "display_value": "1",
            "tags": [],
            "excluded": False,
        },
    ]
    output_xlsx = tmp_path / "output_with_tags.xlsx"

    matched = append_ordered_tags_to_xlsx(
        ordered_tags,
        source_xlsx,
        output_xlsx,
        "ordered_tags",
    )

    result = pd.read_excel(output_xlsx)
    # The excluded row (cycle_id=63, position=2) must be entirely absent --
    # not just left with a blank tags cell.
    assert len(result) == 2
    assert not (
        (result["cycle_id"] == 63) & (result["position"] == 2)
    ).any()
    row63_1 = result[(result["cycle_id"] == 63) & (result["position"] == 1)].iloc[0]
    assert row63_1["ordered_tags"] == "TagA"
    # matched only reflects the two non-excluded rows, both of which matched.
    assert matched == 2
