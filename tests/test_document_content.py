"""Unit tests for src.dataset.document_content.display_column_for()"""

import pandas as pd

from src.dataset import document_content


def _write_output_xlsx(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet", index=False)


def _reset_cache(monkeypatch, xlsx_path) -> None:
    # document_content.py did `from src.config.project import META_OUTPUT_FILE`,
    # so it holds its own bound name -- patching src.config.project.META_OUTPUT_FILE
    # would not affect it; the module's own binding must be patched instead.
    monkeypatch.setattr(
        document_content,
        "META_OUTPUT_FILE",
        {"filename": xlsx_path, "sheetname": "Sheet"},
    )
    monkeypatch.setattr(document_content, "_document_rows_cache", None)


def test_display_column_for_reads_real_position_value_not_index_plus_one(
    tmp_path, monkeypatch
):
    """The whole point: DOCUMENT_ORDER_COLUMN doesn't have to be contiguous from 1 --
    display_column_for() must read whatever is actually there, not index + 1."""
    xlsx = tmp_path / "output.xlsx"
    _write_output_xlsx(
        xlsx,
        [
            {
                "cycle_id": 1,
                "condition": "no advice",
                "think_aloud_id": 7,
                "position": 5,
                "adjusted_transcript": "first",
            },
            {
                "cycle_id": 1,
                "condition": "no advice",
                "think_aloud_id": 7,
                "position": 9,
                "adjusted_transcript": "second",
            },
        ],
    )
    _reset_cache(monkeypatch, xlsx)

    doc_name = document_content.document_names()[0]

    display_column = document_content.display_column_for(doc_name)

    assert display_column.source_column == "position"
    assert display_column.label == "Paragraph"
    assert display_column.values == {0: "5", 1: "9"}


def test_display_column_for_omits_out_of_range_rows(tmp_path, monkeypatch):
    xlsx = tmp_path / "output.xlsx"
    _write_output_xlsx(
        xlsx,
        [
            {
                "cycle_id": 1,
                "condition": "no advice",
                "think_aloud_id": 7,
                "position": 1,
                "adjusted_transcript": "only row",
            },
        ],
    )
    _reset_cache(monkeypatch, xlsx)

    doc_name = document_content.document_names()[0]

    display_column = document_content.display_column_for(doc_name)

    assert display_column.values == {0: "1"}
    assert 5 not in display_column.values


def test_display_column_for_returns_empty_values_for_unknown_document(
    tmp_path, monkeypatch
):
    xlsx = tmp_path / "output.xlsx"
    _write_output_xlsx(
        xlsx,
        [
            {
                "cycle_id": 1,
                "condition": "no advice",
                "think_aloud_id": 7,
                "position": 1,
                "adjusted_transcript": "row",
            },
        ],
    )
    _reset_cache(monkeypatch, xlsx)

    display_column = document_content.display_column_for("not-a-real-document")

    assert display_column.source_column == "position"
    assert display_column.label == "Paragraph"
    assert display_column.values == {}


class TestRowsForDocument:
    def test_filters_rows_by_dataset_column_and_keeps_order(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 2,
                    "adjusted_transcript": "second",
                    "dataset": "excluded",
                },
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "first",
                    "dataset": "training_no advice",
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)
        doc_name = document_content.document_names()[0]

        rows = document_content.rows_for_document(doc_name, lambda v: v != "excluded")

        assert rows is not None
        assert list(rows["adjusted_transcript"]) == ["first"]

    def test_no_dataset_column_treats_every_row_as_none(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "only row",
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)
        doc_name = document_content.document_names()[0]

        rows = document_content.rows_for_document(doc_name, lambda v: v is None)

        assert rows is not None
        assert len(rows) == 1

    def test_returns_none_for_unknown_document(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)

        assert document_content.rows_for_document("not-a-real-document", lambda v: True) is None


class TestKnownConditions:
    def test_returns_distinct_conditions_in_first_seen_order(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
                {
                    "cycle_id": 1,
                    "condition": "advice",
                    "think_aloud_id": 8,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
                {
                    # A different document (distinct think_aloud_id) sharing
                    # the first document's condition -- must not be counted
                    # twice.
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 9,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)

        assert document_content.known_conditions() == ["no advice", "advice"]

    def test_returns_empty_list_for_empty_spreadsheet(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
            ],
        )
        # Drop every row after writing, so the sheet exists but is empty.
        pd.DataFrame(
            columns=["cycle_id", "condition", "think_aloud_id", "position", "adjusted_transcript"]
        ).to_excel(xlsx, sheet_name="Sheet", index=False)
        _reset_cache(monkeypatch, xlsx)

        assert document_content.known_conditions() == []


class TestAllRowsMatching:
    def test_matches_across_documents_and_sorts_numerically(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "doc1-row",
                    "dataset": "validation_advice",
                    "dataset_order": 10,
                },
                {
                    "cycle_id": 2,
                    "condition": "no advice",
                    "think_aloud_id": 8,
                    "position": 1,
                    "adjusted_transcript": "doc2-row",
                    "dataset": "validation_advice",
                    "dataset_order": 2,
                },
                {
                    "cycle_id": 3,
                    "condition": "no advice",
                    "think_aloud_id": 9,
                    "position": 1,
                    "adjusted_transcript": "unrelated",
                    "dataset": "excluded",
                    "dataset_order": None,
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)

        rows = document_content.all_rows_matching(
            lambda row: row["dataset"] == "validation_advice",
            sort_column="dataset_order",
        )

        # Numeric sort (2 before 10), not lexicographic ("10" before "2").
        assert list(rows["adjusted_transcript"]) == ["doc2-row", "doc1-row"]

    def test_no_matches_returns_empty_dataframe(self, tmp_path, monkeypatch):
        xlsx = tmp_path / "output.xlsx"
        _write_output_xlsx(
            xlsx,
            [
                {
                    "cycle_id": 1,
                    "condition": "no advice",
                    "think_aloud_id": 7,
                    "position": 1,
                    "adjusted_transcript": "row",
                },
            ],
        )
        _reset_cache(monkeypatch, xlsx)

        rows = document_content.all_rows_matching(lambda row: False)

        assert rows.empty
