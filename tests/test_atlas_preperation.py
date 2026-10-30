"""Unit tests for document_content.row_text_value()"""

import dataclasses
import math

import pandas as pd

from scripts.atlas_preparation import _write_document_driven, _write_random
from src.config import project as config_project
from src.dataset import document_content
from src.dataset.document_content import row_text_value


def test_text_value_nan_becomes_empty_string():
    assert row_text_value(math.nan) == ""


def test_text_value_plain_string_unchanged():
    assert row_text_value("plain transcript text") == "plain transcript text"


def test_text_value_collapses_embedded_newline():
    assert row_text_value("first line\nsecond line") == "first line second line"


def test_text_value_collapses_embedded_crlf():
    assert row_text_value("first line\r\nsecond line") == "first line second line"


def test_text_value_trims_leading_and_trailing_whitespace():
    assert row_text_value("  plain transcript text  ") == "plain transcript text"


def test_text_value_trims_each_line_before_joining():
    assert row_text_value("first line \n second line") == "first line second line"


def test_text_value_trims_leading_and_trailing_blank_lines():
    assert row_text_value("\nfirst line\nsecond line\n") == "first line second line"


def _write_output_xlsx(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet", index=False)


def _reset_cache(monkeypatch, xlsx_path) -> None:
    monkeypatch.setattr(
        document_content,
        "META_OUTPUT_FILE",
        {"filename": xlsx_path, "sheetname": "Sheet"},
    )
    monkeypatch.setattr(document_content, "_document_rows_cache", None)


def test_write_random_groups_by_cluster_and_orders_by_dataset_order(tmp_path, monkeypatch):
    xlsx = tmp_path / "output.xlsx"
    _write_output_xlsx(
        xlsx,
        [
            # Two different documents, both contributing to the same
            # "validation_advice" cluster file, out of insertion order.
            {
                "cycle_id": 1, "condition": "advice", "think_aloud_id": 7, "position": 1,
                "adjusted_transcript": "second", "dataset": "validation_advice", "dataset_order": 2,
            },
            {
                "cycle_id": 2, "condition": "advice", "think_aloud_id": 8, "position": 1,
                "adjusted_transcript": "first", "dataset": "validation_advice", "dataset_order": 1,
            },
            # A row in a different cluster must not leak into "advice"'s file.
            {
                "cycle_id": 3, "condition": "no advice", "think_aloud_id": 9, "position": 1,
                "adjusted_transcript": "other cluster", "dataset": "validation_no advice", "dataset_order": 1,
            },
            # An excluded row must never appear anywhere.
            {
                "cycle_id": 4, "condition": "advice", "think_aloud_id": 10, "position": 1,
                "adjusted_transcript": "excluded row", "dataset": "excluded", "dataset_order": None,
            },
        ],
    )
    _reset_cache(monkeypatch, xlsx)
    output_dir = tmp_path / "validation"
    output_dir.mkdir()

    entries = _write_random("validation", output_dir)

    entries_by_title = dict(entries)
    assert entries_by_title["validation_advice"] == ["first", "second"]
    assert entries_by_title["validation_no advice"] == ["other cluster"]
    assert (output_dir / "validation_advice.txt").read_text(encoding="utf-8") == "first\nsecond"


def test_write_document_driven_keeps_only_prefixed_rows(tmp_path, monkeypatch):
    xlsx = tmp_path / "output.xlsx"
    _write_output_xlsx(
        xlsx,
        [
            {
                "cycle_id": 1, "condition": "no advice", "think_aloud_id": 7, "position": 1,
                "adjusted_transcript": "keep me", "dataset": "training_no advice",
            },
            {
                "cycle_id": 1, "condition": "no advice", "think_aloud_id": 7, "position": 2,
                "adjusted_transcript": "drop me", "dataset": "excluded",
            },
        ],
    )
    _reset_cache(monkeypatch, xlsx)
    monkeypatch.setitem(
        config_project.RUN_TYPE_CONFIGS,
        "training",
        dataclasses.replace(config_project.RUN_TYPE_CONFIGS["training"], selection_mode="document_driven"),
    )
    output_dir = tmp_path / "training"
    output_dir.mkdir()

    entries = _write_document_driven("training", output_dir)

    assert len(entries) == 1
    doc_name, texts = entries[0]
    assert texts == ["keep me"]
