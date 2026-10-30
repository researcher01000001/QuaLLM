"""Unit tests for src.dataset.selection's DATASET_COLUMN row-membership parsing"""

import dataclasses

import pandas as pd

from src.config import project as config_project
from src.dataset.selection import (
    document_driven_row_filter,
    is_excluded_row,
    is_validation_row,
    parse_dataset_value,
    random_mode_row_filter,
    row_filter_for_txt_subfolder,
)


class TestParseDatasetValue:
    """DATASET_COLUMN's grammar (see scripts/sample_selector.py's module
    docstring): "excluded", or "[training|validation_]{cluster}" -- the
    row's order within its (type, cluster) group lives in the separate
    RANDOM_DATASET_ORDER_COLUMN, not encoded in this string."""

    def test_parses_training_value(self):
        assert parse_dataset_value("training_advice") == {
            "type": "training",
            "cluster": "advice",
        }

    def test_parses_validation_value(self):
        assert parse_dataset_value("validation_advice") == {
            "type": "validation",
            "cluster": "advice",
        }

    def test_parses_untagged_value_with_no_type_prefix(self):
        assert parse_dataset_value("advice") == {
            "type": None,
            "cluster": "advice",
        }

    def test_cluster_may_contain_spaces(self):
        assert parse_dataset_value("validation_no advice") == {
            "type": "validation",
            "cluster": "no advice",
        }
        assert parse_dataset_value("advice with annotation") == {
            "type": None,
            "cluster": "advice with annotation",
        }

    def test_excluded_returns_none(self):
        assert parse_dataset_value("excluded") is None

    def test_empty_value_returns_none(self):
        assert parse_dataset_value("") is None

    def test_type_prefix_with_empty_cluster_returns_none(self):
        assert parse_dataset_value("training_") is None
        assert parse_dataset_value("validation_") is None


class TestRowPredicates:
    """is_excluded_row()/is_validation_row()/document_driven_row_filter() --
    the shared DATASET_COLUMN predicates used by both
    scripts/atlas_preperation.py (writing .txt files) and src/coding/ai_coding.py
    (resolving a .txt file's lines back to source rows)."""

    def test_is_excluded_row(self):
        assert is_excluded_row("excluded") is True
        assert is_excluded_row("training_advice") is False
        assert is_excluded_row(None) is False

    def test_is_validation_row(self):
        assert is_validation_row("validation_advice") is True
        assert is_validation_row("validation_no advice") is True
        assert is_validation_row("training_advice") is False
        assert is_validation_row("excluded") is False
        assert is_validation_row(None) is False

    def test_document_driven_row_filter_matches_only_its_own_prefix(self):
        row_filter = document_driven_row_filter("training")
        assert row_filter("training_advice") is True
        assert row_filter("validation_advice") is False
        assert row_filter("excluded") is False
        assert row_filter(None) is False


class TestRowFilterForTxtSubfolder:
    def test_full_and_full_without_validation_are_always_document_driven(self):
        assert row_filter_for_txt_subfolder("full") is not None
        assert row_filter_for_txt_subfolder("full")("excluded") is False
        assert row_filter_for_txt_subfolder("full")("training_advice") is True

        fwv_filter = row_filter_for_txt_subfolder("full_without_validation")
        assert fwv_filter is not None
        assert fwv_filter("validation_advice") is False
        assert fwv_filter("training_advice") is True
        assert fwv_filter("excluded") is False

    def test_document_driven_subfolder_resolves_a_prefix_filter(self, monkeypatch):
        monkeypatch.setitem(
            config_project.RUN_TYPE_CONFIGS,
            "training",
            dataclasses.replace(
                config_project.RUN_TYPE_CONFIGS["training"], selection_mode="document_driven"
            ),
        )
        row_filter = row_filter_for_txt_subfolder("training")
        assert row_filter is not None
        assert row_filter("training_advice") is True
        assert row_filter("validation_advice") is False

    def test_random_mode_subfolder_returns_none(self, monkeypatch):
        monkeypatch.setitem(
            config_project.RUN_TYPE_CONFIGS,
            "validation",
            dataclasses.replace(config_project.RUN_TYPE_CONFIGS["validation"], selection_mode="random"),
        )
        assert row_filter_for_txt_subfolder("validation") is None

    def test_unrecognized_subfolder_returns_none(self):
        assert row_filter_for_txt_subfolder("not-a-real-subfolder") is None


class TestRandomModeRowFilter:
    def _row(self, dataset_value, order) -> pd.Series:
        return pd.Series({"dataset": dataset_value, "dataset_order": order})

    def test_matches_exact_type_and_cluster_with_a_real_order(self):
        row_filter = random_mode_row_filter("validation", "advice")
        assert row_filter(self._row("validation_advice", 3)) is True

    def test_rejects_a_different_cluster(self):
        row_filter = random_mode_row_filter("validation", "advice")
        assert row_filter(self._row("validation_no advice", 3)) is False

    def test_rejects_a_different_type(self):
        row_filter = random_mode_row_filter("validation", "advice")
        assert row_filter(self._row("training_advice", 3)) is False

    def test_rejects_blank_order(self):
        row_filter = random_mode_row_filter("validation", "advice")
        assert row_filter(self._row("validation_advice", float("nan"))) is False

    def test_rejects_excluded_and_non_string_values(self):
        row_filter = random_mode_row_filter("validation", "advice")
        assert row_filter(self._row("excluded", 3)) is False
        assert row_filter(self._row(None, 3)) is False
