"""Unit tests for src.dataset.document_identity's document identity / condition parsing"""

import pytest

from src.dataset import document_content
from src.dataset.document_identity import (
    condition_for,
    condition_match,
    get_document_key,
    parse_document_name,
)

_KNOWN_CONDITIONS = ["advice", "advice with annotation", "no advice"]


@pytest.fixture(autouse=True)
def _mock_known_conditions(monkeypatch):
    monkeypatch.setattr(document_content, "known_conditions", lambda: _KNOWN_CONDITIONS)


class TestGetDocumentKey:
    def test_parses_think_aloud_id_and_condition_from_valid_name(self):
        assert get_document_key("45_no advice_cycle_63") == {
            "think_aloud_id": 45,
            "condition": "no advice",
        }

    def test_parses_other_conditions_too(self):
        assert get_document_key("12_advice_cycle_7") == {
            "think_aloud_id": 12,
            "condition": "advice",
        }
        assert get_document_key("8_advice with annotation_cycle_99") == {
            "think_aloud_id": 8,
            "condition": "advice with annotation",
        }

    def test_returns_none_for_unrecognised_shape(self):
        assert get_document_key("not-a-valid-name") is None

    def test_returns_none_for_random_mode_sample_file(self):
        """A random-mode training/validation sample file (see
        scripts/atlas_preperation.py) spans many different participants, so
        it has no single think_aloud_id -- get_document_key() must not
        invent one."""
        assert get_document_key("advice") is None
        assert get_document_key("no advice") is None
        assert get_document_key("validation_advice") is None
        assert get_document_key("training_no advice") is None


class TestConditionFor:
    def test_returns_raw_condition_value(self):
        assert condition_for("45_no advice_cycle_63") == "no advice"
        assert condition_for("12_advice_cycle_7") == "advice"
        assert condition_for("8_advice with annotation_cycle_99") == "advice with annotation"

    def test_returns_none_for_unrecognised_shape(self):
        assert condition_for("not-a-valid-name") is None

    def test_returns_condition_for_random_mode_sample_file(self):
        """scripts/atlas_preperation.py names a random-mode sample file
        after its document-cluster value alone (e.g. "advice.txt") --
        condition_for() must still recover the condition from that bare name."""
        assert condition_for("advice") == "advice"
        assert condition_for("no advice") == "no advice"
        assert condition_for("advice with annotation") == "advice with annotation"

    def test_returns_condition_for_type_prefixed_random_mode_sample_file(self):
        """scripts/atlas_preperation.py now names a random-mode sample file
        "{type}_{condition}" (e.g. "validation_advice.txt") so it's
        traceable to its split from the filename alone -- condition_for()
        must recover the condition from that prefixed name too."""
        assert condition_for("validation_advice") == "advice"
        assert condition_for("training_no advice") == "no advice"
        assert condition_for("validation_advice with annotation") == "advice with annotation"


class TestConditionMatch:
    """This study's coding-scheme condition vocabulary (general/supported/human)
    differs from output.xlsx's condition vocabulary (advice/advice with
    annotation/no advice); condition_match() translates between them."""

    def test_advice_matches_general_and_supported(self):
        assert condition_match("12_advice_cycle_7") == ["general", "supported"]

    def test_advice_with_annotation_matches_general_and_supported(self):
        assert condition_match("8_advice with annotation_cycle_99") == [
            "general",
            "supported",
        ]

    def test_no_advice_matches_general_and_human(self):
        assert condition_match("45_no advice_cycle_63") == ["general", "human"]

    def test_unrecognised_name_matches_nothing(self):
        assert condition_match("not-a-valid-name") == []

    def test_matches_random_mode_sample_files_too(self):
        """condition_match() feeds directly into AI coding's tag filtering
        (src.coding.ai_coding.process_documents() via condition_match(filepath.stem));
        a random-mode sample file named just "advice.txt" must still resolve
        to the right tag vocabulary, not an empty list."""
        assert condition_match("advice") == ["general", "supported"]
        assert condition_match("no advice") == ["general", "human"]

    def test_matches_type_prefixed_random_mode_sample_files_too(self):
        assert condition_match("validation_advice") == ["general", "supported"]
        assert condition_match("training_no advice") == ["general", "human"]


class TestParseDocumentName:
    """parse_document_name() is the shared parser get_document_key()/
    condition_for() are both built on top of; ids come back as ints, the
    condition as its raw string."""

    def test_parses_all_columns(self):
        assert parse_document_name("45_no advice_cycle_63") == {
            "think_aloud_id": 45,
            "condition": "no advice",
            "cycle_id": 63,
        }

    def test_returns_none_for_unrecognised_shape(self):
        assert parse_document_name("not-a-valid-name") is None

    def test_returns_none_for_non_numeric_ids(self):
        assert parse_document_name("abc_no advice_cycle_63") is None

    def test_recognizes_random_mode_sample_file_named_after_condition(self):
        """scripts/atlas_preperation.py's "random" mode names a file after
        its document-cluster value alone (e.g. "advice", "no advice",
        "advice with annotation" when VALIDATION_DOCUMENT_CLUSTER_COLUMN ==
        ["condition"]) -- such a file has no single think_aloud_id/cycle_id
        (it's a sample spanning many participants/cycles), so only
        "condition" is returned."""
        assert parse_document_name("advice") == {"condition": "advice"}
        assert parse_document_name("no advice") == {"condition": "no advice"}
        assert parse_document_name("advice with annotation") == {
            "condition": "advice with annotation"
        }

    def test_unrecognized_bare_name_still_returns_none(self):
        assert parse_document_name("garbage-name") is None

    def test_recognizes_type_prefixed_random_mode_sample_file(self):
        """scripts/atlas_preperation.py names a random-mode sample file
        "{type}_{condition}" (e.g. "validation_advice") so it's traceable
        to its split from the filename alone -- still no single
        think_aloud_id/cycle_id, so only "condition" comes back."""
        assert parse_document_name("validation_advice") == {"condition": "advice"}
        assert parse_document_name("training_no advice") == {"condition": "no advice"}
        assert parse_document_name("validation_advice with annotation") == {
            "condition": "advice with annotation"
        }

    def test_type_prefix_without_recognized_condition_returns_none(self):
        assert parse_document_name("validation_not-a-condition") is None
