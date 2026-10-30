from pathlib import Path

import pytest

from src.core import process_log
from src.core.models import ConsensusQuote, DisplayColumn, Document, Row, Tag
from src.post_processing.review_database import DELETED_TAG_VALUE
from src.post_processing.review_page import explode_consensus_records
from src.post_processing.undefined_tags import (
    apply_database_prefill_to_records,
    build_corrected_consensus,
    build_database_rows_from_step1_records,
    build_step1_target_records,
    log_step1_review_report,
)

_DOC_NAME = "45_no advice_cycle_63"


def _document(rows: list[Row] | None = None) -> Document:
    return Document(
        source_name=_DOC_NAME,
        rows=rows
        if rows is not None
        else [
            Row(index=0, text="x" * 100, start_position=0, end_position=100),
            Row(index=1, text="y" * 100, start_position=101, end_position=201),
        ],
        display_column=DisplayColumn(
            source_column="position", label="Paragraph", values={0: "1", 1: "2"}
        ),
    )


@pytest.fixture(autouse=True)
def _document_identity_defaults(monkeypatch):
    """Every test that needs the database prefill to actually resolve a
    document overrides these; other tests get a safe "unresolvable" default
    so a forgotten override fails loudly instead of silently reading real
    output.xlsx-driven state."""
    monkeypatch.setattr("src.dataset.document_identity.get_document_key", lambda name: None)
    monkeypatch.setattr("src.dataset.document_identity.condition_for", lambda name: None)


def _resolve_doc_identity(monkeypatch, identity: dict, condition: str):
    monkeypatch.setattr(
        "src.dataset.document_identity.get_document_key", lambda name: identity if name == _DOC_NAME else None
    )
    monkeypatch.setattr(
        "src.dataset.document_identity.condition_for", lambda name: condition if name == _DOC_NAME else None
    )


class TestBuildStep1TargetRecords:
    def test_keeps_full_paragraph_context_for_target_paragraphs_only(self):
        document = _document()
        target_tag = Tag(name="other")
        other_tag = Tag(name="reasoning")
        quotes_by_doc = {
            _DOC_NAME: [
                ConsensusQuote(text="a", tags=[target_tag], start=0, end=1),
                ConsensusQuote(text="b", tags=[other_tag], start=1, end=2),
                ConsensusQuote(text="c", tags=[other_tag], start=101, end=102),
            ]
        }

        records, filtered_documents = build_step1_target_records(
            {_DOC_NAME: document}, quotes_by_doc, target_tag_names=["other"]
        )

        # Paragraph 0 (row 0) has a target-tag occurrence, so both its
        # records are kept (target and non-target, for context); paragraph
        # 1 (row 1) has none, so it's dropped entirely.
        assert {r["row"] for r in records} == {0}
        assert {r["tag"] for r in records} == {"other", "reasoning"}

        target_record = next(r for r in records if r["tag"] == "other")
        context_record = next(r for r in records if r["tag"] == "reasoning")
        assert target_record["is_target"] is True
        assert target_record["raw_tag"] == "other"
        assert context_record["is_target"] is False

        assert list(filtered_documents.keys()) == [_DOC_NAME]
        assert [row.index for row in filtered_documents[_DOC_NAME].rows] == [0]

    def test_no_target_tag_occurrence_drops_the_document_entirely(self):
        document = _document()
        quotes_by_doc = {
            _DOC_NAME: [ConsensusQuote(text="b", tags=[Tag(name="reasoning")], start=1, end=2)]
        }

        records, filtered_documents = build_step1_target_records(
            {_DOC_NAME: document}, quotes_by_doc, target_tag_names=["other"]
        )

        assert records == []
        assert filtered_documents == {}


class TestApplyDatabasePrefillToRecords:
    def _target_record(self, **overrides) -> dict:
        record = {
            "id": "rec-1",
            "source_name": _DOC_NAME,
            "row": 0,
            "phrase": "some phrase",
            "tag": "other",
            "raw_tag": "other",
            "start": 10,
            "end": 20,
            "status": "original",
            "is_target": True,
        }
        record.update(overrides)
        return record

    def test_matching_database_row_sets_the_record_tag(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._target_record()
        database_rows = [
            {
                "think_aloud_id": 45,
                "condition": "no advice",
                "start": 10,
                "end": 20,
                "new_tag": "RealTag",
            }
        ]

        touched = apply_database_prefill_to_records(
            [record], {_DOC_NAME: _document()}, database_rows, tags_by_name={}, overlap_threshold=0.9
        )

        assert touched == 1
        assert record["tag"] == "RealTag"
        assert record["status"] == "original"

    def test_deleted_marker_sets_status_without_touching_tag(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._target_record()
        database_rows = [
            {
                "think_aloud_id": 45,
                "condition": "no advice",
                "start": 10,
                "end": 20,
                "new_tag": DELETED_TAG_VALUE,
            }
        ]

        touched = apply_database_prefill_to_records(
            [record], {_DOC_NAME: _document()}, database_rows, tags_by_name={}, overlap_threshold=0.9
        )

        assert touched == 1
        assert record["status"] == "deleted"
        assert record["tag"] == "other"

    def test_double_codable_multi_tag_match_adds_sibling_records(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._target_record()
        tags_by_name = {
            "TagA": Tag(name="TagA", can_co_occur=True),
            "TagB": Tag(name="TagB", can_co_occur=True),
        }
        database_rows = [
            {"think_aloud_id": 45, "condition": "no advice", "start": 10, "end": 20, "new_tag": "TagB"},
            {"think_aloud_id": 45, "condition": "no advice", "start": 10, "end": 20, "new_tag": "TagA"},
        ]
        records = [record]

        touched = apply_database_prefill_to_records(
            records, {_DOC_NAME: _document()}, database_rows, tags_by_name, overlap_threshold=0.9
        )

        assert touched == 1
        assert len(records) == 2
        assert record["tag"] == "TagA"  # first of the sorted, comma-joined set
        sibling = records[1]
        assert sibling["tag"] == "TagB"
        assert sibling["is_target"] is True
        assert sibling["status"] == "original"
        assert sibling["id"] == "rec-1::prefill::TagB"

    def test_no_matching_row_leaves_record_untouched(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._target_record()
        database_rows = [
            {"think_aloud_id": 45, "condition": "no advice", "start": 500, "end": 510, "new_tag": "RealTag"}
        ]

        touched = apply_database_prefill_to_records(
            [record], {_DOC_NAME: _document()}, database_rows, tags_by_name={}, overlap_threshold=0.9
        )

        assert touched == 0
        assert record["tag"] == "other"
        assert record["status"] == "original"

    def test_unresolvable_document_identity_is_skipped_without_touching_record(self):
        # _document_identity_defaults leaves get_document_key -> None.
        record = self._target_record()
        database_rows = [
            {"think_aloud_id": 45, "condition": "no advice", "start": 10, "end": 20, "new_tag": "RealTag"}
        ]

        touched = apply_database_prefill_to_records(
            [record], {_DOC_NAME: _document()}, database_rows, tags_by_name={}, overlap_threshold=0.9
        )

        assert touched == 0
        assert record["tag"] == "other"

    def test_non_target_records_are_never_prefilled(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._target_record(is_target=False, tag="reasoning", raw_tag="reasoning")
        database_rows = [
            {"think_aloud_id": 45, "condition": "no advice", "start": 10, "end": 20, "new_tag": "RealTag"}
        ]

        touched = apply_database_prefill_to_records(
            [record], {_DOC_NAME: _document()}, database_rows, tags_by_name={}, overlap_threshold=0.9
        )

        assert touched == 0
        assert record["tag"] == "reasoning"


class TestBuildCorrectedConsensus:
    def test_touched_paragraph_is_replaced_wholesale_untouched_paragraph_passes_through(self):
        document = _document()
        original_tag = Tag(name="other")
        untouched_tag = Tag(name="reasoning")
        quotes_by_doc = {
            _DOC_NAME: [
                ConsensusQuote(text="a", tags=[original_tag], start=0, end=1),
                ConsensusQuote(text="b", tags=[untouched_tag], start=101, end=102),
            ]
        }
        # A reviewer-saved Step 1 record for paragraph 0 that retagged the
        # occurrence -- note it does NOT match quotes_by_doc's own tag,
        # proving the paragraph is replaced wholesale rather than merged.
        step1_records = [
            {
                "id": "45_no advice_cycle_63::0::1::other",
                "source_name": _DOC_NAME,
                "row": 0,
                "phrase": "a",
                "tag": "retagged_value",
                "start": 0,
                "end": 1,
                "status": "retagged",
            }
        ]

        corrected = build_corrected_consensus({_DOC_NAME: document}, quotes_by_doc, step1_records)

        by_start = {rec["start"]: rec for rec in corrected}
        assert by_start[0]["tags"] == ["retagged_value"]
        assert by_start[101]["tags"] == ["reasoning"]

    def test_empty_step1_records_falls_back_to_untouched_consensus(self):
        document = _document()
        tag = Tag(name="other")
        quotes_by_doc = {_DOC_NAME: [ConsensusQuote(text="a", tags=[tag], start=0, end=1)]}

        corrected = build_corrected_consensus({_DOC_NAME: document}, quotes_by_doc, [])

        expected = explode_consensus_records({_DOC_NAME: document}, quotes_by_doc)
        assert len(corrected) == 1
        assert corrected[0]["tags"] == [expected[0]["tag"]]
        assert corrected[0]["start"] == expected[0]["start"]


class TestBuildDatabaseRowsFromStep1Records:
    def _record(self, **overrides) -> dict:
        record = {
            "id": "rec-1",
            "source_name": _DOC_NAME,
            "row": 0,
            "phrase": "some phrase",
            "tag": "other",
            "raw_tag": "other",
            "start": 10,
            "end": 20,
            "status": "original",
            "is_target": True,
        }
        record.update(overrides)
        return record

    def test_retagged_target_record_is_written_back(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._record(tag="RealTag", status="retagged")

        rows = build_database_rows_from_step1_records([record], {_DOC_NAME: _document()})

        assert len(rows) == 1
        assert rows[0]["new_tag"] == "RealTag"
        assert rows[0]["think_aloud_id"] == 45

    def test_deleted_target_record_is_written_back_with_deleted_marker(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._record(status="deleted")

        rows = build_database_rows_from_step1_records([record], {_DOC_NAME: _document()})

        assert len(rows) == 1
        assert rows[0]["new_tag"] == DELETED_TAG_VALUE

    def test_unchanged_target_record_is_not_written_back(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._record()  # tag == raw_tag, status == "original"

        rows = build_database_rows_from_step1_records([record], {_DOC_NAME: _document()})

        assert rows == []

    def test_unchanged_sibling_of_a_resolved_record_is_written_back_as_kept(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        retagged = self._record(
            id="a", group_id="g", raw_tag="Hyp", tag="RealTag", status="changed_tag"
        )
        sibling = self._record(id="b", group_id="g", raw_tag="Merk", tag="Merk")

        rows = build_database_rows_from_step1_records([retagged, sibling], {_DOC_NAME: _document()})

        assert [(row["ai_tag"], row["new_tag"]) for row in rows] == [
            ("Hyp", "RealTag"),
            ("Merk", "Merk"),
        ]

    def test_unchanged_record_in_a_group_without_resolved_records_is_not_written_back(
        self, monkeypatch
    ):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        untouched = self._record(id="a", group_id="g1", raw_tag="Merk", tag="Merk")
        elsewhere = self._record(
            id="b", group_id="g2", raw_tag="Hyp", tag="RealTag", status="changed_tag"
        )

        rows = build_database_rows_from_step1_records(
            [untouched, elsewhere], {_DOC_NAME: _document()}
        )

        assert [(row["ai_tag"], row["new_tag"]) for row in rows] == [("Hyp", "RealTag")]

    def test_deleted_sibling_is_not_written_back_as_kept(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        retagged = self._record(
            id="a", group_id="g", raw_tag="Hyp", tag="RealTag", status="changed_tag"
        )
        deleted = self._record(id="b", group_id="g", raw_tag="Merk", tag="Merk", status="deleted")

        rows = build_database_rows_from_step1_records([retagged, deleted], {_DOC_NAME: _document()})

        assert [(row["ai_tag"], row["new_tag"]) for row in rows] == [
            ("Hyp", "RealTag"),
            ("Merk", DELETED_TAG_VALUE),
        ]

    def test_non_target_record_is_never_written_back(self, monkeypatch):
        _resolve_doc_identity(monkeypatch, {"think_aloud_id": 45}, "no advice")
        record = self._record(is_target=False, tag="RealTag", status="retagged")

        rows = build_database_rows_from_step1_records([record], {_DOC_NAME: _document()})

        assert rows == []


class TestLogStep1ReviewReport:
    def _record(self, **overrides) -> dict:
        record = {
            "id": "rec-1",
            "source_name": _DOC_NAME,
            "row": 0,
            "tag": "other",
            "raw_tag": "other",
            "start": 0,
            "end": 1,
            "status": "original",
            "is_target": True,
        }
        record.update(overrides)
        return record

    def test_logs_every_target_record_plus_only_edited_non_target_records(self, tmp_path: Path):
        records = [
            self._record(status="retagged"),  # target, edited -> logged
            self._record(id="rec-2", status="original"),  # target, kept -> still logged (audit trail)
            self._record(
                id="rec-3", is_target=False, tag="reasoning", raw_tag="reasoning", status="original"
            ),  # non-target, untouched -> not logged
            self._record(
                id="rec-4", is_target=False, tag="reasoning", raw_tag="reasoning", status="moved"
            ),  # non-target, edited despite restrict toggle -> logged
            self._record(id="rec-5", status="deleted"),  # target, deleted -> Warning severity
        ]

        log = process_log.ProcessLog(
            tmp_path / "process_log.xlsx",
            step=process_log.STEP_POST_PROCESS_STEP2,
            owned_steps={process_log.STEP_POST_PROCESS_STEP2, process_log.STEP_MANUAL_REWORK_REVIEW},
        )
        with process_log.active(log):
            report = log_step1_review_report(records, {_DOC_NAME: _document()})
        log.flush()

        assert len(report) == 4
        assert set(report["status"]) == {"retagged", "original", "moved", "deleted"}
        deleted_row = report[report["status"] == "deleted"].iloc[0]
        assert deleted_row["severity"] == "Warning"
        info_rows = report[report["status"] != "deleted"]
        assert (info_rows["severity"] == "Info").all()
