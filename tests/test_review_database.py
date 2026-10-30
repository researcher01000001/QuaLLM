import os
from pathlib import Path

import pandas as pd
import pytest

from src.config import project as config_project
from src.core.models import AIQuote, Tag
from src.post_processing.review_database import (
    DELETED_TAG_VALUE,
    append_to_database,
    build_post_process_columns,
    build_target_tags,
    collect_undefined_tag_names,
    is_kept_row,
    load_database_rows,
    manually_defined_rows,
    prefill_new_tags_from_database,
    read_post_process_rows,
    rebuild_database,
    resolve_database_files,
    spans_overlap_at_least,
    write_data_table,
)

KEY_COLUMNS = ["id", "cycle_id"]


@pytest.fixture(autouse=True)
def _config_project_defaults(monkeypatch):
    """DOCUMENT_KEY_COLUMNS matches this test file's row fixtures.
    get_document_key/condition_for default to "not found" so a test that
    doesn't explicitly patch them can't accidentally read the real
    output.xlsx; tests that need specific identity data override these with
    their own monkeypatch call."""
    monkeypatch.setattr("src.config.project.DOCUMENT_KEY_COLUMNS", KEY_COLUMNS)
    monkeypatch.setattr("src.dataset.document_identity.get_document_key", lambda name: None)
    monkeypatch.setattr("src.dataset.document_identity.condition_for", lambda name: None)


def _row(**overrides) -> dict:
    row = {
        "id": 45,
        "condition": "no advice",
        "cycle_id": 63,
        config_project.DOCUMENT_ORDER_COLUMN_LABEL: "1",
        "span": "irgendetwas anderes",
        "start": 12,
        "end": 31,
        "ai_tag": "Other",
        "new_tag": "",
    }
    row.update(overrides)
    return row


def _tags_by_name(*tags: Tag) -> dict[str, Tag]:
    return {tag.name: tag for tag in tags}


def test_spans_overlap_at_least_true_when_ratio_meets_threshold():
    # intersection=16, union=20 -> ratio=0.8
    assert spans_overlap_at_least(100, 120, 102, 118, 0.5) is True


def test_spans_overlap_at_least_false_when_ratio_below_threshold():
    # intersection=10, union=20 -> ratio=0.5
    assert spans_overlap_at_least(100, 120, 110, 120, 0.9) is False


def test_spans_overlap_at_least_false_when_offset_missing():
    assert spans_overlap_at_least(None, 120, 100, 120, 0.1) is False


def test_collect_undefined_tag_names_returns_distinct_additional_tags():
    schema_tag = Tag(name="indicator: no_bleeding")
    undefined_1 = Tag(name="Other", additional_tag=True)
    undefined_2 = Tag(name="Made Up", additional_tag=True)
    ai_quote_runs = {
        ("config", 0): {
            "doc-1": [
                AIQuote(text="a", tags=[schema_tag], row=0),
                AIQuote(text="b", tags=[undefined_1], row=0),
            ]
        },
        ("config", 1): {
            "doc-1": [
                AIQuote(text="c", tags=[undefined_1, undefined_2], row=0),
            ]
        },
    }

    names = collect_undefined_tag_names(ai_quote_runs)

    assert names == ["Other", "Made Up"]


def test_build_target_tags_merges_configured_and_undefined_tags():
    undefined = Tag(name="Made Up", additional_tag=True)
    already_listed = Tag(name="Other", additional_tag=True)
    ai_quote_runs = {
        ("config", 0): {
            "doc-1": [
                AIQuote(text="a", tags=[undefined, already_listed], row=0),
            ]
        },
    }

    target_tags = build_target_tags(["Other"], ai_quote_runs, include_undefined_tags=True)

    assert target_tags == ["Other", "Made Up"]


def test_build_target_tags_ignores_undefined_tags_when_disabled():
    undefined = Tag(name="Made Up", additional_tag=True)
    ai_quote_runs = {
        ("config", 0): {"doc-1": [AIQuote(text="a", tags=[undefined], row=0)]},
    }

    target_tags = build_target_tags(["Other"], ai_quote_runs, include_undefined_tags=False)

    assert target_tags == ["Other"]


def test_build_target_tags_accepts_a_plain_wrapped_quotes_by_doc_mapping():
    """collect_undefined_tag_names()/build_target_tags() only ever read
    ai_quote_runs.values(), so a caller with a plain {doc_name: [quotes]}
    consensus mapping (not per-configuration/iteration AI runs) can wrap it
    as {"_": quotes_by_doc} instead of reshaping it."""
    undefined = Tag(name="Made Up", additional_tag=True)
    quotes_by_doc = {"doc-1": [AIQuote(text="a", tags=[undefined], row=0)]}

    target_tags = build_target_tags(["Other"], {"_": quotes_by_doc}, include_undefined_tags=True)

    assert target_tags == ["Other", "Made Up"]


def test_build_post_process_columns_does_not_duplicate_condition(monkeypatch):
    """When DOCUMENT_KEY_COLUMNS already includes "condition" (as it does
    for the real dataset), it must not be appended a second time -- a
    duplicate "condition" column silently corrupts the exported workbook
    (pandas' read_excel renames the second one to "condition.1", so a
    reader relying on the "condition" column only ever sees the first,
    identical copy -- confirmed by reproducing the exact write/read
    round-trip)."""
    monkeypatch.setattr("src.config.project.DOCUMENT_KEY_COLUMNS", ["think_aloud_id", "condition"])

    columns = build_post_process_columns()

    assert columns.count("condition") == 1
    assert columns[:2] == ["think_aloud_id", "condition"]


def test_read_post_process_rows_missing_file_returns_empty(tmp_path: Path):
    assert read_post_process_rows(tmp_path / "missing.xlsx") == []


def test_write_data_table_drops_paragraph_text_and_internal_keys(tmp_path: Path):
    """The database workbook keeps its original schema -- paragraph_text and
    the internal bold-span offset keys are extraction-file only."""
    output_file = tmp_path / "database.xlsx"
    row = _row(
        paragraph_text="Das koennte irgendetwas anderes sein.",
        _span_paragraph_start=12,
        _span_paragraph_end=31,
    )

    write_data_table([row], output_file)

    df = pd.read_excel(output_file)
    assert df.columns.tolist() == build_post_process_columns()


def test_manually_defined_rows_filters_empty_new_tag():
    rows = [_row(new_tag=""), _row(new_tag="  "), _row(new_tag="Fixed")]

    result = manually_defined_rows(rows)

    assert result == [_row(new_tag="Fixed")]


def test_append_to_database_adds_new_and_skips_duplicates(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Fixed"))

    added_first = append_to_database([_row(new_tag="Fixed")], database_file, tags_by_name, 0.5)
    added_second = append_to_database([_row(new_tag="Fixed")], database_file, tags_by_name, 0.5)

    assert added_first == 1
    assert added_second == 0
    assert len(read_post_process_rows(database_file)) == 1


def test_append_to_database_skips_duplicate_within_overlap_threshold(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Fixed"))

    # (100,120) vs (102,118): intersection=16, union=20 -> ratio=0.8
    append_to_database(
        [_row(start=100, end=120, new_tag="Fixed")], database_file, tags_by_name, 0.5
    )
    added_second = append_to_database(
        [_row(start=102, end=118, new_tag="Fixed")], database_file, tags_by_name, 0.5
    )

    assert added_second == 0
    assert len(read_post_process_rows(database_file)) == 1


def test_append_to_database_ignores_rows_without_new_tag(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"

    added = append_to_database([_row(new_tag="")], database_file, {}, 0.5)

    assert added == 0
    assert not database_file.exists()


def test_append_to_database_replaces_conflicting_non_double_codable_entry(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(
        Tag(name="TagA", can_co_occur=False), Tag(name="TagB", can_co_occur=False)
    )

    append_to_database(
        [_row(start=100, end=120, new_tag="TagA")], database_file, tags_by_name, 0.5
    )
    added = append_to_database(
        [_row(start=101, end=121, new_tag="TagB")], database_file, tags_by_name, 0.5
    )

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert len(rows) == 1
    assert rows[0]["new_tag"] == "TagB"


def test_append_to_database_replaces_when_new_entry_is_deleted(tmp_path: Path):
    """A DELETED_TAG_VALUE candidate always wins outright over an existing
    real-tag match -- "no tag applies here" isn't compatible with anything."""
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="TagA", can_co_occur=True))

    append_to_database(
        [_row(start=100, end=120, new_tag="TagA")], database_file, tags_by_name, 0.5
    )
    added = append_to_database(
        [_row(start=101, end=121, new_tag=DELETED_TAG_VALUE)], database_file, tags_by_name, 0.5
    )

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert len(rows) == 1
    assert rows[0]["new_tag"] == DELETED_TAG_VALUE


def test_append_to_database_keeps_both_when_double_coding_allowed(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(
        Tag(name="TagA", can_co_occur=True), Tag(name="TagB", can_co_occur=True)
    )

    append_to_database(
        [_row(start=100, end=120, new_tag="TagA")], database_file, tags_by_name, 0.5
    )
    added = append_to_database(
        [_row(start=101, end=121, new_tag="TagB")], database_file, tags_by_name, 0.5
    )

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert {row["new_tag"] for row in rows} == {"TagA", "TagB"}


def test_append_to_database_keeps_both_when_only_one_tag_allows_double_coding(tmp_path: Path):
    """A double_coding=yes tag may pair with anything, including a
    double_coding=no tag -- only two double_coding=no tags conflict."""
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(
        Tag(name="TagA", can_co_occur=True), Tag(name="TagB", can_co_occur=False)
    )

    append_to_database(
        [_row(start=100, end=120, new_tag="TagA")], database_file, tags_by_name, 0.5
    )
    added = append_to_database(
        [_row(start=101, end=121, new_tag="TagB")], database_file, tags_by_name, 0.5
    )

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert {row["new_tag"] for row in rows} == {"TagA", "TagB"}


def test_rebuild_database_deletes_and_recreates_from_scratch(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Stale"), Tag(name="Fresh"))

    append_to_database(
        [_row(start=200, end=220, new_tag="Stale")], database_file, tags_by_name, 0.5
    )
    added = rebuild_database(
        [_row(start=100, end=120, new_tag="Fresh")], database_file, tags_by_name, 0.5
    )

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert [row["new_tag"] for row in rows] == ["Fresh"]


def test_rebuild_database_works_when_no_database_exists(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"

    added = rebuild_database([_row(new_tag="Fresh")], database_file, {}, 0.5)

    rows = read_post_process_rows(database_file)
    assert added == 1
    assert rows[0]["new_tag"] == "Fresh"


def test_resolve_database_files_returns_current_file_when_it_exists(tmp_path: Path):
    current = tmp_path / "prefix-a" / "01_undefined_tags" / "database_prefix-a.xlsx"
    write_data_table([_row()], current)

    files = resolve_database_files(tmp_path, current, use_all_prefixes=False)

    assert files == [current]


def test_resolve_database_files_returns_empty_when_current_file_does_not_exist(
    tmp_path: Path,
):
    """A caller's "N database file(s)" log message should reflect files
    actually found, not paths merely considered."""
    current = tmp_path / "prefix-a" / "01_undefined_tags" / "database_prefix-a.xlsx"

    files = resolve_database_files(tmp_path, current, use_all_prefixes=False)

    assert files == []


def test_resolve_database_files_globs_across_prefixes(tmp_path: Path):
    """Database files live under {prefix}/{run_type}/3-post_process/ before
    the Step 1 subfolder -- see POST_PROCESS_DIR's layout."""
    current = (
        tmp_path / "prefix-a" / "full" / "3-post_process" / "01_undefined_tags"
        / "database_prefix-a.xlsx"
    )
    write_data_table([_row()], current)
    other = (
        tmp_path / "prefix-b" / "training" / "3-post_process" / "01_undefined_tags"
        / "database_prefix-b.xlsx"
    )
    write_data_table([_row()], other)
    # Other stage folders of a prefix are never scanned.
    unrelated = (
        tmp_path / "prefix-b" / "training" / "2-results" / "01_undefined_tags"
        / "database_prefix-b.xlsx"
    )
    write_data_table([_row()], unrelated)

    files = resolve_database_files(tmp_path, current, use_all_prefixes=True)

    assert files == sorted([current, other])


def test_load_database_rows_concatenates_files(tmp_path: Path):
    file_a = tmp_path / "a.xlsx"
    file_b = tmp_path / "b.xlsx"
    write_data_table([_row(new_tag="A")], file_a)
    write_data_table([_row(new_tag="B")], file_b)

    rows = load_database_rows([file_a, file_b])

    assert [row["new_tag"] for row in rows] == ["A", "B"]


def test_prefill_new_tags_from_database_matches_within_overlap_threshold():
    # (100,120) vs (102,118): ratio=0.8
    row = _row(start=100, end=120, new_tag="")
    database_rows = [_row(start=102, end=118, new_tag="Corrected")]
    tags_by_name = _tags_by_name(Tag(name="Corrected"))

    result = prefill_new_tags_from_database([row], database_rows, tags_by_name, 0.5)

    assert result[0]["new_tag"] == "Corrected"


def test_prefill_new_tags_from_database_ignores_matches_below_overlap_threshold():
    # (100,120) vs (110,120): ratio=0.5, below a 0.9 threshold
    row = _row(start=100, end=120, new_tag="")
    database_rows = [_row(start=110, end=120, new_tag="Corrected")]

    result = prefill_new_tags_from_database([row], database_rows, {}, 0.9)

    assert result[0]["new_tag"] == ""


def test_prefill_new_tags_from_database_requires_same_document():
    row = _row(id=45, condition="no advice", cycle_id=63, start=100, end=120, new_tag="")
    database_rows = [
        _row(id=46, condition="no advice", cycle_id=63, start=100, end=120, new_tag="Corrected")
    ]

    result = prefill_new_tags_from_database([row], database_rows, {}, 0.5)

    assert result[0]["new_tag"] == ""


def test_prefill_new_tags_from_database_ignores_db_rows_without_new_tag():
    row = _row(start=100, end=120, new_tag="")
    database_rows = [_row(start=100, end=120, new_tag="")]

    result = prefill_new_tags_from_database([row], database_rows, {}, 0.5)

    assert result[0]["new_tag"] == ""


def test_prefill_new_tags_from_database_resolves_to_deleted_marker():
    row = _row(start=100, end=120, new_tag="")
    database_rows = [_row(start=100, end=120, new_tag=DELETED_TAG_VALUE)]

    result = prefill_new_tags_from_database([row], database_rows, {}, 0.5)

    assert result[0]["new_tag"] == DELETED_TAG_VALUE


def test_prefill_new_tags_from_database_combines_unique_set_when_all_allow_double_coding():
    row = _row(start=100, end=120, new_tag="")
    database_rows = [
        _row(start=100, end=120, new_tag="TagB"),
        _row(start=101, end=121, new_tag="TagA"),
    ]
    tags_by_name = _tags_by_name(
        Tag(name="TagA", can_co_occur=True), Tag(name="TagB", can_co_occur=True)
    )

    result = prefill_new_tags_from_database([row], database_rows, tags_by_name, 0.5)

    assert result[0]["new_tag"] == "TagA, TagB"


def test_prefill_new_tags_from_database_combines_double_coding_no_and_yes_tag():
    """A double_coding=no tag may still pair with a double_coding=yes tag --
    only two double_coding=no tags conflict with each other."""
    row = _row(start=100, end=120, new_tag="")
    database_rows = [
        _row(start=100, end=120, new_tag="NoTag"),
        _row(start=101, end=121, new_tag="YesTag"),
    ]
    tags_by_name = _tags_by_name(
        Tag(name="NoTag", can_co_occur=False), Tag(name="YesTag", can_co_occur=True)
    )

    result = prefill_new_tags_from_database([row], database_rows, tags_by_name, 0.5)

    assert result[0]["new_tag"] == "NoTag, YesTag"


def test_prefill_new_tags_from_database_picks_newest_file_when_conflict(tmp_path: Path):
    row = _row(start=100, end=120, new_tag="")
    older_file = tmp_path / "prefix-a" / "database_prefix-a.xlsx"
    newer_file = tmp_path / "prefix-b" / "database_prefix-b.xlsx"
    write_data_table([_row(start=100, end=120, new_tag="OldTag")], older_file)
    write_data_table([_row(start=101, end=121, new_tag="NewTag")], newer_file)

    old_time = 1_700_000_000
    os.utime(older_file, (old_time, old_time))
    os.utime(newer_file, (old_time + 3600, old_time + 3600))

    database_rows = load_database_rows([older_file, newer_file])
    tags_by_name = _tags_by_name(
        Tag(name="OldTag", can_co_occur=False), Tag(name="NewTag", can_co_occur=False)
    )

    result = prefill_new_tags_from_database([row], database_rows, tags_by_name, 0.5)

    assert result[0]["new_tag"] == "NewTag"


def test_prefill_new_tags_from_database_keeps_double_codable_tag_alongside_newest_conflict(
    tmp_path: Path,
):
    """Two double_coding=no tags conflict and are resolved by save date, but a
    double_coding=yes tag matching the same span is kept regardless."""
    row = _row(start=100, end=120, new_tag="")
    older_file = tmp_path / "prefix-a" / "database_prefix-a.xlsx"
    newer_file = tmp_path / "prefix-b" / "database_prefix-b.xlsx"
    write_data_table(
        [
            _row(start=100, end=120, new_tag="OldNoTag"),
            _row(start=100, end=120, new_tag="YesTag"),
        ],
        older_file,
    )
    write_data_table([_row(start=101, end=121, new_tag="NewNoTag")], newer_file)

    old_time = 1_700_000_000
    os.utime(older_file, (old_time, old_time))
    os.utime(newer_file, (old_time + 3600, old_time + 3600))

    database_rows = load_database_rows([older_file, newer_file])
    tags_by_name = _tags_by_name(
        Tag(name="OldNoTag", can_co_occur=False),
        Tag(name="NewNoTag", can_co_occur=False),
        Tag(name="YesTag", can_co_occur=True),
    )

    result = prefill_new_tags_from_database([row], database_rows, tags_by_name, 0.5)

    assert result[0]["new_tag"] == "NewNoTag, YesTag"


# --- "kept" rows: a reviewed record whose AI tag was left unchanged -------------


def test_is_kept_row_true_only_when_new_tag_equals_ai_tag():
    assert is_kept_row(_row(ai_tag="A", new_tag="A"))
    assert not is_kept_row(_row(ai_tag="A", new_tag="B"))
    assert not is_kept_row(_row(ai_tag="A", new_tag=""))
    assert not is_kept_row(_row(ai_tag="", new_tag=""))


def test_prefill_keeps_sibling_record_of_a_double_coded_span_when_a_kept_row_exists():
    database_rows = [
        _row(ai_tag="Hyp", new_tag="Fixed"),
        _row(ai_tag="Merk", new_tag="Merk"),
    ]
    tags_by_name = _tags_by_name(Tag(name="Fixed"), Tag(name="Merk"), Tag(name="Hyp"))

    hyp, merk = prefill_new_tags_from_database(
        [_row(ai_tag="Hyp"), _row(ai_tag="Merk")], database_rows, tags_by_name, 0.5
    )

    assert hyp["new_tag"] == "Fixed"
    assert merk["new_tag"] == "Merk"


def test_prefill_uses_only_decisions_for_the_records_own_ai_tag():
    database_rows = [
        _row(ai_tag="Hyp", new_tag="X"),
        _row(ai_tag="Merk", new_tag="Y"),
    ]
    tags_by_name = _tags_by_name(Tag(name="X"), Tag(name="Y"))

    hyp, merk = prefill_new_tags_from_database(
        [_row(ai_tag="Hyp"), _row(ai_tag="Merk")], database_rows, tags_by_name, 0.5
    )

    assert hyp["new_tag"] == "X"
    assert merk["new_tag"] == "Y"


def test_prefill_falls_back_to_other_ai_tags_correction_without_a_decision_of_its_own():
    database_rows = [_row(ai_tag="Hyp", new_tag="Fixed")]
    tags_by_name = _tags_by_name(Tag(name="Fixed"))

    (result,) = prefill_new_tags_from_database(
        [_row(ai_tag="Other")], database_rows, tags_by_name, 0.5
    )

    assert result["new_tag"] == "Fixed"


def test_prefill_ignores_a_kept_row_of_a_different_ai_tag():
    database_rows = [_row(ai_tag="Merk", new_tag="Merk")]

    (result,) = prefill_new_tags_from_database(
        [_row(ai_tag="Other")], database_rows, _tags_by_name(Tag(name="Merk")), 0.5
    )

    assert result["new_tag"] == ""


def test_append_stores_kept_row_without_counting_it_as_a_correction(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Fixed"), Tag(name="Merk"))

    added = append_to_database(
        [_row(ai_tag="Hyp", new_tag="Fixed"), _row(ai_tag="Merk", new_tag="Merk")],
        database_file,
        tags_by_name,
        0.5,
    )

    assert added == 1
    stored = {(row["ai_tag"], row["new_tag"]) for row in read_post_process_rows(database_file)}
    assert stored == {("Hyp", "Fixed"), ("Merk", "Merk")}


def test_append_kept_row_never_removes_a_correction_of_the_same_span(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    # Neither tag allows double coding, so a correction and a kept row would
    # clash if kept rows took part in the conflict logic.
    tags_by_name = _tags_by_name(Tag(name="Fixed"), Tag(name="Merk"))
    append_to_database([_row(ai_tag="Hyp", new_tag="Fixed")], database_file, tags_by_name, 0.5)

    append_to_database([_row(ai_tag="Merk", new_tag="Merk")], database_file, tags_by_name, 0.5)

    stored = {(row["ai_tag"], row["new_tag"]) for row in read_post_process_rows(database_file)}
    assert stored == {("Hyp", "Fixed"), ("Merk", "Merk")}


def test_append_kept_row_is_skipped_when_the_same_ai_tag_already_has_a_decision(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Fixed"))
    append_to_database([_row(ai_tag="Merk", new_tag="Fixed")], database_file, tags_by_name, 0.5)

    append_to_database([_row(ai_tag="Merk", new_tag="Merk")], database_file, tags_by_name, 0.5)

    stored = [(row["ai_tag"], row["new_tag"]) for row in read_post_process_rows(database_file)]
    assert stored == [("Merk", "Fixed")]


def test_append_correction_replaces_the_kept_row_of_the_same_ai_tag(tmp_path: Path):
    database_file = tmp_path / "database.xlsx"
    tags_by_name = _tags_by_name(Tag(name="Fixed"), Tag(name="Merk"))
    append_to_database([_row(ai_tag="Merk", new_tag="Merk")], database_file, tags_by_name, 0.5)
    # Only kept rows in the batch: nothing to count, but the file is still written.
    assert len(read_post_process_rows(database_file)) == 1

    added = append_to_database(
        [_row(ai_tag="Merk", new_tag="Fixed")], database_file, tags_by_name, 0.5
    )

    assert added == 1
    stored = [(row["ai_tag"], row["new_tag"]) for row in read_post_process_rows(database_file)]
    assert stored == [("Merk", "Fixed")]
