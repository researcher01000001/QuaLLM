"""Unit tests for run_post_processing.py's manual-review JSON helpers"""

import json
import re
from unittest.mock import Mock

import pytest

import run_post_processing as pipeline
from src.core import process_log
from src.core.models import DisplayColumn, Document, Row, Tag
from src.post_processing.review_database import (
    append_to_database,
    read_post_process_rows,
)
from src.post_processing.review_page import extract_review_data
from run_post_processing import (
    _existing_excluded_paragraph_keys,
    _load_step3_review_data,
    confirm_copy_excluded_paragraphs,
    write_manual_review_json,
)


@pytest.fixture
def step1_files(tmp_path, monkeypatch):
    """Run real Step 1 HTML/database processing with synthetic source data."""
    source_name = "45_no advice_cycle_63"
    document = Document(
        source_name=source_name,
        rows=[Row(index=0, text="test paragraph", start_position=0, end_position=14)],
        display_column=DisplayColumn(
            source_column="position", label="Paragraph", values={0: "1"}
        ),
    )
    tags = [Tag("Known"), Tag("Unknown", additional_tag=True)]
    ai_dir = tmp_path / "ai"
    ai_dir.mkdir()
    (ai_dir / "consensus.json").write_text(
        json.dumps(
            [
                {
                    "phrase": "test",
                    "tags": ["Unknown"],
                    "start": 0,
                    "end": 4,
                    "source": {"source_name": source_name, "row": 0},
                },
            ]
        )
    )
    review_file = tmp_path / "review.html"
    database_file = tmp_path / "database.xlsx"
    settings = {
        "PREFIX": "synthetic_test",
        "AI_CODING_FILES": ai_dir,
        "DATA_RESULTS_DIR": tmp_path,
        "POST_PROCESS_STEP1_REVIEW_HTML": review_file,
        "POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE": tmp_path
        / "consensus_unmodified.json",
        "POST_PROCESS_DATABASE_FILE": database_file,
        "POST_PROCESS_TAGS": [],
        "POST_PROCESS_INCLUDE_UNDEFINED_TAGS": True,
        "POST_PROCESS_LOAD_DATABASE": True,
        "POST_PROCESS_DATABASE_USE_ALL_PREFIXES": False,
        "POST_PROCESS_STEP1_REBUILD_DATABASE": False,
        "POST_PROCESS_SPAN_OVERLAP_THRESHOLD": 0.9,
    }
    for name, value in settings.items():
        monkeypatch.setattr(pipeline, name, value)
    monkeypatch.setattr(pipeline, "load_code_tags", lambda: tags)
    monkeypatch.setattr(pipeline, "paragraph_meta", lambda: {})
    monkeypatch.setattr(
        pipeline.run_scope,
        "find_source_documents",
        lambda _: [tmp_path / f"{source_name}.txt"],
    )
    monkeypatch.setattr(
        pipeline.run_scope,
        "load_source_documents",
        lambda *a, **k: {source_name: document},
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    return review_file, database_file


def _save_step1_decision(review_file, decision):
    data = extract_review_data(review_file)
    data["metadata"]["test_marker"] = "saved page"
    record = data["records"][0]
    if decision == "retagged":
        record.update(tag="Known", status="changed_tag")
    elif decision == "deleted":
        record["status"] = "deleted"
    html = re.sub(
        r'(<script type="application/json" id="review-data">).*?(</script>)',
        lambda match: match[1] + json.dumps(data) + match[2],
        review_file.read_text(),
        flags=re.DOTALL,
    )
    review_file.write_text(html)


@pytest.mark.parametrize("integrate", [False, True])
def test_step1_first_run_generates_page_with_either_integration_value(
    step1_files, monkeypatch, integrate
):
    review_file, database_file = step1_files
    monkeypatch.setattr(
        pipeline, "POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE", integrate
    )

    pipeline._run_step1_body()

    records = extract_review_data(review_file)["records"]
    assert len(records) == 1
    assert records[0]["raw_tag"] == "Unknown"
    assert records[0]["status"] == "original"
    assert not database_file.exists()


@pytest.mark.parametrize("decision", ["unchanged", "retagged", "deleted"])
@pytest.mark.parametrize("integrate", [False, True])
def test_step1_existing_page_saves_and_stops_or_regenerates(
    step1_files, monkeypatch, decision, integrate
):
    review_file, database_file = step1_files
    monkeypatch.setattr(
        pipeline, "POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE", integrate
    )
    pipeline._run_step1_body()
    _save_step1_decision(review_file, decision)
    original_page = review_file.read_bytes()
    export = Mock(wraps=pipeline.export_step1_review_html)
    monkeypatch.setattr(pipeline, "export_step1_review_html", export)

    pipeline._run_step1_body()

    if integrate:
        export.assert_not_called()
        assert review_file.read_bytes() == original_page
        assert database_file.exists() == (decision != "unchanged")
        if database_file.exists():
            rows = read_post_process_rows(database_file)
            assert len(rows) == 1
            assert rows[0]["ai_tag"] == "Unknown"
            assert rows[0]["new_tag"] == (
                "Known" if decision == "retagged" else "<deleted>"
            )
    else:
        export.assert_called_once()
        data = extract_review_data(review_file)
        assert "test_marker" not in data["metadata"]
        assert data["records"][0]["tag"] == "Unknown"
        assert data["records"][0]["status"] == "original"
        assert not database_file.exists()


@pytest.mark.parametrize("integrate", [False, True])
def test_step1_loads_database_suggestions_independently_of_integration(
    step1_files, monkeypatch, integrate
):
    review_file, database_file = step1_files
    append_to_database(
        [
            {
                "think_aloud_id": 45,
                "condition": "no advice",
                "Paragraph": "1",
                "span": "test",
                "start": 0,
                "end": 4,
                "ai_tag": "Unknown",
                "new_tag": "<deleted>",
            },
        ],
        database_file,
        {},
        0.9,
    )
    original_database = database_file.read_bytes()
    monkeypatch.setattr(
        pipeline, "POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE", integrate
    )

    pipeline._run_step1_body()

    records = extract_review_data(review_file)["records"]
    assert len(records) == 1
    assert records[0]["raw_tag"] == "Unknown"
    assert records[0]["status"] == "deleted"
    assert database_file.read_bytes() == original_database


def _write_review_html(path, paragraph_checks) -> None:
    payload = json.dumps(
        {"metadata": {}, "paragraph_checks": paragraph_checks, "records": []}
    )
    path.write_text(
        "<!doctype html><html><body>\n"
        f'<script type="application/json" id="review-data">\n{payload}\n</script>\n'
        "</body></html>",
        encoding="utf-8",
    )


def test_write_manual_review_json_records_excluded_paragraphs(tmp_path):
    review_data = {
        "paragraph_checks": [
            {"source_name": "d", "row": 0, "checked": True, "excluded": False},
            {"source_name": "d", "row": 1, "checked": False, "excluded": True},
        ],
        "records": [
            {
                "id": "1",
                "group_id": "g1",
                "source_name": "d",
                "row": 0,
                "phrase": "span",
                "tag": "TagA",
                "start": 10,
                "end": 20,
                "status": "original",
            }
        ],
    }
    output_file = tmp_path / "manual_review_test.json"

    write_manual_review_json(review_data, output_file)

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["excluded_paragraphs"] == [["d", 1]]
    assert len(payload["consensus_rows"]) == 1
    assert payload["consensus_rows"][0]["tags"] == ["TagA"]


def test_load_step3_review_data_reads_new_dict_format(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "consensus_rows": [{"phrase": "x", "tags": ["TagA"]}],
                "excluded_paragraphs": [["d", 1], ["e", 0]],
            }
        ),
        encoding="utf-8",
    )

    consensus_rows, excluded_keys = _load_step3_review_data(path)

    assert consensus_rows == [{"phrase": "x", "tags": ["TagA"]}]
    assert excluded_keys == frozenset({("d", 1), ("e", 0)})


def test_load_step3_review_data_accepts_pre_exclude_marker_bare_list(tmp_path):
    """A POST_PROCESS_STEP3_REVIEW_JSON file written before the exclude
    marker existed is a bare list of consensus rows -- must still load,
    with zero exclusions."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps([{"phrase": "x", "tags": ["TagA"]}]),
        encoding="utf-8",
    )

    consensus_rows, excluded_keys = _load_step3_review_data(path)

    assert consensus_rows == [{"phrase": "x", "tags": ["TagA"]}]
    assert excluded_keys == frozenset()


def test_existing_excluded_paragraph_keys_missing_file_returns_empty(tmp_path):
    assert _existing_excluded_paragraph_keys(tmp_path / "missing.html") == frozenset()


def test_existing_excluded_paragraph_keys_reads_excluded_flags(tmp_path):
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path,
        [
            {"source_name": "d", "row": 0, "checked": True, "excluded": False},
            {"source_name": "d", "row": 1, "checked": False, "excluded": True},
        ],
    )

    assert _existing_excluded_paragraph_keys(path) == frozenset({("d", 1)})


def test_confirm_copy_excluded_paragraphs_no_file_skips_prompt(tmp_path, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError(
            "input() should not be called when there's nothing to copy"
        )

    monkeypatch.setattr("builtins.input", _fail_if_called)

    assert confirm_copy_excluded_paragraphs(tmp_path / "missing.html") == frozenset()


def test_confirm_copy_excluded_paragraphs_no_excluded_skips_prompt(
    tmp_path, monkeypatch
):
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path, [{"source_name": "d", "row": 0, "checked": True, "excluded": False}]
    )

    def _fail_if_called(*a, **k):
        raise AssertionError(
            "input() should not be called when there's nothing to copy"
        )

    monkeypatch.setattr("builtins.input", _fail_if_called)

    assert confirm_copy_excluded_paragraphs(path) == frozenset()


def test_confirm_copy_excluded_paragraphs_yes_answer_copies(tmp_path, monkeypatch):
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path, [{"source_name": "d", "row": 1, "checked": False, "excluded": True}]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    assert confirm_copy_excluded_paragraphs(path) == frozenset({("d", 1)})


def test_confirm_copy_excluded_paragraphs_no_answer_skips_copy(tmp_path, monkeypatch):
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path, [{"source_name": "d", "row": 1, "checked": False, "excluded": True}]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    assert confirm_copy_excluded_paragraphs(path) == frozenset()


def test_confirm_copy_excluded_paragraphs_default_answer_copies(tmp_path, monkeypatch):
    """An empty answer (just pressing Enter) defaults to copying -- losing a
    human's exclusion decisions silently is worse than copying one extra
    marker forward by default."""
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path, [{"source_name": "d", "row": 1, "checked": False, "excluded": True}]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    assert confirm_copy_excluded_paragraphs(path) == frozenset({("d", 1)})


def test_confirm_copy_excluded_paragraphs_logs_decision(tmp_path, monkeypatch):
    path = tmp_path / "manual_review_test.html"
    _write_review_html(
        path, [{"source_name": "d", "row": 1, "checked": False, "excluded": True}]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    log_file = tmp_path / "process_log.xlsx"
    log = process_log.ProcessLog(log_file, step="post_process_step2")
    with process_log.active(log):
        confirm_copy_excluded_paragraphs(path)
    log.flush()

    import pandas as pd

    df = pd.read_excel(log_file, sheet_name=process_log.AUDIT_TRAIL)
    assert len(df) == 1
    assert df.iloc[0]["decision_point"] == "step2_copy_excluded_paragraphs"
    assert df.iloc[0]["outcome"] == "copied"
