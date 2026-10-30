import threading

import pandas as pd
import pytest

from src.config import project as config_project
from src.core import process_log
from src.core.process_log import ProcessLog


def _read_sheet(path, sheet_name):
    return pd.read_excel(path, sheet_name=sheet_name)


def test_flush_empty_run_creates_all_sheets(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main")
    log.mark_step_start(prefix="test")
    log.mark_step_end(outcome="success")
    log.flush()

    workbook = pd.ExcelFile(output_file)
    assert process_log.RUN_INFO in workbook.sheet_names
    for sheet_name in process_log.SHEET_SCHEMAS:
        assert sheet_name in workbook.sheet_names
        df = _read_sheet(output_file, sheet_name)
        assert list(df.columns) == process_log.SHEET_SCHEMAS[sheet_name]
        assert len(df) == 0

    run_info = _read_sheet(output_file, process_log.RUN_INFO)
    assert len(run_info) == 1
    assert run_info.iloc[0]["step"] == "run_main"
    assert run_info.iloc[0]["outcome"] == "success"


def test_log_paragraph_excluded_populates_columns(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="post_process_step3")
    log.log_paragraph_excluded(
        severity="Info",
        source_name="45_no advice_cycle_63",
        row=0,
        display_value="1",
        message="excluded",
    )
    log.flush()

    df = _read_sheet(output_file, process_log.PARAGRAPH_EXCLUSIONS)
    assert len(df) == 1
    assert df.iloc[0]["source_name"] == "45_no advice_cycle_63"
    assert df.iloc[0]["row"] == 0
    # The order-column value round-trips through Excel as a number when it
    # looks numeric (pandas infers dtype on read), not the original string.
    assert str(df.iloc[0][config_project.DOCUMENT_ORDER_COLUMN_LABEL]) == "1"
    assert df.iloc[0]["message"] == "excluded"
    assert df.iloc[0]["severity"] == "Info"


def test_rerun_of_same_step_replaces_rows_not_accumulates(tmp_path):
    output_file = tmp_path / "process_log.xlsx"

    log1 = ProcessLog(output_file, step="post_process_step1")
    log1.log_missing_source_document(
        severity="Warning", source_name="doc-a", context="ctx", message="first"
    )
    log1.log_missing_source_document(
        severity="Warning", source_name="doc-b", context="ctx", message="second"
    )
    log1.flush()

    df = _read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)
    assert len(df) == 2

    log2 = ProcessLog(output_file, step="post_process_step1")
    log2.log_missing_source_document(
        severity="Warning", source_name="doc-c", context="ctx", message="third"
    )
    log2.flush()

    df = _read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)
    assert len(df) == 1
    assert df.iloc[0]["source_name"] == "doc-c"


def test_rerun_with_no_fresh_rows_still_wipes_stale_rows(tmp_path):
    output_file = tmp_path / "process_log.xlsx"

    log1 = ProcessLog(output_file, step="post_process_step1")
    log1.log_missing_source_document(
        severity="Warning", source_name="doc-a", context="ctx", message="stale"
    )
    log1.flush()
    assert len(_read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)) == 1

    log2 = ProcessLog(output_file, step="post_process_step1")
    log2.flush()

    df = _read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)
    assert len(df) == 0


def test_different_steps_writing_same_sheet_dont_clobber_each_other(tmp_path):
    output_file = tmp_path / "process_log.xlsx"

    log_a = ProcessLog(output_file, step="run_main")
    log_a.log_missing_source_document(
        severity="Warning", source_name="from-run-main", context="ctx", message="a"
    )
    log_a.flush()

    log_b = ProcessLog(output_file, step="post_process_step1")
    log_b.log_missing_source_document(
        severity="Warning", source_name="from-step1", context="ctx", message="b"
    )
    log_b.flush()

    df = _read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)
    assert set(df["source_name"]) == {"from-run-main", "from-step1"}
    assert set(df["step"]) == {"run_main", "post_process_step1"}

    # Rerunning run_main again must not touch post_process_step1's row.
    log_a2 = ProcessLog(output_file, step="run_main")
    log_a2.log_missing_source_document(
        severity="Warning", source_name="from-run-main-2", context="ctx", message="a2"
    )
    log_a2.flush()

    df = _read_sheet(output_file, process_log.MISSING_SOURCE_DOCUMENT)
    assert set(df["source_name"]) == {"from-run-main-2", "from-step1"}


def test_owned_steps_manual_rework_review_replaced_by_latest_owner(tmp_path):
    output_file = tmp_path / "process_log.xlsx"

    step2_log = ProcessLog(
        output_file,
        step="post_process_step2",
        owned_steps={"post_process_step2", "manual_rework_review"},
    )
    step2_log.log_rework_still_undefined(
        severity="Warning",
        source_name="doc-a",
        row=0,
        tag_name="Mystery Tag",
        span_start=0,
        span_end=5,
        message="from step 2",
        step="manual_rework_review",
    )
    step2_log.flush()

    df = _read_sheet(output_file, process_log.REWORK_STILL_UNDEFINED)
    assert len(df) == 1
    assert df.iloc[0]["tag_name"] == "Mystery Tag"

    step3_log = ProcessLog(
        output_file,
        step="post_process_step3",
        owned_steps={"post_process_step3", "manual_rework_review"},
    )
    step3_log.log_rework_still_undefined(
        severity="Warning",
        source_name="doc-b",
        row=1,
        tag_name="Other Mystery Tag",
        span_start=10,
        span_end=15,
        message="from step 3",
        step="manual_rework_review",
    )
    step3_log.flush()

    df = _read_sheet(output_file, process_log.REWORK_STILL_UNDEFINED)
    assert len(df) == 1
    assert df.iloc[0]["tag_name"] == "Other Mystery Tag"
    assert df.iloc[0]["step"] == "manual_rework_review"

    # Other sheets/steps must be unaffected by either flush.
    run_info = _read_sheet(output_file, process_log.RUN_INFO)
    assert set(run_info["step"]) == {"post_process_step2", "post_process_step3"}


def test_step_not_in_owned_steps_raises(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="post_process_step2", owned_steps={"post_process_step2"})
    with pytest.raises(ValueError):
        log.log_rework_still_undefined(
            severity="Warning",
            source_name="doc-a",
            row=0,
            tag_name="Tag",
            span_start=0,
            span_end=1,
            message="msg",
            step="manual_rework_review",
        )


def test_explicit_step_kwarg_namespaces_a_row_independent_of_others(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main", owned_steps={"run_main", "run_main/cfg-a"})
    log.log_general_warning(severity="Warning", message="scoped", step="run_main/cfg-a")
    log.log_general_warning(severity="Warning", message="unscoped")
    log.flush()

    df = _read_sheet(output_file, process_log.GENERAL_WARNINGS)
    steps_by_message = dict(zip(df["message"], df["step"]))
    assert steps_by_message["scoped"] == "run_main/cfg-a"
    assert steps_by_message["unscoped"] == "run_main"


def test_every_typed_log_method_accepts_an_explicit_step_kwarg(tmp_path):
    """Every typed log_*() method (not just log_rework_still_undefined,
    which had it first) accepts a plain step= override -- this is how a
    caller several functions deep from the active ProcessLog (e.g.
    AIQuote.find_position()) tags a row with a finer-grained sub-step: it's
    just an ordinary parameter passed down from whichever caller actually
    knows the configuration name, no hidden state involved."""
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(
        output_file, step="run_main", owned_steps={"run_main", "run_main/cfg-a"}
    )
    log.log_quote_position_issue(
        severity="Error",
        source_name="doc-a",
        row=0,
        issue_type="not_found",
        quote_text="text",
        message="msg",
        step="run_main/cfg-a",
    )
    log.log_undefined_tag(
        severity="Info",
        source_name="doc-a",
        tag_name="Tag",
        reason="unknown_name",
        resolution="kept_as_additional_tag",
        message="msg",
        step="run_main/cfg-a",
    )
    log.flush()

    assert _read_sheet(output_file, process_log.QUOTE_POSITION_ISSUES).iloc[0]["step"] == (
        "run_main/cfg-a"
    )
    assert _read_sheet(output_file, process_log.UNDEFINED_TAGS).iloc[0]["step"] == "run_main/cfg-a"


def test_explicit_step_kwarg_raises_when_not_owned(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="post_process_step1", owned_steps={"post_process_step1"})
    with pytest.raises(ValueError):
        log.log_general_warning(
            severity="Warning", message="boom", step="post_process_step1/cfg-a"
        )


def test_run_info_counts_include_sub_step_rows(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main", owned_steps={"run_main", "run_main/cfg-a"})
    log.mark_step_start(prefix="test")
    log.log_general_warning(severity="Error", message="e1", step="run_main/cfg-a")
    log.log_general_warning(severity="Warning", message="w1")
    log.mark_step_end(outcome="success")
    log.flush()

    run_info = _read_sheet(output_file, process_log.RUN_INFO).iloc[0]
    assert run_info["error_count"] == 1
    assert run_info["warning_count"] == 1
    assert run_info[f"count_{process_log.GENERAL_WARNINGS}"] == 2


def test_run_info_counts_exclude_unrelated_cross_cutting_sub_step(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(
        output_file,
        step="post_process_step2",
        owned_steps={"post_process_step2", "manual_rework_review"},
    )
    log.mark_step_start(prefix="test")
    log.log_rework_still_undefined(
        severity="Warning",
        source_name="doc-a",
        row=0,
        tag_name="Tag",
        span_start=0,
        span_end=1,
        message="msg",
        step="manual_rework_review",
    )
    log.mark_step_end(outcome="success")
    log.flush()

    run_info = _read_sheet(output_file, process_log.RUN_INFO).iloc[0]
    assert run_info[f"count_{process_log.REWORK_STILL_UNDEFINED}"] == 0


def test_run_info_rows_refreshed_not_accumulated_per_step(tmp_path):
    output_file = tmp_path / "process_log.xlsx"

    log1 = ProcessLog(output_file, step="run_main")
    log1.mark_step_start(prefix="test")
    log1.mark_step_end(outcome="success")
    log1.flush()

    log2 = ProcessLog(output_file, step="post_process_step1")
    log2.mark_step_start(prefix="test")
    log2.mark_step_end(outcome="success")
    log2.flush()

    run_info = _read_sheet(output_file, process_log.RUN_INFO)
    assert len(run_info) == 2

    log1_again = ProcessLog(output_file, step="run_main")
    log1_again.mark_step_start(prefix="test")
    log1_again.mark_step_end(outcome="error", notes="boom")
    log1_again.flush()

    run_info = _read_sheet(output_file, process_log.RUN_INFO)
    assert len(run_info) == 2
    run_main_row = run_info[run_info["step"] == "run_main"].iloc[0]
    assert run_main_row["outcome"] == "error"
    assert run_main_row["notes"] == "boom"


def test_run_info_counts_severity_and_per_sheet(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main")
    log.mark_step_start(prefix="test")
    log.log_general_warning(severity="Error", message="e1")
    log.log_general_warning(severity="Warning", message="w1")
    log.log_general_warning(severity="Info", message="i1")
    log.log_missing_source_document(
        severity="Warning", source_name="doc", context="ctx", message="m1"
    )
    log.mark_step_end(outcome="success")
    log.flush()

    run_info = _read_sheet(output_file, process_log.RUN_INFO).iloc[0]
    assert run_info["error_count"] == 1
    assert run_info["warning_count"] == 2
    assert run_info["info_count"] == 1
    assert run_info[f"count_{process_log.GENERAL_WARNINGS}"] == 3
    assert run_info[f"count_{process_log.MISSING_SOURCE_DOCUMENT}"] == 1


def test_call_stack_column_captures_real_caller_chain(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main")

    def _inner():
        log.log_general_warning(severity="Warning", message="boom")

    def _outer():
        _inner()

    _outer()
    log.flush()

    df = _read_sheet(output_file, process_log.GENERAL_WARNINGS)
    call_stack = df.iloc[0]["call_stack"]
    assert "test_process_log.py" in call_stack
    assert "_outer" in call_stack
    assert "_inner" in call_stack
    # Root (outermost) caller first.
    assert call_stack.index("_outer") < call_stack.index("_inner")
    # process_log.py's own frames (log_general_warning, _append) are filtered out.
    assert "log_general_warning" not in call_stack
    assert "_append" not in call_stack


def test_current_outside_active_is_working_noop():
    log = process_log.current()
    # None of these should raise or write anything.
    log.log_ai_coding_error(
        severity="Error",
        source_name="doc",
        row=0,
        configuration_name="cfg",
        error_type="TestError",
        message="msg",
    )
    log.log_missing_source_document(
        severity="Warning", source_name="doc", context="ctx", message="msg"
    )
    log.log_general_warning(severity="Warning", message="msg")
    log.mark_step_start(prefix="test")
    log.mark_step_end(outcome="success")
    log.flush()


def test_active_restores_previous_log_on_exit(tmp_path):
    outer = ProcessLog(tmp_path / "outer.xlsx", step="run_main")
    with process_log.active(outer):
        assert process_log.current() is outer
        inner = ProcessLog(tmp_path / "inner.xlsx", step="post_process_step1")
        with process_log.active(inner):
            assert process_log.current() is inner
        assert process_log.current() is outer
    assert process_log.current() is not outer


def test_active_restores_previous_log_on_exception(tmp_path):
    outer = ProcessLog(tmp_path / "outer.xlsx", step="run_main")
    with process_log.active(outer):
        inner = ProcessLog(tmp_path / "inner.xlsx", step="post_process_step1")
        try:
            with process_log.active(inner):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert process_log.current() is outer


def test_thread_safety_concurrent_log_calls(tmp_path):
    output_file = tmp_path / "process_log.xlsx"
    log = ProcessLog(output_file, step="run_main")

    call_count = 200

    def _log_one(i):
        log.log_general_warning(severity="Warning", message=f"warning {i}")

    with process_log.active(log):
        threads = [threading.Thread(target=_log_one, args=(i,)) for i in range(call_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    log.flush()
    df = _read_sheet(output_file, process_log.GENERAL_WARNINGS)
    assert len(df) == call_count
