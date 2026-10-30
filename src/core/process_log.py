"""Consolidated cross-lifecycle process log"""

from __future__ import annotations

import sys
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import project as config_project
from src.io.io_utils import write_workbook_sheets

Severity = Literal["Info", "Warning", "Error"]

CALL_STACK_COLUMN = "call_stack"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SELF_FILE = Path(__file__).resolve()


def _capture_call_path(max_frames: int = 25) -> str:
    """The chain of project-code call frames leading to this log call, root
    (outermost) first, formatted as ``relative/path.py:function:line``
    entries joined by ``" > "``.

    Frames inside this module and anything outside the project (stdlib,
    dependencies, pytest/test-runner internals) are skipped -- not just
    stopped at -- so the chain reflects only real caller code even when a
    library frame sits between two project frames (e.g. a
    ThreadPoolExecutor worker call).
    """
    entries: list[str] = []
    frame = sys._getframe(1)
    while frame is not None and len(entries) < max_frames:
        try:
            resolved = Path(frame.f_code.co_filename).resolve()
        except OSError:
            frame = frame.f_back
            continue
        if resolved == _SELF_FILE:
            frame = frame.f_back
            continue
        try:
            rel = resolved.relative_to(_PROJECT_ROOT)
        except ValueError:
            frame = frame.f_back
            continue
        entries.append(f"{rel.as_posix()}:{frame.f_code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    entries.reverse()
    return " > ".join(entries)


# ==========================================
# Step names
#
# The one and only place these strings are spelled out -- every entry point
# and module that needs to construct or compare a step tag imports the
# relevant constant from here rather than hardcoding the literal, so a rename
# only ever has to happen in one place.
# ==========================================
STEP_RUN_MAIN = "run_main"
STEP_POST_PROCESS_STEP1 = "post_process_step1"
STEP_POST_PROCESS_STEP2 = "post_process_step2"
STEP_POST_PROCESS_STEP3 = "post_process_step3"
STEP_MANUAL_REWORK_REVIEW = "manual_rework_review"


# ==========================================
# Sheet names
# ==========================================
RUN_INFO = "run_info"
AI_CODING_ERRORS = "ai_coding_errors"
MISSING_SOURCE_DOCUMENT = "missing_source_document"
QUOTE_POSITION_ISSUES = "quote_position_issues"
UNDEFINED_TAGS = "undefined_tags"
DOCUMENT_KEY_MATCH_FAILURES = "document_key_match_failures"
POST_PROCESSING_MATCH_FAILURES = "post_processing_match_failures"
BOLD_SPAN_OUT_OF_BOUNDS = "bold_span_out_of_bounds"
DOUBLE_CODING_CONFLICTS = "double_coding_conflicts"
REWORK_STILL_UNDEFINED = "rework_still_undefined"
STEP1_REVIEW_OUTCOMES = "step1_review_outcomes"
AUDIT_TRAIL = "audit_trail"
GENERAL_WARNINGS = "general_warnings"
PARAGRAPH_EXCLUSIONS = "paragraph_exclusions"

# Every issue sheet shares these three columns, in addition to the
# sheet-specific ones listed below.
_BASE_COLUMNS = ["severity", "step", "timestamp"]

SHEET_SCHEMAS: dict[str, list[str]] = {
    AI_CODING_ERRORS: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "configuration_name",
        "error_type",
        "message",
        "raw_output",
    ],
    MISSING_SOURCE_DOCUMENT: [*_BASE_COLUMNS, "source_name", "context", "message"],
    QUOTE_POSITION_ISSUES: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "issue_type",
        "quote_text",
        "context_text",
        "message",
    ],
    UNDEFINED_TAGS: [
        *_BASE_COLUMNS,
        "source_name",
        "tag_name",
        "reason",
        "resolution",
        "quote_text",
        "message",
    ],
    DOCUMENT_KEY_MATCH_FAILURES: [*_BASE_COLUMNS, "source_name", "call_site", "message"],
    POST_PROCESSING_MATCH_FAILURES: [
        *_BASE_COLUMNS,
        "source_name",
        "ai_tag",
        "new_tag",
        "start",
        "end",
        "failure_type",
        "message",
    ],
    BOLD_SPAN_OUT_OF_BOUNDS: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "span_start",
        "span_end",
        "paragraph_length",
        "message",
    ],
    DOUBLE_CODING_CONFLICTS: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "span_start",
        "span_end",
        "quote_text",
        "tag_a",
        "tag_b",
        "message",
    ],
    REWORK_STILL_UNDEFINED: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "tag_name",
        "span_start",
        "span_end",
        "message",
    ],
    STEP1_REVIEW_OUTCOMES: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "tag_name",
        "span_start",
        "span_end",
        "is_target",
        "status",
        "message",
    ],
    AUDIT_TRAIL: [*_BASE_COLUMNS, "decision_point", "detail", "answer_given", "outcome"],
    GENERAL_WARNINGS: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "message",
    ],
    PARAGRAPH_EXCLUSIONS: [
        *_BASE_COLUMNS,
        "source_name",
        "row",
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "message",
    ],
}

# Every issue sheet gets the call-stack column last, so the primary fields
# (severity/message/etc.) stay the first thing read; this is appended here
# rather than listed in every schema above to guarantee no sheet is missed.
for _columns in SHEET_SCHEMAS.values():
    _columns.append(CALL_STACK_COLUMN)

RUN_INFO_COLUMNS: list[str] = [
    "step",
    "started_at",
    "ended_at",
    "outcome",
    "prefix",
    "undefined_tags_enabled",
    "post_process_step",
    "coding_configurations",
    "error_count",
    "warning_count",
    "info_count",
    *[f"count_{sheet_name}" for sheet_name in SHEET_SCHEMAS],
    "notes",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stringify(value: object) -> object:
    """Excel cells can't hold list/tuple/set objects -- flatten them to a
    readable comma-separated string. Everything else passes through as-is."""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return value


def _read_existing_workbook(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    try:
        return pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception:
        # A corrupt/half-written/locked workbook shouldn't crash the pipeline;
        # treat it as if this is the first run (existing rows are lost, but
        # that's no worse than the log not existing yet).
        return {}


def _belongs_to_step(row_step: object, step: str) -> bool:
    """True if ``row_step`` is exactly ``step``, or a ``f"{step}/..."``
    namespaced sub-step of it (see each typed ``log_*()`` method's ``step``
    parameter). Used so a top-level step's ``run_info`` counts still include
    rows logged under a finer-grained sub-step (e.g. ``"run_main/ai-delegation"``
    counts toward ``"run_main"``'s totals), without pulling in unrelated
    cross-cutting sub-steps like ``"manual_rework_review"`` that don't share
    the prefix.
    """
    return row_step == step or (isinstance(row_step, str) and row_step.startswith(f"{step}/"))


class ProcessLog:
    """Accumulates this step's log rows in memory; nothing touches disk
    until ``flush()``. Not meant to be constructed directly for "no logging
    active" cases -- use ``process_log.current()`` instead, which returns a
    working no-op when no log is active."""

    def __init__(
        self,
        path: Path | None,
        step: str,
        owned_steps: set[str] | None = None,
        *,
        _noop: bool = False,
    ) -> None:
        self.path = path
        self.step = step
        self.owned_steps = owned_steps if owned_steps is not None else {step}
        self._noop = _noop
        self._rows: dict[str, list[dict]] = defaultdict(list)
        self._run_info: dict[str, object] = {}
        self._lock = threading.Lock()

    # ---- step lifecycle -------------------------------------------------

    def mark_step_start(self, **config_snapshot: object) -> None:
        if self._noop:
            return
        with self._lock:
            self._run_info.update(config_snapshot)
            self._run_info["started_at"] = _now()

    def mark_step_end(self, outcome: str = "success", notes: str = "") -> None:
        if self._noop:
            return
        with self._lock:
            self._run_info["ended_at"] = _now()
            self._run_info["outcome"] = outcome
            self._run_info["notes"] = notes

    # ---- typed log methods ------------------------------------------------

    def log_ai_coding_error(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        configuration_name: str | None,
        error_type: str,
        message: str,
        raw_output: str | None = None,
        step: str | None = None,
    ) -> None:
        self._append(
            AI_CODING_ERRORS,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            configuration_name=configuration_name,
            error_type=error_type,
            message=message,
            raw_output=raw_output,
        )

    def log_missing_source_document(
        self,
        *,
        severity: Severity,
        source_name: str,
        context: str,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            MISSING_SOURCE_DOCUMENT,
            step=step,
            severity=severity,
            source_name=source_name,
            context=context,
            message=message,
        )

    def log_quote_position_issue(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        issue_type: str,
        quote_text: str,
        context_text: str | None = None,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            QUOTE_POSITION_ISSUES,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            issue_type=issue_type,
            quote_text=quote_text,
            context_text=context_text,
            message=message,
        )

    def log_undefined_tag(
        self,
        *,
        severity: Severity,
        source_name: str | None,
        tag_name: str,
        reason: str,
        resolution: str,
        quote_text: str | None = None,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            UNDEFINED_TAGS,
            step=step,
            severity=severity,
            source_name=source_name,
            tag_name=tag_name,
            reason=reason,
            resolution=resolution,
            quote_text=quote_text,
            message=message,
        )

    def log_document_key_match_failure(
        self,
        *,
        severity: Severity,
        source_name: str,
        call_site: str,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            DOCUMENT_KEY_MATCH_FAILURES,
            step=step,
            severity=severity,
            source_name=source_name,
            call_site=call_site,
            message=message,
        )

    def log_post_processing_match_failure(
        self,
        *,
        severity: Severity,
        source_name: str | None,
        ai_tag: str | None,
        new_tag: str | None,
        start: int | None,
        end: int | None,
        failure_type: str,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            POST_PROCESSING_MATCH_FAILURES,
            step=step,
            severity=severity,
            source_name=source_name,
            ai_tag=ai_tag,
            new_tag=new_tag,
            start=start,
            end=end,
            failure_type=failure_type,
            message=message,
        )

    def log_bold_span_out_of_bounds(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        span_start: int,
        span_end: int,
        paragraph_length: int,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            BOLD_SPAN_OUT_OF_BOUNDS,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            span_start=span_start,
            span_end=span_end,
            paragraph_length=paragraph_length,
            message=message,
        )

    def log_double_code_conflict(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        span_start: int | None,
        span_end: int | None,
        quote_text: str,
        tag_a: str,
        tag_b: str,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            DOUBLE_CODING_CONFLICTS,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            span_start=span_start,
            span_end=span_end,
            quote_text=quote_text,
            tag_a=tag_a,
            tag_b=tag_b,
            message=message,
        )

    def log_rework_still_undefined(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        tag_name: str,
        span_start: int | None,
        span_end: int | None,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            REWORK_STILL_UNDEFINED,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            tag_name=tag_name,
            span_start=span_start,
            span_end=span_end,
            message=message,
        )

    def log_step1_review_outcome(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        tag_name: str,
        span_start: int | None,
        span_end: int | None,
        is_target: bool,
        status: str,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            STEP1_REVIEW_OUTCOMES,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            tag_name=tag_name,
            span_start=span_start,
            span_end=span_end,
            is_target=is_target,
            status=status,
            message=message,
        )

    def log_audit_decision(
        self,
        *,
        severity: Severity,
        decision_point: str,
        detail: str,
        outcome: str,
        answer_given: str | None = None,
        step: str | None = None,
    ) -> None:
        self._append(
            AUDIT_TRAIL,
            step=step,
            severity=severity,
            decision_point=decision_point,
            detail=detail,
            answer_given=answer_given,
            outcome=outcome,
        )

    def log_general_warning(
        self,
        *,
        severity: Severity,
        source_name: str | None = None,
        row: int | None = None,
        display_value: str | None = None,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            GENERAL_WARNINGS,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            message=message,
        )

    def log_paragraph_excluded(
        self,
        *,
        severity: Severity,
        source_name: str,
        row: int | None,
        display_value: str | None = None,
        message: str,
        step: str | None = None,
    ) -> None:
        self._append(
            PARAGRAPH_EXCLUSIONS,
            step=step,
            severity=severity,
            source_name=source_name,
            row=row,
            **{config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value},
            message=message,
        )

    # ---- internals --------------------------------------------------------

    def _append(self, sheet: str, *, step: str | None = None, **fields: object) -> None:
        if self._noop:
            return
        effective_step = step if step is not None else self.step
        if effective_step not in self.owned_steps:
            raise ValueError(
                f"step {effective_step!r} is not in this ProcessLog's "
                f"owned_steps {self.owned_steps!r}"
            )
        row = {
            "step": effective_step,
            "timestamp": _now(),
            **fields,
            CALL_STACK_COLUMN: _capture_call_path(),
        }
        with self._lock:
            self._rows[sheet].append(row)

    def _build_run_info_row(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        error_count = warning_count = info_count = 0
        for sheet_name in SHEET_SCHEMAS:
            rows_this_step = [
                row
                for row in self._rows.get(sheet_name, [])
                if _belongs_to_step(row.get("step"), self.step)
            ]
            counts[f"count_{sheet_name}"] = len(rows_this_step)
            for row in rows_this_step:
                severity = row.get("severity")
                if severity == "Error":
                    error_count += 1
                elif severity == "Warning":
                    warning_count += 1
                elif severity == "Info":
                    info_count += 1

        return {
            "step": self.step,
            "started_at": self._run_info.get("started_at"),
            "ended_at": self._run_info.get("ended_at"),
            "outcome": self._run_info.get("outcome"),
            "prefix": _stringify(self._run_info.get("prefix")),
            "undefined_tags_enabled": self._run_info.get("undefined_tags_enabled"),
            "post_process_step": self._run_info.get("post_process_step"),
            "coding_configurations": _stringify(self._run_info.get("coding_configurations")),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            **counts,
            "notes": self._run_info.get("notes", ""),
        }

    def flush(self) -> None:
        """Read the existing workbook (if any), replace every row tagged
        with one of ``self.owned_steps`` (and, for ``run_info``, the row for
        ``self.step``) with this run's fresh rows, and write the whole
        workbook back. Preserves rows belonging to every other step
        untouched, even in sheets this run didn't log anything to."""
        if self._noop or self.path is None:
            return

        with self._lock:
            rows_snapshot = {sheet: list(rows) for sheet, rows in self._rows.items()}
            run_info_row = self._build_run_info_row()

        existing_sheets = _read_existing_workbook(self.path)

        combined: dict[str, pd.DataFrame] = {}

        existing_run_info = existing_sheets.get(RUN_INFO)
        if existing_run_info is None or "step" not in existing_run_info.columns:
            existing_run_info = pd.DataFrame(columns=RUN_INFO_COLUMNS)
        preserved_run_info = existing_run_info[existing_run_info["step"] != self.step]
        fresh_run_info = pd.DataFrame([run_info_row], columns=RUN_INFO_COLUMNS)
        combined[RUN_INFO] = pd.concat(
            [preserved_run_info, fresh_run_info], ignore_index=True
        ).reindex(columns=RUN_INFO_COLUMNS)

        for sheet_name, columns in SHEET_SCHEMAS.items():
            existing_df = existing_sheets.get(sheet_name)
            if existing_df is None or "step" not in existing_df.columns:
                existing_df = pd.DataFrame(columns=columns)
            preserved = existing_df[~existing_df["step"].isin(self.owned_steps)]
            fresh_rows = rows_snapshot.get(sheet_name, [])
            fresh_df = pd.DataFrame(fresh_rows, columns=columns)
            combined[sheet_name] = pd.concat(
                [preserved, fresh_df], ignore_index=True
            ).reindex(columns=columns)

        write_workbook_sheets(combined, self.path)


# ==========================================
# Module-level "current log" singleton.
#
# A plain global (not contextvars.ContextVar): coding/ai_coding.py's
# ThreadPoolExecutor workers do not inherit ContextVar state set in the
# submitting thread, so a contextvars-based design would silently stop
# logging inside the one place in this codebase with real concurrency.
# ProcessLog's own internal lock makes concurrent log_* calls from worker
# threads safe.
# ==========================================
_NOOP = ProcessLog(path=None, step="__noop__", owned_steps=set(), _noop=True)
_current: ProcessLog | None = None
_current_lock = threading.Lock()


def current() -> ProcessLog:
    """The active ProcessLog, or a shared no-op instance if none is active."""
    with _current_lock:
        return _current if _current is not None else _NOOP


def confirm(
    prompt: str,
    *,
    decision_point: str,
    detail: str,
    default_yes: bool = False,
    severity_yes: Severity = "Info",
    severity_no: Severity = "Warning",
    outcome_yes: str = "confirmed",
    outcome_no: str = "declined",
) -> bool:
    """Ask a y/n question on the console, log the decision via
    ``log_audit_decision()``, and return it.

    ``default_yes=False`` (the default) shows a "[y/N]"-style prompt where
    only an explicit "y" counts as yes; ``default_yes=True`` shows "[Y/n]"
    where anything but "n" counts as yes. ``severity_*``/``outcome_*`` let a
    caller match its own audit-trail wording; the defaults match the most
    common shape (Info/"confirmed" on yes, Warning/"declined" on no).

    Callers that must stop the run on "no" still do that themselves (e.g.
    via ``sys.exit(1)``) -- this helper has no exit side effect, and logs
    only this one decision. A caller that needs to log something different
    after acting on "yes" (e.g. once it knows how many rows were affected)
    still does that itself.

    A closed/non-interactive stdin (piped from ``/dev/null``, a cron job, a
    CI runner with no attached TTY) raises ``EOFError`` from ``input()``;
    that is treated as an empty answer, so it falls through to each
    prompt's own printed default (``[y/N]``/``[Y/n]``) instead of crashing.
    """
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    decision = (answer.strip().lower() != "n") if default_yes else (answer.strip().lower() == "y")
    current().log_audit_decision(
        severity=severity_yes if decision else severity_no,
        decision_point=decision_point,
        detail=detail,
        outcome=outcome_yes if decision else outcome_no,
        answer_given=answer,
    )
    return decision


@contextmanager
def active(log: ProcessLog) -> Iterator[ProcessLog]:
    """Makes ``log`` the module-wide ``current()`` for the duration of the
    block, restoring whatever was active before on exit (including on
    exception)."""
    global _current
    with _current_lock:
        previous = _current
        _current = log
    try:
        yield log
    finally:
        with _current_lock:
            _current = previous
