import sys
from pathlib import Path

from src.analysis.metrics import calculate_metrics_for_quote_sets
from src.coding.consensus import (
    build_consensus_records,
    log_double_coding_conflicts,
    write_consensus_json,
)
from src.config.general import (
    AI_CODING_FILES,
    CONSENSUS_FILENAME,
    DATA_RESULTS_DIR,
    PREFIX,
    PROCESS_LOG_FILE,
    RUN_TYPE,
    SUPERCONSENSUS_NAME,
    UNDEFINED_TAGS_ENABLED,
)
from src.config.post_processing import (
    POST_PROCESS_DATABASE_FILE,
    POST_PROCESS_DATABASE_USE_ALL_PREFIXES,
    POST_PROCESS_INCLUDE_UNDEFINED_TAGS,
    POST_PROCESS_LOAD_DATABASE,
    POST_PROCESS_SPAN_OVERLAP_THRESHOLD,
    POST_PROCESS_STEP,
    POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE,
    POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE,
    POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE,
    POST_PROCESS_STEP1_REBUILD_DATABASE,
    POST_PROCESS_STEP1_REVIEW_HTML,
    POST_PROCESS_STEP2_CONSENSUS_FILE,
    POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE,
    POST_PROCESS_STEP2_REPORT_HTML,
    POST_PROCESS_STEP2_REPORT_XLSX_EXCLUDING_UNDEFINED_TAGS,
    POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS,
    POST_PROCESS_STEP2_REVIEW_HTML,
    POST_PROCESS_STEP3_ORDERED_TAGS_JSON,
    POST_PROCESS_STEP3_OUTPUT_XLSX,
    POST_PROCESS_STEP3_REVIEW_JSON,
    POST_PROCESS_STEP3_TAGS_COLUMN,
    POST_PROCESS_STEP3_USE_EXISTING_JSON,
    POST_PROCESS_TAGS,
)
from src.config.project import (
    DOCUMENT_KEY_COLUMNS,
    HUMAN_CODING_FILES,
    META_OUTPUT_FILE,
    PARA_META_COLUMNS,
)
from src.core import process_log
from src.core.models import Document, Quote, Tag
from src.dataset import run_scope
from src.dataset.document_content import paragraph_meta
from src.dataset.document_identity import condition_match
from src.io.io_utils import read_json, write_json, write_workbook_sheets, resolve_output_dir
from src.io.loaders import AtlasQuoteLoader, ConsensusQuoteLoader, load_default_tags
from src.post_processing.review_database import (
    append_to_database,
    build_target_tags,
    load_database_rows,
    rebuild_database,
    resolve_database_files,
)
from src.post_processing.review_page import (
    build_document_identity,
    build_excluded_paragraph_keys,
    build_paragraph_checks,
    build_paragraph_meta,
    build_review_metadata,
    collect_review_tags,
    count_reviewed_paragraphs,
    explode_consensus_records,
    extract_review_data,
    find_still_undefined_tags,
    read_review_protect_file_flag,
    render_manual_review_html,
)
from src.post_processing.undefined_tags import (
    apply_database_prefill_to_records,
    build_corrected_consensus,
    build_database_rows_from_step1_records,
    build_step1_consensus_file,
    build_step1_target_records,
    copy_and_load_consensus,
    log_step1_review_report,
)
from src.post_processing.tag_export import (
    append_ordered_tags_to_xlsx,
    build_ordered_tags,
    collapse_to_consensus_schema,
    find_still_undefined_tag_names_in_rows,
)
from src.report.html import DocumentProcessor, apply_coder_comparison_scores, build_render_codings
from src.report.xlsx import export_metrics_workbook


def main():
    print(f"Running post-processing step {POST_PROCESS_STEP} (PREFIX={PREFIX})...")
    if POST_PROCESS_STEP == 1:
        run_step1()
    elif POST_PROCESS_STEP == 2:
        run_step2()
    elif POST_PROCESS_STEP == 3:
        run_step3()
    else:
        raise ValueError(f"Unknown POST_PROCESS_STEP: {POST_PROCESS_STEP!r}")


# ==========================================
# Step 1: undefined-tags review (HTML) & database
# ==========================================


def run_step1() -> None:
    """Build/refresh "the created consensus" (consensus_unmodified.json) and
    the undefined-tags-only manual-review HTML page."""
    log = process_log.ProcessLog(PROCESS_LOG_FILE, step=process_log.STEP_POST_PROCESS_STEP1)
    log.mark_step_start(
        prefix=PREFIX,
        undefined_tags_enabled=UNDEFINED_TAGS_ENABLED,
        post_process_step=1,
    )
    with process_log.active(log):
        try:
            _run_step1_body()
            log.mark_step_end(outcome="success")
        except SystemExit as exc:
            log.mark_step_end(outcome="aborted", notes=str(exc))
            raise
        except Exception as exc:
            log.mark_step_end(outcome="error", notes=str(exc))
            raise
        finally:
            log.flush()


def _run_step1_body() -> None:
    tags = load_code_tags()
    tags_by_name = {tag.name: tag for tag in tags}
    document_paths = run_scope.find_source_documents(RUN_TYPE)
    source_doc_names = [path.stem for path in document_paths]
    documents = run_scope.load_source_documents(
        document_paths, context="post_process_document_loading"
    )

    if not handle_existing_step1_review(tags_by_name, documents):
        return

    superconsensus_file = AI_CODING_FILES / CONSENSUS_FILENAME
    raw_quotes = ConsensusQuoteLoader(
        superconsensus_file, tags, condition_match=condition_match
    ).load_quotes()
    raw_quotes = {doc_name: raw_quotes.get(doc_name, []) for doc_name in source_doc_names}

    quotes_by_doc = build_step1_consensus_file(
        documents, raw_quotes, tags, POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE
    )

    target_tags = build_target_tags(
        POST_PROCESS_TAGS, {"consensus": quotes_by_doc}, POST_PROCESS_INCLUDE_UNDEFINED_TAGS
    )
    print(f"Target tags: {target_tags}")

    records, target_documents = build_step1_target_records(documents, quotes_by_doc, target_tags)

    if POST_PROCESS_LOAD_DATABASE:
        database_files = resolve_database_files(
            DATA_RESULTS_DIR, POST_PROCESS_DATABASE_FILE, POST_PROCESS_DATABASE_USE_ALL_PREFIXES
        )
        database_rows = load_database_rows(database_files)
        touched = apply_database_prefill_to_records(
            records, documents, database_rows, tags_by_name, POST_PROCESS_SPAN_OVERLAP_THRESHOLD
        )
        print(f"Pre-filled {touched} record(s) from {len(database_files)} database file(s).")

    export_step1_review_html(target_documents, records, tags, quotes_by_doc)


def load_code_tags() -> list[Tag]:
    print("Loading tags...")
    return load_default_tags()


def handle_existing_step1_review(
    tags_by_name: dict[str, Tag], documents: dict[str, Document]
) -> bool:
    """Mirrors the old xlsx-settings-sheet integrate/rebuild/protect/discard-
    warning flow, sourced instead from the previous run's saved
    ``POST_PROCESS_STEP1_REVIEW_HTML``. Returns True if Step 1 should
    proceed to rebuild "the created consensus" and regenerate the review
    page this run; False if it already handled everything that needed doing
    (protected, or just rebuilt/integrated the database) and the run should
    stop here.
    """
    review_data = extract_review_data(POST_PROCESS_STEP1_REVIEW_HTML)
    if review_data is None:
        return True

    existing_records = review_data.get("records", [])
    database_rows_from_existing = build_database_rows_from_step1_records(
        existing_records, documents
    )

    if read_review_protect_file_flag(POST_PROCESS_STEP1_REVIEW_HTML):
        if POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE and database_rows_from_existing:
            added = append_to_database(
                database_rows_from_existing,
                POST_PROCESS_DATABASE_FILE,
                tags_by_name,
                POST_PROCESS_SPAN_OVERLAP_THRESHOLD,
            )
            print(f"Saved {added} manually defined tag(s) to database: {POST_PROCESS_DATABASE_FILE}")
            process_log.current().log_audit_decision(
                severity="Info",
                decision_point="step1_integrate_into_database",
                detail=f"{added} manually defined tag(s) merged into {POST_PROCESS_DATABASE_FILE}",
                outcome="merged",
            )
        print(
            f"{POST_PROCESS_STEP1_REVIEW_HTML} is protected (protect_file=True); "
            "it was not regenerated."
        )
        process_log.current().log_audit_decision(
            severity="Info",
            decision_point="step1_protect_file",
            detail=f"{POST_PROCESS_STEP1_REVIEW_HTML} has protect_file=True",
            outcome="file_not_overwritten",
        )
        return False

    if POST_PROCESS_STEP1_REBUILD_DATABASE:
        try:
            answer = input(
                "POST_PROCESS_STEP1_REBUILD_DATABASE is True: this will delete the existing "
                f"database at {POST_PROCESS_DATABASE_FILE} and rebuild it from "
                f"{POST_PROCESS_STEP1_REVIEW_HTML}'s manually defined tags. Continue? [y/N]: "
            )
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            print("Aborted: database was not rebuilt.")
            process_log.current().log_audit_decision(
                severity="Warning",
                decision_point="step1_rebuild_database",
                detail="POST_PROCESS_STEP1_REBUILD_DATABASE=True",
                outcome="aborted",
                answer_given=answer,
            )
            sys.exit(1)

        added = rebuild_database(
            database_rows_from_existing,
            POST_PROCESS_DATABASE_FILE,
            tags_by_name,
            POST_PROCESS_SPAN_OVERLAP_THRESHOLD,
        )
        print(f"Rebuilt database from {added} manually defined tag(s): {POST_PROCESS_DATABASE_FILE}")
        process_log.current().log_audit_decision(
            severity="Info",
            decision_point="step1_rebuild_database",
            detail=f"{POST_PROCESS_DATABASE_FILE} rebuilt from {added} manually defined tag(s)",
            outcome="rebuilt",
            answer_given=answer,
        )
        return False

    if POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE:
        added = append_to_database(
            database_rows_from_existing,
            POST_PROCESS_DATABASE_FILE,
            tags_by_name,
            POST_PROCESS_SPAN_OVERLAP_THRESHOLD,
        )
        print(f"Saved {added} manually defined tag(s) to database: {POST_PROCESS_DATABASE_FILE}")
        process_log.current().log_audit_decision(
            severity="Info",
            decision_point="step1_integrate_into_database",
            detail=f"{added} manually defined tag(s) merged into {POST_PROCESS_DATABASE_FILE}",
            outcome="merged",
        )
        return False

    if database_rows_from_existing:
        decision = process_log.confirm(
            f"{len(database_rows_from_existing)} manually defined tag(s) found in "
            f"{POST_PROCESS_STEP1_REVIEW_HTML}, but POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE "
            "is False. Overwrite and discard them? [y/N]: ",
            decision_point="step1_overwrite_prompt",
            detail=(
                f"{len(database_rows_from_existing)} manually defined row(s) in "
                f"{POST_PROCESS_STEP1_REVIEW_HTML}"
            ),
            severity_yes="Warning",
            outcome_yes="overwritten",
            outcome_no="aborted",
        )
        if not decision:
            print("Aborted: existing Step 1 review page was not overwritten.")
            sys.exit(1)

    return True


def export_step1_review_html(
    target_documents: dict[str, Document],
    records: list[dict],
    tags: list[Tag],
    quotes_by_doc: dict[str, list[Quote]],
) -> None:
    """Generate the undefined-tags-only manual-review HTML from the target-
    tag-filtered records."""
    print("Generating Step 1 undefined-tags review HTML...")
    paragraph_checks = build_paragraph_checks(target_documents)
    excluded_keys_to_copy = confirm_copy_excluded_paragraphs(POST_PROCESS_STEP1_REVIEW_HTML)
    if excluded_keys_to_copy:
        for check in paragraph_checks:
            if (check["source_name"], check["row"]) in excluded_keys_to_copy:
                check["excluded"] = True
        print(f"Copied {len(excluded_keys_to_copy)} excluded paragraph marker(s).")
    metadata = build_review_metadata(POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE, PREFIX)
    metadata["restrict_to_target_tags"] = True
    review_tags = collect_review_tags(tags, quotes_by_doc)
    zero_based_paragraph_meta = build_paragraph_meta(target_documents, paragraph_meta())
    doc_identity = build_document_identity(target_documents)
    filter_column_labels = dict(PARA_META_COLUMNS)
    filter_columns = [(col, filter_column_labels.get(col, col)) for col in DOCUMENT_KEY_COLUMNS]
    render_manual_review_html(
        target_documents,
        records,
        paragraph_checks,
        review_tags,
        metadata,
        POST_PROCESS_STEP1_REVIEW_HTML,
        paragraph_meta=zero_based_paragraph_meta,
        document_identity=doc_identity,
        filter_columns=filter_columns,
        meta_columns=PARA_META_COLUMNS,
        mode="restricted",
    )
    print(f"Step 1 review HTML created: {POST_PROCESS_STEP1_REVIEW_HTML}")


# ==========================================
# Step 2: manual supervision (corrected consensus + validation)
# ==========================================


def run_step2() -> None:
    """Close out Step 1 (read its saved HTML, write
    consensus_with_defined_tags.json), then build the full, unrestricted
    manual-review HTML plus a validation report."""
    log = process_log.ProcessLog(
        PROCESS_LOG_FILE,
        step=process_log.STEP_POST_PROCESS_STEP2,
        owned_steps={process_log.STEP_POST_PROCESS_STEP2, process_log.STEP_MANUAL_REWORK_REVIEW},
    )
    log.mark_step_start(
        prefix=PREFIX,
        undefined_tags_enabled=UNDEFINED_TAGS_ENABLED,
        post_process_step=2,
    )
    with process_log.active(log):
        try:
            _build_manual_supervision_body(SUPERCONSENSUS_NAME)
            log.mark_step_end(outcome="success")
        except SystemExit as exc:
            log.mark_step_end(outcome="aborted", notes=str(exc))
            raise
        except Exception as exc:
            log.mark_step_end(outcome="error", notes=str(exc))
            raise
        finally:
            log.flush()


def _build_manual_supervision_body(label: str) -> None:
    tags = load_code_tags()
    document_paths = run_scope.find_source_documents(RUN_TYPE)
    source_doc_names = [path.stem for path in document_paths]
    documents = run_scope.load_source_documents(
        document_paths, context="post_process_document_loading"
    )

    # ---- Phase 1: close out Step 1 -------------------------------------
    step1_quotes = ConsensusQuoteLoader(
        POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE, tags, condition_match=condition_match
    ).load_quotes()
    step1_quotes = {doc_name: step1_quotes.get(doc_name, []) for doc_name in source_doc_names}

    step1_review_data = extract_review_data(POST_PROCESS_STEP1_REVIEW_HTML) or {}
    step1_records = step1_review_data.get("records", [])

    corrected_records = build_corrected_consensus(documents, step1_quotes, step1_records)
    write_consensus_json(corrected_records, POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE)
    print(f"Wrote created consensus with defined tags: {POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE}")

    report_df = log_step1_review_report(step1_records, documents)

    # ---- Phase 2: Step 2 proper -----------------------------------------
    updated_quotes = copy_and_load_consensus(
        POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE,
        POST_PROCESS_STEP2_CONSENSUS_FILE,
        tags,
        source_doc_names,
    )

    log_double_coding_conflicts(
        documents, updated_quotes, context_message="after manual correction"
    )
    still_undefined = find_still_undefined_tags(documents, updated_quotes)
    if still_undefined:
        print(
            f"Warning: {len(still_undefined)} tag occurrence(s) are still undefined "
            "after manual rework (see the process log)."
        )

    records = build_consensus_records(documents, updated_quotes)
    write_consensus_json(records, POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE)
    print(f"Wrote reviewed consensus: {POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE}")

    if read_review_protect_file_flag(POST_PROCESS_STEP2_REVIEW_HTML):
        print(
            f"{POST_PROCESS_STEP2_REVIEW_HTML} is protected (protect_file=True); "
            "it was not regenerated."
        )
    else:
        export_manual_review_html(
            documents, updated_quotes, tags, POST_PROCESS_STEP2_REVIEW_HTML
        )

    atlas_quotes = load_human_quotes(tags, source_doc_names)

    export_validation_report(
        documents,
        atlas_quotes,
        updated_quotes,
        label,
        POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS,
    )
    if UNDEFINED_TAGS_ENABLED:
        export_validation_report(
            documents,
            atlas_quotes,
            updated_quotes,
            label,
            POST_PROCESS_STEP2_REPORT_XLSX_EXCLUDING_UNDEFINED_TAGS,
            drop_undefined_tags=True,
        )
    export_validation_html(
        documents, atlas_quotes, updated_quotes, label, POST_PROCESS_STEP2_REPORT_HTML
    )

    if not report_df.empty:
        write_workbook_sheets(
            {"step1_review_outcomes": report_df},
            POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS,
            append_to_existing_file=True,
            replace_existing_sheet=True,
        )
        print(
            f"Appended step1_review_outcomes ({len(report_df)} row(s)) to "
            f"{POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS}"
        )


def _existing_excluded_paragraph_keys(existing_file: Path) -> frozenset[tuple[str, int]]:
    """Paragraphs marked "excluded" in ``existing_file``'s currently-saved
    review data, or an empty set if the file doesn't exist / has no
    review-data blob / has no excluded paragraphs."""
    if not existing_file.exists():
        return frozenset()
    review_data = extract_review_data(existing_file)
    if review_data is None:
        return frozenset()
    return build_excluded_paragraph_keys(review_data.get("paragraph_checks", []))


def confirm_copy_excluded_paragraphs(existing_file: Path) -> frozenset[tuple[str, int]]:
    """Regenerating ``existing_file`` always seeds a fresh, all-unchecked
    ``paragraph_checks`` -- which would silently drop any "excluded" markers
    a human already set on the previous version. If the existing file has
    any, ask on the console whether to carry them forward. Returns the keys
    to copy (empty if none exist or the answer is no)."""
    excluded_keys = _existing_excluded_paragraph_keys(existing_file)
    if not excluded_keys:
        return frozenset()

    copy = process_log.confirm(
        f"{len(excluded_keys)} paragraph(s) are marked excluded in the existing "
        f"{existing_file.name}. Copy them into the regenerated file? [Y/n]: ",
        decision_point="step2_copy_excluded_paragraphs",
        detail=f"{len(excluded_keys)} excluded paragraph(s) found in existing review file",
        default_yes=True,
        severity_no="Info",
        outcome_yes="copied",
        outcome_no="not_copied",
    )
    return excluded_keys if copy else frozenset()


def export_manual_review_html(
    documents: dict[str, Document],
    updated_quotes: dict[str, list[Quote]],
    tags: list[Tag],
    output_file: Path,
) -> None:
    """Generate the editable manual-review HTML from the corrected consensus."""
    print("Generating manual-review HTML...")
    records = explode_consensus_records(documents, updated_quotes)
    paragraph_checks = build_paragraph_checks(documents)
    excluded_keys_to_copy = confirm_copy_excluded_paragraphs(output_file)
    if excluded_keys_to_copy:
        for check in paragraph_checks:
            if (check["source_name"], check["row"]) in excluded_keys_to_copy:
                check["excluded"] = True
        print(f"Copied {len(excluded_keys_to_copy)} excluded paragraph marker(s).")
    metadata = build_review_metadata(POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE, PREFIX)
    review_tags = collect_review_tags(tags, updated_quotes)
    zero_based_paragraph_meta = build_paragraph_meta(documents, paragraph_meta())
    document_identity = build_document_identity(documents)
    filter_column_labels = dict(PARA_META_COLUMNS)
    filter_columns = [(col, filter_column_labels.get(col, col)) for col in DOCUMENT_KEY_COLUMNS]
    render_manual_review_html(
        documents,
        records,
        paragraph_checks,
        review_tags,
        metadata,
        output_file,
        paragraph_meta=zero_based_paragraph_meta,
        document_identity=document_identity,
        filter_columns=filter_columns,
        meta_columns=PARA_META_COLUMNS,
    )
    print(f"Manual-review HTML created: {output_file}")


def load_human_quotes(tags: list[Tag], source_doc_names: list[str]) -> dict[str, list[Quote]]:
    print("Loading ATLAS quotes (Human)...")
    atlas_quotes = AtlasQuoteLoader(
        HUMAN_CODING_FILES,
        tags,
        condition_match=condition_match,
    ).load_quotes()
    return run_scope.align_quote_documents(atlas_quotes, source_doc_names)


def export_validation_report(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    updated_quotes: dict[str, list[Quote]],
    label: str,
    output_file: Path,
    drop_undefined_tags: bool = False,
) -> None:
    print("Calculating validation metrics...")
    coded_results = calculate_metrics_for_quote_sets(
        documents=documents,
        human_quotes=atlas_quotes,
        ai_quotes=updated_quotes,
        configuration_name=label,
        iteration=None,
        is_consensus=True,
        drop_undefined_tags=drop_undefined_tags,
    )
    metric_results = [result.metrics for result in coded_results]
    export_metrics_workbook(metric_results, output_file)
    print(f"Validation report created: {output_file}")


def export_validation_html(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    updated_quotes: dict[str, list[Quote]],
    label: str,
    output_file: Path,
) -> None:
    print("Generating validation HTML...")
    resolve_output_dir(output_file)
    para_meta = paragraph_meta()

    html_consensus_quotes = {label: updated_quotes}
    apply_coder_comparison_scores(documents, atlas_quotes, html_consensus_quotes)
    codings = build_render_codings(atlas_quotes, html_consensus_quotes)

    processor = DocumentProcessor(
        run_scope.processing_dirs(RUN_TYPE),
        para_meta=para_meta,
        documents=documents,
    )
    processor.generate_html(codings, output_file, prefix=PREFIX)
    print(f"Validation HTML created: {output_file}")


# ==========================================
# Step 3: export (ordered per-paragraph tags + output.xlsx column)
# ==========================================


def extract_manual_review_data(html_file: Path) -> dict | None:
    """Extract the raw review data (metadata + paragraph_checks + records)
    from whatever is currently saved in ``html_file`` -- regardless of its
    ``protect_file`` flag; Step 3 always reads it, protected or not, since
    that flag only stops Step 2 from *regenerating* the HTML."""
    review_data = extract_review_data(html_file)
    if review_data is None:
        print(f"No manual-review data found at {html_file}.")
    return review_data


def confirm_paragraphs_reviewed(paragraph_checks: list[dict]) -> None:
    """Ask on the console before continuing if not every paragraph is
    marked as reviewed yet. 'y' continues anyway; anything else stops the
    run, matching Step 1's rebuild_database confirmation pattern."""
    checked, total = count_reviewed_paragraphs(paragraph_checks)
    if total == 0 or checked == total:
        process_log.current().log_audit_decision(
            severity="Info",
            decision_point="step3_paragraphs_reviewed_confirmation",
            detail=f"{checked}/{total} paragraphs reviewed",
            outcome="all_reviewed" if total else "no_paragraphs",
        )
        return

    decision = process_log.confirm(
        f"{checked}/{total} paragraphs are marked as reviewed in the manual-review "
        "HTML -- not all of them yet. Continue anyway? [y/N]: ",
        decision_point="step3_paragraphs_reviewed_confirmation",
        detail=f"{checked}/{total} paragraphs reviewed",
        severity_yes="Warning",
        outcome_yes="continued_anyway",
        outcome_no="aborted",
    )
    if not decision:
        print("Aborted: Step 3 was not run.")
        sys.exit(1)


def write_manual_review_json(review_data: dict, output_file: Path) -> None:
    """Collapse an already-extracted review_data dict's records into
    consensus.json-style rows, and write them to output_file alongside which
    paragraphs were marked "excluded" on the manual-review page -- see
    ``_load_step3_review_data()`` for the read side of this shape."""
    consensus_rows = collapse_to_consensus_schema(review_data.get("records", []))
    excluded_keys = build_excluded_paragraph_keys(review_data.get("paragraph_checks", []))
    payload = {
        "consensus_rows": consensus_rows,
        "excluded_paragraphs": [list(key) for key in sorted(excluded_keys)],
    }
    write_json(payload, output_file)
    print(f"Manual-review JSON extracted: {output_file}")


def _load_step3_review_data(path: Path) -> tuple[list[dict], frozenset[tuple[str, int]]]:
    """Read ``path`` (as written by ``write_manual_review_json()``), returning
    ``(consensus_rows, excluded_keys)``.

    Also accepts the pre-"exclude marker" format -- a bare JSON list of
    consensus rows, with no exclusion info -- so a ``POST_PROCESS_STEP3_REVIEW_JSON``
    file written before this feature existed still loads under
    ``POST_PROCESS_STEP3_USE_EXISTING_JSON`` (treated as zero exclusions).
    """
    data = read_json(path)
    if isinstance(data, list):
        return data, frozenset()
    consensus_rows = data.get("consensus_rows", [])
    excluded_keys = frozenset(
        (source_name, row) for source_name, row in data.get("excluded_paragraphs", [])
    )
    return consensus_rows, excluded_keys


def run_step3() -> None:
    """Get the manual-review JSON (extracted from Step 2's HTML, or reused
    as already extracted per POST_PROCESS_STEP3_USE_EXISTING_JSON), build
    the ordered per-paragraph tag list, and append it as a new column to a
    copy of output.xlsx.

    When extracting fresh, asks for confirmation if not every paragraph is
    marked as reviewed yet -- POST_PROCESS_STEP3_USE_EXISTING_JSON skips this
    entirely, since the already-extracted JSON never carried the full
    paragraph_checks list to begin with (only the collapsed consensus rows
    plus which paragraphs were marked "excluded" are saved to it -- see
    write_manual_review_json()/_load_step3_review_data())."""
    log = process_log.ProcessLog(
        PROCESS_LOG_FILE,
        step=process_log.STEP_POST_PROCESS_STEP3,
        owned_steps={process_log.STEP_POST_PROCESS_STEP3, process_log.STEP_MANUAL_REWORK_REVIEW},
    )
    log.mark_step_start(
        prefix=PREFIX,
        undefined_tags_enabled=UNDEFINED_TAGS_ENABLED,
        post_process_step=3,
    )
    with process_log.active(log):
        try:
            _run_step3_body()
            log.mark_step_end(outcome="success")
        except SystemExit as exc:
            log.mark_step_end(outcome="aborted", notes=str(exc))
            raise
        except Exception as exc:
            log.mark_step_end(outcome="error", notes=str(exc))
            raise
        finally:
            log.flush()


def _run_step3_body() -> None:
    if POST_PROCESS_STEP3_USE_EXISTING_JSON:
        if not POST_PROCESS_STEP3_REVIEW_JSON.exists():
            raise FileNotFoundError(
                f"POST_PROCESS_STEP3_USE_EXISTING_JSON is True but "
                f"{POST_PROCESS_STEP3_REVIEW_JSON} does not exist."
            )
        print(f"Using existing review JSON: {POST_PROCESS_STEP3_REVIEW_JSON}")
    else:
        review_data = extract_manual_review_data(POST_PROCESS_STEP2_REVIEW_HTML)
        if review_data is None:
            print("Aborted: Step 3 was not run.")
            sys.exit(1)
        confirm_paragraphs_reviewed(review_data.get("paragraph_checks", []))
        write_manual_review_json(review_data, POST_PROCESS_STEP3_REVIEW_JSON)

    consensus_rows, excluded_keys = _load_step3_review_data(POST_PROCESS_STEP3_REVIEW_JSON)

    tags = load_code_tags()
    tags_by_name = {tag.name: tag for tag in tags}
    document_paths = run_scope.find_source_documents(RUN_TYPE)
    documents = run_scope.load_source_documents(
        document_paths, context="post_process_document_loading"
    )

    still_undefined = find_still_undefined_tag_names_in_rows(
        consensus_rows, tags_by_name, excluded_keys
    )
    if still_undefined:
        print(
            f"Warning: {len(still_undefined)} tag occurrence(s) are still undefined "
            "after manual rework (see the process log)."
        )

    ordered_tags = build_ordered_tags(documents, consensus_rows, tags_by_name, excluded_keys)
    write_json(ordered_tags, POST_PROCESS_STEP3_ORDERED_TAGS_JSON)
    print(f"Wrote ordered tags: {POST_PROCESS_STEP3_ORDERED_TAGS_JSON}")

    if excluded_keys:
        print(
            f"{len(excluded_keys)} paragraph(s) marked excluded; their rows were "
            "omitted from the output.xlsx export (see the process log)."
        )

    matched = append_ordered_tags_to_xlsx(
        ordered_tags,
        META_OUTPUT_FILE["filename"],
        POST_PROCESS_STEP3_OUTPUT_XLSX,
        POST_PROCESS_STEP3_TAGS_COLUMN,
        sheet_name=META_OUTPUT_FILE["sheetname"],
    )
    print(
        f"Appended '{POST_PROCESS_STEP3_TAGS_COLUMN}' column ({matched} matched row(s)): "
        f"{POST_PROCESS_STEP3_OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()
