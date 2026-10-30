"""Logic for Step 1 to review the undefined/target tags"""

import shutil
from pathlib import Path

import pandas as pd

from src.coding.consensus import write_consensus_json
from src.config import project as config_project
from src.core import process_log
from src.core.models import Document, Quote, Tag
from src.dataset import document_identity
from src.dataset.document_identity import condition_match
from src.io.io_utils import resolve_output_dir
from src.io.loaders import ConsensusQuoteLoader
from src.post_processing import review_database, review_page, tag_export


def copy_and_load_consensus(
    source_file: Path,
    dest_file: Path,
    tags: list[Tag],
    source_doc_names: list[str],
) -> dict[str, list[Quote]]:
    """Copy ``source_file`` to ``dest_file`` and load it back via
    ``ConsensusQuoteLoader``. Used by Step 2 to make its own import copy from
    Step 1's finished output (``consensus_with_defined_tags.json``)."""
    resolve_output_dir(dest_file)
    shutil.copyfile(source_file, dest_file)
    print(f"Copied consensus: {source_file} -> {dest_file}")

    quotes_by_doc = ConsensusQuoteLoader(dest_file, tags, condition_match=condition_match).load_quotes()
    aligned = {doc_name: quotes_by_doc.get(doc_name, []) for doc_name in source_doc_names}
    quote_count = sum(len(quotes) for quotes in aligned.values())
    print(f"Loaded {quote_count} consensus quote(s).")
    return aligned


def build_step1_consensus_file(
    documents: dict[str, Document],
    quotes_by_doc: dict[str, list[Quote]],
    tags: list[Tag],
    dest_file: Path,
) -> dict[str, list[Quote]]:
    """Build and write "the created consensus file": every quote's tags are
    exploded into one single-tag consensus entry each, so a double-coded
    span (two co-occurring tags) becomes two separate entries, matching the
    granularity the review page edits at. Reloads the written file via
    ``ConsensusQuoteLoader`` and returns the result, so Step 1's rest of the
    pipeline (and Step 2, on the next run) always work from the file, not
    from in-memory state.
    """
    exploded_records = review_page.explode_consensus_records(documents, quotes_by_doc)
    consensus_rows = [
        {
            "phrase": record["phrase"],
            "tags": [record["tag"]],
            "source": {"source_name": record["source_name"], "row": record["row"]},
            "start": record["start"],
            "end": record["end"],
        }
        for record in exploded_records
    ]
    write_consensus_json(consensus_rows, dest_file)
    print(f"Wrote created consensus ({len(consensus_rows)} single-tag entries): {dest_file}")

    quotes = ConsensusQuoteLoader(dest_file, tags, condition_match=condition_match).load_quotes()
    return {doc_name: quotes.get(doc_name, []) for doc_name in documents}


def build_step1_target_records(
    documents: dict[str, Document],
    quotes_by_doc: dict[str, list[Quote]],
    target_tag_names: list[str],
) -> tuple[list[dict], dict[str, Document]]:
    """Explode ``quotes_by_doc`` into per-tag review records (one row per
    tag, mirroring the final review's schema), flag which ones carry a
    target tag (``is_target``, computed *before* any database prefill can
    change ``tag``, so it always reflects what the AI/consensus originally
    proposed), and keep every record -- target and non-target alike --
    belonging to a paragraph with >=1 target-tag record (non-target records
    stay visible for context, just frozen in the restricted page).

    Returns ``(records, filtered_documents)`` -- ``filtered_documents`` has
    each ``Document``'s ``rows`` cut down to just the kept paragraphs, ready
    to hand to ``render_manual_review_html()`` unchanged.
    """
    full_records = review_page.explode_consensus_records(documents, quotes_by_doc)
    target_tag_set = set(target_tag_names)
    for record in full_records:
        record["is_target"] = record["tag"] in target_tag_set
        record["raw_tag"] = record["tag"]

    target_paragraph_keys = {
        (record["source_name"], record["row"]) for record in full_records if record["is_target"]
    }
    target_records = [
        record
        for record in full_records
        if (record["source_name"], record["row"]) in target_paragraph_keys
    ]

    filtered_documents: dict[str, Document] = {}
    for doc_name, document in documents.items():
        kept_rows = [
            row for row in document.rows if (doc_name, row.index) in target_paragraph_keys
        ]
        if kept_rows:
            filtered_documents[doc_name] = Document(
                source_name=document.source_name,
                rows=kept_rows,
                display_column=document.display_column,
            )

    return target_records, filtered_documents


def _database_row_for_lookup(record: dict, documents: dict[str, Document]) -> dict | None:
    """Forward adapter: a target-tag review record -> the xlsx-database row
    shape (``{*DOCUMENT_KEY_COLUMNS, GENERAL_CLUSTER_COLUMN, start, end,
    ai_tag, new_tag, ...}``), the same shape ``review_database``'s
    tolerance-matching functions expect. Returns ``None`` (skip) when the
    document name doesn't resolve to an identity."""
    doc_name = record["source_name"]
    identity = document_identity.get_document_key(doc_name)
    if identity is None:
        process_log.current().log_document_key_match_failure(
            severity="Warning",
            source_name=doc_name,
            call_site="undefined_tags._database_row_for_lookup",
            message=(
                f"Document name '{doc_name}' doesn't match DOCUMENT_NAME_TEMPLATE's "
                "shape; its target-tag record was skipped for database matching."
            ),
        )
        return None
    condition = document_identity.condition_for(doc_name)
    document = documents.get(doc_name)
    display_value = document.display_value(record["row"]) if document else None
    return {
        **identity,
        config_project.GENERAL_CLUSTER_COLUMN: condition,
        config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value,
        "span": record["phrase"],
        "start": record["start"],
        "end": record["end"],
        "ai_tag": record["raw_tag"],
        "new_tag": "",
    }


def apply_database_prefill_to_records(
    records: list[dict],
    documents: dict[str, Document],
    database_rows: list[dict],
    tags_by_name: dict[str, Tag],
    overlap_threshold: float,
) -> int:
    """Reverse adapter: pre-fill each target-tag record's ``tag`` from the
    cross-run database (tolerance-matched via
    ``review_database.prefill_new_tags_from_database()``, unchanged).

    A resolved value of ``review_database.DELETED_TAG_VALUE`` pre-marks
    the record ``status="deleted"`` instead of touching ``tag`` -- the review
    page won't show it as an active occurrence by default (mirrors any other
    deleted record), but the reviewer can still restore it like any edit.

    A comma-joined multi-tag resolution (double-coding) sets ``tag`` to the
    first name and appends a new sibling record (same span/paragraph,
    ``is_target=True``, fresh ``id``, ``status="original"``) for each
    additional name -- mutates ``records`` in place with the new siblings.

    Prefilled records keep ``status="original"``: this is a suggestion the
    reviewer can override like any other edit, not a change to log/track.

    Returns the number of records touched (tag changed or pre-deleted).
    """
    target_records = [record for record in records if record.get("is_target")]
    lookup_rows: list[dict] = []
    record_by_lookup_id: dict[int, dict] = {}
    for record in target_records:
        lookup_row = _database_row_for_lookup(record, documents)
        if lookup_row is None:
            continue
        lookup_rows.append(lookup_row)
        record_by_lookup_id[id(lookup_row)] = record

    prefilled_rows = review_database.prefill_new_tags_from_database(
        lookup_rows, database_rows, tags_by_name, overlap_threshold
    )

    touched = 0
    new_records: list[dict] = []
    for lookup_row, prefilled in zip(lookup_rows, prefilled_rows):
        record = record_by_lookup_id[id(lookup_row)]
        new_tag_value = str(prefilled.get("new_tag", "")).strip()
        if not new_tag_value or new_tag_value == record["tag"]:
            continue

        if new_tag_value == review_database.DELETED_TAG_VALUE:
            record["status"] = "deleted"
            touched += 1
            continue

        names = [part.strip() for part in new_tag_value.split(",") if part.strip()]
        record["tag"] = names[0]
        for extra_name in names[1:]:
            new_records.append(
                {
                    **record,
                    "id": f"{record['id']}::prefill::{extra_name}",
                    "tag": extra_name,
                    "status": "original",
                }
            )
        touched += 1

    records.extend(new_records)
    return touched


def build_database_rows_from_step1_records(
    records: list[dict],
    documents: dict[str, Document],
) -> list[dict]:
    """Write-back adapter for ``append_to_database()``/``rebuild_database()``:
    one row per target-tag record that carries a resolved decision -- either
    retagged (``tag != raw_tag``) or explicitly deleted (``new_tag`` becomes
    ``review_database.DELETED_TAG_VALUE``).

    A target record that keeps its AI tag but shares its span (``group_id``)
    with a resolved record also gets a row with ``new_tag == ai_tag``
    (``review_database.is_kept_row()``). That records the decision to keep it,
    so a later prefill does not apply the sibling's correction to it too.
    """
    rows: list[dict] = []
    resolved_groups: set[str] = set()
    kept_candidates: list[dict] = []
    for record in records:
        if not record.get("is_target"):
            continue
        if record["status"] == "deleted":
            new_tag = review_database.DELETED_TAG_VALUE
        elif record["tag"] != record.get("raw_tag"):
            new_tag = record["tag"]
        else:
            kept_candidates.append(record)
            continue

        lookup_row = _database_row_for_lookup(record, documents)
        if lookup_row is None:
            continue
        rows.append({**lookup_row, "new_tag": new_tag})
        if record.get("group_id"):
            resolved_groups.add(record["group_id"])

    for record in kept_candidates:
        if record.get("group_id") not in resolved_groups:
            continue  # also skips records without a group_id
        lookup_row = _database_row_for_lookup(record, documents)
        if lookup_row is None:
            continue
        rows.append({**lookup_row, "new_tag": record["tag"]})
    return rows


def build_corrected_consensus(
    documents: dict[str, Document],
    quotes_by_doc: dict[str, list[Quote]],
    step1_records: list[dict],
) -> list[dict]:
    """Merge the reviewer's saved Step 1 edits back into the full consensus:
    every paragraph the Step 1 page touched is replaced wholesale by
    ``step1_records``; every other paragraph passes through untouched.
    Falls back to "untouched only" when ``step1_records`` is empty (no Step 1
    page saved yet).
    """
    full_records = review_page.explode_consensus_records(documents, quotes_by_doc)
    step1_paragraph_keys = {(record["source_name"], record["row"]) for record in step1_records}
    untouched = [
        record
        for record in full_records
        if (record["source_name"], record["row"]) not in step1_paragraph_keys
    ]
    return tag_export.collapse_to_consensus_schema(untouched + step1_records)


def log_step1_review_report(
    step1_records: list[dict],
    documents: dict[str, Document],
) -> pd.DataFrame:
    """Log, and return as a DataFrame, one outcome row per target-tag record
    (always, regardless of status -- a complete audit trail) plus one row
    per non-target record that was actually edited (only possible when the
    restrict toggle was turned off; logged as its own category, not an
    error). Each saved record already carries its own final ``status``
    (kept/retagged/moved/moved+retagged/deleted), so no before/after diffing
    is needed.
    """
    rows: list[dict] = []
    for record in step1_records:
        is_target = bool(record.get("is_target"))
        status = record["status"]
        if not is_target and status == "original":
            continue

        document = documents.get(record["source_name"])
        display_value = document.display_value(record["row"]) if document else None
        tag_name = record.get("raw_tag", record["tag"])
        severity = "Warning" if status == "deleted" else "Info"
        message = (
            f"Target tag '{tag_name}' in '{record['source_name']}' [{status}]"
            if is_target
            else (
                f"Non-target tag '{record['tag']}' in '{record['source_name']}' was "
                f"edited ({status}) despite the restrict toggle."
            )
        )

        process_log.current().log_step1_review_outcome(
            severity=severity,
            source_name=record["source_name"],
            row=record["row"],
            display_value=display_value,
            tag_name=tag_name,
            span_start=record["start"],
            span_end=record["end"],
            is_target=is_target,
            status=status,
            message=message,
            step=process_log.STEP_MANUAL_REWORK_REVIEW,
        )
        rows.append(
            {
                "severity": severity,
                "source_name": record["source_name"],
                "row": record["row"],
                config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value,
                "tag_name": tag_name,
                "span_start": record["start"],
                "span_end": record["end"],
                "is_target": is_target,
                "status": status,
                "message": message,
            }
        )

    return pd.DataFrame(rows)
