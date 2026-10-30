"""Logic for Step 3 for exporting after post processing"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.config import project as config_project
from src.core import process_log
from src.core.models import Document, Tag
from src.dataset import document_identity
from src.io.io_utils import resolve_output_dir


def collapse_to_consensus_schema(records: list[dict]) -> list[dict]:
    """Regroup edited per-tag review rows back into consensus.json-style
    records: one per distinct (source_name, start, end), each with a
    ``tags`` list.

    Grouped by *current* position rather than the original ``group_id``,
    since a Move can cause tags that started on the same span to end up at
    different positions afterwards — they must then export as separate
    records. Rows with ``status == "deleted"`` are dropped.
    """
    groups: dict[tuple[str, int, int], dict] = {}
    order: list[tuple[str, int, int]] = []

    for row in records:
        if row["status"] == "deleted":
            continue
        key = (row["source_name"], row["start"], row["end"])
        if key not in groups:
            groups[key] = {
                "phrase": row["phrase"],
                "tags": [],
                "source": {
                    "source_name": row["source_name"],
                    "row": row["row"],
                },
                "start": row["start"],
                "end": row["end"],
            }
            order.append(key)
        groups[key]["tags"].append(row["tag"])

    return [groups[key] for key in order]


def find_still_undefined_tag_names_in_rows(
    consensus_rows: Sequence[dict],
    tags_by_name: Mapping[str, Tag],
    excluded_keys: frozenset[tuple[str, int]] = frozenset(),
) -> list[dict]:
    """Sibling of ``review_page.find_still_undefined_tags()`` for Step 3's data shape:
    ``collapse_to_consensus_schema()``'s plain tag-name-string records. A name
    is still-undefined when it isn't in the codebook at all, or resolves to a
    codebook tag with ``additional_tag=True``.

    A record whose ``(source_name, row)`` is in ``excluded_keys`` is skipped
    entirely -- its tags will never reach output.xlsx (see
    ``append_ordered_tags_to_xlsx()``), so flagging them as still-undefined
    would be pure noise.

    Logs each occurrence to the process log (``step=STEP_MANUAL_REWORK_REVIEW``)
    and also returns the rows, for testability.
    """
    rows: list[dict] = []
    for record in consensus_rows:
        source = record.get("source", {})
        doc_name = source.get("source_name")
        row_index = source.get("row")
        if (doc_name, row_index) in excluded_keys:
            continue
        for tag_name in record.get("tags", []):
            tag = tags_by_name.get(tag_name)
            if tag is not None and not tag.additional_tag:
                continue
            row = {
                "source_name": doc_name,
                "row": row_index,
                "tag_name": tag_name,
                "span_start": record.get("start"),
                "span_end": record.get("end"),
            }
            rows.append(row)
            process_log.current().log_rework_still_undefined(
                severity="Warning",
                source_name=doc_name,
                row=row_index,
                tag_name=tag_name,
                span_start=record.get("start"),
                span_end=record.get("end"),
                message=(
                    f"Tag '{tag_name}' in '{doc_name}' is still undefined after manual rework."
                ),
                step=process_log.STEP_MANUAL_REWORK_REVIEW,
            )
    return rows


def _can_co_occur(tag_name: str, tags_by_name: Mapping[str, Tag]) -> bool:
    """Whether ``tag_name`` is allowed to double-code with another tag on the
    same span. Tags not in the codebook (undefined/additional) aren't subject
    to the double-coding rule, so they default to True."""
    tag = tags_by_name.get(tag_name)
    if tag is None:
        return True
    return bool(tag.can_co_occur)


def _deterministic_shuffle(items: list[str], seed_material: str) -> list[str]:
    """Shuffle ``items`` using a PRNG seeded from a hash of ``seed_material``.

    Looks unpredictable (nothing meaningful like alphabetical order) but is
    perfectly reproducible: the same seed_material always yields the same
    permutation, so re-running on unchanged data produces identical output.
    """
    if len(items) <= 1:
        return list(items)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_ordered_tags(
    documents: Mapping[str, Document],
    consensus_rows: Sequence[dict],
    tags_by_name: Mapping[str, Tag],
    excluded_keys: frozenset[tuple[str, int]] = frozenset(),
) -> list[dict]:
    """Build one ordered tag-name list per paragraph of every document, from
    collapsed consensus-schema rows (see ``collapse_to_consensus_schema()``).

    Within a paragraph, records are ordered by span start ascending, ties
    broken by span length ascending (shortest first). Within one record
    (tags co-occurring on the exact same span), any tag that cannot
    double-code comes first, followed by the double-codable tags in a
    deterministic-but-unpredictable order (see ``_deterministic_shuffle()``,
    seeded from that paragraph's text plus the span's phrase).

    Every paragraph of every document gets an entry, even with an empty tag
    list, so downstream consumers (the output.xlsx append) can rely on full
    coverage rather than treating a missing entry as "no tags". Documents
    whose identity can't be resolved via ``document_identity.get_document_key()``
    are skipped, since there's no way to join them back to source rows.

    A paragraph whose ``(doc_name, doc_row.index)`` is in ``excluded_keys``
    (see ``review_page.build_excluded_paragraph_keys()``) still gets an entry here --
    marked ``"excluded": True`` -- so this JSON stays a full audit trail of
    every paragraph; ``append_ordered_tags_to_xlsx()`` is what actually drops
    it from the exported file. Each excluded paragraph is also logged.
    """
    by_row: dict[tuple[str, int], list[dict]] = {}
    for row in consensus_rows:
        source = row["source"]
        if source.get("row") is None:
            continue
        key = (source["source_name"], source["row"])
        by_row.setdefault(key, []).append(row)

    results: list[dict] = []
    for doc_name, document in documents.items():
        identity = document_identity.get_document_key(doc_name)
        if identity is None:
            process_log.current().log_document_key_match_failure(
                severity="Warning",
                source_name=doc_name,
                call_site="tag_export.build_ordered_tags",
                message=(
                    f"Document name '{doc_name}' doesn't match DOCUMENT_NAME_TEMPLATE's "
                    "shape; skipped -- can't be joined back to output.xlsx."
                ),
            )
            continue

        for doc_row in document.rows:
            rows = by_row.get((doc_name, doc_row.index), [])
            rows_sorted = sorted(rows, key=lambda r: (r["start"], r["end"] - r["start"]))

            ordered_tags: list[str] = []
            for row in rows_sorted:
                fixed = [t for t in row["tags"] if not _can_co_occur(t, tags_by_name)]
                flexible = [t for t in row["tags"] if _can_co_occur(t, tags_by_name)]
                seed_material = f"{doc_row.text}|{row['phrase']}"
                ordered_tags.extend(fixed)
                ordered_tags.extend(_deterministic_shuffle(flexible, seed_material))

            is_excluded = (doc_name, doc_row.index) in excluded_keys
            if is_excluded:
                display_value = document.display_value(doc_row.index)
                process_log.current().log_paragraph_excluded(
                    severity="Info",
                    source_name=doc_name,
                    row=doc_row.index,
                    display_value=display_value,
                    message=(
                        f"Paragraph {display_value!r} of '{doc_name}' is marked "
                        f"excluded ({len(ordered_tags)} tag(s) discarded); omitted "
                        "from the exported output.xlsx."
                    ),
                )

            results.append(
                {
                    "source_name": doc_name,
                    "identity": identity,
                    "row": doc_row.index,
                    "display_value": document.display_value(doc_row.index),
                    "tags": ordered_tags,
                    "excluded": is_excluded,
                }
            )

    return results


def append_ordered_tags_to_xlsx(
    ordered_tags: Sequence[dict],
    source_xlsx: Path,
    output_xlsx: Path,
    column_name: str,
    sheet_name: str | int = 0,
) -> int:
    """Copy ``source_xlsx`` to ``output_xlsx`` with one new column added: a
    comma-separated tag list per row, matched by
    ``(*config_project.DOCUMENT_KEY_COLUMNS, config_project.DOCUMENT_ORDER_COLUMN)`` --
    the order column's value is each row's ``display_value`` (see
    ``build_ordered_tags()``/``Document.display_value()``), parsed back to
    an int to match ``source_xlsx``'s numeric ``DOCUMENT_ORDER_COLUMN`` cells.

    Any ``ordered_tags`` entry with ``"excluded": True`` (see
    ``build_ordered_tags()``) has its matching row dropped from
    ``output_xlsx`` entirely, rather than merely left with an empty tags
    cell -- an excluded paragraph is not exported at all.

    ``sheet_name`` selects which sheet of ``source_xlsx`` to read (defaults
    to the first sheet, matching ``pandas.read_excel``'s own default).

    Returns the number of (non-excluded) rows that received a (possibly
    empty) match.
    """
    key_columns = config_project.DOCUMENT_KEY_COLUMNS
    order_column = config_project.DOCUMENT_ORDER_COLUMN
    lookup = {
        (
            *(row["identity"][col] for col in key_columns),
            int(row["display_value"]),
        ): ", ".join(row["tags"])
        for row in ordered_tags
    }
    excluded_keys = {
        (*(row["identity"][col] for col in key_columns), int(row["display_value"]))
        for row in ordered_tags
        if row.get("excluded")
    }

    df = pd.read_excel(source_xlsx, sheet_name=sheet_name)
    matched_rows: list[bool] = []

    def _row_key(row: pd.Series) -> tuple:
        return (*(row[col] for col in key_columns), row[order_column])

    def _lookup(row: pd.Series) -> str:
        key = _row_key(row)
        value = lookup.get(key)
        if key not in excluded_keys:
            matched_rows.append(value is not None)
        return value or ""

    df[column_name] = df.apply(_lookup, axis=1)

    if excluded_keys:
        excluded_mask = df.apply(lambda row: _row_key(row) in excluded_keys, axis=1)
        df = df[~excluded_mask]

    resolve_output_dir(output_xlsx)
    df.to_excel(output_xlsx, index=False)
    return sum(matched_rows)
