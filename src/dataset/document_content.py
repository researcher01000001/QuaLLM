"""Groups a project's source spreadsheet into documents"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.config.project import (
    DATASET_COLUMN,
    DOCUMENT_KEY_COLUMNS,
    DOCUMENT_NAME_TEMPLATE,
    DOCUMENT_ORDER_COLUMN,
    DOCUMENT_ORDER_COLUMN_LABEL,
    GENERAL_CLUSTER_COLUMN,
    META_OUTPUT_FILE,
    PARA_META_COLUMNS,
)
from src.core import process_log
from src.core.models import DisplayColumn

_document_rows_cache: dict[str, pd.DataFrame] | None = None


def row_text_value(value: Any) -> str:
    """A META_OUTPUT_FILE TEXT_COLUMN cell as a string, or "" for a missing
    (NaN) transcript row -- pandas' Series.astype(str) does NOT stringify
    real float NaN values on an object-dtype column (it leaves them as NaN),
    so joining raw .astype(str) output can raise a TypeError; this handles
    that explicitly instead.

    Also collapses any embedded line break within the cell into a single
    space: DocumentLoader.load() (src/io/loaders.py) splits a written .txt file
    into Row objects via str.splitlines() (one row per line), so a cell
    containing its own newline would otherwise split into more than one
    Row, inflating the row count and desyncing every following row's index
    from its spreadsheet position. Each line is stripped of leading/trailing
    whitespace (and the joined result stripped again, to cover a leading or
    trailing blank line) before joining, so a padded cell value doesn't leak
    stray spaces into the written line.

    The single definition of "what a row's transcript text looks like as a
    written line" -- shared by scripts/atlas_preperation.py (writing one
    line per row into a .txt file) and src/coding/ai_coding.py (matching a .txt
    file's lines back to their source rows), so the two can never drift
    apart into disagreeing about it."""
    if pd.isna(value):
        return ""
    return " ".join(line.strip() for line in str(value).splitlines()).strip()


def _normalize_number(value: Any) -> Any:
    """Convert a pandas/numpy scalar (e.g. numpy.int64) to the equivalent
    native Python type, so a formatted document name doesn't leak numpy's
    repr or a spurious ".0" suffix (from a column upcast to float64 because
    some other row has a missing value). Leaves every other value untouched."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _document_rows() -> dict[str, pd.DataFrame]:
    """doc_name -> that document's META_OUTPUT_FILE rows, sorted by
    DOCUMENT_ORDER_COLUMN. Reads and groups the whole spreadsheet on first call, then
    reuses the cached result for the lifetime of the process."""
    global _document_rows_cache
    if _document_rows_cache is None:
        df = pd.read_excel(
            META_OUTPUT_FILE["filename"], sheet_name=META_OUTPUT_FILE["sheetname"]
        )
        documents: dict[str, pd.DataFrame] = {}
        for _, group in df.groupby(DOCUMENT_KEY_COLUMNS, sort=False):
            group = group.sort_values(DOCUMENT_ORDER_COLUMN)
            first_row = group.iloc[0]
            format_values: dict = {}
            try:
                format_values = {
                    column: _normalize_number(value)
                    for column, value in first_row.to_dict().items()
                }
                doc_name = DOCUMENT_NAME_TEMPLATE.format(**format_values)
            except (KeyError, ValueError) as e:
                row_label = str(format_values)
                process_log.current().log_document_key_match_failure(
                    severity="Warning",
                    source_name=row_label,
                    call_site="document_content._document_rows",
                    message=(
                        f"Row group {row_label} couldn't be formatted into "
                        f"DOCUMENT_NAME_TEMPLATE ({e}); this document is unreachable."
                    ),
                )
                continue
            documents[doc_name] = group
        _document_rows_cache = documents
    return _document_rows_cache


def document_names() -> list[str]:
    """Every canonical document name resolved from META_OUTPUT_FILE."""
    return list(_document_rows().keys())


def known_conditions() -> list[str]:
    """Every distinct document condition actually present in META_OUTPUT_FILE
    (the "condition" column of config_project.DOCUMENT_KEY_COLUMNS), in
    first-seen order -- the dataset's own answer to "what conditions exist,"
    rather than a hardcoded guess that can drift out of sync with the
    spreadsheet. Each _document_rows() group has one constant "condition"
    value by construction (DOCUMENT_KEY_COLUMNS groups by it), so this reuses
    that cache instead of re-reading the spreadsheet."""
    return list(
        dict.fromkeys(group[GENERAL_CLUSTER_COLUMN].iloc[0] for group in _document_rows().values())
    )


def text_rows_for(doc_name: str) -> pd.DataFrame | None:
    """doc_name's source rows, sorted by DOCUMENT_ORDER_COLUMN."""
    return _document_rows().get(doc_name)


def rows_for_document(
    doc_name: str, row_filter: Callable[[object], bool]
) -> pd.DataFrame | None:
    """doc_name's source rows (sorted by DOCUMENT_ORDER_COLUMN, see text_rows_for())
    kept where row_filter(dataset_value) is True -- dataset_value is each
    row's config_project.DATASET_COLUMN value, or None for every row if the
    document has no DATASET_COLUMN at all (e.g. scripts/sample_selector.py
    hasn't been run yet on this checkout). Returns None if doc_name doesn't
    exist. Generic fetch-filter-sort mechanics only, reused by
    scripts/atlas_preperation.py (writing .txt files) and src/coding/ai_coding.py
    (resolving a .txt file's lines back to these same rows) -- see
    src.dataset.selection.row_filter_for_txt_subfolder()."""
    rows = text_rows_for(doc_name)
    if rows is None:
        return None
    if DATASET_COLUMN in rows.columns:
        dataset_values = rows[DATASET_COLUMN]
    else:
        dataset_values = pd.Series([None] * len(rows), index=rows.index)
    return rows[dataset_values.map(row_filter)]


def all_rows_matching(
    row_filter: Callable[[pd.Series], bool], sort_column: str | None = None
) -> pd.DataFrame:
    """Every META_OUTPUT_FILE row (across all documents at once) where
    row_filter(row) is True, optionally sorted by sort_column (cast to int
    first, so a mixed/object dtype sorts numerically rather than as
    strings). Built from the same cached _document_rows() groups (via
    pd.concat()), so this never re-reads the spreadsheet.

    This is the whole-row counterpart to rows_for_document(): a "random"-mode
    split's selection (see src.dataset.selection.random_mode_row_filter()) isn't
    scoped to one document at all -- it's a filter over the whole sheet --
    so its lookup mechanics are equally generic, just a different filter
    granularity (a whole row instead of one column's value)."""
    groups = _document_rows()
    if not groups:
        return pd.DataFrame()
    all_rows = pd.concat(groups.values(), ignore_index=False)
    if all_rows.empty:
        return all_rows
    mask = all_rows.apply(row_filter, axis=1)
    filtered = all_rows[mask]
    if sort_column and sort_column in filtered.columns:
        filtered = filtered.sort_values(sort_column, key=lambda s: s.astype(int))
    return filtered


def display_column_for(doc_name: str) -> DisplayColumn:
    """Build doc_name's DisplayColumn: DOCUMENT_ORDER_COLUMN's value for every row, in
    the same order DocumentLoader.load() builds Row objects from the .txt
    file (both ultimately derive from this same sorted-by-DOCUMENT_ORDER_COLUMN slice
    of META_OUTPUT_FILE, via text_rows_for())."""
    rows = text_rows_for(doc_name)
    values = (
        {}
        if rows is None
        else {i: str(value) for i, value in enumerate(rows[DOCUMENT_ORDER_COLUMN])}
    )
    return DisplayColumn(
        source_column=DOCUMENT_ORDER_COLUMN, label=DOCUMENT_ORDER_COLUMN_LABEL, values=values
    )


def paragraph_meta() -> dict[str, dict[int, dict[str, str]]]:
    """Per-paragraph metadata for every document: doc_name -> DOCUMENT_ORDER_COLUMN
    value -> {column: value}, restricted to PARA_META_COLUMNS."""
    result: dict[str, dict[int, dict[str, str]]] = {}
    for doc_name, rows in _document_rows().items():
        per_position: dict[int, dict[str, str]] = {}
        for _, row in rows.iterrows():
            try:
                position = int(row[DOCUMENT_ORDER_COLUMN])
            except (TypeError, ValueError):
                continue

            meta: dict[str, str] = {}
            for column, _label in PARA_META_COLUMNS:
                if column not in rows.columns:
                    continue
                value = row[column]
                meta[column] = "" if pd.isna(value) else str(value).strip()
            per_position[position] = meta
        result[doc_name] = per_position
    return result
