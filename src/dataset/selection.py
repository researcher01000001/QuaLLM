"""DATASET_COLUMN row-membership parsing"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.project import (
    DATASET_COLUMN,
    EXCLUDED_VALUES,
    NAMING_BY_MODE,
    RANDOM_DATASET_ORDER_COLUMN,
    RUN_TYPE_CONFIGS,
)
from src.dataset import document_content


def is_excluded_row(value: object) -> bool:
    """True if a DATASET_COLUMN value marks a row as technically unusable
    (never part of any split, see parse_dataset_value())."""
    return value in EXCLUDED_VALUES


def is_validation_row(value: object) -> bool:
    """True if a DATASET_COLUMN value marks a row as selected for
    validation (any cluster)."""
    return isinstance(value, str) and value.startswith("validation_")


def document_driven_row_filter(dataset_type: str) -> Callable[[object], bool]:
    """A DATASET_COLUMN predicate for a "document_driven"-mode split: a row
    belongs to dataset_type if its DATASET_COLUMN value carries that type's
    prefix (e.g. "training_no advice" for dataset_type="training")."""
    prefix = f"{dataset_type}_"

    def row_is_valid(value: object) -> bool:
        return isinstance(value, str) and value.startswith(prefix)

    return row_is_valid


# Ready-made DATASET_COLUMN predicates for the two split types that are
# always whole-document (their RUN_TYPE_CONFIGS entry's selection_mode is
# the fixed literal "document_driven", not a TRAINING_SELECTION_MODE-style
# toggle -- see row_filter_for_txt_subfolder()'s name-based special case).
def _full_row_filter(value: object) -> bool:
    return not is_excluded_row(value)


def _full_without_validation_row_filter(value: object) -> bool:
    return not is_excluded_row(value) and not is_validation_row(value)


FULL_ROW_FILTER: Callable[[object], bool] = _full_row_filter
FULL_WITHOUT_VALIDATION_ROW_FILTER: Callable[[object], bool] = _full_without_validation_row_filter


def row_filter_for_txt_subfolder(subfolder_name: str) -> Callable[[object], bool] | None:
    """The DATASET_COLUMN predicate for a document-driven .txt subfolder
    ("full", "full_without_validation", or a "training"/"validation"
    subfolder whose RUN_TYPE_CONFIGS entry has selection_mode ==
    "document_driven"). Returns None for a "random"-mode subfolder (see
    random_mode_row_filter() instead -- random mode needs a whole-row
    filter, not a single-column one), "test" (selection_mode == "manual",
    resolved by sniffing instead -- see resolve_rows_for_txt_file()),
    or an unrecognized subfolder name.

    Reads RUN_TYPE_CONFIGS live on every call (not a value captured once) --
    a test can still monkeypatch one entry's selection_mode via
    monkeypatch.setitem()/dataclasses.replace() and see it take effect here
    immediately, the same way monkeypatching the old SELECTION_MODE_BY_TYPE
    dict used to.
    """
    if subfolder_name == "full":
        return FULL_ROW_FILTER
    if subfolder_name == "full_without_validation":
        return FULL_WITHOUT_VALIDATION_ROW_FILTER
    run_type_config = RUN_TYPE_CONFIGS.get(subfolder_name)
    if run_type_config is not None and run_type_config.selection_mode == "document_driven":
        return document_driven_row_filter(subfolder_name)
    return None


def random_mode_row_filter(dataset_type: str, cluster: str) -> Callable[[pd.Series], bool]:
    """A whole-row predicate for a "random"-mode split's cluster file (e.g.
    "validation_advice.txt"): a row belongs if its DATASET_COLUMN value
    parses (via parse_dataset_value()) to this exact dataset_type/cluster,
    and RANDOM_DATASET_ORDER_COLUMN isn't blank -- mirrors
    scripts/atlas_preperation.py::_write_random()'s selection exactly, so
    both writing and reading back use the same logic."""

    def row_is_valid(row: pd.Series) -> bool:
        value = row[DATASET_COLUMN]
        if not isinstance(value, str):
            return False
        parsed = parse_dataset_value(value)
        if parsed is None or parsed["type"] != dataset_type or parsed["cluster"] != cluster:
            return False
        return pd.notna(row[RANDOM_DATASET_ORDER_COLUMN])

    return row_is_valid


def parse_dataset_value(value: str) -> dict[str, Any] | None:
    """Parses DATASET_COLUMN's grammar: one of EXCLUDED_VALUES (a
    technically unusable row, not part of any split), or
    "[training|validation_]{cluster}" (e.g. "validation_advice", or
    "no advice" for an untagged, full-dataset-only row). Returns
    {"type": "training"|"validation"|None, "cluster": str}, or None if
    value is one of EXCLUDED_VALUES or doesn't match this shape.

    The row's order within its (type, cluster) group is a separate value
    (RANDOM_DATASET_ORDER_COLUMN), not encoded here -- keeping DATASET_COLUMN to
    just these two parts is what makes it easy to parse back with a plain
    prefix check, with no ambiguity from a trailing number.

    Pure string parsing only -- no spreadsheet access -- mirroring
    src.dataset.document_identity.parse_document_name().
    """
    if value in EXCLUDED_VALUES or not value:
        return None

    for candidate in ("training", "validation"):
        prefix = f"{candidate}_"
        if value.startswith(prefix):
            cluster = value[len(prefix) :]
            return {"type": candidate, "cluster": cluster} if cluster else None

    return {"type": None, "cluster": value}


def _resolve_test_subfolder_rows(stem: str) -> pd.DataFrame | None:
    """Row resolution for atlas_import/test/ (RUN_TYPE == "test", see
    config/general.py's RUN_TYPE docstring) -- files the user manually places there,
    not written by scripts/atlas_preperation.py, so there's no real
    selection_mode to key off of (RUN_TYPE_CONFIGS["test"].selection_mode ==
    "manual"). Instead, sniffs stem against every naming scheme the other
    subfolders actually use:

    - "{type}_{cluster}" (e.g. "validation_advice", "training_no advice"),
      matching the "training"/"validation" random-mode subfolders -- resolved
      the same way (random_mode_row_filter() + all_rows_matching(), ordered
      by NAMING_BY_MODE["manual"].order_column).
    - anything else (e.g. "45_advice_cycle_111", a document-driven name)
      resolved like the "full" subfolder: every non-excluded row for that
      document (document_content.rows_for_document() + FULL_ROW_FILTER).

    Returns None (document_content.rows_for_document()'s convention) when
    stem matches neither shape or names an unknown document."""
    for dataset_type in ("training", "validation"):
        prefix = f"{dataset_type}_"
        if stem.startswith(prefix):
            cluster = stem[len(prefix) :]
            return document_content.all_rows_matching(
                random_mode_row_filter(dataset_type, cluster),
                sort_column=NAMING_BY_MODE["manual"].order_column,
            )
    return document_content.rows_for_document(stem, FULL_ROW_FILTER)


def _align_test_rows(df: pd.DataFrame, paragraphs: list[str]) -> pd.DataFrame | None:
    """A contiguous len(paragraphs)-row window of df (already in document
    order) whose NAMING_BY_MODE["manual"].text_column values equal
    paragraphs exactly, in order -- or None if paragraphs is empty, longer
    than df, or no such window exists.

    Only used as a fallback for the "test" subfolder, whose files are placed
    manually and may hold just a contiguous excerpt of a document (e.g. the
    user trimmed a long transcript down for a quick check) rather than every
    row scripts/atlas_preperation.py would have written -- an exact row-count
    match is the common case and stays a plain equality check; this handles
    the shortened case by finding where in the full document that excerpt
    actually starts.

    Row text goes through document_content.row_text_value() -- the same
    function scripts/atlas_preperation.py uses to turn a row into the line
    it writes to a .txt file -- plus a .strip() on both sides to match
    src.coding.ai_coding.prepare_document()'s paragraph chunking
    (`[p.strip() for p in text.split("\\n") if p.strip()]`), so a row's text
    here is byte-comparable to a paragraph read back from disk."""
    n = len(paragraphs)
    if n == 0 or n > len(df):
        return None
    row_texts = [
        document_content.row_text_value(v).strip()
        for v in df[NAMING_BY_MODE["manual"].text_column]
    ]
    wanted = [p.strip() for p in paragraphs]
    for start in range(len(row_texts) - n + 1):
        if row_texts[start : start + n] == wanted:
            return df.iloc[start : start + n]
    return None


def resolve_rows_for_txt_file(
    filepath: Path, paragraphs: list[str]
) -> list[dict[str, Any] | None] | None:
    """The real META_OUTPUT_FILE row (every column) behind each of
    filepath's paragraphs, in order -- resolved the same way
    scripts/atlas_preperation.py selected rows for this file (or, for
    "test", by sniffing the filename and realigning -- that subfolder isn't
    written by that script at all), so this can never disagree with what's
    actually on disk.

    Returns None if the rows can't be resolved at all (an unrecognized
    subfolder/file name, or a genuine row-count mismatch) -- the caller
    should log this and degrade gracefully (no row data for that document)
    rather than risk pairing the wrong row with a paragraph.
    """
    subfolder = filepath.parent.name
    run_type_config = RUN_TYPE_CONFIGS.get(subfolder)
    if run_type_config is None:
        return None

    if subfolder == "test":
        df = _resolve_test_subfolder_rows(filepath.stem)
        if df is not None and len(df) != len(paragraphs):
            df = _align_test_rows(df, paragraphs)
    elif run_type_config.selection_mode == "random":
        prefix = f"{subfolder}_"
        if not filepath.stem.startswith(prefix):
            return None
        cluster = filepath.stem[len(prefix) :]
        df = document_content.all_rows_matching(
            random_mode_row_filter(subfolder, cluster),
            sort_column=NAMING_BY_MODE[run_type_config.selection_mode].order_column,
        )
        if len(df) != len(paragraphs):
            return None
    else:
        row_filter = row_filter_for_txt_subfolder(subfolder)
        if row_filter is None:
            return None
        df = document_content.rows_for_document(filepath.stem, row_filter)
        if df is None or len(df) != len(paragraphs):
            return None

    if df is None:
        return None
    return [row.to_dict() for _, row in df.iterrows()]
