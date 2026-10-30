"""Builds per-document .txt transcript files from META_OUTPUT_FILE rows,
organized under TXT_FILES into four subfolders: "training", "validation",
"full", and "full_without_validation" -- plus a "word" subfolder holding
one combined .docx per type (see the bottom of this docstring), and a
"test" subfolder that this script only ensures exists -- its content is
placed manually by the user for ad-hoc test runs (RUN_TYPE = "test"),
never generated here.

Which rows go where is entirely governed by the DATASET_COLUMN  in the file. It is created by scripts/sample_selector.py's
DATASET_COLUMN.

- "training"/"validation": how each is built depends on that type's
  SELECTION_MODE (config_project.TRAINING_SELECTION_MODE /
  VALIDATION_SELECTION_MODE):

Row selection for every subfolder is delegated to
src.dataset.selection.row_filter_for_txt_subfolder()/random_mode_row_filter() +
document_content.rows_for_document()/all_rows_matching() -- the same shared
logic src/coding/ai_coding.py uses to resolve a written .txt file's lines back to
their source rows, so there is exactly one definition of "which rows belong
to this file", not two that could drift apart.
"""

from pathlib import Path

from docx import Document as WordDocument

from src.config import project as config_project
from src.config.project import (
    DATASET_COLUMN,
    RANDOM_DATASET_ORDER_COLUMN,
    RUN_TYPE_CONFIGS,
    TEXT_COLUMN,
    TXT_FILES,
)
from src.dataset import selection as dataset_selection, document_content
from src.dataset.document_content import document_names, row_text_value
from src.dataset.selection import parse_dataset_value


def _whole_document_entries(subfolder_name: str) -> list[tuple[str, list[str]]]:
    """[(doc_name, [row_text, ...]), ...] for every document that has at
    least one row valid for subfolder_name (see
    src.dataset.selection.row_filter_for_txt_subfolder()) -- only those rows
    are kept, in the same order document_content already sorted them in. A
    document with zero valid rows is omitted entirely."""
    row_filter = dataset_selection.row_filter_for_txt_subfolder(subfolder_name)
    if row_filter is None:
        return []

    entries: list[tuple[str, list[str]]] = []
    for doc_name in document_names():
        valid_rows = document_content.rows_for_document(doc_name, row_filter)
        if valid_rows is None or valid_rows.empty:
            continue
        entries.append((doc_name, [row_text_value(v) for v in valid_rows[TEXT_COLUMN]]))
    return entries


def _write_txt_files(output_dir: Path, entries: list[tuple[str, list[str]]]) -> None:
    for title, row_texts in entries:
        output_path = output_dir / f"{title}.txt"
        output_path.write_text("\n".join(row_texts), encoding="utf-8")
        print(f"Wrote {output_path}")


def _write_whole_documents(
    output_dir: Path, subfolder_name: str
) -> list[tuple[str, list[str]]]:
    """One .txt file per document, keeping only the rows valid for
    subfolder_name (see _whole_document_entries()). Returns the same
    (title, rows) entries that were written, for reuse by the combined Word
    export."""
    entries = _whole_document_entries(subfolder_name)
    _write_txt_files(output_dir, entries)
    return entries


def _write_document_driven(dataset_type: str, output_dir: Path) -> list[tuple[str, list[str]]]:
    """Every document that has at least one row carrying dataset_type's
    prefix in DATASET_COLUMN, written with only those rows -- any other
    row in the same document (typically one individually marked
    "excluded") is left out of the content, not the whole document (see
    _write_whole_documents())."""
    return _write_whole_documents(output_dir, dataset_type)


def _write_random(dataset_type: str, output_dir: Path) -> list[tuple[str, list[str]]]:
    """One .txt file per cluster (DATASET_COLUMN's own cluster segment),
    concatenating every dataset_type-tagged row's TEXT_COLUMN value in
    RANDOM_DATASET_ORDER_COLUMN order. Named "{dataset_type}_{cluster}.txt".

    Clusters are discovered once from the whole spreadsheet, then each
    cluster's rows are resolved via
    document_content.all_rows_matching()/dataset_selection.random_mode_row_filter()
    -- the same shared logic src/coding/ai_coding.py uses to resolve these files' lines
    back to rows, so this is the one place that selection is implemented."""
    all_rows = document_content.all_rows_matching(lambda row: True)
    if DATASET_COLUMN not in all_rows.columns or RANDOM_DATASET_ORDER_COLUMN not in all_rows.columns:
        return []

    clusters: list[str] = []
    seen: set[str] = set()
    for value in all_rows[DATASET_COLUMN]:
        if not isinstance(value, str):
            continue
        parsed = parse_dataset_value(value)
        if parsed is None or parsed["type"] != dataset_type or parsed["cluster"] in seen:
            continue
        seen.add(parsed["cluster"])
        clusters.append(parsed["cluster"])

    entries: list[tuple[str, list[str]]] = []
    for cluster in clusters:
        rows = document_content.all_rows_matching(
            dataset_selection.random_mode_row_filter(dataset_type, cluster),
            sort_column=RANDOM_DATASET_ORDER_COLUMN,
        )
        if rows.empty:
            continue
        title = config_project.NAMING_BY_MODE["random"].name_template.format(
            type=dataset_type, cluster=cluster
        )
        entries.append((title, [row_text_value(v) for v in rows[TEXT_COLUMN]]))

    _write_txt_files(output_dir, entries)
    return entries


def _write_word_document(output_path: Path, entries: list[tuple[str, list[str]]]) -> None:
    """One .docx containing every (title, rows) entry, in order: the
    title as a heading, followed by each row as its own paragraph/line."""
    document = WordDocument()
    for title, row_texts in entries:
        document.add_heading(title, level=1)
        for row_text in row_texts:
            document.add_paragraph(row_text)
    document.save(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    word_dir = TXT_FILES / "word"
    word_dir.mkdir(parents=True, exist_ok=True)

    # Manually populated by the user for ad-hoc test runs -- nothing here
    # writes its content, only ensures the folder exists.
    test_dir = TXT_FILES / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # "full"/"full_without_validation" are written by the explicit calls
    # below instead (whole-document, not selection_mode-driven); "test" is
    # never written here at all (see test_dir above) -- only these two go
    # through the generic mode dispatch.
    for dataset_type in ("training", "validation"):
        run_type_config = RUN_TYPE_CONFIGS[dataset_type]
        output_dir = TXT_FILES / dataset_type
        output_dir.mkdir(parents=True, exist_ok=True)

        mode = run_type_config.selection_mode
        if mode == "document_driven":
            entries = _write_document_driven(dataset_type, output_dir)
        elif mode == "random":
            entries = _write_random(dataset_type, output_dir)
        else:
            raise ValueError(f"Unknown selection mode for {dataset_type!r}: {mode!r}")

        print(f"Wrote {len(entries)} document(s) to {output_dir} ({dataset_type}, mode={mode!r})")
        _write_word_document(word_dir / f"{dataset_type}.docx", entries)

    full_dir = TXT_FILES / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_entries = _write_whole_documents(full_dir, "full")
    print(f"Wrote {len(full_entries)} document(s) to {full_dir} (full)")
    _write_word_document(word_dir / "full.docx", full_entries)

    full_without_validation_dir = TXT_FILES / "full_without_validation"
    full_without_validation_dir.mkdir(parents=True, exist_ok=True)
    full_without_validation_entries = _write_whole_documents(
        full_without_validation_dir, "full_without_validation"
    )
    print(
        f"Wrote {len(full_without_validation_entries)} document(s) to "
        f"{full_without_validation_dir} (full_without_validation)"
    )
    _write_word_document(word_dir / "full_without_validation.docx", full_without_validation_entries)


if __name__ == "__main__":
    main()
