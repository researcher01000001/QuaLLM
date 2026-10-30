"""File-IO helpers"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.comments import Comment


def resolve_output_dir(path: Path) -> Path:
    """Returns the directory for a path (file → parent, folder → itself) and creates it."""
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_matching_files(
    source_paths: Path | list[Path], glob_pattern: str, recursive: bool = False
) -> list[Path]:
    """Helper to resolve all files based on the given paths and patterns.

    Args:
        source_paths: Path or list of Paths to the source file(s) or directory containing them.
        glob_pattern: The glob pattern to match inside directories (e.g. "*.xlsx").
        recursive: If True, uses rglob instead of glob.
    """
    if isinstance(source_paths, Path):
        source_paths = [source_paths]

    files = []
    for path in source_paths:
        if path.is_dir():
            if recursive:
                files.extend(path.rglob(glob_pattern))
            else:
                files.extend(path.glob(glob_pattern))
        elif path.is_file():
            files.append(path)
    return files


def read_json(path: Path) -> Any:
    """Read and parse a JSON file. Raises if it's missing or malformed --
    use read_json_safe() when a missing/corrupt file should fall back to a
    default instead."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_safe(path: Path, *, default: Any = None) -> Any:
    """Read and parse a JSON file, returning default if it's missing or
    malformed instead of raising."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(data: Any, path: Path, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Write data as JSON to path, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def write_workbook_sheets(
    sheets: Mapping[str, pd.DataFrame],
    output_file: Path,
    *,
    append_to_existing_file: bool = False,
    replace_existing_sheet: bool = False,
    header_notes: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Write each named DataFrame to its own sheet in output_file.

    By default creates/overwrites output_file. append_to_existing_file=True
    opens an existing workbook and adds to it instead of replacing it;
    replace_existing_sheet=True (only meaningful together with
    append_to_existing_file) overwrites a same-named sheet instead of
    erroring.

    header_notes, when given, attaches an Excel cell comment explaining a
    column's meaning to that column's header cell: {sheet_name: {column_name:
    note_text}}. A sheet or column missing from header_notes simply gets no
    comment -- this is opt-in per column, not required for every sheet.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append_to_existing_file else "w"
    if_sheet_exists = "replace" if (append_to_existing_file and replace_existing_sheet) else None
    with pd.ExcelWriter(
        output_file, engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists
    ) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            notes = (header_notes or {}).get(sheet_name)
            if not notes:
                continue
            worksheet = writer.sheets[sheet_name]
            for column_index, column_name in enumerate(df.columns, start=1):
                note_text = notes.get(column_name)
                if note_text:
                    worksheet.cell(row=1, column=column_index).comment = Comment(
                        note_text, "QuaLLM"
                    )
