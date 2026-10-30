from pathlib import Path

import pandas as pd

from src.io.io_utils import get_matching_files


def load_txt_files(document_paths: list[Path]) -> dict[str, str]:
    return {
        str(p.resolve()): p.read_text(encoding="utf-8").replace("\r\n", "\n")
        for p in document_paths
    }


def extract_quotes(xlsx_path: Path, document_paths: list[Path]) -> None:
    """Extracts quote text from source files and adds it to the Excel export.

    Skips only when "quote_text" already has at least one non-blank value --
    an existing-but-entirely-empty column (e.g. from a previous run whose
    document_paths didn't cover any of this file's Source Name documents, so
    every row fell back to None) is treated the same as a missing column and
    recomputed, rather than being permanently stuck empty."""
    df = pd.read_excel(xlsx_path)
    if "quote_text" in df.columns:
        if df["quote_text"].notna().any():
            return
        df = df.drop(columns=["quote_text"])

    txt_cache = load_txt_files(document_paths)
    # Map document name (stem) to resolved path for quick lookup
    doc_map = {p.stem: str(p.resolve()) for p in document_paths}

    quote_texts = []
    for _, row in df.iterrows():
        doc_name = str(row["Source Name"]).split("/")[-1].strip()
        resolved = doc_map.get(doc_name)

        if (
            resolved
            and resolved in txt_cache
            and pd.notna(row.get("quote_start"))
            and pd.notna(row.get("quote_end"))
        ):
            content = txt_cache[resolved]
            quote_texts.append(
                content[int(row["quote_start"]) : int(row["quote_end"]) + 1]
            )
        else:
            quote_texts.append(None)

    insert_pos = df.columns.get_loc("quote_end") + 1
    df.insert(insert_pos, "quote_text", quote_texts)
    df.to_excel(xlsx_path, index=False)


def extract_quotes_from_atlas_exports(
    human_coding_dir: Path, document_paths: list[Path]
) -> None:
    for xlsx_path in get_matching_files(human_coding_dir, "*.xlsx"):
        if not xlsx_path.name.startswith("~$"):
            extract_quotes(xlsx_path, document_paths)
