import openpyxl
import pandas as pd

from src.io.io_utils import read_json, read_json_safe, write_json, write_workbook_sheets


def test_write_json_then_read_json_round_trips(tmp_path):
    path = tmp_path / "nested" / "data.json"
    write_json({"a": 1, "b": "café"}, path)

    assert path.exists()
    assert read_json(path) == {"a": 1, "b": "café"}
    assert "café" in path.read_text(encoding="utf-8")


def test_read_json_raises_on_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    try:
        read_json(missing)
        raised = False
    except OSError:
        raised = True
    assert raised


def test_read_json_safe_returns_default_for_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    assert read_json_safe(missing, default={}) == {}
    assert read_json_safe(missing) is None


def test_read_json_safe_returns_default_for_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert read_json_safe(corrupt, default=[]) == []


def test_read_json_safe_reads_valid_file(tmp_path):
    path = tmp_path / "valid.json"
    write_json({"x": 1}, path)
    assert read_json_safe(path) == {"x": 1}


def test_write_workbook_sheets_creates_multiple_sheets(tmp_path):
    output_file = tmp_path / "nested" / "book.xlsx"
    write_workbook_sheets(
        {
            "overview": pd.DataFrame({"a": [1, 2]}),
            "detail": pd.DataFrame({"b": [3, 4]}),
        },
        output_file,
    )

    assert output_file.exists()
    sheets = pd.read_excel(output_file, sheet_name=None)
    assert list(sheets.keys()) == ["overview", "detail"]
    assert sheets["overview"]["a"].tolist() == [1, 2]
    assert sheets["detail"]["b"].tolist() == [3, 4]


def test_write_workbook_sheets_append_adds_sheet_to_existing_file(tmp_path):
    output_file = tmp_path / "book.xlsx"
    write_workbook_sheets({"first": pd.DataFrame({"a": [1]})}, output_file)

    write_workbook_sheets(
        {"second": pd.DataFrame({"b": [2]})},
        output_file,
        append_to_existing_file=True,
    )

    sheets = pd.read_excel(output_file, sheet_name=None)
    assert list(sheets.keys()) == ["first", "second"]


def test_write_workbook_sheets_header_notes_attach_comments_to_header_cells(tmp_path):
    output_file = tmp_path / "book.xlsx"
    write_workbook_sheets(
        {
            "with_notes": pd.DataFrame({"a": [1], "b": [2]}),
            "without_notes": pd.DataFrame({"c": [3]}),
        },
        output_file,
        header_notes={"with_notes": {"a": "Column a's meaning.", "z": "Unused column, ignored."}},
    )

    workbook = openpyxl.load_workbook(output_file)
    with_notes = workbook["with_notes"]
    assert with_notes["A1"].comment.text == "Column a's meaning."
    assert with_notes["B1"].comment is None

    without_notes = workbook["without_notes"]
    assert without_notes["A1"].comment is None


def test_write_workbook_sheets_replace_existing_sheet_overwrites_it(tmp_path):
    output_file = tmp_path / "book.xlsx"
    write_workbook_sheets(
        {"first": pd.DataFrame({"a": [1]}), "second": pd.DataFrame({"b": [2]})}, output_file
    )

    write_workbook_sheets(
        {"second": pd.DataFrame({"b": [99]})},
        output_file,
        append_to_existing_file=True,
        replace_existing_sheet=True,
    )

    sheets = pd.read_excel(output_file, sheet_name=None)
    assert list(sheets.keys()) == ["first", "second"]
    assert sheets["second"]["b"].tolist() == [99]
