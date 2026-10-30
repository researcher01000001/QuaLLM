import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.models import ConsensusQuote, DisplayColumn, Document, Row, Tag
from src.post_processing.review_page import (
    build_excluded_paragraph_keys,
    build_paragraph_checks,
    count_reviewed_paragraphs,
    explode_consensus_records,
    extract_review_data,
    read_review_protect_file_flag,
    render_manual_review_html,
)


def _document(source_name: str = "45_no advice_cycle_63") -> Document:
    return Document(
        source_name=source_name,
        rows=[
            Row(index=0, text="x" * 100, start_position=0, end_position=100),
            Row(index=1, text="y" * 100, start_position=101, end_position=201),
        ],
        display_column=DisplayColumn(
            source_column="position", label="Paragraph", values={0: "1", 1: "2"}
        ),
    )


@pytest.mark.parametrize("additional", [False, True], ids=["defined", "undefined"])
@pytest.mark.parametrize("locked", [False, True], ids=["editable", "locked"])
def test_badge_click_highlights_current_spans_after_editor_render(additional, locked):
    """Run the template's click handler with a DOM replaced on every render."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute review-page JavaScript")
    template = (Path(__file__).resolve().parents[1] / "src/templates/manual_review.html").read_text()
    start = template.index("  function flashRecordSpans(")
    end = template.index("  function renderEditPanel(", start)
    handlers = template[start:end]
    harness = r"""
const assert = require('node:assert/strict');
const record = {id: 'test-record', tag: 'Test tag', status: 'original'};
let openPanelId = null;
let renderCount = 0;
let toastCount = 0;
const timers = [];
const STATUS_LABELS = {original: 'original'};
const displayLabel = 'Paragraph';
const tagInfo = () => ({additional: ADDITIONAL, system: false});
const tagColor = () => '#808080';
const recordLocked = () => LOCKED;
const showToast = () => toastCount++;
const setTimeout = (callback) => timers.push(callback);
const document = {
  createTextNode: (text) => text,
  createElement: () => ({
    dataset: {}, style: {}, handlers: {},
    appendChild() {},
    addEventListener(type, callback) { this.handlers[type] = callback; },
  }),
};
function makeMarks() {
  return ['test-record', 'other,test-record', 'test-record-suffix', 'other'].map((ids) => ({
    dataset: {ids}, scrolls: 0,
    classList: new Set(),
    scrollIntoView() { this.scrolls++; },
  })).map((mark) => {
    mark.classList.remove = mark.classList.delete.bind(mark.classList);
    return mark;
  });
}
let marks = makeMarks();
const docsEl = {querySelectorAll: () => marks};
function render() { renderCount++; marks = makeMarks(); }
for (let click = 0; click < 2; click++) {
  const previousMarks = marks;
  const badge = renderBadge(record, LOCKED, false);
  badge.handlers.click();
  assert.equal(renderCount, LOCKED ? 0 : click + 1);
  assert.equal(openPanelId, LOCKED || click === 1 ? null : record.id);
  if (!LOCKED) {
    assert.notEqual(marks, previousMarks);
    assert(previousMarks.every((m) => !m.classList.has('flash')));
  }
  assert.deepEqual(marks.map((m) => m.classList.has('flash')), [true, true, false, false]);
  assert.equal(marks[0].scrolls, LOCKED ? click + 1 : 1);
  assert(marks.slice(1).every((m) => m.scrolls === 0));
  timers.splice(0).forEach((callback) => callback());
  assert(marks.every((m) => !m.classList.has('flash')));
}
assert.equal(toastCount, LOCKED ? 2 : 0);
"""
    script = (
        f"const ADDITIONAL = {json.dumps(additional)}; const LOCKED = {json.dumps(locked)};\n"
        + handlers
        + harness
    )
    result = subprocess.run([node, "-"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_explode_consensus_records_one_row_per_span_and_tag():
    tag_a = Tag(name="TagA")
    tag_b = Tag(name="TagB")
    quote = ConsensusQuote(text="span", tags=[tag_a, tag_b], start=10, end=20)
    documents = {"45_no advice_cycle_63": _document()}
    quotes_by_doc = {"45_no advice_cycle_63": [quote]}

    records = explode_consensus_records(documents, quotes_by_doc)

    assert len(records) == 2
    assert {r["tag"] for r in records} == {"TagA", "TagB"}
    assert records[0]["group_id"] == records[1]["group_id"]
    assert records[0]["id"] != records[1]["id"]
    for r in records:
        assert r["source_name"] == "45_no advice_cycle_63"
        assert r["start"] == 10
        assert r["end"] == 20
        assert r["status"] == "original"


def test_explode_consensus_records_skips_unresolved_position():
    tag_a = Tag(name="TagA")
    quote = ConsensusQuote(text="span", tags=[tag_a], start=None, end=None)
    documents = {"45_no advice_cycle_63": _document()}
    quotes_by_doc = {"45_no advice_cycle_63": [quote]}

    records = explode_consensus_records(documents, quotes_by_doc)

    assert records == []


def test_build_paragraph_checks_one_per_paragraph_seeded_false():
    documents = {"45_no advice_cycle_63": _document()}

    checks = build_paragraph_checks(documents)

    assert len(checks) == 2
    assert all(c["checked"] is False for c in checks)
    assert all(c["excluded"] is False for c in checks)
    assert {c["row"] for c in checks} == {0, 1}
    assert all(c["source_name"] == "45_no advice_cycle_63" for c in checks)


def test_render_manual_review_html_embeds_row_and_converted_paragraph(tmp_path):
    """Regression test for the bug where manual_review.html displayed the raw
    0-based row as the paragraph number (e.g. "¶ 0" for the first
    paragraph). document_texts must carry both the technical row and the
    converted, 1-based paragraph number."""
    documents = {"45_no advice_cycle_63": _document()}
    output_file = tmp_path / "manual_review_test.html"

    render_manual_review_html(
        documents=documents,
        records=[],
        paragraph_checks=[],
        tags=[],
        metadata={"prefix": "test"},
        output_file=output_file,
    )

    html = output_file.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="document-texts">\s*(.*?)\s*</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    document_texts = json.loads(match.group(1))
    paragraphs = document_texts["45_no advice_cycle_63"]["paragraphs"]
    assert [p["row"] for p in paragraphs] == [0, 1]
    assert [p["paragraph"] for p in paragraphs] == ["1", "2"]


def test_render_manual_review_html_uses_output_filename_for_save(tmp_path):
    output_file = tmp_path / "review_interface_undefined_tags_test.html"

    render_manual_review_html(
        documents={"45_no advice_cycle_63": _document()},
        records=[],
        paragraph_checks=[],
        tags=[],
        metadata={"prefix": "test"},
        output_file=output_file,
    )

    html = output_file.read_text(encoding="utf-8")
    match = re.search(
        r'<script type="application/json" id="review-data">\s*(.*?)\s*</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    review_data = json.loads(match.group(1))
    assert review_data["metadata"]["suggested_filename"] == output_file.name
    assert "state.metadata.suggested_filename ||" in html


def _write_review_html(tmp_path: Path, review_data: dict) -> Path:
    import json

    html_file = tmp_path / "manual_review_test.html"
    payload = json.dumps(review_data, indent=2, ensure_ascii=False)
    html_file.write_text(
        "<!doctype html><html><body>\n"
        f'<script type="application/json" id="review-data">\n{payload}\n</script>\n'
        "</body></html>",
        encoding="utf-8",
    )
    return html_file


def test_extract_review_data_round_trips(tmp_path: Path):
    review_data = {
        "metadata": {"protect_file": False, "prefix": "test"},
        "paragraph_checks": [],
        "records": [{"id": "a", "tag": "TagA", "status": "original"}],
    }
    html_file = _write_review_html(tmp_path, review_data)

    result = extract_review_data(html_file)

    assert result == review_data


def test_extract_review_data_missing_file_returns_none(tmp_path: Path):
    assert extract_review_data(tmp_path / "does_not_exist.html") is None


def test_extract_review_data_no_script_tag_returns_none(tmp_path: Path):
    html_file = tmp_path / "plain.html"
    html_file.write_text("<html><body>no data here</body></html>", encoding="utf-8")

    assert extract_review_data(html_file) is None


def test_read_review_protect_file_flag_true(tmp_path: Path):
    html_file = _write_review_html(
        tmp_path,
        {"metadata": {"protect_file": True}, "paragraph_checks": [], "records": []},
    )

    assert read_review_protect_file_flag(html_file) is True


def test_read_review_protect_file_flag_false_when_missing(tmp_path: Path):
    assert read_review_protect_file_flag(tmp_path / "missing.html") is False


def test_count_reviewed_paragraphs_partial():
    checks = [
        {"source_name": "d", "row": 0, "checked": True},
        {"source_name": "d", "row": 1, "checked": False},
        {"source_name": "d", "row": 2, "checked": True},
    ]

    assert count_reviewed_paragraphs(checks) == (2, 3)


def test_count_reviewed_paragraphs_empty():
    assert count_reviewed_paragraphs([]) == (0, 0)


def test_count_reviewed_paragraphs_all_checked():
    checks = [{"source_name": "d", "row": i, "checked": True} for i in range(4)]

    assert count_reviewed_paragraphs(checks) == (4, 4)


def test_count_reviewed_paragraphs_excluded_counts_as_reviewed():
    checks = [
        {"source_name": "d", "row": 0, "checked": False, "excluded": True},
        {"source_name": "d", "row": 1, "checked": False, "excluded": False},
    ]

    assert count_reviewed_paragraphs(checks) == (1, 2)


def test_build_excluded_paragraph_keys_only_includes_excluded():
    checks = [
        {"source_name": "d", "row": 0, "checked": True, "excluded": False},
        {"source_name": "d", "row": 1, "checked": False, "excluded": True},
        {"source_name": "e", "row": 0, "checked": True, "excluded": True},
    ]

    assert build_excluded_paragraph_keys(checks) == frozenset(
        {("d", 1), ("e", 0)}
    )


def test_build_excluded_paragraph_keys_empty_when_none_excluded():
    checks = [{"source_name": "d", "row": 0, "checked": True, "excluded": False}]

    assert build_excluded_paragraph_keys(checks) == frozenset()
