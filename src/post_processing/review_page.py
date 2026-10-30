"""Logic of Step 2 for an editable manual-review page"""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2

from src.coding.consensus import build_consensus_records, row_for_quote
from src.config import project as config_project
from src.core import process_log
from src.core.models import Document, Quote, Tag
from src.dataset import document_identity
from src.io.io_utils import resolve_output_dir

REVIEW_DATA_SCRIPT_ID = "review-data"
_REVIEW_DATA_PATTERN = re.compile(
    r'<script type="application/json" id="review-data">(.*?)</script>',
    re.DOTALL,
)


def explode_consensus_records(
    documents: Mapping[str, Document],
    quotes_by_doc: Mapping[str, Sequence[Quote]],
) -> list[dict]:
    """Explode consensus quotes into one editable row per (span, tag).

    Mirrors ``build_consensus_records()``'s span/paragraph resolution, but
    never merges co-occurring tags into a single row: a ``group_id`` ties
    rows that started on the same span, while each row tracks its own
    ``id``/``status`` so it can be independently retagged, moved, or deleted
    in the manual-review HTML. Rows whose position could not be resolved
    (``start``/``end`` is ``None``) are skipped, since they can't be
    rendered or edited.
    """
    exploded: list[dict] = []
    for record in build_consensus_records(documents, quotes_by_doc):
        start, end = record["start"], record["end"]
        if start is None or end is None:
            continue
        source_name = record["source"]["source_name"]
        group_id = f"{source_name}::{start}::{end}"
        for tag_name in record["tags"]:
            exploded.append(
                {
                    "id": f"{group_id}::{tag_name}",
                    "group_id": group_id,
                    "source_name": source_name,
                    "row": record["source"]["row"],
                    "phrase": record["phrase"],
                    "tag": tag_name,
                    "start": start,
                    "end": end,
                    "status": "original",
                }
            )
    return exploded


def build_paragraph_checks(documents: Mapping[str, Document]) -> list[dict]:
    """One reviewed/excluded-flag row per paragraph across all documents,
    seeded False. "excluded" is a second, independent marker (see
    ``build_excluded_paragraph_keys()``) -- checking it on the manual-review
    page also counts as reviewed (see ``count_reviewed_paragraphs()``) and
    drops the paragraph from Step 3's exported output.xlsx (see
    ``tag_export.append_ordered_tags_to_xlsx()``)."""
    return [
        {
            "source_name": doc_name,
            "row": row.index,
            "checked": False,
            "excluded": False,
        }
        for doc_name, document in documents.items()
        for row in document.rows
    ]


def build_paragraph_meta(
    documents: Mapping[str, Document],
    para_meta_by_position: Mapping[str, Mapping[int, dict[str, str]]],
) -> dict[str, dict[int, dict[str, str]]]:
    """Re-key per-paragraph output.xlsx metadata (performance, truth, etc.)
    from 1-based position (as returned by ``src.dataset.document_content.paragraph_meta()``)
    to 0-based ``row.index`` (as used by ``Document.rows``), for every row of
    every loaded document.
    """
    result: dict[str, dict[int, dict[str, str]]] = {}
    for doc_name, document in documents.items():
        by_position = para_meta_by_position.get(doc_name)
        if not by_position:
            continue
        per_paragraph = {}
        for row in document.rows:
            try:
                position = int(document.display_value(row.index))
            except ValueError:
                continue
            if position in by_position:
                per_paragraph[row.index] = by_position[position]
        if per_paragraph:
            result[doc_name] = per_paragraph
    return result


def build_document_identity(documents: Mapping[str, Document]) -> dict[str, dict[str, Any]]:
    """Resolve each document's identity + condition for the manual-review
    page's filter panels, keyed the same way as ``document_texts`` (the
    ``documents`` dict key). Documents whose identity can't be resolved are
    omitted, since they have nothing to filter by.
    """
    result: dict[str, dict[str, Any]] = {}
    for doc_name, document in documents.items():
        identity = document_identity.resolve_document_identity(document.source_name)
        if identity is None:
            process_log.current().log_document_key_match_failure(
                severity="Warning",
                source_name=document.source_name,
                call_site="review_page.build_document_identity",
                message=(
                    f"Document name '{document.source_name}' doesn't match "
                    "DOCUMENT_NAME_TEMPLATE's shape; omitted from the filter panels."
                ),
            )
            continue
        result[doc_name] = identity
    return result


def build_review_metadata(source_file: Path, prefix: str) -> dict:
    """Build the metadata block embedded alongside a fresh set of review records."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "last_saved_at": None,
        "source_file": source_file.name,
        "protect_file": False,
        "render_undefined_tags": True,
        "show_system_tags": False,
        "prefix": prefix,
    }


def collect_review_tags(
    base_tags: Sequence[Tag],
    quotes_by_doc: Mapping[str, Sequence[Quote]],
) -> list[Tag]:
    """Combine the codebook tag list with any additional (undefined) tags
    actually used in ``quotes_by_doc``, so the manual-review page can color
    and flag every tag it might encounter as defined/undefined, not just the
    ones in the codebook.
    """
    by_name = {tag.name: tag for tag in base_tags}
    for quotes in quotes_by_doc.values():
        for quote in quotes:
            for tag in quote.tags:
                if tag.name not in by_name:
                    by_name[tag.name] = tag
    return list(by_name.values())


def render_manual_review_html(
    documents: Mapping[str, Document],
    records: list[dict],
    paragraph_checks: list[dict],
    tags: list[Tag],
    metadata: dict,
    output_file: Path,
    paragraph_meta: Mapping[str, Mapping[int, dict[str, str]]] | None = None,
    meta_columns: Sequence[tuple[str, str]] = (),
    document_identity: Mapping[str, dict[str, Any]] | None = None,
    filter_columns: Sequence[tuple[str, str]] = (),
    mode: str = "full",
) -> None:
    """Render the self-contained, editable manual-review HTML page.

    The page embeds ``records``/``paragraph_checks``/``metadata`` (the
    editable dataset), the source documents' paragraph text/offsets plus any
    ``paragraph_meta`` (output.xlsx fields such as ``performance``, shown via
    a toggleable dropdown mirroring compare.html's meta-selector), the tag
    list (name + color, for the retag dropdown and highlight colors), and
    each document's ``document_identity`` (see ``build_document_identity()``)
    -- one filter dropdown per ``filter_columns`` entry is built client-side
    from this data, so the page never re-derives identity from a document
    name itself. No external files are referenced; the reviewer opens and
    edits it, then saves changes back into the same file from the browser.

    ``mode`` is either ``"full"`` (the default -- every record freely
    editable, "+ Add tag" available) or ``"restricted"`` (the Step 1
    undefined-tags review page: "+ Add tag" disabled outright, and a
    client-side "Restrict to undefined tags" toggle -- default on -- locks
    editing to records whose ``is_target`` field is true). ``records`` in
    restricted mode are expected to already carry an ``is_target`` key (see
    ``undefined_tags.build_step1_target_records()``); it is
    otherwise unused.
    """
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template("manual_review.html")

    paragraph_meta = paragraph_meta or {}
    document_texts = {
        doc_name: {
            "paragraphs": [
                {
                    "row": row.index,
                    "paragraph": document.display_value(row.index),
                    "text": row.text,
                    "start_position": row.start_position,
                    "end_position": row.end_position,
                    "meta": paragraph_meta.get(doc_name, {}).get(row.index, {}),
                }
                for row in document.rows
            ]
        }
        for doc_name, document in documents.items()
    }
    tag_list = [
        {
            "name": tag.name,
            "color": tag.color or "#808080",
            "additional_tag": bool(tag.additional_tag),
            "system": bool(tag.system),
        }
        for tag in tags
    ]
    review_data = {
        "metadata": {**metadata, "suggested_filename": output_file.name},
        "paragraph_checks": paragraph_checks,
        "records": records,
    }
    meta_column_list = [{"col": col, "label": label} for col, label in meta_columns]
    filter_column_list = [{"col": col, "label": label} for col, label in filter_columns]

    html = template.render(
        prefix=metadata.get("prefix", ""),
        review_data_json=_embed_json(review_data),
        document_texts_json=_embed_json(document_texts),
        tag_list_json=_embed_json(tag_list),
        meta_columns_json=_embed_json(meta_column_list),
        document_identity_json=_embed_json(dict(document_identity or {})),
        filter_columns_json=_embed_json(filter_column_list),
        display_label=config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        display_label_json=_embed_json(config_project.DOCUMENT_ORDER_COLUMN_LABEL),
        mode=mode,
        mode_json=_embed_json(mode),
    )

    resolve_output_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def _embed_json(obj: object) -> str:
    """Serialize ``obj`` for embedding inside a ``<script>`` tag.

    Escapes "</" so a literal "</script>" occurring inside any string value
    (e.g. quoted document text) can't prematurely close the script block.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False).replace("</", "<\\/")


def extract_review_data(html_file: Path) -> dict | None:
    """Pull the embedded review-data JSON back out of a saved manual-review HTML.

    Returns ``None`` if the file doesn't exist or the expected script tag
    can't be found/parsed (e.g. a file that isn't one of our generated pages).
    """
    if not html_file.exists():
        return None

    html_text = html_file.read_text(encoding="utf-8")
    match = _REVIEW_DATA_PATTERN.search(html_text)
    if match is None:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def read_review_protect_file_flag(html_file: Path) -> bool:
    """Read the ``protect_file`` flag out of an existing manual-review HTML.

    Returns False if the file doesn't exist or the flag can't be found,
    meaning: regenerate freely.

    Shared by both the Step 1 (restricted) and Step 2 (full) manual-review
    pages -- both embed ``protect_file`` the same way, in their ``metadata``.
    """
    review_data = extract_review_data(html_file)
    if review_data is None:
        return False
    return bool(review_data.get("metadata", {}).get("protect_file", False))


def count_reviewed_paragraphs(paragraph_checks: Sequence[dict]) -> tuple[int, int]:
    """Return ``(checked_count, total_count)`` from a review page's
    ``paragraph_checks`` list, for reporting how much of the manual review
    is actually done before Step 3 exports it. A paragraph marked "excluded"
    counts as reviewed even if its own "checked" flag was never set."""
    total = len(paragraph_checks)
    checked = sum(1 for p in paragraph_checks if p.get("checked") or p.get("excluded"))
    return checked, total


def build_excluded_paragraph_keys(
    paragraph_checks: Sequence[dict],
) -> frozenset[tuple[str, int]]:
    """``(source_name, row)`` pairs marked "excluded" on the manual-review
    page -- paragraphs dropped entirely from Step 3's exported output.xlsx
    (see ``tag_export.build_ordered_tags()``/``tag_export.append_ordered_tags_to_xlsx()``)."""
    return frozenset(
        (p["source_name"], p["row"]) for p in paragraph_checks if p.get("excluded")
    )


def find_still_undefined_tags(
    documents: Mapping[str, Document],
    quotes_by_doc: Mapping[str, Sequence[Quote]],
) -> list[dict]:
    """Find quote tags still ``Tag.additional_tag=True`` after manual rework
    (Step 2's corrected consensus). Same test ``collect_undefined_tag_names()``
    (``src/post_processing/review_database.py``) applies to raw AI output, applied here
    post-rework instead, at per-occurrence (not just per-name) granularity.

    Logs each occurrence to the process log (``step=STEP_MANUAL_REWORK_REVIEW``)
    and also returns the rows, for testability.
    """
    rows: list[dict] = []
    for doc_name, quotes in quotes_by_doc.items():
        document = documents.get(doc_name)
        for quote in quotes:
            row_index = row_for_quote(document, quote)
            for tag in quote.tags:
                if not tag.additional_tag:
                    continue
                row = {
                    "source_name": doc_name,
                    "row": row_index,
                    "tag_name": tag.name,
                    "span_start": quote.start,
                    "span_end": quote.end,
                }
                rows.append(row)
                process_log.current().log_rework_still_undefined(
                    severity="Warning",
                    source_name=doc_name,
                    row=row_index,
                    tag_name=tag.name,
                    span_start=quote.start,
                    span_end=quote.end,
                    message=(
                        f"Tag '{tag.name}' on the quote at [{quote.start}:{quote.end}) in "
                        f"'{doc_name}' is still undefined after manual rework."
                    ),
                    step=process_log.STEP_MANUAL_REWORK_REVIEW,
                )
    return rows
