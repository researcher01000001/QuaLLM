from pathlib import Path

import pandas as pd

from src.analysis.tag_analysis import span_overlap_ratio
from src.config import project as config_project
from src.core.models import Tag
from src.io.io_utils import write_workbook_sheets

DATA_SHEET_NAME = "data"

# Reserved ``new_tag`` value marking a span as deliberately deleted (the
# reviewer decided no tag should apply here at all) rather than left
# undecided (empty ``new_tag``) or reassigned to a real tag name. Recognized
# by the database matching functions below and by
# ``undefined_tags.apply_database_prefill_to_records()``, which
# turns a match on this value into a pre-marked ``status="deleted"`` record
# instead of a tag name.
DELETED_TAG_VALUE = "<deleted>"


def build_post_process_columns() -> list[str]:
    """Column order for the post-processing extraction/database workbooks:
    one column per document-identity key column (see
    ``document_identity.get_document_key()``), plus ``config_project.GENERAL_CLUSTER_COLUMN``
    if it isn't already one of them, then the quote-specific columns."""
    columns = list(config_project.DOCUMENT_KEY_COLUMNS)
    if config_project.GENERAL_CLUSTER_COLUMN not in columns:
        columns.append(config_project.GENERAL_CLUSTER_COLUMN)
    return [
        *columns,
        config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        "span",
        "start",
        "end",
        "ai_tag",
        "new_tag",
    ]


def collect_undefined_tag_names(
    ai_quote_runs: dict[tuple[str, int], dict[str, list]],
) -> list[str]:
    """Return distinct names of AI-generated tags that are not part of the coding schema.

    These are tags loaded with ``Tag.additional_tag = True`` (see
    ``AIQuoteLoader._get_or_create_undefined_tag``), i.e. tag names the AI
    produced that do not match any tag in the codebook. Only ``.values()`` is
    ever read, so a caller with a plain ``{doc_name: [quotes]}`` mapping
    (e.g. a consensus, not per-configuration/iteration AI runs) can wrap it
    as ``{"_": quotes_by_doc}`` and pass that instead.
    """
    names: list[str] = []
    seen: set[str] = set()

    for quotes_by_doc in ai_quote_runs.values():
        for quotes in quotes_by_doc.values():
            for quote in quotes:
                for tag in quote.tags:
                    if tag.additional_tag and tag.name not in seen:
                        seen.add(tag.name)
                        names.append(tag.name)

    return names


def build_target_tags(
    post_process_tags: list[str],
    ai_quote_runs: dict[tuple[str, int], dict[str, list]],
    include_undefined_tags: bool,
) -> list[str]:
    """Combine the configured tag list with auto-detected undefined tags."""
    target_tags = list(post_process_tags)

    if include_undefined_tags:
        for tag_name in collect_undefined_tag_names(ai_quote_runs):
            if tag_name not in target_tags:
                target_tags.append(tag_name)

    return target_tags


def is_kept_row(row: dict) -> bool:
    """True for a database row recording that the reviewer looked at a record
    and kept its AI tag (``new_tag`` equals ``ai_tag``). Such a row exists so
    that a sibling record on a double-coded span is not overwritten by a
    correction made to the other record on that span."""
    ai_tag = str(row.get("ai_tag", "")).strip()
    return bool(ai_tag) and str(row.get("new_tag", "")).strip() == ai_tag


def manually_defined_rows(rows: list[dict]) -> list[dict]:
    """Return the rows that carry a non-empty ``new_tag`` value."""
    return [row for row in rows if str(row.get("new_tag", "")).strip()]


def read_post_process_rows(file: Path) -> list[dict]:
    """Read the data sheet of a post-processing/database workbook.

    Returns an empty list if the file does not exist.
    """
    if not file.exists():
        return []
    df = pd.read_excel(file, sheet_name=DATA_SHEET_NAME)
    df = df.where(pd.notna(df), "")
    return df.to_dict(orient="records")


def _same_document(row_a: dict, row_b: dict) -> bool:
    return all(
        row_a.get(col) == row_b.get(col) for col in config_project.DOCUMENT_KEY_COLUMNS
    ) and row_a.get(config_project.GENERAL_CLUSTER_COLUMN) == row_b.get(
        config_project.GENERAL_CLUSTER_COLUMN
    )


def spans_overlap_at_least(
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
    overlap_threshold: float,
) -> bool:
    """True if both spans' intersection-over-union ratio is at least
    ``overlap_threshold`` (see ``tag_analysis.span_overlap_ratio()`` -- the
    same method the pipeline's validation matching uses). False if either
    span is missing an offset."""
    return span_overlap_ratio(a_start, a_end, b_start, b_end) >= overlap_threshold


def _same_span(row_a: dict, row_b: dict, overlap_threshold: float) -> bool:
    """True if both rows describe the same document and their spans overlap
    by at least ``overlap_threshold`` (intersection-over-union)."""
    if not _same_document(row_a, row_b):
        return False

    return spans_overlap_at_least(
        row_a.get("start"), row_a.get("end"), row_b.get("start"), row_b.get("end"), overlap_threshold
    )


def _allows_double_coding(tag_name: str, tags_by_name: dict[str, Tag]) -> bool:
    """True if the codebook marks ``tag_name`` as allowed to co-occur with other
    tags on the same span (``double_coding`` == "yes" in the description file).
    Unknown tag names default to not allowing double coding."""
    tag = tags_by_name.get(tag_name)
    return bool(tag and tag.can_co_occur)


def _tags_compatible(tag_a: str, tag_b: str, tags_by_name: dict[str, Tag]) -> bool:
    """True if ``tag_a`` and ``tag_b`` may both apply to the same span.

    ``double_coding`` on a tag means *that tag* may not be paired with another
    tag that also disallows double coding; a tag with ``double_coding = yes``
    may be paired with anything. So two tags conflict only when *both* of
    them individually disallow double coding.
    """
    return _allows_double_coding(tag_a, tags_by_name) or _allows_double_coding(
        tag_b, tags_by_name
    )


def append_to_database(
    rows: list[dict],
    database_file: Path,
    tags_by_name: dict[str, Tag],
    overlap_threshold: float,
) -> int:
    """Merge manually-defined rows into the per-prefix database.

    A candidate row is compared against existing database rows for the same
    span (document identity plus an intersection-over-union overlap ratio of
    at least ``overlap_threshold``):

    - A match with the exact same ``new_tag`` means the candidate is already
      saved; it is skipped.
    - A match with a *different* ``new_tag`` is kept alongside the candidate
      when the two tags are compatible (``_tags_compatible()``: at least one
      of them allows double coding); otherwise the candidate replaces it, so
      a single database never ends up with two tags that both disallow
      double coding on the same span. ``new_tag`` may also be
      ``DELETED_TAG_VALUE``, meaning "no tag applies here" rather than a real
      tag name -- it is treated like any other value for matching/replacement
      purposes.

    Args:
        rows: Candidate rows (only those with a non-empty ``new_tag`` are used).
        database_file: Path to the per-prefix database workbook.
        tags_by_name: Codebook tags keyed by name, for ``can_co_occur`` lookups.
        overlap_threshold: Minimum intersection-over-union ratio to treat two
            spans as the same (see ``POST_PROCESS_SPAN_OVERLAP_THRESHOLD``).

    Rows whose ``new_tag`` equals their ``ai_tag`` (see ``is_kept_row()``) record
    "reviewed and kept". They are stored only when the database has no decision
    yet for that span and AI tag, are replaced by a later real decision for the
    same AI tag, and never conflict with other rows.

    Returns:
        Number of corrections (retags or deletions) added or replaced in the
        database; kept rows are not counted.
    """
    manual_rows = manually_defined_rows(rows)
    if not manual_rows:
        return 0

    database_rows = read_post_process_rows(database_file)
    changed = 0
    wrote_kept_rows = False

    corrections = [row for row in manual_rows if not is_kept_row(row)]
    kept_rows = [row for row in manual_rows if is_kept_row(row)]

    for candidate in corrections:
        matches = [
            db_row for db_row in database_rows if _same_span(db_row, candidate, overlap_threshold)
        ]
        # A kept row is superseded by a real decision for the same AI tag and
        # otherwise never takes part in the conflict logic below.
        superseded_ids = {
            id(match)
            for match in matches
            if is_kept_row(match) and match.get("ai_tag") == candidate.get("ai_tag")
        }
        matches = [match for match in matches if not is_kept_row(match)]

        if any(match["new_tag"] == candidate["new_tag"] for match in matches):
            if superseded_ids:
                database_rows = [row for row in database_rows if id(row) not in superseded_ids]
                wrote_kept_rows = True
            continue  # already saved

        candidate_is_deleted = candidate["new_tag"] == DELETED_TAG_VALUE
        conflicting_ids = {
            id(match)
            for match in matches
            if candidate_is_deleted
            or match["new_tag"] == DELETED_TAG_VALUE
            or not _tags_compatible(candidate["new_tag"], match["new_tag"], tags_by_name)
        }
        removed_ids = conflicting_ids | superseded_ids
        if removed_ids:
            database_rows = [row for row in database_rows if id(row) not in removed_ids]

        database_rows.append(candidate)
        changed += 1

    for candidate in kept_rows:
        already_decided = any(
            _same_span(db_row, candidate, overlap_threshold)
            and db_row.get("ai_tag") == candidate.get("ai_tag")
            for db_row in database_rows
        )
        if not already_decided:
            database_rows.append(candidate)
            wrote_kept_rows = True

    if changed or wrote_kept_rows:
        write_data_table(database_rows, database_file)
    return changed


def rebuild_database(
    rows: list[dict],
    database_file: Path,
    tags_by_name: dict[str, Tag],
    overlap_threshold: float,
) -> int:
    """Discard the existing per-prefix database and rebuild it from ``rows``.

    Deletes ``database_file`` if it exists, then re-populates it via
    ``append_to_database()`` exactly as if no database file had existed yet.

    Returns the number of rows written to the freshly rebuilt database.
    """
    if database_file.exists():
        database_file.unlink()
    return append_to_database(rows, database_file, tags_by_name, overlap_threshold)


def resolve_database_files(
    results_dir: Path,
    current_database_file: Path,
    use_all_prefixes: bool,
) -> list[Path]:
    """Resolve which per-prefix database file(s) to read for pre-filling new_tag values.

    Database files are expected at
    ``results_dir / {prefix} / {run_type} / {post_process_subdir} / {step1_subdir} / database_{prefix}.xlsx``
    (see ``POST_PROCESS_DATABASE_FILE`` -- ``POST_PROCESS_DIR`` is
    ``DATA_RESULTS_DIR / PREFIX / RUN_TYPE / POST_PROCESS_SUBDIR``), where
    ``{post_process_subdir}``/``{step1_subdir}`` are taken from
    ``current_database_file``'s own path so this doesn't need to hard-code
    either folder name. use_all_prefixes scans across every prefix *and*
    every run type under it. Only paths that actually exist on
    disk are returned, so a caller's "N database file(s)" count reflects
    what was actually found, not paths merely considered.
    """
    if not use_all_prefixes:
        return [current_database_file] if current_database_file.exists() else []
    step1_subdir = current_database_file.parent.name
    post_process_subdir = current_database_file.parents[1].name
    return sorted(
        results_dir.glob(f"*/*/{post_process_subdir}/{step1_subdir}/database_*.xlsx")
    )


def load_database_rows(database_files: list[Path]) -> list[dict]:
    """Read and concatenate rows from one or more database workbooks.

    Each row is tagged with an internal ``_source_file`` key (not part of
    ``build_post_process_columns()``, so it is dropped again on write)
    recording which file it came from, so conflicting matches across files
    can be resolved by the file's save date.
    """
    rows: list[dict] = []
    for file in database_files:
        for row in read_post_process_rows(file):
            rows.append({**row, "_source_file": file})
    return rows


def prefill_new_tags_from_database(
    rows: list[dict],
    database_rows: list[dict],
    tags_by_name: dict[str, Tag],
    overlap_threshold: float,
) -> list[dict]:
    """Pre-fill ``new_tag`` for rows matching one or more database entries.

    A row matches a database entry when both belong to the same document
    (``config_project.DOCUMENT_KEY_COLUMNS`` plus ``condition``) and their
    spans overlap by at least ``overlap_threshold`` (intersection-over-union).
    ``ai_tag`` is not required to match, so a correction is reused even if a
    later AI run or model labelled the same phrase with a different
    placeholder tag. When the database holds a decision for the row's own
    ``ai_tag`` on that span, only those decisions are used, so the records of a
    double-coded span are prefilled independently.

    When several distinct ``new_tag`` values match the same row (e.g. from
    different prefix databases), all of them that are pairwise compatible
    (``_tags_compatible()``: a tag that allows double coding may be paired
    with anything) are combined into a comma-separated, alphabetically
    sorted set. Tags that individually disallow double coding conflict with
    each other, so at most one of them can be kept; among those, the match
    from the most recently saved database file wins (see
    ``load_database_rows()``). ``DELETED_TAG_VALUE`` is never combined with
    anything else -- a "no tag applies here" decision always wins outright.
    """
    return [_prefill_row(row, database_rows, tags_by_name, overlap_threshold) for row in rows]


def _prefill_row(
    row: dict,
    database_rows: list[dict],
    tags_by_name: dict[str, Tag],
    overlap_threshold: float,
) -> dict:
    matches = [
        db_row
        for db_row in database_rows
        if str(db_row.get("new_tag", "")).strip() and _same_span(db_row, row, overlap_threshold)
    ]
    # A decision made for this very AI tag outranks decisions for the other tags
    # on the same span. Without one, any correction of the span applies, but a
    # "kept" row says nothing about a record with a different AI tag.
    same_ai_tag = [match for match in matches if match.get("ai_tag") == row.get("ai_tag")]
    matches = same_ai_tag or [match for match in matches if not is_kept_row(match)]
    if not matches:
        return row

    return {**row, "new_tag": _resolve_new_tag(matches, tags_by_name)}


def _resolve_new_tag(matches: list[dict], tags_by_name: dict[str, Tag]) -> str:
    distinct_tags = sorted({match["new_tag"] for match in matches})
    if len(distinct_tags) == 1:
        return distinct_tags[0]

    if DELETED_TAG_VALUE in distinct_tags:
        # A "no tag applies here" decision always wins outright, regardless
        # of what else matched -- it isn't a taggable value to combine.
        deleted_matches = [m for m in matches if m["new_tag"] == DELETED_TAG_VALUE]
        newest_match = max(
            deleted_matches, key=lambda match: _file_mtime(match.get("_source_file"))
        )
        return newest_match["new_tag"]

    double_codable = {tag for tag in distinct_tags if _allows_double_coding(tag, tags_by_name)}
    non_double_codable = [tag for tag in distinct_tags if tag not in double_codable]

    if len(non_double_codable) > 1:
        # These tags mutually conflict (none of them allows double coding),
        # so only one can be kept: the one from the most recently saved
        # database file.
        conflicting_matches = [m for m in matches if m["new_tag"] in non_double_codable]
        newest_match = max(
            conflicting_matches, key=lambda match: _file_mtime(match.get("_source_file"))
        )
        non_double_codable = [newest_match["new_tag"]]

    return ", ".join(sorted(double_codable.union(non_double_codable)))


def _file_mtime(file: Path | None) -> float:
    if file is None or not Path(file).exists():
        return 0.0
    return Path(file).stat().st_mtime


def write_data_table(rows: list[dict], output_file: Path) -> None:
    """Write rows to a workbook's data sheet only (no settings sheet), e.g. for the database."""
    df = pd.DataFrame(rows, columns=build_post_process_columns())
    write_workbook_sheets({DATA_SHEET_NAME: df}, output_file)
