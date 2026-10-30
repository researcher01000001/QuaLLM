import dataclasses
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openpyxl
import pandas as pd

from src.config import project as config_project
from src.config.general import UNDEFINED_TAGS_ENABLED
from src.core import process_log
from src.core.models import (
    AIQuote,
    AtlasQuote,
    CodingScheme,
    ConsensusQuote,
    Document,
    Quote,
    Row,
    Tag,
)
from src.dataset import document_content
from src.io.io_utils import get_matching_files

if TYPE_CHECKING:
    from src.config.coding import CodingConfiguration


class QuoteLoader(ABC):
    """Abstract base class for loading quotes from different sources."""

    def __init__(
        self,
        source_paths: Path | list[Path],
        tags: list[Tag],
        condition_match: Callable[[str], list[str]] | None = None,
        log_step: str | None = None,
    ) -> None:
        """Initialise with source paths and a list of tags.

        Args:
            source_paths: Path or list of Paths to the source file(s) or directory containing them.
            tags: List of Tag objects, used to resolve tag references.
            condition_match: Optional lookup from document name to the list
                of coding-scheme condition values valid for that document
                (see ``src.dataset.document_identity.condition_match()``). When
                omitted, tag resolution skips the condition check entirely.
                A tag whose condition isn't in the list is treated like an
                undefined tag (see ``_apply_condition()``).
            log_step: Optional process-log ``step`` override for any
                ``log_undefined_tag()`` calls made while loading (see
                ``_resolve_tag()``/``_apply_condition()``) -- e.g. a specific
                configuration's namespaced sub-step. Omitted/None falls back
                to the active log's own step.
        """
        self.tags = tags
        self.source_paths = source_paths
        self.condition_match = condition_match
        self.log_step = log_step

    @abstractmethod
    def load_quotes(self) -> dict[str, list[Quote]]:
        """Load quotes from the data source.

        Returns:
            Dictionary mapping document name to a list of Quote objects with tags resolved.
        """
        pass


class AtlasQuoteLoader(QuoteLoader):
    """Load quotes from ATLAS Excel export."""

    def load_quotes(self) -> dict[str, list[Quote]]:
        """Load quotes from the ATLAS Excel files.

        Returns:
            Dictionary mapping document name to a list of AtlasQuote objects. Rows with no quote text or no assigned
            tags are skipped.
        """
        all_quotes: dict[str, list[Quote]] = {}
        tags_by_name = {tag.name: tag for tag in self.tags}

        for file_path in get_matching_files(self.source_paths, "*.xlsx"):
            try:
                quotes_df = pd.read_excel(file_path)
            except Exception as e:
                message = f"Error reading {file_path}: {e}"
                print(message)
                process_log.current().log_general_warning(
                    severity="Error", source_name=file_path.name, message=message
                )
                continue

            for _, row in quotes_df.iterrows():
                quote_text = row["quote_text"]
                doc_name = str(row["Source Name"]).split("/")[-1].strip()

                if pd.isna(quote_text):
                    continue

                doc_valid_conditions = (
                    self.condition_match(doc_name) if self.condition_match else None
                )

                quote_tags = []
                for tag_name, tag in tags_by_name.items():
                    if (
                        tag_name in quotes_df.columns
                        and pd.notna(row[tag_name])
                        and row[tag_name] != 0
                    ):
                        resolved = _apply_condition(
                            tag, doc_valid_conditions, doc_name=doc_name, step=self.log_step
                        )
                        if resolved is not None:
                            quote_tags.append(resolved)

                if quote_tags:
                    if doc_name not in all_quotes:
                        all_quotes[doc_name] = []
                    all_quotes[doc_name].append(
                        AtlasQuote(
                            text=quote_text,
                            tags=quote_tags,
                            start=int(row["quote_start"]),
                            end=int(row["quote_end"]) + 1,  # end is inclusive in Excel
                        )
                    )

        return all_quotes


class AIQuoteLoader(QuoteLoader):
    """Load quotes from AI-generated coding."""

    def load_quotes(self) -> dict[str, list[Quote]]:
        """Load quotes from the AI coding format.

        Returns:
            Dictionary mapping document name to a list of AIQuote objects.
        """
        all_quotes: dict[str, list[Quote]] = {}

        tags_by_name = {tag.name: tag for tag in self.tags}

        for data_file in get_matching_files(
            self.source_paths, "*_classified.json", recursive=True
        ):
            if not data_file.exists():
                continue

            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                quote_text = item.get("phrase")
                if not quote_text:
                    continue

                source = item.get("source", {})
                sourcefile = source.get("sourcefile", "")
                # Remove extension to match ATLAS document names typically used
                doc_name = Path(sourcefile).stem if sourcefile else data_file.stem

                doc_valid_conditions = (
                    self.condition_match(doc_name) if self.condition_match else None
                )

                item_tags = []
                for tag_name in item.get("tags", []):
                    resolved = _resolve_tag(
                        tag_name,
                        tags_by_name,
                        doc_valid_conditions,
                        doc_name=doc_name,
                        step=self.log_step,
                    )
                    if resolved is not None:
                        item_tags.append(resolved)

                if not item_tags:
                    continue

                context = item.get("context")
                if not isinstance(context, str) or not context:
                    context = None

                # Row indices in JSON are 0-based, matching AIQuote
                row_num = source.get("row")

                if doc_name not in all_quotes:
                    all_quotes[doc_name] = []

                all_quotes[doc_name].append(
                    AIQuote(
                        text=quote_text,
                        tags=item_tags,
                        row=row_num,
                        context=context,
                    )
                )

        return all_quotes

    @classmethod
    def load_runs(
        cls,
        base_dir: Path,
        configurations: list["CodingConfiguration"],
        tags: list[Tag],
        iterations: int | None = None,
        condition_match: Callable[[str], list[str]] | None = None,
        log_step_prefix: str | None = None,
    ) -> dict[tuple[str, int], dict[str, list[Quote]]]:
        """Load AI quotes for all configured iterations without merging runs.

        ``load_quotes()`` remains the entrypoint for one concrete run directory.
        This helper only orchestrates repeated calls over the nested output
        layout: ``base_dir / configuration_name / iteration-N``.
        When ``iterations`` is omitted, each configuration's own iteration
        count is used. ``condition_match`` is forwarded to each constructed
        loader (see ``QuoteLoader.__init__``).

        ``log_step_prefix`` is the top-level process-log step this call is
        running under (e.g. ``"run_main"`` or ``"post_process_step1"`` --
        this method is shared by both entry points, so the caller must say
        which). Each configuration's loader is then given
        ``log_step=f"{log_step_prefix}/{configuration_name}"``, so undefined
        tags found while loading are attributed to that specific
        configuration rather than the coarse top-level step. Omitted/None
        (e.g. in tests that call this directly) falls back to the active
        log's own step, unchanged from before this parameter existed.
        """
        runs: dict[tuple[str, int], dict[str, list[Quote]]] = {}
        for configuration in configurations:
            iteration_count = (
                iterations if iterations is not None else configuration.iterations
            )
            log_step = (
                f"{log_step_prefix}/{configuration.configuration_name}"
                if log_step_prefix is not None
                else None
            )
            for iteration in range(iteration_count):
                iteration_dir = (
                    base_dir
                    / configuration.configuration_name
                    / f"iteration-{iteration}"
                )
                runs[(configuration.configuration_name, iteration)] = cls(
                    iteration_dir,
                    tags,
                    condition_match=condition_match,
                    log_step=log_step,
                ).load_quotes()
        return runs


class ConsensusQuoteLoader(QuoteLoader):
    """Load quotes from a consensus.json file (see ``src.coding.consensus.build_consensus_records``)."""

    def load_quotes(self) -> dict[str, list[Quote]]:
        """Load quotes from consensus.json file(s).

        Records with an unresolved position (``start``/``end`` is ``null`` --
        e.g. a single-iteration fallback record whose phrase could not be
        located in the source text) are skipped, since they cannot be
        rendered or compared without a position.

        Returns:
            Dictionary mapping document name to a list of ConsensusQuote objects.
        """
        all_quotes: dict[str, list[Quote]] = {}
        tags_by_name = {tag.name: tag for tag in self.tags}

        for data_file in get_matching_files(self.source_paths, "*.json"):
            if not data_file.exists():
                continue

            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                quote_text = item.get("phrase")
                start = item.get("start")
                end = item.get("end")
                if not quote_text or start is None or end is None:
                    continue

                doc_name = item.get("source", {}).get("source_name") or data_file.stem
                doc_valid_conditions = (
                    self.condition_match(doc_name) if self.condition_match else None
                )

                item_tags = []
                for tag_name in item.get("tags", []):
                    resolved = _resolve_tag(
                        tag_name,
                        tags_by_name,
                        doc_valid_conditions,
                        doc_name=doc_name,
                        step=self.log_step,
                    )
                    if resolved is not None:
                        item_tags.append(resolved)

                if not item_tags:
                    continue

                if doc_name not in all_quotes:
                    all_quotes[doc_name] = []

                all_quotes[doc_name].append(
                    ConsensusQuote(
                        text=quote_text,
                        tags=item_tags,
                        start=start,
                        end=end,
                        source_iterations=item.get("source_iterations"),
                    )
                )

        return all_quotes


def _get_or_create_undefined_tag(tags_by_name: dict[str, Tag], tag_name: str) -> Tag:
    """Return an ad-hoc Tag for a name that is not part of the codebook.

    Used when UNDEFINED_TAGS_ENABLED is enabled so tags a loader encounters
    that aren't in the codebook (AI-invented tags, typos, hallucinated
    categories, etc.) are still visible in the rendered output instead of
    being silently dropped. The colour is derived deterministically from the
    tag name so the same undefined tag always renders with the same colour,
    and the lookup is mutated in place so repeated occurrences reuse the same
    Tag instance.

    Args:
        tags_by_name: Lookup of already-resolved tags, keyed by name.
        tag_name: Raw tag name as it appeared in the source data.

    Returns:
        The cached or newly created Tag, marked with additional_tag=True.
    """
    if tag_name not in tags_by_name:
        tags_by_name[tag_name] = Tag(
            name=tag_name,
            color=_hash_color(tag_name),
            additional_tag=True,
        )
    return tags_by_name[tag_name]


def _hash_color(tag_name: str) -> str:
    """Deterministic hex colour derived from a tag name."""
    digest = hashlib.md5(tag_name.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def _apply_condition(
    tag: Tag,
    valid_conditions: list[str] | None,
    doc_name: str | None = None,
    step: str | None = None,
) -> Tag | None:
    """Check a resolved codebook tag against a document's valid conditions.

    If ``tag.condition`` isn't in ``valid_conditions`` (see
    ``src.dataset.document_identity.condition_match()``), the tag is treated like an
    undefined tag: a copy with ``additional_tag=True`` when
    ``UNDEFINED_TAGS_ENABLED`` is enabled (so its real color/description/group
    survive, unlike the synthetic hash-colored tags
    ``_get_or_create_undefined_tag()`` builds for unknown names), or dropped
    (None) otherwise.

    Returns ``tag`` unchanged when the check can't be made (no condition
    restriction on the tag, or no valid-conditions list configured) or when
    it matches.
    """
    if not tag.condition or valid_conditions is None or tag.condition in valid_conditions:
        return tag
    if UNDEFINED_TAGS_ENABLED:
        process_log.current().log_undefined_tag(
            severity="Info",
            source_name=doc_name,
            tag_name=tag.name,
            reason="condition_mismatch",
            resolution="kept_as_additional_tag",
            message=(
                f"Tag '{tag.name}' condition '{tag.condition}' not in {valid_conditions} "
                f"for document {doc_name!r}; kept as an additional/undefined tag."
            ),
            step=step,
        )
        return dataclasses.replace(tag, additional_tag=True)
    process_log.current().log_undefined_tag(
        severity="Warning",
        source_name=doc_name,
        tag_name=tag.name,
        reason="condition_mismatch",
        resolution="dropped",
        message=(
            f"Tag '{tag.name}' condition '{tag.condition}' not in {valid_conditions} "
            f"for document {doc_name!r}; dropped (UNDEFINED_TAGS_ENABLED is False)."
        ),
        step=step,
    )
    return None


def _resolve_tag(
    tag_name: str,
    tags_by_name: dict[str, Tag],
    valid_conditions: list[str] | None,
    doc_name: str | None = None,
    step: str | None = None,
) -> Tag | None:
    """Resolve a raw tag name (from AI/consensus JSON) to a Tag.

    A name matching the codebook is passed through ``_apply_condition()``. A
    name the codebook doesn't recognise at all falls back to
    ``_get_or_create_undefined_tag()`` (or is dropped) exactly as before,
    independent of any condition check.
    """
    tag = tags_by_name.get(tag_name)
    if tag is None:
        if UNDEFINED_TAGS_ENABLED:
            process_log.current().log_undefined_tag(
                severity="Info",
                source_name=doc_name,
                tag_name=tag_name,
                reason="unknown_name",
                resolution="kept_as_additional_tag",
                message=(
                    f"Tag name '{tag_name}' not found in the codebook for document "
                    f"{doc_name!r}; kept as an additional/undefined tag."
                ),
                step=step,
            )
            return _get_or_create_undefined_tag(tags_by_name, tag_name)
        process_log.current().log_undefined_tag(
            severity="Warning",
            source_name=doc_name,
            tag_name=tag_name,
            reason="unknown_name",
            resolution="dropped",
            message=(
                f"Tag name '{tag_name}' not found in the codebook for document "
                f"{doc_name!r}; dropped (UNDEFINED_TAGS_ENABLED is False)."
            ),
            step=step,
        )
        return None
    return _apply_condition(tag, valid_conditions, doc_name=doc_name, step=step)


_CODING_SCHEME_SHEET = "Coding_Scheme"
_GENERAL_RULES_SHEET = "General_Rules"
_MEDICAL_KNOWLEDGE_SHEET = "Medical_Knowledge"


def _parse_examples(examples_raw: str) -> list[str]:
    """Parses a Coding_Scheme "examples" cell's quoted-comma-separated list
    (e.g. '"ex1", "ex2"') into a list of strings. Returns [] for an empty
    input, and falls back to a single-item list if the value doesn't match
    the expected quoted format."""
    if not examples_raw:
        return []
    try:
        return [e.strip(" \"'") for e in examples_raw.split('", "')]
    except Exception:
        return [examples_raw]


def _coding_scheme_rows(descriptions_file: Path) -> list[dict[str, Any]]:
    """Coding_Scheme sheet rows as plain, already-cleaned dicts -- one per
    non-blank row, each with a "key" entry (code_category + ": " + code, or
    just code_category when code is blank) plus every other column
    (code_category, code, group, definition, note, examples, condition,
    double_coding, system) verbatim. Every field is NaN-guarded, so neither
    a raw pandas NaN nor the literal string "nan" ever reaches a Tag or a
    src.core.models.CodingScheme prompt-building method.

    Generic: whatever columns the sheet has beyond code_category/code carry
    straight through, so a different dataset's Coding_Scheme sheet with
    different extra columns still works -- only "key" is synthesized.

    "code" is normalized to "" rather than None when blank -- callers rely
    on its truthiness (both "key"'s own ": code" suffix, and the "{n1}: {n2}"
    if n2 else n1 pattern load_tags() uses for Tag.name).

    Note: code_category can itself legitimately be the string "system" (one
    of its ~10 distinct real values) -- unrelated to the boolean "system"
    column below, which load_tags() copies onto Tag.system.
    """
    df = pd.read_excel(descriptions_file, sheet_name=_CODING_SCHEME_SHEET)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        category_value = row.get("code_category", row.get("name_1"))
        if pd.isna(category_value):
            continue

        code_category = str(category_value).strip()
        code_value = row.get("code", row.get("name_2"))
        code = str(code_value).strip() if pd.notna(code_value) else ""
        examples_raw = (
            str(row["examples"]).strip()
            if "examples" in row and pd.notna(row["examples"])
            else ""
        )

        rows.append(
            {
                "key": f"{code_category}: {code}" if code else code_category,
                "code_category": code_category,
                "code": code,
                "group": (
                    str(row["group"]).strip()
                    if "group" in row and pd.notna(row["group"])
                    else None
                ),
                "definition": (
                    str(row["definition"]).strip()
                    if "definition" in row and pd.notna(row["definition"])
                    else None
                ),
                "note": (
                    str(row["note"]).strip()
                    if "note" in row and pd.notna(row["note"])
                    else None
                ),
                "examples": _parse_examples(examples_raw),
                "condition": (
                    str(row["condition"]).strip()
                    if "condition" in row and pd.notna(row["condition"])
                    else None
                ),
                "double_coding": (
                    str(row["double_coding"]).strip().lower() == "yes"
                    if "double_coding" in row and pd.notna(row["double_coding"])
                    else False
                ),
                "system": (
                    str(row["system"]).strip().lower() == "yes"
                    if "system" in row and pd.notna(row["system"])
                    else False
                ),
            }
        )
    return rows


def _general_rules(descriptions_file: Path) -> list[str]:
    """General_Rules sheet's "rule" column as a plain list of strings,
    skipping blank rows."""
    df = pd.read_excel(descriptions_file, sheet_name=_GENERAL_RULES_SHEET)
    return [str(v).strip() for v in df["rule"].dropna()]


def _medical_knowledge(descriptions_file: Path) -> dict[str, list[str]]:
    """Medical_Knowledge sheet as one list per named column -- each column
    is an independent list of free-text items, not row-aligned records, so
    columns commonly end up different lengths. Columns with no real header
    (pandas' auto-generated "Unnamed: N") are skipped."""
    df = pd.read_excel(descriptions_file, sheet_name=_MEDICAL_KNOWLEDGE_SHEET)
    return {
        str(col): [str(v).strip() for v in df[col].dropna()]
        for col in df.columns
        if not str(col).startswith("Unnamed")
    }


def load_prompt_context(descriptions_file: Path) -> CodingScheme:
    """Raw coding-scheme data (Coding_Scheme/General_Rules/Medical_Knowledge
    sheets of descriptions_file) as a CodingScheme, for AI-coding prompt
    construction (see src.coding.prompt_builders). Distinct from load_tags(): that
    builds Tag objects for the rest of the pipeline (matching/scoring/
    rendering); this is a raw, prompt-facing view of the same Coding_Scheme
    sheet plus the two sheets load_tags() doesn't touch at all."""
    return CodingScheme(
        coding_scheme=_coding_scheme_rows(descriptions_file),
        general_rules=_general_rules(descriptions_file),
        medical_knowledge=_medical_knowledge(descriptions_file),
    )


def load_tags(codes_file: Path, descriptions_file: Path | None = None) -> list[Tag]:
    """Load tag definitions.

    Creates a Tags objects based on the the descriptions file and also the codes file for colouring.

    Args:
        codes_file: Path to the ATLAS codes .xlsx file (used for colours).
        descriptions_file: Path to the descriptions .xlsx file.

    Returns:
        List of Tag objects.
    """
    loaded_colors = _load_code_colors(codes_file)
    colors = loaded_colors or {}

    tags = []

    if descriptions_file:
        for row in _coding_scheme_rows(descriptions_file):
            n1 = row["code_category"]
            n2 = row["code"]
            tag_name = f"{n1}: {n2}" if n2 else n1

            # Normalize some known inconsistencies if necessary, or just rely on lowercasing
            if loaded_colors is None:
                color = _hash_color(tag_name)
            else:
                color = colors.get(tag_name.lower(), "#808080")

            tags.append(
                Tag(
                    name=tag_name,
                    color=color,
                    group=row["group"],
                    description=row["definition"],
                    note=row["note"],
                    examples=row["examples"],
                    condition=row["condition"],
                    can_co_occur=row["double_coding"],
                    system=row["system"],
                )
            )
    else:
        # Fallback if no descriptions file provided
        for tag_name_lower, color in colors.items():
            # without descriptions file, we don't have the properly cased tag name easily
            tags.append(Tag(name=tag_name_lower.title(), color=color))

    return tags


def _load_code_colors(codes_file: Path) -> dict[str, str] | None:
    """Read tag colours (lowercased name -> hex) from the ATLAS codes file's
    font colours, or None (with a logged warning) if the file is missing."""
    if not codes_file.exists():
        message = f"Codes file {codes_file} not found; using name-derived colours for all tags."
        print(f"Warning: {message}")
        process_log.current().log_general_warning(
            severity="Warning", source_name=codes_file.name, message=message
        )
        return None

    codes_ws = openpyxl.load_workbook(codes_file).active
    if codes_ws is None:
        raise ValueError(f"{codes_file} has no active worksheet.")
    colors = {}

    for row in codes_ws.iter_rows(min_row=2, min_col=1, max_col=1):
        cell = row[0]
        if not cell.value:
            continue

        tag_name_str = str(cell.value).strip()
        hex_color = "#808080"

        if cell.font and cell.font.color and cell.font.color.rgb:
            rgb = str(cell.font.color.rgb)
            if len(rgb) == 8:
                hex_color = f"#{rgb[2:]}"
            elif len(rgb) == 6:
                hex_color = f"#{rgb}"

        colors[tag_name_str.lower()] = hex_color

    return colors


def load_default_tags() -> list[Tag]:
    """Load tag definitions from config_project.CODES_FILE/DESCRIPTION_FILE
    -- the codebook every entry point (run.py, run_post_processing.py) loads
    its tags from."""
    return load_tags(config_project.CODES_FILE, config_project.DESCRIPTION_FILE)


class DocumentLoader:
    """Loads a text document and parses it into paragraphs with absolute character offsets."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> Document:
        """Reads the file and returns a populated Document object."""
        source_name = self.file_path.stem  # .stem gets filename without extension
        rows = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Normalise Windows line endings so that character offsets stay correct
        text = text.replace("\r\n", "\n")

        character_counter = 0
        row_number = 0

        for t in text.splitlines():
            start_position = character_counter
            end_position = character_counter + len(t)

            rows.append(
                Row(
                    index=row_number,
                    text=t,
                    start_position=start_position,
                    end_position=end_position,
                )
            )
            character_counter += len(t) + 1  # +1 for the newline character
            row_number += 1

        display_column = document_content.display_column_for(source_name)
        return Document(source_name=source_name, rows=rows, display_column=display_column)
