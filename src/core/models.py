from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.analysis.agreement import (
    CharacterLevelF1Result,
    character_level_f1,
    f1_score,
    jaccard_similarity,
    normalized_levenshtein_distance,
)
from src.core import process_log


@dataclass
class Tag:
    """Represents a coding tag with its properties.

    Attributes:
        name: Display name of the tag.
        color: Optional Hex color string (e.g. "#ff0000").
        group: Optional group name for organising tags.
        description: Optional detailed description of the tag.
        note: Optional additional notes about the tag.
        examples: Optional list of example usages of the tag.
        condition: Optional condition this tag is restricted to. The
            vocabulary is project-defined (see
            ``src.dataset.document_identity.condition_match()``); a tag with no
            condition set applies to every condition (see
            ``tags_for_condition()``).
        can_co_occur: Optional boolean indicating whether this tag may
            co-occur with a *different* tag on the *same* quote span
            ("double coding") -- not about reuse across separate quotes. See
            ``tags_disallow_co_occurrence()`` below and
            ``review_database._tags_compatible()``.
        additional_tag: Optional boolean marking this tag as not part of the
            defined coding schema (undefined/out-of-codebook) -- applies to
            any source (AI, human/ATLAS, a condition-mismatched codebook
            tag), not just AI output. See ``config.UNDEFINED_TAGS_ENABLED``,
            ``loaders._get_or_create_undefined_tag()``/``_apply_condition()``,
            and ``src.report.html._tag_modifier_classes()``'s "Not part of the
            predefined codebook" tooltip.
        system: Optional boolean from the coding scheme's ``system`` column
            marking an administrative tag that supports AI coding but is not
            part of the analysis: excluded from every agreement metric and
            hidden by default in HTML. See ``quotes_excluding_system()``.
    """

    name: str
    color: str | None = None
    group: str | None = None
    description: str | None = None
    note: str | None = None
    examples: list[str] | None = None
    condition: str | None = None
    can_co_occur: bool | None = None
    additional_tag: bool | None = None
    system: bool | None = None

    def __hash__(self) -> int:
        """Return hash based on tag name."""
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        """Compare tags by name.

        Args:
            other: Object to compare against.

        Returns:
            True if other is a Tag with the same name.
        """
        if isinstance(other, Tag):
            return self.name == other.name
        return False


def tags_for_condition(tags: Sequence[Tag], valid_conditions: Collection[str]) -> list[Tag]:
    """Filter tags to those valid for a document.

    A tag with no ``condition`` set applies to every document. Otherwise a
    tag is kept when its ``condition`` is in ``valid_conditions`` -- the
    coding-scheme condition values valid for one specific document (see
    ``src.dataset.document_identity.condition_match()``).
    """
    return [tag for tag in tags if not tag.condition or tag.condition in valid_conditions]


def tags_disallow_co_occurrence(tag_a: Tag, tag_b: Tag) -> bool:
    """True if ``tag_a``/``tag_b`` may not both apply to the same quote span:
    both individually disallow double coding (``Tag.can_co_occur`` falsy).

    A ``can_co_occur=True`` tag may always be paired with anything, so two
    tags conflict only when neither allows it.
    """
    return not (bool(tag_a.can_co_occur) or bool(tag_b.can_co_occur))


def _one_line(value: str) -> str:
    """Joins any embedded line breaks in value with ", ", so a single
    coding-scheme entry's text (which may itself span multiple physical
    lines in the source spreadsheet cell) can never split a
    CodingScheme.coding_scheme_text() line into more than one."""
    return ", ".join(value.splitlines())


@dataclass(frozen=True)
class CodingScheme:
    """The raw content of DESCRIPTION_FILE's three sheets (see
    src.io.loaders.load_prompt_context()), for AI-coding prompt construction
    (src.coding.prompt_builders). Deliberately generic rather than a fixed
    per-column model -- config/project.py's source spreadsheet columns are
    dataset-specific, so coding_scheme/medical_knowledge carry whatever
    columns the sheet actually has, keyed by header name.

    Attributes:
        coding_scheme: One dict per Coding_Scheme row, each with a "key"
            entry (code_category + ": " + code, or just code_category when
            code is blank) plus every other column verbatim.
        general_rules: General_Rules sheet's "rule" column, one string per
            row.
        medical_knowledge: One list per Medical_Knowledge column (header
            name -> that column's non-blank cell values); columns are
            independent, not row-aligned records.
    """

    coding_scheme: list[dict[str, Any]]
    general_rules: list[str]
    medical_knowledge: dict[str, list[str]]

    def general_rules_text(self) -> str:
        """Every general rule, one per line."""
        return "\n".join(self.general_rules)

    def medical_knowledge_text(self, columns: list[str] | None = None) -> str:
        """Text for the given medical_knowledge column(s) (every column if
        columns is None) -- one line per item, headed by the column name
        when more than one column is included."""
        selected = columns if columns is not None else list(self.medical_knowledge)
        lines: list[str] = []
        for column in selected:
            items = self.medical_knowledge.get(column, [])
            if len(selected) > 1:
                lines.append(f"{column}:")
            lines.extend(f"- {item}" for item in items)
        return "\n".join(lines)

    def coding_scheme_text(
        self,
        entry_filter: Callable[[dict[str, Any]], bool] | None = None,
        *,
        group: bool = False,
        definition: bool = False,
        note: bool = False,
        examples: bool = False,
        condition: bool = False,
        double_coding: bool = False,
        system: bool = False,
        row_join_symbole: str = ";"
    ) -> str:
        """One line per coding_scheme entry passing entry_filter (every
        entry if entry_filter is None; the filter receives the whole entry
        dict, so it can filter on any column, not just condition) -- the
        entry's "key" alone, or "key: ..." followed by whichever of the
        flagged columns that entry has a non-empty value for, in the fixed
        order above. Every flag defaults to False: a caller opts in per
        column."""
        lines: list[str] = []
        for entry in self.coding_scheme:
            if entry_filter is not None and not entry_filter(entry):
                continue
            parts: list[str] = []
            if group and entry.get("group"):
                parts.append(f"Group: {_one_line(entry['group'])}")
            if definition and entry.get("definition"):
                parts.append(_one_line(entry["definition"]))
            if note and entry.get("note"):
                parts.append(f"(Note: {_one_line(entry['note'])})")
            if examples and entry.get("examples"):
                examples_text = ", ".join(_one_line(e) for e in entry["examples"])
                parts.append(f"Examples: {examples_text}")
            if condition and entry.get("condition"):
                parts.append(f"Condition: {_one_line(entry['condition'])}")
            if double_coding and entry.get("double_coding"):
                parts.append("Can co-occur with other tags")
            if double_coding and not entry.get("double_coding"):
                parts.append("Cannot co-occur with other tags that also cannot co-occur.")
            if system and entry.get("system"):
                parts.append("System")
            key = _one_line(entry["key"])
            lines.append(f"<tag>{key}</tag>: {f'{row_join_symbole} '.join(parts)}" if parts else key)
        return "\n".join(lines)


@dataclass
class Row:
    """Represents one row of the document with its text and character
    offsets. This is the technical, 0-based indexed unit the backend works
    with; the human-facing display value is looked up separately, from the
    owning ``Document``'s ``display_column`` (see ``Document.display_value()``).

    Attributes
        index: Zero-based index of this row within the document.
        text: The text content of the row.
        start_position: Start character offset of the row.
        end_position: End character offset of the row.
    """

    index: int
    text: str
    start_position: int
    end_position: int

    def __eq__(self, other: object) -> bool:
        """Compare rows by index and text."""
        if isinstance(other, Row):
            return self.index == other.index and self.text == other.text
        return False


@dataclass
class Quote(ABC):
    """Abstract base class for coded quotes.

    Attributes:
        text: The quoted text as it appears in the document.
        tags: List of tags assigned to this quote.
        start: Start character offset in the document text.
        end: End character offset in the document text.
        quote_id: Unique identifier used for rendering correlation.
    """

    text: str
    tags: list[Tag]
    start: int | None = None
    end: int | None = None
    quote_id: int | None = None

    @abstractmethod
    def find_position(
        self,
        search_text: str,
        start_offset: int = 0,
        step: str | None = None,
        document: "Document | None" = None,
    ) -> tuple[int, int] | None:
        """Find the character position of this quote in the provided text.

        Args:
            search_text: The text to search within.
            start_offset: The absolute start position of search_text in the document.
            step: Optional process-log step override (see
                ``src.core.process_log.ProcessLog``) for any position-not-found
                logging this lookup triggers -- e.g. a specific
                configuration's namespaced sub-step. Only ``AIQuote`` (the
                only subclass that actually searches for a position, and
                therefore the only one that can fail to find it) uses this;
                the others accept and ignore it for signature consistency.
            document: Optional owning Document, used only for best-effort
                ``display_value`` resolution on any position-not-found log
                entry this lookup triggers. Same "accept and ignore" rule as
                ``step`` for subclasses that never fail to find a position.

        Returns:
            (start, end) character offsets if found, None otherwise.
        """
        pass

    @abstractmethod
    def in_row(self, row: Row) -> bool:
        """Check if the quote is within the given row's boundaries.

        Args:
            row: The row to check against.
        Returns:
            True if the quote is within the row, False otherwise.
        """
        pass


def quotes_excluding_system(
    quotes: Sequence[Quote], exclude_system_tags: bool
) -> list[Quote]:
    """Return copies of ``quotes`` with any ``Tag.system`` tag removed from
    each quote's ``tags`` list, when ``exclude_system_tags`` is True.

    Used to keep a document's administrative/meta tags (the coding scheme's
    ``system`` column) out of agreement calculations while leaving them
    untouched everywhere else (e.g. rendering). A quote whose tags become
    empty is kept, not dropped -- it simply contributes nothing to
    downstream metrics. ``exclude_system_tags=False`` disables filtering.
    """
    if not exclude_system_tags:
        return list(quotes)
    return [
        replace(quote, tags=[tag for tag in quote.tags if not tag.system])
        for quote in quotes
    ]


def quotes_excluding_undefined(
    quotes: Sequence[Quote], drop_undefined_tags: bool
) -> list[Quote]:
    """Return copies of ``quotes`` with any ``Tag.additional_tag`` tag removed
    from each quote's ``tags`` list, when ``drop_undefined_tags`` is True.

    Sibling of ``quotes_excluding_system()`` for the toggleable undefined-tags
    feature (see ``src.config.general.UNDEFINED_TAGS_ENABLED``): a quote whose tags
    become empty is kept, not dropped. ``drop_undefined_tags=False`` disables
    filtering (today's default -- undefined tags are included).
    """
    if not drop_undefined_tags:
        return list(quotes)
    return [
        replace(quote, tags=[tag for tag in quote.tags if not tag.additional_tag])
        for quote in quotes
    ]


@dataclass
class AtlasQuote(Quote):
    """Quote with exact character positions from ATLAS."""

    def find_position(
        self,
        search_text: str,
        start_offset: int = 0,
        step: str | None = None,
        document: "Document | None" = None,
    ) -> tuple[int, int] | None:
        """Return the stored character offsets if they are within the search_text.

        Args:
            search_text: The text to search within.
            start_offset: The absolute start position of search_text in the document.

        Returns:
            (start, end) absolute character offsets if valid, None otherwise.
        """
        if self.start is None or self.end is None:
            return None

        # Check if the quote is within the boundaries of the provided search_text
        if self.start < start_offset or self.end > start_offset + len(search_text):
            return None

        return (self.start, self.end)

    def in_row(self, row: Row) -> bool:
        if self.start is None:
            return False
        return (
            self.start >= row.start_position
            and self.start <= row.end_position
        )


@dataclass
class ConsensusQuote(Quote):
    """Quote span derived from character-level consensus across AI iterations."""

    source_iterations: int | None = None

    def find_position(
        self,
        search_text: str,
        start_offset: int = 0,
        step: str | None = None,
        document: "Document | None" = None,
    ) -> tuple[int, int] | None:
        if self.start is None or self.end is None:
            return None
        if self.start < start_offset or self.end > start_offset + len(search_text):
            return None
        return (self.start, self.end)

    def in_row(self, row: Row) -> bool:
        if self.start is None or self.end is None:
            return False
        return (
            self.start < row.end_position and self.end > row.start_position
        )


@dataclass
class AIQuote(Quote):
    """Quote from AI coding."""

    row: int | None = None
    context: str | None = None

    def find_position(
        self,
        search_text: str,
        start_offset: int = 0,
        step: str | None = None,
        document: "Document | None" = None,
    ) -> tuple[int, int] | None:
        """Find the quote text in *search_text* and cache absolute coordinates.

        Uniqueness-first strategy:

        1. If the quote text occurs exactly once in ``search_text``, use it
           directly -- ``context`` is irrelevant.
        2. Otherwise, if ``context`` occurs exactly once in ``search_text``,
           narrow the search to that window.
        3. If the quote also occurs exactly once inside that context window,
           use it.
        4. Otherwise (the context itself is not unique, or the quote still
           repeats inside a unique context) fall back to the first available
           occurrence -- inside the first context match if the quote text can
           be found there, otherwise the first raw occurrence of the quote
           anywhere in ``search_text``. This is logged as an error and every
           tag on this quote is renamed to ``position_unsure_<name>`` and
           marked ``additional_tag=True`` so the uncertain placement stays
           visible in the rendered output.

        On first success the coordinates are stored on ``self.start`` /
        ``self.end`` so downstream metrics and renderers can read them
        directly. Subsequent calls return the cached value.

        Args:
            search_text: The text to search in (e.g. a paragraph's text).
            start_offset: The absolute start position of ``search_text``
                in the document.
            step: Optional process-log step override for any
                ``log_quote_position_issue()`` calls this lookup triggers --
                e.g. a specific configuration's namespaced sub-step.
                Omitted/None falls back to the active log's own step.
            document: Optional owning Document, used to resolve
                ``display_value`` and ``source_name`` on any
                ``log_quote_position_issue()`` call this lookup triggers.

        Returns:
            (start, end) absolute character offsets if found, None otherwise.
        """
        if self.start is not None and self.end is not None:
            return (self.start, self.end)

        display_value = (
            document.display_value(self.row)
            if document is not None and self.row is not None
            else None
        )

        search_text_lower = search_text.lower()
        quote_text_lower = self.text.lower()
        quote_count = search_text_lower.count(quote_text_lower)

        # Step 1: the quote alone is unambiguous.
        if quote_count == 1:
            relative_pos = search_text_lower.find(quote_text_lower)
            return self._finalize_position(start_offset, relative_pos)

        if quote_count == 0:
            return self._log_not_found(document, display_value, step)

        # From here the quote occurs more than once in search_text; try to
        # narrow it down via context.
        context_lower = self.context.lower() if self.context else None
        context_count = search_text_lower.count(context_lower) if context_lower else 0

        if context_count >= 1:
            # Steps 2/6: the first (and, when context_count == 1, only)
            # context occurrence.
            context_pos = search_text_lower.find(context_lower)
            context_window = search_text_lower[
                context_pos : context_pos + len(context_lower)
            ]
            quote_count_in_context = context_window.count(quote_text_lower)
            relative_in_context = context_window.find(quote_text_lower)

            if relative_in_context != -1:
                # Step 4: context unique and quote unique within it.
                if context_count == 1 and quote_count_in_context == 1:
                    return self._finalize_position(
                        start_offset, context_pos + relative_in_context
                    )

                # Steps 5/6: either the context itself is not unique (took
                # the first occurrence), or the quote still repeats inside
                # an otherwise-unique context -- best effort, flagged unsure.
                if context_count != 1:
                    reason = (
                        f"context '{self.context}' occurs {context_count} times in "
                        f"the text"
                    )
                else:
                    reason = (
                        f"quote '{self.text}' occurs {quote_count_in_context} times "
                        f"within its context '{self.context}'"
                    )
                message = (
                    f"Row {self.row}: {reason}; using the first occurrence and "
                    f"marking the position as unsure."
                )
                return self._finalize_unsure_position(
                    start_offset,
                    context_pos + relative_in_context,
                    message,
                    document,
                    display_value,
                    step,
                )

        # No usable context (missing, or the quote text could not be found
        # inside the chosen context window) -- fall back to the first raw
        # occurrence of the quote anywhere in search_text.
        relative_pos = search_text_lower.find(quote_text_lower)
        message = (
            f"Row {self.row}: quote '{self.text}' occurs {quote_count} times and its "
            f"context ({self.context!r}) did not resolve a unique position; using "
            f"the first occurrence and marking the position as unsure."
        )
        return self._finalize_unsure_position(
            start_offset, relative_pos, message, document, display_value, step
        )

    def _finalize_position(
        self, start_offset: int, relative_pos: int
    ) -> tuple[int, int]:
        """Cache and return an unambiguous match's absolute coordinates."""
        self.start = start_offset + relative_pos
        self.end = self.start + len(self.text)
        return (self.start, self.end)

    def _finalize_unsure_position(
        self,
        start_offset: int,
        relative_pos: int,
        message: str,
        document: "Document | None",
        display_value: str | None,
        step: str | None,
    ) -> tuple[int, int]:
        """Cache a best-effort match, log it as an error, and rename every
        tag on this quote to ``position_unsure_<name>`` (additional_tag=True)
        so the uncertain placement stays visible in the rendered output."""
        print(message)
        process_log.current().log_quote_position_issue(
            severity="Error",
            source_name=document.source_name if document else None,
            row=self.row,
            display_value=display_value,
            issue_type="position_unsure",
            quote_text=self.text,
            context_text=self.context,
            message=message,
            step=step,
        )
        self.tags = [
            replace(tag, name=f"position_unsure_{tag.name}", additional_tag=True)
            for tag in self.tags
        ]
        return self._finalize_position(start_offset, relative_pos)

    def _log_not_found(
        self,
        document: "Document | None",
        display_value: str | None,
        step: str | None,
    ) -> None:
        import warnings

        message = (
            f"Could not find quote in text in row {self.row}: {self.text[:50]}..."
        )
        warnings.warn(message, stacklevel=2)
        process_log.current().log_quote_position_issue(
            severity="Error",
            source_name=document.source_name if document else None,
            row=self.row,
            display_value=display_value,
            issue_type="not_found",
            quote_text=self.text,
            context_text=self.context,
            message=message,
            step=step,
        )
        return None

    def in_row(self, row: Row) -> bool:
        return row.index == self.row


@dataclass(frozen=True)
class DisplayColumn:
    """A document's generic, source-file-driven per-row display value.

    Attributes:
        source_column: Name of the source-file column this was read from
            (config_project.DOCUMENT_ORDER_COLUMN).
        label: Human-facing label for source_column (config_project.DOCUMENT_ORDER_COLUMN_LABEL).
        values: row.index -> that row's source_column value, as a string.
    """

    source_column: str
    label: str
    values: dict[int, str]


@dataclass
class Document:
    """Represents a document with its source name and its rows.

    Attributes:
        source_name: Name of the source document.
        rows: List of Row objects contained in the document.
        display_column: Generic per-row display value read from the source
            file (see src.dataset.document_content.display_column_for()).
    """

    source_name: str
    rows: list[Row]
    display_column: DisplayColumn

    def display_value(self, row_index: int) -> str:
        """This document's display value for row_index, or "" if unresolved."""
        return self.display_column.values.get(row_index, "")

    def append_quotes(self, quotes_1: list[Quote], quotes_2: list[Quote]) -> None:
        """Append quotes to the rows of the document.

        Args:
            quotes_1: List of Quote objects from the first coder.
            quotes_2: List of Quote objects from the second coder.
        """
        coded_rows = []
        for row in self.rows:
            p_quotes_1 = [q for q in quotes_1 if q.in_row(row)]
            p_quotes_2 = [q for q in quotes_2 if q.in_row(row)]
            coded_rows.append(
                CodedRow(
                    index=row.index,
                    text=row.text,
                    start_position=row.start_position,
                    end_position=row.end_position,
                    quotes_1=p_quotes_1,
                    quotes_2=p_quotes_2,
                )
            )
        self.rows = coded_rows

    def compare_coders(
        self, exclude_system_tags: bool = False, drop_undefined_tags: bool = False
    ) -> tuple[float, float, float, float, float, float, float]:
        """Compares the tags of both coders and returns average agreement metrics.

        Args:
            exclude_system_tags: If True, drop every ``Tag.system`` tag from
                every row's calculation. False compares every tag.
            drop_undefined_tags: If True, also drop every tag with
                ``additional_tag=True`` from every row's calculation
                (see ``src.config.general.UNDEFINED_TAGS_ENABLED``). False compares
                every tag not already excluded by ``exclude_system_tags``.

        Returns:
            A tuple of average normalized Levenshtein distance, Jaccard similarity,
            F1 score, original character-level F1, original character-level agreement,
            pruned character-level F1, and pruned character-level agreement.
        """
        lev_distances = []
        jaccard_scores = []
        f1_scores = []
        pruned_char_f1_scores = []
        pruned_char_agreement_scores = []
        original_char_f1_scores = []
        original_char_agreement_scores = []
        for row in self.rows:
            if isinstance(row, CodedRow):
                lev_dist, jaccard_sim, f1, char_f1_result = row.compare_coders(
                    exclude_system_tags, drop_undefined_tags, document=self
                )
                lev_distances.append(lev_dist)
                jaccard_scores.append(jaccard_sim)
                f1_scores.append(f1)
                pruned_char_f1_scores.append(char_f1_result["pruned"]["char_f1"])
                pruned_char_agreement_scores.append(
                    char_f1_result["pruned"]["agreement"]
                )
                original_char_f1_scores.append(char_f1_result["original"]["char_f1"])
                original_char_agreement_scores.append(
                    char_f1_result["original"]["agreement"]
                )
                print(
                    f"Row {row.index}: Levenshtein = {lev_dist:.4f}, Jaccard = {jaccard_sim:.4f}, F1 = {f1:.4f}, "
                    f"Original Char-F1 = {char_f1_result['original']['char_f1']:.4f}, Original Agreement = {char_f1_result['original']['agreement']:.4f}, "
                    f"Pruned Char-F1 = {char_f1_result['pruned']['char_f1']:.4f}, Pruned Agreement = {char_f1_result['pruned']['agreement']:.4f}"
                )
            else:
                message = f"Warning: Row {row.index} is not a CodedRow."
                print(message)
                process_log.current().log_general_warning(
                    severity="Warning",
                    source_name=self.source_name,
                    row=row.index,
                    display_value=self.display_value(row.index),
                    message=message,
                )
        avg_lev = sum(lev_distances) / len(lev_distances) if lev_distances else 0.0
        avg_jaccard = (
            sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        )
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_pruned_char_f1 = (
            sum(pruned_char_f1_scores) / len(pruned_char_f1_scores)
            if pruned_char_f1_scores
            else 0.0
        )
        avg_pruned_char_agreement = (
            sum(pruned_char_agreement_scores) / len(pruned_char_agreement_scores)
            if pruned_char_agreement_scores
            else 0.0
        )
        avg_original_char_f1 = (
            sum(original_char_f1_scores) / len(original_char_f1_scores)
            if original_char_f1_scores
            else 0.0
        )
        avg_original_char_agreement = (
            sum(original_char_agreement_scores) / len(original_char_agreement_scores)
            if original_char_agreement_scores
            else 0.0
        )
        return (
            avg_lev,
            avg_jaccard,
            avg_f1,
            avg_original_char_f1,
            avg_original_char_agreement,
            avg_pruned_char_f1,
            avg_pruned_char_agreement,
        )


@dataclass
class CodedRow(Row):
    """A row enriched with quotes from two different sources for comparison.

    Attributes:
        quotes_1: Sequence of quotes from the first coder.
        quotes_2: Sequence of quotes from the second coder.
    """

    quotes_1: list[Quote]
    quotes_2: list[Quote]

    def compare_coders(
        self,
        exclude_system_tags: bool = False,
        drop_undefined_tags: bool = False,
        step: str | None = None,
        document: "Document | None" = None,
    ) -> tuple[float, float, float, CharacterLevelF1Result]:
        """Compares the tags of both coders and returns normalized Levenshtein distance, Jaccard similarity, F1 score, and character-level F1.

        Args:
            exclude_system_tags: If True, drop every ``Tag.system`` tag
                before comparing. False compares every tag.
            drop_undefined_tags: If True, also drop every tag with
                ``additional_tag=True`` before comparing (see
                ``src.config.general.UNDEFINED_TAGS_ENABLED``). False compares every
                tag not already excluded by ``exclude_system_tags``.
            step: Optional process-log step override, forwarded to any
                ``AIQuote.find_position()`` call this comparison triggers --
                e.g. a specific configuration's namespaced sub-step.
                Omitted/None falls back to the active log's own step.
            document: Optional owning Document, forwarded to
                ``AIQuote.find_position()`` for best-effort ``display_value``
                resolution on any position-not-found log entry.

        Returns:
            A tuple of (normalized Levenshtein distance, Jaccard similarity, F1 score, character-level F1 result dict).
        """
        # Resolve any AIQuote positions that haven't been cached yet
        for quote in self.quotes_1 + self.quotes_2:
            if isinstance(quote, AIQuote) and quote.start is None:
                quote.find_position(self.text, self.start_position, step=step, document=document)

        def get_sort_key(quote: Quote):
            # Use cached position if available, otherwise try to find it
            if quote.start is not None:
                return quote.start
            pos = quote.find_position(self.text, self.start_position, step=step, document=document)
            return pos[0] if pos else 0

        sorted_quotes_1 = sorted(self.quotes_1, key=get_sort_key)
        sorted_quotes_2 = sorted(self.quotes_2, key=get_sort_key)

        scored_quotes_1 = quotes_excluding_system(sorted_quotes_1, exclude_system_tags)
        scored_quotes_2 = quotes_excluding_system(sorted_quotes_2, exclude_system_tags)
        scored_quotes_1 = quotes_excluding_undefined(scored_quotes_1, drop_undefined_tags)
        scored_quotes_2 = quotes_excluding_undefined(scored_quotes_2, drop_undefined_tags)

        # Get the list of tags for each quote in the comparison row
        tags_1 = [tag for quote in scored_quotes_1 for tag in quote.tags]
        tags_2 = [tag for quote in scored_quotes_2 for tag in quote.tags]

        lev_dist = normalized_levenshtein_distance(tags_1, tags_2)
        jaccard_sim = jaccard_similarity(tags_1, tags_2)
        f1 = f1_score(tags_1, tags_2)

        # Calculate character-level F1
        char_f1_result = character_level_f1(
            scored_quotes_1, scored_quotes_2, self.text, self.start_position
        )

        return lev_dist, jaccard_sim, f1, char_f1_result
