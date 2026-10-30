from collections.abc import Mapping, Sequence
from pathlib import Path

import jinja2

from src.config import project as config_project
from src.config.general import SCORE_COLUMNS, UNDEFINED_TAGS_ENABLED
from src.config.project import PARA_META_COLUMNS
from src.core import process_log
from src.core.models import AIQuote, CodedRow, Document, Quote, Tag
from src.dataset.document_identity import parse_document_name


def apply_coder_comparison_scores(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    html_consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
) -> None:
    """Turn each document's rows into CodedRows so the renderer can display
    per-row and per-document agreement scores.

    Compares Human (ATLAS) quotes against the first row of
    ``html_consensus_quotes`` (the superconsensus by default, or the first
    entry of ``HTML_CONSENSUS_CONFIGURATIONS`` when set, in ``run.py``'s
    usage). Mutates ``documents`` in place, which is safe here since this
    runs right before the documents are handed to the HTML renderer.
    """
    if not html_consensus_quotes:
        return

    primary_ai_quotes = next(iter(html_consensus_quotes.values()))

    for doc_name, document in documents.items():
        document.append_quotes(
            list(atlas_quotes.get(doc_name, [])),
            list(primary_ai_quotes.get(doc_name, [])),
        )


def build_render_codings(
    atlas_quotes: dict[str, list[Quote]],
    html_consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
) -> dict[str, dict[str, list[Quote]]]:
    """Build the ``codings`` mapping (coder label -> document -> quotes) that
    ``DocumentProcessor.generate_html()`` expects: one "Human (ATLAS)" column
    plus one "AI consensus ({label})" column per entry in
    ``html_consensus_quotes``.
    """
    codings: dict[str, dict[str, list[Quote]]] = {"Human (ATLAS)": atlas_quotes}
    for label, quotes_by_doc in html_consensus_quotes.items():
        codings[f"AI consensus ({label})"] = {
            doc_name: list(quotes) for doc_name, quotes in quotes_by_doc.items()
        }
    return codings


class DocumentProcessor:
    """Handles HTML generation for coded quotes."""

    def __init__(
        self,
        txt_dir: Path | Sequence[Path],
        para_meta: dict[str, dict[int, dict[str, str]]] | None = None,
        documents: dict[str, Document] | None = None,
    ) -> None:
        """Initialise with the source documents directory.

        Args:
            txt_dir: Directory (or directories, searched in order) containing
                the plain-text source documents.
            para_meta: Optional per-paragraph metadata, keyed by document name
                then by 1-based paragraph position (see
                ``src.dataset.document_content.paragraph_meta()``).
            documents: Optional pre-loaded Document objects keyed by document name.
        """
        self.txt_dirs = [txt_dir] if isinstance(txt_dir, Path) else list(txt_dir)
        self.para_meta = para_meta or {}
        self.documents = documents or {}

        templates_dir = Path(__file__).resolve().parent.parent / "templates"
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(templates_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )
        self._base_template = self._jinja_env.get_template("compare.html")

    def get_document_path(self, doc_name: str) -> Path:
        """Resolve a document name to its plain-text file path.

        Args:
            doc_name: Document name as used in the quotes data.

        Returns:
            Path to the corresponding .txt file.

        Raises:
            FileNotFoundError: If no matching file can be found.
        """
        for txt_dir in self.txt_dirs:
            file_path = txt_dir / f"{doc_name}.txt"
            if file_path.exists():
                return file_path

        raise FileNotFoundError(f"Could not find text file for document: {doc_name}")

    @staticmethod
    def get_row_boundaries(text: str) -> list[tuple[int, int]]:
        """Return the character offsets of each non-empty row (paragraph) in the text.

        Args:
            text: The full document text, with rows separated by newlines.

        Returns:
            List of (start, end) character offsets, one per non-empty row.
            If no newlines are found the whole text is returned as one row.
        """
        row_boundaries = []
        start = 0

        parts = text.split("\n")
        for part in parts:
            end = start + len(part)
            if part.strip():
                row_boundaries.append((start, end))
            start = end + 1  # +1 for the '\n'

        if not row_boundaries:
            row_boundaries.append((0, len(text)))

        return row_boundaries

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
        """Convert a hex colour string to an rgba() CSS value.

        Args:
            hex_color: Six-digit hex colour, with or without a leading "#".
            alpha: Opacity value between 0.0 and 1.0.

        Returns:
            CSS rgba() string, e.g. "rgba(255, 0, 0, 0.3)". Falls back to
            grey if the hex string cannot be parsed.
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return f"rgba(128, 128, 128, {alpha})"
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        return f"rgba({r}, {g}, {b}, {alpha})"

    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML special characters in a plain-text string.

        Args:
            text: Raw text that may contain "&", "<", or ">".

        Returns:
            Text with "&", "<", and ">" replaced by their HTML entities.
        """
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _build_background_style(self, colors: list[str]) -> str:
        """Build a CSS background value for one or more tag colours.

        Args:
            colors: List of hex colour strings. An empty list produces grey.

        Returns:
            A plain rgba() string for a single colour, or a linear-gradient
            with equal-height horizontal stripes for multiple colours.
        """
        if not colors:
            return "rgba(128, 128, 128, 0.3)"
        if len(colors) == 1:
            return self.hex_to_rgba(colors[0], 0.3)

        stripe_height = 100 / len(colors)
        gradient_parts = [
            f"{self.hex_to_rgba(color, 0.3)} {i * stripe_height}%, {self.hex_to_rgba(color, 0.3)} {(i + 1) * stripe_height}%"
            for i, color in enumerate(colors)
        ]
        return f"linear-gradient(to bottom, {', '.join(gradient_parts)})"

    @staticmethod
    def _build_meta_checkboxes() -> str:
        """Build the checkbox list for choosing which metadata to display in
        the paragraph-number column -- a labelled section meant to sit inside
        the toolbar's "View" dropdown panel, not a standalone control.

        Returns:
            HTML string of a section label plus one checkbox per column.
        """
        checkboxes = ['<div class="dropdown-section-label"># column info</div>']
        for col, label in PARA_META_COLUMNS + SCORE_COLUMNS:
            checked = "checked" if col.startswith("score_") else ""
            checkboxes.append(
                f'<label class="dropdown-check-item">'
                f'<input type="checkbox" class="meta-col-checkbox" value="{col}" {checked}> {label}'
                f"</label>"
            )
        return "".join(checkboxes)

    def render_header(self, meta_checkboxes_html: str) -> str:
        """Render the HTML header via the Jinja2 base template.

        Args:
            meta_checkboxes_html: Pre-built HTML for the meta column checkboxes.

        Returns:
            Rendered HTML header string.
        """
        return self._base_template.render(
            meta_checkboxes_html=meta_checkboxes_html,
            legend_html="",
            documents=[],
            system_tags_tooltip="No system tags",
            has_system_tags=False,
            undefined_tags_feature_enabled=False,
            undefined_tags_tooltip="No undefined tags",
            has_undefined_tags=False,
            document_identity={},
            filter_columns=[],
            prefix="",
            display_label=config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        )

    @staticmethod
    def _system_tags_button_attrs(tags: list[Tag]) -> tuple[str, bool]:
        """Build the (tooltip, has_system_tags) pair for the "Show system
        tags" toggle button: the tooltip lists every system tag name found
        in ``tags``, or explains there are none -- the button is disabled
        in that case (see compare.html's #btn-toggle-system).
        """
        system_tag_names = sorted(tag.name for tag in tags if tag.system)
        if not system_tag_names:
            return "No system tags", False
        return "System tags:\n" + "\n".join(system_tag_names), True

    @staticmethod
    def _undefined_tags_button_attrs(tags: list[Tag]) -> tuple[str, bool]:
        """Build the (tooltip, has_undefined_tags) pair for the "Show/Hide
        undefined tags" toggle button: the tooltip lists every undefined tag
        name found in ``tags``, or explains there are none -- the button is
        disabled in that case (see compare.html's #btn-toggle-undefined).
        """
        undefined_tag_names = sorted(tag.name for tag in tags if tag.additional_tag)
        if not undefined_tag_names:
            return "No undefined tags", False
        return "Undefined tags:\n" + "\n".join(undefined_tag_names), True

    @staticmethod
    def _tag_modifier_classes(tag: Tag) -> str:
        """CSS modifier classes for a tag: "undefined" for tags outside the
        codebook (or whose condition doesn't fit their document), "system"
        for Tag.system tags. Both are toggled client-side
        via a body class -- see compare.html's btn-toggle-* buttons."""
        classes = []
        if tag.additional_tag:
            classes.append("undefined")
        if tag.system:
            classes.append("system")
        return " ".join(classes)

    def _build_legend(self, tags: list[Tag]) -> str:
        """Build the tag colour legend HTML block.

        Args:
            tags: All tags to include in the legend.

        Returns:
            HTML string containing a styled legend div with one entry per tag.
        """
        parts = [
            '<div class="legend"><div class="legend-title">Code Legend</div><div class="legend-items">'
        ]
        for tag in sorted(tags, key=lambda t: t.name):
            item_class = f"legend-item {self._tag_modifier_classes(tag)}".strip()
            title = (
                ' title="Not part of the predefined codebook"'
                if tag.additional_tag
                else ""
            )
            parts.append(
                f'<div class="{item_class}"{title}>'
                f'<div class="legend-color" style="background-color: {tag.color};"></div>'
                f'<div class="legend-label">{tag.name}</div>'
                f"</div>"
            )
        parts.append("</div></div>")
        return "".join(parts)

    def _build_tooltip(self, tags: list[Tag]) -> str:
        """Build the hover tooltip HTML for a highlighted quote span.

        Args:
            tags: Tags to display inside the tooltip.

        Returns:
            HTML string for a <span class="quote-tooltip"> element.
        """
        parts = []
        for tag in tags:
            css_class = f"quote-tooltip-tag {self._tag_modifier_classes(tag)}".strip()
            parts.append(
                f'<div class="{css_class}">'
                f'<div class="quote-tooltip-color" data-color="{tag.color}" style="background-color: {tag.color};"></div>'
                f'<div class="quote-tooltip-label">{tag.name}</div></div>'
            )
        tag_html = "".join(parts)
        return f'<span class="quote-tooltip">{tag_html}</span>'

    def _render_paragraph_html(
        self,
        para_text: str,
        para_start: int,
        quotes: list[Quote],
    ) -> str:
        """Render a paragraph as HTML with annotated spans highlighted.

        Splits the paragraph text at annotation boundaries and wraps each
        annotated segment in a <span class="quote"> with a tooltip.

        Args:
            para_text: Plain text of the paragraph.
            para_start: Character offset of the paragraph start in the full document.
            quotes: List of Quote objects overlapping this paragraph.

        Returns:
            HTML string of the paragraph with highlighted spans inserted.
        """
        para_end = para_start + len(para_text)

        boundaries = sorted(
            set(
                [para_start, para_end]
                + [
                    q.start
                    for q in quotes
                    if q.start is not None and para_start <= q.start < para_end
                ]
                + [
                    q.end
                    for q in quotes
                    if q.end is not None and para_start < q.end <= para_end
                ]
            )
        )

        html_parts = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_text = para_text[seg_start - para_start : seg_end - para_start]

            active_quotes = [
                q
                for q in quotes
                if q.start is not None
                and q.end is not None
                and q.start <= seg_start
                and seg_end <= q.end
            ]

            if not active_quotes:
                html_parts.append(self.escape_html(seg_text))
                continue

            tags = list(
                {tag.name: tag for q in active_quotes for tag in q.tags}.values()
            )
            colors = [tag.color or "#808080" for tag in tags]
            bg_style = self._build_background_style(colors)
            tooltip = self._build_tooltip(tags)
            quote_ids = ",".join(str(q.quote_id) for q in active_quotes)
            tag_names = "|".join(self.escape_html(tag.name) for tag in tags)

            html_parts.append(
                f'<span class="quote" data-quote-ids="{quote_ids}" data-tags="{tag_names}" '
                f'style="background: {bg_style};">'
                f"{tooltip}{self.escape_html(seg_text)}</span>"
            )

        return "".join(html_parts)

    def _render_combined_paragraph_html(
        self,
        para_text: str,
        para_start: int,
        coder_quotes: dict[str, list[Quote]],
    ) -> str:
        """Render a paragraph with combined highlights from all coders, styled by source."""
        para_end = para_start + len(para_text)

        labeled_quotes = []
        for label, quotes in coder_quotes.items():
            for q in quotes:
                if (
                    q.start is not None
                    and q.end is not None
                    and q.start < para_end
                    and q.end > para_start
                ):
                    labeled_quotes.append((label, q))

        if not labeled_quotes:
            return self.escape_html(para_text)

        boundaries = sorted(
            set(
                [para_start, para_end]
                + [
                    q.start
                    for _, q in labeled_quotes
                    if q.start is not None and para_start <= q.start < para_end
                ]
                + [
                    q.end
                    for _, q in labeled_quotes
                    if q.end is not None and para_start < q.end <= para_end
                ]
            )
        )

        html_parts = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_text = para_text[seg_start - para_start : seg_end - para_start]

            active = [
                (label, q)
                for label, q in labeled_quotes
                if q.start is not None
                and q.end is not None
                and q.start <= seg_start
                and seg_end <= q.end
            ]

            if not active:
                html_parts.append(self.escape_html(seg_text))
                continue

            active_labels = set(label for label, _ in active)

            tags = list({tag.name: tag for _, q in active for tag in q.tags}.values())
            colors = [tag.color or "#808080" for tag in tags]
            bg_style = self._build_background_style(colors)
            tooltip = self._build_tooltip(tags)
            quote_ids = ",".join(str(q.quote_id) for _, q in active)
            tag_names = "|".join(self.escape_html(tag.name) for tag in tags)

            style_props = [f"background: {bg_style}"]

            all_coders = list(coder_quotes.keys())
            if len(all_coders) >= 2:
                coder1, coder2 = all_coders[0], all_coders[1]
                if coder1 in active_labels and coder2 not in active_labels:
                    style_props.append("text-decoration: underline")
                elif coder2 in active_labels and coder1 not in active_labels:
                    style_props.append("font-weight: bold")

            style_str = "; ".join(style_props)

            html_parts.append(
                f'<span class="quote" data-quote-ids="{quote_ids}" data-tags="{tag_names}" '
                f'style="{style_str};">'
                f"{tooltip}{self.escape_html(seg_text)}</span>"
            )

        return "".join(html_parts)

    def _render_tag_badges(self, tags: list[Tag]) -> str:
        """Render tags as a row of coloured pill badges.

        Args:
            tags: Tags to render. An empty list renders a placeholder dash.

        Returns:
            HTML string of <span class="tag-badge"> elements, or a grey dash
            if tags is empty.
        """
        if not tags:
            return '<span style="color:#bbb;font-size:12px;">—</span>'
        parts = []
        for tag in tags:
            badge_class = f"tag-badge {self._tag_modifier_classes(tag)}".strip()
            title = self.escape_html(tag.name)
            if tag.additional_tag:
                title += " (not in codebook)"
            parts.append(
                f'<span class="{badge_class}" data-tag-name="{title}" data-tag="{self.escape_html(tag.name)}"'
                f' style="background-color: {tag.color};" title="{title}">'
                f"{self.escape_html(tag.name)}</span>"
            )
        return "".join(parts)

    def _render_para_num_cell(
        self,
        row: int,
        doc_name: str,
        scores_by_variant: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """Render the ``<td>`` for the row-display / metadata column.

        The cell always contains a ``para-num-primary`` span with the row's
        display value (see ``Document.display_value()``) plus one
        ``para-num-meta`` span per metadata column so that the JavaScript
        selector can show/hide the right one without any round-trips.

        Args:
            row: Zero-based technical row index.
            doc_name: Document name, used to look up ``self.documents`` and
                ``self.para_meta``.
            scores_by_variant: Calculated scores for this row, keyed by
                drop_undefined_tags variant ("including"/"excluding"). A
                ``data-variant`` attribute is only added to the emitted spans
                when there are two variants -- with a single variant (the
                undefined-tags feature disabled) the markup is identical to
                a plain, un-variant-aware ``scores`` dict.

        Returns:
            HTML string for the complete ``<td class="col-para-num">`` element.
        """
        document = self.documents.get(doc_name)
        display_value = document.display_value(row) if document else ""

        # PARA_META_COLUMNS is keyed by output.xlsx's raw DOCUMENT_ORDER_COLUMN value
        # (see document_content.paragraph_meta()); since DOCUMENT_ORDER_COLUMN is also
        # display_value's source_column, parsing it back to an int recovers
        # the same join key without a separate row -> position conversion.
        try:
            position_key: int | None = int(display_value)
        except ValueError:
            position_key = None

        meta_row: dict[str, str] = (
            self.para_meta.get(doc_name, {}).get(position_key, {})
            if position_key is not None
            else {}
        )

        # Fold toggle + primary display value + reviewed checkbox -- JS reads
        # the enclosing <tr>'s data-row and the enclosing .document's
        # data-doc-name to key these.
        parts = [
            f'<button class="para-fold-toggle" type="button" aria-label="Toggle {config_project.DOCUMENT_ORDER_COLUMN_LABEL}"></button>',
            f'<span class="para-num-primary">{self.escape_html(display_value)}</span>',
            '<label class="para-reviewed">'
            '<input type="checkbox" class="para-reviewed-checkbox"> Reviewed'
            "</label>",
        ]

        # One hidden span per metadata column
        for col, label in PARA_META_COLUMNS:
            value = self.escape_html(meta_row.get(col, ""))
            parts.append(
                f'<span class="para-num-meta" data-col="{col}"><b>{label}:</b> {value}</span>'
            )

        # Add score columns, one set of spans per variant
        if scores_by_variant:
            multi_variant = len(scores_by_variant) > 1
            for variant, scores in scores_by_variant.items():
                variant_attr = f' data-variant="{variant}"' if multi_variant else ""
                for col, label in SCORE_COLUMNS:
                    value = f"{scores.get(col, 0.0):.4f}"
                    parts.append(
                        f'<span class="para-num-meta" data-col="{col}"{variant_attr}>'
                        f"<b>{label}:</b> {value}</span>"
                    )

        return f'<td class="col-para-num">{"".join(parts)}</td>'

    def process_document(
        self,
        doc_name: str,
        doc_codings: dict[str, list[Quote]],
        text: str,
        row_boundaries: list[tuple[int, int]],
    ) -> str:
        """Render a single document as a scrollable paragraph comparison table.

        Each table row corresponds to one document row (paragraph). Each
        coder gets a column showing the row's text with their own
        highlights and tag badges below.

        Args:
            doc_name: Display name shown in the table header.
            doc_codings: Mapping of coder label to that coder's quotes for
                this document.
            text: Full plain text of the document.
            row_boundaries: List of (start, end) character offsets, one per
                row.

        Returns:
            HTML string for the complete document <div> including its table.
        """
        # Ensure all quotes have positions and IDs
        document = self.documents.get(doc_name)
        quote_id = 0
        for label, quotes in doc_codings.items():
            for quote in quotes:
                quote.quote_id = quote_id
                quote_id += 1
                if isinstance(quote, AIQuote):
                    if (
                        quote.row is not None
                        and 0 <= quote.row < len(row_boundaries)
                    ):
                        p_start, p_end = row_boundaries[quote.row]
                        p_text = text[p_start:p_end]
                        quote.find_position(p_text, p_start, document=document)
                else:
                    quote.find_position(text, 0, document=document)

        coder_labels = list(doc_codings.keys())

        html_parts = [
            f'<div class="document" data-doc-name="{self.escape_html(doc_name)}">',
            '<div class="document-title">'
            '<button class="doc-fold-toggle" type="button" aria-label="Toggle document"></button>'
            f'<span class="doc-title-text">{self.escape_html(doc_name)}</span>'
            '<span class="doc-progress"></span>'
            "</div>",
            '<table class="comparison-table">',
            "<thead><tr>",
            f'<th class="col-para-num">{self.escape_html(config_project.DOCUMENT_ORDER_COLUMN_LABEL)}</th>',
            '<th class="col-combined" style="display: none;">Combined View</th>',
            *[
                f'<th class="col-coder-header">{self.escape_html(label)}</th>'
                for label in coder_labels
            ],
            "</tr></thead>",
            "<tbody>",
        ]

        for row, (para_start, para_end) in enumerate(row_boundaries):
            para_text = text[para_start:para_end]

            # Filter quotes belonging to this row
            para_quotes_by_coder = {}
            for label in coder_labels:
                para_quotes_by_coder[label] = [
                    q
                    for q in doc_codings[label]
                    if q.start is not None
                    and q.end is not None
                    and q.start < para_end
                    and q.end > para_start
                ]

            # Calculate scores for this row if we have a Document object
            scores_by_variant: dict[str, dict[str, float]] = {}
            if doc_name in self.documents:
                doc = self.documents[doc_name]
                if row < len(doc.rows):
                    coded_row = doc.rows[row]
                    if isinstance(coded_row, CodedRow):
                        variants = (
                            [("including", False), ("excluding", True)]
                            if UNDEFINED_TAGS_ENABLED
                            else [("including", False)]
                        )
                        for variant_name, drop_undefined_tags in variants:
                            lev, jaccard, f1, char_f1_result = coded_row.compare_coders(
                                True, drop_undefined_tags
                            )
                            # Values line up positionally with SCORE_COLUMNS
                            # (src/config/general.py) -- the single source of truth
                            # for these score keys, also used by
                            # _build_meta_checkboxes()/_render_para_num_cell().
                            score_values = (
                                lev,
                                jaccard,
                                f1,
                                char_f1_result["original"]["agreement"],
                                char_f1_result["original"]["char_f1"],
                                char_f1_result["pruned"]["agreement"],
                                char_f1_result["pruned"]["char_f1"],
                            )
                            scores_by_variant[variant_name] = dict(
                                zip((key for key, _ in SCORE_COLUMNS), score_values, strict=True)
                            )

            row_tag_names = "|".join(
                dict.fromkeys(
                    self.escape_html(tag.name)
                    for quotes in para_quotes_by_coder.values()
                    for q in quotes
                    for tag in q.tags
                )
            )
            html_parts.append(f'<tr data-row="{row}" data-tags="{row_tag_names}">')
            html_parts.append(
                self._render_para_num_cell(
                    row, doc_name, scores_by_variant=scores_by_variant
                )
            )

            # Render combined cell
            combined_html = self._render_combined_paragraph_html(
                para_text, para_start, para_quotes_by_coder
            )
            used_tags_combined = []
            for label in coder_labels:
                for q in para_quotes_by_coder[label]:
                    used_tags_combined.extend(q.tags)

            html_parts.append(
                f'<td class="col-combined" style="display: none;">'
                f'<div class="coder-text">{combined_html}</div>'
                f'<div class="coder-badges">{self._render_tag_badges(used_tags_combined)}</div>'
                f"</td>"
            )

            for label in coder_labels:
                para_quotes = para_quotes_by_coder[label]
                used_tags = [
                    tag
                    for q in sorted(para_quotes, key=lambda q: q.start or 0)
                    for tag in q.tags
                ]

                para_html = self._render_paragraph_html(
                    para_text, para_start, para_quotes
                )
                html_parts.append(
                    f'<td class="col-coder">'
                    f'<div class="coder-text">{para_html}</div>'
                    f'<div class="coder-badges">{self._render_tag_badges(used_tags)}</div>'
                    f"</td>"
                )

            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div>")
        return "".join(html_parts)

    def _build_document_score_block(
        self, doc: Document, variant_name: str | None, drop_undefined_tags: bool
    ) -> str:
        """Build the ``<div class="document-scores">`` block for one
        drop_undefined_tags variant. ``variant_name`` is None when the
        undefined-tags feature is disabled (single variant, no
        ``data-variant`` attribute -- byte-identical to the pre-feature
        markup); otherwise it is "including" or "excluding".
        """
        (
            avg_lev,
            avg_jaccard,
            avg_f1,
            original_char_f1,
            original_char_agreement,
            pruned_char_f1,
            pruned_char_agreement,
        ) = doc.compare_coders(True, drop_undefined_tags)
        variant_attr = f' data-variant="{variant_name}"' if variant_name else ""
        return (
            f'<div class="document-scores"{variant_attr}>'
            f"<span><b>Avg Levenshtein:</b> {avg_lev:.4f}</span>"
            f"<span><b>Avg Jaccard:</b> {avg_jaccard:.4f}</span>"
            f"<span><b>Avg F1:</b> {avg_f1:.4f}</span>"
            f"<span><b>Avg Original Agreement:</b> {original_char_agreement:.4f}</span>"
            f"<span><b>Avg Original Char F1:</b> {original_char_f1:.4f}</span>"
            f"<span><b>Avg Pruned Agreement:</b> {pruned_char_agreement:.4f}</span>"
            f"<span><b>Avg Pruned Char F1:</b> {pruned_char_f1:.4f}</span>"
            f"</div>"
        )

    def _build_document_identity(self, doc_names: list[str]) -> dict[str, dict[str, str]]:
        """Resolve each of doc_names' identity (one column per
        config_project.DOCUMENT_NAME_TEMPLATE placeholder, e.g.
        think_aloud_id/condition/cycle_id) for the client-side filter panel.
        Documents not in self.documents, or whose name doesn't resolve, are
        skipped -- they have nothing to filter by. Values are stringified
        since the client-side filter code compares them against DOM
        dataset-attribute strings when restoring saved state.
        """
        identity: dict[str, dict[str, str]] = {}
        for doc_name in doc_names:
            document = self.documents.get(doc_name)
            if document is None:
                continue
            parsed = parse_document_name(document.source_name)
            if parsed is not None:
                identity[doc_name] = {col: str(value) for col, value in parsed.items()}
        return identity

    def generate_html(
        self,
        codings: dict[str, dict[str, list[Quote]]],
        output_file: Path,
        prefix: str = "",
    ) -> None:
        """Generate and write the full HTML comparison file.

        Iterates over all documents referenced by any coder, renders each as a
        comparison table, and writes the result to disk.

        Args:
            codings: Mapping of coder label to a mapping of document name to that coder's quote list.
            output_file: Destination path for the generated HTML file.
            prefix: Dataset prefix (config.PREFIX), used client-side as the
                localStorage key for the page's filter/fold/reviewed state.
        """
        all_tags = list(
            {
                tag.name: tag
                for doc_quotes in codings.values()
                for quotes in doc_quotes.values()
                for quote in quotes
                for tag in quote.tags
            }.values()
        )
        legend_html = self._build_legend(all_tags)
        meta_checkboxes_html = self._build_meta_checkboxes()
        system_tags_tooltip, has_system_tags = self._system_tags_button_attrs(all_tags)
        undefined_tags_tooltip, has_undefined_tags = self._undefined_tags_button_attrs(
            all_tags
        )

        # Determine the documents for which quotes are available
        seen: set[str] = set()
        all_doc_names: list[str] = []
        for doc_quotes in codings.values():
            for doc_name in doc_quotes.keys():
                if doc_name not in seen:
                    seen.add(doc_name)
                    all_doc_names.append(doc_name)

        # Sorted {name, color} list for the HTML report's tag-select filter
        # (compare.html's "Tags Filter" dropdown) -- same source as legend_html,
        # just serialized for client-side use instead of rendered as markup.
        tag_filter_options = [
            {"name": tag.name, "color": tag.color or "#808080"}
            for tag in sorted(all_tags, key=lambda tag: tag.name)
        ]

        document_identity = self._build_document_identity(all_doc_names)
        # Filter columns come from the parsed identity dicts' keys (insertion
        # order matches parse_document_name()'s column order); empty when no
        # document name resolved, in which case there's nothing to filter.
        first_identity = next(iter(document_identity.values()), {})
        filter_columns = [{"col": col, "label": col} for col in first_identity]

        # Render each document into a list of HTML fragments
        documents: list[str] = []
        for doc_name in all_doc_names:
            if doc_name in self.documents:
                doc = self.documents[doc_name]
                text = "\n".join(r.text for r in doc.rows)
                row_boundaries = [
                    (r.start_position, r.end_position) for r in doc.rows
                ]

                # Calculate document-level scores only when this renderer was
                # given a two-coder CodedRow document.
                if any(
                    isinstance(row, CodedRow)
                    for row in doc.rows
                ):
                    try:
                        if UNDEFINED_TAGS_ENABLED:
                            doc_score_html = self._build_document_score_block(
                                doc, "including", False
                            ) + self._build_document_score_block(
                                doc, "excluding", True
                            )
                        else:
                            doc_score_html = self._build_document_score_block(
                                doc, None, False
                            )
                    except Exception:
                        doc_score_html = ""
                else:
                    doc_score_html = ""
            else:
                doc_score_html = ""
                try:
                    txt_file = self.get_document_path(doc_name)
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    process_log.current().log_missing_source_document(
                        severity="Warning",
                        source_name=doc_name,
                        context="html_rendering",
                        message=str(e),
                    )
                    continue

                # Calculate row boundaries for this document
                row_boundaries = self.get_row_boundaries(text)

            # Reduce codings to those relevant for this document
            doc_codings: dict[str, list[Quote]] = {
                label: doc_quotes.get(doc_name, [])
                for label, doc_quotes in codings.items()
            }

            # Render the document's comparison table
            doc_html = self.process_document(
                doc_name, doc_codings, text, row_boundaries
            )
            if doc_score_html:
                # Insert scores after the title
                doc_html = doc_html.replace(
                    "</div><table", f"</div>{doc_score_html}<table", 1
                )
            documents.append(doc_html)

        # Render the full page via Jinja2
        html = self._base_template.render(
            meta_checkboxes_html=meta_checkboxes_html,
            legend_html=legend_html,
            documents=documents,
            system_tags_tooltip=system_tags_tooltip,
            has_system_tags=has_system_tags,
            undefined_tags_feature_enabled=UNDEFINED_TAGS_ENABLED,
            undefined_tags_tooltip=undefined_tags_tooltip,
            has_undefined_tags=has_undefined_tags,
            document_identity=document_identity,
            filter_columns=filter_columns,
            tag_filter_options=tag_filter_options,
            prefix=prefix,
            suggested_filename=output_file.name,
            display_label=config_project.DOCUMENT_ORDER_COLUMN_LABEL,
        )

        # Write the complete HTML to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"HTML file created: {output_file}")
