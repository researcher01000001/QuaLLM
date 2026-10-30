"""Unit tests for HTML renderer utilities"""

from src.core.models import DisplayColumn, Document, Tag
from src.report.html import DocumentProcessor

_EMPTY_DISPLAY_COLUMN = DisplayColumn(source_column="", label="", values={})


class TestEscapeHtml:
    """Tests for DocumentProcessor.escape_html."""

    def test_basic_escaping(self):
        assert (
            DocumentProcessor.escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"
        )

    def test_no_special_chars(self):
        assert DocumentProcessor.escape_html("hello world") == "hello world"

    def test_empty_string(self):
        assert DocumentProcessor.escape_html("") == ""


class TestHexToRgba:
    """Tests for DocumentProcessor.hex_to_rgba."""

    def test_with_hash(self):
        assert DocumentProcessor.hex_to_rgba("#ff0000", 0.5) == "rgba(255, 0, 0, 0.5)"

    def test_without_hash(self):
        assert DocumentProcessor.hex_to_rgba("00ff00", 0.3) == "rgba(0, 255, 0, 0.3)"

    def test_invalid_hex_fallback(self):
        assert DocumentProcessor.hex_to_rgba("zzz", 0.3) == "rgba(128, 128, 128, 0.3)"

    def test_default_alpha(self):
        result = DocumentProcessor.hex_to_rgba("#0000ff")
        assert result == "rgba(0, 0, 255, 0.3)"


class TestGetParagraphBoundaries:
    """Tests for DocumentProcessor.get_row_boundaries."""

    def test_multiple_lines(self):
        text = "line one\nline two\nline three"
        bounds = DocumentProcessor.get_row_boundaries(text)
        assert bounds == [(0, 8), (9, 17), (18, 28)]

    def test_empty_lines_ignored(self):
        text = "line one\n\nline two"
        bounds = DocumentProcessor.get_row_boundaries(text)
        assert bounds == [(0, 8), (10, 18)]

    def test_no_newlines(self):
        text = "single paragraph"
        bounds = DocumentProcessor.get_row_boundaries(text)
        assert bounds == [(0, 16)]

    def test_trailing_newline(self):
        text = "line one\nline two\n"
        bounds = DocumentProcessor.get_row_boundaries(text)
        assert bounds == [(0, 8), (9, 17)]


class TestBuildBackgroundStyle:
    """Tests for DocumentProcessor._build_background_style."""

    def test_empty_list(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        assert proc._build_background_style([]) == "rgba(128, 128, 128, 0.3)"

    def test_single_color(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        assert proc._build_background_style(["#ff0000"]) == "rgba(255, 0, 0, 0.3)"

    def test_multiple_colors(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        result = proc._build_background_style(["#ff0000", "#00ff00"])
        assert "linear-gradient" in result
        assert "rgba(255, 0, 0, 0.3)" in result
        assert "rgba(0, 255, 0, 0.3)" in result


class TestSystemTags:
    """System tags come from Tag.system, not from the tag's group."""

    def test_system_button_lists_only_flagged_tags(self):
        tags = [
            Tag(name="other", group="Other", system=True),
            Tag(name="Grouped", group="system"),
            Tag(name="Monitoring"),
        ]
        tooltip, has_tags = DocumentProcessor._system_tags_button_attrs(tags)
        assert has_tags is True
        assert tooltip == "System tags:\nother"

    def test_modifier_class_follows_system_flag(self):
        assert DocumentProcessor._tag_modifier_classes(Tag(name="other", system=True)) == "system"
        assert DocumentProcessor._tag_modifier_classes(Tag(name="x", group="system")) == ""


class TestUndefinedTagsButtonAttrs:
    """Tests for DocumentProcessor._undefined_tags_button_attrs."""

    def test_no_undefined_tags(self):
        tags = [Tag(name="Monitoring"), Tag(name="Internal Note", system=True)]
        tooltip, has_tags = DocumentProcessor._undefined_tags_button_attrs(tags)
        assert has_tags is False
        assert tooltip == "No undefined tags"

    def test_some_undefined_tags_sorted_in_tooltip(self):
        tags = [
            Tag(name="Zeta", additional_tag=True),
            Tag(name="Alpha", additional_tag=True),
            Tag(name="Monitoring"),
        ]
        tooltip, has_tags = DocumentProcessor._undefined_tags_button_attrs(tags)
        assert has_tags is True
        assert tooltip == "Undefined tags:\nAlpha\nZeta"


class TestRenderParaNumCell:
    """Tests for DocumentProcessor._render_para_num_cell's scores_by_variant handling."""

    @staticmethod
    def _processor() -> DocumentProcessor:
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.para_meta = {}
        proc.documents = {}
        return proc

    def test_no_scores_renders_only_primary_and_meta_spans(self):
        proc = self._processor()
        html = proc._render_para_num_cell(0, "doc")
        assert 'class="para-num-primary"' in html
        assert "data-variant" not in html

    def test_single_variant_omits_data_variant_attribute(self):
        proc = self._processor()
        html = proc._render_para_num_cell(
            0, "doc", scores_by_variant={"including": {"score_lev": 0.5}}
        )
        assert "data-variant" not in html
        assert "0.5000" in html

    def test_two_variants_adds_data_variant_attribute_per_span(self):
        proc = self._processor()
        html = proc._render_para_num_cell(
            0,
            "doc",
            scores_by_variant={
                "including": {"score_lev": 0.5},
                "excluding": {"score_lev": 0.8},
            },
        )
        assert 'data-variant="including"' in html
        assert 'data-variant="excluding"' in html
        assert "0.5000" in html
        assert "0.8000" in html


class TestBuildDocumentIdentity:
    """Tests for DocumentProcessor._build_document_identity."""

    def test_includes_all_template_columns_for_resolvable_name(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        doc_name = "45_no advice_cycle_63"
        proc.documents = {doc_name: Document(source_name=doc_name, rows=[], display_column=_EMPTY_DISPLAY_COLUMN)}

        identity = proc._build_document_identity([doc_name])

        assert identity == {
            doc_name: {
                "think_aloud_id": "45",
                "condition": "no advice",
                "cycle_id": "63",
            }
        }

    def test_skips_documents_not_in_self_documents(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.documents = {}
        assert proc._build_document_identity(["missing_doc"]) == {}

    def test_skips_unresolvable_names(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        doc_name = "not-a-valid-name"
        proc.documents = {doc_name: Document(source_name=doc_name, rows=[], display_column=_EMPTY_DISPLAY_COLUMN)}
        assert proc._build_document_identity([doc_name]) == {}


class TestProcessDocumentDataDocName:
    """Tests for the data-doc-name attribute on process_document()'s output."""

    def test_document_wrapper_carries_data_doc_name(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.documents = {}
        proc.para_meta = {}

        html = proc.process_document("45_no advice_cycle_63", {}, "", [])

        assert 'data-doc-name="45_no advice_cycle_63"' in html


class TestGenerateHtmlSuggestedFilename:
    """Tests for the browser save-name embedded by generate_html()."""

    def test_embeds_output_file_name(self, tmp_path):
        proc = DocumentProcessor(tmp_path, documents={})
        output_file = tmp_path / "comparison_interface_pp2_test.html"

        proc.generate_html({}, output_file, prefix="test")

        html = output_file.read_text(encoding="utf-8")
        assert 'id="suggested-filename">"comparison_interface_pp2_test.html"' in html
