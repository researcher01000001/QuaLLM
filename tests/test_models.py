import pandas as pd

from src.core import process_log
from src.core.models import (
    AIQuote,
    AtlasQuote,
    CodingScheme,
    Tag,
    quotes_excluding_system,
    quotes_excluding_undefined,
    tags_for_condition,
)


def test_ai_quote_find_position_uses_quote_directly_when_unique():
    """Step 1: a unique quote is used as-is, context is irrelevant."""
    tag = Tag("Monitoring")
    search_text = "Das ist einzigartig hier im Text."
    quote = AIQuote(
        text="einzigartig",
        tags=[tag],
        row=0,
        context="hier steht nichts passendes",
    )

    expected_start = search_text.index("einzigartig")

    assert quote.find_position(search_text) == (
        expected_start,
        expected_start + len("einzigartig"),
    )
    assert quote.tags == [tag]


def test_ai_quote_find_position_uses_context_to_disambiguate_repeated_phrase():
    """Steps 2-4: quote repeats, but a unique context resolves it uniquely."""
    tag = Tag("Monitoring")
    search_text = "ja das ist links und später ist ja rechts relevant"
    quote = AIQuote(
        text="ja",
        tags=[tag],
        row=0,
        context="später ist ja rechts relevant",
    )

    expected_start = search_text.index("ja", search_text.index("später"))

    assert quote.find_position(search_text) == (expected_start, expected_start + 2)
    # A confidently resolved position leaves the tag untouched.
    assert quote.tags == [tag]


def test_ai_quote_find_position_flags_unsure_when_quote_repeats_in_unique_context():
    """Step 5: the context is unique, but the quote still repeats inside it
    -- the first occurrence within the context is used, and the tag is
    renamed to flag the uncertain position."""
    tag = Tag("Monitoring")
    search_text = "vorher ja mal ja und ja wieder ja am ende ja nachher"
    quote = AIQuote(
        text="ja",
        tags=[tag],
        row=0,
        context="ja und ja wieder ja am ende ja",
    )

    context_start = search_text.index("ja und ja wieder ja am ende ja")
    expected_start = search_text.index("ja", context_start)

    assert quote.find_position(search_text) == (expected_start, expected_start + 2)
    assert [t.name for t in quote.tags] == ["position_unsure_Monitoring"]
    assert quote.tags[0].additional_tag is True


def test_ai_quote_find_position_flags_unsure_when_context_is_not_unique():
    """Step 6: the context itself repeats -- the first context occurrence is
    used, and the tag is renamed to flag the uncertain position."""
    tag = Tag("Monitoring")
    search_text = "a mitte ja b mitte ja c mitte ja d"
    quote = AIQuote(
        text="ja",
        tags=[tag],
        row=0,
        context="mitte ja",
    )

    first_context_start = search_text.index("mitte ja")
    expected_start = search_text.index("ja", first_context_start)

    assert quote.find_position(search_text) == (expected_start, expected_start + 2)
    assert [t.name for t in quote.tags] == ["position_unsure_Monitoring"]
    assert quote.tags[0].additional_tag is True


def test_ai_quote_find_position_flags_unsure_when_context_is_missing():
    """Step 6 (context not found in the text at all): falls back to the
    first raw occurrence of the quote and flags the position as unsure."""
    tag = Tag("Monitoring")
    search_text = "ja vorne und ja hinten"
    quote = AIQuote(
        text="ja",
        tags=[tag],
        row=0,
        context="nicht vorhanden ja",
    )

    assert quote.find_position(search_text) == (0, 2)
    assert [t.name for t in quote.tags] == ["position_unsure_Monitoring"]
    assert quote.tags[0].additional_tag is True


def test_ai_quote_find_position_unsure_logs_to_process_log(tmp_path):
    tag = Tag("Monitoring")
    quote = AIQuote(
        text="ja",
        tags=[tag],
        row=1,
        context="nicht vorhanden ja",
    )

    log_path = tmp_path / "process_log.xlsx"
    log = process_log.ProcessLog(log_path, step="run_main")
    with process_log.active(log):
        result = quote.find_position("ja vorne und ja hinten")
    log.flush()

    assert result == (0, 2)
    df = pd.read_excel(log_path, sheet_name=process_log.QUOTE_POSITION_ISSUES)
    assert len(df) == 1
    assert df.iloc[0]["issue_type"] == "position_unsure"
    assert df.iloc[0]["severity"] == "Error"
    assert df.iloc[0]["row"] == 1


def test_ai_quote_find_position_not_found_logs_to_process_log(tmp_path):
    tag = Tag("Monitoring")
    quote = AIQuote(text="nonexistent phrase", tags=[tag], row=2)

    log_path = tmp_path / "process_log.xlsx"
    log = process_log.ProcessLog(log_path, step="run_main")
    with process_log.active(log):
        result = quote.find_position("completely different text")
    log.flush()

    assert result is None
    df = pd.read_excel(log_path, sheet_name=process_log.QUOTE_POSITION_ISSUES)
    assert len(df) == 1
    assert df.iloc[0]["issue_type"] == "not_found"
    assert df.iloc[0]["row"] == 2
    # No Document was passed to find_position(), so the display value is
    # unresolved (None -> NaN after the Excel round-trip).
    assert pd.isna(df.iloc[0]["Paragraph"])


class TestTagsForCondition:
    """Tests for tags_for_condition(). valid_conditions is the coding-scheme
    condition list for one document (see document_identity.condition_match())."""

    def test_tag_without_condition_always_kept(self):
        tag = Tag(name="A")
        result = tags_for_condition([tag], ["supported"])
        assert result == [tag]

    def test_tag_with_condition_in_list_kept(self):
        tag = Tag(name="A", condition="supported")
        result = tags_for_condition([tag], ["general", "supported"])
        assert result == [tag]

    def test_tag_with_condition_not_in_list_dropped(self):
        tag = Tag(name="A", condition="supported")
        result = tags_for_condition([tag], ["general", "human"])
        assert result == []

    def test_empty_valid_conditions_keeps_only_unconditional_tags(self):
        tag_a = Tag(name="A", condition="supported")
        tag_b = Tag(name="B")
        result = tags_for_condition([tag_a, tag_b], [])
        assert result == [tag_b]


class TestQuotesExcludingSystem:
    """Tests for quotes_excluding_system() (Tag.system, from the coding
    scheme's ``system`` column)."""

    def test_false_disables_filtering(self):
        quote = AtlasQuote(text="x", tags=[Tag(name="A", system=True)], start=0, end=1)
        result = quotes_excluding_system([quote], False)
        assert result == [quote]
        assert result[0].tags == quote.tags

    def test_strips_only_system_tags(self):
        kept = Tag(name="A")
        dropped = Tag(name="B", system=True)
        quote = AtlasQuote(text="x", tags=[kept, dropped], start=0, end=1)

        result = quotes_excluding_system([quote], True)

        assert len(result) == 1
        assert result[0].tags == [kept]

    def test_quote_with_only_system_tags_kept_with_empty_tags(self):
        quote = AtlasQuote(text="x", tags=[Tag(name="A", system=True)], start=0, end=1)

        result = quotes_excluding_system([quote], True)

        assert len(result) == 1
        assert result[0].tags == []

    def test_group_named_system_is_not_a_system_tag(self):
        """Only the system flag counts; a group value of "system" does not."""
        tag = Tag(name="A", group="system", system=False)
        quote = AtlasQuote(text="x", tags=[tag], start=0, end=1)

        result = quotes_excluding_system([quote], True)

        assert result[0].tags == [tag]


class TestQuotesExcludingUndefined:
    """Tests for quotes_excluding_undefined() (see config.UNDEFINED_TAGS_ENABLED)."""

    def test_false_disables_filtering(self):
        quote = AtlasQuote(
            text="x", tags=[Tag(name="A", additional_tag=True)], start=0, end=1
        )
        result = quotes_excluding_undefined([quote], False)
        assert result == [quote]

    def test_strips_only_additional_tags(self):
        kept = Tag(name="A")
        dropped = Tag(name="B", additional_tag=True)
        quote = AtlasQuote(text="x", tags=[kept, dropped], start=0, end=1)

        result = quotes_excluding_undefined([quote], True)

        assert len(result) == 1
        assert result[0].tags == [kept]

    def test_quote_with_only_additional_tags_kept_with_empty_tags(self):
        quote = AtlasQuote(
            text="x", tags=[Tag(name="A", additional_tag=True)], start=0, end=1
        )

        result = quotes_excluding_undefined([quote], True)

        assert len(result) == 1
        assert result[0].tags == []

    def test_quote_without_additional_tags_passes_through_unchanged(self):
        tag = Tag(name="A")
        quote = AtlasQuote(text="x", tags=[tag], start=0, end=1)

        result = quotes_excluding_undefined([quote], True)

        assert result[0].tags == [tag]


def _coding_scheme_entry(**overrides) -> dict:
    entry = {
        "key": "Indicator: Anomaly",
        "code_category": "Indicator",
        "code": "Anomaly",
        "group": "Group A",
        "definition": "A definition.",
        "note": "A note.",
        "examples": ["ex 1", "ex 2"],
        "condition": "general",
        "double_coding": True,
        "system": False,
    }
    entry.update(overrides)
    return entry


class TestCodingSchemeGeneralRulesText:
    def test_joins_rules_one_per_line(self):
        scheme = CodingScheme(
            coding_scheme=[], general_rules=["Rule one.", "Rule two."], medical_knowledge={}
        )

        assert scheme.general_rules_text() == "Rule one.\nRule two."

    def test_empty_rules_gives_empty_string(self):
        scheme = CodingScheme(coding_scheme=[], general_rules=[], medical_knowledge={})

        assert scheme.general_rules_text() == ""


class TestCodingSchemeMedicalKnowledgeText:
    def test_all_columns_are_headed_when_more_than_one(self):
        scheme = CodingScheme(
            coding_scheme=[],
            general_rules=[],
            medical_knowledge={"healthy": ["A", "B"], "abnormal": ["C"]},
        )

        text = scheme.medical_knowledge_text()

        assert text == "healthy:\n- A\n- B\nabnormal:\n- C"

    def test_specific_column_omits_header_when_only_one_selected(self):
        scheme = CodingScheme(
            coding_scheme=[],
            general_rules=[],
            medical_knowledge={"healthy": ["A", "B"], "abnormal": ["C"]},
        )

        text = scheme.medical_knowledge_text(["healthy"])

        assert text == "- A\n- B"

    def test_unknown_column_yields_nothing_for_that_column(self):
        scheme = CodingScheme(
            coding_scheme=[], general_rules=[], medical_knowledge={"healthy": ["A"]}
        )

        assert scheme.medical_knowledge_text(["missing"]) == ""


class TestCodingSchemeCodingSchemeText:
    def test_no_flags_gives_bare_key_per_entry(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A"), _coding_scheme_entry(key="B")],
            general_rules=[],
            medical_knowledge={},
        )

        assert scheme.coding_scheme_text() == "A\nB"

    def test_definition_flag_appends_after_colon(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A", definition="Def A")],
            general_rules=[],
            medical_knowledge={},
        )

        assert scheme.coding_scheme_text(definition=True) == "<tag>A</tag>: Def A"

    def test_unflagged_columns_are_never_included(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A", note="A note.")],
            general_rules=[],
            medical_knowledge={},
        )

        # note is present on the entry but its flag defaults to False.
        assert scheme.coding_scheme_text() == "A"

    def test_multiple_flags_appear_in_fixed_order(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A")],
            general_rules=[],
            medical_knowledge={},
        )

        text = scheme.coding_scheme_text(
            definition=True, note=True, examples=True, double_coding=True
        )

        assert text == (
            "<tag>A</tag>: A definition.; (Note: A note.); Examples: ex 1, ex 2; "
            "Can co-occur with other tags"
        )

    def test_falsy_definition_is_skipped_but_double_coding_false_still_renders(self):
        """A truly-absent value (``definition=None``) is skipped, but
        ``double_coding=False`` is a meaningful boolean, not an absent value
        -- it always renders one of the two double-coding messages."""
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A", definition=None, double_coding=False)],
            general_rules=[],
            medical_knowledge={},
        )

        assert scheme.coding_scheme_text(definition=True, double_coding=True) == (
            "<tag>A</tag>: Cannot co-occur with other tags that also cannot co-occur."
        )

    def test_entry_filter_excludes_non_matching_entries(self):
        scheme = CodingScheme(
            coding_scheme=[
                _coding_scheme_entry(key="A", condition="general"),
                _coding_scheme_entry(key="B", condition="supported"),
            ],
            general_rules=[],
            medical_knowledge={},
        )

        text = scheme.coding_scheme_text(
            entry_filter=lambda e: e["condition"] in ("general", "human")
        )

        assert text == "A"

    def test_entry_filter_none_keeps_every_entry(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A"), _coding_scheme_entry(key="B")],
            general_rules=[],
            medical_knowledge={},
        )

        assert scheme.coding_scheme_text(entry_filter=None) == "A\nB"

    def test_embedded_newlines_in_a_cell_are_joined_with_comma(self):
        scheme = CodingScheme(
            coding_scheme=[_coding_scheme_entry(key="A", definition="Line 1\nLine 2")],
            general_rules=[],
            medical_knowledge={},
        )

        assert scheme.coding_scheme_text(definition=True) == "<tag>A</tag>: Line 1, Line 2"
