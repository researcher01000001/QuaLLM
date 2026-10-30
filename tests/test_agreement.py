"""Unit tests for agreement metrics."""

from src.analysis.agreement import (
    character_level_f1,
    f1_score,
    jaccard_similarity,
    normalized_levenshtein_distance,
)
from src.core.models import AIQuote, AtlasQuote, CodedRow, Row, Tag


class TestNormalizedLevenshteinDistance:
    """Tests for normalized_levenshtein_distance."""

    def test_identical_strings(self):
        assert normalized_levenshtein_distance("hello", "hello") == 0.0

    def test_completely_different(self):
        assert normalized_levenshtein_distance("abc", "xyz") == 1.0

    def test_partial_overlap(self):
        result = normalized_levenshtein_distance("kitten", "sitting")
        assert 0.0 < result < 1.0

    def test_empty_strings(self):
        assert normalized_levenshtein_distance("", "") == 0.0

    def test_one_empty(self):
        assert normalized_levenshtein_distance("hello", "") == 1.0
        assert normalized_levenshtein_distance("", "hello") == 1.0


class TestJaccardSimilarity:
    """Tests for jaccard_similarity."""

    def test_identical_sets(self):
        tags = [Tag("A", "#fff"), Tag("B", "#000")]
        assert jaccard_similarity(tags, tags) == 1.0

    def test_no_overlap(self):
        t1 = [Tag("A", "#fff")]
        t2 = [Tag("B", "#000")]
        assert jaccard_similarity(t1, t2) == 0.0

    def test_partial_overlap(self):
        t1 = [Tag("A", "#fff"), Tag("B", "#000")]
        t2 = [Tag("B", "#000"), Tag("C", "#f00")]
        assert jaccard_similarity(t1, t2) == 1 / 3

    def test_both_empty(self):
        assert jaccard_similarity([], []) == 1.0


class TestF1Score:
    """Tests for f1_score."""

    def test_identical(self):
        tags = [Tag("A", "#fff")]
        assert f1_score(tags, tags) == 1.0

    def test_no_overlap(self):
        t1 = [Tag("A", "#fff")]
        t2 = [Tag("B", "#000")]
        assert f1_score(t1, t2) == 0.0

    def test_perfect_precision_no_recall(self):
        t1 = [Tag("A", "#fff"), Tag("B", "#000")]
        t2 = []
        assert f1_score(t1, t2) == 0.0

    def test_both_empty(self):
        assert f1_score([], []) == 0.0


class TestCharacterLevelF1:
    """Tests for character_level_f1."""

    def test_exact_match(self):
        """Two identical quotes covering the same span should yield F1 = 1.0."""
        tag = Tag("Monitoring", "#ff0000")
        q1 = AtlasQuote(text="abc", tags=[tag], start=0, end=3)
        q2 = AIQuote(text="abc", tags=[tag], row=0)
        q2.start = 0
        q2.end = 3

        result = character_level_f1([q1], [q2], "abc", 0)
        assert result["original"]["agreement"] == 1.0
        assert result["original"]["char_f1"] == 1.0
        assert result["pruned"]["agreement"] == 1.0
        assert result["pruned"]["char_f1"] == 1.0

    def test_no_overlap(self):
        """Quotes on disjoint spans with different tags should yield F1 = 0.0."""
        t1 = Tag("A", "#fff")
        t2 = Tag("B", "#000")
        q1 = AtlasQuote(text="abc", tags=[t1], start=0, end=3)
        q2 = AIQuote(text="def", tags=[t2], row=0)
        q2.start = 4
        q2.end = 7

        result = character_level_f1([q1], [q2], "abc def", 0)
        assert result["original"]["char_f1"] == 0.0
        assert result["pruned"]["char_f1"] == 0.0

    def test_partial_overlap_same_tag(self):
        """Overlapping spans with the same tag should yield intermediate F1."""
        tag = Tag("Monitoring", "#ff0000")
        q1 = AtlasQuote(text="abcd", tags=[tag], start=0, end=4)
        q2 = AIQuote(text="cdef", tags=[tag], row=0)
        q2.start = 2
        q2.end = 6

        result = character_level_f1([q1], [q2], "abcdef", 0)
        assert 0.0 < result["original"]["char_f1"] < 1.0
        assert 0.0 < result["pruned"]["char_f1"] < 1.0

    def test_empty_quotes(self):
        """Empty predictions and empty gold should return perfect agreement."""
        result = character_level_f1([], [], "text", 0)
        assert result["original"]["agreement"] == 1.0
        assert result["original"]["char_f1"] == 0.0
        assert result["pruned"]["agreement"] == 1.0
        assert result["pruned"]["char_f1"] == 0.0

    def test_returns_expected_score_keys(self):
        """The result should contain only agreement and char_f1 per mode."""
        result = character_level_f1([], [], "abc", 0)
        assert set(result) == {"original", "pruned"}
        assert set(result["original"]) == {"agreement", "char_f1"}
        assert set(result["pruned"]) == {"agreement", "char_f1"}

    def test_pruned_scores_ignore_whitespace_dots_and_commas(self):
        """Pruned metrics should ignore whitespace, dots, and commas."""
        tag = Tag("A", "#fff")
        q1 = AtlasQuote(text="a, b.", tags=[tag], start=0, end=5)
        q2 = AIQuote(text="a", tags=[tag], row=0)
        q2.start = 0
        q2.end = 1
        q3 = AIQuote(text="b", tags=[tag], row=0)
        q3.start = 3
        q3.end = 4

        result = character_level_f1([q1], [q2, q3], "a, b.", 0)
        assert result["original"]["agreement"] < 1.0
        assert result["original"]["char_f1"] < 1.0
        assert result["pruned"]["agreement"] == 1.0
        assert result["pruned"]["char_f1"] == 1.0


class TestCodedRowComparison:
    """Integration-level tests for CodedRow.compare_coders."""

    def test_perfect_agreement(self, coded_paragraph: CodedRow):
        """When both coders tag the same span with the same tag, metrics should be 1.0."""
        lev, jaccard, f1, char_f1 = coded_paragraph.compare_coders()
        assert lev == 0.0
        assert jaccard == 1.0
        assert f1 == 1.0
        assert char_f1["original"]["agreement"] == 1.0
        assert char_f1["original"]["char_f1"] == 1.0
        assert char_f1["pruned"]["agreement"] == 1.0
        assert char_f1["pruned"]["char_f1"] == 1.0

    def test_empty_coders(self):
        """When both coders have no quotes, metrics should reflect emptiness."""
        para = Row(
            index=0,
            text="some text here",
            start_position=0,
            end_position=14,
        )
        coded = CodedRow(
            index=para.index,
            text=para.text,
            start_position=para.start_position,
            end_position=para.end_position,
            quotes_1=[],
            quotes_2=[],
        )
        lev, jaccard, f1, char_f1 = coded.compare_coders()
        assert lev == 0.0
        assert jaccard == 1.0
        assert f1 == 0.0
        assert char_f1["original"]["char_f1"] == 0.0
        assert char_f1["pruned"]["char_f1"] == 0.0


class TestCodedRowCompareCodersExcludeSystemTags:
    """exclude_system_tags drops Tag.system tags (the coding scheme's
    ``system`` column) from every calculation."""

    @staticmethod
    def _coded_paragraph() -> CodedRow:
        """Both coders agree on "patient shows", but coder 1 additionally
        attached a system tag that coder 2 didn't -- a mismatch unless
        that tag is excluded."""
        text = "The patient shows clear signs of improvement."
        normal_tag = Tag(name="Monitoring")
        system_tag = Tag(name="Internal Note", system=True)

        q1 = AtlasQuote(text="patient shows", tags=[normal_tag, system_tag], start=4, end=17)
        q2 = AtlasQuote(text="patient shows", tags=[normal_tag], start=4, end=17)

        return CodedRow(
            index=0,
            text=text,
            start_position=0,
            end_position=len(text),
            quotes_1=[q1],
            quotes_2=[q2],
        )

    def test_without_exclusion_system_tag_causes_mismatch(self):
        coded = self._coded_paragraph()
        _, jaccard, f1, _ = coded.compare_coders()
        assert jaccard < 1.0
        assert f1 < 1.0

    def test_excluding_system_tags_restores_perfect_agreement(self):
        coded = self._coded_paragraph()
        lev, jaccard, f1, char_f1 = coded.compare_coders(exclude_system_tags=True)
        assert lev == 0.0
        assert jaccard == 1.0
        assert f1 == 1.0
        assert char_f1["original"]["agreement"] == 1.0
        assert char_f1["original"]["char_f1"] == 1.0

    def test_exclusion_does_not_mutate_original_quotes(self):
        """quotes_excluding_system() copies quotes rather than mutating them,
        so the paragraph's own quotes_1/quotes_2 keep the system tag."""
        coded = self._coded_paragraph()
        coded.compare_coders(exclude_system_tags=True)
        assert len(coded.quotes_1[0].tags) == 2


class TestCodedRowCompareCodersDropUndefinedTags:
    """exclude_system_tags and drop_undefined_tags are independent knobs, each
    catching only the tags matching its own criterion (see
    models.compare_coders)."""

    @staticmethod
    def _coded_paragraph() -> CodedRow:
        """Coder 1 additionally attached one undefined-only tag and one
        system-only tag; coder 2 didn't. Neither exclusion alone removes
        both, so only applying both together restores agreement."""
        text = "The patient shows clear signs of improvement."
        normal_tag = Tag(name="Monitoring")
        undefined_only_tag = Tag(name="Stray", additional_tag=True)
        system_only_tag = Tag(name="Internal Note", system=True)

        q1 = AtlasQuote(
            text="patient shows",
            tags=[normal_tag, undefined_only_tag, system_only_tag],
            start=4,
            end=17,
        )
        q2 = AtlasQuote(text="patient shows", tags=[normal_tag], start=4, end=17)

        return CodedRow(
            index=0,
            text=text,
            start_position=0,
            end_position=len(text),
            quotes_1=[q1],
            quotes_2=[q2],
        )

    def test_without_either_exclusion_extra_tags_cause_mismatch(self):
        coded = self._coded_paragraph()
        _, jaccard, f1, _ = coded.compare_coders()
        assert jaccard < 1.0
        assert f1 < 1.0

    def test_exclude_system_tags_alone_still_mismatches_on_undefined_tag(self):
        coded = self._coded_paragraph()
        _, jaccard, f1, _ = coded.compare_coders(exclude_system_tags=True)
        assert jaccard < 1.0
        assert f1 < 1.0

    def test_drop_undefined_tags_alone_still_mismatches_on_system_tag(self):
        coded = self._coded_paragraph()
        _, jaccard, f1, _ = coded.compare_coders(drop_undefined_tags=True)
        assert jaccard < 1.0
        assert f1 < 1.0

    def test_both_exclusions_together_restore_perfect_agreement(self):
        coded = self._coded_paragraph()
        lev, jaccard, f1, char_f1 = coded.compare_coders(
            exclude_system_tags=True, drop_undefined_tags=True
        )
        assert lev == 0.0
        assert jaccard == 1.0
        assert f1 == 1.0
        assert char_f1["original"]["agreement"] == 1.0
        assert char_f1["original"]["char_f1"] == 1.0

    def test_exclusions_do_not_mutate_original_quotes(self):
        coded = self._coded_paragraph()
        coded.compare_coders(exclude_system_tags=True, drop_undefined_tags=True)
        assert len(coded.quotes_1[0].tags) == 3
