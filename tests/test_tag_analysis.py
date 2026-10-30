"""Unit tests for src.analysis.tag_analysis to achieve span-level match classification
and the tag_analysis/tag_confusion row builders"""

from src.analysis.tag_analysis import (
    ALL_GENERAL_CLUSTERS_LABEL,
    SpanMatchCounts,
    classify_span_matches,
    compute_tag_analysis_rows,
    compute_tag_confusion_rows,
    has_any_quotes,
)
from src.core.models import AtlasQuote, ConsensusQuote, Document, DisplayColumn, Row, Tag


def _document(name: str, text: str) -> Document:
    return Document(
        source_name=name,
        rows=[Row(index=0, text=text, start_position=0, end_position=len(text))],
        display_column=DisplayColumn(
            source_column="position", label="Paragraph", values={0: "1"}
        ),
    )


class TestClassifySpanMatches:
    """classify_span_matches() greedily pairs gold/pred spans by overlap
    ratio (intersection / union) and classifies each pair as exact (ratio
    == 1.0), threshold (>= overlap_threshold), or part (below it)."""

    def test_no_spans(self):
        assert classify_span_matches([], []) == SpanMatchCounts(0, 0, 0, 0, 0, 0, 0)

    def test_zero_overlap_counts_as_unfound_and_additional(self):
        result = classify_span_matches([(0, 5)], [(10, 15)])
        assert result.unfound == 1
        assert result.additional == 1
        assert (result.exact, result.threshold, result.part) == (0, 0, 0)

    def test_identical_spans_are_exact(self):
        result = classify_span_matches([(0, 10)], [(0, 10)])
        assert result.exact == 1
        assert (result.threshold, result.part, result.unfound, result.additional) == (
            0,
            0,
            0,
            0,
        )

    def test_ratio_at_threshold_is_threshold_not_exact(self):
        # gold [0,10) len 10, pred [0,9) len 9 -> intersection 9, union 10 -> ratio 0.9
        result = classify_span_matches([(0, 10)], [(0, 9)], overlap_threshold=0.9)
        assert result.threshold == 1
        assert result.exact == 0
        assert result.part == 0

    def test_ratio_below_threshold_is_part(self):
        # gold [0,10), pred [0,5) -> intersection 5, union 10 -> ratio 0.5
        result = classify_span_matches([(0, 10)], [(0, 5)], overlap_threshold=0.9)
        assert result.part == 1
        assert (result.exact, result.threshold) == (0, 0)

    def test_greedy_matching_prefers_highest_overlap_pairing(self):
        """Two pred spans could each match the first gold span; the greedy
        matcher must take the higher-overlap pairing (0,10)-(0,10) first,
        leaving the second pred span and second gold span unmatched."""
        gold = [(0, 10), (20, 30)]
        pred = [(0, 10), (5, 10)]
        result = classify_span_matches(gold, pred, overlap_threshold=0.9)
        assert result.exact == 1
        assert result.unfound == 1
        assert result.additional == 1
        assert result.part == 0


class TestComputeTagAnalysisRows:
    NORMAL_TAG = Tag(name="Monitoring")
    ADVICE_ONLY_TAG = Tag(name="AdviceOnly", condition="supported")

    def _scenario(self):
        doc1 = _document("1_advice_cycle_1", "abcdefghij")  # len 10
        doc2 = _document("2_no advice_cycle_1", "klmnopqrst")  # len 10
        documents = {doc1.source_name: doc1, doc2.source_name: doc2}

        human_quotes = {
            doc1.source_name: [
                AtlasQuote(text="abcde", tags=[self.NORMAL_TAG, self.ADVICE_ONLY_TAG], start=0, end=5)
            ],
            doc2.source_name: [
                AtlasQuote(text="klmn", tags=[self.NORMAL_TAG], start=0, end=4)
            ],
        }
        ai_quotes_by_configuration = {
            "cfgA": {
                doc1.source_name: [
                    ConsensusQuote(
                        text="abcde", tags=[self.NORMAL_TAG, self.ADVICE_ONLY_TAG], start=0, end=5
                    )
                ],
                doc2.source_name: [
                    ConsensusQuote(text="klmn", tags=[self.NORMAL_TAG], start=0, end=4)
                ],
            }
        }
        return documents, human_quotes, ai_quotes_by_configuration

    def _rows_for_tag(self, rows, tag_name):
        return {row["general_cluster"]: row for row in rows if row["tag"] == tag_name}

    def test_tag_with_no_condition_applies_to_every_cluster(self):
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        rows = compute_tag_analysis_rows(
            documents, human_quotes, ai_quotes_by_configuration, [self.NORMAL_TAG]
        )
        by_cluster = self._rows_for_tag(rows, "Monitoring")
        assert set(by_cluster) == {"advice", "no advice", ALL_GENERAL_CLUSTERS_LABEL}

    def test_condition_restricted_tag_only_appears_where_allowed(self):
        """ADVICE_ONLY_TAG's condition ("supported") is only in the "advice"
        document's allowed set (condition_match() maps advice ->
        general/supported), not the "no advice" document's (general/human)
        -- it must not get a manufactured "no advice" row."""
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        rows = compute_tag_analysis_rows(
            documents, human_quotes, ai_quotes_by_configuration, [self.ADVICE_ONLY_TAG]
        )
        by_cluster = self._rows_for_tag(rows, "AdviceOnly")
        assert set(by_cluster) == {"advice", ALL_GENERAL_CLUSTERS_LABEL}
        assert by_cluster["advice"]["human_tag_count"] == 1
        assert by_cluster[ALL_GENERAL_CLUSTERS_LABEL]["human_tag_count"] == 1

    def test_exact_match_counts_and_character_percentages_per_cluster(self):
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        rows = compute_tag_analysis_rows(
            documents, human_quotes, ai_quotes_by_configuration, [self.NORMAL_TAG]
        )
        by_cluster = self._rows_for_tag(rows, "Monitoring")

        advice_row = by_cluster["advice"]
        assert advice_row["human_tag_count"] == 1
        assert advice_row["ai_tag_count"] == 1
        assert advice_row["exact_match_count"] == 1
        assert (advice_row["threshold_match_count"], advice_row["part_match_count"]) == (0, 0)
        assert (advice_row["unfound_count"], advice_row["additional_count"]) == (0, 0)
        assert advice_row["ai_char_precision"] == 1.0
        assert advice_row["ai_char_recall"] == 1.0
        assert advice_row["ai_char_f1"] == 1.0
        assert advice_row["ai_pct_of_coded_characters"] == 1.0
        assert advice_row["ai_pct_of_all_characters"] == 0.5  # 5 of 10 doc1 characters
        assert advice_row["human_pct_of_all_characters"] == 0.5

        no_advice_row = by_cluster["no advice"]
        assert no_advice_row["human_pct_of_all_characters"] == 0.4  # 4 of 10 doc2 characters

        all_clusters_row = by_cluster[ALL_GENERAL_CLUSTERS_LABEL]
        assert all_clusters_row["human_tag_count"] == 2
        assert all_clusters_row["ai_tag_count"] == 2
        assert all_clusters_row["exact_match_count"] == 2
        # 9 tagged characters (5 + 4) out of 20 total document characters.
        assert all_clusters_row["human_pct_of_all_characters"] == 9 / 20

    def test_configuration_axis_is_preserved(self):
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        ai_quotes_by_configuration["cfgB"] = ai_quotes_by_configuration["cfgA"]
        rows = compute_tag_analysis_rows(
            documents, human_quotes, ai_quotes_by_configuration, [self.NORMAL_TAG]
        )
        configurations = {row["configuration"] for row in rows if row["tag"] == "Monitoring"}
        assert configurations == {"cfgA", "cfgB"}

    def test_undefined_tag_observed_in_a_cluster_gets_a_row_there(self):
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        undefined_tag = Tag(name="Surprise", additional_tag=True)
        human_quotes["1_advice_cycle_1"].append(
            AtlasQuote(text="fgh", tags=[undefined_tag], start=5, end=8)
        )
        rows = compute_tag_analysis_rows(
            documents, human_quotes, ai_quotes_by_configuration, [self.NORMAL_TAG]
        )
        by_cluster = self._rows_for_tag(rows, "Surprise")
        assert set(by_cluster) == {"advice", ALL_GENERAL_CLUSTERS_LABEL}
        assert by_cluster["advice"]["human_tag_count"] == 1
        assert by_cluster["advice"]["unfound_count"] == 1  # AI never coded it

    def test_drop_undefined_tags_excludes_undefined_tag_rows(self):
        documents, human_quotes, ai_quotes_by_configuration = self._scenario()
        undefined_tag = Tag(name="Surprise", additional_tag=True)
        human_quotes["1_advice_cycle_1"].append(
            AtlasQuote(text="fgh", tags=[undefined_tag], start=5, end=8)
        )
        rows = compute_tag_analysis_rows(
            documents,
            human_quotes,
            ai_quotes_by_configuration,
            [self.NORMAL_TAG],
            drop_undefined_tags=True,
        )
        assert "Surprise" not in {row["tag"] for row in rows}


class TestHasAnyQuotes:
    def test_true_when_some_document_has_a_quote(self):
        tag = Tag(name="Monitoring")
        quote = AtlasQuote(text="a", tags=[tag], start=0, end=1)
        assert has_any_quotes({"doc1": [], "doc2": [quote]}) is True

    def test_false_for_empty_dict(self):
        assert has_any_quotes({}) is False

    def test_false_when_every_document_has_no_quotes(self):
        """This is the shape run._slice_to_scope() always produces -- one
        entry per scoped document, defaulting missing ones to [] -- so a
        source mapping that was genuinely empty (e.g. no superconsensus was
        ever built) still ends up as a non-empty dict of empty lists after
        slicing. Plain dict truthiness would wrongly call this "has data"."""
        assert has_any_quotes({"doc1": [], "doc2": []}) is False


class TestComputeTagConfusionRows:
    def test_returns_empty_when_no_superconsensus(self):
        doc = _document("1_advice_cycle_1", "abcde")
        documents = {doc.source_name: doc}
        rows = compute_tag_confusion_rows(documents, {doc.source_name: []}, {})
        assert rows == []

    def test_returns_empty_when_superconsensus_is_a_dict_of_empty_lists(self):
        """Reproduces the shape run._slice_to_scope() produces when no real
        superconsensus was ever built (fewer than two coding configurations):
        a non-empty dict whose every value is an empty list. compute_tag_confusion_rows()
        must treat this the same as a genuinely empty mapping, not proceed
        and silently find nothing to report for the wrong reason."""
        doc1 = _document("1_advice_cycle_1", "abcde")
        doc2 = _document("2_no advice_cycle_1", "fghij")
        documents = {doc1.source_name: doc1, doc2.source_name: doc2}
        human_quotes = {doc1.source_name: [], doc2.source_name: []}
        superconsensus_quotes = {doc1.source_name: [], doc2.source_name: []}

        rows = compute_tag_confusion_rows(documents, human_quotes, superconsensus_quotes)
        assert rows == []

    def test_tallies_character_level_substitution(self):
        human_tag = Tag(name="Monitoring")
        ai_tag = Tag(name="Diagnosis")
        doc = _document("1_advice_cycle_1", "abcde")
        documents = {doc.source_name: doc}
        human_quotes = {
            doc.source_name: [AtlasQuote(text="abcde", tags=[human_tag], start=0, end=5)]
        }
        superconsensus_quotes = {
            doc.source_name: [ConsensusQuote(text="abcde", tags=[ai_tag], start=0, end=5)]
        }

        rows = compute_tag_confusion_rows(documents, human_quotes, superconsensus_quotes)
        by_cluster = {row["general_cluster"]: row for row in rows}

        assert by_cluster["advice"] == {
            "human_tag": "Monitoring",
            "ai_tag": "Diagnosis",
            "general_cluster": "advice",
            "character_count": 5,
        }
        assert by_cluster[ALL_GENERAL_CLUSTERS_LABEL]["character_count"] == 5

    def test_agreeing_characters_produce_no_confusion_row(self):
        tag = Tag(name="Monitoring")
        doc = _document("1_advice_cycle_1", "abcde")
        documents = {doc.source_name: doc}
        human_quotes = {
            doc.source_name: [AtlasQuote(text="abcde", tags=[tag], start=0, end=5)]
        }
        superconsensus_quotes = {
            doc.source_name: [ConsensusQuote(text="abcde", tags=[tag], start=0, end=5)]
        }

        assert compute_tag_confusion_rows(documents, human_quotes, superconsensus_quotes) == []
