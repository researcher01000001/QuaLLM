from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from src.analysis.metrics import DocumentMetricResult
from src.config.coding import CodingConfiguration
from src.io.io_utils import write_workbook_sheets, resolve_output_dir
from src.report.xlsx_rows import MetricsWorkbookData, build_metrics_workbook_data


TAG_ANALYSIS_HEADER_NOTES: dict[str, str] = {
    "tag": "Tag name.",
    "configuration": "Configuration name, or the superconsensus label.",
    "general_cluster": (
        "One GENERAL_CLUSTER_COLUMN value this tag is valid for, or "
        "'all_general_clusters' for the roll-up across all of them."
    ),
    "human_tag_count": "Number of human-coded spans carrying this tag, in scope.",
    "ai_tag_count": "Number of AI/consensus-coded spans carrying this tag, in scope.",
    "exact_match_count": "Matched spans whose boundaries are character-for-character identical.",
    "threshold_match_count": "Matched spans overlapping >=90% but not identical.",
    "part_match_count": "Matched spans overlapping less than 90%.",
    "unfound_count": "Human coded this tag here; AI did not.",
    "additional_count": "AI coded this tag here; human did not.",
    "ai_char_precision": (
        "Character-level precision scoring the AI's characters against the "
        "human's, for this tag only."
    ),
    "ai_char_recall": (
        "Character-level recall scoring the AI's characters against the "
        "human's, for this tag only."
    ),
    "ai_char_f1": (
        "Character-level F1 scoring the AI's characters against the human's, "
        "for this tag only."
    ),
    "human_char_precision": (
        "Character-level precision scoring the human's characters against "
        "the AI's, for this tag only (precision/recall swapped relative to "
        "the ai_char_* columns)."
    ),
    "human_char_recall": (
        "Character-level recall scoring the human's characters against the "
        "AI's, for this tag only."
    ),
    "human_char_f1": (
        "Character-level F1 scoring the human's characters against the AI's, "
        "for this tag only."
    ),
    "ai_pct_of_coded_characters": (
        "This tag's AI-coded characters divided by all AI-coded characters "
        "(any tag), in scope."
    ),
    "ai_pct_of_all_characters": (
        "This tag's AI-coded characters divided by total characters in "
        "scope (coded or not)."
    ),
    "human_pct_of_coded_characters": (
        "This tag's human-coded characters divided by all human-coded "
        "characters (any tag), in scope."
    ),
    "human_pct_of_all_characters": (
        "This tag's human-coded characters divided by total characters in "
        "scope (coded or not)."
    ),
}

TAG_CONFUSION_HEADER_NOTES: dict[str, str] = {
    "human_tag": "Tag the human applied.",
    "ai_tag": "Different tag the superconsensus applied to the same character(s).",
    "general_cluster": (
        "One GENERAL_CLUSTER_COLUMN value, or 'all_general_clusters' for the "
        "roll-up across all of them."
    ),
    "character_count": "Number of characters where this substitution occurs.",
}


def export_metrics_workbook(
    metric_results: list[DocumentMetricResult],
    output_file: Path,
    coding_configurations: Iterable[CodingConfiguration] | None = None,
    iteration_count: int | None = None,
    consensus_progression_scores: Mapping[str, Mapping[int, float]] | None = None,
    tag_analysis_rows: list[dict[str, object]] | None = None,
    tag_confusion_rows: list[dict[str, object]] | None = None,
) -> None:
    """Build and write the metrics workbook."""
    resolve_output_dir(output_file)
    workbook_data = build_metrics_workbook_data(
        metric_results=metric_results,
        coding_configurations=coding_configurations,
        iteration_count=iteration_count,
        consensus_progression_scores=consensus_progression_scores,
        tag_analysis_rows=tag_analysis_rows,
        tag_confusion_rows=tag_confusion_rows,
    )
    write_metrics_workbook(workbook_data, output_file)


def write_metrics_workbook(
    workbook_data: MetricsWorkbookData,
    output_file: Path,
) -> None:
    """Write pre-built metrics workbook rows to XLSX."""
    write_workbook_sheets(
        {
            "overview": pd.DataFrame(workbook_data.overview_rows),
            "document_metrics": pd.DataFrame(workbook_data.document_metric_rows),
            "row_metrics": pd.DataFrame(workbook_data.paragraph_metric_rows),
            "iteration_metrics": pd.DataFrame(workbook_data.iteration_metric_rows),
            "consensus_progression": pd.DataFrame(workbook_data.consensus_progression_rows),
            "comparison": pd.DataFrame(workbook_data.comparison_rows),
            "metadata": pd.DataFrame(workbook_data.metadata_rows),
            "tag_analysis": pd.DataFrame(workbook_data.tag_analysis_rows),
            "tag_confusion": pd.DataFrame(workbook_data.tag_confusion_rows),
        },
        output_file,
        header_notes={
            "tag_analysis": TAG_ANALYSIS_HEADER_NOTES,
            "tag_confusion": TAG_CONFUSION_HEADER_NOTES,
        },
    )
