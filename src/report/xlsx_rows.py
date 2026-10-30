from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from src.analysis.metrics import DocumentMetricResult, RowMetricResult
from src.config import project as config_project
from src.config.coding import CodingConfiguration

METRIC_COLUMNS = [
    "avg_lev_distance",
    "avg_jaccard",
    "avg_f1",
    "avg_original_char_f1",
    "avg_original_char_agreement",
    "avg_pruned_char_f1",
    "avg_pruned_char_agreement",
]

PARAGRAPH_METRIC_COLUMNS = [
    "lev_distance",
    "jaccard",
    "f1",
    "original_char_f1",
    "original_char_agreement",
    "pruned_char_f1",
    "pruned_char_agreement",
]

CONFIG_METADATA_COLUMNS = [
    "configuration_name",
    "short_name",
    "model_name",
    "temperature",
    "seed",
    "provider_names",
    "max_threads",
    "iterations",
    "consensus_threshold",
    "system_prompt_name",
    "user_prompt_name",
]
assert set(CONFIG_METADATA_COLUMNS) <= set(CodingConfiguration.METADATA_FIELDS), (
    "CONFIG_METADATA_COLUMNS references a field CodingConfiguration.METADATA_FIELDS "
    "doesn't have -- keep the two in sync (see CodingConfiguration.to_metadata())."
)


@dataclass(frozen=True)
class MetricScoreTables:
    """Metric scores split into mean-aggregation and consensus-aggregation paths."""

    all_document_scores: list[DocumentMetricResult]
    mean_source_document_scores: list[DocumentMetricResult]
    consensus_document_scores: list[DocumentMetricResult]
    all_paragraph_scores: list[RowMetricResult]
    mean_source_paragraph_scores: list[RowMetricResult]
    consensus_paragraph_scores: list[RowMetricResult]


@dataclass(frozen=True)
class MetricsWorkbookData:
    """Rows for each metrics workbook sheet before writing to XLSX."""

    overview_rows: list[dict[str, object]]
    document_metric_rows: list[dict[str, object]]
    paragraph_metric_rows: list[dict[str, object]]
    iteration_metric_rows: list[dict[str, object]]
    consensus_progression_rows: list[dict[str, object]]
    comparison_rows: list[dict[str, object]]
    metadata_rows: list[dict[str, object]]
    tag_analysis_rows: list[dict[str, object]]
    tag_confusion_rows: list[dict[str, object]]


def build_metrics_workbook_data(
    metric_results: list[DocumentMetricResult],
    coding_configurations: Iterable[CodingConfiguration] | None = None,
    iteration_count: int | None = None,
    consensus_progression_scores: Mapping[str, Mapping[int, float]] | None = None,
    tag_analysis_rows: list[dict[str, object]] | None = None,
    tag_confusion_rows: list[dict[str, object]] | None = None,
) -> MetricsWorkbookData:
    """Build XLSX sheet rows from metric scores.

    The workbook has two explicit aggregation paths:
    - mean columns aggregate individual run scores;
    - consensus columns score prediction-level consensuses directly.

    tag_analysis_rows/tag_confusion_rows are computed upstream (see
    src.analysis.tag_analysis.compute_tag_analysis_rows()/compute_tag_confusion_rows(),
    called from run.py) and passed through as-is -- unlike every other sheet
    here, their computation isn't derived from metric_results, so this
    function only shapes them into the workbook, it doesn't build them.
    """
    score_tables = _split_metric_scores(metric_results)
    config_metadata = _config_metadata_by_name(metric_results, coding_configurations)

    return MetricsWorkbookData(
        overview_rows=_overview_rows(score_tables, config_metadata, iteration_count),
        document_metric_rows=_document_metric_rows(score_tables),
        paragraph_metric_rows=_paragraph_metric_rows(score_tables),
        iteration_metric_rows=_iteration_metric_rows(score_tables),
        consensus_progression_rows=_consensus_progression_rows(
            consensus_progression_scores,
            config_metadata,
            iteration_count,
        ),
        comparison_rows=[],
        metadata_rows=list(config_metadata.values()),
        tag_analysis_rows=list(tag_analysis_rows or []),
        tag_confusion_rows=list(tag_confusion_rows or []),
    )


def _split_metric_scores(
    metric_results: list[DocumentMetricResult],
) -> MetricScoreTables:
    paragraph_scores = _all_paragraph_results(metric_results)
    return MetricScoreTables(
        all_document_scores=metric_results,
        mean_source_document_scores=[
            result for result in metric_results if not result.is_consensus
        ],
        consensus_document_scores=[
            result for result in metric_results if result.is_consensus
        ],
        all_paragraph_scores=paragraph_scores,
        mean_source_paragraph_scores=[
            paragraph for paragraph in paragraph_scores if not paragraph.is_consensus
        ],
        consensus_paragraph_scores=[
            paragraph for paragraph in paragraph_scores if paragraph.is_consensus
        ],
    )


def _overview_rows(
    score_tables: MetricScoreTables,
    config_metadata: dict[str, dict[str, Any]],
    iteration_count: int | None,
) -> list[dict[str, object]]:
    rows = []
    all_documents_by_config = _group_documents_by_configuration(
        score_tables.all_document_scores
    )
    mean_documents_by_config = _group_documents_by_configuration(
        score_tables.mean_source_document_scores
    )
    consensus_documents_by_config = _group_documents_by_configuration(
        score_tables.consensus_document_scores
    )
    all_paragraphs_by_config = _group_paragraphs_by_configuration(
        score_tables.all_paragraph_scores
    )

    for configuration_name in config_metadata:
        config_results = all_documents_by_config.get(configuration_name, [])
        iteration_results = mean_documents_by_config.get(configuration_name, [])
        consensus_results = consensus_documents_by_config.get(configuration_name, [])

        row = _metadata_row(config_metadata[configuration_name])
        row.update(
            {
                "document_count": len(
                    {result.document_name for result in config_results}
                ),
                "paragraph_count": len(
                    {
                        (paragraph.document_name, paragraph.row)
                        for paragraph in all_paragraphs_by_config.get(
                            configuration_name,
                            [],
                        )
                    }
                ),
                "iteration_count": _iteration_count_for_configuration(
                    config_metadata[configuration_name],
                    iteration_results,
                    iteration_count,
                ),
            }
        )
        row.update(
            _mean_and_consensus_metric_columns(
                iteration_results,
                consensus_results,
                METRIC_COLUMNS,
                current_level="document",
            )
        )
        rows.append(row)
    return rows


def _document_metric_rows(
    score_tables: MetricScoreTables,
) -> list[dict[str, object]]:
    rows = []
    mean_by_key = _group_documents_by_configuration_and_document(
        score_tables.mean_source_document_scores
    )
    consensus_by_key = _group_documents_by_configuration_and_document(
        score_tables.consensus_document_scores
    )

    for configuration_name, document_name in sorted(
        mean_by_key.keys() | consensus_by_key.keys()
    ):
        row: dict[str, object] = {
            "configuration_name": configuration_name,
            "document": document_name,
        }
        row.update(
            _mean_and_consensus_metric_columns(
                mean_by_key.get((configuration_name, document_name), []),
                consensus_by_key.get((configuration_name, document_name), []),
                METRIC_COLUMNS,
                current_level="document",
            )
        )
        rows.append(row)
    return rows


def _paragraph_metric_rows(
    score_tables: MetricScoreTables,
) -> list[dict[str, object]]:
    rows = []
    mean_by_key = _group_paragraphs_by_configuration_document_and_paragraph(
        score_tables.mean_source_paragraph_scores
    )
    consensus_by_key = _group_paragraphs_by_configuration_document_and_paragraph(
        score_tables.consensus_paragraph_scores
    )

    for configuration_name, document_name, row_number in sorted(
        mean_by_key.keys() | consensus_by_key.keys()
    ):
        mean_scores = mean_by_key.get((configuration_name, document_name, row_number), [])
        consensus_scores = consensus_by_key.get(
            (configuration_name, document_name, row_number), []
        )
        display_value = (mean_scores or consensus_scores)[0].display_value
        row = {
            "configuration_name": configuration_name,
            "document": document_name,
            "row": row_number,
            config_project.DOCUMENT_ORDER_COLUMN_LABEL: display_value,
        }
        row.update(
            _mean_and_consensus_metric_columns(
                mean_scores,
                consensus_scores,
                PARAGRAPH_METRIC_COLUMNS,
                current_level="paragraph",
            )
        )
        rows.append(row)
    return rows


def _iteration_metric_rows(
    score_tables: MetricScoreTables,
) -> list[dict[str, object]]:
    rows = []
    for paragraph in sorted(
        score_tables.mean_source_paragraph_scores,
        key=lambda paragraph: (
            paragraph.configuration_name,
            paragraph.document_name,
            paragraph.row,
            paragraph.iteration or 0,
        ),
    ):
        row = {
            "configuration_name": paragraph.configuration_name,
            "document": paragraph.document_name,
            "row": paragraph.row,
            config_project.DOCUMENT_ORDER_COLUMN_LABEL: paragraph.display_value,
            "iteration": paragraph.iteration,
        }
        row.update(
            {metric: getattr(paragraph, metric) for metric in PARAGRAPH_METRIC_COLUMNS}
        )
        rows.append(row)
    return rows


def _consensus_progression_rows(
    consensus_progression_scores: Mapping[str, Mapping[int, float]] | None,
    config_metadata: dict[str, dict[str, Any]],
    iteration_count: int | None,
) -> list[dict[str, object]]:
    rows = []
    if consensus_progression_scores is None:
        return rows

    max_iteration = (
        iteration_count - 1
        if iteration_count is not None
        else max(
            (
                iteration
                for scores_by_iteration in consensus_progression_scores.values()
                for iteration in scores_by_iteration
            ),
            default=-1,
        )
    )

    for configuration_name in config_metadata:
        if configuration_name not in consensus_progression_scores:
            continue

        scores_by_iteration = consensus_progression_scores[configuration_name]
        row: dict[str, object] = {"configuration_name": configuration_name}
        for iteration in range(max_iteration + 1):
            row[f"consensus_0_to_{iteration}"] = scores_by_iteration.get(iteration)
        rows.append(row)
    return rows


def _mean_and_consensus_metric_columns(
    mean_source_scores: list[DocumentMetricResult] | list[RowMetricResult],
    consensus_scores: list[DocumentMetricResult] | list[RowMetricResult],
    metric_columns: list[str],
    current_level: str,
) -> dict[str, object]:
    row: dict[str, object] = {}
    mean_source_scores_by_iteration: dict[
        int,
        list[DocumentMetricResult | RowMetricResult],
    ] = defaultdict(list)
    for result in mean_source_scores:
        if result.iteration is not None:
            mean_source_scores_by_iteration[result.iteration].append(result)

    for metric in metric_columns:
        mean_path_values = [
            _average([float(getattr(result, metric)) for result in results])
            for _, results in sorted(mean_source_scores_by_iteration.items())
        ]
        consensus_path_values = [
            float(getattr(result, metric)) for result in consensus_scores
        ]
        row[_metric_column_name(metric, current_level, "mean", "mean")] = (
            _average(mean_path_values) if mean_path_values else None
        )
        row[_metric_column_name(metric, current_level, "sd", "mean")] = (
            _std(mean_path_values) if mean_path_values else None
        )
        row[
            _metric_column_name(
                metric,
                current_level,
                _consensus_aggregation_for_level(current_level),
                "consensus",
            )
        ] = _average(consensus_path_values) if consensus_path_values else None
    return row


def _group_documents_by_configuration(
    scores: list[DocumentMetricResult],
) -> dict[str, list[DocumentMetricResult]]:
    grouped: dict[str, list[DocumentMetricResult]] = defaultdict(list)
    for score in scores:
        grouped[score.configuration_name].append(score)
    return grouped


def _group_paragraphs_by_configuration(
    scores: list[RowMetricResult],
) -> dict[str, list[RowMetricResult]]:
    grouped: dict[str, list[RowMetricResult]] = defaultdict(list)
    for score in scores:
        grouped[score.configuration_name].append(score)
    return grouped


def _group_documents_by_configuration_and_document(
    scores: list[DocumentMetricResult],
) -> dict[tuple[str, str], list[DocumentMetricResult]]:
    grouped: dict[tuple[str, str], list[DocumentMetricResult]] = defaultdict(list)
    for score in scores:
        grouped[(score.configuration_name, score.document_name)].append(score)
    return grouped


def _group_paragraphs_by_configuration_document_and_paragraph(
    scores: list[RowMetricResult],
) -> dict[tuple[str, str, int], list[RowMetricResult]]:
    grouped: dict[tuple[str, str, int], list[RowMetricResult]] = defaultdict(list)
    for score in scores:
        grouped[
            (
                score.configuration_name,
                score.document_name,
                score.row,
            )
        ].append(score)
    return grouped


def _metric_column_name(
    metric: str,
    current_level: str,
    current_level_aggregation: str,
    first_level_aggregation: str,
) -> str:
    return (
        f"{_metric_label(metric)}_"
        f"{current_level}-{current_level_aggregation}_{first_level_aggregation}"
    )


def _metric_label(metric: str) -> str:
    return metric.removeprefix("avg_").replace("_", "-")


def _consensus_aggregation_for_level(current_level: str) -> str:
    return "consensus" if current_level == "paragraph" else "mean"


def _iteration_count_for_configuration(
    metadata: dict[str, Any],
    mean_source_document_scores: list[DocumentMetricResult],
    iteration_count: int | None,
) -> int:
    if iteration_count is not None:
        return iteration_count

    configured_iterations = metadata.get("iterations")
    if isinstance(configured_iterations, int):
        return configured_iterations

    return len({result.iteration for result in mean_source_document_scores})


def _all_paragraph_results(
    metric_results: list[DocumentMetricResult],
) -> list[RowMetricResult]:
    return [
        paragraph for result in metric_results for paragraph in result.row_metrics
    ]


def _config_metadata_by_name(
    metric_results: list[DocumentMetricResult],
    coding_configurations: Iterable[CodingConfiguration] | None,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if coding_configurations is not None:
        metadata.update(
            {
                config.configuration_name: _normalize_metadata(config.to_metadata())
                for config in coding_configurations
            }
        )

    for configuration_name in dict.fromkeys(
        result.configuration_name for result in metric_results
    ):
        metadata.setdefault(
            configuration_name,
            {"configuration_name": configuration_name},
        )
    return metadata


def _metadata_row(metadata: dict[str, Any]) -> dict[str, object]:
    return {column: metadata.get(column) for column in CONFIG_METADATA_COLUMNS}


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    provider_names = metadata.get("provider_names")
    if isinstance(provider_names, list):
        metadata["provider_names"] = ", ".join(provider_names)
    return metadata


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0
