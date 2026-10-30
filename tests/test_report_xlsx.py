import pandas as pd
import pytest

from src.analysis.metrics import DocumentMetricResult, RowMetricResult
from src.config.coding import CodingConfiguration
from src.report.xlsx import export_metrics_workbook


def test_export_writes_new_metric_sheet_structure(tmp_path):
    output_file = tmp_path / "metrics.xlsx"
    config = CodingConfiguration(
        short_name="sonnet-t0",
        iterations=2,
        consensus_threshold=0.75,
    )
    metrics = [
        _metric_result("doc-1", "sonnet-t0", 0, False, 0.2, 0.5),
        _metric_result("doc-1", "sonnet-t0", 1, False, 0.4, 0.7),
        _metric_result("doc-1", "sonnet-t0", None, True, 0.1, 0.8),
    ]

    export_metrics_workbook(
        metrics,
        output_file,
        coding_configurations=[config],
        consensus_progression_scores={"sonnet-t0": {0: 0.8, 1: 0.9}},
    )

    workbook = pd.ExcelFile(output_file)
    assert workbook.sheet_names == [
        "overview",
        "document_metrics",
        "row_metrics",
        "iteration_metrics",
        "consensus_progression",
        "comparison",
        "metadata",
        "tag_analysis",
        "tag_confusion",
    ]

    overview = pd.read_excel(output_file, sheet_name="overview")
    overview_row = overview.iloc[0]
    assert overview_row["configuration_name"] == "sonnet-t0"
    assert overview_row["document_count"] == 1
    assert overview_row["paragraph_count"] == 1
    assert overview_row["iteration_count"] == 2
    assert overview_row["iterations"] == 2
    assert overview_row["consensus_threshold"] == pytest.approx(0.75)
    assert "iteration_mean_avg_f1" not in overview.columns
    assert overview_row["f1_document-mean_mean"] == pytest.approx(0.6)
    assert overview_row["f1_document-sd_mean"] == pytest.approx(0.1)
    assert overview_row["f1_document-mean_consensus"] == pytest.approx(0.8)
    assert overview_row["lev-distance_document-mean_mean"] == pytest.approx(0.3)
    assert overview_row["lev-distance_document-sd_mean"] == pytest.approx(0.1)
    assert overview_row["lev-distance_document-mean_consensus"] == pytest.approx(0.1)

    paragraph_metrics = pd.read_excel(output_file, sheet_name="row_metrics")
    paragraph_row = paragraph_metrics.iloc[0]
    assert paragraph_row["document"] == "doc-1"
    assert paragraph_row["row"] == 1
    # pandas infers a numeric dtype on read-back since every value in the
    # column looks numeric, regardless of the in-memory display_value being a str.
    assert paragraph_row["Paragraph"] == 2
    assert "iteration_mean_f1" not in paragraph_metrics.columns
    assert paragraph_row["f1_paragraph-mean_mean"] == pytest.approx(0.6)
    assert paragraph_row["f1_paragraph-sd_mean"] == pytest.approx(0.1)
    assert paragraph_row["f1_paragraph-consensus_consensus"] == pytest.approx(0.8)
    assert paragraph_row["lev-distance_paragraph-mean_mean"] == pytest.approx(0.3)
    assert paragraph_row["lev-distance_paragraph-sd_mean"] == pytest.approx(0.1)
    assert paragraph_row["lev-distance_paragraph-consensus_consensus"] == pytest.approx(
        0.1
    )

    iteration_metrics = pd.read_excel(output_file, sheet_name="iteration_metrics")
    assert list(iteration_metrics["iteration"]) == [0, 1]
    assert list(iteration_metrics["f1"]) == [0.5, 0.7]

    progression = pd.read_excel(output_file, sheet_name="consensus_progression")
    progression_row = progression.iloc[0]
    assert progression_row["configuration_name"] == "sonnet-t0"
    assert progression_row["consensus_0_to_0"] == pytest.approx(0.8)
    assert progression_row["consensus_0_to_1"] == pytest.approx(0.9)


def test_export_leaves_unavailable_consensus_progression_cells_blank(tmp_path):
    output_file = tmp_path / "metrics.xlsx"
    short_config = CodingConfiguration(short_name="short", iterations=1)
    long_config = CodingConfiguration(short_name="long", iterations=2)

    export_metrics_workbook(
        [],
        output_file,
        coding_configurations=[short_config, long_config],
        consensus_progression_scores={
            "short": {0: 0.5},
            "long": {0: 0.0, 1: 0.7},
        },
    )

    progression = pd.read_excel(output_file, sheet_name="consensus_progression")
    short_row = progression[progression["configuration_name"] == "short"].iloc[0]
    long_row = progression[progression["configuration_name"] == "long"].iloc[0]

    assert short_row["consensus_0_to_0"] == pytest.approx(0.5)
    assert pd.isna(short_row["consensus_0_to_1"])
    assert long_row["consensus_0_to_0"] == pytest.approx(0.0)
    assert long_row["consensus_0_to_1"] == pytest.approx(0.7)


def test_export_keeps_superconsensus_metrics_out_of_progression(tmp_path):
    output_file = tmp_path / "metrics.xlsx"
    config = CodingConfiguration(short_name="config", iterations=1)
    metrics = [
        _metric_result("doc-1", "config", None, True, 0.2, 0.7),
        _metric_result("doc-1", "superconsensus", None, True, 0.1, 0.8),
    ]

    export_metrics_workbook(
        metrics,
        output_file,
        coding_configurations=[config],
        consensus_progression_scores={"config": {0: 0.7}},
    )

    overview = pd.read_excel(output_file, sheet_name="overview")
    assert set(overview["configuration_name"]) == {"config", "superconsensus"}
    superconsensus_row = overview[
        overview["configuration_name"] == "superconsensus"
    ].iloc[0]
    assert pd.isna(superconsensus_row["f1_document-mean_mean"])
    assert pd.isna(superconsensus_row["f1_document-sd_mean"])
    assert superconsensus_row["f1_document-mean_consensus"] == pytest.approx(0.8)

    progression = pd.read_excel(output_file, sheet_name="consensus_progression")
    assert list(progression["configuration_name"]) == ["config"]


def _metric_result(
    document_name: str,
    configuration_name: str,
    iteration: int | None,
    is_consensus: bool,
    avg_lev_distance: float,
    avg_f1: float,
) -> DocumentMetricResult:
    return DocumentMetricResult(
        document_name=document_name,
        configuration_name=configuration_name,
        iteration=iteration,
        is_consensus=is_consensus,
        avg_lev_distance=avg_lev_distance,
        avg_jaccard=0.0,
        avg_f1=avg_f1,
        avg_original_char_f1=0.0,
        avg_original_char_agreement=0.0,
        avg_pruned_char_f1=0.0,
        avg_pruned_char_agreement=0.0,
        row_metrics=[
            RowMetricResult(
                document_name=document_name,
                configuration_name=configuration_name,
                iteration=iteration,
                is_consensus=is_consensus,
                row=1,
                display_value="2",
                lev_distance=avg_lev_distance,
                jaccard=0.0,
                f1=avg_f1,
                original_char_f1=0.0,
                original_char_agreement=0.0,
                pruned_char_f1=0.0,
                pruned_char_agreement=0.0,
            )
        ],
    )
