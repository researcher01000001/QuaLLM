import dataclasses
import webbrowser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.analysis import tag_analysis
from src.analysis.metrics import DocumentMetricResult, calculate_metrics_for_quote_sets
from src.coding.ai_coding import process_documents, render_general_prompt, write_prompt_snapshots
from src.coding.consensus import (
    build_consensus_progression_quote_sets,
    build_consensus_records,
    build_superconsensus_quotes,
    log_double_coding_conflicts,
    write_consensus_json,
)
from src.coding.prompt_builders import DocumentPromptScheme
from src.config.coding import CodingConfiguration
from src.config.general import (
    AI_CODING_FILES,
    CODING_CONFIGURATIONS,
    COMPARISON_PREFIXES,
    CONSENSUS_FILENAME,
    COST_ESTIMATION_ENABLED,
    COST_ESTIMATION_RECALC_THRESHOLD_FRACTION,
    HTML_CONSENSUS_CONFIGURATIONS,
    OUTPUT_HTML_FILE,
    OUTPUT_XLSX_FILE_EXCLUDING_UNDEFINED_TAGS,
    OUTPUT_XLSX_FILE_INCLUDING_UNDEFINED_TAGS,
    PREFIX,
    PROCESS_LOG_FILE,
    RESULTS_SUBDIR,
    RUN_AI_CODING,
    RUN_TYPE,
    SUPERCONSENSUS_NAME,
    SUPERCONSENSUS_THRESHOLD,
    UNDEFINED_TAGS_ENABLED,
    scoped_output_html,
    scoped_output_xlsx_excluding,
    scoped_output_xlsx_including,
    run_results_dir,
)
from src.config.project import (
    DESCRIPTION_FILE,
    HUMAN_CODING_FILES,
)
from src.core import process_log
from src.core.models import Document, Quote, Tag
from src.cost_estimation import calibration as cost_calibration, estimation as cost_estimation
from src.cost_estimation.pricing import fetch_model_pricing
from src.dataset import run_scope
from src.dataset.document_content import known_conditions, paragraph_meta
from src.dataset.document_identity import condition_for, condition_match
from src.io.io_utils import resolve_output_dir, write_workbook_sheets
from src.io.loaders import (
    AIQuoteLoader,
    AtlasQuoteLoader,
    ConsensusQuoteLoader,
    load_default_tags,
    load_prompt_context,
)
from src.io.quote_extraction import extract_quotes_from_atlas_exports
from src.report.html import (
    DocumentProcessor,
    apply_coder_comparison_scores,
    build_render_codings,
)
from src.report.xlsx import export_metrics_workbook


def _run_main_owned_steps(
    coding_configurations: Sequence[CodingConfiguration],
    html_consensus_configurations: Sequence[str],
) -> set[str]:
    """The set of run_main sub-steps this invocation is allowed to replace
    on flush: the coarse STEP_RUN_MAIN step itself, its "/superconsensus"
    sub-step (always -- export_consensus_json() always (re)writes the
    superconsensus artifact, even as a copy of a single configuration's own
    consensus), one per currently-configured CodingConfiguration, and one per
    HTML_CONSENSUS_CONFIGURATIONS entry (defensive: nothing enforces those
    names are a subset of coding_configurations, and an unmatched entry
    would otherwise raise the first time it logs anything, instead of
    continuing to silently work as it does today).

    A configuration's sub-step rows are therefore only ever replaced when
    that configuration is actually part of *this* run -- dropping it from
    CODING_CONFIGURATIONS on a later run preserves its previously-logged
    rows untouched rather than wiping them just because run_main reran.
    """
    return (
        {process_log.STEP_RUN_MAIN, f"{process_log.STEP_RUN_MAIN}/{SUPERCONSENSUS_NAME}"}
        | {f"{process_log.STEP_RUN_MAIN}/{c.configuration_name}" for c in coding_configurations}
        | {f"{process_log.STEP_RUN_MAIN}/{name}" for name in html_consensus_configurations}
    )


def main():
    log = process_log.ProcessLog(
        PROCESS_LOG_FILE,
        step=process_log.STEP_RUN_MAIN,
        owned_steps=_run_main_owned_steps(CODING_CONFIGURATIONS, HTML_CONSENSUS_CONFIGURATIONS),
    )
    log.mark_step_start(
        prefix=PREFIX,
        undefined_tags_enabled=UNDEFINED_TAGS_ENABLED,
        coding_configurations=[c.configuration_name for c in CODING_CONFIGURATIONS],
    )
    with process_log.active(log):
        try:
            _run_main_pipeline()
            log.mark_step_end(outcome="success")
        except Exception as exc:
            log.mark_step_end(outcome="error", notes=str(exc))
            raise
        finally:
            log.flush()


def _run_main_pipeline():
    document_paths = run_scope.find_source_documents(RUN_TYPE)
    source_doc_names = [path.stem for path in document_paths]

    preprocess_atlas_exports()
    tags = load_code_tags()
    run_ai_coding(tags)

    atlas_quotes = load_human_quotes(tags, source_doc_names)
    ai_quote_runs = load_ai_quote_runs(tags, source_doc_names)
    documents = run_scope.load_source_documents(
        document_paths, context="run_main_document_loading"
    )

    configuration_names = get_configuration_names()
    final_iterations_by_configuration = get_final_iterations_by_configuration()
    consensus_progression_quotes = build_ai_consensus_quotes(
        documents,
        ai_quote_runs,
        CODING_CONFIGURATIONS,
    )
    consensus_quotes = get_final_consensus_quotes(
        consensus_progression_quotes,
        final_iterations_by_configuration,
    )
    superconsensus_quotes = build_superconsensus_quote_set(
        documents,
        consensus_quotes,
        configuration_names,
    )

    export_consensus_json(
        documents,
        ai_quote_runs,
        consensus_quotes,
        superconsensus_quotes,
        CODING_CONFIGURATIONS,
    )

    html_consensus_quotes = load_html_consensus_quotes(tags, source_doc_names)

    scopes = run_scope.report_scopes(RUN_TYPE)
    primary_label = run_scope.PRIMARY_SCOPE_LABEL if len(scopes) > 1 else next(iter(scopes))

    for scope_label, scope_doc_names in scopes.items():
        if not scope_doc_names:
            message = (
                f"Warning: scope '{scope_label}' (RUN_TYPE={RUN_TYPE!r}) resolved to "
                "zero documents -- has scripts/sample_selector.py been run, and does "
                "META_OUTPUT_FILE have a populated 'dataset' column?"
            )
            print(message)
            process_log.current().log_general_warning(severity="Warning", message=message)
        print(f"\n--- Building report for scope: {scope_label} ---")
        scoped = _slice_to_scope(
            documents,
            atlas_quotes,
            ai_quote_runs,
            consensus_progression_quotes,
            superconsensus_quotes,
            html_consensus_quotes,
            scope_doc_names,
        )
        is_primary = scope_label == primary_label

        if len(scopes) == 1:
            html_file = OUTPUT_HTML_FILE
            xlsx_including = OUTPUT_XLSX_FILE_INCLUDING_UNDEFINED_TAGS
            xlsx_excluding = OUTPUT_XLSX_FILE_EXCLUDING_UNDEFINED_TAGS
        else:
            html_file = scoped_output_html(scope_label)
            xlsx_including = scoped_output_xlsx_including(scope_label)
            xlsx_excluding = scoped_output_xlsx_excluding(scope_label)

        metric_results, consensus_progression_scores = build_metric_results(
            scoped.documents,
            scoped.atlas_quotes,
            scoped.ai_quote_runs,
            scoped.consensus_progression_quotes,
            final_iterations_by_configuration,
            configuration_names,
            scoped.superconsensus_quotes,
            drop_undefined_tags=False,
        )
        tag_analysis_rows, tag_confusion_rows = compute_tag_analysis_and_confusion_rows(
            scoped, tags, final_iterations_by_configuration, drop_undefined_tags=False
        )
        export_metrics(
            metric_results,
            consensus_progression_scores,
            xlsx_including,
            tag_analysis_rows=tag_analysis_rows,
            tag_confusion_rows=tag_confusion_rows,
        )
        if is_primary and COMPARISON_PREFIXES:
            compare_coders(
                output_file=xlsx_including,
                prefix=PREFIX,
                comparison_prefixes=COMPARISON_PREFIXES,
                run_type=RUN_TYPE,
                scope_label=scope_label if len(scopes) > 1 else None,
            )

        if UNDEFINED_TAGS_ENABLED:
            excluding_metric_results, excluding_progression_scores = build_metric_results(
                scoped.documents,
                scoped.atlas_quotes,
                scoped.ai_quote_runs,
                scoped.consensus_progression_quotes,
                final_iterations_by_configuration,
                configuration_names,
                scoped.superconsensus_quotes,
                drop_undefined_tags=True,
            )
            excluding_tag_analysis_rows, excluding_tag_confusion_rows = (
                compute_tag_analysis_and_confusion_rows(
                    scoped, tags, final_iterations_by_configuration, drop_undefined_tags=True
                )
            )
            export_metrics(
                excluding_metric_results,
                excluding_progression_scores,
                xlsx_excluding,
                tag_analysis_rows=excluding_tag_analysis_rows,
                tag_confusion_rows=excluding_tag_confusion_rows,
            )

        render_html_comparison(
            scoped.documents,
            scoped.atlas_quotes,
            scoped.html_consensus_quotes,
            html_file,
        )

        if is_primary:
            open_html_output(html_file)
        else:
            print(f"Report for scope '{scope_label}' written to {html_file}")


def preprocess_atlas_exports() -> None:
    """Backfill quote_text in HUMAN_CODING_FILES for any row whose source
    document text is resolvable. Deliberately NOT scoped to the current
    RUN_TYPE's document_paths (see run_scope.find_source_documents()) --
    HUMAN_CODING_FILES is one shared file spanning documents from every
    subfolder, and a
    RUN_TYPE with a narrow document set (e.g. "test", a handful of manually
    placed files) would otherwise only ever see its own few documents,
    leaving every other row's quote_text unresolvable. Uses every TXT_FILES
    subfolder instead (one whole document per name, see
    run_scope.find_canonical_documents()), so this never depends on which
    RUN_TYPE happens to be active."""
    print("Preprocessing ATLAS Excel files to extract quotes if missing...")
    extract_quotes_from_atlas_exports(
        HUMAN_CODING_FILES, run_scope.find_canonical_documents()
    )


def load_code_tags() -> list[Tag]:
    print("\nLoading tags...")
    return load_default_tags()


def _apply_reasoning_capability_check(
    coding_configurations: list[CodingConfiguration],
) -> list[CodingConfiguration]:
    """For every configuration requesting reasoning, verify (via live
    OpenRouter model data) that its model actually supports it. A model
    confirmed unsupported has reasoning warned-and-dropped for that
    configuration (dataclasses.replace(), since CodingConfiguration is
    frozen) rather than aborting the run. When support can't be determined
    (fetch failed / model missing from OpenRouter's list), reasoning is left
    as configured -- best-effort, matching this codebase's "unresolved =>
    warn, don't block" philosophy elsewhere."""
    reasoning_model_names = {c.model_name for c in coding_configurations if c.reasoning}
    if not reasoning_model_names:
        return coding_configurations

    pricing = fetch_model_pricing(reasoning_model_names)
    adjusted: list[CodingConfiguration] = []
    for coding_config in coding_configurations:
        if not coding_config.reasoning:
            adjusted.append(coding_config)
            continue
        model_pricing = pricing.get(coding_config.model_name)
        if model_pricing is None:
            process_log.current().log_general_warning(
                severity="Warning",
                message=(
                    f"Reasoning support for model '{coding_config.model_name}' "
                    f"(configuration '{coding_config.short_name}') could not be "
                    "verified against OpenRouter's model list -- requesting "
                    "reasoning anyway."
                ),
            )
            adjusted.append(coding_config)
        elif not model_pricing.supports_reasoning:
            process_log.current().log_general_warning(
                severity="Warning",
                message=(
                    f"Model '{coding_config.model_name}' (configuration "
                    f"'{coding_config.short_name}') does not support reasoning "
                    "per OpenRouter's model list -- disabling reasoning for this "
                    "configuration."
                ),
            )
            adjusted.append(
                dataclasses.replace(coding_config, reasoning=False, reasoning_effort=None)
            )
        else:
            adjusted.append(coding_config)
    return adjusted


def run_ai_coding(tags: list[Tag]) -> None:
    if RUN_AI_CODING:
        print("Running AI coding...")
        source_paths = run_scope.find_source_documents(RUN_TYPE)

        coding_configurations = _apply_reasoning_capability_check(CODING_CONFIGURATIONS)

        run_cost_estimate = None
        estimates_by_name: dict[str, cost_estimation.ConfigurationCostEstimate] = {}
        if COST_ESTIMATION_ENABLED:
            run_cost_estimate = cost_estimation.estimate_run_cost(
                coding_configurations, tags, source_paths
            )
            cost_estimation.confirm_cost_estimate(run_cost_estimate.estimates)
            estimates_by_name = {
                estimate.configuration_name: estimate
                for estimate in run_cost_estimate.estimates
            }

        prompt_context = load_prompt_context(DESCRIPTION_FILE)
        # Every condition in the dataset (see document_content.known_conditions()),
        # not just the ones this RUN_TYPE's documents happen to use -- and,
        # separately, which of those are actually present among this
        # RUN_TYPE's documents, so write_prompt_snapshots() can flag the rest
        # as unused rather than silently omitting or including them
        # unmarked.
        all_conditions = known_conditions()
        active_conditions = {
            condition
            for condition in (condition_for(f.stem) for f in source_paths)
            if condition is not None
        }

        prompt_entries: list[tuple[str, CodingConfiguration, str, str]] = []
        # Actual usage summaries returned by process_documents(), grouped by
        # (model_name, effort_bucket) (several configurations can share one
        # model at different effort levels), used to update the per-model,
        # per-effort calibration file after the run.
        usage_by_model: dict[tuple[str, str], list[dict]] = {}
        cost_summary: list[cost_estimation.CostSummaryRow] = []
        for coding_config in coding_configurations:
            effort_bucket = coding_config.effective_reasoning_effort or cost_calibration.NO_REASONING_BUCKET

            warm_start = None
            original_estimate = recalculated_estimate = None
            if COST_ESTIMATION_ENABLED and run_cost_estimate is not None:
                original_estimate = estimates_by_name.get(coding_config.configuration_name)
                probe = cost_estimation.probe_first_paragraph(
                    coding_config, tags, source_paths, run_cost_estimate.prompt_context
                )
                if probe is not None:
                    warm_start = (probe.filepath, probe.paragraph_index, probe.results)
                    recalculated_estimate = cost_estimation.recalculate_from_probe(
                        coding_config,
                        tags,
                        source_paths,
                        run_cost_estimate.prompt_context,
                        run_cost_estimate.pricing,
                        run_cost_estimate.calibration,
                        probe,
                    )
                    if original_estimate is not None:
                        cost_estimation.confirm_recalculated_estimate(
                            coding_config,
                            original_estimate,
                            recalculated_estimate,
                            COST_ESTIMATION_RECALC_THRESHOLD_FRACTION,
                        )

            actual_cost = 0.0
            for iteration in range(coding_config.iterations):
                usage = process_documents(
                    source_paths,
                    coding_config,
                    iteration,
                    tags,
                    warm_start=warm_start if iteration == 0 else None,
                )
                if usage is not None:
                    usage_by_model.setdefault(
                        (coding_config.model_name, effort_bucket), []
                    ).append(usage)
                    actual_cost += usage.get("cost") or 0.0

            if COST_ESTIMATION_ENABLED:
                cost_summary.append(
                    cost_estimation.CostSummaryRow(
                        short_name=coding_config.short_name,
                        model_name=coding_config.model_name,
                        initial_estimate=(
                            original_estimate.estimated_cost if original_estimate else None
                        ),
                        recalculated_estimate=(
                            recalculated_estimate.estimated_cost
                            if recalculated_estimate
                            else None
                        ),
                        actual_cost=actual_cost,
                    )
                )

            # One prompt snapshot per document condition, since each
            # condition's valid_conditions (see document_identity.condition_match())
            # filters which tags are allowed and so changes the rendered prompt.
            for condition in all_conditions:
                condition_scheme = DocumentPromptScheme(
                    coding_scheme=prompt_context,
                    valid_conditions=condition_match(condition),
                    document_condition=condition,
                )
                system, user = render_general_prompt(coding_config, tags, condition_scheme)
                prompt_entries.append((condition, coding_config, system, user))
            process_log.current().log_general_warning(
                severity="Info",
                message=(
                    f"AI-coding prompt snapshot recorded for configuration "
                    f"'{coding_config.configuration_name}'."
                ),
            )
        if prompt_entries:
            output_paths = write_prompt_snapshots(
                prompt_entries, active_conditions=active_conditions
            )
            for output_path in output_paths:
                print(f"Saved AI-coding prompt snapshot to {output_path}")

        if COST_ESTIMATION_ENABLED:
            update_cost_calibration(usage_by_model)
            if cost_summary:
                print()
                print(cost_estimation.format_cost_summary(cost_summary))
    else:
        print("Skipping AI coding (RUN_AI_CODING is False)...")


def update_cost_calibration(usage_by_model: dict[tuple[str, str], list[dict]]) -> None:
    """Aggregate this run's actual usage per (model, effort) and refresh
    src.cost_estimation.calibration's per-model, per-effort calibration file, so future
    cost estimates for that model/effort start from what actually happened
    this time."""
    for (model_name, effort), usage_list in usage_by_model.items():
        calibration = cost_calibration.update_calibration(
            model_name,
            effort,
            paragraph_tokens=sum(u["paragraph_tokens"] for u in usage_list),
            completion_tokens=sum(u["completion_tokens"] for u in usage_list),
            reasoning_tokens=sum(u["reasoning_tokens"] for u in usage_list),
            calls=sum(u["calls"] for u in usage_list),
            prefix=PREFIX,
        )
        if calibration is not None:
            print(
                f"Updated cost calibration for '{model_name}' (effort: {effort}): "
                f"completion_token_factor={calibration.completion_token_factor:.3f}, "
                f"reasoning_token_factor={calibration.reasoning_token_factor:.3f}"
            )


def load_human_quotes(
    tags: dict[str, Tag],
    source_doc_names: list[str],
) -> dict[str, list[Quote]]:
    print("Loading ATLAS quotes (Human)...")
    atlas_quotes = AtlasQuoteLoader(
        HUMAN_CODING_FILES,
        tags,
        condition_match=condition_match,
    ).load_quotes()
    return run_scope.align_quote_documents(atlas_quotes, source_doc_names)


def load_ai_quote_runs(
    tags: dict[str, Tag],
    source_doc_names: list[str],
) -> dict[tuple[str, int], dict[str, list[Quote]]]:
    print("Loading AI quote runs...")
    ai_quote_runs = AIQuoteLoader.load_runs(
        base_dir=AI_CODING_FILES,
        configurations=CODING_CONFIGURATIONS,
        tags=tags,
        condition_match=condition_match,
        log_step_prefix=process_log.STEP_RUN_MAIN,
    )
    return {
        run_key: run_scope.align_quote_documents(quotes_by_doc, source_doc_names)
        for run_key, quotes_by_doc in ai_quote_runs.items()
    }


def get_configuration_names() -> list[str]:
    return [config.configuration_name for config in CODING_CONFIGURATIONS]


def get_final_iterations_by_configuration() -> dict[str, int]:
    return {
        config.configuration_name: config.iterations - 1
        for config in CODING_CONFIGURATIONS
    }


def get_final_consensus_quotes(
    consensus_progression_quotes: Mapping[
        int,
        Mapping[str, Mapping[str, Sequence[Quote]]],
    ],
    final_iterations_by_configuration: Mapping[str, int],
) -> dict[str, dict[str, list[Quote]]]:
    final_consensus: dict[str, dict[str, list[Quote]]] = {}
    for (
        configuration_name,
        final_iteration,
    ) in final_iterations_by_configuration.items():
        quotes_by_doc = consensus_progression_quotes.get(final_iteration, {}).get(
            configuration_name,
            {},
        )
        final_consensus[configuration_name] = {
            doc_name: list(quotes) for doc_name, quotes in quotes_by_doc.items()
        }
    return final_consensus


def build_ai_consensus_quotes(
    documents: dict[str, Document],
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]],
    coding_configurations: Sequence[CodingConfiguration],
) -> dict[int, dict[str, dict[str, list[Quote]]]]:
    print("Building AI consensus quotes...")
    progression: dict[int, dict[str, dict[str, list[Quote]]]] = {}

    for coding_config in coding_configurations:
        config_progression = build_consensus_progression_quote_sets(
            documents=documents,
            ai_quote_runs=ai_quote_runs,
            configuration_names=[coding_config.configuration_name],
            iterations=coding_config.iterations,
            threshold=coding_config.consensus_threshold,
        )
        for iteration_limit, quotes_by_configuration in config_progression.items():
            progression.setdefault(iteration_limit, {}).update(quotes_by_configuration)

    return progression


def build_superconsensus_quote_set(
    documents: dict[str, Document],
    consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    configuration_names: Sequence[str],
) -> dict[str, list[Quote]]:
    """With a single configuration, the superconsensus is just that
    configuration's own consensus -- build_superconsensus_quotes() handles
    this without special-casing (see its docstring). Only zero
    configurations has nothing to build from."""
    if not configuration_names:
        return {}

    print("Building AI superconsensus quotes...")
    return {
        doc_name: list(quotes)
        for doc_name, quotes in build_superconsensus_quotes(
            documents=documents,
            consensus_quotes=consensus_quotes,
            configuration_names=configuration_names,
            threshold=SUPERCONSENSUS_THRESHOLD,
        ).items()
    }


def export_consensus_json(
    documents: dict[str, Document],
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]],
    consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    superconsensus_quotes: Mapping[str, Sequence[Quote]],
    coding_configurations: Sequence[CodingConfiguration],
) -> None:
    """Write each configuration's final consensus, plus a top-level superconsensus,
    to consensus.json files under AI_CODING_FILES.

    A configuration with only one iteration has nothing to vote across, so its
    consensus.json is a direct copy of that single iteration's quotes rather
    than character-level-voted output. The top-level superconsensus.json is
    always written: with 2+ configurations it's the real superconsensus, with
    exactly one configuration it's a copy of that configuration's own
    consensus.json.
    """
    print("Exporting consensus JSON...")
    records_by_configuration: dict[str, list[dict]] = {}

    for coding_config in coding_configurations:
        config_name = coding_config.configuration_name
        quotes_by_doc = (
            ai_quote_runs.get((config_name, 0), {})
            if coding_config.iterations == 1
            else consensus_quotes.get(config_name, {})
        )
        log_double_coding_conflicts(
            documents,
            quotes_by_doc,
            context_message=f"in the final consensus ({config_name})",
            step=f"{process_log.STEP_RUN_MAIN}/{config_name}",
        )
        records = build_consensus_records(documents, quotes_by_doc)
        records_by_configuration[config_name] = records

        output_file = AI_CODING_FILES / config_name / CONSENSUS_FILENAME
        write_consensus_json(records, output_file)
        print(f"  Wrote {len(records)} record(s): {output_file}")

    if len(coding_configurations) >= 2:
        log_double_coding_conflicts(
            documents,
            superconsensus_quotes,
            context_message=f"in the final consensus ({SUPERCONSENSUS_NAME})",
            step=f"{process_log.STEP_RUN_MAIN}/{SUPERCONSENSUS_NAME}",
        )
        superconsensus_records = build_consensus_records(documents, superconsensus_quotes)
    elif len(coding_configurations) == 1:
        superconsensus_records = records_by_configuration[
            coding_configurations[0].configuration_name
        ]
    else:
        superconsensus_records = []

    superconsensus_output_file = AI_CODING_FILES / CONSENSUS_FILENAME
    write_consensus_json(superconsensus_records, superconsensus_output_file)
    print(f"  Wrote {len(superconsensus_records)} record(s): {superconsensus_output_file}")


def load_html_consensus_quotes(
    tags: list[Tag],
    source_doc_names: list[str],
) -> dict[str, dict[str, list[Quote]]]:
    """Load the consensus.json source(s) that feed the HTML report, one row per source.

    HTML_CONSENSUS_CONFIGURATIONS empty (default): a single row labelled
    SUPERCONSENSUS_NAME, loaded from the top-level superconsensus.json.
    Non-empty: one row per listed CodingConfiguration, each loaded from that
    configuration's own consensus.json.
    """
    print("Loading consensus quotes for HTML report...")
    if HTML_CONSENSUS_CONFIGURATIONS:
        sources = {
            config_name: AI_CODING_FILES / config_name / CONSENSUS_FILENAME
            for config_name in HTML_CONSENSUS_CONFIGURATIONS
        }
    else:
        sources = {
            SUPERCONSENSUS_NAME: AI_CODING_FILES / CONSENSUS_FILENAME
        }

    quotes_by_label: dict[str, dict[str, list[Quote]]] = {}
    for label, consensus_file in sources.items():
        quotes_by_doc = ConsensusQuoteLoader(
            consensus_file,
            tags,
            condition_match=condition_match,
            log_step=f"{process_log.STEP_RUN_MAIN}/{label}",
        ).load_quotes()
        quotes_by_label[label] = run_scope.align_quote_documents(quotes_by_doc, source_doc_names)
        quote_count = sum(len(quotes) for quotes in quotes_by_doc.values())
        print(f"  Loaded {label}: {quote_count} quote(s) from {consensus_file}")

    return quotes_by_label


def build_metric_results(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]],
    consensus_progression_quotes: Mapping[
        int,
        Mapping[str, Mapping[str, Sequence[Quote]]],
    ],
    final_iterations_by_configuration: Mapping[str, int],
    configuration_names: list[str],
    superconsensus_quotes: Mapping[str, Sequence[Quote]],
    drop_undefined_tags: bool,
) -> tuple[list[DocumentMetricResult], dict[str, dict[int, float]]]:
    """Run the full metric-calculation sequence for one drop_undefined_tags variant.

    Reuses the already-built quote/consensus sets (independent of
    drop_undefined_tags) and only re-runs scoring, once per call.
    """
    metric_results = calculate_iteration_metrics(
        documents,
        atlas_quotes,
        ai_quote_runs,
        drop_undefined_tags,
    )
    final_consensus_results, consensus_progression_scores = (
        calculate_consensus_progression_metrics(
            documents=documents,
            human_quotes=atlas_quotes,
            consensus_progression_quotes=consensus_progression_quotes,
            final_iterations_by_configuration=final_iterations_by_configuration,
            drop_undefined_tags=drop_undefined_tags,
        )
    )
    add_final_consensus_metrics(
        metric_results,
        final_consensus_results,
        configuration_names,
    )
    metric_results.extend(
        calculate_superconsensus_metrics(
            documents,
            atlas_quotes,
            superconsensus_quotes,
            drop_undefined_tags,
        )
    )
    return metric_results, consensus_progression_scores


def calculate_iteration_metrics(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]],
    drop_undefined_tags: bool = False,
) -> list[DocumentMetricResult]:
    print("Calculating metrics...")
    metric_results = []
    for coding_config in CODING_CONFIGURATIONS:
        config_name = coding_config.configuration_name

        for iteration in range(coding_config.iterations):
            coded_results = calculate_metrics_for_quote_sets(
                documents=documents,
                human_quotes=atlas_quotes,
                ai_quotes=ai_quote_runs.get((config_name, iteration), {}),
                configuration_name=config_name,
                iteration=iteration,
                is_consensus=False,
                drop_undefined_tags=drop_undefined_tags,
                step=f"{process_log.STEP_RUN_MAIN}/{config_name}",
            )
            metric_results.extend(result.metrics for result in coded_results)

    return metric_results


def add_final_consensus_metrics(
    metric_results: list[DocumentMetricResult],
    final_consensus_results: dict[str, list[DocumentMetricResult]],
    configuration_names: list[str],
) -> None:
    for config_name in configuration_names:
        metric_results.extend(final_consensus_results.get(config_name, []))


def calculate_superconsensus_metrics(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    superconsensus_quotes: Mapping[str, Sequence[Quote]],
    drop_undefined_tags: bool = False,
) -> list[DocumentMetricResult]:
    if not superconsensus_quotes:
        return []

    print("Calculating superconsensus metrics...")
    coded_results = calculate_metrics_for_quote_sets(
        documents=documents,
        human_quotes=atlas_quotes,
        ai_quotes=superconsensus_quotes,
        configuration_name=SUPERCONSENSUS_NAME,
        iteration=None,
        is_consensus=True,
        drop_undefined_tags=drop_undefined_tags,
        step=f"{process_log.STEP_RUN_MAIN}/{SUPERCONSENSUS_NAME}",
    )
    return [result.metrics for result in coded_results]


def export_metrics(
    metric_results: list[DocumentMetricResult],
    consensus_progression_scores: dict[str, dict[int, float]],
    output_file: Path,
    tag_analysis_rows: list[dict[str, object]] | None = None,
    tag_confusion_rows: list[dict[str, object]] | None = None,
) -> None:
    print("Exporting metrics...")
    export_metrics_workbook(
        metric_results,
        output_file,
        coding_configurations=CODING_CONFIGURATIONS,
        consensus_progression_scores=consensus_progression_scores,
        tag_analysis_rows=tag_analysis_rows,
        tag_confusion_rows=tag_confusion_rows,
    )
    print(f"Metrics workbook created: {output_file}")


def compute_tag_analysis_and_confusion_rows(
    scoped: "_ScopedPipelineData",
    tags: list[Tag],
    final_iterations_by_configuration: Mapping[str, int],
    drop_undefined_tags: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Per-tag analysis + tag-substitution confusion rows for one report
    scope (see src.analysis.tag_analysis). Reuses this scope's already-built final
    consensus quotes (one set per CodingConfiguration) and superconsensus
    quotes -- the same "consensus-level only" quote sets build_metric_results()
    scores -- rather than resolving them a second time."""
    print("Calculating per-tag analysis...")
    final_consensus_quotes = get_final_consensus_quotes(
        scoped.consensus_progression_quotes,
        final_iterations_by_configuration,
    )
    ai_quotes_by_configuration: dict[str, dict[str, list[Quote]]] = dict(
        final_consensus_quotes
    )
    # scoped.superconsensus_quotes is always a non-empty dict -- one entry
    # per scoped document, per _slice_to_scope() -- even when there's no
    # real superconsensus at all (zero coding configurations, see
    # build_superconsensus_quote_set()'s guard; one configuration's
    # superconsensus is that configuration's own consensus, not empty).
    # has_any_quotes() checks whether any of those entries actually holds a
    # quote, not just whether the dict itself is non-empty.
    if tag_analysis.has_any_quotes(scoped.superconsensus_quotes):
        ai_quotes_by_configuration[SUPERCONSENSUS_NAME] = dict(scoped.superconsensus_quotes)

    tag_analysis_rows = tag_analysis.compute_tag_analysis_rows(
        scoped.documents,
        scoped.atlas_quotes,
        ai_quotes_by_configuration,
        tags,
        drop_undefined_tags=drop_undefined_tags,
    )
    tag_confusion_rows = tag_analysis.compute_tag_confusion_rows(
        scoped.documents,
        scoped.atlas_quotes,
        scoped.superconsensus_quotes,
        drop_undefined_tags=drop_undefined_tags,
    )
    return tag_analysis_rows, tag_confusion_rows


def render_html_comparison(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    html_consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    html_file: Path = OUTPUT_HTML_FILE,
) -> None:
    print("Generating HTML comparison...")
    resolve_output_dir(html_file)

    print("Loading paragraph metadata...")
    para_meta = paragraph_meta()
    print(f"Loaded metadata for {len(para_meta)} document(s)")

    apply_coder_comparison_scores(documents, atlas_quotes, html_consensus_quotes)

    codings = build_render_codings(atlas_quotes, html_consensus_quotes)

    print("Rendering HTML output...")
    processor = DocumentProcessor(
        run_scope.processing_dirs(RUN_TYPE),
        para_meta=para_meta,
        documents=documents,
    )
    processor.generate_html(codings, html_file, prefix=PREFIX)


def open_html_output(html_file: Path = OUTPUT_HTML_FILE) -> None:
    print("Done!")
    print(f"Opening {html_file.name} in your browser...")
    webbrowser.open(f"file://{html_file.resolve()}")


class _ScopedPipelineData(NamedTuple):
    """The subset of _run_main_pipeline()'s per-document data belonging to
    one report scope (see src.dataset.run_scope.report_scopes())."""

    documents: dict[str, Document]
    atlas_quotes: dict[str, list[Quote]]
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]]
    consensus_progression_quotes: Mapping[
        int, Mapping[str, Mapping[str, Sequence[Quote]]]
    ]
    superconsensus_quotes: Mapping[str, Sequence[Quote]]
    html_consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]]


def _slice_to_scope(
    documents: dict[str, Document],
    atlas_quotes: dict[str, list[Quote]],
    ai_quote_runs: dict[tuple[str, int], dict[str, list[Quote]]],
    consensus_progression_quotes: Mapping[
        int, Mapping[str, Mapping[str, Sequence[Quote]]]
    ],
    superconsensus_quotes: Mapping[str, Sequence[Quote]],
    html_consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    scope_doc_names: Sequence[str],
) -> _ScopedPipelineData:
    """Narrows every per-document mapping down to scope_doc_names, tolerating
    documents missing from any particular mapping (e.g. a document with no
    atlas/AI quotes at all) rather than raising a KeyError."""
    names = list(scope_doc_names)
    return _ScopedPipelineData(
        documents={name: documents[name] for name in names if name in documents},
        atlas_quotes={name: atlas_quotes.get(name, []) for name in names},
        ai_quote_runs={
            run_key: {name: quotes_by_doc.get(name, []) for name in names}
            for run_key, quotes_by_doc in ai_quote_runs.items()
        },
        consensus_progression_quotes={
            iteration_limit: {
                config_name: {name: quotes_by_doc.get(name, []) for name in names}
                for config_name, quotes_by_doc in by_config.items()
            }
            for iteration_limit, by_config in consensus_progression_quotes.items()
        },
        superconsensus_quotes={
            name: superconsensus_quotes.get(name, []) for name in names
        },
        html_consensus_quotes={
            label: {name: quotes_by_doc.get(name, []) for name in names}
            for label, quotes_by_doc in html_consensus_quotes.items()
        },
    )


def calculate_consensus_progression_metrics(
    documents: dict[str, Document],
    human_quotes: dict[str, list[Quote]],
    consensus_progression_quotes: Mapping[
        int,
        Mapping[str, Mapping[str, Sequence[Quote]]],
    ],
    final_iterations_by_configuration: Mapping[str, int],
    drop_undefined_tags: bool = False,
) -> tuple[dict[str, list[DocumentMetricResult]], dict[str, dict[int, float]]]:
    """Score cumulative consensuses and return final metrics plus progression scores."""
    print("Calculating consensus progression...")
    final_results: dict[str, list[DocumentMetricResult]] = {}
    progression_scores: dict[str, dict[int, float]] = {}

    for (
        iteration_limit,
        quotes_by_configuration,
    ) in consensus_progression_quotes.items():
        for configuration_name, cumulative_consensus in quotes_by_configuration.items():
            coded_results = calculate_metrics_for_quote_sets(
                documents=documents,
                human_quotes=human_quotes,
                ai_quotes=cumulative_consensus,
                configuration_name=configuration_name,
                iteration=None,
                is_consensus=True,
                drop_undefined_tags=drop_undefined_tags,
                step=f"{process_log.STEP_RUN_MAIN}/{configuration_name}",
            )
            document_metrics = [result.metrics for result in coded_results]
            progression_scores.setdefault(configuration_name, {})[iteration_limit] = (
                _average(
                    [result.avg_pruned_char_agreement for result in document_metrics]
                )
            )
            if iteration_limit == final_iterations_by_configuration.get(
                configuration_name
            ):
                final_results[configuration_name] = document_metrics

    return final_results, progression_scores


def compare_coders(
    output_file: Path,
    prefix: str,
    comparison_prefixes: list[str],
    run_type: str,
    scope_label: str | None = None,
) -> None:
    """Add a comparison sheet from existing overview XLSX exports.

    Reads each prefix's "including undefined tags" workbook -- the one that
    always exists regardless of that prefix's own UNDEFINED_TAGS_ENABLED
    setting (see src/config/general.py). Compares against the *same* run_type for
    every comparison_prefix (RESULTS_DIR is nested per run type, so a
    prefix's "full" results live in a different folder than its "training"
    results) and, for multi-scope run types (RUN_TYPE == "full"), the same
    scope_label suffix (see src.config.general.scoped_output_xlsx_including()) --
    this is only ever called for the primary scope's report (see
    _run_main_pipeline()), so every comparison_prefix is looked up at its
    own primary scope too.
    """
    prefixes = list(dict.fromkeys([prefix, *comparison_prefixes]))

    rows = []
    for comparison_prefix in prefixes:
        filename = (
            f"configuration_matrix_main_{comparison_prefix}_including_undefined_tags.xlsx"
            if scope_label is None
            else f"configuration_matrix_main_{comparison_prefix}_{scope_label}_including_undefined_tags.xlsx"
        )
        workbook = run_results_dir(comparison_prefix, run_type) / RESULTS_SUBDIR / filename
        if not workbook.exists():
            message = f"Warning: Comparison workbook not found: {workbook}"
            print(message)
            process_log.current().log_general_warning(severity="Warning", message=message)
            continue

        try:
            overview = pd.read_excel(workbook, sheet_name="overview")
        except ValueError:
            message = f"Warning: No overview sheet in {workbook}"
            print(message)
            process_log.current().log_general_warning(severity="Warning", message=message)
            continue

        overview.insert(0, "prefix", comparison_prefix)
        rows.extend(overview.to_dict(orient="records"))

    if not rows:
        return

    write_workbook_sheets(
        {"comparison": pd.DataFrame(rows)},
        output_file,
        append_to_existing_file=True,
        replace_existing_sheet=True,
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
