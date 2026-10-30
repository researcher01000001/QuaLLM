"""Pre-run AI-coding cost estimation"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.coding.ai_coding import analyse_with_genai, prepare_document, render_prompt
from src.config.coding import CodingConfiguration
from src.config.general import AI_CODING_MIN_PARAGRAPH_CHARS, PREFIX
from src.config.project import DESCRIPTION_FILE
from src.core import process_log
from src.core.models import CodingScheme, Tag
from src.cost_estimation.calibration import (
    ModelCalibration,
    NO_REASONING_BUCKET,
    load_calibration,
)
from src.cost_estimation.pricing import ModelPricing, fetch_model_pricing
from src.cost_estimation.tokenization import count_tokens
from src.io.io_utils import get_matching_files
from src.io.loaders import load_prompt_context


@dataclass
class ConfigurationCostEstimate:
    """Estimated token counts/cost for one CodingConfiguration's full run
    (every paragraph, across all its iterations).

    input_tokens (the full system+user prompt) is what's actually billed as
    prompt tokens -- used directly for input cost. paragraph_tokens (just the
    raw paragraph text) is the basis for the completion/reasoning estimate
    instead: paragraph length/complexity drives how many codes get
    extracted, not the size of the (largely fixed, per-document) coding
    scheme/instructions overhead that dominates input_tokens."""

    configuration_name: str
    short_name: str
    model_name: str
    effort: str | None
    iterations: int
    prompt_caching: bool = False
    paragraph_count: int = 0
    paragraph_tokens: int = 0
    input_tokens: int = 0
    system_tokens: int = 0
    user_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    completion_token_factor: float | None = None
    completion_token_factor_source: str = "unresolved"
    reasoning_token_factor: float | None = None
    reasoning_token_factor_source: str = "unresolved"
    input_price_per_token: float | None = None
    cache_read_price_per_token: float | None = None
    cache_write_price_per_token: float | None = None
    completion_price_per_token: float | None = None
    reasoning_price_per_token: float | None = None
    unresolved_reasons: list[str] = field(default_factory=list)

    @property
    def estimated_completion_tokens(self) -> int | None:
        """paragraph_tokens (not input_tokens) is the basis -- completion
        size scales with what's actually in the paragraph, not with the
        fixed prompt overhead around it. Includes reasoning tokens (a
        subset), matching how OpenRouter/OpenAI's usage.completion_tokens
        is calibrated (src/cost_estimation/calibration.py)."""
        if self.completion_token_factor is None:
            return None
        return round(self.paragraph_tokens * self.completion_token_factor)

    @property
    def estimated_reasoning_tokens(self) -> int | None:
        """A subset of estimated_completion_tokens, not additional to it."""
        completion_tokens = self.estimated_completion_tokens
        if completion_tokens is None or self.reasoning_token_factor is None:
            return None
        return round(completion_tokens * self.reasoning_token_factor)

    @property
    def estimated_visible_output_tokens(self) -> int | None:
        """completion tokens minus reasoning tokens -- the actual visible
        text output. Falls back to the full completion-token estimate when
        the reasoning split is unresolved (can't split what isn't known),
        so a cost estimate is still shown rather than blocked."""
        completion_tokens = self.estimated_completion_tokens
        if completion_tokens is None:
            return None
        reasoning_tokens = self.estimated_reasoning_tokens
        if reasoning_tokens is None:
            return completion_tokens
        return completion_tokens - reasoning_tokens

    @property
    def cache_pricing_applied(self) -> bool:
        return (
            self.prompt_caching
            and self.cache_read_price_per_token is not None
            and self.cache_write_price_per_token is not None
        )

    @property
    def estimated_input_cost(self) -> float | None:
        if self.input_price_per_token is None:
            return None
        cache_read_price = self.cache_read_price_per_token
        cache_write_price = self.cache_write_price_per_token
        if self.prompt_caching and cache_read_price is not None and cache_write_price is not None:
            return (
                self.user_tokens * self.input_price_per_token
                + self.cache_write_tokens * cache_write_price
                + self.cache_read_tokens * cache_read_price
            )
        return self.input_tokens * self.input_price_per_token

    @property
    def estimated_uncached_input_cost(self) -> float | None:
        if self.input_price_per_token is None:
            return None
        return self.input_tokens * self.input_price_per_token

    @property
    def estimated_output_cost(self) -> float | None:
        completion_tokens = self.estimated_completion_tokens
        if completion_tokens is None or self.completion_price_per_token is None:
            return None
        reasoning_tokens = self.estimated_reasoning_tokens or 0
        visible_output_tokens = completion_tokens - reasoning_tokens
        reasoning_price = (
            self.reasoning_price_per_token
            if self.reasoning_price_per_token is not None
            else self.completion_price_per_token
        )
        return (
            visible_output_tokens * self.completion_price_per_token
            + reasoning_tokens * reasoning_price
        )

    @property
    def estimated_cost(self) -> float | None:
        input_cost = self.estimated_input_cost
        output_cost = self.estimated_output_cost
        if input_cost is None or output_cost is None:
            return None
        return input_cost + output_cost

    @property
    def estimated_uncached_cost(self) -> float | None:
        input_cost = self.estimated_uncached_input_cost
        output_cost = self.estimated_output_cost
        if input_cost is None or output_cost is None:
            return None
        return input_cost + output_cost


def resolve_completion_token_factor(
    coding_config: CodingConfiguration,
    calibration: dict[str, dict[str, ModelCalibration]],
    override: float | None = None,
) -> tuple[float | None, str]:
    """Resolution chain: override (only ever passed by recalculate_from_probe(),
    a real ratio observed from actually calling the model) -> learned
    calibration for this model at its effective effort bucket -> learned
    calibration for this model at ANY other effort bucket (still far more
    informative than nothing, clearly labeled as a cross-effort fallback)
    -> unresolved.

    The resolved factor is multiplied by a paragraph's own token count (not
    the full prompt's) to estimate its completion tokens -- see
    ConfigurationCostEstimate.estimated_completion_tokens."""
    if override is not None:
        return override, "probe"
    model_entries = calibration.get(coding_config.model_name)
    if model_entries:
        effort_bucket = coding_config.effective_reasoning_effort or NO_REASONING_BUCKET
        entry = model_entries.get(effort_bucket)
        if entry is not None:
            return entry.completion_token_factor, f"calibration({effort_bucket})"
        other_effort, other_entry = next(iter(model_entries.items()))
        return (
            other_entry.completion_token_factor,
            f"calibration({other_effort}, cross-effort fallback)",
        )
    return None, "unresolved"


def resolve_reasoning_token_factor(
    coding_config: CodingConfiguration,
    calibration: dict[str, dict[str, ModelCalibration]],
    override: float | None = None,
) -> tuple[float | None, str]:
    """A call that won't request reasoning at all never produces reasoning
    tokens, regardless of what calibration data exists for this model --
    short-circuit before consulting any resolution source. Otherwise: override
    (see resolve_completion_token_factor()) -> calibration (exact effort) ->
    calibration (any other effort, excluding the reasoning-disabled bucket)
    -> unresolved."""
    if not coding_config.reasoning:
        return 0.0, "reasoning_disabled"
    if override is not None:
        return override, "probe"
    model_entries = calibration.get(coding_config.model_name)
    if model_entries:
        effort_bucket = coding_config.effective_reasoning_effort or "medium"
        entry = model_entries.get(effort_bucket)
        if entry is not None:
            return entry.reasoning_token_factor, f"calibration({effort_bucket})"
        other_effort, other_entry = next(
            (
                (effort, e)
                for effort, e in model_entries.items()
                if effort != NO_REASONING_BUCKET
            ),
            (None, None),
        )
        if other_entry is not None:
            return (
                other_entry.reasoning_token_factor,
                f"calibration({other_effort}, cross-effort fallback)",
            )
    return None, "unresolved"


def estimate_configuration_cost(
    coding_config: CodingConfiguration,
    tags: list[Tag],
    source_dir: Path | list[Path],
    prompt_context: CodingScheme,
    pricing: Mapping[str, ModelPricing | None],
    calibration: dict[str, dict[str, ModelCalibration]],
    *,
    completion_token_factor_override: float | None = None,
    reasoning_token_factor_override: float | None = None,
) -> ConfigurationCostEstimate:
    """Tokenize every prompt a real run of coding_config would send, without
    calling the model. Reuses src.coding.ai_coding.prepare_document()/render_prompt()
    so the paragraphs/rows/scheme counted here can never disagree with what
    process_documents() would actually send.

    completion_token_factor_override/reasoning_token_factor_override are
    only ever passed by recalculate_from_probe() -- a real, single-call
    ratio observed from actually calling the model, which takes priority
    over calibration (see resolve_completion_token_factor()/
    resolve_reasoning_token_factor())."""
    estimate = ConfigurationCostEstimate(
        configuration_name=coding_config.configuration_name,
        short_name=coding_config.short_name,
        model_name=coding_config.model_name,
        effort=coding_config.effective_reasoning_effort,
        iterations=coding_config.iterations,
        prompt_caching=coding_config.prompt_caching,
    )

    system_prompt_tokens: dict[str, int] = {}
    files = get_matching_files(source_dir, "*.txt", recursive=True)
    for filepath in files:
        prep = prepare_document(
            filepath, coding_config, tags, prompt_context, log_warnings=False
        )
        if prep is None:
            continue
        for paragraph, row in zip(prep.paragraphs, prep.rows):
            if len(paragraph) < AI_CODING_MIN_PARAGRAPH_CHARS:
                # Mirrors analyse_with_genai()'s skip -- no call would be made.
                continue
            system, user = render_prompt(
                coding_config, prep.tags, paragraph, row, prep.scheme
            )
            system_tokens = count_tokens(system)
            user_tokens = count_tokens(user)
            estimate.paragraph_count += 1
            estimate.paragraph_tokens += count_tokens(paragraph)
            estimate.system_tokens += system_tokens
            estimate.user_tokens += user_tokens
            estimate.input_tokens += system_tokens + user_tokens
            system_prompt_tokens.setdefault(system, system_tokens)

    # Prompt content is identical across iterations of the same
    # configuration (only temperature/seed differ), so token counts scale
    # linearly with iterations without re-walking the documents per iteration.
    estimate.paragraph_count *= coding_config.iterations
    estimate.paragraph_tokens *= coding_config.iterations
    estimate.system_tokens *= coding_config.iterations
    estimate.user_tokens *= coding_config.iterations
    estimate.input_tokens *= coding_config.iterations

    if coding_config.prompt_caching:
        # Structural cache assumption: the first occurrence of each distinct
        # rendered system prompt writes the cache; every later occurrence,
        # including later iterations, reads it. The dynamic user message is
        # always billed normally. Provider fallback/expiry can cause fewer
        # hits in reality, so format_cost_overview() also shows no-cache cost.
        estimate.cache_write_tokens = sum(system_prompt_tokens.values())
        estimate.cache_read_tokens = estimate.system_tokens - estimate.cache_write_tokens

    estimate.completion_token_factor, estimate.completion_token_factor_source = (
        resolve_completion_token_factor(
            coding_config, calibration, override=completion_token_factor_override
        )
    )
    estimate.reasoning_token_factor, estimate.reasoning_token_factor_source = (
        resolve_reasoning_token_factor(
            coding_config, calibration, override=reasoning_token_factor_override
        )
    )
    if estimate.completion_token_factor is None:
        estimate.unresolved_reasons.append(
            f"no completion_token_factor available for model '{coding_config.model_name}' "
            "(no calibration data yet -- resolved automatically once this "
            "configuration's first row is probed)"
        )
    elif coding_config.reasoning and estimate.reasoning_token_factor is None:
        estimate.unresolved_reasons.append(
            f"no reasoning_token_factor available for model '{coding_config.model_name}' "
            f"at effort '{estimate.effort}' -- estimate excludes reasoning tokens, may "
            "understate cost for this reasoning-enabled configuration"
        )

    model_pricing = pricing.get(coding_config.model_name)
    if model_pricing is None:
        estimate.unresolved_reasons.append(
            f"no live pricing available for model '{coding_config.model_name}' from OpenRouter"
        )
    else:
        estimate.input_price_per_token = model_pricing.prompt_price_per_token
        estimate.cache_read_price_per_token = model_pricing.cache_read_price_per_token
        estimate.cache_write_price_per_token = model_pricing.cache_write_price_per_token
        estimate.completion_price_per_token = model_pricing.completion_price_per_token
        estimate.reasoning_price_per_token = model_pricing.reasoning_price_per_token

    return estimate


@dataclass
class RunCostEstimate:
    """estimate_run_cost()'s full result: the per-configuration estimates
    plus the shared resources (prompt_context/pricing/calibration) used to
    compute them, so run.py can reuse them for the first-row probe
    recalculation (recalculate_from_probe()) instead of re-fetching."""

    estimates: list[ConfigurationCostEstimate]
    prompt_context: CodingScheme
    pricing: dict[str, ModelPricing | None]
    calibration: dict[str, dict[str, ModelCalibration]]


def estimate_run_cost(
    coding_configurations: list[CodingConfiguration],
    tags: list[Tag],
    source_dir: Path | list[Path],
) -> RunCostEstimate:
    """One ConfigurationCostEstimate per entry in coding_configurations."""
    prompt_context = load_prompt_context(DESCRIPTION_FILE)
    calibration = load_calibration()
    pricing = fetch_model_pricing({c.model_name for c in coding_configurations})
    estimates = [
        estimate_configuration_cost(
            coding_config, tags, source_dir, prompt_context, pricing, calibration
        )
        for coding_config in coding_configurations
    ]
    return RunCostEstimate(
        estimates=estimates, prompt_context=prompt_context, pricing=pricing, calibration=calibration
    )


@dataclass
class ProbeResult:
    """The result of actually calling the model for one configuration's
    first eligible paragraph (see probe_first_paragraph()) -- a genuine
    production call, not a throwaway test, so its results/usage are reused
    (not discarded) by run.py's warm_start wiring into process_documents()."""

    filepath: Path
    paragraph_index: int
    results: list[dict[str, Any]]
    usage: dict[str, Any] | None


def probe_first_paragraph(
    coding_config: CodingConfiguration,
    tags: list[Tag],
    source_dir: Path | list[Path],
    prompt_context: CodingScheme,
) -> ProbeResult | None:
    """Find and really call the model on coding_config's first eligible
    (>= 10 char) paragraph, walking files/paragraphs the same deterministic
    order estimate_configuration_cost() uses. Returns None when the
    configuration has no eligible paragraphs at all (nothing to probe or
    run)."""
    files = get_matching_files(source_dir, "*.txt", recursive=True)
    for filepath in files:
        prep = prepare_document(
            filepath, coding_config, tags, prompt_context, log_warnings=False
        )
        if prep is None:
            continue
        for i, (paragraph, row) in enumerate(zip(prep.paragraphs, prep.rows)):
            if len(paragraph) < AI_CODING_MIN_PARAGRAPH_CHARS:
                continue
            results = analyse_with_genai(
                i,
                paragraph,
                len(prep.paragraphs),
                str(filepath),
                coding_config,
                prep.tags,
                row,
                prep.scheme,
            )
            usage = results[0].get("usage") if results else None
            return ProbeResult(
                filepath=filepath, paragraph_index=i, results=results, usage=usage
            )
    return None


def recalculate_from_probe(
    coding_config: CodingConfiguration,
    tags: list[Tag],
    source_dir: Path | list[Path],
    prompt_context: CodingScheme,
    pricing: Mapping[str, ModelPricing | None],
    calibration: dict[str, dict[str, ModelCalibration]],
    probe: ProbeResult,
) -> ConfigurationCostEstimate:
    """Recalculate coding_config's full-run cost estimate using the real,
    single-call token ratios just observed from probe. Reuses
    estimate_configuration_cost() rather than duplicating its logic, passing
    the probed ratios as its completion/reasoning_token_factor_override
    params -- the highest-priority source in resolve_completion_token_factor()
    /resolve_reasoning_token_factor() -- so the whole configuration's
    remaining paragraphs estimate from just-observed reality instead of a
    (possibly absent) calibration entry."""
    usage = probe.usage or {}
    paragraph_tokens = usage.get("paragraph_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    reasoning_tokens = (
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    )

    completion_factor = (
        completion_tokens / paragraph_tokens if paragraph_tokens and completion_tokens else None
    )
    reasoning_factor = (
        reasoning_tokens / completion_tokens
        if coding_config.reasoning and completion_tokens
        else 0.0
    )

    return estimate_configuration_cost(
        coding_config,
        tags,
        source_dir,
        prompt_context,
        pricing,
        calibration,
        completion_token_factor_override=completion_factor,
        reasoning_token_factor_override=reasoning_factor,
    )


def confirm_recalculated_estimate(
    coding_config: CodingConfiguration,
    original: ConfigurationCostEstimate,
    recalculated: ConfigurationCostEstimate,
    threshold_fraction: float,
) -> None:
    """After probing coding_config's first real row, compare the
    recalculated full-run estimate against the original pre-run estimate
    and, per COST_ESTIMATION_RECALC_THRESHOLD_FRACTION (0.1 = 10%, i.e. the
    recalculated cost is more than 1.1x the original), ask the user to
    re-confirm before the rest of the run proceeds. Mirrors
    confirm_cost_estimate()'s print + process_log.confirm() +
    sys.exit(1)-on-decline pattern, under a distinct decision_point so it's
    distinguishable in the audit log."""
    new_cost = recalculated.estimated_cost
    if new_cost is None:
        # Still nothing more informative to show than the original estimate.
        return

    old_cost = original.estimated_cost
    if old_cost is None:
        should_confirm = True
        detail = (
            f"configuration '{coding_config.short_name}': initial estimate was "
            f"unresolved; after probing the first row, the estimated cost is "
            f"{_format_cost(new_cost)}"
        )
    elif old_cost == 0:
        # No baseline to take a ratio against -- any positive recalculated
        # cost is an undefined (effectively infinite) increase, so always
        # surface it rather than silently letting it through.
        should_confirm = new_cost > 0
        detail = (
            f"configuration '{coding_config.short_name}': initial estimate was "
            f"$0.0000; recalculated estimate is {_format_cost(new_cost)}"
        )
    else:
        increase_fraction = (new_cost - old_cost) / old_cost
        should_confirm = increase_fraction > threshold_fraction
        detail = (
            f"configuration '{coding_config.short_name}': initial estimate "
            f"{_format_cost(old_cost)} -> recalculated {_format_cost(new_cost)} "
            f"({increase_fraction * 100:+.1f}%)"
        )

    if not should_confirm:
        return

    print()
    print(
        f"Recalculated cost estimate for '{coding_config.short_name}' after "
        "probing its first row:"
    )
    print(f"  Initial estimate:      {_format_cost(old_cost)}")
    print(f"  Recalculated estimate: {_format_cost(new_cost)}")
    print()
    decision = process_log.confirm(
        "Continue with the rest of this configuration's AI coding? [y/N]: ",
        decision_point="ai_coding_cost_recalculation_confirmation",
        detail=detail,
        outcome_no="aborted",
    )
    if not decision:
        print("Aborted: AI coding was not run.")
        sys.exit(1)


@dataclass
class CostSummaryRow:
    """One configuration's three cost figures for the end-of-run summary
    (format_cost_summary()): the original pre-run estimate, the post-probe
    recalculated estimate, and the real actual cost OpenRouter billed."""

    short_name: str
    model_name: str
    initial_estimate: float | None
    recalculated_estimate: float | None
    actual_cost: float | None


def format_cost_summary(rows: list[CostSummaryRow]) -> str:
    """One block per configuration: initial estimate, recalculated
    (post-probe) estimate, and real actual cost, plus a TOTAL row. Any
    missing value prints as N/A and is excluded from the totals, per the
    'show as much as possible' philosophy used throughout this module."""
    lines = ["AI-coding cost summary", ""]
    total_initial = total_recalculated = total_actual = 0.0
    any_missing = False

    for row in rows:
        lines.append(f"{row.short_name} ({row.model_name}):")
        lines.append(f"  Initial estimate:      {_format_cost(row.initial_estimate)}")
        lines.append(f"  Second estimation:     {_format_cost(row.recalculated_estimate)}")
        lines.append(f"  Actual cost:           {_format_cost(row.actual_cost)}")
        lines.append("")
        if row.initial_estimate is not None:
            total_initial += row.initial_estimate
        else:
            any_missing = True
        if row.recalculated_estimate is not None:
            total_recalculated += row.recalculated_estimate
        else:
            any_missing = True
        if row.actual_cost is not None:
            total_actual += row.actual_cost
        else:
            any_missing = True

    lines.append("TOTAL:")
    lines.append(f"  Initial estimate:      {_format_cost(total_initial)}")
    lines.append(f"  Second estimation:     {_format_cost(total_recalculated)}")
    lines.append(f"  Actual cost:           {_format_cost(total_actual)}")
    if any_missing:
        lines.append("")
        lines.append(
            "Note: one or more rows are missing a value (see 'N/A' above) -- "
            "totals exclude those."
        )
    return "\n".join(lines)


def _format_tokens(value: int | None) -> str:
    return f"{value:,}" if value is not None else "N/A"


def _format_cost(value: float | None) -> str:
    return f"${value:,.4f}" if value is not None else "N/A"


def format_cost_overview(estimates: list[ConfigurationCostEstimate]) -> str:
    header = (
        f"{'Configuration':<22} {'Model':<32} {'Effort':<7} {'Iter':>4} {'Paragraphs':>10} "
        f"{'Para. tok':>10} {'Input tok':>12} {'Cache W':>10} {'Cache R':>10} "
        f"{'Completion tok':>15} {'Reason. tok':>12} {'Est. cost':>12} {'No-cache':>12}"
    )
    separator = "-" * len(header)
    lines = ["AI-coding cost estimate (approximate)", "", header, separator]

    total_paragraph = 0
    total_input = 0
    total_cache_write = 0
    total_cache_read = 0
    total_completion = 0
    total_reasoning = 0
    total_cost = 0.0
    total_uncached_cost = 0.0
    any_unresolved = False

    for est in estimates:
        completion_tokens = est.estimated_completion_tokens
        reasoning_tokens = est.estimated_reasoning_tokens
        cost = est.estimated_cost
        uncached_cost = est.estimated_uncached_cost
        lines.append(
            f"{est.short_name:<22} {est.model_name:<32} {est.effort or '-':<7} {est.iterations:>4} "
            f"{est.paragraph_count:>10,} {est.paragraph_tokens:>10,} "
            f"{est.input_tokens:>12,} {est.cache_write_tokens:>10,} "
            f"{est.cache_read_tokens:>10,} {_format_tokens(completion_tokens):>15} "
            f"{_format_tokens(reasoning_tokens):>12} {_format_cost(cost):>12} "
            f"{_format_cost(uncached_cost):>12}"
        )
        total_paragraph += est.paragraph_tokens
        total_input += est.input_tokens
        total_cache_write += est.cache_write_tokens
        total_cache_read += est.cache_read_tokens
        total_completion += completion_tokens or 0
        total_reasoning += reasoning_tokens or 0
        if cost is not None:
            total_cost += cost
        if uncached_cost is not None:
            total_uncached_cost += uncached_cost
        if est.prompt_caching and not est.cache_pricing_applied:
            lines.append(
                "    -> cache pricing unavailable; estimate uses ordinary input pricing"
            )
        if est.unresolved_reasons:
            any_unresolved = True
            for reason in est.unresolved_reasons:
                lines.append(f"    -> {reason}")

    lines.append(separator)
    lines.append(
        f"{'TOTAL':<22} {'':<32} {'':<7} {'':>4} {'':>10} {total_paragraph:>10,} "
        f"{total_input:>12,} {total_cache_write:>10,} {total_cache_read:>10,} "
        f"{total_completion:>15,} {total_reasoning:>12,} "
        f"{_format_cost(total_cost):>12} {_format_cost(total_uncached_cost):>12}"
    )
    if any_unresolved:
        lines.append("")
        lines.append(
            "Note: the total above is a PARTIAL estimate -- one or more "
            "configurations have unresolved values (see '->' notes above); "
            "the actual cost may be higher."
        )
    return "\n".join(lines)


def confirm_cost_estimate(estimates: list[ConfigurationCostEstimate]) -> None:
    """Print the cost overview and require console confirmation before the
    caller proceeds with real AI-coding API calls. 'y' continues; anything
    else aborts the whole run (sys.exit(1)), matching the confirmation
    pattern already used throughout run_post_processing.py."""
    print()
    print(format_cost_overview(estimates))
    print()
    decision = process_log.confirm(
        "Continue with AI coding? [y/N]: ",
        decision_point="ai_coding_cost_estimate_confirmation",
        detail=f"{len(estimates)} configuration(s) estimated for prefix '{PREFIX}'",
        outcome_no="aborted",
    )
    if not decision:
        print("Aborted: AI coding was not run.")
        sys.exit(1)
