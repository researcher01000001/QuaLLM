from unittest.mock import Mock

import pytest

from src.config.coding import CodingConfiguration
from src.core.models import CodingScheme
from src.cost_estimation.calibration import ModelCalibration
from src.cost_estimation.estimation import (
    ConfigurationCostEstimate,
    CostSummaryRow,
    ProbeResult,
    confirm_cost_estimate,
    confirm_recalculated_estimate,
    estimate_configuration_cost,
    estimate_run_cost,
    format_cost_overview,
    format_cost_summary,
    probe_first_paragraph,
    recalculate_from_probe,
    resolve_completion_token_factor,
    resolve_reasoning_token_factor,
)
from src.cost_estimation.pricing import ModelPricing

_EMPTY_SCHEME = CodingScheme(coding_scheme=[], general_rules=[], medical_knowledge={})
_CALIBRATION_ENTRY = ModelCalibration(
    completion_token_factor=0.33,
    reasoning_token_factor=0.1,
    sample_calls=5,
    last_updated="now",
    last_prefix="p",
)


# --- resolve_completion_token_factor() / resolve_reasoning_token_factor() --
#
# There is no manual/config way to set these factors any more -- the only
# sources are a real, probed override (recalculate_from_probe(), passed
# through as the `override` param) and learned calibration.


def test_resolve_completion_token_factor_prefers_override_over_calibration():
    config = CodingConfiguration(short_name="c", model_name="m")

    assert resolve_completion_token_factor(
        config, {"m": {"none": _CALIBRATION_ENTRY}}, override=0.7
    ) == (0.7, "probe")


def test_resolve_completion_token_factor_falls_back_to_calibration():
    config = CodingConfiguration(short_name="c", model_name="m")

    assert resolve_completion_token_factor(config, {"m": {"none": _CALIBRATION_ENTRY}}) == (
        0.33,
        "calibration(none)",
    )


def test_resolve_completion_token_factor_falls_back_across_effort_buckets():
    """A model calibrated only at a different effort level is still more
    informative than nothing -- clearly labeled as a cross-effort fallback."""
    config = CodingConfiguration(
        short_name="c", model_name="m", reasoning=True, reasoning_effort="high"
    )

    factor, source = resolve_completion_token_factor(config, {"m": {"medium": _CALIBRATION_ENTRY}})

    assert factor == 0.33
    assert "cross-effort fallback" in source


def test_resolve_completion_token_factor_unresolved_when_nothing_available():
    config = CodingConfiguration(short_name="c", model_name="m")

    assert resolve_completion_token_factor(config, {}) == (None, "unresolved")


def test_resolve_reasoning_token_factor_prefers_override_over_calibration():
    config = CodingConfiguration(short_name="c", model_name="m", reasoning=True)

    assert resolve_reasoning_token_factor(
        config, {"m": {"medium": _CALIBRATION_ENTRY}}, override=0.9
    ) == (0.9, "probe")


def test_resolve_reasoning_token_factor_short_circuits_when_reasoning_disabled():
    config = CodingConfiguration(short_name="c", model_name="m", reasoning=False)

    # Even with an override given, a call that won't request reasoning at
    # all never produces reasoning tokens.
    assert resolve_reasoning_token_factor(
        config, {"m": {"medium": _CALIBRATION_ENTRY}}, override=0.9
    ) == (0.0, "reasoning_disabled")


def test_resolve_reasoning_token_factor_unresolved_when_nothing_available():
    config = CodingConfiguration(short_name="c", model_name="m", reasoning=True)

    assert resolve_reasoning_token_factor(config, {}) == (None, "unresolved")


# --- ConfigurationCostEstimate computed properties -------------------------


def test_estimated_completion_tokens_none_without_factor():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        input_tokens=1000,
    )

    assert estimate.estimated_completion_tokens is None
    assert estimate.estimated_reasoning_tokens is None
    assert estimate.estimated_cost is None


def test_estimated_completion_tokens_uses_paragraph_tokens_not_input_tokens():
    """The full prompt (input_tokens) includes the largely-fixed
    coding-scheme/instructions overhead; only paragraph_tokens should drive
    the completion-token estimate."""
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        paragraph_tokens=200,
        input_tokens=5000,  # large fixed overhead -- must not affect the estimate below
        completion_token_factor=0.5,
    )

    assert estimate.estimated_completion_tokens == 100


def test_estimated_cost_does_not_double_count_reasoning_tokens():
    """Regression test: reasoning tokens are a SUBSET of completion tokens
    (matching OpenRouter/OpenAI usage semantics), not additional to them --
    visible_output_tokens + reasoning_tokens must equal completion_tokens,
    each billed once."""
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort="medium",
        iterations=1,
        paragraph_tokens=1000,
        input_tokens=1000,
        completion_token_factor=0.5,
        reasoning_token_factor=0.2,
        input_price_per_token=0.001,
        completion_price_per_token=0.002,
    )

    assert estimate.estimated_completion_tokens == 500
    assert estimate.estimated_reasoning_tokens == 100
    assert estimate.estimated_visible_output_tokens == 400
    # completion_tokens (500) is billed once, split 400 visible + 100
    # reasoning -- NOT 500 + 100 = 600.
    assert estimate.estimated_cost == pytest.approx(1000 * 0.001 + 500 * 0.002)


def test_estimated_cost_applies_cache_prices_and_keeps_uncached_comparison():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        prompt_caching=True,
        paragraph_tokens=1000,
        input_tokens=1000,
        user_tokens=200,
        cache_write_tokens=100,
        cache_read_tokens=700,
        completion_token_factor=0.5,
        reasoning_token_factor=0.0,
        input_price_per_token=0.001,
        cache_read_price_per_token=0.0001,
        cache_write_price_per_token=0.00125,
        completion_price_per_token=0.002,
    )

    assert estimate.cache_pricing_applied is True
    assert estimate.estimated_input_cost == pytest.approx(0.2 + 0.125 + 0.07)
    assert estimate.estimated_cost == pytest.approx(0.395 + 1.0)
    assert estimate.estimated_uncached_cost == pytest.approx(1.0 + 1.0)


def test_estimated_cost_falls_back_to_normal_input_price_without_cache_rates():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        prompt_caching=True,
        paragraph_tokens=100,
        input_tokens=1000,
        user_tokens=200,
        cache_write_tokens=100,
        cache_read_tokens=700,
        completion_token_factor=0.5,
        reasoning_token_factor=0.0,
        input_price_per_token=0.001,
        completion_price_per_token=0.002,
    )

    assert estimate.cache_pricing_applied is False
    assert estimate.estimated_cost == estimate.estimated_uncached_cost


def test_estimated_cost_uses_distinct_reasoning_price_when_available():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort="medium",
        iterations=1,
        paragraph_tokens=1000,
        input_tokens=1000,
        completion_token_factor=0.5,
        reasoning_token_factor=0.2,
        input_price_per_token=0.001,
        completion_price_per_token=0.002,
        reasoning_price_per_token=0.01,
    )

    assert estimate.estimated_cost == pytest.approx(
        1000 * 0.001 + 400 * 0.002 + 100 * 0.01
    )


# --- estimate_configuration_cost() ------------------------------------------


def test_estimate_configuration_cost_counts_tokens_and_scales_by_iterations(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.\n"
        "Zweiter Testabsatz mit genug Zeichen.",
        encoding="utf-8",
    )
    config = CodingConfiguration(short_name="c", model_name="m", iterations=3)
    pricing = {"m": ModelPricing(prompt_price_per_token=0.001, completion_price_per_token=0.002)}
    calibration = {"m": {"none": _CALIBRATION_ENTRY}}

    estimate = estimate_configuration_cost(
        config, [], tmp_path, _EMPTY_SCHEME, pricing, calibration
    )

    assert estimate.paragraph_count == 2 * 3
    assert estimate.completion_token_factor_source == "calibration(none)"
    # reasoning=False (default) short-circuits the reasoning factor.
    assert estimate.reasoning_token_factor_source == "reasoning_disabled"
    assert estimate.input_price_per_token == 0.001
    assert estimate.completion_price_per_token == 0.002
    assert estimate.unresolved_reasons == []
    assert estimate.estimated_cost is not None


def test_estimate_configuration_cost_groups_identical_system_prompts_for_caching(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.\n"
        "Zweiter Testabsatz mit genug Zeichen.",
        encoding="utf-8",
    )
    config = CodingConfiguration(short_name="c", model_name="m", iterations=3)
    pricing = {
        "m": ModelPricing(
            prompt_price_per_token=0.001,
            completion_price_per_token=0.002,
            cache_read_price_per_token=0.0001,
            cache_write_price_per_token=0.00125,
        )
    }

    estimate = estimate_configuration_cost(
        config,
        [],
        tmp_path,
        _EMPTY_SCHEME,
        pricing,
        {"m": {"none": _CALIBRATION_ENTRY}},
    )

    assert estimate.prompt_caching is True
    assert estimate.cache_pricing_applied is True
    assert estimate.input_tokens == estimate.system_tokens + estimate.user_tokens
    # Both paragraphs render the same system prompt. Across three iterations,
    # one system-prefix occurrence writes and the other five read the cache.
    assert estimate.cache_write_tokens * 6 == estimate.system_tokens
    assert estimate.cache_read_tokens == estimate.system_tokens - estimate.cache_write_tokens
    assert estimate.estimated_cost is not None
    assert estimate.estimated_uncached_cost is not None
    assert estimate.estimated_cost < estimate.estimated_uncached_cost


def test_estimate_configuration_cost_does_not_assume_cache_when_opted_out(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Ein ausreichend langer Testabsatz.", encoding="utf-8")
    config = CodingConfiguration(
        short_name="c", model_name="m", prompt_caching=False
    )
    pricing = {
        "m": ModelPricing(
            prompt_price_per_token=0.001,
            completion_price_per_token=0.002,
            cache_read_price_per_token=0.0001,
            cache_write_price_per_token=0.00125,
        )
    }

    estimate = estimate_configuration_cost(
        config,
        [],
        tmp_path,
        _EMPTY_SCHEME,
        pricing,
        {"m": {"none": _CALIBRATION_ENTRY}},
    )

    assert estimate.cache_write_tokens == 0
    assert estimate.cache_read_tokens == 0
    assert estimate.cache_pricing_applied is False
    assert estimate.estimated_cost == estimate.estimated_uncached_cost


def test_estimate_configuration_cost_uses_override_over_calibration(tmp_path, monkeypatch):
    """completion_token_factor_override/reasoning_token_factor_override --
    only ever passed by recalculate_from_probe() -- take priority over a
    present calibration entry."""
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein ausreichend langer Testabsatz fuer die Verarbeitung.", encoding="utf-8"
    )
    config = CodingConfiguration(short_name="c", model_name="m", reasoning=True, iterations=1)
    calibration = {"m": {"medium": _CALIBRATION_ENTRY}}

    estimate = estimate_configuration_cost(
        config,
        [],
        tmp_path,
        _EMPTY_SCHEME,
        pricing={},
        calibration=calibration,
        completion_token_factor_override=9.0,
        reasoning_token_factor_override=0.75,
    )

    assert estimate.completion_token_factor == 9.0
    assert estimate.completion_token_factor_source == "probe"
    assert estimate.reasoning_token_factor == 0.75
    assert estimate.reasoning_token_factor_source == "probe"


def test_estimate_configuration_cost_skips_paragraphs_under_ten_characters(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("short\nEin ausreichend langer Testabsatz.", encoding="utf-8")
    config = CodingConfiguration(short_name="c", model_name="m", iterations=1)

    estimate = estimate_configuration_cost(
        config, [], tmp_path, _EMPTY_SCHEME, pricing={}, calibration={}
    )

    assert estimate.paragraph_count == 1


def test_estimate_configuration_cost_reports_unresolved_factors_and_pricing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein ausreichend langer Testabsatz fuer die Verarbeitung.", encoding="utf-8"
    )
    config = CodingConfiguration(short_name="c", model_name="m", iterations=1)

    estimate = estimate_configuration_cost(
        config, [], tmp_path, _EMPTY_SCHEME, pricing={}, calibration={}
    )

    assert estimate.completion_token_factor is None
    assert estimate.estimated_cost is None
    assert any("completion_token_factor" in reason for reason in estimate.unresolved_reasons)
    assert any("pricing" in reason for reason in estimate.unresolved_reasons)


# --- estimate_run_cost() ----------------------------------------------------


def test_estimate_run_cost_fetches_pricing_once_and_returns_one_estimate_per_configuration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.cost_estimation.estimation.load_prompt_context", lambda *a, **k: _EMPTY_SCHEME
    )
    monkeypatch.setattr("src.cost_estimation.estimation.load_calibration", lambda: {})

    captured_model_names: set[str] = set()

    def fake_fetch_pricing(model_names):
        captured_model_names.update(model_names)
        return {name: None for name in model_names}

    monkeypatch.setattr("src.cost_estimation.estimation.fetch_model_pricing", fake_fetch_pricing)

    calls: list[str] = []

    def fake_estimate_configuration_cost(
        coding_config, tags, source_dir, prompt_context, pricing, calibration
    ):
        calls.append(coding_config.short_name)
        return ConfigurationCostEstimate(
            configuration_name=coding_config.configuration_name,
            short_name=coding_config.short_name,
            model_name=coding_config.model_name,
            effort=coding_config.effective_reasoning_effort,
            iterations=coding_config.iterations,
        )

    monkeypatch.setattr(
        "src.cost_estimation.estimation.estimate_configuration_cost", fake_estimate_configuration_cost
    )

    config_a = CodingConfiguration(short_name="a", model_name="model-a")
    config_b = CodingConfiguration(short_name="b", model_name="model-b")

    run_cost_estimate = estimate_run_cost([config_a, config_b], [], tmp_path)

    assert calls == ["a", "b"]
    assert captured_model_names == {"model-a", "model-b"}
    assert [e.short_name for e in run_cost_estimate.estimates] == ["a", "b"]
    assert run_cost_estimate.prompt_context is _EMPTY_SCHEME
    assert run_cost_estimate.pricing == {"model-a": None, "model-b": None}
    assert run_cost_estimate.calibration == {}


# --- format_cost_overview() -------------------------------------------------


def test_format_cost_overview_shows_na_and_partial_note_for_unresolved_estimate():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        paragraph_count=5,
        input_tokens=1000,
        unresolved_reasons=["no completion_token_factor available for model 'm'"],
    )

    overview = format_cost_overview([estimate])

    assert "N/A" in overview
    assert "PARTIAL estimate" in overview
    assert "no completion_token_factor available" in overview


def test_format_cost_overview_no_partial_note_when_fully_resolved():
    estimate = ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort="medium",
        iterations=1,
        paragraph_count=5,
        input_tokens=1000,
        completion_token_factor=0.5,
        reasoning_token_factor=0.0,
        input_price_per_token=0.001,
        completion_price_per_token=0.002,
    )

    overview = format_cost_overview([estimate])

    assert "PARTIAL estimate" not in overview
    assert "N/A" not in overview
    assert "medium" in overview
    assert "No-cache" in overview


# --- confirm_cost_estimate() -----------------------------------------------


def test_confirm_cost_estimate_aborts_on_non_y_answer(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.cost_estimation.estimation.process_log.current", lambda: mock_log)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    estimate = ConfigurationCostEstimate(
        configuration_name="c", short_name="c", model_name="m", effort=None, iterations=1
    )

    with pytest.raises(SystemExit) as exc_info:
        confirm_cost_estimate([estimate])

    assert exc_info.value.code == 1
    mock_log.log_audit_decision.assert_called_once()
    assert mock_log.log_audit_decision.call_args.kwargs["outcome"] == "aborted"


def test_confirm_cost_estimate_continues_on_y_answer(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.cost_estimation.estimation.process_log.current", lambda: mock_log)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    estimate = ConfigurationCostEstimate(
        configuration_name="c", short_name="c", model_name="m", effort=None, iterations=1
    )

    confirm_cost_estimate([estimate])

    mock_log.log_audit_decision.assert_called_once()
    assert mock_log.log_audit_decision.call_args.kwargs["outcome"] == "confirmed"


# --- probe_first_paragraph() / recalculate_from_probe() --------------------


def test_probe_first_paragraph_calls_model_on_first_eligible_paragraph(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("short\nEin ausreichend langer Testabsatz.", encoding="utf-8")
    config = CodingConfiguration(short_name="c", model_name="m", iterations=1)

    captured = {}

    def fake_analyse(i, paragraph, total_paragraphs, filepath, coding_config, tags, row, scheme):
        captured["i"] = i
        captured["paragraph"] = paragraph
        return [{"phrase": "x", "usage": {"paragraph_tokens": 10, "completion_tokens": 20}}]

    monkeypatch.setattr("src.cost_estimation.estimation.analyse_with_genai", fake_analyse)

    probe = probe_first_paragraph(config, [], tmp_path, _EMPTY_SCHEME)

    assert probe is not None
    assert probe.filepath == doc_path
    # "short" (index 0) is under 10 chars and skipped -- the eligible
    # paragraph is at index 1.
    assert probe.paragraph_index == 1
    assert captured["paragraph"] == "Ein ausreichend langer Testabsatz."
    assert probe.usage == {"paragraph_tokens": 10, "completion_tokens": 20}


def test_probe_first_paragraph_returns_none_when_nothing_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("short", encoding="utf-8")
    config = CodingConfiguration(short_name="c", model_name="m", iterations=1)

    probe = probe_first_paragraph(config, [], tmp_path, _EMPTY_SCHEME)

    assert probe is None


def test_recalculate_from_probe_uses_observed_ratio_over_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: None)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Ein ausreichend langer Testabsatz.", encoding="utf-8")
    config = CodingConfiguration(short_name="c", model_name="m", iterations=1)
    pricing = {"m": ModelPricing(prompt_price_per_token=0.001, completion_price_per_token=0.002)}
    # A calibration entry is present for this model too -- the probe's real,
    # just-observed ratio must still win over it.
    calibration = {"m": {"none": _CALIBRATION_ENTRY}}
    probe = ProbeResult(
        filepath=doc_path,
        paragraph_index=0,
        results=[{}],
        usage={"paragraph_tokens": 100, "completion_tokens": 250},
    )

    recalculated = recalculate_from_probe(
        config, [], tmp_path, _EMPTY_SCHEME, pricing, calibration, probe=probe
    )

    assert recalculated.completion_token_factor == pytest.approx(2.5)
    assert recalculated.completion_token_factor_source == "probe"
    assert recalculated.completion_token_factor != _CALIBRATION_ENTRY.completion_token_factor


# --- confirm_recalculated_estimate() ----------------------------------------


def _estimate_with_cost(cost: float | None) -> ConfigurationCostEstimate:
    if cost is None:
        return ConfigurationCostEstimate(
            configuration_name="c", short_name="c", model_name="m", effort=None, iterations=1
        )
    return ConfigurationCostEstimate(
        configuration_name="c",
        short_name="c",
        model_name="m",
        effort=None,
        iterations=1,
        paragraph_tokens=1,
        input_tokens=1,
        completion_token_factor=cost,
        input_price_per_token=1.0,
        completion_price_per_token=1.0,
    )


def test_confirm_recalculated_estimate_skips_when_recalculated_still_unresolved(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.cost_estimation.estimation.process_log.current", lambda: mock_log)
    config = CodingConfiguration(short_name="c", model_name="m")

    confirm_recalculated_estimate(
        config, _estimate_with_cost(1.0), _estimate_with_cost(None), threshold_fraction=0.1
    )

    mock_log.log_audit_decision.assert_not_called()


def test_confirm_recalculated_estimate_always_confirms_when_original_was_unresolved(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.cost_estimation.estimation.process_log.current", lambda: mock_log)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    config = CodingConfiguration(short_name="c", model_name="m")

    confirm_recalculated_estimate(
        config, _estimate_with_cost(None), _estimate_with_cost(1.0), threshold_fraction=0.1
    )

    mock_log.log_audit_decision.assert_called_once()


def test_confirm_recalculated_estimate_prompts_only_above_threshold(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.cost_estimation.estimation.process_log.current", lambda: mock_log)
    config = CodingConfiguration(short_name="c", model_name="m")

    # +5% is under the 10% threshold -- no prompt.
    confirm_recalculated_estimate(
        config, _estimate_with_cost(1.0), _estimate_with_cost(1.05), threshold_fraction=0.1
    )
    mock_log.log_audit_decision.assert_not_called()

    # +50% is over the 10% threshold -- prompts, and aborts on "n".
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(SystemExit):
        confirm_recalculated_estimate(
            config, _estimate_with_cost(1.0), _estimate_with_cost(1.5), threshold_fraction=0.1
        )
    mock_log.log_audit_decision.assert_called_once()


# --- format_cost_summary() --------------------------------------------------


def test_format_cost_summary_shows_all_three_figures_and_totals():
    rows = [
        CostSummaryRow(
            short_name="a",
            model_name="model-a",
            initial_estimate=1.0,
            recalculated_estimate=1.1,
            actual_cost=1.05,
        ),
        CostSummaryRow(
            short_name="b",
            model_name="model-b",
            initial_estimate=2.0,
            recalculated_estimate=None,
            actual_cost=1.9,
        ),
    ]

    summary = format_cost_summary(rows)

    assert "Initial estimate:      $1.0000" in summary
    assert "Second estimation:     $1.1000" in summary
    assert "Actual cost:           $1.0500" in summary
    assert "N/A" in summary
    assert "TOTAL:" in summary
    # Missing recalculated_estimate for "b" is excluded from the total, not
    # treated as 0.
    assert "Second estimation:     $1.1000" in summary  # total == only row a's value
