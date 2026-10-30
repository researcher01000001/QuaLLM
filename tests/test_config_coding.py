import pytest

from src.coding.prompt_builders import (
    build_system_prompt_v0,
    build_system_prompt_v1,
    build_user_prompt_v0,
)
from src.config.coding import CodingConfiguration


def test_configuration_name_uses_required_short_name():
    config = CodingConfiguration(
        short_name="Sonnet T1",
        model_name="anthropic/claude-sonnet-4.6",
        temperature=1,
    )

    assert config.configuration_name == "sonnet-t1"


def test_configuration_metadata_serializes_provider_tuple():
    config = CodingConfiguration(
        short_name="test-config",
        model_name="test/model",
        provider_names=("provider",),
    )

    metadata = config.to_metadata()

    assert metadata["configuration_name"] == "test-config"
    assert metadata["short_name"] == "test-config"
    assert metadata["provider_names"] == ["provider"]
    assert metadata["prompt_caching"] is True
    assert metadata["iterations"] == 1
    assert metadata["consensus_threshold"] == 0.5


def test_configuration_metadata_serializes_prompt_caching_opt_out():
    config = CodingConfiguration(short_name="test-config", prompt_caching=False)

    assert config.to_metadata()["prompt_caching"] is False


def test_configuration_metadata_serializes_consensus_settings():
    config = CodingConfiguration(
        short_name="test-config",
        iterations=5,
        consensus_threshold=0.6,
    )

    metadata = config.to_metadata()

    assert metadata["iterations"] == 5
    assert metadata["consensus_threshold"] == 0.6


def test_configuration_has_no_manual_cost_estimation_factor_fields():
    """completion_token_factor/reasoning_token_factor were removed once the
    first-row probe (src.cost_estimation.estimation.recalculate_from_probe()) made a
    real, measured observation available on every run instead -- there is
    no longer a manual/config way to set these."""
    with pytest.raises(TypeError):
        CodingConfiguration(short_name="test-config", completion_token_factor=0.5)
    with pytest.raises(TypeError):
        CodingConfiguration(short_name="test-config", reasoning_token_factor=0.2)


def test_configuration_metadata_serializes_reasoning_fields_default():
    config = CodingConfiguration(short_name="test-config")

    metadata = config.to_metadata()

    assert metadata["reasoning"] is False
    assert metadata["reasoning_effort"] is None


# --- reasoning / reasoning_effort -------------------------------------------


def test_reasoning_defaults_off_and_effort_none():
    config = CodingConfiguration(short_name="test-config")

    assert config.reasoning is False
    assert config.reasoning_effort is None
    assert config.effective_reasoning_effort is None


def test_effective_reasoning_effort_defaults_to_medium_when_reasoning_enabled():
    config = CodingConfiguration(short_name="test-config", reasoning=True)

    assert config.effective_reasoning_effort == "medium"


def test_effective_reasoning_effort_uses_explicit_effort():
    config = CodingConfiguration(short_name="test-config", reasoning=True, reasoning_effort="high")

    assert config.effective_reasoning_effort == "high"


def test_reasoning_effort_rejects_invalid_value():
    with pytest.raises(ValueError, match="reasoning_effort"):
        CodingConfiguration(short_name="test-config", reasoning=True, reasoning_effort="bogus")


def test_reasoning_effort_rejects_being_set_without_reasoning_enabled():
    with pytest.raises(ValueError, match="reasoning=False"):
        CodingConfiguration(short_name="test-config", reasoning=False, reasoning_effort="high")


def test_system_prompt_builder_resolves_default_name_by_function_identity():
    config = CodingConfiguration(short_name="test-config")

    assert config.system_prompt_builder is build_system_prompt_v0


def test_system_prompt_builder_resolves_configured_name():
    config = CodingConfiguration(short_name="test-config", system_prompt_name="SYSTEM_PROMPT_V1")

    assert config.system_prompt_builder is build_system_prompt_v1


def test_user_prompt_builder_resolves_default_name_by_function_identity():
    config = CodingConfiguration(short_name="test-config")

    assert config.user_prompt_builder is build_user_prompt_v0


def test_system_prompt_builder_raises_on_unknown_name():
    config = CodingConfiguration(short_name="test-config", system_prompt_name="NOPE")

    with pytest.raises(ValueError, match="NOPE"):
        config.system_prompt_builder


def test_user_prompt_builder_raises_on_unknown_name():
    config = CodingConfiguration(short_name="test-config", user_prompt_name="NOPE")

    with pytest.raises(ValueError, match="NOPE"):
        config.user_prompt_builder
