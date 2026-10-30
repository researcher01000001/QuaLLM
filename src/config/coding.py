"""CodingConfiguration model for per-configuration AI models"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from src.coding.prompt_builders import (
    PromptContext,
    resolve_system_prompt_builder,
    resolve_user_prompt_builder,
)

# OpenRouter reasoning-effort levels we support
VALID_REASONING_EFFORTS = ("low", "medium", "high")


@dataclass(frozen=True, kw_only=True)
class CodingConfiguration:
    """Runtime settings for one AI-coding configuration."""
    short_name: str
    model_name: str = "anthropic/claude-sonnet-4.6"
    temperature: float = 0
    seed: int | None = 42
    provider_names: tuple[str, ...] | None = None
    prompt_caching: bool = True  # Enable Prompt caching of the System Prompt
    max_threads: int = 10
    iterations: int = 1
    consensus_threshold: float = 0.5
    system_prompt_name: str = "SYSTEM_PROMPT_V0"
    user_prompt_name: str = "USER_PROMPT_V0"
    reasoning: bool = False  # Enable reasoning for the model
    reasoning_effort: str | None = None  # Reasoning effort when reasoning is enabled

    # Fields that are included in the metadata
    METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "configuration_name",
        "model_name",
        "short_name",
        "temperature",
        "seed",
        "provider_names",
        "prompt_caching",
        "max_threads",
        "iterations",
        "consensus_threshold",
        "system_prompt_name",
        "user_prompt_name",
        "reasoning",
        "reasoning_effort",
    )

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None and self.reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {VALID_REASONING_EFFORTS} or None, "
                f"got {self.reasoning_effort!r}"
            )
        if self.reasoning_effort is not None and not self.reasoning:
            raise ValueError(
                "reasoning_effort is set but reasoning=False -- it would never take "
                "effect; set reasoning=True or leave reasoning_effort unset"
            )

    @property
    def effective_reasoning_effort(self) -> str | None:
        """The effort level that will be sent."""
        if not self.reasoning:
            return None
        return self.reasoning_effort or "medium"

    @property
    def configuration_name(self) -> str:
        """Return a stable, filesystem-safe name for this configuration."""
        return _slugify(self.short_name)

    @property
    def system_prompt_builder(self) -> Callable[[PromptContext], str]:
        """Resolve the configured system prompt builder function by name."""
        return resolve_system_prompt_builder(self.system_prompt_name)

    @property
    def user_prompt_builder(self) -> Callable[[PromptContext], str]:
        """Resolve the configured user prompt builder function by name."""
        return resolve_user_prompt_builder(self.user_prompt_name)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the configuration for run metadata output."""
        values: dict[str, Any] = {
            field: getattr(self, field) for field in self.METADATA_FIELDS
        }
        values["provider_names"] = list(self.provider_names) if self.provider_names else None
        return values


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "default"
