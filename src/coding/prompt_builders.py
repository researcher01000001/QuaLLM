"""Logic of the AI-coding prompt building"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.templates import prompts

if TYPE_CHECKING:
    # Deferred to avoid a circular import: src.core.models -> src.core.process_log ->
    # src.config.project -> src.config.general -> src.config.coding -> here. Safe
    # since this module only uses CodingScheme/Tag as type annotations
    # (never constructs them), and `from __future__ import annotations`
    # makes every annotation in this file a lazy string, never evaluated at
    # class/function definition time.
    from src.core.models import CodingScheme, Tag


@dataclass(frozen=True)
class PromptContext:
    """Everything a prompt-builder function may draw on for one AI-coding
    call. Explicit, typed fields (not a loose extra_context mapping) so a
    builder's signature/body shows exactly what it uses."""

    tags: list[Tag]
    coding_scheme: CodingScheme
    valid_conditions: list[str] | None
    document_condition: str | None
    paragraph: str
    row: dict[str, Any] | None


@dataclass(frozen=True)
class DocumentPromptScheme:
    """Everything about a document's prompt context that's constant across
    all its paragraphs -- built once per document (or once per saved
    snapshot), not once per paragraph/row."""

    coding_scheme: CodingScheme
    valid_conditions: list[str] | None
    document_condition: str | None


def _condition_entry_filter(
    valid_conditions: list[str] | None,
) -> Callable[[dict[str, Any]], bool]:
    """A coding_scheme_text() entry_filter keeping an entry when it has no
    condition restriction, or when its condition is in valid_conditions
    (None = no filtering at all)."""

    def entry_filter(entry: dict[str, Any]) -> bool:
        if not entry.get("condition"):
            return True
        if valid_conditions is None:
            return True
        return entry["condition"] in valid_conditions

    return entry_filter


def build_system_prompt_base(ctx: PromptContext) -> str:
    """prompts.SYSTEM_PROMPT_PREAMBLE + the coding scheme (via
    CodingScheme.coding_scheme_text(), filtered by ctx.valid_conditions, with
    definition/note/examples/double_coding included) +
    prompts.EXPECTED_JSON_FORMAT."""
    tag_descriptions = ctx.coding_scheme.coding_scheme_text(
        _condition_entry_filter(ctx.valid_conditions),
        definition=True,
        double_coding=True,
    )
    return (
        f"{prompts.BASE_SYSTEM_PROMPT_PREAMBLE}\n"
        f"Here are the ALLOWED TAGS:\n{tag_descriptions}\n\n"
        f"{prompts.EXPECTED_JSON_FORMAT}"
    )

def build_system_prompt_v0(ctx: PromptContext) -> str:
    """prompts.SYSTEM_PROMPT_PREAMBLE + the coding scheme (via
    CodingScheme.coding_scheme_text(), filtered by ctx.valid_conditions, with
    definition/note/examples/double_coding included) +
    prompts.EXPECTED_JSON_FORMAT."""
    tag_descriptions = ctx.coding_scheme.coding_scheme_text(
        _condition_entry_filter(ctx.valid_conditions),
        definition=True,
        double_coding=True,
    )
    return (
        f"{prompts.SYSTEM_PROMPT_PREAMBLE}\n"
        f"Here are the ALLOWED TAGS:\n{tag_descriptions}\n\n"
        f"{prompts.EXPECTED_JSON_FORMAT}"
    )


def build_user_prompt_v0(ctx: PromptContext) -> str:
    """Fills prompts.USER_PROMPT_V0's single {paragraph} placeholder."""
    return prompts.USER_PROMPT_V0.format(paragraph=ctx.paragraph)


def build_system_prompt_v1(ctx: PromptContext) -> str:
    """prompts.SYSTEM_PROMPT_PREAMBLE + the coding scheme (filtered by
    ctx.valid_conditions, with definition/note/examples/double_coding
    included) + general rules + prompts.EXPECTED_JSON_FORMAT. Same tag
    formatting as build_system_prompt_v0, plus a GENERAL RULES section."""
    coding_scheme_text = ctx.coding_scheme.coding_scheme_text(
        _condition_entry_filter(ctx.valid_conditions),
        definition=True,
        note=True,
        examples=True,
        double_coding=True,
    )
    general_rules_text = ctx.coding_scheme.general_rules_text()
    return (
        f"{prompts.SYSTEM_PROMPT_PREAMBLE}\n"
        f"Here are the ALLOWED TAGS:\n{coding_scheme_text}\n\n"
        f"GENERAL RULES:\n{general_rules_text}\n\n"
        f"{prompts.EXPECTED_JSON_FORMAT}"
    )


def build_ai_generated_system_prompt(ctx: PromptContext) -> str:
    """Wraps the fully static AI_GENERATED_SYSTEM_PROMPT (no per-call
    variables) -- byte-identical output regardless of ctx."""
    return prompts.AI_GENERATED_SYSTEM_PROMPT


def build_ai_generated_user_prompt(ctx: PromptContext) -> str:
    """Fills AI_GENERATED_USER_PROMPT's single {paragraph} placeholder."""
    return prompts.AI_GENERATED_USER_PROMPT.format(paragraph=ctx.paragraph)


SYSTEM_PROMPT_BUILDERS: dict[str, Callable[[PromptContext], str]] = {
    "SYSTEM_PROMPT_BASE" : build_system_prompt_base,
    "SYSTEM_PROMPT_V0": build_system_prompt_v0,
    "SYSTEM_PROMPT_V1": build_system_prompt_v1,
    "AI_GENERATED_SYSTEM_PROMPT": build_ai_generated_system_prompt,
}

USER_PROMPT_BUILDERS: dict[str, Callable[[PromptContext], str]] = {
    "USER_PROMPT_BASE" : build_user_prompt_v0,
    "USER_PROMPT_V0": build_user_prompt_v0,
    "AI_GENERATED_USER_PROMPT": build_ai_generated_user_prompt,
}


def resolve_system_prompt_builder(name: str) -> Callable[[PromptContext], str]:
    try:
        return SYSTEM_PROMPT_BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown system prompt builder: {name}") from None


def resolve_user_prompt_builder(name: str) -> Callable[[PromptContext], str]:
    try:
        return USER_PROMPT_BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown user prompt builder: {name}") from None
