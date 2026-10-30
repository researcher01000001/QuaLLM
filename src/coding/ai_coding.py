import concurrent.futures
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

from src.coding.prompt_builders import DocumentPromptScheme, PromptContext
from src.config.coding import CodingConfiguration, _slugify
from src.config.general import (
    AI_CODING_FILES,
    AI_CODING_MAX_PARAGRAPHS,
    AI_CODING_MIN_PARAGRAPH_CHARS,
    COST_ESTIMATION_ENABLED,
    OPENROUTER_API_KEY,
    WRITE_METADATA,
)
from src.config.project import CODES_FILE, DESCRIPTION_FILE, TXT_FILES
from src.core import process_log
from src.core.models import CodingScheme, Tag, tags_for_condition
from src.cost_estimation.tokenization import count_tokens
from src.dataset import document_content
from src.dataset.document_identity import condition_for, condition_match
from src.dataset.selection import resolve_rows_for_txt_file
from src.io.io_utils import write_json, get_matching_files, resolve_output_dir
from src.io.loaders import load_prompt_context, load_tags

# Initialize the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Stands in for a specific row's transcript text in a saved prompt snapshot
# (see render_general_prompt()) -- the snapshot is meant to show the general
# (coding-scheme) shape of a configuration's prompt, not any one document's
# content.
PARAGRAPH_PLACEHOLDER = "[paragraph text]"

_EMPTY_CODING_SCHEME = CodingScheme(coding_scheme=[], general_rules=[], medical_knowledge={})


def get_ai_iteration_dir(
    coding_config: CodingConfiguration,
    iteration: int,
    base_dir: Path = AI_CODING_FILES,
) -> Path:
    """Return the nested output directory for one AI configuration iteration.

    base_dir defaults to AI_CODING_FILES, which is already specific to this
    PREFIX and RUN_TYPE."""
    return base_dir / coding_config.configuration_name / f"iteration-{iteration}"


def render_prompt(
    coding_config: CodingConfiguration,
    tags: list[Tag],
    paragraph: str,
    row: dict[str, Any] | None = None,
    scheme: DocumentPromptScheme | None = None,
) -> tuple[str, str]:
    """The exact (system, user) message strings analyse_with_genai() sends
    for one paragraph -- built by calling coding_config's registered
    system/user prompt-builder functions (see src.coding.prompt_builders) with one
    shared PromptContext, so either builder can reference any of these
    fields regardless of its role."""
    ctx = PromptContext(
        tags=tags,
        coding_scheme=scheme.coding_scheme if scheme else _EMPTY_CODING_SCHEME,
        valid_conditions=scheme.valid_conditions if scheme else None,
        document_condition=scheme.document_condition if scheme else None,
        paragraph=paragraph,
        row=row,
    )
    system = coding_config.system_prompt_builder(ctx)
    user = coding_config.user_prompt_builder(ctx)
    return system, user


def render_general_prompt(
    coding_config: CodingConfiguration,
    tags: list[Tag],
    scheme: DocumentPromptScheme | None = None,
) -> tuple[str, str]:
    """The system/user prompt with every general (coding-scheme) piece
    resolved from the full tag list, but the row-specific paragraph text
    replaced by PARAGRAPH_PLACEHOLDER and no row -- a representative
    snapshot of this configuration's prompt shape, not tied to any one
    document/row."""
    return render_prompt(coding_config, tags, PARAGRAPH_PLACEHOLDER, None, scheme)


def write_prompt_snapshots(
    entries: list[tuple[str, CodingConfiguration, str, str]],
    active_conditions: set[str] | None = None,
    output_dir: Path = AI_CODING_FILES,
) -> list[Path]:
    """Write one file per document condition, each grouping that condition's
    (coding_config, system, user) entries' resolved prompt (see
    render_general_prompt()) under a heading naming its configuration(s) --
    configurations that resolve to byte-identical (system, user) text within
    the same condition share one heading/section instead of repeating the
    same content once per configuration. Prompts differ by condition because
    each document's valid_conditions (see document_identity.condition_match())
    filters which tags are allowed, which changes the rendered prompt text.

    entries normally cover every condition in the dataset (see
    document_content.known_conditions()), not just the ones exercised by the
    current RUN_TYPE's documents -- active_conditions, when given, is that
    narrower "actually present in this run" set, and any condition not in it
    gets a "__unused" marker appended to its filename so it's still generated
    for reference but clearly flagged as not exercised this run. None (the
    default) marks nothing as unused.

    Overwrites output_dir/prompt_{condition-slug}[__unused].txt
    on every call, so re-running the pipeline with the same PREFIX/RUN_TYPE
    simply replaces them. Returns the list of paths written, in the order
    conditions first appear in entries."""
    groups_by_condition: dict[str, dict[tuple[str, str], list[str]]] = {}
    order_by_condition: dict[str, list[tuple[str, str]]] = {}
    condition_order: list[str] = []
    for condition, coding_config, system, user in entries:
        if condition not in groups_by_condition:
            groups_by_condition[condition] = {}
            order_by_condition[condition] = []
            condition_order.append(condition)
        groups = groups_by_condition[condition]
        order = order_by_condition[condition]
        key = (system, user)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(coding_config.configuration_name)

    output_paths = []
    for condition in condition_order:
        groups = groups_by_condition[condition]
        order = order_by_condition[condition]
        sections = []
        for key in order:
            system, user = key
            names = ", ".join(groups[key])
            sections.append(
                f"{'=' * 60}\nConfiguration(s): {names}\n{'=' * 60}\n\n"
                f"SYSTEM MESSAGE\n{'-' * 40}\n{system}\n\n"
                f"USER MESSAGE\n{'-' * 40}\n{user}\n"
            )

        slug = _slugify(condition)
        if active_conditions is not None and condition not in active_conditions:
            slug += "__unused"
        output_path = output_dir / f"prompt_{slug}.txt"
        resolve_output_dir(output_path)
        output_path.write_text("\n\n".join(sections), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


@dataclass
class DocumentPrompts:
    """One document's resolved paragraphs/rows/scheme/condition-filtered
    tags -- everything render_prompt() needs per paragraph, without any
    document actually being sent to the model yet. Shared by
    process_documents() (the real run) and
    src.cost_estimation.estimation.estimate_configuration_cost() (the pre-run token
    estimate), so both can never disagree about which paragraphs/rows apply
    to a document."""

    filepath: Path
    scheme: DocumentPromptScheme
    tags: list[Tag]
    paragraphs: list[str]
    rows: list[dict[str, Any] | None]


def prepare_document(
    filepath: Path,
    coding_config: CodingConfiguration,
    tags: list[Tag],
    prompt_context: CodingScheme,
    *,
    log_warnings: bool = True,
) -> DocumentPrompts | None:
    """Resolve filepath into a DocumentPrompts, or None if the file doesn't
    exist. log_warnings=False suppresses the missing-file/unresolved-rows
    process_log entries -- used by the cost estimator so a dry-run estimate
    doesn't double-log warnings that the real run will log again anyway."""
    if not filepath.exists():
        if log_warnings:
            message = f"Warning: Could not find document text for {filepath.name}"
            print(message)
            process_log.current().log_missing_source_document(
                severity="Warning",
                source_name=filepath.name,
                context="ai_coding",
                message=message,
                step=f"{process_log.STEP_RUN_MAIN}/{coding_config.configuration_name}",
            )
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Paragraph Chunking
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Rows are resolved against the whole file (the row-count check needs every
    # paragraph) and only then cut down to AI_CODING_MAX_PARAGRAPHS.
    resolved_rows = resolve_rows_for_txt_file(filepath, paragraphs)
    if resolved_rows is None:
        if log_warnings:
            message = (
                f"Warning: Could not resolve source rows for {filepath.name} "
                f"({len(paragraphs)} paragraph(s)) -- prompts for this document "
                "will have no row data."
            )
            print(message)
            process_log.current().log_general_warning(
                severity="Warning",
                source_name=filepath.name,
                message=message,
                step=f"{process_log.STEP_RUN_MAIN}/{coding_config.configuration_name}",
            )
        # A homogeneous None-per-paragraph fallback is a valid
        # list[dict[str, Any] | None]; list's generic invariance just can't
        # express that from a bare `[None] * n` literal.
        resolved_rows = cast(list[dict[str, Any] | None], [None] * len(paragraphs))

    if AI_CODING_MAX_PARAGRAPHS:
        paragraphs = paragraphs[:AI_CODING_MAX_PARAGRAPHS]
        resolved_rows = resolved_rows[:AI_CODING_MAX_PARAGRAPHS]

    valid_conditions = condition_match(filepath.stem)
    document_tags = tags_for_condition(tags, valid_conditions)
    scheme = DocumentPromptScheme(
        coding_scheme=prompt_context,
        valid_conditions=valid_conditions,
        document_condition=condition_for(filepath.stem),
    )

    return DocumentPrompts(
        filepath=filepath,
        scheme=scheme,
        tags=document_tags,
        paragraphs=paragraphs,
        rows=resolved_rows,
    )


def _best_effort_display_value(filepath: str, row: int) -> str | None:
    """Best-effort row -> display value lookup for AI-coding error logs.

    This runs on text chunked independently of DocumentLoader.load() (see
    docs/architecture-notes.md's note on divergent paragraph-splitting), so
    ``row`` isn't guaranteed to line up with the real Document's rows -- any
    failure (unknown document, out-of-range row, bad path) is swallowed and
    yields None rather than risking a wrong or crashing lookup.
    """
    try:
        return document_content.display_column_for(Path(filepath).stem).values.get(row)
    except Exception:
        return None


def _extract_usage(response: Any, paragraph: str) -> dict[str, Any] | None:
    """Pull actual token/cost usage off an OpenRouter chat-completion
    response, requires the request to have been sent with
    extra_body={"usage": {"include": True}} (see analyse_with_genai()) for
    cost, cache, and reasoning-token details to be populated. Returns None
    when no usage object is present at all.

    Also includes "paragraph_tokens" -- a local tokenization of just the
    paragraph text (not part of the API response) -- so
    src.cost_estimation.calibration.update_calibration() can calibrate
    completion_token_factor against the same basis
    src.cost_estimation.estimation.ConfigurationCostEstimate.estimated_completion_tokens
    uses to apply it (paragraph tokens, not full prompt tokens)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cost": getattr(usage, "cost", None),
        "cache_discount": getattr(
            usage, "cache_discount", getattr(response, "cache_discount", None)
        ),
        "prompt_tokens_details": {
            "cached_tokens": getattr(prompt_details, "cached_tokens", None),
            "cache_write_tokens": getattr(prompt_details, "cache_write_tokens", None),
        }
        if prompt_details is not None
        else None,
        "completion_tokens_details": {
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        }
        if completion_details is not None
        else None,
        "paragraph_tokens": count_tokens(paragraph),
    }


def analyse_with_genai(
    i: int,
    paragraph: str,
    total_paragraphs: int,
    filepath: str,
    coding_config: CodingConfiguration,
    tags: list[Tag],
    row: dict[str, Any] | None = None,
    scheme: DocumentPromptScheme | None = None,
):
    if len(paragraph) < AI_CODING_MIN_PARAGRAPH_CHARS:
        return []

    try:
        system_message, user_message = render_prompt(
            coding_config, tags, paragraph, row, scheme
        )
        extra_body: dict[str, Any] = {}
        if coding_config.provider_names:
            extra_body["provider"] = {"only": list(coding_config.provider_names)}
        if coding_config.reasoning:
            # Omitted entirely (not sent as disabled) when reasoning=False --
            # see CodingConfiguration.reasoning/.effective_reasoning_effort
            # and run.py's reasoning-capability check, which can downgrade
            # this to False for a model confirmed not to support it.
            extra_body["reasoning"] = {"effort": coding_config.effective_reasoning_effort}
        if COST_ESTIMATION_ENABLED:
            # Ask OpenRouter to return actual cost/token usage on the
            # response (response.usage.cost, reasoning-token breakdown) --
            # see _extract_usage() and src/cost_estimation/calibration.py.
            extra_body["usage"] = {"include": True}
        system_content: str | list[dict[str, Any]] = system_message
        if coding_config.prompt_caching:
            system_content = [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]
        response = client.chat.completions.create(
            model=coding_config.model_name,
            # OpenRouter supports cache_control on text blocks; the OpenAI
            # SDK's ChatCompletionMessageParam types do not model that
            # provider extension, so keep the cast at this API boundary.
            messages=cast(Any, messages),
            response_format={"type": "json_object"},  # Force JSON
            extra_body=extra_body or None,
            temperature=coding_config.temperature,
            seed=coding_config.seed,
        )

        content = (response.choices[0].message.content or "").strip()

        # Remove any code block markers if present (some models might wrap JSON in ```json ... ```)
        content = content.replace("```json", "").replace("```", "").strip()

        # Auto-fix missing closing brace if the output was slightly truncated
        if content and not content.endswith("}"):
            content += "\n}"

        try:
            parsed_json = json.loads(content)
            parsed_results = parsed_json.get("classifications", [])
            results: list[dict[str, Any]] = [
                item for item in parsed_results if isinstance(item, dict)
            ]
        except json.JSONDecodeError as e:
            print(f"Error parsing row {i}. Raw output: {content}")
            process_log.current().log_ai_coding_error(
                severity="Error",
                source_name=filepath,
                row=i,
                display_value=_best_effort_display_value(filepath, i),
                configuration_name=coding_config.configuration_name,
                error_type="JSONDecodeError",
                message=str(e),
                raw_output=content,
                step=f"{process_log.STEP_RUN_MAIN}/{coding_config.configuration_name}",
            )
            results = [
                {"error": "JSONDecodeError", "message": str(e), "raw_output": content}
            ]

        if results:
            usage = _extract_usage(response, paragraph) if COST_ESTIMATION_ENABLED else None
            for result in results:
                result["source"] = {
                    "sourcefile": filepath,
                    "row": i,
                }
                result["model"] = getattr(response, "model", coding_config.model_name)
                result["provider"] = getattr(response, "provider", None)
                result["configuration_name"] = coding_config.configuration_name
            # Usage belongs to the API call/paragraph, not to each individual
            # classification returned by that call. Store it once to avoid
            # multiplying tokens and cost by the number of classifications.
            if usage is not None:
                results[0]["usage"] = usage
        print(f"Processing row {i}/{total_paragraphs - 1}...")
        return results
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        process_log.current().log_ai_coding_error(
            severity="Error",
            source_name=filepath,
            row=i,
            display_value=_best_effort_display_value(filepath, i),
            configuration_name=coding_config.configuration_name,
            error_type=type(e).__name__,
            message=str(e),
            step=f"{process_log.STEP_RUN_MAIN}/{coding_config.configuration_name}",
        )
        return [
            {
                "error": type(e).__name__,
                "message": str(e),
                "source": {
                    "sourcefile": filepath,
                    "row": i,
                },
                "configuration_name": coding_config.configuration_name,
            }
        ]


def process_documents(
    source_paths: Path | list[Path],
    coding_config: CodingConfiguration,
    iteration: int,
    tags: list[Tag] | None = None,
    *,
    warm_start: tuple[Path, int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Process documents for one coding configuration and iteration.

    warm_start, when given, is (filepath, paragraph_index, results) for a
    paragraph that was already really sent to the model -- see
    src.cost_estimation.estimation.probe_first_paragraph()/run.py's per-configuration
    first-row probe. That paragraph is not resubmitted; its already-computed
    results (and their usage) are spliced in as-is, so the probe call is
    reused as this run's genuine result for that row instead of being
    discarded or paid for twice.

    Returns this call's aggregate actual usage (prompt/paragraph/completion/
    reasoning tokens, cost, call count) when COST_ESTIMATION_ENABLED and at
    least one call succeeded, else None -- callers use this to update the
    per-model calibration file after a run (see src.cost_estimation.calibration).

    Each document's own valid conditions (see
    ``document_identity.condition_match()``) filter the coding scheme sent to
    the model, via ``src.core.models.tags_for_condition()``, so the AI only sees
    the part of the scheme valid for that document. The same valid
    conditions (plus the raw document condition and the raw coding-scheme
    sheets, see ``src.io.loaders.load_prompt_context()``) are also made
    available to the prompt builder itself via a ``DocumentPromptScheme``,
    so it can filter/format the coding scheme directly. Each paragraph's
    real source row (every META_OUTPUT_FILE column, resolved via
    ``resolve_rows_for_txt_file()``) is threaded through too, when it can be
    resolved.
    """
    output_dir = resolve_output_dir(get_ai_iteration_dir(coding_config, iteration))
    tags = tags if tags is not None else load_tags(CODES_FILE, DESCRIPTION_FILE)
    prompt_context = load_prompt_context(DESCRIPTION_FILE)

    files = get_matching_files(source_paths, "*.txt", recursive=True)
    all_metadata = []

    run_start = datetime.now().isoformat()

    # Collect unique models and providers used across the run
    used_models = set()
    used_providers = set()

    usage_totals: dict[str, int | float] = {
        "prompt_tokens": 0,
        "paragraph_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_prompt_tokens": 0,
        "cache_write_tokens": 0,
        "cost": 0.0,
        "calls": 0,
    }
    # OpenRouter's normalized usage schema does not guarantee a
    # cache_discount field. Keep missing as null rather than claiming $0.
    cache_discount_total: float | None = None

    for filepath in files:
        prep = prepare_document(filepath, coding_config, tags, prompt_context)
        if prep is None:
            continue

        print(
            f"\n--- Processing document: {filepath.name} "
            f"[{coding_config.configuration_name}, iteration {iteration}] ---"
        )

        all_classifications = []

        warm_start_index: int | None = None
        if warm_start is not None and warm_start[0] == filepath:
            warm_start_index, warm_start_results = warm_start[1], warm_start[2]
            all_classifications.extend(warm_start_results)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=coding_config.max_threads
        ) as executor:
            futures = [
                executor.submit(
                    analyse_with_genai,
                    i,
                    paragraph,
                    len(prep.paragraphs),
                    str(filepath),
                    coding_config,
                    prep.tags,
                    prep.rows[i],
                    prep.scheme,
                )
                for i, paragraph in enumerate(prep.paragraphs)
                if i != warm_start_index
            ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    all_classifications.extend(res)

        output_path = output_dir / f"{filepath.stem}_classified.json"
        write_json(all_classifications, output_path)

        print(f"Saved {len(all_classifications)} classifications to {output_path}")

        # Collect models, providers, and actual usage used for this document.
        # The source key is also a defensive deduplication guard for old/warm
        # results that may still carry the same call-level usage on every
        # classification from one paragraph.
        seen_usage_calls: set[tuple[str, int]] = set()
        for res in all_classifications:
            if "model" in res:
                used_models.add(res["model"])
            if res.get("provider"):
                used_providers.add(res["provider"])
            usage = res.get("usage")
            if usage:
                source = res.get("source") or {}
                sourcefile = source.get("sourcefile")
                row = source.get("row")
                call_key = (
                    (str(sourcefile), int(row))
                    if sourcefile is not None and row is not None
                    else ("usage-object", id(usage))
                )
                if call_key in seen_usage_calls:
                    continue
                seen_usage_calls.add(call_key)
                usage_totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
                usage_totals["paragraph_tokens"] += usage.get("paragraph_tokens") or 0
                usage_totals["completion_tokens"] += usage.get("completion_tokens") or 0
                reasoning = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                usage_totals["reasoning_tokens"] += reasoning or 0
                prompt_details = usage.get("prompt_tokens_details") or {}
                usage_totals["cached_prompt_tokens"] += (
                    prompt_details.get("cached_tokens") or 0
                )
                usage_totals["cache_write_tokens"] += (
                    prompt_details.get("cache_write_tokens") or 0
                )
                cache_discount = usage.get("cache_discount")
                if cache_discount is not None:
                    cache_discount_total = (cache_discount_total or 0.0) + cache_discount
                usage_totals["cost"] += usage.get("cost") or 0.0
                usage_totals["calls"] += 1

        all_metadata.append(
            {
                "source_file": str(filepath),
                "total_paragraphs": len(prep.paragraphs),
                "processed_paragraphs": sum(
                    1 for paragraph in prep.paragraphs if len(paragraph) >= 10
                ),
            }
        )

    run_end = datetime.now().isoformat()
    usage_summary: dict[str, int | float | None] = {
        **usage_totals,
        "cache_discount": cache_discount_total,
    }

    if WRITE_METADATA and all_metadata:
        metadata_output_path = output_dir / "run_metadata.json"
        general_scheme = DocumentPromptScheme(
            coding_scheme=prompt_context, valid_conditions=None, document_condition=None
        )
        general_system_prompt, general_user_prompt = render_general_prompt(
            coding_config, tags, general_scheme
        )
        global_meta = {
            "timestamp_start": run_start,
            "timestamp_end": run_end,
            "iteration": iteration,
            "coding_configuration": coding_config.to_metadata(),
            "system_prompt": general_system_prompt,
            "user_prompt_template": general_user_prompt,
            "models": sorted(used_models),
            "providers": sorted(used_providers),
            "coded_files": all_metadata,
        }
        if COST_ESTIMATION_ENABLED:
            global_meta["actual_usage"] = usage_summary
        write_json(global_meta, metadata_output_path)
        print(f"Saved run metadata to {metadata_output_path}")

    return usage_summary if COST_ESTIMATION_ENABLED and usage_totals["calls"] else None


if __name__ == "__main__":
    from src.config.general import CODING_CONFIGURATIONS

    loaded_tags = load_tags(CODES_FILE, DESCRIPTION_FILE)
    for config in CODING_CONFIGURATIONS:
        for iteration_number in range(config.iterations):
            process_documents(TXT_FILES, config, iteration_number, loaded_tags)
