# AI Coding

`src/coding/ai_coding.py` sends transcript paragraphs to OpenRouter through the OpenAI SDK. `run.py::run_ai_coding()` orchestrates configurations, iterations, capability checks, cost confirmation, and prompt snapshots.

## Configuration and execution

Set `RUN_AI_CODING` in `src/config/general.py`:

- `False` reuses existing iteration JSON without making classification calls.
- `True` runs every entry in `CODING_CONFIGURATIONS` for its own `iterations` count.

These flags are Python constants, not environment-variable overrides. `OPENROUTER_API_KEY` is read from the environment or `.env`.

The SDK client is constructed when `src/coding/ai_coding.py` is imported. Consequently, the main `run.py` entry point currently needs a nonempty key value even when AI coding is disabled. A placeholder such as `OPENROUTER_API_KEY=unused` is sufficient for a saved-output run with `RUN_AI_CODING=False`; real classification calls require a valid key. `run_post_processing.py` does not import this client and needs no API key.

`CodingConfiguration` is a frozen dataclass in `src/config/coding.py`. Its relevant defaults are:

| Field | Default / purpose |
| --- | --- |
| `short_name` | Required; slugged to form the output grouping key `configuration_name` |
| `model_name` | `anthropic/claude-sonnet-4.6`; the active configuration can override it |
| `temperature`, `seed` | `0`, `42`; the same seed is sent on every iteration |
| `provider_names` | `None`; otherwise sent as an OpenRouter provider allowlist |
| `prompt_caching` | `True` |
| `max_threads` | `10` concurrent paragraph workers per document |
| `iterations`, `consensus_threshold` | `1`, `0.5` |
| `system_prompt_name`, `user_prompt_name` | `SYSTEM_PROMPT_V0`, `USER_PROMPT_V0` |
| `reasoning`, `reasoning_effort` | `False`, `None` |

Configuration names come from `short_name` alone: `gemini_3_7_flash` becomes `gemini-3-7-flash`. Use distinct resulting names for distinct configurations; model/temperature changes do not automatically create a new output folder. The main configuration list explicitly selects models, prompt versions, and iteration counts independently of these defaults.

## Paragraph preparation and prompts

`prepare_document()` reads a transcript, strips each line, and drops empty lines. It resolves source spreadsheet rows against all remaining paragraphs, then applies `AI_CODING_MAX_PARAGRAPHS` per document when truthy (paragraphs and rows are cut together), and constructs a `DocumentPromptScheme`. Both the real coding path and cost estimator call this helper.

`resolve_rows_for_txt_file(filepath, paragraphs)` uses `src/dataset/selection.py` and `src/dataset/document_content.py`, sharing selection rules with `scripts/atlas_preperation.py`. Prepared files are aligned by expected row count. Manually placed `test/` files can also be matched to a contiguous source excerpt by normalized text when their count differs. Unresolvable files receive `row=None` for every paragraph and a warning. Matching counts alone do not verify file contents. See [input preparation and alignment limitations](data-loading.md#prompt-row-resolution-and-paragraph-splitting).

`render_prompt()` creates a `PromptContext` containing:

- Condition-filtered `tags` and the loaded `CodingScheme`.
- `valid_conditions` from `document_identity.condition_match()` and the raw `document_condition`.
- The paragraph text and its source spreadsheet `row` dictionary, when resolved.

`system_prompt_name` and `user_prompt_name` resolve registered Python functions in `src/coding/prompt_builders.py`; unknown names raise `ValueError`. Shared static strings live in `src/templates/prompts.py`. There are no Jinja prompt templates.

| Builder | Content |
| --- | --- |
| `SYSTEM_PROMPT_V0` | Shared instructions, condition-filtered coding scheme with definitions/notes/examples/double-coding rules, expected JSON format |
| `SYSTEM_PROMPT_V1` | V0 content plus the `General_Rules` sheet |
| `USER_PROMPT_V0` | Paragraph text |
| `AI_GENERATED_SYSTEM_PROMPT` | Static alternative coding scheme; ignores the context, including condition filtering |
| `AI_GENERATED_USER_PROMPT` | Alternative paragraph wrapper |

`load_prompt_context()` loads `Coding_Scheme`, `General_Rules`, and `Medical_Knowledge` from the description workbook. `CodingScheme` exposes text-formatting methods for these sections. The loader materializes known coding-scheme fields, rather than passing arbitrary spreadsheet columns through. The standard builders use the scheme and optional general rules; medical knowledge and source row dictionaries are available to custom builders but are not inserted by these standard builders.

The standard prompts request exact, contiguous phrases, surrounding verbatim context, tags, and rationales in a JSON object containing `classifications`. Tag-condition filtering uses the study vocabulary mapping in `src/dataset/document_identity.py`; unknown conditions get only unconditional entries. This filtering applies to standard builders; a custom or static builder controls its own output.

### Prompt snapshots and run metadata

After a configuration's iterations finish, `run_ai_coding()` renders a representative prompt for every condition in `document_content.known_conditions()`. Snapshots use `[paragraph text]` as the paragraph and no source row. Byte-identical system/user pairs within a condition share a heading listing their configurations.

Files are overwritten at:

```text
data/results/{PREFIX}/{RUN_TYPE}/1-ai_coding/prompt_{condition-slug}[__unused].txt
```

`__unused` marks conditions absent from this run's transcript files. Snapshots are condition-specific, unlike `run_metadata.json`'s general system/user prompt snapshot, which uses `valid_conditions=None` (unfiltered). Builders that require row data must handle `row=None` for both kinds of snapshot.

## Model request and response handling

`analyse_with_genai()` skips paragraphs shorter than `AI_CODING_MIN_PARAGRAPH_CHARS` (default 10) characters. Otherwise it renders prompts and sends `response_format={"type": "json_object"}`, the configured model, temperature, seed, optional provider allowlist, and optional reasoning/usage fields.

The response handler strips Markdown code fences, appends a missing final `}` when needed, parses `classifications`, and retains dictionary entries. It enriches entries with `source: {"sourcefile": filepath, "row": i}`, model, provider, and configuration name. The zero-based technical `source.row` is distinct from the source spreadsheet position; the old `source.paragraphnumber` field is not read by the iteration loader.

Malformed JSON becomes an error record including raw output. Other call/processing exceptions become error records with source information. Both are logged. The quote loader later ignores entries without a phrase or usable tags; a completed pipeline does not by itself imply that every API call yielded coding.

Do not remove the missing-closing-brace repair without explicit approval.

## Concurrency, files, and reuse

`process_documents()` processes one configuration/iteration. It walks the source files, handles each document's paragraphs concurrently, and writes:

```text
data/results/{PREFIX}/{RUN_TYPE}/1-ai_coding/{configuration_name}/iteration-{iteration}/
+-- {document_stem}_classified.json
+-- run_metadata.json
```

The classified file is a flat JSON list in future-completion order, not paragraph order. Use `source.row`, not array position. Files are rewritten on reruns; there is no automatic resume/skip-existing mechanism. Do not recursively load a prefix as a single run; `AIQuoteLoader.load_runs()` keeps configurations and iterations separate.

`WRITE_METADATA` controls `run_metadata.json`. It records timestamps, configuration, general prompt snapshots, actual model/provider names, and file/paragraph counts. With cost estimation enabled it also contains aggregated `actual_usage`.

Worker concurrency defaults to 10. There is no project-level retry/backoff loop around `analyse_with_genai()` beyond behavior provided by the SDK. Increasing worker count can increase rate-limit failures. The seed remains unchanged across repeated iterations; reproducibility and variation depend on the configured model/provider.

## Reasoning

When `reasoning=False`, the request omits the reasoning key entirely. With `reasoning=True`, it sends `extra_body["reasoning"] = {"effort": effective_reasoning_effort}`. Supported effort strings in this project are `low`, `medium`, and `high`; `None` resolves to `medium` when reasoning is enabled. Setting an effort while reasoning is disabled raises `ValueError`.

Before classification, `run_ai_coding()` checks each reasoning-enabled model through `pricing.fetch_model_pricing()` and its `supports_reasoning` field. That lookup uses cached or fetched OpenRouter model metadata. Confirmed-unsupported models are replaced with reasoning-disabled configurations and a warning; unresolved support leaves reasoning configured and logs a warning. This check can fetch model metadata even when cost estimation is disabled.

## Prompt caching

When `prompt_caching=True`, the request's system message is a text content block with `cache_control={"type": "ephemeral"}`. The user message is separate and uncached. Setting `False` uses a plain system string.

The hint does not guarantee a cache hit. Provider/model support, minimum prefix lengths, TTL, routing, and concurrent first calls affect reuse. `provider_names=None` leaves routing unpinned; an explicit allowlist remains independent of caching. System text can differ by condition, so only matching prefixes can be reused.

The estimator groups byte-identical system prompts, assumes one write per distinct prefix and reads for subsequent calls across iterations, and charges dynamic user tokens at the normal input rate. It also shows a no-cache comparison. This is an estimate, not a simulation of actual provider cache state.

## Cost estimation

`COST_ESTIMATION_ENABLED=True` enables estimation, console confirmation, a first-real-paragraph probe, usage capture, calibration, and an end-of-run summary **when AI coding runs**. Disabling it removes these cost-specific operations; reasoning capability lookup remains independent.

### Initial estimate

`estimate_run_cost()` prepares and renders all eligible paragraphs once per configuration, multiplying totals by the iteration count. `count_tokens()` uses a cached generic `cl100k_base` tiktoken encoding, not a model-specific tokenizer. On first use, tiktoken may download its encoding data; offline estimation and tokenizer tests need that data cached already.

Completion and reasoning factors resolve in this order:

```text
observed probe override → calibration for the same model/effort
                        → another effort bucket for that model → unresolved
```

There are no manual factor settings in the configuration. Before the probe, estimates use calibration or show `N/A` with reasons. Reasoning-disabled configurations use a reasoning factor of zero.

```text
completion_tokens = paragraph_tokens * completion_token_factor
reasoning_tokens  = completion_tokens * reasoning_token_factor
visible_tokens    = completion_tokens - reasoning_tokens
```

Completion tokens include reasoning; reasoning is not added a second time. Input cost is cache-aware when cache prices are available, otherwise ordinary input pricing. Output cost prices visible and reasoning tokens separately, falling back to the completion rate if no separate reasoning price is available. If the reasoning split is unresolved, the implementation prices all estimated completion tokens at the completion rate.

`src/cost_estimation/pricing.py` fetches OpenRouter's model list and caches prices at `data/cache/openrouter_pricing.json` for `COST_ESTIMATION_PRICING_CACHE_TTL_HOURS`. A failed refresh can use stale cached pricing; absent pricing remains unresolved. The pricing code adds the ordinary prompt rate to Google-model cache-write surcharges; other providers' write rates pass through. The cache has a schema version to trigger refreshes when fields change.

`confirm_cost_estimate()` prints the overview and requires `y` to proceed to classification. Its decision is logged in the process workbook.

### First-paragraph probe and confirmation

Before each configuration's batch, `probe_first_paragraph()` calls the model on its first eligible paragraph in file traversal order. `recalculate_from_probe()` uses observed completion/paragraph and reasoning/completion ratios to estimate the configuration's full run, including all iterations. If usable usage is missing, calibration/unresolved fallbacks still apply.

The result is passed to iteration zero as `warm_start`, so that paragraph is not called again in that iteration. It is a real, potentially billable call even if the user declines the subsequent recalculation confirmation.

When the recalculated total is resolved, confirmation is required if the original total was unresolved, if a zero original total becomes positive, or if the relative increase exceeds `COST_ESTIMATION_RECALC_THRESHOLD_FRACTION` (currently `0.1`, meaning more than 10%). Declining stops the run. If the new total is still unresolved, there is no second confirmation.

### Recorded usage and calibration

With cost accounting enabled, requests include `extra_body["usage"] = {"include": True}`. `_extract_usage()` captures returned token/cost/cache fields and adds locally counted `paragraph_tokens`. Usage is attached only to the first returned classification/error dictionary, then defensively deduplicated by source file and row during aggregation. It is not repeated for every classification.

`process_documents()` aggregates `prompt_tokens`, `paragraph_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_prompt_tokens`, `cache_write_tokens`, `cache_discount`, `cost`, and `calls`. Missing cache discount remains `null`; other absent numeric fields are summed as zero. A call returning an empty classifications list has no result dictionary to carry usage, so its usage is not currently captured. Failed calls without returned usage are also absent from totals. Treat the printed actual-cost summary as recorded usage, not a guaranteed complete billing reconciliation.

After all configuration iterations, `update_cost_calibration()` groups recorded usage by `(model_name, effective_effort or "none")` and overwrites that pair's latest ratios in the global `data/results/cost_calibration.json`:

```text
completion_token_factor = completion_tokens / paragraph_tokens
reasoning_token_factor  = reasoning_tokens / completion_tokens
```

There is no smoothing across runs. Flat legacy calibration entries are read into the model's `none` bucket. The global file is shared across prefixes and run types; cross-effort fallback is labelled as such.

Finally, the summary prints initial, recalculated, and recorded actual costs for each configuration and their totals. Missing estimate figures show `N/A` and are excluded from totals. Configurations without captured usage can show zero recorded actual cost.
