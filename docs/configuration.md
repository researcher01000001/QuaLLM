# Configuration Reference

This guide covers the constants in the four modules in `src/config/`, the per-model settings, and the sampling settings in `scripts/sample_selector.py`. Derived paths are listed alongside settings you would normally edit.

Edit the Python files before launching a script. They are imported at startup; changes do not affect an already running process. The entry points have no command-line switches for prefix, run type, or review step, and environment variables do not override these constants. The API key is the exception.

## Where to make changes

| File | Responsibility |
| --- | --- |
| [`src/config/general.py`](../src/config/general.py) | Main run, active model configurations, consensus, reports, and costs |
| [`src/config/coding.py`](../src/config/coding.py) | `CodingConfiguration` fields and defaults |
| [`src/config/project.py`](../src/config/project.py) | Source files, spreadsheet schema, naming, and dataset selection |
| [`src/config/post_processing.py`](../src/config/post_processing.py) | Review steps, correction database, and final export |
| [`scripts/sample_selector.py`](../scripts/sample_selector.py) | Participant selection, sample sizes, and random seed |

For a new experiment, usually change `PREFIX`, `RUN_TYPE`, and `CODING_CONFIGURATIONS`, then decide whether `RUN_AI_CODING` should generate new annotations or load saved files. Keep prefix and run type unchanged while reviewing and exporting that experiment.

## API connection

`OPENROUTER_API_KEY` is read by `src/config/general.py` from the environment after `load_dotenv()` loads the repository's `.env` file. Put the key there or export it in your shell; do not put it in a Python source file.

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

The `OpenAI` client in `src/coding/ai_coding.py` has `base_url="https://openrouter.ai/api/v1"` set directly in its constructor. There is no configurable base-URL constant or environment override. Changing endpoints means editing that constructor and checking compatibility with the OpenRouter-specific request fields. The pricing/capability lookup separately uses OpenRouter's model endpoint in `src/cost_estimation/pricing.py`.

The main script imports this client even with `RUN_AI_CODING=False`. A nonempty placeholder such as `OPENROUTER_API_KEY=unused` allows SDK construction for a saved-output run; use a valid key for real calls. Post-processing does not initialize the client.

## Main runtime settings — `src/config/general.py`

| Constant | Current value / meaning |
| --- | --- |
| `PREFIX` | `"test_300926"`; run label used in output folders and filenames |
| `RUN_TYPE` | `"test"`; selects the transcript pool and report scopes described below |
| `RUN_AI_CODING` | `False`; reuse existing iteration JSON. `True` runs model calls. |
| `CODING_CONFIGURATIONS` | List of `CodingConfiguration` instances. Currently one Gemini configuration with three iterations and `SYSTEM_PROMPT_V1`; see the model fields below. |
| `AI_CODING_MAX_PARAGRAPHS` | `None`; a truthy value limits paragraphs sent per document after empty lines are dropped |
| `AI_CODING_MIN_PARAGRAPH_CHARS` | `10`; paragraphs shorter than this (after stripping) are skipped without a model call and excluded from cost estimation |
| `WRITE_METADATA` | `True`; write `run_metadata.json` for each configuration/iteration that processes files |
| `COMPARISON_PREFIXES` | Other run prefixes whose including-workbook `overview` rows enter the primary report's `comparison` sheet. Uses the same run type and primary scope. `[]` disables this import. |
| `SUPERCONSENSUS_THRESHOLD` | `0.5`; minimum share of final configuration consensuses assigning a character/tag |
| `SUPERCONSENSUS_NAME` | `"superconsensus"`; internal label for the synthetic configuration and default HTML source |
| `HTML_CONSENSUS_CONFIGURATIONS` | `[]` selects the top-level consensus file. A list of configuration names selects those files as separate HTML columns; displayed scores use the first AI source. |
| `UNDEFINED_TAGS_ENABLED` | `True`; retain unknown/condition-mismatched tags, enable the main report's undefined-tag toggle and alternate scores, and write the excluding workbook. `False` drops such tags while loading. |
| `SCORE_COLUMNS` | `(internal key, display label)` pairs for the seven HTML score options: tag-sequence distance, tag Jaccard/F1, and original/pruned character agreement/F1 |

Missing iteration files are treated as empty coding. When reusing output, match the configured names and iteration counts to the files you intend to evaluate.

### Run types

| Value | Processing | Main report coverage |
| --- | --- | --- |
| `training` | `TXT_FILES/training/` | Training files |
| `validation` | `TXT_FILES/validation/` | Validation files |
| `test` | Manually placed files in `TXT_FILES/test/` | Test files |
| `full_without_validation` | `TXT_FILES/full_without_validation/` | Training filenames only |
| `full` | `TXT_FILES/training/`, `validation/` and `full/`, each document once | Training, validation, and combined training + validation reports |

Files already on disk determine the processing pool. Names are keyed by file stem, and when a stem exists in several folders only the first file is used (see [run scopes](pipeline.md#run-types-and-scopes)). Post-processing uses the processing pool without main-report scope slicing.

### Cost settings

| Constant | Current value / meaning |
| --- | --- |
| `COST_ESTIMATION_ENABLED` | `True`; when AI coding runs, enable estimates, confirmation, a first-paragraph probe, usage capture, calibration, and cost summary |
| `COST_ESTIMATION_RECALC_THRESHOLD_FRACTION` | `0.1`; ask again if the probe-based estimate is more than 10% above the original. A newly resolved estimate also asks for confirmation. |
| `COST_ESTIMATION_PRICING_CACHE_TTL_HOURS` | `24`; reuse fetched model pricing for this long before refreshing |
| `COST_ESTIMATION_PRICING_CACHE_FILE` | `data/cache/openrouter_pricing.json`; shared pricing/capability cache |
| `COST_ESTIMATION_CALIBRATION_FILE` | `data/results/cost_calibration.json`; shared observed token ratios by model and reasoning effort |

Token factors cannot be set manually. They come from the current probe or saved calibration. Disabling cost estimation does not disable a reasoning-enabled model's capability lookup. See [cost estimation](ai-coding.md#cost-estimation) for calculation and accounting limits.

### Derived paths and filenames

These normally follow `PREFIX` and `RUN_TYPE` automatically. Paths below are relative to the repository root unless noted.

| Constant | Path / purpose |
| --- | --- |
| `ROOT_DIR` | Repository root, derived from `config/general.py`'s location |
| `DATA_DIR` | `ROOT_DIR / "data"` |
| `DATA_SOURCES_DIR` | `data/sources/`; every input file |
| `DATA_RESULTS_DIR` | `data/results/`; every generated output |
| `RUN_RESULTS_DIR` | `data/results/{PREFIX}/{RUN_TYPE}/`; one folder per prefix and run type, built by `run_results_dir(prefix, run_type)` |
| `AI_CODING_SUBDIR`, `RESULTS_SUBDIR`, `POST_PROCESS_SUBDIR`, `PROCESS_LOG_SUBDIR` | `1-ai_coding`, `2-results`, `3-post_process`, `process_log`; the stage folders inside `RUN_RESULTS_DIR` |
| `RESULTS_DIR` | `data/results/{PREFIX}/{RUN_TYPE}/2-results/` |
| `OUTPUT_HTML_FILE` | `RESULTS_DIR / "comparison_interface_{PREFIX}.html"` |
| `OUTPUT_XLSX_FILE_INCLUDING_UNDEFINED_TAGS` | `RESULTS_DIR / "configuration_matrix_main_{PREFIX}_including_undefined_tags.xlsx"`; always written |
| `OUTPUT_XLSX_FILE_EXCLUDING_UNDEFINED_TAGS` | `RESULTS_DIR / "configuration_matrix_main_{PREFIX}_excluding_undefined_tags.xlsx"`; written when undefined tags are enabled |
| `AI_CODING_FILES` | `data/results/{PREFIX}/{RUN_TYPE}/1-ai_coding/`; iteration, consensus, and prompt snapshot output |
| `CONSENSUS_FILENAME` | `"consensus.json"`; per-configuration and top-level consensus filename |
| `PROCESS_LOG_DIR` | `data/results/{PREFIX}/{RUN_TYPE}/process_log/` |
| `PROCESS_LOG_FILE` | `PROCESS_LOG_DIR / "process_log_{PREFIX}.xlsx"` |

For `full`, `scoped_output_html()`, `scoped_output_xlsx_including()`, and `scoped_output_xlsx_excluding()` add `_{scope}` after the prefix. Scopes are `training`, `validation`, and `combined`. Prompt snapshots live in `AI_CODING_FILES` as `prompt_{condition-slug}[__unused].txt`.

## Model settings — `CodingConfiguration`

Create entries in `src/config/general.py`'s `CODING_CONFIGURATIONS`. The dataclass in `src/config/coding.py` supplies these defaults; an entry can override each field:

| Field | Default | Meaning |
| --- | --- | --- |
| `short_name` | Required | Slugged into `configuration_name`, the output grouping key |
| `model_name` | `"anthropic/claude-sonnet-4.6"` | Model identifier sent to OpenRouter |
| `temperature` | `0` | Sampling temperature passed to the provider |
| `seed` | `42` | Same seed passed on every iteration; may be `None` |
| `provider_names` | `None` | Optional tuple of allowed provider names, sent as `provider.only` |
| `prompt_caching` | `True` | Mark the system-message text block as an ephemeral cache prefix |
| `max_threads` | `10` | Concurrent paragraph workers within each document |
| `iterations` | `1` | Number of repeated coding runs |
| `consensus_threshold` | `0.5` | Include a character/tag when its iteration vote share is at least this value |
| `system_prompt_name` | `"SYSTEM_PROMPT_V0"` | Name in `SYSTEM_PROMPT_BUILDERS` |
| `user_prompt_name` | `"USER_PROMPT_V0"` | Name in `USER_PROMPT_BUILDERS` |
| `reasoning` | `False` | Include an OpenRouter reasoning request when enabled |
| `reasoning_effort` | `None` | With reasoning enabled, resolves to `"medium"`; otherwise leave unset |

`VALID_REASONING_EFFORTS` is `("low", "medium", "high")`. An unsupported effort or an effort set while `reasoning=False` raises `ValueError`. Confirmed lack of model reasoning support causes the run to disable it with a warning; unresolved support leaves it configured.

`configuration_name` depends only on `short_name`: `gemini_3_7_flash` becomes `gemini-3-7-flash`. Different settings need distinct names or prefixes if their outputs should be kept separate. `effective_reasoning_effort` returns the resolved effort. The `system_prompt_builder` and `user_prompt_builder` properties resolve the named functions; unknown names raise `ValueError`. `METADATA_FIELDS` is the class's internal list of fields serialized by `to_metadata()`.

Available system builders are `SYSTEM_PROMPT_V0`, `SYSTEM_PROMPT_V1`, and `AI_GENERATED_SYSTEM_PROMPT`; user builders are `USER_PROMPT_V0` and `AI_GENERATED_USER_PROMPT`. V1 adds general rules to V0. The static AI-generated system prompt embeds its own coding scheme and ignores runtime condition filtering. See [prompts](ai-coding.md#paragraph-preparation-and-prompts) before changing versions.

## Study configuration — `src/config/project.py`

### Source files

| Constant | Current value / meaning |
| --- | --- |
| `CODES_FILE` | `data/sources/additional_sources/HeadCT_ThinkAlouds_codebook.xlsx`; tag colours |
| `DESCRIPTION_FILE` | `data/sources/additional_sources/description_codingschema.xlsx`; coding scheme and prompt context |
| `META_OUTPUT_FILE` | Dictionary with `filename=data/sources/additional_sources/output.xlsx` and `sheetname="Sheet"`; source text, metadata, and selection |
| `HUMAN_CODING_FILES` | `data/sources/human_coding/HeadCT_ThinkAlouds.xlsx`; human quotation export. The loader also accepts paths to folders or lists of paths. |
| `TXT_FILES` | `data/sources/atlas_import/`; transcript root with run-type subfolders |

### Document shape and display

| Constant | Current value / meaning |
| --- | --- |
| `GENERAL_CLUSTER_COLUMN` | `"condition"`; shared clustering column and input to the study's tag-condition mapping |
| `DOCUMENT_KEY_COLUMNS` | `["think_aloud_id", "condition"]`; group source rows into canonical documents |
| `DOCUMENT_NAME_TEMPLATE` | `"{think_aloud_id}_{condition}_cycle_{cycle_id}"`; format each document stem |
| `DOCUMENT_ORDER_COLUMN` | `"position"`; order source rows and join reviewed tags back to the spreadsheet |
| `DOCUMENT_ORDER_COLUMN_LABEL` | `"Paragraph"`; label for source order values in HTML, workbooks, and logs |
| `TEXT_COLUMN` | `"adjusted_transcript"`; paragraph text |
| `PARA_META_COLUMNS` | `(column, label)` pairs shown in HTML: patient information, truth, diagnosis, performance, confidence, advice usefulness, condition, difficulty, type, patient UID, cycle type, and group |

System tags are not configured here: they come from the `system` column of the coding scheme's `Coding_Scheme` sheet.

The name parser and condition mapping live in `src/dataset/document_identity.py`. Changing the template or vocabularies may require editing that module. Source metadata/display mapping currently assumes canonical document row order; see [alignment limitations](data-loading.md#source-documents-and-coordinates) for filtered and trimmed files.

### Dataset membership and sampling modes

| Constant | Current value / meaning |
| --- | --- |
| `DATASET_COLUMN` | `"dataset"`; stores training/validation membership and exclusions |
| `RANDOM_DATASET_ORDER_COLUMN` | `"dataset_order"`; shuffled row order within selected cluster files |
| `EXCLUDED_VALUES` | `["excluded"]`; values excluded by shared row-selection predicates. The study selector specifically preserves the literal `excluded`. |
| `TRAINING_SELECTION_MODE` | `"document_driven"`; training materialization mode, also read by the study selector |
| `VALIDATION_SELECTION_MODE` | `"random"`; validation materialization mode. The study selector currently always draws validation randomly. |
| `TRAINING_DOCUMENT_CLUSTER_COLUMN` | `["think_aloud_id", "condition"]`; grouping for random training draws |
| `TRAINING_SUB_CLUSTER_COLUMN` | `[]`; no further balancing groups within a training cluster |
| `VALIDATION_DOCUMENT_CLUSTER_COLUMN` | `["condition"]`; grouping for random validation draws |
| `VALIDATION_SUB_CLUSTER_COLUMN` | `["type", "performance"]`; balance the validation draw across these combinations |
| `NAMING_BY_MODE` | `DocumentNamingConfig` entries for `document_driven`, `random`, and `manual`; names/order/text/selection grouping described below |
| `RUN_TYPE_CONFIGS` | `RunTypeConfig` entries for the five run types; associates selection mode and cluster columns with each |

`DocumentNamingConfig` has `name_template`, `order_column`, `order_column_label`, `text_column`, and `group_column` fields. Document-driven mode uses the document template/order settings and selects whole `think_aloud_id` groups. Random mode uses `{type}_{cluster}`, sorts by `dataset_order`, and has no display label or whole-participant group. Manual mode has no single template and resolves manually supplied files using recognized name shapes.

`RunTypeConfig` has `selection_mode`, `document_cluster_column`, and `sub_cluster_column` fields. Training and validation use their corresponding settings; `test` uses `manual`. The `full` entries inherit training's table values, but preparation explicitly writes full/without-validation files through the whole-document filters, and runtime scopes are defined separately in `dataset/run_scope.py`.

These modes describe selection and file construction; they do not train model weights. Changing tables alone does not generalize the study-specific sampling script. See [dataset preparation](data-loading.md#preparing-the-dataset).

## Review settings — `src/config/post_processing.py`

### Behavior

The main configuration groups the review step, target tags, and Step 1 database-integration operation. Additional configuration controls database lookup, matching, rebuilding, and export options.

| Constant | Current value / meaning |
| --- | --- |
| `POST_PROCESS_STEP` | `1`; choose `1` for target-tag review, `2` for merging it and generating full review, or `3` for export |
| `POST_PROCESS_TAGS` | `["other"]`; tags always selected for Step 1 review |
| `POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE` | `True`; on a rerun with saved Step 1 data, merge resolved target-tag corrections into the database (creating it if needed) and stop without regenerating the existing page. `False` skips saving, e.g. for tests, and allows regeneration subject to protection/rebuild rules. With no review page yet, either value allows generation. |
| `POST_PROCESS_INCLUDE_UNDEFINED_TAGS` | `True`; add observed undefined tags to the target list. Tags must have survived loading via `UNDEFINED_TAGS_ENABLED`. |
| `POST_PROCESS_LOAD_DATABASE` | `True`; prefill target records from saved corrections |
| `POST_PROCESS_DATABASE_USE_ALL_PREFIXES` | `False`; current database only. `True` scans all prefixes and run types beneath the post-process root. |
| `POST_PROCESS_SPAN_OVERLAP_THRESHOLD` | `0.9`; minimum span intersection-over-union for database reuse/replacement; independent of validation's match threshold |
| `POST_PROCESS_STEP1_REBUILD_DATABASE` | `False`; on a rerun with saved Step 1 data, ask before replacing the database with that page's corrections, then stop |
| `POST_PROCESS_STEP3_USE_EXISTING_JSON` | `False`; extract saved full-review HTML. `True` requires existing review JSON and skips extraction plus the review-completion check. |
| `POST_PROCESS_STEP3_TAGS_COLUMN` | `"ordered_tags"`; fixed convention for the exported spreadsheet column |

Protection is a saved HTML setting, not a Python constant: `protect_file=True` prevents that page's regeneration. In Step 1 it takes precedence over rebuilding the database; enabled integration can still run before stopping. With protection off, rebuilding takes precedence over integration. Step 3 reads a protected full-review page normally. See [rerun rules](pipeline.md#rerunning-step-1).

### Folders and files

| Constant | Derived path / purpose |
| --- | --- |
| `POST_PROCESS_DIR` | `RUN_RESULTS_DIR / POST_PROCESS_SUBDIR`, i.e. `data/results/{PREFIX}/{RUN_TYPE}/3-post_process/` |
| `POST_PROCESS_STEP1_DIR` | `POST_PROCESS_DIR / "01_undefined_tags"` |
| `POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE` | Step 1 folder's `consensus_unmodified.json`; baseline with one tag per span entry |
| `POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE` | Step 1 folder's `consensus_with_defined_tags.json`; corrected output written when Step 2 closes out Step 1 |
| `POST_PROCESS_STEP1_REVIEW_HTML` | Step 1 folder's `review_interface_undefined_tags_{PREFIX}.html` |
| `POST_PROCESS_DATABASE_FILE` | Step 1 folder's `database_{PREFIX}.xlsx`; created when corrections are saved |
| `POST_PROCESS_STEP2_DIR` | `POST_PROCESS_DIR / "02_manual_supervision"` |
| `POST_PROCESS_STEP2_CONSENSUS_FILE` | Step 2 folder's `consensus.json`; copy of the corrected Step 1 output |
| `POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE` | Step 2 folder's `consensus_reviewed.json`; seeds full review from Step 1 corrections |
| `POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS` | Step 2 folder's `configuration_matrix_pp2_{PREFIX}_including_undefined_tags.xlsx` |
| `POST_PROCESS_STEP2_REPORT_XLSX_EXCLUDING_UNDEFINED_TAGS` | Step 2 folder's `configuration_matrix_pp2_{PREFIX}_excluding_undefined_tags.xlsx`; only when enabled |
| `POST_PROCESS_STEP2_REPORT_HTML` | Step 2 folder's `comparison_interface_pp2_{PREFIX}.html`; validation comparison |
| `POST_PROCESS_STEP2_REVIEW_HTML` | Step 2 folder's `review_interface_general_{PREFIX}.html`; full editable review |
| `POST_PROCESS_STEP3_DIR` | `POST_PROCESS_DIR / "03_export"` |
| `POST_PROCESS_STEP3_REVIEW_JSON` | Step 3 folder's `manual_review_{PREFIX}.json`; saved annotations and exclusions |
| `POST_PROCESS_STEP3_ORDERED_TAGS_JSON` | Step 3 folder's `ordered_tags_{PREFIX}.json`; tag sequences per paragraph |
| `POST_PROCESS_STEP3_OUTPUT_XLSX` | Step 3 folder's `output_{PREFIX}.xlsx`; source-sheet copy with tag column and excluded rows removed |

The Step 2 consensus JSON and reports do not absorb edits later made in the full-review page. Step 3 consumes those edits from saved HTML or its extracted JSON. Browser autosave is not a file save; replace the configured file if the browser downloads a copy. See [the review workflow](pipeline.md#post-processing).

## Sampling settings — `scripts/sample_selector.py`

These study-specific constants live in the script rather than the four configuration modules:

| Constant | Current value / meaning |
| --- | --- |
| `TRAINING_PARTICIPANT_IDS` | `[20, 50, 42, 45]`; participants selected for document-driven training |
| `TRAINING_CASES_PER_CLUSTER` | `50`; sample size per cluster when training selection is random |
| `VALIDATION_CASES_PER_CLUSTER` | `50`; sample size per validation cluster, subject to eligible rows available |
| `RANDOM_SEED` | `42`; reproducible selection and shuffling |

The selector writes only after confirmation and preserves existing `excluded` values. Its printed participant-exclusion statistics do not add exclusions to the spreadsheet. The current validation draw excludes selected training rows. `scripts/atlas_preperation.py` subsequently materializes the recorded selections; neither helper is called automatically by the runtime pipeline.
