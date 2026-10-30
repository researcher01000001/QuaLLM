# Architecture Notes

This guide records implementation boundaries and behavior that matters when changing the pipeline. Operational instructions are in [Pipeline](pipeline.md), source schemas in [Data Loading](data-loading.md), and score semantics in [Metrics and Rendering](metrics-and-rendering.md).

## Module responsibilities

- `run.py` and `run_post_processing.py` orchestrate I/O, console decisions, and processing stages.
- `config/general.py` holds generic runtime settings; `config/coding.py` defines the frozen `CodingConfiguration`; `config/project.py` holds study paths/schema/naming tables; `config/post_processing.py` selects review steps and database/export behavior.
- `dataset/document_identity.py` parses study-specific names and translates condition vocabularies. `dataset/selection.py` implements shared row-membership predicates. Neither set of functions lives in `config/project.py`.
- `dataset/document_content.py` groups and caches source spreadsheet rows, supplies display values and paragraph metadata, and normalizes transcript cells. `dataset/run_scope.py` discovers materialized files and report subsets.
- `io/loaders.py` reads tags, source documents, raw AI iteration quotes, human quotes, and consensus files. `core/models.py` defines `Tag`, `CodingScheme`, `Row`, `CodedRow`, `DisplayColumn`, `Document`, and quote subclasses.
- `coding/ai_coding.py` prepares paragraphs, renders prompts, invokes OpenRouter, and writes iteration output. Prompt builders, pricing, token estimates, and calibration are separate modules.
- `coding/consensus.py` aggregates predictions into voted spans. `analysis/metrics.py` calculates reusable result objects, `analysis/agreement.py` implements metrics, and `analysis/tag_analysis.py` calculates span/character diagnostics.
- `report/xlsx_rows.py` shapes report rows; `report/xlsx.py` writes metrics workbooks. `report/html.py` and `templates/compare.html` generate comparison pages.
- `post_processing/review_database.py` handles the correction database; `post_processing/undefined_tags.py` builds restricted review records and merges Step 1 edits. `post_processing/review_page.py` builds, renders, and reads the Step 2 manual-review page, and `post_processing/tag_export.py` implements the Step 3 collapse, ordered-tag, and workbook export. Step 2 is built from the paragraph merge in `undefined_tags.py` and the page helpers in `review_page.py`.
- `core/process_log.py` collects lifecycle diagnostics; `io/io_utils.py` supplies shared JSON/workbook I/O.

## Invariants

- `Tag` equality and hashing use `Tag.name` only. Metrics and deduplication rely on that choice.
- Internal spans use absolute document offsets with exclusive ends. ATLAS exports use inclusive `quote_end`, so extraction and loading add `1`.
- `AIQuote.find_position()` caches `start` and `end` on the quote. It uses a unique phrase directly, then context to resolve repetition. Ambiguous fallback placement logs an error and replaces quote tags with undefined `position_unsure_<name>` tags.
- `AIQuote.in_row()` uses its zero-based technical `row`; displayed source positions are independent. `ConsensusQuote` uses overlap with stored spans, while `AtlasQuote` assigns a quote by its starting position.
- `Document` contains `source_name`, `rows`, and `display_column`. Identity is resolved separately from the name. `append_quotes()` replaces its rows with `CodedRow` objects; it does not append annotations onto existing row objects in place.
- `document_content._document_rows()` caches grouped spreadsheet content for the process lifetime. Editing the spreadsheet does not refresh that cache in a running process.
- Numeric values are normalized before filename formatting: whole-number floats become integers and numpy scalars become native values.
- Character/tag consensus uses `votes / iteration_count >= threshold`, including the half-vote case. Missing configured run data counts as empty coding, not a smaller denominator.
- Superconsensus gives each final configuration consensus equal weight. With one configuration it preserves that configuration's character labels.
- Configuration folder keys are slugged `short_name` values, independent of model settings. Use unique names to avoid overwriting/grouping different configurations together.
- Prediction-level consensus scores and averages of iteration scores remain separate. Overview/document means are not pooled character-level statistics.
- System tags (`system = yes` in the coding scheme, `Tag.system`) are excluded from scores even if shown in HTML. Undefined-tag filtering is a separate option.

## Changes requiring explicit approval

These restrictions mirror the repository's `AGENTS.md`:

- Do not remove missing-closing-brace repair in `coding.analyse_with_genai()`.
- Do not remove CRLF normalization in `DocumentLoader.load()` or `quote_extraction.load_txt_files()`.
- Do not remove ATLAS inclusive-end adjustment.
- Do not replace the loose `examples` parser unless there is a concrete parsing bug.
- Do not change `document_identity.condition_match()`'s `general` / `supported` / `human` mapping without confirmation. The document and coding-scheme vocabularies differ; their mapping was explicitly confirmed.

## Known limitations

### Paragraphs, display values, and file identity

AI preparation strips text and drops empty lines, but source loading preserves empty line positions. Blank lines can shift the technical `source.row` used by AI quotes. Generated transcripts normalize embedded cell line breaks, but missing cells can still produce empty lines.

Prompt row resolution checks counts, with a contiguous-text fallback for shortened `test/` files. It does not verify every prepared file's text. Display values and report metadata are built separately by enumerating the canonical document's source rows; they are not realigned for filtered files or test excerpts. Step 3 joins by those display values, so prompt alignment alone does not establish correct export alignment.

Random cluster sample names can resolve a condition but no full participant/document key. They can be coded and reported, but are skipped by correction-database matching and Step 3 ordered-tag generation. Arbitrary cluster naming changes need corresponding parser changes.

`RUN_TYPE="full"` processes the entire transcript tree, including manually placed test files. Duplicate stems share document and AI-output keys; differing copies can overwrite each other. Report slicing uses stems, not source-row selection. The combined scope is training + validation only. See [run types](pipeline.md#run-types-and-scopes).

### Annotation and score representations

Raw AI output uses `source.sourcefile` and `source.row`; consensus JSON uses `source.source_name`, `source.row`, and absolute `start` / `end`. These are distinct schemas. The old raw `source.paragraphnumber` field is not supported by the current iteration loader.

AI results are written in completion order. Failed/unparseable calls can leave error dictionaries, which loaders skip when no quote is usable. A missing phrase has no character span but may still contribute to an iteration's tag-level scores. Ambiguous phrases remain as explicitly renamed undefined tags.

Single-iteration configuration JSON preserves original quote spans, while in-memory consensus metrics use compressed voted spans. HTML reloads the JSON. Identical character labels need not give identical tag-sequence or span-count results across these representations.

HTML scores refer to the human coder and the first selected AI source only. Additional AI columns do not gain their own score blocks. The tag-confusion sheet counts unequal label pairs on shared coded characters, including multiple cross-pairs for multi-tag assignments; it is not an exclusive classification matrix.

### Persistence and regeneration

The preparation script overwrites matching transcript files without removing stale files. Runtime discovery trusts the files present. ATLAS quote extraction can rewrite the human workbook and only rebuilds a missing or entirely empty text column, not partial gaps.

Saving the correction database in Step 1 stops that run. Step 2 reports and `consensus_reviewed.json` describe Step 1 corrections and seed the full review page; they do not incorporate that page's later edits. Step 3 reads the saved full review but does not recalculate validation reports. `protect_file` protects the review HTML only. Browser local storage is not a Python-readable save; edited HTML must be saved at the configured path.

Step 1 exclusion flags are not merged into Step 2. Exclusions used by Step 3 come from saved Step 2 review data or reused Step 3 JSON. Step 3 writes a new workbook containing the selected source sheet and the tag column; original workbook formatting and other sheets are not preserved.

The process workbook stores the latest diagnostics per owned step, preserving other steps. It is not a chronological run archive; Steps 2 and 3 share ownership of `manual_rework_review` diagnostics.

### AI client and accounting

The main entry point imports an eagerly constructed OpenRouter client, so it currently requires a nonempty API-key value even when reusing saved output. Post-processing has no such client dependency. `RUN_AI_CODING` is a Python setting, not an environment override.

Worker count defaults to 10; there is no additional project retry/backoff loop around SDK calls. Empty classification responses and failed calls without usage are absent from recorded usage totals. Cost estimates use a generic tokenizer, probe/calibration ratios, and assumed cache behavior. See [AI Coding](ai-coding.md) for the exact accounting limits.

## Adapting to another dataset

Start with `config/project.py` for input paths, spreadsheet key/order/text columns, display labels, metadata, and selection/naming tables. System tags are marked in the coding scheme's `system` column. Further study assumptions remain in code:

1. `dataset/document_identity.py` hardcodes the filename parser, identity fields, and condition mapping. A different name template or vocabulary needs changes here too.
2. `io/loaders.py` hardcodes ATLAS quotation columns and the description workbook's sheets/known columns. Prompt context does not automatically pass through arbitrary extra coding-scheme columns.
3. `coding/prompt_builders.py` and `templates/prompts.py` contain Head CT instructions; the static alternative prompt embeds its own coding scheme and ignores runtime condition filtering.
4. `scripts/sample_selector.py` contains participant IDs, sample sizes, study performance statistics, and some hardcoded study columns. Its settings are not all in `config/project.py`.
5. Display/export joins currently rely on integer source order values and canonical document identities. A different ordering scheme or cluster-level export needs explicit alignment work.

Thus changing `config/project.py` alone handles compatible paths/column settings, but does not support every new dataset shape without code changes. Preserve offset and metric invariants when generalizing these boundaries.
