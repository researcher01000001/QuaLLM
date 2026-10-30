# Data Loading

This guide covers preparation scripts, input schemas, document identity, and text coordinates. Paths and spreadsheet settings are defined in `src/config/project.py`; file readers live in `src/io/loaders.py` and `src/dataset/document_content.py`.

## Preparing the dataset

Run the preparation scripts from the repository root:

```bash
uv run python -m scripts.sample_selector
uv run python -m scripts.atlas_preperation
```

Neither script is called by `run.py`.

### Select training and validation rows

`scripts/sample_selector.py` reads `META_OUTPUT_FILE` and, after console confirmation, rewrites its configured sheet with `DATASET_COLUMN` (`dataset`) and `RANDOM_DATASET_ORDER_COLUMN` (`dataset_order`). It asks again before replacing existing non-excluded selection values. This is a pandas rewrite, not a workbook-preserving cell edit.

- Existing `excluded` values are maintained manually and preserved. The participant-exclusion statistics printed by the script do not create these values.
- Document-driven training selects the participants in `TRAINING_PARTICIPANT_IDS`, across their conditions. The participant list, sample sizes, and seed live in the script.
- Random selection draws without replacement within configured document clusters, balances across sub-clusters, and fills shortfalls from the remaining eligible pool. Training rows and existing excluded rows cannot enter the validation draw.
- Selected rows carry `training_{cluster}` or `validation_{cluster}`. Their shuffled order is stored separately as an integer in `dataset_order`. Unselected rows are left blank and remain eligible for full-dataset preparation.

`RUN_TYPE_CONFIGS` and `NAMING_BY_MODE` describe selection/naming modes in `config/project.py`. The selector itself still contains study-specific participant and performance logic; changing the configuration tables alone does not generalize that script.

### Materialize transcript files

`scripts/atlas_preperation.py` writes under `TXT_FILES` (`data/sources/atlas_import/`):

| Subfolder | Contents |
| --- | --- |
| `training`, `validation` | Document-driven: one file per document with only that split's rows. Random: one `{type}_{cluster}.txt` file per selected cluster, ordered by `dataset_order`. |
| `full` | One file per canonical document, excluding rows marked excluded |
| `full_without_validation` | One file per canonical document, excluding validation and excluded rows |
| `test` | Created if missing; populated manually |
| `word` | One combined `.docx` per generated run type |

The shared selection functions are in `src/dataset/selection.py`: `row_filter_for_txt_subfolder()`, `random_mode_row_filter()`, and `parse_dataset_value()`. Spreadsheet fetching/filtering is in `document_content.rows_for_document()` / `all_rows_matching()`.

`row_text_value()` converts missing text to an empty string, strips surrounding whitespace, and collapses embedded line breaks to spaces. The script then joins cells with newlines, giving each source row one transcript line. Matching filenames are overwritten; stale files are not removed. The runtime discovers files on disk, so stale files remain part of a run until removed.

## Source spreadsheet and document identity

`META_OUTPUT_FILE` specifies a workbook filename and sheet name. For this study:

| Setting | Value / purpose |
| --- | --- |
| `DOCUMENT_KEY_COLUMNS` | `think_aloud_id`, `condition`: group rows into one document |
| `DOCUMENT_NAME_TEMPLATE` | `{think_aloud_id}_{condition}_cycle_{cycle_id}` |
| `DOCUMENT_ORDER_COLUMN` | `position`: source row order and export join key |
| `DOCUMENT_ORDER_COLUMN_LABEL` | `Paragraph`: display label in HTML, metrics, and logs |
| `TEXT_COLUMN` | `adjusted_transcript` |
| `PARA_META_COLUMNS` | Source columns and labels to show in reports |

`document_content._document_rows()` reads the configured sheet once, groups by the key columns, sorts by the order column, and caches the result for the process lifetime. Names use values from the first row of each group; template values should be constant within a group. Whole-number floats are normalized to integers when names are formatted.

Name parsing lives in **`src/dataset/document_identity.py`**, separate from the configuration:

```python
parse_document_name("45_no advice_cycle_63")
# {"think_aloud_id": 45, "condition": "no advice", "cycle_id": 63}
get_document_key("45_no advice_cycle_63")
# {"think_aloud_id": 45, "condition": "no advice"}
```

That document-driven shape is parsed without spreadsheet access. The parser also accepts known condition names and `training_{condition}` / `validation_{condition}` sample names, checking them against `document_content.known_conditions()` from the spreadsheet. Those cluster sample names resolve only a condition; `get_document_key()` returns `None`, since there is no single participant identity. Arbitrary multi-column cluster names are not covered by this parser automatically.

`condition_match()` translates the study vocabularies:

| Document condition | Valid coding-scheme conditions |
| --- | --- |
| `advice`, `advice with annotation` | `general`, `supported` |
| `no advice` | `general`, `human` |
| Unresolved / other | None; only tags without a condition restriction apply |

Change the parser and mapping explicitly when adapting filename shapes or condition vocabularies. There is no spreadsheet identity fallback for an arbitrary filename.

## Tags and prompt context

`load_tags()` reads tag colours from first-column font colours in the active sheet of `CODES_FILE` (`HeadCT_ThinkAlouds_codebook.xlsx`). Codes without a font colour, or missing from that sheet, get `#808080`. If `CODES_FILE` does not exist, a warning is printed and logged, and each tag gets its own colour derived from the MD5 hash of its name (the same scheme undefined tags use). With `DESCRIPTION_FILE` configured, names and definitions come from its `Coding_Scheme` sheet:

- `code_category` and optional `code` form `category: code`, or just `category` when `code` is blank. The legacy `name_1` / `name_2` columns are also accepted.
- `definition`, `note`, `examples`, `group`, and `condition` supply tag metadata.
- `double_coding = yes` sets `Tag.can_co_occur`. Two tags conflict only when both disallow co-occurrence; a co-occurring tag may be paired with anything.
- `system = yes` sets `Tag.system`. System tags support AI coding but are excluded from every agreement metric and the per-tag analysis, and are hidden by default in the HTML pages. A blank value means not a system tag. The `group` column has no effect on this.
- Tag equality and hashing use `Tag.name` only.

`load_prompt_context()` additionally reads `General_Rules` (`rule` column) and `Medical_Knowledge` (each named column independently) into `CodingScheme`. The loaders use the known coding-scheme fields; arbitrary extra `Coding_Scheme` columns are not passed through. Prompt builders decide which context sections to use.

All three quote loaders can apply `document_identity.condition_match()`. Unknown names in AI/consensus JSON, and known tags whose condition does not fit the document, are undefined (`additional_tag=True`) when `UNDEFINED_TAGS_ENABLED` is enabled; otherwise they are dropped. A condition-mismatched tag retains its codebook metadata. The ATLAS loader iterates known codebook tags, so arbitrary unknown Excel tag columns are ignored.

## Source documents and coordinates

`DocumentLoader.load()` reads UTF-8 text, normalizes CRLF to LF, and builds a `Document` containing one `Row` per `splitlines()` entry. Each row has a zero-based `index`, text, and absolute document `start_position` / `end_position`. Internal character spans are **start-inclusive, end-exclusive**.

`Document.display_column` is a `DisplayColumn` populated from source `position` values by `document_content.display_column_for()`. Technical `row` indices and displayed paragraph values are separate: row zero need not represent source position one. Metadata comes from `paragraph_meta()`, keyed by canonical document name and source position; the renderer and review page look it up using `Document.display_value(row.index)`.

The current display mapping enumerates the whole canonical document's spreadsheet rows. It does not realign filtered subsets or manually trimmed test excerpts, and cluster files have no canonical display mapping. Source prompts have a separate row resolver (below); successful prompt alignment does not guarantee that display metadata or Step 3 export positions are realigned. Keep this limitation in mind when exporting subsets.

## Human ATLAS quotes

`AtlasQuoteLoader` reads the first worksheet of each configured Excel export. Required columns are:

| Column | Meaning |
| --- | --- |
| `Source Name` | Canonical document stem; the final slash-separated path component is used |
| `quote_start` | Absolute start offset |
| `quote_end` | Absolute **inclusive** end offset |
| `quote_text` | Quoted text |
| A column per known tag | A nonblank, nonzero value assigns that tag |

The loader stores `end = quote_end + 1` for Python slicing. Rows with missing text or no assigned tags are skipped.

Before loading, `run.py` scans every `TXT_FILES` subfolder and calls `io/quote_extraction.py`. A missing or entirely empty `quote_text` column is extracted from transcript offsets and the Excel file is rewritten **in place**. When a stem exists in several subfolders, `run_scope.find_canonical_documents()` takes the whole document from `full/` first (then `test/`, `training/`, `validation/`, `full_without_validation/`), because ATLAS offsets refer to the whole transcript and `full_without_validation/` copies are trimmed. If any `quote_text` is already nonblank, extraction skips the file entirely; individual gaps are not backfilled. Preserve the CRLF normalization and inclusive-end adjustment.

## AI iteration JSON

`AIQuoteLoader.load_quotes()` recursively reads `*_classified.json` files within one iteration directory:

```text
data/results/{PREFIX}/{RUN_TYPE}/1-ai_coding/{configuration_name}/iteration-{iteration}/
```

Each file contains a JSON list, for example:

```json
[
  {
    "phrase": "exact transcript text",
    "context": "surrounding exact transcript text",
    "tags": ["Tag Name"],
    "rationale": "Why the tag applies",
    "source": {
      "sourcefile": "/path/to/45_no advice_cycle_63.txt",
      "row": 0
    }
  }
]
```

`source.row` is the zero-based paragraph index, not the source spreadsheet's `position`. The loader takes `Path(sourcefile).stem` as the document name, falling back to the JSON file's stem when no sourcefile is present. It does **not** read the old `source.paragraphnumber` field. Results are saved in completion order, so use the source row rather than array position.

`AIQuoteLoader.load_runs()` reads configurations and iterations independently. Loading a whole prefix recursively with `load_quotes()` would merge runs.

`AIQuote.find_position()` searches case-insensitively within the assigned row:

1. A unique phrase match is used directly.
2. Repeated phrases can be resolved by a unique context containing one occurrence of the phrase.
3. Remaining ambiguity uses the first available match, logs `position_unsure`, and renames every tag to `position_unsure_<name>` with `additional_tag=True`.
4. A missing phrase logs `not_found` and has no resolved span.

## Consensus JSON

`ConsensusQuoteLoader` reads a flat list of records with absolute, exclusive-end offsets:

```json
[
  {
    "phrase": "exact transcript text",
    "tags": ["Tag Name"],
    "source": {"source_name": "45_no advice_cycle_63", "row": 0},
    "start": 0,
    "end": 21,
    "source_iterations": 3
  }
]
```

`source_name` is already a document stem. Missing text, tags, or unresolved (`null`) start/end positions cause a record to be skipped. This format is distinct from raw iteration JSON; review exports also use this consensus shape.

## Prompt row resolution and paragraph splitting

`dataset/selection.py::resolve_rows_for_txt_file(filepath, paragraphs)` reuses preparation's selection logic to attach source spreadsheet dictionaries to prompts. In `test/`, it recognizes random sample names or canonical document names; if the count differs, it can match a contiguous excerpt by normalized text. Other prepared subsets require matching row counts. Unresolved files yield `row=None` in prompt context and a warning.

The count checks do not verify all file contents against the spreadsheet. Regenerate prepared files after changing their source selection or text. AI coding strips paragraphs and drops empty lines, while `DocumentLoader` preserves empty line positions; blank lines can shift `source.row` indices. This remains a known limitation, not a supported format conversion.
