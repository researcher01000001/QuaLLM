# QuaLLM

## Overview

This project explores how generative AI can support the analysis of think-aloud data at a larger scale while preserving analytical depth. It generates AI coding of transcripts and compares the results with human coding exported from ATLAS.ti. Researchers can review and correct the AI annotations, with feedback influencing further iterations.

The general workflow is:

1. Prepare the source spreadsheet, coding scheme, transcripts and human coding export.
2. Generate AI annotations through OpenRouter if requested.
3. Combine repeated AI runs into a consensus and these consensusses through multiple configurations into a superconsensus.
4. Calculate overlap metrics and save the results in reports. Reviewers can then compare the AI-coded annotations with the Human coding.
5. Review and correct annotations in editable HTML pages.
6. Export the saved review as JSON and a source-spreadsheet copy containing ordered tag lists.

## Setup

You'll need Python 3.13+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root, install the project dependencies:

```bash
uv sync
```

To connect to OpenRouter, copy `.env.example` to `.env` in the repository root and enter your API key. You can also set it directly in your environment:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

If you're reusing saved AI output, the main script still needs a key value but a placeholder also works with `RUN_AI_CODING=False`. You won't need an API key for the separate review and export steps.

## Input Data

Start with the files below. These are the default paths, if your files are stored elsewhere, you can change the paths in `src/config/project.py`:

| Path | Contents |
| --- | --- |
| `data/sources/additional_sources/output.xlsx` | Source spreadsheet containing transcript text and metadata |
| `data/sources/additional_sources/HeadCT_ThinkAlouds_codebook.xlsx` | ATLAS.ti codebook export used for tag colours |
| `data/sources/additional_sources/description_codingschema.xlsx` | Coding scheme, tag definitions and rules |
| `data/sources/human_coding/HeadCT_ThinkAlouds.xlsx` | Human ATLAS.ti quotation export used as the comparison reference |
| `data/sources/atlas_import/{run_type}/*.txt` | UTF-8 transcripts, with one paragraph per line |

If you're preparing these inputs for the first time, [Data Loading](docs/data-loading.md) explains the required spreadsheet columns, transcript format and document naming.

## Configuration

Before you start, choose the settings for your run. The table below shows where to find them:

| File | Settings to change |
| --- | --- |
| `src/config/general.py` | Main run, model configurations and reports |
| `src/config/coding.py` | Per-model settings and their defaults |
| `src/config/project.py` | Input paths, data layout and dataset selection |
| `src/config/post_processing.py` | Review and export settings |

For each run, choose:

- **`PREFIX`:** a run label. Use a new prefix to keep a new experiment's outputs separate.
- **`RUN_TYPE`:** which transcript subset to process.
- **`RUN_AI_CODING`:** `True` generates new annotations, `False` loads existing iteration files.
- **`CODING_CONFIGURATIONS`:** the model and prompt settings to use for AI coding.
- **`POST_PROCESS_STEP`:** which review or export stage to execute after the main pipeline has run.

Here is a full example of `CODING_CONFIGURATIONS` in `src/config/general.py`:

```python
from src.config_coding import CodingConfiguration

CODING_CONFIGURATIONS = [
    CodingConfiguration(
        short_name="gemini_3_8_flash",
        model_name="google/gemini-3.8-flash",
        temperature=0,
        seed=42,
        provider_names=None,  # Optional tuple of allowed provider names
        max_threads=10,  # Concurrent paragraph workers
        iterations=3,
        consensus_threshold=0.5,
        system_prompt_name="SYSTEM_PROMPT_V1",
        user_prompt_name="USER_PROMPT_V0",
        prompt_caching=True,
        reasoning=False,
        reasoning_effort=None,
    ),
]
```

This runs the model three times. At a consensus threshold of `0.5`, a tag must be assigned to a character in at least two of those runs to appear in the consensus. Give each configuration a distinct `short_name` so its output is stored separately.

Add more entries to compare models or prompt settings. Their final consensuses are combined into the superconsensus. Fields omitted from an entry use the defaults in `src/config/coding.py`.

[Configuration Reference](docs/configuration.md) covers all constants in the four configuration modules, per-model fields and sampling settings.

## Usage

The following steps take a dataset from input files to a reviewed export. Keep the same `PREFIX` and `RUN_TYPE` throughout one run's coding and post-processing stages.

### Prepare the Dataset

1. Place the spreadsheets and human export at the paths listed under [Input Data](#input-data), or change the source paths in `src/config/project.py`.
2. Choose training/validation selection settings in `src/config/project.py`. For document-driven training, choose participant IDs and sample sizes in `scripts/sample_selector.py`.
3. If selection has not already been recorded, run the selector. It asks before writing `dataset` and `dataset_order` to the source spreadsheet and preserves existing manual `excluded` values:

   ```bash
   uv run python -m scripts.sample_selector
   ```

4. Generate transcript files and combined Word exports:

   ```bash
   uv run python -m scripts.atlas_preperation
   ```

   This writes the `training`, `validation`, `full` and `full_without_validation` folders under `data/sources/atlas_import/`, plus Word files under `word/`. It creates `test/` for manually supplied files but does not populate it. Matching transcript files are overwritten, stale files are not removed automatically.

Choose the run type according to the intended processing and report coverage:

| `RUN_TYPE` | Files processed | Report coverage |
| --- | --- | --- |
| `training` | `atlas_import/training/` | Training files |
| `validation` | `atlas_import/validation/` | Validation files |
| `full_without_validation` | `atlas_import/full_without_validation/` | Training files only |
| `full` | `atlas_import/training/`, `validation/` and `full/`, each document once | Separate training, validation and combined training + validation reports |
| `test` | Manually placed `atlas_import/test/` files | Those test files |

Before using a full-dataset mode, check [run scopes](docs/pipeline.md#run-types-and-scopes) for how transcript copies and report coverage are handled.

### Run Coding and Comparison

For a new AI run, edit `src/config/general.py`, for example:

```python
PREFIX = "my_run"
RUN_TYPE = "training"
RUN_AI_CODING = True
```

Once your coding configuration is ready, run:

```bash
uv run python run.py
```

If cost estimation is enabled, the script will ask you to confirm before proceeding. When the run finishes, it saves the reports and opens the main HTML comparison.

If you already have AI coding from an earlier run, you can regenerate its reports with `RUN_AI_CODING=False`. Keep the prefix, run type, configuration names and iteration counts matched to the saved run.

### Review and Correct Annotations

After comparing the results, you can review and correct the AI annotations without making further AI calls. Select one step at a time in `src/config/post_processing.py` and run:

```bash
uv run python run_post_processing.py
```

1. Set `POST_PROCESS_STEP=1` to generate `01_undefined_tags/review_interface_undefined_tags_{PREFIX}.html` beneath `data/results/{PREFIX}/{RUN_TYPE}/3-post_process/`. If no review page exists yet, it is generated whether `POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE` is `True` or `False`. Open it, review `other` and undefined-tag occurrences and save the edited HTML under the same name. Previous correction databases can prefill suggestions when `POST_PROCESS_LOAD_DATABASE=True`.
2. Set `POST_PROCESS_STEP=2` and run the command again. This merges saved Step 1 corrections, writes validation reports and generates `02_manual_supervision/review_interface_general_{PREFIX}.html` for full review.
3. Open the full review page to retag, move, delete, or add annotations. Mark completed paragraphs reviewed and mark paragraphs excluded if their source rows should be removed from the final spreadsheet export. Save the HTML before continuing.

Before moving to the next step, save the edited HTML at its original path. If your browser downloads a copy, replace the original file with it. You can use the page's protection toggle to prevent regeneration of saved work.

To reuse your Step 1 decisions in later runs, save the edited review HTML, keep `POST_PROCESS_STEP=1`, set or leave `POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE=True` and run the command again. This saves tag replacements and deletions to `01_undefined_tags/database_{PREFIX}.xlsx`, creating the database if needed, then stops without regenerating the existing review page. Set the flag to `False` for test runs whose decisions should not be saved to the database, or to regenerate an existing Step 1 review page, subject to protection and overwrite confirmation. With `POST_PROCESS_LOAD_DATABASE=True`, matching database decisions are applied when a page is generated, independently of the integration flag. Keep the database file when regenerating review outputs.

Saving decisions to the database is optional for continuing to Step 2. See [the review guide](docs/pipeline.md#post-processing) for database matching, reuse across runs and what happens when you rerun a step.

### Export Reviewed Coding

When you're happy with the saved full review, set `POST_PROCESS_STEP=3` and run:

```bash
uv run python run_post_processing.py
```

By default, Step 3 reads the saved full-review HTML and asks before proceeding if paragraphs remain unreviewed. It writes review JSON, ordered per-paragraph tags and `03_export/output_{PREFIX}.xlsx` with an `ordered_tags` column. Rows marked excluded in the Step 2 page are removed from that exported workbook.

If you're working with sampled or trimmed transcripts, check the [export requirements](docs/pipeline.md#step-3-export-the-saved-full-review) before exporting.

## Outputs and How to read them

Outputs are grouped by run label and run type:

| Location | What to open or inspect |
| --- | --- |
| `data/results/{PREFIX}/{RUN_TYPE}/2-results/` | Main HTML comparison and Excel metrics |
| `data/results/{PREFIX}/{RUN_TYPE}/1-ai_coding/` | Raw per-configuration/iteration coding and consensus JSON |
| `data/results/{PREFIX}/{RUN_TYPE}/3-post_process/` | Numbered review stages and final export |
| `data/results/{PREFIX}/{RUN_TYPE}/process_log/process_log_{PREFIX}.xlsx` | Errors, warnings, review outcomes and console decisions |

The **HTML comparison** is a good place to start. Open `comparison_interface_{PREFIX}.html` to compare human and AI spans, inspect labels and metadata and filter documents/tags. You can also switch to a combined view that overlays the annotations. By default, the AI column shows the superconsensus.

For a summary in **Excel**, open `configuration_matrix_main_{PREFIX}_including_undefined_tags.xlsx`. When undefined tags are enabled, an additional excluding-undefined-tags workbook lets you see how those annotations affect the scores.

| Sheet | How to use it |
| --- | --- |
| `overview` | Compare configurations' consensus scores and iteration mean/SD |
| `document_metrics`, `row_metrics` | Find documents or paragraphs with lower agreement |

When reading scores:

- Levenshtein is a tag-sequence **distance**: lower is better. Jaccard, F1 and character agreement are better when higher.
- Tag-set metrics compare which labels occur, character metrics also measure placement. Pruned metrics ignore whitespace, dots and commas.
- Consensus scores evaluate the combined annotations separately from the average iteration score.

The Step 2 comparison reports reflect Step 1 corrections. Later edits in the full review page appear in the Step 3 export. The [metrics guide](docs/metrics-and-rendering.md) explains the remaining workbook sheets, scoring rules and aggregation.

##Known issues
The human reference coding (HeadCT_ThinkAlouds_codes.xlsx) uses five code names from an earlier version of the coding scheme (333 of 5,169 codes). These are not mapped and count as disagreements in the reported metrics; see TA5.2 of the technical appendix
Krippendorff’s α (Hayes and Krippendorff 2007) is currently not implemented
The computation of standard deviations in the improvement run is currently not implemented

## Further Documentation

- [Configuration Reference](docs/configuration.md) - all configuration constants, model fields, API connection and sampling settings.
- [Pipeline](docs/pipeline.md) - run scopes, post-processing lifecycle, saving, protection and process logs.
- [Data Loading](docs/data-loading.md) - input schemas, transcript generation, identity and offset rules.
- [AI Coding](docs/ai-coding.md) - prompts, requests, caching, reasoning, cost estimates and recorded usage.
- [Metrics and Rendering](docs/metrics-and-rendering.md) - score interpretation, workbook structure and HTML controls.
- [Architecture Notes](docs/architecture-notes.md) - implementation boundaries, invariants, adaptation and known limitations.
