# AGENTS.md — QuaLLM

## Project

Python 3.13+ pipeline comparing human ATLAS.ti coding with AI coding of think-aloud transcripts. `run.py` handles coding and comparison; `run_post_processing.py` handles AI-free review and spreadsheet export.

## Commands

| Task | Command |
| --- | --- |
| Install dependencies | `uv sync` |
| Run main pipeline | `uv run python run.py` |
| Run selected review/export step | `uv run python run_post_processing.py` |
| Test | `uv run pytest` |
| Lint | `uv run ruff check .` |

## Code organization

- `src/config/general.py` / `config/coding.py`: runtime and per-model settings. `config/project.py`: study paths/schema. `config/post_processing.py`: review/export settings.
- `src/dataset/document_identity.py`: study-specific filename parsing and condition mapping. `dataset/document_content.py` / `dataset/selection.py`: shared source-row lookup and selection; reuse these in preparation and coding.
- Keep reusable logic in `src/`; orchestration in the entry points. See the [module map](docs/pipeline.md#module-map) for ownership.

## Working rules

- Be concise, answer the question first, and ask only when blocked. After edits, report changes, files touched, and validation run.
- Preserve existing behavior unless the task calls for a change. Use type hints and small, focused functions.
- Do not read `data/results/` or `results/` unless explicitly requested; these contain generated outputs.
- Keep secrets in the environment or `.env`. Do not hard-code or log keys, secrets, or unnecessary sensitive transcript data.
- Update relevant `docs/` guides when changing workflows, formats, metrics, or invariants. Keep this file limited to agent instructions.

## Invariants and protected behavior

- Preserve exact character offsets. Internal span ends are exclusive; ATLAS `quote_end` is inclusive and needs the `+1` adjustment.
- `Tag` equality/hash use its name. `AIQuote.find_position()` caches offsets. `Document.append_quotes()` replaces `Document.rows` with `CodedRow` objects.
- AI preparation drops blank lines; `DocumentLoader` preserves line positions. AI results arrive in completion order: use `source.row`, not array position, and load iterations separately.
- Consensus voting uses `>= threshold`; each configuration has one superconsensus vote. Keep consensus scores separate from iteration-score averages.
- Get explicit approval before removing JSON closing-brace repair, CRLF normalization, or the ATLAS inclusive-end adjustment.
- Do not replace the loose `load_tags()` examples parser unless fixing a concrete parsing bug.
- Get explicit approval before changing `document_identity.condition_match()`'s confirmed mapping: advice/advice with annotation → general + supported; no advice → general + human.

## Validation

- Add or update tests for behavior changes. Run `uv run pytest` before finishing non-trivial changes.
- Run `uv run ruff check .` after formatting or import changes.
- Claim checks passed only when the commands succeeded; report anything not run or failing.

## Documentation

- [Configuration](docs/configuration.md): constants and model options.
- [Pipeline](docs/pipeline.md): run scopes, review lifecycle, and export.
- [Data loading](docs/data-loading.md): schemas, identity, and offset rules.
- [AI coding](docs/ai-coding.md): prompts, API calls, caching, and costs.
- [Metrics](docs/metrics-and-rendering.md): scoring, workbooks, and rendering.
- [Architecture](docs/architecture-notes.md): invariants and known limitations.
