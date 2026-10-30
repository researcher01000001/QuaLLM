# Metrics and Rendering

Metrics are computed by `src/analysis/agreement.py`, orchestrated through `CodedRow.compare_coders()` and `src/analysis/metrics.py`, assembled by `src/report/xlsx_rows.py`, and written by `src/report/xlsx.py`. HTML uses `src/report/html.py::DocumentProcessor` and `src/templates/compare.html`.

## What is compared

Each comparison pairs human ATLAS quotations with an AI iteration or a derived consensus. Quotes map labels onto exact text spans. Technical row indices are zero-based; a separate `DisplayColumn` supplies source paragraph values, currently labelled `Paragraph`.

Before scoring, system tags (`Tag.system`, from the coding scheme's `system` column) are always excluded. `drop_undefined_tags=True` additionally removes `Tag.additional_tag` annotations. These filters remove tags from copied quotes; they do not delete the underlying annotations.

## Tag-level metrics

Quotes are ordered by their resolved start offsets and their tags are flattened into sequences.

| Metric | Interpretation | Best value |
| --- | --- | --- |
| Normalized Levenshtein distance | Edit distance between tag sequences divided by the longer length; preserves order and repetition | `0` |
| Jaccard similarity | Intersection / union of unique tag names; ignores order and duplicates | `1` |
| Tag F1 | Set-based precision/recall F1; ignores order and duplicates | `1` |

Empty comparisons have deliberately different conventions: two empty tag sequences have distance `0` and Jaccard `1`, but tag F1 `0`. A single empty side gives Jaccard and F1 `0`.

## Character-level metrics

`character_level_f1()` projects each quote's tags onto the characters in `[start, end)`, then computes:

- `agreement`: fraction of characters with exactly equal tag sets. Characters unlabelled by both sides count as agreements.
- `char_f1`: micro F1 over character/label assignments, summing true positives, false positives, and false negatives over all labels. Shared unlabelled characters do not increase F1.

Both are returned for `original` text and `pruned` text. Pruning ignores whitespace, dots, and commas only. If no character/label positives exist, F1 is `0`; with no characters to score, agreement is `1`.

Unresolved quote positions do not contribute character labels. An iteration quote can still contribute tags to the tag-level comparison even if it cannot be located. Ambiguous positions are retained as `position_unsure_<tag>` undefined tags; excluding undefined tags removes those renamed labels too. See [quote positioning](data-loading.md#ai-iteration-json).

## Consensus voting

`src/coding/consensus.py` votes independently on every character/tag pair. A label is retained when:

```text
agreeing iterations / configured iterations >= consensus_threshold
```

At threshold `0.5`, three iterations need two votes; two iterations need only one. Missing iteration output is loaded as empty coding and still contributes to the configured denominator. Adjacent characters with the same tag set are compressed into `ConsensusQuote` spans.

Cumulative consensuses use iterations `0..k`. The final cumulative result is reused for final consensus metrics. A superconsensus then votes over the final configuration consensuses using `SUPERCONSENSUS_THRESHOLD`, giving each configuration one vote regardless of its iteration count. With one configuration its character labels are unchanged.

Consensus scores evaluate a new prediction, not an average of run scores:

```text
consensus metric = score(human, consensus(iteration predictions))
iteration mean  = mean(score(human, each iteration prediction))
```

Keep these separate: consensus is derived from the runs, not an independent extra iteration.

The exporter preserves original quotes for single-iteration configuration files, whereas in-memory consensus metrics use compressed character-voted spans. HTML reloads the exported file. Character labels may agree while tag-sequence and span counts differ between these representations.

## Workbook sheets

Single-scope main reports are written under `data/results/{PREFIX}/{RUN_TYPE}/2-results/`:

- `configuration_matrix_main_{PREFIX}_including_undefined_tags.xlsx` is always written.
- `configuration_matrix_main_{PREFIX}_excluding_undefined_tags.xlsx` is also written when `UNDEFINED_TAGS_ENABLED=True`.

For `RUN_TYPE="full"`, insert `__training`, `__validation`, or `__combined` after `{PREFIX}` in each filename.

| Sheet | Contents |
| --- | --- |
| `overview` | One row per configuration with metadata, run mean/SD, and consensus scores; superconsensus appears as a synthetic configuration |
| `document_metrics` | One row per configuration/document with run summaries and consensus scores |
| `row_metrics` | One row per configuration/document/row, including the source display value |
| `iteration_metrics` | Raw paragraph scores ordered by configuration → document → row → iteration; no consensus rows |
| `consensus_progression` | Per-configuration `consensus_0_to_k` columns of average pruned character agreement; unavailable iterations are blank |
| `comparison` | Primary including report's overview rows plus configured cross-prefix comparisons; otherwise empty |
| `metadata` | Full `CodingConfiguration.to_metadata()` values, with fallback metadata for synthetic configurations |
| `tag_analysis` | Final-consensus span matches, per-tag character precision/recall/F1 and coverage |
| `tag_confusion` | Human/superconsensus unequal tag-pair character counts |

`COMPARISON_PREFIXES` looks up each prefix's including workbook at the same `RUN_TYPE` and, for `full`, the same primary `combined` scope. Missing comparison workbooks are warned about and skipped. This only populates the primary including report, and only when the comparison-prefix list is nonempty.

Step 2 writes the same workbook structure beneath `data/results/{PREFIX}/{RUN_TYPE}/3-post_process/02_manual_supervision/`, using `configuration_matrix_pp2_{PREFIX}_*.xlsx`. It passes only corrected-consensus metrics: iteration, progression, comparison, tag-analysis, and confusion sheets have no data. A `step1_review_outcomes` sheet is appended to the including workbook when there are outcomes. These reports measure Step 1 corrections, not subsequent edits in the full-review HTML.

### Aggregation and column names

`src/analysis/metrics.py` averages paragraph scores to obtain a document score. The overview gives documents equal weight, rather than pooling all characters or paragraphs across documents. For each iteration, document scores are averaged first; run mean and population SD (`statistics.pstdev`) are then taken across those per-iteration averages. Document and row sheets summarize their own run values similarly. One iteration has SD `0`.

The two aggregation paths are explicit:

```text
mean path:
iteration predictions → paragraph scores → document means
                     → iteration summaries (mean / population SD)

consensus path:
iteration predictions → voted spans → paragraph consensus scores
                                  → document / overview means
```

Superconsensus rows contain consensus scores and no iteration summaries. `consensus_progression` uses document means of pruned agreement at each cumulative step, and excludes superconsensus.

Aggregate columns use:

```text
{metric}_{level}-{aggregation}_{source}
```

Examples:

```text
lev-distance_paragraph-mean_mean
lev-distance_paragraph-sd_mean
lev-distance_paragraph-consensus_consensus
lev-distance_document-mean_mean
lev-distance_document-sd_mean
lev-distance_document-mean_consensus
```

Metric names use hyphens in aggregate columns (`original-char-f1`, `pruned-char-agreement`); raw iteration columns retain Python field names such as `original_char_f1`.

### Per-tag analysis

`src/analysis/tag_analysis.py::compute_tag_analysis_rows()` compares final configuration consensuses and nonempty superconsensus coding against human spans. Rows are broken out by raw document condition (`general_cluster`) and an `all_general_clusters` roll-up. Valid codebook tags are included even with zero observations; observed undefined tags are included when enabled. System tags are excluded.

Within each document/tag, spans are matched greedily by descending intersection-over-union, each span used at most once:

- `exact_match_count`: identical boundaries.
- `threshold_match_count`: overlap at least `0.9`, excluding exact matches.
- `part_match_count`: positive overlap below `0.9`.
- `unfound_count`: unmatched human spans.
- `additional_count`: unmatched AI spans.

The analysis threshold is the function's default `0.9`; `POST_PROCESS_SPAN_OVERLAP_THRESHOLD` independently controls correction-database matching.

Per-tag character counts are pooled across documents before deriving precision, recall, and F1. Human-side precision/recall mirror AI-side recall/precision; F1 is the same. Coverage columns are fractions, despite their `pct` names: tagged characters divided by all coded characters or by all document characters. Counts use original text, including whitespace/punctuation; newline separators enter the total document-character denominator. Overlapping tags can make the sum of per-tag coverage exceed one. Workbook header comments explain each column.

### Tag confusion

`compute_tag_confusion_rows()` compares human coding with superconsensus only. At each character labelled by both sides, it counts every unequal `(human_tag, ai_tag)` combination, broken out by condition plus a roll-up. It omits unlabelled-vs-labelled cases; those are handled by span/character metrics.

With multiple tags on a character, there can be several counts, including unequal cross-pairs even when both coders share the same multi-tag set. Interpret this as an unequal tag-pair tally, not a mutually exclusive classification matrix or a count of all annotation errors.

## HTML comparison

`DocumentProcessor` renders one row per transcript paragraph, with human and selected AI consensus columns, highlights, tag badges, tooltips, source metadata, and agreement scores. Default output is `data/results/{PREFIX}/{RUN_TYPE}/2-results/comparison_interface_{PREFIX}.html`; `full` adds scope suffixes.

`HTML_CONSENSUS_CONFIGURATIONS=[]` selects the top-level superconsensus file. A nonempty list selects those configurations' own `consensus.json` files, each as a column. There is no `SUPERCONSENSUS` display boolean. When several AI columns are shown, displayed paragraph/document scores compare the human coder with the **first** selected AI source only; Excel still contains metrics for all configured runs/consensuses.

Controls include:

- Separate or combined annotation views, tag badges and duplicate-tag display.
- Metadata/metric column selection, identity filters, and a tag filter for visible rows.
- Document/paragraph folding, reviewed markers, and per-document progress.
- System-tag visibility, hidden by default; showing system tags never adds them to scores.
- Undefined-tag visibility when enabled; this switches between precomputed including/excluding score variants as well as changing highlights. Tag/identity filters do not recalculate aggregate scores.
- Browser local-storage state restoration and a saved HTML copy containing view state. These comparison-page review markers are separate from manual-review workflow data.

The main report's filter columns come from parsed document identity keys; the manual-review page uses `DOCUMENT_KEY_COLUMNS`. See [the review workflow](pipeline.md#saving-and-review-controls) for editing and saving actual annotations.
