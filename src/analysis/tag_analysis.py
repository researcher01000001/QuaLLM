"""Per-tag coding-quality analysis"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.analysis.agreement import character_level_confusion
from src.core.models import (
    Document,
    Quote,
    Tag,
    quotes_excluding_system,
    quotes_excluding_undefined,
)
from src.dataset.document_identity import condition_for, condition_match

ALL_GENERAL_CLUSTERS_LABEL = "all_general_clusters"


def has_any_quotes(quotes_by_document: Mapping[str, Sequence[Quote]]) -> bool:
    """True if quotes_by_document holds at least one quote in any document.

    A mapping sliced to one report scope (see run._slice_to_scope()) always
    has one entry per scoped document name -- even a genuinely empty source
    mapping (e.g. no superconsensus was built at all, because zero coding
    configurations are configured -- see run.build_superconsensus_quote_set()
    docstring; one configuration's superconsensus is that configuration's own
    consensus, not empty) becomes {doc_name: [] for doc_name in scope} after
    slicing, which is a non-empty dict. Plain truthiness on such a mapping is
    therefore always True; this checks the thing that actually matters --
    whether any document has a quote in it.
    """
    return any(quotes_by_document.values())


@dataclass(frozen=True)
class SpanMatchCounts:
    """Span-level match classification counts for one tag in one scope."""

    human_total: int
    ai_total: int
    exact: int
    threshold: int
    part: int
    unfound: int
    additional: int


def span_overlap_ratio(
    a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None
) -> float:
    """Intersection-over-union of two spans, 0.0 if they don't overlap (or
    either offset is missing). Shared by ``classify_span_matches()`` (below)
    and the cross-run database span matching in ``src.post_processing.review_database``,
    so both use the exact same overlap-ratio formula."""
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0.0
    intersection = min(a_end, b_end) - max(a_start, b_start)
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def classify_span_matches(
    gold_spans: Sequence[tuple[int, int]],
    pred_spans: Sequence[tuple[int, int]],
    overlap_threshold: float = 0.9,
) -> SpanMatchCounts:
    """Greedily match gold_spans against pred_spans by overlap ratio
    (intersection length / union length of the two spans), highest overlap
    first, each span used at most once (standard maximum-weight greedy
    bipartite matching -- good enough for span alignment, doesn't require an
    exact assignment solver).

    A matched pair with ratio == 1.0 (identical start/end) is "exact";
    >= overlap_threshold but < 1.0 is "threshold"; anything lower is "part".
    A gold span left unmatched (no overlapping pred span at all) is
    "unfound"; a pred span left unmatched is "additional".
    """
    candidates: list[tuple[float, int, int]] = []
    for gold_index, (gold_start, gold_end) in enumerate(gold_spans):
        for pred_index, (pred_start, pred_end) in enumerate(pred_spans):
            ratio = span_overlap_ratio(gold_start, gold_end, pred_start, pred_end)
            if ratio <= 0:
                continue
            candidates.append((ratio, gold_index, pred_index))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    exact = threshold = part = 0
    for ratio, gold_index, pred_index in candidates:
        if gold_index in matched_gold or pred_index in matched_pred:
            continue
        matched_gold.add(gold_index)
        matched_pred.add(pred_index)
        if ratio >= 1.0:
            exact += 1
        elif ratio >= overlap_threshold:
            threshold += 1
        else:
            part += 1

    return SpanMatchCounts(
        human_total=len(gold_spans),
        ai_total=len(pred_spans),
        exact=exact,
        threshold=threshold,
        part=part,
        unfound=len(gold_spans) - len(matched_gold),
        additional=len(pred_spans) - len(matched_pred),
    )


def compute_tag_analysis_rows(
    documents: Mapping[str, Document],
    human_quotes: Mapping[str, Sequence[Quote]],
    ai_quotes_by_configuration: Mapping[str, Mapping[str, Sequence[Quote]]],
    tags: Sequence[Tag],
    drop_undefined_tags: bool = False,
    overlap_threshold: float = 0.9,
) -> list[dict[str, object]]:
    """One row per (tag, configuration, general_cluster), plus one
    "all_general_clusters" roll-up row per (tag, configuration) -- see the
    module docstring and the tag_analysis feature plan for full column
    definitions. ai_quotes_by_configuration keys are configuration names
    (e.g. one per CodingConfiguration.configuration_name, plus
    config.SUPERCONSENSUS_NAME when available) each mapped to that
    configuration's final consensus quotes by document name.
    """
    doc_names = list(documents.keys())
    buckets = _document_buckets(doc_names)
    codebook_tags = [tag for tag in tags if not tag.system]

    rows: list[dict[str, object]] = []
    for configuration_name, ai_quotes in ai_quotes_by_configuration.items():
        prepared_human = {
            name: _prepare_quotes(human_quotes.get(name, []), drop_undefined_tags)
            for name in doc_names
        }
        prepared_ai = {
            name: _prepare_quotes(ai_quotes.get(name, []), drop_undefined_tags)
            for name in doc_names
        }

        tag_bucket_labels: dict[str, list[str]] = defaultdict(list)
        for bucket_label, bucket_doc_names in buckets.items():
            allowed_conditions = condition_match(bucket_doc_names[0])
            bucket_tag_names = _tags_for_bucket(
                codebook_tags, allowed_conditions, bucket_doc_names, prepared_human, prepared_ai
            )
            for tag_name in bucket_tag_names:
                rows.append(
                    _tag_row(
                        tag_name,
                        configuration_name,
                        bucket_label,
                        bucket_doc_names,
                        documents,
                        prepared_human,
                        prepared_ai,
                        overlap_threshold,
                    )
                )
                tag_bucket_labels[tag_name].append(bucket_label)

        for tag_name, bucket_labels in tag_bucket_labels.items():
            all_doc_names = [name for label in bucket_labels for name in buckets[label]]
            rows.append(
                _tag_row(
                    tag_name,
                    configuration_name,
                    ALL_GENERAL_CLUSTERS_LABEL,
                    all_doc_names,
                    documents,
                    prepared_human,
                    prepared_ai,
                    overlap_threshold,
                )
            )

    return rows


def compute_tag_confusion_rows(
    documents: Mapping[str, Document],
    human_quotes: Mapping[str, Sequence[Quote]],
    superconsensus_quotes: Mapping[str, Sequence[Quote]],
    drop_undefined_tags: bool = False,
) -> list[dict[str, object]]:
    """Character-level tag-substitution tally: for every character where
    both the human and the superconsensus assign at least one tag but
    disagree, count one (human_tag, ai_tag) pair per differing combination.
    This captures mislabeling independent of exact span boundaries --
    deliberately distinct from compute_tag_analysis_rows()'s
    unfound/additional counts, which are about presence/absence.

    Computed only against the superconsensus (never a per-configuration
    consensus or iteration); empty only when there's no real superconsensus
    data at all (zero coding configurations -- see
    run.build_superconsensus_quote_set()). With exactly one configuration,
    the superconsensus is that configuration's own consensus, so this
    compares human coding against it same as with any other configuration
    count.

    Note: superconsensus_quotes is checked via has_any_quotes() rather than
    a plain truthiness check -- run._slice_to_scope() always rebuilds this
    mapping as {doc_name: quotes_by_doc.get(doc_name, []) for doc_name in
    scope}, so it has one entry per scoped document (and is therefore always
    a non-empty dict) even when every one of those entries is an empty list
    because no superconsensus was ever built at all.
    """
    if not has_any_quotes(superconsensus_quotes):
        return []

    doc_names = list(documents.keys())
    buckets = _document_buckets(doc_names)

    prepared_human = {
        name: _prepare_quotes(human_quotes.get(name, []), drop_undefined_tags)
        for name in doc_names
    }
    prepared_ai = {
        name: _prepare_quotes(superconsensus_quotes.get(name, []), drop_undefined_tags)
        for name in doc_names
    }

    def confusion_counts(bucket_doc_names: Sequence[str]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for name in bucket_doc_names:
            document = documents.get(name)
            if document is None:
                continue
            text = _document_text(document)
            human_tags_by_char, ai_tags_by_char = _char_tag_sets(
                prepared_human.get(name, []), prepared_ai.get(name, []), len(text)
            )
            for human_tags, ai_tags in zip(human_tags_by_char, ai_tags_by_char):
                if not human_tags or not ai_tags:
                    continue
                for human_tag in human_tags:
                    for ai_tag in ai_tags:
                        if human_tag != ai_tag:
                            counts[(human_tag, ai_tag)] += 1
        return counts

    rows: list[dict[str, object]] = []
    all_doc_names: list[str] = []
    for bucket_label, bucket_doc_names in buckets.items():
        all_doc_names.extend(bucket_doc_names)
        for (human_tag, ai_tag), count in confusion_counts(bucket_doc_names).items():
            rows.append(
                {
                    "human_tag": human_tag,
                    "ai_tag": ai_tag,
                    "general_cluster": bucket_label,
                    "character_count": count,
                }
            )

    for (human_tag, ai_tag), count in confusion_counts(all_doc_names).items():
        rows.append(
            {
                "human_tag": human_tag,
                "ai_tag": ai_tag,
                "general_cluster": ALL_GENERAL_CLUSTERS_LABEL,
                "character_count": count,
            }
        )

    return rows


def _tag_row(
    tag_name: str,
    configuration_name: str,
    general_cluster: str,
    doc_names: Sequence[str],
    documents: Mapping[str, Document],
    prepared_human: Mapping[str, Sequence[Quote]],
    prepared_ai: Mapping[str, Sequence[Quote]],
    overlap_threshold: float,
) -> dict[str, object]:
    span_totals = SpanMatchCounts(0, 0, 0, 0, 0, 0, 0)
    true_positives = false_positives = false_negatives = 0
    tag_ai_chars = total_ai_chars = 0
    tag_human_chars = total_human_chars = 0
    total_chars = 0

    for name in doc_names:
        document = documents.get(name)
        if document is None:
            continue
        text = _document_text(document)
        total_chars += len(text)

        human = prepared_human.get(name, [])
        ai = prepared_ai.get(name, [])

        human_spans = _tag_spans(human, tag_name)
        ai_spans = _tag_spans(ai, tag_name)
        span_totals = _add_span_counts(
            span_totals, classify_span_matches(human_spans, ai_spans, overlap_threshold)
        )

        tp, fp, fn = character_level_confusion(human, ai, tag_name, text)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        tag_ai_chars += _tag_char_count(ai, tag_name, len(text))
        total_ai_chars += _any_tag_char_count(ai, len(text))
        tag_human_chars += _tag_char_count(human, tag_name, len(text))
        total_human_chars += _any_tag_char_count(human, len(text))

    ai_precision, ai_recall, ai_f1 = _precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    # Human-side view is the mirror comparison: same true positives, but
    # precision/recall swapped (false_negatives become the human's own false
    # positives against the AI, and vice versa).
    human_precision, human_recall, human_f1 = _precision_recall_f1(
        true_positives, false_negatives, false_positives
    )

    return {
        "tag": tag_name,
        "configuration": configuration_name,
        "general_cluster": general_cluster,
        "human_tag_count": span_totals.human_total,
        "ai_tag_count": span_totals.ai_total,
        "exact_match_count": span_totals.exact,
        "threshold_match_count": span_totals.threshold,
        "part_match_count": span_totals.part,
        "unfound_count": span_totals.unfound,
        "additional_count": span_totals.additional,
        "ai_char_precision": ai_precision,
        "ai_char_recall": ai_recall,
        "ai_char_f1": ai_f1,
        "human_char_precision": human_precision,
        "human_char_recall": human_recall,
        "human_char_f1": human_f1,
        "ai_pct_of_coded_characters": _safe_div(tag_ai_chars, total_ai_chars),
        "ai_pct_of_all_characters": _safe_div(tag_ai_chars, total_chars),
        "human_pct_of_coded_characters": _safe_div(tag_human_chars, total_human_chars),
        "human_pct_of_all_characters": _safe_div(tag_human_chars, total_chars),
    }


def _document_buckets(doc_names: Sequence[str]) -> dict[str, list[str]]:
    """Group doc_names by their raw GENERAL_CLUSTER_COLUMN value
    (document_identity.condition_for()). Documents whose name doesn't
    resolve a condition are skipped -- nothing to group them by."""
    buckets: dict[str, list[str]] = {}
    for doc_name in doc_names:
        value = condition_for(doc_name)
        if value is None:
            continue
        buckets.setdefault(value, []).append(doc_name)
    return buckets


def _tags_for_bucket(
    codebook_tags: Sequence[Tag],
    allowed_conditions: Sequence[str],
    bucket_doc_names: Sequence[str],
    prepared_human: Mapping[str, Sequence[Quote]],
    prepared_ai: Mapping[str, Sequence[Quote]],
) -> dict[str, Tag]:
    """Codebook tags valid for this bucket's allowed conditions, plus any
    undefined/additional tag actually observed in this bucket's quotes
    (those aren't restricted by Tag.condition -- they're outside the
    codebook, so they belong wherever they were actually coded)."""
    bucket_tags: dict[str, Tag] = {
        tag.name: tag
        for tag in codebook_tags
        if tag.condition is None or tag.condition in allowed_conditions
    }
    for name in bucket_doc_names:
        for quote in list(prepared_human.get(name, [])) + list(prepared_ai.get(name, [])):
            for tag in quote.tags:
                if tag.additional_tag and tag.name not in bucket_tags:
                    bucket_tags[tag.name] = tag
    return bucket_tags


def _prepare_quotes(quotes: Sequence[Quote], drop_undefined_tags: bool) -> list[Quote]:
    """System tags are always excluded (administrative/meta, excluded
    from every agreement calculation elsewhere too -- see
    models.quotes_excluding_system()); undefined tags are additionally
    excluded when drop_undefined_tags is True."""
    scoped = quotes_excluding_system(quotes, True)
    return quotes_excluding_undefined(scoped, drop_undefined_tags)


def _document_text(document: Document) -> str:
    return "\n".join(row.text for row in document.rows)


def _tag_spans(quotes: Sequence[Quote], tag_name: str) -> list[tuple[int, int]]:
    return [
        (quote.start, quote.end)
        for quote in quotes
        if quote.start is not None
        and quote.end is not None
        and any(tag.name == tag_name for tag in quote.tags)
    ]


def _tag_char_count(quotes: Sequence[Quote], tag_name: str, text_length: int) -> int:
    covered = [False] * text_length
    for start, end in _tag_spans(quotes, tag_name):
        for i in range(max(0, start), min(text_length, end)):
            covered[i] = True
    return sum(covered)


def _any_tag_char_count(quotes: Sequence[Quote], text_length: int) -> int:
    covered = [False] * text_length
    for quote in quotes:
        if quote.start is None or quote.end is None or not quote.tags:
            continue
        start = max(0, quote.start)
        end = min(text_length, quote.end)
        for i in range(start, end):
            covered[i] = True
    return sum(covered)


def _char_tag_sets(
    human_quotes: Sequence[Quote], ai_quotes: Sequence[Quote], text_length: int
) -> tuple[list[set[str]], list[set[str]]]:
    human_tags: list[set[str]] = [set() for _ in range(text_length)]
    ai_tags: list[set[str]] = [set() for _ in range(text_length)]
    for quotes, char_tags in [(human_quotes, human_tags), (ai_quotes, ai_tags)]:
        for quote in quotes:
            if quote.start is None or quote.end is None:
                continue
            start = max(0, quote.start)
            end = min(text_length, quote.end)
            for i in range(start, end):
                char_tags[i].update(tag.name for tag in quote.tags)
    return human_tags, ai_tags


def _add_span_counts(a: SpanMatchCounts, b: SpanMatchCounts) -> SpanMatchCounts:
    return SpanMatchCounts(
        human_total=a.human_total + b.human_total,
        ai_total=a.ai_total + b.ai_total,
        exact=a.exact + b.exact,
        threshold=a.threshold + b.threshold,
        part=a.part + b.part,
        unfound=a.unfound + b.unfound,
        additional=a.additional + b.additional,
    )


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
