from collections.abc import Sequence
from typing import TypedDict

from Levenshtein import distance


class PrecisionRecallF1(TypedDict):
    """Precision, recall, and F1 metrics for one comparison scope."""

    precision: float
    recall: float
    f1: float


class CharacterLevelScores(TypedDict):
    """Character-level scores for one text normalization mode."""

    agreement: float
    char_f1: float


class CharacterLevelF1Result(TypedDict):
    """Character-level F1 scores for original and pruned text."""

    original: CharacterLevelScores
    pruned: CharacterLevelScores


def normalized_levenshtein_distance(obj1: Sequence, obj2: Sequence) -> float:
    """Calculate the normalized Levenshtein distance between two strings.

    Args:
        obj1: First string.
        obj2: Second string.

    Returns:
        Normalized Levenshtein distance between 0.0 and 1.0.
    """
    # Calculate Levenshtein distance
    lev_distance = distance(obj1, obj2)
    max_len = max(len(obj1), len(obj2))

    # Normalize the distance by the length of the longer sequence
    return lev_distance / max_len if max_len > 0 else 0.0


def jaccard_similarity(tags1: list, tags2: list) -> float:
    """Calculate Jaccard similarity between two lists of tags.

    Args:
        tags1: First list of tags.
        tags2: Second list of tags.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 1.0


def f1_score(tags1: list, tags2: list) -> float:
    """Calculate F1 score between two lists of tags.

    Args:
        tags1: First list of tags.
        tags2: Second list of tags.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1 & set2)

    precision = intersection / len(set1) if len(set1) > 0 else 0.0
    recall = intersection / len(set2) if len(set2) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def character_level_f1(
    gold_quotes: list,
    pred_quotes: list,
    text: str,
    start_offset: int = 0,
) -> CharacterLevelF1Result:
    """Calculate character-level F1 scores between two sets of quotes.

    Args:
        gold_quotes: Reference (ground-truth) quotes.
        pred_quotes: Predicted quotes to evaluate.
        text: The full text against which character positions are resolved.
        start_offset: Optional offset subtracted from quote start/end indices.

    Returns:
        Dictionary with two score sets:
        - ``original`` uses the original text unchanged.
        - ``pruned`` ignores whitespace, dots, and commas.

        Each score set contains ``agreement`` and ``char_f1``.
    """

    text_length = len(text)
    gold_tags: list[set[str]] = [set() for _ in range(text_length)]
    pred_tags: list[set[str]] = [set() for _ in range(text_length)]

    # Map each quote's tags to the characters it spans.
    for quotes, char_tags in [(gold_quotes, gold_tags), (pred_quotes, pred_tags)]:
        for quote in quotes:
            if quote.start is not None and quote.end is not None:
                start = max(0, quote.start - start_offset)
                end = min(text_length, quote.end - start_offset)
                for i in range(start, end):
                    char_tags[i].update(tag.name for tag in quote.tags)

    original = _calculate_character_level_scores(gold_tags, pred_tags)
    pruned = _calculate_character_level_scores(
        gold_tags,
        pred_tags,
        included_indices=[
            i for i, character in enumerate(text) if not _is_pruned_character(character)
        ],
    )

    return {"original": original, "pruned": pruned}


def character_level_confusion(
    gold_quotes: list,
    pred_quotes: list,
    tag_name: str,
    text: str,
    start_offset: int = 0,
) -> tuple[int, int, int]:
    """Character-level true/false positive/negative counts for one tag.

    The same per-label bookkeeping _calculate_character_level_scores() does
    internally for every label in gold_quotes/pred_quotes at once, scoped to
    a single tag and exposed as a reusable primitive (see src.analysis.tag_analysis,
    which sums these across documents before deriving precision/recall/F1).

    Args:
        gold_quotes: Reference (ground-truth) quotes.
        pred_quotes: Predicted quotes to evaluate.
        tag_name: The single tag to count characters for.
        text: The full text against which character positions are resolved.
        start_offset: Optional offset subtracted from quote start/end indices.

    Returns:
        (true_positives, false_positives, false_negatives) character counts.
    """
    text_length = len(text)
    gold_has_tag = [False] * text_length
    pred_has_tag = [False] * text_length

    for quotes, has_tag in [(gold_quotes, gold_has_tag), (pred_quotes, pred_has_tag)]:
        for quote in quotes:
            if quote.start is None or quote.end is None:
                continue
            if not any(tag.name == tag_name for tag in quote.tags):
                continue
            start = max(0, quote.start - start_offset)
            end = min(text_length, quote.end - start_offset)
            for i in range(start, end):
                has_tag[i] = True

    true_positives = sum(1 for g, p in zip(gold_has_tag, pred_has_tag) if g and p)
    false_positives = sum(1 for g, p in zip(gold_has_tag, pred_has_tag) if not g and p)
    false_negatives = sum(1 for g, p in zip(gold_has_tag, pred_has_tag) if g and not p)
    return true_positives, false_positives, false_negatives


def _is_pruned_character(character: str) -> bool:
    """Return whether a character is ignored by pruned character metrics."""
    return character.isspace() or character in {".", ","}


def _calculate_character_level_scores(
    gold_tags: list[set[str]],
    pred_tags: list[set[str]],
    included_indices: list[int] | None = None,
) -> CharacterLevelScores:
    """Calculate character-level F1 and agreement scores for selected positions."""

    def _precision_recall_f1(
        true_positives: int, false_positives: int, false_negatives: int
    ) -> PrecisionRecallF1:
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
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    if included_indices is None:
        included_indices = list(range(len(gold_tags)))

    filtered_gold_tags = [gold_tags[i] for i in included_indices]
    filtered_pred_tags = [pred_tags[i] for i in included_indices]
    score_length = len(included_indices)

    # Collect all labels that appear in either set.
    all_labels = set(
        tag for char_set in filtered_gold_tags + filtered_pred_tags for tag in char_set
    )

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for label in all_labels:
        true_positives = sum(
            1
            for gold_set, pred_set in zip(filtered_gold_tags, filtered_pred_tags)
            if label in gold_set and label in pred_set
        )
        false_positives = sum(
            1
            for gold_set, pred_set in zip(filtered_gold_tags, filtered_pred_tags)
            if label not in gold_set and label in pred_set
        )
        false_negatives = sum(
            1
            for gold_set, pred_set in zip(filtered_gold_tags, filtered_pred_tags)
            if label in gold_set and label not in pred_set
        )

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    exact_matches = sum(
        gold_set == pred_set
        for gold_set, pred_set in zip(filtered_gold_tags, filtered_pred_tags)
    )
    agreement = exact_matches / score_length if score_length > 0 else 1.0

    overall = _precision_recall_f1(
        total_true_positives, total_false_positives, total_false_negatives
    )
    return {"agreement": agreement, "char_f1": overall["f1"]}
