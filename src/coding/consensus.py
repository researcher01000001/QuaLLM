import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from itertools import combinations

from src.core import process_log
from src.core.models import (
    AIQuote,
    ConsensusQuote,
    Document,
    Quote,
    Row,
    Tag,
    tags_disallow_co_occurrence,
)
from src.io.io_utils import resolve_output_dir


def build_consensus_quotes(
    document: Document,
    iteration_quotes: Sequence[Sequence[Quote]],
    threshold: float,
    step: str | None = None,
) -> list[ConsensusQuote]:
    """Build character-level majority-vote quotes for one document.

    A tag is included for a character when the share of iterations assigning
    that tag is greater than or equal to ``threshold``. With ``threshold=0.5``
    and three iterations, this requires at least two agreeing iterations.

    ``step`` is an optional process-log step override (see
    ``src.core.process_log.ProcessLog``), passed down to any
    ``AIQuote.find_position()`` call made while resolving quote positions --
    e.g. a specific configuration's namespaced sub-step. Omitted/None falls
    back to the active log's own step.
    """
    if not iteration_quotes:
        return []

    consensus_quotes: list[ConsensusQuote] = []
    iteration_count = len(iteration_quotes)

    for row in document.rows:
        char_tags_by_iteration = _row_char_tags_by_iteration(
            row,
            iteration_quotes,
            step=step,
            document=document,
        )
        consensus_char_tags = _vote_character_tags(
            char_tags_by_iteration,
            iteration_count,
            threshold,
        )
        new_quotes = _compress_character_tags_to_quotes(
            row,
            consensus_char_tags,
            iteration_count,
        )
        consensus_quotes.extend(new_quotes)

    return consensus_quotes


def build_consensus_quote_sets(
    documents: dict[str, Document],
    ai_quote_runs: Mapping[tuple[str, int], Mapping[str, Sequence[Quote]]],
    configuration_names: Iterable[str],
    iterations: int,
    threshold: float,
) -> dict[str, dict[str, list[ConsensusQuote]]]:
    """Build consensus quote mappings for all configurations and documents."""
    consensus_by_configuration: dict[str, dict[str, list[ConsensusQuote]]] = {}

    for configuration_name in configuration_names:
        consensus_by_document: dict[str, list[ConsensusQuote]] = {}
        # This function is only ever called from run.py, always under the
        # STEP_RUN_MAIN top-level step -- see build_consensus_progression_quote_sets().
        step = f"{process_log.STEP_RUN_MAIN}/{configuration_name}"
        for doc_name, document in documents.items():
            iteration_doc_quotes = [
                ai_quote_runs.get((configuration_name, iteration), {}).get(doc_name, [])
                for iteration in range(iterations)
            ]
            consensus_by_document[doc_name] = build_consensus_quotes(
                document,
                iteration_doc_quotes,
                threshold,
                step=step,
            )
        consensus_by_configuration[configuration_name] = consensus_by_document

    return consensus_by_configuration


def build_consensus_progression_quote_sets(
    documents: dict[str, Document],
    ai_quote_runs: Mapping[tuple[str, int], Mapping[str, Sequence[Quote]]],
    configuration_names: Iterable[str],
    iterations: int,
    threshold: float,
) -> dict[int, dict[str, dict[str, list[ConsensusQuote]]]]:
    """Build cumulative consensuses from iterations 0..k for each k."""
    return {
        iteration_limit: build_consensus_quote_sets(
            documents=documents,
            ai_quote_runs=ai_quote_runs,
            configuration_names=configuration_names,
            iterations=iteration_limit + 1,
            threshold=threshold,
        )
        for iteration_limit in range(iterations)
    }


def build_superconsensus_quotes(
    documents: dict[str, Document],
    consensus_quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    configuration_names: Sequence[str],
    threshold: float,
) -> dict[str, list[ConsensusQuote]]:
    """Build one character-level consensus across configuration consensuses.

    With a single configuration there's nothing to vote *across*, but no
    special-casing is needed for it: every character trivially clears any
    threshold <= 1.0 (1 vote out of 1), so build_consensus_quotes() already
    reduces to a character-level copy of that configuration's own consensus
    -- see tests/test_consensus.py::TestBuildSuperconsensusQuotes for the
    single-configuration case. With zero configurations there's nothing to
    build from at all.
    """
    if not configuration_names:
        return {}

    return {
        doc_name: build_consensus_quotes(
            document,
            [
                consensus_quotes.get(configuration_name, {}).get(doc_name, [])
                for configuration_name in configuration_names
            ],
            threshold,
        )
        for doc_name, document in documents.items()
    }


def build_consensus_records(
    documents: Mapping[str, Document],
    quotes_by_doc: Mapping[str, Sequence[Quote]],
) -> list[dict]:
    """Serialize a per-document quote mapping (consensus or a single AI iteration)
    into consensus.json-style records: ``{phrase, tags, source: {source_name,
    row}, start, end, source_iterations}``.

    Works for ``ConsensusQuote`` (row resolved by locating the document row
    containing ``start``) and ``AIQuote`` (uses its own ``row`` directly)
    alike, so the same function serializes both a real character-level
    consensus and the single-iteration fallback used when a configuration
    has only one iteration.
    """
    records: list[dict] = []
    for doc_name, quotes in quotes_by_doc.items():
        document = documents.get(doc_name)
        for quote in quotes:
            records.append(
                {
                    "phrase": quote.text,
                    "tags": [tag.name for tag in quote.tags],
                    "source": {
                        "source_name": doc_name,
                        "row": row_for_quote(document, quote),
                    },
                    "start": quote.start,
                    "end": quote.end,
                    "source_iterations": getattr(quote, "source_iterations", None) or 1,
                }
            )
    return records


def row_for_quote(document: Document | None, quote: Quote) -> int | None:
    """The 0-based row index ``quote`` belongs to within ``document``.

    Uses ``AIQuote``'s own ``row`` directly when available (more reliable
    than re-deriving it from character position); otherwise locates the row
    containing ``quote.start``.
    """
    if isinstance(quote, AIQuote) and quote.row is not None:
        return quote.row
    if document is None or quote.start is None:
        return None
    for row in document.rows:
        if row.start_position <= quote.start <= row.end_position:
            return row.index
    return None


def write_consensus_json(records: list[dict], output_file: Path) -> None:
    """Write consensus records to a JSON file, creating parent folders as needed."""
    resolve_output_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def log_double_coding_conflicts(
    documents: Mapping[str, Document],
    quotes_by_doc: Mapping[str, Sequence[Quote]],
    *,
    context_message: str,
    step: str | None = None,
) -> None:
    """Flag any quote whose co-occurring tags include a pair that both
    disallow double coding (see ``models.tags_disallow_co_occurrence()``).

    ``context_message`` is interpolated into the logged message (e.g. "in
    the final consensus (superconsensus)" or "after manual correction").
    ``step`` is an optional process-log step override; omitted/None logs
    under the active log's own step.
    """
    for doc_name, quotes in quotes_by_doc.items():
        document = documents.get(doc_name)
        for quote in quotes:
            row = row_for_quote(document, quote)
            for tag_a, tag_b in combinations(quote.tags, 2):
                if tags_disallow_co_occurrence(tag_a, tag_b):
                    process_log.current().log_double_code_conflict(
                        severity="Warning",
                        source_name=doc_name,
                        row=row,
                        span_start=quote.start,
                        span_end=quote.end,
                        quote_text=quote.text,
                        tag_a=tag_a.name,
                        tag_b=tag_b.name,
                        message=(
                            f"Tags '{tag_a.name}' and '{tag_b.name}' both disallow double "
                            f"coding but co-occur {context_message} in '{doc_name}'."
                        ),
                        step=step,
                    )


def _row_char_tags_by_iteration(
    row: Row,
    iteration_quotes: Sequence[Sequence[Quote]],
    step: str | None = None,
    document: Document | None = None,
) -> list[list[set[Tag]]]:
    char_tags_by_iteration = [
        [set() for _ in range(len(row.text))] for _ in iteration_quotes
    ]

    for iteration_index, quotes in enumerate(iteration_quotes):
        for quote in quotes:
            if not quote.in_row(row):
                continue

            _resolve_quote_position(quote, row, step=step, document=document)
            if quote.start is None or quote.end is None:
                continue

            start = max(0, quote.start - row.start_position)
            end = min(len(row.text), quote.end - row.start_position)
            if start >= end:
                continue

            for char_index in range(start, end):
                char_tags_by_iteration[iteration_index][char_index].update(quote.tags)

    return char_tags_by_iteration


def _resolve_quote_position(
    quote: Quote, row: Row, step: str | None = None, document: Document | None = None
) -> None:
    if quote.start is not None and quote.end is not None:
        return
    quote.find_position(row.text, row.start_position, step=step, document=document)


def _vote_character_tags(
    char_tags_by_iteration: list[list[set[Tag]]],
    iteration_count: int,
    threshold: float,
) -> list[set[Tag]]:
    if not char_tags_by_iteration:
        return []

    text_length = len(char_tags_by_iteration[0])
    consensus_char_tags: list[set[Tag]] = []

    for char_index in range(text_length):
        counts: Counter[Tag] = Counter()
        for iteration_tags in char_tags_by_iteration:
            counts.update(iteration_tags[char_index])

        consensus_tags = {
            tag for tag, count in counts.items() if count / iteration_count >= threshold
        }
        consensus_char_tags.append(consensus_tags)

    return consensus_char_tags


def _compress_character_tags_to_quotes(
    row: Row,
    consensus_char_tags: list[set[Tag]],
    source_iterations: int,
) -> list[ConsensusQuote]:
    quotes: list[ConsensusQuote] = []
    span_start: int | None = None
    active_tags: set[Tag] = set()

    for index, tags in enumerate(consensus_char_tags + [set()]):
        if tags == active_tags:
            continue

        if active_tags and span_start is not None:
            quotes.append(
                _make_consensus_quote(
                    row,
                    span_start,
                    index,
                    active_tags,
                    source_iterations,
                )
            )

        span_start = index if tags else None
        active_tags = set(tags)

    return quotes


def _make_consensus_quote(
    row: Row,
    start_index: int,
    end_index: int,
    tags: set[Tag],
    source_iterations: int,
) -> ConsensusQuote:
    absolute_start = row.start_position + start_index
    absolute_end = row.start_position + end_index
    return ConsensusQuote(
        text=row.text[start_index:end_index],
        tags=sorted(tags, key=lambda tag: tag.name),
        start=absolute_start,
        end=absolute_end,
        source_iterations=source_iterations,
    )
