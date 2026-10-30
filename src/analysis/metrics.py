from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.core.models import CodedRow, Document, Quote, Row


@dataclass(frozen=True)
class RowMetricResult:
    document_name: str
    configuration_name: str
    iteration: int | None
    is_consensus: bool
    row: int
    display_value: str
    lev_distance: float
    jaccard: float
    f1: float
    original_char_f1: float
    original_char_agreement: float
    pruned_char_f1: float
    pruned_char_agreement: float


@dataclass(frozen=True)
class DocumentMetricResult:
    document_name: str
    configuration_name: str
    iteration: int | None
    is_consensus: bool
    avg_lev_distance: float
    avg_jaccard: float
    avg_f1: float
    avg_original_char_f1: float
    avg_original_char_agreement: float
    avg_pruned_char_f1: float
    avg_pruned_char_agreement: float
    row_metrics: list[RowMetricResult]


@dataclass(frozen=True)
class CodedDocumentMetricResult:
    document: Document
    metrics: DocumentMetricResult


def calculate_document_metrics(
    document: Document,
    human_quotes: Sequence[Quote],
    ai_quotes: Sequence[Quote],
    configuration_name: str,
    iteration: int | None,
    is_consensus: bool,
    drop_undefined_tags: bool = False,
    step: str | None = None,
) -> CodedDocumentMetricResult:
    """Compare human quotes with one AI run/consensus for one document.

    System tags (Tag.system) are always excluded from every
    calculated metric (see CodedRow.compare_coders()). Tags with
    additional_tag=True are additionally excluded when drop_undefined_tags is
    True (see src.config.general.UNDEFINED_TAGS_ENABLED).

    ``step`` is an optional process-log step override, forwarded to
    ``CodedRow.compare_coders()`` for any position-not-found logging
    an unresolved ``AIQuote`` triggers -- e.g. a specific configuration's
    namespaced sub-step. Omitted/None falls back to the active log's own
    step.
    """
    coded_document = _copy_as_plain_document(document)
    coded_document.append_quotes(list(human_quotes), list(ai_quotes))

    row_results: list[RowMetricResult] = []
    for row in coded_document.rows:
        if not isinstance(row, CodedRow):
            continue

        lev_dist, jaccard, f1, char_f1_result = row.compare_coders(
            True, drop_undefined_tags, step=step, document=coded_document
        )
        row_results.append(
            RowMetricResult(
                document_name=document.source_name,
                configuration_name=configuration_name,
                iteration=iteration,
                is_consensus=is_consensus,
                row=row.index,
                display_value=document.display_value(row.index),
                lev_distance=lev_dist,
                jaccard=jaccard,
                f1=f1,
                original_char_f1=char_f1_result["original"]["char_f1"],
                original_char_agreement=char_f1_result["original"]["agreement"],
                pruned_char_f1=char_f1_result["pruned"]["char_f1"],
                pruned_char_agreement=char_f1_result["pruned"]["agreement"],
            )
        )

    metrics = DocumentMetricResult(
        document_name=document.source_name,
        configuration_name=configuration_name,
        iteration=iteration,
        is_consensus=is_consensus,
        avg_lev_distance=_average([p.lev_distance for p in row_results]),
        avg_jaccard=_average([p.jaccard for p in row_results]),
        avg_f1=_average([p.f1 for p in row_results]),
        avg_original_char_f1=_average([p.original_char_f1 for p in row_results]),
        avg_original_char_agreement=_average(
            [p.original_char_agreement for p in row_results]
        ),
        avg_pruned_char_f1=_average([p.pruned_char_f1 for p in row_results]),
        avg_pruned_char_agreement=_average(
            [p.pruned_char_agreement for p in row_results]
        ),
        row_metrics=row_results,
    )
    return CodedDocumentMetricResult(document=coded_document, metrics=metrics)


def calculate_metrics_for_quote_sets(
    documents: dict[str, Document],
    human_quotes: Mapping[str, Sequence[Quote]],
    ai_quotes: Mapping[str, Sequence[Quote]],
    configuration_name: str,
    iteration: int | None,
    is_consensus: bool,
    drop_undefined_tags: bool = False,
    step: str | None = None,
) -> list[CodedDocumentMetricResult]:
    """Compare all source documents against one AI quote mapping.

    ``step`` is an optional process-log step override forwarded to
    ``calculate_document_metrics()`` -- see there.
    """
    results: list[CodedDocumentMetricResult] = []
    for doc_name, document in documents.items():
        results.append(
            calculate_document_metrics(
                document=document,
                human_quotes=human_quotes.get(doc_name, []),
                ai_quotes=ai_quotes.get(doc_name, []),
                configuration_name=configuration_name,
                iteration=iteration,
                is_consensus=is_consensus,
                drop_undefined_tags=drop_undefined_tags,
                step=step,
            )
        )
    return results


def _copy_as_plain_document(document: Document) -> Document:
    return Document(
        source_name=document.source_name,
        rows=[
            Row(
                index=row.index,
                text=row.text,
                start_position=row.start_position,
                end_position=row.end_position,
            )
            for row in document.rows
        ],
        display_column=document.display_column,
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
