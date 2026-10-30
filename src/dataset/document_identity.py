"""Parses a document's identity/condition directly out of its .txt file name"""

from typing import Any


def parse_document_name(doc_name: str) -> dict[str, Any] | None:
    """Every DOCUMENT_NAME_TEMPLATE column's value parsed according to the
    logic in DOCUMENT_NAME_TEMPLATE (ids as ints, condition as str), for a
    document-driven document ("45_no advice_cycle_63") -- this shape alone
    needs no spreadsheet access.

    Also recognizes a random-mode training/validation sample file (named
    "{training|validation}_{condition}" -- see scripts/atlas_preperation.py's
    module docstring, e.g. "validation_advice", "training_no advice" --
    the type prefix makes such a file traceable to its split from the
    filename alone) or, for backward compatibility, one named by its bare
    document-cluster value alone (e.g. "advice", "no advice"): such a file
    has no single think_aloud_id/cycle_id of its own, since it's a sample
    spanning many different participants/cycles, so only "condition" is
    returned for it. Recognizing either of these shapes checks the
    remaining text against document_content.known_conditions() -- the
    dataset's own real conditions, read from META_OUTPUT_FILE, rather than a
    hardcoded guess -- so a name is only accepted as a condition-named file
    if it actually names one. Imported locally to avoid a
    document_identity <-> document_content import cycle (document_content.py
    imports several constants from config/project.py, which document_identity
    is independent of, but keeping the import local here keeps this module
    safe to import from anywhere regardless).

    Returns None if doc_name matches none of these shapes.
    """
    parts = doc_name.split("_")
    if len(parts) == 4:
        try:
            return {
                "think_aloud_id": int(parts[0]),
                "condition": parts[1],
                "cycle_id": int(parts[3]),
            }
        except ValueError:
            return None

    from src.dataset import document_content

    conditions = document_content.known_conditions()

    if doc_name in conditions:
        return {"condition": doc_name}

    for dataset_type in ("training", "validation"):
        prefix = f"{dataset_type}_"
        if doc_name.startswith(prefix) and doc_name[len(prefix) :] in conditions:
            return {"condition": doc_name[len(prefix) :]}

    return None


def get_document_key(doc_name: str) -> dict[str, Any] | None:
    """The DOCUMENT_KEY_COLUMNS values identifying doc_name, or None if
    doc_name doesn't match DOCUMENT_NAME_TEMPLATE's shape, or if doc_name is
    a random-mode sample file (no single think_aloud_id -- it spans many)."""
    parsed = parse_document_name(doc_name)
    if parsed is None or "think_aloud_id" not in parsed:
        return None
    return {"think_aloud_id": parsed["think_aloud_id"], "condition": parsed["condition"]}


def condition_for(doc_name: str) -> str | None:
    """doc_name's raw document condition (e.g. "no advice") -- the same
    value as the document's own config_project.GENERAL_CLUSTER_COLUMN
    column, recovered from the name instead of the spreadsheet. Or None if
    doc_name doesn't match DOCUMENT_NAME_TEMPLATE's shape."""
    parsed = parse_document_name(doc_name)
    return parsed["condition"] if parsed else None


def resolve_document_identity(doc_name: str) -> dict[str, Any] | None:
    """get_document_key() plus condition_for(), combined into the one dict
    shape review_page.build_document_identity() needs. None if doc_name
    doesn't match DOCUMENT_NAME_TEMPLATE's shape (see get_document_key())."""
    identity = get_document_key(doc_name)
    if identity is None:
        return None
    return {**identity, "condition": condition_for(doc_name)}


def condition_match(doc_name: str) -> list[str]:
    """Coding-scheme condition values (Tag.condition) valid for doc_name --
    this is the actual filter that decides which coding-scheme entries
    apply to a document (see src.core.models.tags_for_condition()), driven by
    doc_name's condition_for() value (config_project.GENERAL_CLUSTER_COLUMN).

    The coding scheme's condition vocabulary ("general"/"supported"/"human")
    doesn't overlap with the document condition vocabulary ("advice"/"advice
    with annotation"/"no advice") by plain equality -- "supported" means a
    tag only applies when the participant received AI advice, "human" means
    it only applies to unassisted (no advice) reasoning, and "general"
    always applies. This function is the one place that translation
    happens; never compare the two vocabularies directly elsewhere.
    """
    match condition_for(doc_name):
        case "advice" | "advice with annotation":
            return ["general", "supported"]
        case "no advice":
            return ["general", "human"]
        case _:
            return []
