"""Resolves config.RUN_TYPE into concrete document-name sets and paths"""

from pathlib import Path

from src.config.project import RUN_TYPE_CONFIGS, TXT_FILES
from src.core import process_log
from src.core.models import Document, Quote
from src.io.io_utils import get_matching_files
from src.io.loaders import DocumentLoader


def folder(run_type: str) -> Path:
    """The subfolder under TXT_FILES for run_type (does not validate
    run_type -- callers that need that should go through processing_dirs()/
    report_scopes(), which check against RUN_TYPE_CONFIGS)."""
    return TXT_FILES / run_type


def training_folder() -> Path:
    return folder("training")


def validation_folder() -> Path:
    return folder("validation")


def full_folder() -> Path:
    return folder("full")


def full_without_validation_folder() -> Path:
    return folder("full_without_validation")


def test_folder() -> Path:
    return folder("test")


def _stems(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(path.stem for path in folder.glob("*.txt"))


def documents(run_type: str) -> list[str]:
    """Every .txt file already materialized under folder(run_type)."""
    return _stems(folder(run_type))


def training_documents() -> list[str]:
    """Every .txt file already materialized under training_folder()."""
    return documents("training")


def validation_documents() -> list[str]:
    """Every .txt file already materialized under validation_folder()."""
    return documents("validation")


def full_documents() -> list[str]:
    """Every .txt file already materialized under full_folder()."""
    return documents("full")


def full_without_validation_documents() -> list[str]:
    """Every .txt file already materialized under full_without_validation_folder()."""
    return documents("full_without_validation")


def test_documents() -> list[str]:
    """Every .txt file already materialized under test_folder(). Unlike the
    other four folders, nothing writes these -- the user places .txt files
    here manually for a small, hand-picked test run."""
    return documents("test")


def processing_dirs(run_type: str) -> list[Path]:
    """The directories find_source_documents()/run_ai_coding() glob *.txt
    files from (recursively) for run_type, in precedence order.

    "training"/"validation"/"full_without_validation"/"test" each point at
    their own single subfolder. "full" covers training/, validation/ and
    full/: its report_scopes() need training's and validation's documents,
    and full/ adds the untagged whole documents. full_without_validation/ is
    left out, because it only holds trimmed copies of documents that full/
    already has in whole.

    The same document name can exist in several of these folders (a
    document-driven training/ file is also in full/), so
    find_source_documents() keeps only the first one in this order.
    """
    if run_type == "full":
        return [training_folder(), validation_folder(), full_folder()]
    if run_type not in RUN_TYPE_CONFIGS:
        raise ValueError(f"Unknown RUN_TYPE: {run_type!r}")
    return [folder(run_type)]


def report_scopes(run_type: str) -> dict[str, list[str]]:
    """label -> document subset, for each report run_type must produce.

    "training"/"validation"/"test" report on exactly what they process.
    "full_without_validation" processes a bigger pool than it reports on --
    every non-excluded, non-validation-touched whole document (training +
    untagged), via full_without_validation_folder() -- but its report still
    covers the training subset only, per its own definition ("run on the
    full dataset except validation; validate only the training dataset").
    "full" processes everything (including untagged documents, via
    full_folder(), alongside validation's cluster samples) and produces
    three reports: training, validation, and their combination (training
    union validation -- not literally everything in full_folder(), see the
    project history: "combination of both" means training+validation only).
    """
    if run_type == "full":
        training = training_documents()
        validation = validation_documents()
        combined = sorted(set(training) | set(validation))
        return {
            "training": training,
            "validation": validation,
            "combined": combined,
        }
    if run_type == "full_without_validation":
        return {run_type: training_documents()}
    if run_type not in RUN_TYPE_CONFIGS:
        raise ValueError(f"Unknown RUN_TYPE: {run_type!r}")
    return {run_type: documents(run_type)}


# The report_scopes() label whose output is auto-opened in the browser and
# used for cross-prefix comparison (compare_coders()), for run types that
# produce more than one report. Single-report run types always use their
# one (and only) scope for both.
PRIMARY_SCOPE_LABEL = "combined"


def _first_file_per_stem(directories: list[Path]) -> list[Path]:
    files_by_stem: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(get_matching_files(directory, "*.txt", recursive=True)):
            files_by_stem.setdefault(path.stem, path)
    return list(files_by_stem.values())


def find_source_documents(run_type: str) -> list[Path]:
    """Every .txt file to process for run_type -- one file per document name.

    Files are collected from processing_dirs(run_type) in order, and when a
    document name appears in more than one folder only its first file is
    kept, so a document is coded and loaded exactly once.
    """
    return _first_file_per_stem(processing_dirs(run_type))


def find_canonical_documents() -> list[Path]:
    """One .txt file per document name across every TXT_FILES subfolder,
    independent of RUN_TYPE.

    A document name can exist in several subfolders with different content
    (full_without_validation/ drops validation rows, so character offsets
    shift). The human ATLAS offsets refer to the whole document, so full/
    comes first, then test/ (manually placed whole documents), then the
    remaining subfolders.
    """
    return _first_file_per_stem(
        [
            full_folder(),
            test_folder(),
            training_folder(),
            validation_folder(),
            full_without_validation_folder(),
        ]
    )


def load_source_documents(document_paths: list[Path], *, context: str) -> dict[str, Document]:
    """Load each of document_paths into a Document, keyed by its stem.

    A path that no longer exists on disk is skipped and logged via
    log_missing_source_document() instead of raising -- context identifies
    which entry point/step logged the gap (e.g. "run_main_document_loading"
    vs. "post_process_document_loading").
    """
    print("Loading documents...")
    documents: dict[str, Document] = {}
    for doc_path in document_paths:
        doc_name = doc_path.stem
        if doc_path.exists():
            documents[doc_name] = DocumentLoader(doc_path).load()
            print(f"  Loaded: {doc_name}")
        else:
            message = f"  Warning: Could not find document text for {doc_path.name}"
            print(message)
            process_log.current().log_missing_source_document(
                severity="Warning",
                source_name=doc_path.name,
                context=context,
                message=message,
            )
    return documents


def align_quote_documents(
    quotes_by_doc: dict[str, list[Quote]],
    source_doc_names: list[str],
) -> dict[str, list[Quote]]:
    """Re-key quotes_by_doc to exactly source_doc_names, filling in an empty
    list for any name with no quotes of its own."""
    return {doc_name: quotes_by_doc.get(doc_name, []) for doc_name in source_doc_names}
