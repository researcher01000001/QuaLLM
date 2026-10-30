"""Dataset-specific configuration"""

from dataclasses import dataclass
from typing import Literal

from src.config.general import DATA_SOURCES_DIR


# ==========================================
# Source files
# ==========================================
ADDITIONAL_SOURCES = DATA_SOURCES_DIR / "additional_sources"
# Atlas export of the codebook to give the same colors to the codes
CODES_FILE = ADDITIONAL_SOURCES / "HeadCT_ThinkAlouds_codebook.xlsx"
# File with the coding scheme itself
DESCRIPTION_FILE = ADDITIONAL_SOURCES / "description_coding_scheme.xlsx"
# Source file with the experiment data
META_OUTPUT_FILE = {
    "filename": ADDITIONAL_SOURCES / "output.xlsx",
    "sheetname": "Sheet",
}
# File with the Atlas export of the codes
HUMAN_CODING_FILES = DATA_SOURCES_DIR / "human_coding" / "HeadCT_ThinkAlouds_codes.xlsx"
# Folder that contains the folders with the files of the different clusters
TXT_FILES = DATA_SOURCES_DIR / "atlas_import"

# ==========================================
# Data configuration
# ==========================================

## Document configuration
# Name tamplate for a document drive approache
DOCUMENT_NAME_TEMPLATE = "{think_aloud_id}_{condition}_cycle_{cycle_id}"
# Column name of the general cluster
GENERAL_CLUSTER_COLUMN = "condition"
# Columns that uniquely group META_OUTPUT_FILE rows into documents
DOCUMENT_KEY_COLUMNS: list[str] = ["think_aloud_id", GENERAL_CLUSTER_COLUMN]
# Column that defines the order within one document
DOCUMENT_ORDER_COLUMN = "position"
# Label for the order column
DOCUMENT_ORDER_COLUMN_LABEL = "Paragraph"

# Column with the text of each paragraph
TEXT_COLUMN = "adjusted_transcript"

# Column name of the dataset
DATASET_COLUMN = "dataset"
# Column indicating the order of the random dataset
RANDOM_DATASET_ORDER_COLUMN = "dataset_order"

# Labels of the META_OUTPUT_FILE columns for the HTML rendering
PARA_META_COLUMNS: list[tuple[str, str]] = [
    ("patient_information", "Patient info"),
    ("truth", "Truth"),
    ("diagnosis", "Diagnosis"),
    ("performance", "Performance"),
    ("self_confidence", "Self-confidence"),
    ("ai_report_usefulness", "AI report usefulness"),
    ("condition", "Condition"),
    ("difficulty", "Difficulty"),
    ("type", "Type"),
    ("patient_uid", "Patient UID"),
    ("cycle_type", "Cycle type"),
    ("group", "Group"),
]


# ==========================================
# Run configuration
# ==========================================

# Values to exclude from the DATASET_COLUMN
EXCLUDED_VALUES: list[str] = ["excluded"]

# Defines the selection mode for the training dataset
TRAINING_SELECTION_MODE: Literal["document_driven", "random"] = "document_driven"
# Defines the selection mode for the validation dataset
VALIDATION_SELECTION_MODE: Literal["document_driven", "random"] = "random"

# ==========================================
# Additional configurations
# ==========================================

@dataclass(frozen=True, kw_only=True)
class DocumentNamingConfig:
    """Defines the naming and ordering logic for different selection type"""

    name_template: str | None
    order_column: str
    order_column_label: str | None
    text_column: str
    group_column: str | None

# Defines the naming scheme based on the different selection types
NAMING_BY_MODE: dict[str, DocumentNamingConfig] = {
    "document_driven": DocumentNamingConfig(
        name_template=DOCUMENT_NAME_TEMPLATE,
        order_column=DOCUMENT_ORDER_COLUMN,
        order_column_label=DOCUMENT_ORDER_COLUMN_LABEL,
        text_column=TEXT_COLUMN,
        group_column="think_aloud_id",
    ),
    "random": DocumentNamingConfig(
        name_template="{type}_{cluster}",
        order_column=RANDOM_DATASET_ORDER_COLUMN,
        order_column_label=None,
        text_column=TEXT_COLUMN,
        group_column=None,
    ),
    "manual": DocumentNamingConfig(
        name_template=None,
        order_column=RANDOM_DATASET_ORDER_COLUMN,
        order_column_label=None,
        text_column=TEXT_COLUMN,
        group_column=None,
    ),
}


# Configuration for random selection mode
TRAINING_DOCUMENT_CLUSTER_COLUMN: list[str] = ["think_aloud_id", GENERAL_CLUSTER_COLUMN]
TRAINING_SUB_CLUSTER_COLUMN: list[str] = []
VALIDATION_DOCUMENT_CLUSTER_COLUMN: list[str] = [GENERAL_CLUSTER_COLUMN]
VALIDATION_SUB_CLUSTER_COLUMN: list[str] = ["type", "performance"]


# ==========================================
# Predefined configuration
# ==========================================

@dataclass(frozen=True, kw_only=True)
class RunTypeConfig:
    """Defines run type informations"""

    selection_mode: Literal["document_driven", "random","manual"]
    document_cluster_column: list[str]
    sub_cluster_column: list[str]


RUN_TYPE_CONFIGS: dict[str, RunTypeConfig] = {
    "training": RunTypeConfig(
        selection_mode=TRAINING_SELECTION_MODE,
        document_cluster_column=TRAINING_DOCUMENT_CLUSTER_COLUMN,
        sub_cluster_column=TRAINING_SUB_CLUSTER_COLUMN,
    ),
    "validation": RunTypeConfig(
        selection_mode=VALIDATION_SELECTION_MODE,
        document_cluster_column=VALIDATION_DOCUMENT_CLUSTER_COLUMN,
        sub_cluster_column=VALIDATION_SUB_CLUSTER_COLUMN,
    ),
    "full": RunTypeConfig(
        selection_mode=TRAINING_SELECTION_MODE,
        document_cluster_column=TRAINING_DOCUMENT_CLUSTER_COLUMN,
        sub_cluster_column=TRAINING_SUB_CLUSTER_COLUMN,
    ),
    "full_without_validation": RunTypeConfig(
        selection_mode=TRAINING_SELECTION_MODE,
        document_cluster_column=TRAINING_DOCUMENT_CLUSTER_COLUMN,
        sub_cluster_column=TRAINING_SUB_CLUSTER_COLUMN,
    ),
    "test": RunTypeConfig(
        selection_mode="manual",
        document_cluster_column=["think_aloud_id", GENERAL_CLUSTER_COLUMN],
        sub_cluster_column=[],
    ),
}
