"""Configuration for the Post Processing Pipeline"""

from src.config.general import (
    CONSENSUS_FILENAME,
    POST_PROCESS_SUBDIR,
    PREFIX,
    RUN_RESULTS_DIR,
)


# ==========================================
# Main post processing configuration
# ==========================================

# IMPORTANT: Which post-processing step run_post_processing.py executes:
# 1 = creates a review-file to classify the undefined tags.
#     If activated it also loads existing manually coded tags from a data file
# 2 = creates a review-file for the full dataset of the current run configuration
#     Uses the informations of the step 1 human coding
# 3 = Export of the data to the initial input file
POST_PROCESS_STEP = 1

# List of tags to be manually reviewed
POST_PROCESS_TAGS: list[str] = ["other"]

# Save Step 1 review decisions to the database for reuse in later runs.
# Set False for test runs whose decisions should not be saved.
# With an existing review page, True imports saved decisions without regenerating it.
POST_PROCESS_STEP1_INTEGRATE_INTO_DATABASE = True


# ==========================================
# Additional configurations
# ==========================================

# Whether to include undefined tags in post processing
POST_PROCESS_INCLUDE_UNDEFINED_TAGS = True

## Database
# Whether the Database of existing post process runs should be loaded
POST_PROCESS_LOAD_DATABASE = True
# Flag indicating if all prefixes should be used with the database
POST_PROCESS_DATABASE_USE_ALL_PREFIXES = True
# Threshold, when a span is considered a match to the span from the database depending on their overlap
POST_PROCESS_SPAN_OVERLAP_THRESHOLD = 0.9
# Creats a new database based on the current results and drops the old one
POST_PROCESS_STEP1_REBUILD_DATABASE = False

# If Step 3 should use the existing JSON instead of deriving it again
POST_PROCESS_STEP3_USE_EXISTING_JSON = False

# Name of the new column in the source file that the results are appended to
POST_PROCESS_STEP3_TAGS_COLUMN = "ordered_tags"


# ==========================================
# Path configurations
# ==========================================

POST_PROCESS_DIR = RUN_RESULTS_DIR / POST_PROCESS_SUBDIR

POST_PROCESS_STEP1_DIR = POST_PROCESS_DIR / "01_undefined_tags"
POST_PROCESS_STEP1_CONSENSUS_UNMODIFIED_FILE = POST_PROCESS_STEP1_DIR / "consensus_unmodified.json"
POST_PROCESS_STEP1_CONSENSUS_WITH_DEFINED_TAGS_FILE = POST_PROCESS_STEP1_DIR / "consensus_with_defined_tags.json"
POST_PROCESS_STEP1_REVIEW_HTML = POST_PROCESS_STEP1_DIR / f"review_interface_undefined_tags_{PREFIX}.html"
POST_PROCESS_DATABASE_FILE = POST_PROCESS_STEP1_DIR / f"database_{PREFIX}.xlsx"

POST_PROCESS_STEP2_DIR = POST_PROCESS_DIR / "02_manual_supervision"

POST_PROCESS_STEP2_CONSENSUS_FILE = POST_PROCESS_STEP2_DIR / CONSENSUS_FILENAME
POST_PROCESS_STEP2_CONSENSUS_REVIEWED_FILE = POST_PROCESS_STEP2_DIR / "consensus_reviewed.json"
POST_PROCESS_STEP2_REPORT_XLSX_INCLUDING_UNDEFINED_TAGS = (
    POST_PROCESS_STEP2_DIR / f"configuration_matrix_pp2_{PREFIX}_including_undefined_tags.xlsx"
)
POST_PROCESS_STEP2_REPORT_XLSX_EXCLUDING_UNDEFINED_TAGS = (
    POST_PROCESS_STEP2_DIR / f"configuration_matrix_pp2_{PREFIX}_excluding_undefined_tags.xlsx"
)
POST_PROCESS_STEP2_REPORT_HTML = POST_PROCESS_STEP2_DIR / f"comparison_interface_pp2_{PREFIX}.html"
POST_PROCESS_STEP2_REVIEW_HTML = POST_PROCESS_STEP2_DIR / f"review_interface_general_{PREFIX}.html"

POST_PROCESS_STEP3_DIR = POST_PROCESS_DIR / "03_export"
POST_PROCESS_STEP3_REVIEW_JSON = POST_PROCESS_STEP3_DIR / f"manual_review_{PREFIX}.json"
POST_PROCESS_STEP3_ORDERED_TAGS_JSON = POST_PROCESS_STEP3_DIR / f"ordered_tags_{PREFIX}.json"
POST_PROCESS_STEP3_OUTPUT_XLSX = POST_PROCESS_STEP3_DIR / f"output_{PREFIX}.xlsx"
