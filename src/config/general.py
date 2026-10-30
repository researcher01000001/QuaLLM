"""General run configuration, read docs/configuration.md for further information"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from src.config.coding import CodingConfiguration

# Load environment variables from a .env file if it exists
load_dotenv()


# ==========================================
# Main run configuration
# ==========================================

# Prefix on wicht the data is saved
PREFIX = "model_selection_v2_0"
# Set to True to run the AI coding process, False to skip it
RUN_AI_CODING = False
# Caps how many paragraphs of one document are sent for AI coding
AI_CODING_MAX_PARAGRAPHS = None
# Paragraphs shorter than this many characters (after stripping) are not sent for AI coding
AI_CODING_MIN_PARAGRAPH_CHARS = 10
# Configuration for the AI coding
CODING_CONFIGURATIONS = [
    # CodingConfiguration(
    #     short_name="gemini-3.8-flash_260902",
    #     iterations=3,
    #     model_name="google/gemini-3.8-flash",
    #     seed=None,
    #     system_prompt_name="SYSTEM_PROMPT_V1",
    #     user_prompt_name="USER_PROMPT_BASE",
    #     max_threads=10
    # ),
    CodingConfiguration(
        short_name="mistral-small-4-2603_260316",
        iterations=1,
        model_name="mistralai/mistral-small-2603",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=1
    ),
    CodingConfiguration(
        short_name="mistral-large-4-0_261006",
        iterations=1,
        model_name="mistralai/mistral-large-4-0",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=1
    ),
    CodingConfiguration(
        short_name="qwen3.8-27b_260814",
        iterations=1,
        model_name="qwen/qwen3.8-27b",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="z-ai-glm-5.3_260818",
        iterations=1,
        model_name="z-ai/glm-5.3",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="gpt-6-sol_260922",
        iterations=1,
        model_name="openai/gpt-6-sol",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="gpt-6-luna_260922",
        iterations=1,
        model_name="openai/gpt-6-luna",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="deepseek-v4-pro-0813_260812",
        iterations=1,
        model_name="deepseek/deepseek-v4-pro-0813",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="gemini-3.8-flash_260902",
        iterations=1,
        model_name="google/gemini-3.8-flash",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="gemma-4-31b-it_260402",
        iterations=1,
        model_name="google/gemma-4-31b-it",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="claude-sonnet-5.5_260928",
        iterations=1,
        model_name="anthropic/claude-sonnet-5.5",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="llama-3.3-70b-instruct_241206",
        iterations=1,
        model_name="meta-llama/llama-3.3-70b-instruct",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    ),
    CodingConfiguration(
        short_name="llama-3.2-3b-instruct_240925",
        iterations=1,
        model_name="meta-llama/llama-3.2-3b-instruct",
        seed=None,
        system_prompt_name="SYSTEM_PROMPT_BASE",
        user_prompt_name="USER_PROMPT_BASE",
        max_threads=10
    )
]
# Type of the run to use - depending on the type, it uses the documents
# of the corresponding run types subfolder in data/sources/atlas_import/
RUN_TYPE: Literal["training", "validation", "full_without_validation", "full", "test"] = "training"
# Give older prefixes the current results should be compared to
COMPARISON_PREFIXES: list[str] = []
# Your Openrouter API key to use
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Flag that indicates that the undefined tags are shown and not just dropped.
UNDEFINED_TAGS_ENABLED = True


# ==========================================
# Additional configurations
# ==========================================

## Cost estimation
# Flag indicating if Cost Estimation should be run prior to the AI coding run
COST_ESTIMATION_ENABLED = False
# Threshold that indicates the divergence, of when cost estimation should recalculate
COST_ESTIMATION_RECALC_THRESHOLD_FRACTION: float = 0.1
# How long a cached OpenRouter pricing response is reused
COST_ESTIMATION_PRICING_CACHE_TTL_HOURS = 24
# Flag indicating if the Metadata are written Flag indicating if
WRITE_METADATA = True

## Superconsensus
# Name, under which the superconsensus appears in the results
SUPERCONSENSUS_NAME = "superconsensus"
# Threshold that has to be reached for one tag during superconsensus consensus
# building so that the tag is applied to this character.
SUPERCONSENSUS_THRESHOLD = 0.5
# Which consensus.json source the HTML report is built from
HTML_CONSENSUS_CONFIGURATIONS: list[str] = []

# Short forms for the different metrics
SCORE_COLUMNS: list[tuple[str, str]] = [
    ("score_lev", "List-Tag Levenshtein Score"),
    ("score_jaccard", "Set-Tag Jaccard Score"),
    ("score_f1", "Set-Tag F1 Score"),
    ("score_agreement_original", "Character-Level Agreement (Original)"),
    ("score_char_f1_original", "Character-Level F1 (Original)"),
    ("score_agreement_pruned", "Character-Level Agreement (Pruned)"),
    ("score_char_f1_pruned", "Character-Level F1 (Pruned)"),
]


# ==========================================
# Path configurations
# ==========================================
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
DATA_SOURCES_DIR = DATA_DIR / "sources"
DATA_RESULTS_DIR = DATA_DIR / "results"

AI_CODING_SUBDIR = "1-ai_coding"
RESULTS_SUBDIR = "2-results"
POST_PROCESS_SUBDIR = "3-post_process"

PROCESS_LOG_SUBDIR = "process_log"

def run_results_dir(prefix: str, run_type: str) -> Path:
    """The folder holding everything one prefix + run type produces."""
    return DATA_RESULTS_DIR / prefix / run_type
RUN_RESULTS_DIR = run_results_dir(PREFIX, RUN_TYPE)

RESULTS_DIR = RUN_RESULTS_DIR / RESULTS_SUBDIR
OUTPUT_HTML_FILE = RESULTS_DIR / f"comparison_interface_{PREFIX}.html"

OUTPUT_XLSX_FILE_INCLUDING_UNDEFINED_TAGS = (
    RESULTS_DIR / f"configuration_matrix_main_{PREFIX}_including_undefined_tags.xlsx"
)
OUTPUT_XLSX_FILE_EXCLUDING_UNDEFINED_TAGS = (
    RESULTS_DIR / f"configuration_matrix_main_{PREFIX}_excluding_undefined_tags.xlsx"
)

CONSENSUS_FILENAME = "consensus.json"

PROCESS_LOG_DIR = RUN_RESULTS_DIR / PROCESS_LOG_SUBDIR
PROCESS_LOG_FILE = PROCESS_LOG_DIR / f"process_log_{PREFIX}.xlsx"

COST_ESTIMATION_PRICING_CACHE_FILE = DATA_DIR / "cache" / "openrouter_pricing.json"
COST_ESTIMATION_CALIBRATION_FILE = DATA_RESULTS_DIR / "cost_calibration.json"

AI_CODING_FILES = RUN_RESULTS_DIR / AI_CODING_SUBDIR

def scoped_output_html(scope_label: str) -> Path:
    return RESULTS_DIR / f"comparison_interface_{PREFIX}_{scope_label}.html"


def scoped_output_xlsx_including(scope_label: str) -> Path:
    return RESULTS_DIR / f"configuration_matrix_main_{PREFIX}_{scope_label}_including_undefined_tags.xlsx"


def scoped_output_xlsx_excluding(scope_label: str) -> Path:
    return RESULTS_DIR / f"configuration_matrix_main_{PREFIX}_{scope_label}_excluding_undefined_tags.xlsx"
