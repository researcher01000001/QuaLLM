"""Writes config_project.DATASET_COLUMN + config_project.RANDOM_DATASET_ORDER_COLUMN
back into config_project.META_OUTPUT_FILE (output.xlsx). Selection is at ROW
(paragraph/case) granularity, not whole-document granularity -- a
document's 50 rows can carry different tags. DATASET_COLUMN's grammar:

    "excluded"              -- technically unusable, never part of any split
    "{cluster}"              -- untagged: part of the full dataset only
    "training_{cluster}"    -- selected for training
    "validation_{cluster}"  -- selected for validation

parsed by src.dataset.selection.parse_dataset_value(). RANDOM_DATASET_ORDER_COLUMN holds
the row's shuffled position within its (type, cluster) group as a plain
int (blank for "excluded"/untagged rows) -- kept out of DATASET_COLUMN
itself so that string stays trivial to parse back (just a prefix check, no
trailing-number ambiguity).

A row's identity is its DOCUMENT_KEY_COLUMNS values plus its DOCUMENT_ORDER_COLUMN
value (e.g. (cycle_id, position) for this study) -- see _row_keys().

This script never writes "excluded" itself -- exclusion is entirely
manual: whatever rows are already tagged "excluded" in META_OUTPUT_FILE's
DATASET_COLUMN (maintained by hand) are read via _existing_excluded_rows()
and treated as off-limits for the random draw below, but this script never
adds, removes, or changes an "excluded" tag.

Section A (kept verbatim): identifies participants to exclude entirely
(incomplete data / technical recording problems) and a small hand-picked
set of extreme/near-mean no-advice performers historically reserved for
manual coding -- unrelated to the training/validation split, still
print-only. Its id_to_exclude is NOT used to write "excluded" tags (see
above) -- it's kept purely for its original purpose (filtering the
manual-coding-candidate stats below it).

Section B: the training/validation split.
- TRAINING_SELECTION_MODE == "document_driven" (this study's configuration):
  every row belonging to a TRAINING_PARTICIPANT_IDS participant becomes
  training -- the whole participant, all their rows, taken together.
  == "random": training is instead drawn via random_sample(), using
  config_project.TRAINING_DOCUMENT_CLUSTER_COLUMN/TRAINING_SUB_CLUSTER_COLUMN.
- Validation is drawn via random_sample() using
  config_project.VALIDATION_DOCUMENT_CLUSTER_COLUMN/VALIDATION_SUB_CLUSTER_COLUMN
  (VALIDATION_SELECTION_MODE == "random", this study's configuration).
- Whichever rows training selects (however it selects them) are excluded
  from validation's eligible pool, alongside every row already tagged
  "excluded" on disk -- training and validation never overlap, and neither
  ever lands on an already-excluded row.

random_sample() balances a draw across document-clusters (e.g. "condition"):
sample_size rows are drawn per document-cluster, split as evenly as
possible across that cluster's sub-clusters (the cartesian product of the
sub_cluster columns' distinct values found within the cluster), with any
shortfall (floor-rounding remainder, or a sub-cluster running short) topped
up from the document-cluster's remaining eligible rows as a whole. Always
without replacement, so no row is ever selected twice.

At the end, the selection is printed and the console is asked whether to
write it to META_OUTPUT_FILE. If DATASET_COLUMN already has entries other
than "excluded", a second confirmation is required before overwriting them.
Rows already tagged "excluded" (on disk) are never overwritten with
anything else, regardless of what this run computes -- this script can only
ever add/replace training/validation tags, never touch exclusion.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.config.project import (
    DATASET_COLUMN,
    DOCUMENT_KEY_COLUMNS,
    DOCUMENT_ORDER_COLUMN,
    META_OUTPUT_FILE,
    NAMING_BY_MODE,
    RANDOM_DATASET_ORDER_COLUMN,
    TRAINING_DOCUMENT_CLUSTER_COLUMN,
    TRAINING_SELECTION_MODE,
    TRAINING_SUB_CLUSTER_COLUMN,
    VALIDATION_DOCUMENT_CLUSTER_COLUMN,
    VALIDATION_SUB_CLUSTER_COLUMN,
)

# ==========================================
# Named constants -- easy to change without touching the logic below.
# ==========================================

# Rows drawn per document-cluster for validation (and, if
# TRAINING_SELECTION_MODE is switched to "random", for training).
VALIDATION_CASES_PER_CLUSTER = 50
TRAINING_CASES_PER_CLUSTER = 50  # only used when TRAINING_SELECTION_MODE == "random"

# Reproducible random selection/shuffling; re-running the script without
# changing the data reproduces the same selection and numbering.
RANDOM_SEED = 42

# Only used when TRAINING_SELECTION_MODE == "document_driven" (this study's
# active configuration): the NAMING_BY_MODE["document_driven"].group_column
# (think_aloud_id) values whose rows all become training.
TRAINING_PARTICIPANT_IDS: list[int] = [20, 50, 42, 45]


# ==========================================
# Row identity
# ==========================================


def _row_key_columns() -> list[str]:
    return [*DOCUMENT_KEY_COLUMNS, DOCUMENT_ORDER_COLUMN]


def _row_keys(df: pd.DataFrame) -> list[tuple]:
    """Every row's unique identity: its DOCUMENT_KEY_COLUMNS values, then
    its DOCUMENT_ORDER_COLUMN value -- (cycle_id, position) for this study."""
    return list(df[_row_key_columns()].itertuples(index=False, name=None))


def _rows_for_participants(df: pd.DataFrame, think_aloud_ids: list[int]) -> set[tuple]:
    return set(_row_keys(df[df["think_aloud_id"].isin(think_aloud_ids)]))


def _existing_excluded_rows(full_df: pd.DataFrame) -> set[tuple]:
    """Rows already tagged "excluded" in DATASET_COLUMN on disk -- manually
    maintained, never written by this script (see module docstring). Empty
    if DATASET_COLUMN doesn't exist yet (nothing has been excluded manually
    so far)."""
    if DATASET_COLUMN not in full_df.columns:
        return set()
    return set(_row_keys(full_df[full_df[DATASET_COLUMN] == "excluded"]))


# ==========================================
# Random selection
# ==========================================


def random_sample(
    df: pd.DataFrame,
    document_cluster: list[str],
    sub_cluster: list[str],
    excluded_rows: set[tuple],
    sample_size: int,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Randomly samples individual rows from df, balanced across every
    combination of sub_cluster columns' values ("sub-clusters"), separately
    for each combination of document_cluster columns' values
    ("document-clusters").

    For each document-cluster group: builds the sub-cluster matrix (the
    cartesian product of each sub_cluster column's distinct values found
    within that group), draws sample_size // len(matrix) rows from each
    sub-cluster (fewer if it doesn't have that many eligible rows), then
    tops up any shortfall (from the floor-rounding remainder, or a
    sub-cluster running short) from the document-cluster's remaining
    eligible rows as a whole. Always without replacement, so no row is ever
    selected twice.

    document_cluster/sub_cluster empty ([]) means "one group covering
    everything" for that dimension.

    Returns one dict per document-cluster group: {"document_name": "-".join
    of the group's column values (or "all" if document_cluster is empty) --
    "-" rather than "_" so a multi-column document_name can't be confused
    with the "_"-separated type/cluster split in the dataset value string
    (see _numbered_tags()), "document_elements": [row_key, ...]}, each
    row_key matching _row_keys()'s shape. Up to sample_size elements per
    group (fewer if the group doesn't have that many eligible rows).
    """
    rng = np.random.default_rng(seed)

    keys = _row_keys(df)
    eligible = df.loc[[key not in excluded_rows for key in keys]].copy()
    eligible["_row_key"] = [key for key in keys if key not in excluded_rows]

    if document_cluster:
        groups = list(eligible.groupby(document_cluster, sort=False))
    else:
        groups = [((), eligible)]

    results: list[dict] = []
    for group_values, group_df in groups:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        document_name = "-".join(str(value) for value in group_values) if group_values else "all"

        if sub_cluster:
            combos = list(itertools.product(*(group_df[col].unique() for col in sub_cluster)))
        else:
            combos = [()]

        quota = sample_size // len(combos) if combos else 0
        selected: set[tuple] = set()
        for combo in combos:
            sub_df = group_df
            for column, value in zip(sub_cluster, combo):
                sub_df = sub_df[sub_df[column] == value]
            available = [key for key in sub_df["_row_key"] if key not in selected]
            take = min(quota, len(available))
            if take:
                chosen = rng.choice(len(available), size=take, replace=False)
                selected.update(available[i] for i in chosen)

        deficit = sample_size - len(selected)
        if deficit > 0:
            leftovers = [key for key in group_df["_row_key"] if key not in selected]
            take = min(deficit, len(leftovers))
            if take:
                chosen = rng.choice(len(leftovers), size=take, replace=False)
                selected.update(leftovers[i] for i in chosen)

        results.append({"document_name": document_name, "document_elements": sorted(selected)})

    return results


def _selected_rows(results: list[dict]) -> set[tuple]:
    combined: set[tuple] = set()
    for entry in results:
        combined.update(entry["document_elements"])
    return combined


def run_random_selection(
    full_df: pd.DataFrame, excluded_rows: set[tuple]
) -> tuple[list[dict] | None, list[dict]]:
    """Runs the random-selection phase: training (only if
    TRAINING_SELECTION_MODE == "random") then validation, ensuring
    validation's eligible pool excludes everything already excluded
    (excluded_rows -- see _existing_excluded_rows(), read from
    DATASET_COLUMN on disk, never computed by this script) or selected for
    training -- regardless of whether training was drawn via random_sample()
    or the document-driven TRAINING_PARTICIPANT_IDS list.

    Returns (training_results, validation_results). training_results is
    None when TRAINING_SELECTION_MODE is "document_driven" (training isn't
    drawn via random_sample() at all in that case)."""
    excluded_rows = set(excluded_rows)

    training_results: list[dict] | None = None
    if TRAINING_SELECTION_MODE == "random":
        training_results = random_sample(
            full_df,
            TRAINING_DOCUMENT_CLUSTER_COLUMN,
            TRAINING_SUB_CLUSTER_COLUMN,
            excluded_rows,
            TRAINING_CASES_PER_CLUSTER,
        )
        excluded_rows = excluded_rows | _selected_rows(training_results)
    else:
        excluded_rows = excluded_rows | _rows_for_participants(full_df, TRAINING_PARTICIPANT_IDS)

    validation_results = random_sample(
        full_df,
        VALIDATION_DOCUMENT_CLUSTER_COLUMN,
        VALIDATION_SUB_CLUSTER_COLUMN,
        excluded_rows,
        VALIDATION_CASES_PER_CLUSTER,
    )
    return training_results, validation_results


# ==========================================
# Dataset column construction
# ==========================================


def _numbered_tags(
    row_keys: list[tuple], type_prefix: str, cluster: str, seed: int
) -> dict[tuple, tuple[str, int]]:
    """Shuffles row_keys and numbers them 0..N-1 within (type_prefix,
    cluster). Returns {row_key: (dataset_value, order)} -- dataset_value is
    just "{type_prefix}_{cluster}" (no number in it, see module docstring);
    order is the row's shuffled position, meant for RANDOM_DATASET_ORDER_COLUMN."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(row_keys))
    shuffled = [row_keys[i] for i in order]
    dataset_value = f"{type_prefix}_{cluster}"
    return {key: (dataset_value, i) for i, key in enumerate(shuffled)}


def _document_driven_training_tags(
    full_df: pd.DataFrame, participant_ids: list[int], seed: int
) -> dict[tuple, tuple[str, int]]:
    """Document-driven training has no document-cluster/sub-cluster of its
    own to name -- NAMING_BY_MODE["document_driven"].group_column's value
    (the participant) is used as the cluster instead, numbered within each
    participant's rows."""
    tags: dict[tuple, tuple[str, int]] = {}
    group_column = NAMING_BY_MODE["document_driven"].group_column
    for participant_id in participant_ids:
        subset = full_df[full_df[group_column] == participant_id]
        tags.update(_numbered_tags(_row_keys(subset), "training", str(participant_id), seed))
    return tags


def _build_fresh_dataset_columns(
    full_df: pd.DataFrame,
    training_results: list[dict] | None,
    validation_results: list[dict],
) -> tuple[pd.Series, pd.Series]:
    """The (dataset, dataset_order) columns as this run computes them from
    scratch: training/validation tags + order for selected rows, left blank
    (NaN) for every other row. Never writes "excluded" -- that's manually
    maintained and only ever read (see _existing_excluded_rows());
    _resolve_dataset_columns() is what carries forward whatever "excluded"
    tags already exist on disk."""
    tags: dict[tuple, tuple[str, int]] = {}

    if training_results is not None:
        for entry in training_results:
            tags.update(
                _numbered_tags(
                    entry["document_elements"], "training", entry["document_name"], RANDOM_SEED
                )
            )
    else:
        tags.update(_document_driven_training_tags(full_df, TRAINING_PARTICIPANT_IDS, RANDOM_SEED))

    for entry in validation_results:
        tags.update(
            _numbered_tags(
                entry["document_elements"], "validation", entry["document_name"], RANDOM_SEED
            )
        )

    row_keys = _row_keys(full_df)
    dataset_values = [tags[key][0] if key in tags else None for key in row_keys]
    order_values = [tags[key][1] if key in tags else None for key in row_keys]
    dataset_column = pd.Series(dataset_values, index=full_df.index, dtype="object")
    order_column = pd.Series(order_values, index=full_df.index, dtype="object")
    return dataset_column, order_column


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def _resolve_dataset_columns(
    full_df: pd.DataFrame, fresh_dataset: pd.Series, fresh_order: pd.Series
) -> tuple[pd.Series, pd.Series] | None:
    """Decides the final (dataset, dataset_order) columns to write, honoring
    the confirmation and never-overwrite-exclusion rules. Returns None if
    the write should be aborted (existing non-"excluded" DATASET_COLUMN
    entries found and the user declined to overwrite them)."""
    existing_dataset = full_df[DATASET_COLUMN] if DATASET_COLUMN in full_df.columns else None
    existing_order = full_df[RANDOM_DATASET_ORDER_COLUMN] if RANDOM_DATASET_ORDER_COLUMN in full_df.columns else None

    if existing_dataset is not None:
        has_other_than_excluded = (
            (existing_dataset.notna()) & (existing_dataset != "excluded")
        ).any()
        if has_other_than_excluded:
            if not _confirm(
                f"{DATASET_COLUMN!r} column already has training/validation/other "
                "entries. Overwrite them? [y/N] "
            ):
                print("Aborted: dataset column was not changed.")
                return None
        # Exclusion tags are sacrosanct: a row already "excluded" on disk
        # stays "excluded" (and keeps its old order value, if any) no
        # matter what this run computes for it.
        previously_excluded = existing_dataset == "excluded"
        fresh_dataset = fresh_dataset.where(~previously_excluded, "excluded")
        if existing_order is not None:
            fresh_order = fresh_order.where(~previously_excluded, existing_order)

    return fresh_dataset, fresh_order


def _print_selection(label: str, results: list[dict]) -> None:
    print(f"\n{label} selection:")
    for entry in results:
        print(f"  {entry['document_name']}: {len(entry['document_elements'])} row(s)")
        print(f"    {entry['document_elements']}")


def main() -> None:
    # Load the sheet from the Excel file
    df = pd.read_excel(
        META_OUTPUT_FILE["filename"], sheet_name=META_OUTPUT_FILE["sheetname"]
    )
    full_df = df.copy()  # unfiltered; the "old part" below reassigns df itself

    # drop ids that are incomplete
    # Identify all ids, that have not done three trials with 50 cases each
    id_to_exclude_df = df.groupby(df["think_aloud_id"]).filter(lambda x: len(x) != 150)
    id_to_exclude = id_to_exclude_df["think_aloud_id"].unique().tolist()
    # Additionally identified ids that have technical problems during the recording
    id_to_exclude.extend([28, 47])
    print("IDs to exclude:", id_to_exclude)
    df = df[~df["think_aloud_id"].isin(id_to_exclude)]

    # Filter for "no advice" condition and calculate average performance for each think_aloud_id
    no_advice_df = df[df["condition"] == "no advice"]
    grouped_no_advice_df = no_advice_df.groupby("think_aloud_id").agg(
        average_performance_binary=("performance_binary", "mean")
    )
    # Sort the performance in descending order
    grouped_no_advice_df = grouped_no_advice_df.sort_values(
        by="average_performance_binary", ascending=False
    )

    # Calculate the mean performance across all "no advice" entries
    mean_no_advice_performance = grouped_no_advice_df["average_performance_binary"].mean()

    # Get the two highest and two lowest entries
    top_two = grouped_no_advice_df.head(2)
    bottom_two = grouped_no_advice_df.tail(2)

    # Find entries closest to the mean
    grouped_no_advice_df["distance_from_mean"] = abs(
        grouped_no_advice_df["average_performance_binary"] - mean_no_advice_performance
    )
    above_mean = grouped_no_advice_df[
        grouped_no_advice_df["average_performance_binary"] >= mean_no_advice_performance
    ].tail(1)
    below_mean = grouped_no_advice_df[
        grouped_no_advice_df["average_performance_binary"] < mean_no_advice_performance
    ].head(1)

    # Combine the IDs from the top two, bottom two, and those closest to the mean into a final list for manual coding
    final_list = (
        top_two.index.tolist()
        + bottom_two.index.tolist()
        + above_mean.index.tolist()
        + below_mean.index.tolist()
    )
    print("Final list of IDs:", final_list)

    # ---- Section B: training/validation random selection ----
    # Exclusion is read from the file, never computed/written here -- see
    # module docstring. Rows already tagged "excluded" on disk are kept out
    # of the random draw's eligible pool.
    excluded_rows = _existing_excluded_rows(full_df)
    training_results, validation_results = run_random_selection(full_df, excluded_rows)

    if training_results is not None:
        _print_selection("Training (random)", training_results)
    _print_selection("Validation", validation_results)

    if not _confirm("\nAdd this selection to the dataset column? [y/N] "):
        print("Aborted: nothing written.")
        return

    fresh_dataset, fresh_order = _build_fresh_dataset_columns(
        full_df, training_results, validation_results
    )
    resolved = _resolve_dataset_columns(full_df, fresh_dataset, fresh_order)
    if resolved is None:
        return
    final_dataset, final_order = resolved

    full_df[DATASET_COLUMN] = final_dataset
    full_df[RANDOM_DATASET_ORDER_COLUMN] = final_order
    full_df.to_excel(
        META_OUTPUT_FILE["filename"],
        sheet_name=META_OUTPUT_FILE["sheetname"],
        index=False,
    )

    print(
        f"\nWrote {DATASET_COLUMN!r}/{RANDOM_DATASET_ORDER_COLUMN!r} columns to "
        f"{META_OUTPUT_FILE['filename']}"
    )
    print(full_df[DATASET_COLUMN].value_counts(dropna=False))


if __name__ == "__main__":
    main()
