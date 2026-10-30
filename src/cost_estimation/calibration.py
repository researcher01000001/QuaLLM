"""Global, per-model, per-effort calibration store for cost-estimation factors"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.general import COST_ESTIMATION_CALIBRATION_FILE
from src.io.io_utils import read_json_safe, write_json

# Effort bucket used for calibration entries gathered while reasoning was not
# requested at all (CodingConfiguration.reasoning=False).
NO_REASONING_BUCKET = "none"


@dataclass(frozen=True)
class ModelCalibration:
    """completion_token_factor is relative to a single paragraph's own token
    count (completion_tokens / paragraph_tokens), matching how
    src.cost_estimation.estimation.ConfigurationCostEstimate.estimated_completion_tokens
    applies it -- NOT relative to the full prompt's token count, which is
    dominated by the largely-fixed coding-scheme/instructions overhead and
    doesn't track completion size. reasoning_token_factor is relative to
    completion_tokens (completion output size is what drives reasoning
    length)."""

    completion_token_factor: float
    reasoning_token_factor: float
    sample_calls: int
    last_updated: str
    last_prefix: str


def _is_flat_entry(value: Any) -> bool:
    """True for the old per-model schema: a dict holding calibration fields
    directly, rather than nested under effort-bucket names."""
    return isinstance(value, dict) and (
        "completion_token_factor" in value or "output_token_factor" in value
    )


def _parse_entry(entry: dict[str, Any]) -> ModelCalibration | None:
    try:
        completion_factor = entry.get("completion_token_factor")
        if completion_factor is None:
            completion_factor = entry["output_token_factor"]  # old field name
        return ModelCalibration(
            completion_token_factor=float(completion_factor),
            reasoning_token_factor=float(entry["reasoning_token_factor"]),
            sample_calls=int(entry["sample_calls"]),
            last_updated=str(entry["last_updated"]),
            last_prefix=str(entry["last_prefix"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_calibration(
    *, calibration_file: Path = COST_ESTIMATION_CALIBRATION_FILE
) -> dict[str, dict[str, ModelCalibration]]:
    """Load the calibration store, keyed by model_name -> effort_bucket.
    Missing/unreadable file -> empty dict, never an error. A model entry in
    the old flat schema is treated as that model's NO_REASONING_BUCKET
    entry."""
    raw = read_json_safe(calibration_file, default={})

    calibration: dict[str, dict[str, ModelCalibration]] = {}
    for model_name, value in raw.items():
        efforts = {NO_REASONING_BUCKET: value} if _is_flat_entry(value) else value
        if not isinstance(efforts, dict):
            continue
        bucket: dict[str, ModelCalibration] = {}
        for effort, entry in efforts.items():
            if not isinstance(entry, dict):
                continue
            parsed = _parse_entry(entry)
            if parsed is not None:
                bucket[effort] = parsed
        if bucket:
            calibration[model_name] = bucket
    return calibration


def save_calibration(
    calibration: dict[str, dict[str, ModelCalibration]],
    *,
    calibration_file: Path = COST_ESTIMATION_CALIBRATION_FILE,
) -> None:
    payload = {
        model_name: {effort: asdict(entry) for effort, entry in efforts.items()}
        for model_name, efforts in calibration.items()
    }
    write_json(payload, calibration_file)


def update_calibration(
    model_name: str,
    effort: str,
    *,
    paragraph_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    calls: int,
    prefix: str,
    calibration_file: Path = COST_ESTIMATION_CALIBRATION_FILE,
) -> ModelCalibration | None:
    """Recompute (model_name, effort)'s calibration entry from this run's
    aggregate actual usage and persist it. effort is the effort_bucket the
    usage was gathered under (NO_REASONING_BUCKET when reasoning was
    disabled -- see CodingConfiguration.effective_reasoning_effort).
    paragraph_tokens is the sum of tokenizing just the real paragraph text
    sent on each call (not the full billed prompt, which includes the fixed
    coding-scheme/instructions overhead) -- see ModelCalibration's docstring
    for why. Returns None (and leaves the store unchanged) when there isn't
    enough data to compute a meaningful factor (no calls, or zero
    paragraph/completion tokens)."""
    if calls <= 0 or paragraph_tokens <= 0 or completion_tokens <= 0:
        return None

    entry = ModelCalibration(
        completion_token_factor=completion_tokens / paragraph_tokens,
        reasoning_token_factor=reasoning_tokens / completion_tokens,
        sample_calls=calls,
        last_updated=datetime.now(timezone.utc).isoformat(),
        last_prefix=prefix,
    )

    calibration = load_calibration(calibration_file=calibration_file)
    calibration.setdefault(model_name, {})[effort] = entry
    save_calibration(calibration, calibration_file=calibration_file)
    return entry
