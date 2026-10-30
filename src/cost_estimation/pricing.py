"""Per-token pricing lookup for OpenRouter models, used by cost estimation"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.config.general import (
    COST_ESTIMATION_PRICING_CACHE_FILE,
    COST_ESTIMATION_PRICING_CACHE_TTL_HOURS,
    OPENROUTER_API_KEY,
)
from src.io.io_utils import read_json_safe, write_json

MODELS_URL = "https://openrouter.ai/api/v1/models"
_REQUEST_TIMEOUT_SECONDS = 15
_PRICING_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ModelPricing:
    """Price per single token, in the provider's billing currency (USD)."""

    prompt_price_per_token: float
    completion_price_per_token: float
    # OpenRouter's discounted cache-read rate. None means the model-list
    # response did not expose cache pricing, so estimates must use ordinary
    # prompt pricing even if the request carries a cache hint.
    cache_read_price_per_token: float | None = None
    # Effective total price for writing one cached input token. OpenRouter's
    # Google pricing reports input_cache_write as a storage surcharge, whereas
    # other providers report the full write rate; _fetch_all_pricing()
    # normalizes both shapes into this field.
    cache_write_price_per_token: float | None = None
    # None when OpenRouter doesn't expose a distinct reasoning-token rate for
    # this model (pricing.internal_reasoning) -- callers should fall back to
    # completion_price_per_token in that case.
    reasoning_price_per_token: float | None = None
    # Whether this model's supported_parameters (per OpenRouter's live model
    # list) includes "reasoning" -- see run.py's reasoning-capability check.
    supports_reasoning: bool = False


def _load_cache(cache_file: Path) -> dict[str, Any] | None:
    return read_json_safe(cache_file)


def _save_cache(cache_file: Path, prices_by_model: dict[str, dict[str, Any]]) -> None:
    payload = {
        "schema_version": _PRICING_CACHE_SCHEMA_VERSION,
        "fetched_at": time.time(),
        "prices": prices_by_model,
    }
    write_json(payload, cache_file)


def _cache_is_fresh(cache: dict[str, Any], ttl_hours: float) -> bool:
    if cache.get("schema_version") != _PRICING_CACHE_SCHEMA_VERSION:
        return False
    fetched_at = cache.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return False
    return (time.time() - fetched_at) < ttl_hours * 3600


def _fetch_all_pricing() -> dict[str, dict[str, Any]] | None:
    """Fetch every model's pricing from OpenRouter. Returns None on failure."""
    headers = {}
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
    try:
        response = requests.get(MODELS_URL, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    prices_by_model: dict[str, dict[str, Any]] = {}
    for entry in data.get("data", []):
        model_id = entry.get("id")
        pricing = entry.get("pricing") or {}
        if not model_id or "prompt" not in pricing or "completion" not in pricing:
            continue
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
        except (TypeError, ValueError):
            continue
        reasoning_price = _optional_float(pricing.get("internal_reasoning"))
        cache_read_price = _optional_float(pricing.get("input_cache_read"))
        cache_write_price = _optional_float(pricing.get("input_cache_write"))
        # Gemini's endpoint pricing exposes cache storage as a surcharge on
        # top of ordinary input; OpenRouter's prompt-caching documentation
        # specifies: input price + five minutes of storage. Other explicit-
        # caching providers (for example Anthropic) expose the full write rate.
        if (
            cache_write_price is not None
            and model_id.startswith("google/")
            and cache_write_price < prompt
        ):
            cache_write_price += prompt
        supported_parameters = entry.get("supported_parameters") or []
        prices_by_model[model_id] = {
            "prompt": prompt,
            "completion": completion,
            "internal_reasoning": reasoning_price,
            "input_cache_read": cache_read_price,
            "input_cache_write": cache_write_price,
            "supports_reasoning": "reasoning" in supported_parameters,
        }
    return prices_by_model


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_model_pricing(
    model_names: set[str],
    *,
    cache_file: Path = COST_ESTIMATION_PRICING_CACHE_FILE,
    ttl_hours: float = COST_ESTIMATION_PRICING_CACHE_TTL_HOURS,
) -> dict[str, ModelPricing | None]:
    """Resolve current per-token pricing for each of model_names.

    A model missing from OpenRouter's response, or any total fetch/cache
    failure, resolves to None for the affected model(s) -- callers must treat
    that as "price unknown" (surfaced as N/A in the cost overview), not an
    error.
    """
    cache = _load_cache(cache_file)
    prices_by_model: dict[str, dict[str, Any]] | None = None

    if cache is not None and _cache_is_fresh(cache, ttl_hours):
        prices_by_model = cache.get("prices", {})
    else:
        fetched = _fetch_all_pricing()
        if fetched is not None:
            prices_by_model = fetched
            _save_cache(cache_file, fetched)
        elif cache is not None:
            # Live fetch failed -- fall back to a stale cache rather than
            # reporting every model as unpriced.
            prices_by_model = cache.get("prices", {})

    if prices_by_model is None:
        return {name: None for name in model_names}

    result: dict[str, ModelPricing | None] = {}
    for name in model_names:
        entry = prices_by_model.get(name)
        if entry is None:
            result[name] = None
        else:
            result[name] = ModelPricing(
                prompt_price_per_token=entry["prompt"],
                completion_price_per_token=entry["completion"],
                cache_read_price_per_token=entry.get("input_cache_read"),
                cache_write_price_per_token=entry.get("input_cache_write"),
                reasoning_price_per_token=entry.get("internal_reasoning"),
                supports_reasoning=bool(entry.get("supports_reasoning", False)),
            )
    return result
