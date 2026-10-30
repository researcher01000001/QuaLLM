import json
import time
from unittest.mock import Mock, patch

import requests

from src.cost_estimation.pricing import ModelPricing, fetch_model_pricing


def _fake_models_response(prices: dict[str, tuple[str, str]]) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "data": [
            {"id": model_id, "pricing": {"prompt": prompt, "completion": completion}}
            for model_id, (prompt, completion) in prices.items()
        ]
    }
    return response


def test_fetch_model_pricing_returns_parsed_prices(tmp_path):
    cache_file = tmp_path / "pricing.json"
    response = _fake_models_response({"test/model": ("0.000003", "0.000015")})

    with patch("src.cost_estimation.pricing.requests.get", return_value=response):
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    assert result == {
        "test/model": ModelPricing(
            prompt_price_per_token=0.000003, completion_price_per_token=0.000015
        )
    }
    assert cache_file.exists()


def test_fetch_model_pricing_captures_cache_prices(tmp_path):
    cache_file = tmp_path / "pricing.json"
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "data": [
            {
                "id": "anthropic/test-model",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "input_cache_read": "0.0000003",
                    "input_cache_write": "0.00000375",
                },
            }
        ]
    }

    with patch("src.cost_estimation.pricing.requests.get", return_value=response):
        result = fetch_model_pricing(
            {"anthropic/test-model"}, cache_file=cache_file, ttl_hours=24
        )

    assert result["anthropic/test-model"] == ModelPricing(
        prompt_price_per_token=0.000003,
        completion_price_per_token=0.000015,
        cache_read_price_per_token=0.0000003,
        cache_write_price_per_token=0.00000375,
    )


def test_fetch_model_pricing_adds_gemini_cache_storage_surcharge_to_write_price(tmp_path):
    cache_file = tmp_path / "pricing.json"
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "data": [
            {
                "id": "google/gemini-test",
                "pricing": {
                    "prompt": "0.000000375",
                    "completion": "0.000001875",
                    "input_cache_read": "0.0000000375",
                    "input_cache_write": "0.0000000208333333333333",
                },
            }
        ]
    }

    with patch("src.cost_estimation.pricing.requests.get", return_value=response):
        result = fetch_model_pricing(
            {"google/gemini-test"}, cache_file=cache_file, ttl_hours=24
        )

    pricing = result["google/gemini-test"]
    assert pricing is not None
    assert pricing.cache_read_price_per_token == 0.0000000375
    assert pricing.cache_write_price_per_token == 0.000000375 + 0.0000000208333333333333


def test_fetch_model_pricing_returns_none_for_unknown_model(tmp_path):
    cache_file = tmp_path / "pricing.json"
    response = _fake_models_response({"other/model": ("0.000001", "0.000002")})

    with patch("src.cost_estimation.pricing.requests.get", return_value=response):
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    assert result == {"test/model": None}


def test_fetch_model_pricing_uses_fresh_cache_without_refetching(tmp_path):
    cache_file = tmp_path / "pricing.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fetched_at": time.time(),
                "prices": {"test/model": {"prompt": 0.1, "completion": 0.2}},
            }
        ),
        encoding="utf-8",
    )

    with patch("src.cost_estimation.pricing.requests.get") as mock_get:
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    mock_get.assert_not_called()
    assert result == {
        "test/model": ModelPricing(prompt_price_per_token=0.1, completion_price_per_token=0.2)
    }


def test_fetch_model_pricing_refetches_when_cache_expired(tmp_path):
    cache_file = tmp_path / "pricing.json"
    cache_file.write_text(
        json.dumps(
            {
                "fetched_at": time.time() - 3600 * 48,
                "prices": {"test/model": {"prompt": 0.1, "completion": 0.2}},
            }
        ),
        encoding="utf-8",
    )
    response = _fake_models_response({"test/model": ("0.5", "0.6")})

    with patch("src.cost_estimation.pricing.requests.get", return_value=response) as mock_get:
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    mock_get.assert_called_once()
    assert result == {
        "test/model": ModelPricing(prompt_price_per_token=0.5, completion_price_per_token=0.6)
    }


def test_fetch_model_pricing_falls_back_to_stale_cache_on_network_failure(tmp_path):
    cache_file = tmp_path / "pricing.json"
    cache_file.write_text(
        json.dumps(
            {
                "fetched_at": time.time() - 3600 * 48,
                "prices": {"test/model": {"prompt": 0.1, "completion": 0.2}},
            }
        ),
        encoding="utf-8",
    )

    with patch("src.cost_estimation.pricing.requests.get", side_effect=requests.ConnectionError("boom")):
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    assert result == {
        "test/model": ModelPricing(prompt_price_per_token=0.1, completion_price_per_token=0.2)
    }


def test_fetch_model_pricing_returns_none_on_failure_without_any_cache(tmp_path):
    cache_file = tmp_path / "pricing.json"

    with patch("src.cost_estimation.pricing.requests.get", side_effect=requests.ConnectionError("boom")):
        result = fetch_model_pricing({"test/model"}, cache_file=cache_file, ttl_hours=24)

    assert result == {"test/model": None}


def test_fetch_model_pricing_captures_reasoning_capability_and_price(tmp_path):
    cache_file = tmp_path / "pricing.json"
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "data": [
            {
                "id": "test/reasoning-model",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "internal_reasoning": "0.000015",
                },
                "supported_parameters": ["reasoning", "reasoning_effort", "temperature"],
            },
            {
                "id": "test/plain-model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "supported_parameters": ["temperature"],
            },
        ]
    }

    with patch("src.cost_estimation.pricing.requests.get", return_value=response):
        result = fetch_model_pricing(
            {"test/reasoning-model", "test/plain-model"}, cache_file=cache_file, ttl_hours=24
        )

    assert result["test/reasoning-model"] == ModelPricing(
        prompt_price_per_token=0.000003,
        completion_price_per_token=0.000015,
        reasoning_price_per_token=0.000015,
        supports_reasoning=True,
    )
    assert result["test/plain-model"] == ModelPricing(
        prompt_price_per_token=0.000001,
        completion_price_per_token=0.000002,
        reasoning_price_per_token=None,
        supports_reasoning=False,
    )
