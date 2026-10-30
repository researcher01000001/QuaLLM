import json

from src.cost_estimation.calibration import ModelCalibration, load_calibration, update_calibration


def test_load_calibration_missing_file_returns_empty_dict(tmp_path):
    assert load_calibration(calibration_file=tmp_path / "missing.json") == {}


def test_load_calibration_unreadable_file_returns_empty_dict(tmp_path):
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text("not json", encoding="utf-8")

    assert load_calibration(calibration_file=calibration_file) == {}


def test_load_calibration_migrates_old_flat_schema(tmp_path):
    """A model entry from before per-effort calibration was added (fields
    directly under the model name, no effort nesting) is treated as that
    model's 'none' (reasoning-disabled) bucket."""
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(
        json.dumps(
            {
                "test/model": {
                    "output_token_factor": 0.2,
                    "reasoning_token_factor": 0.0,
                    "sample_calls": 10,
                    "last_updated": "then",
                    "last_prefix": "p",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_calibration(calibration_file=calibration_file)

    assert set(loaded.keys()) == {"test/model"}
    assert loaded["test/model"]["none"] == ModelCalibration(
        completion_token_factor=0.2,
        reasoning_token_factor=0.0,
        sample_calls=10,
        last_updated="then",
        last_prefix="p",
    )


def test_update_calibration_computes_factors_and_persists(tmp_path):
    calibration_file = tmp_path / "calibration.json"

    result = update_calibration(
        "test/model",
        "medium",
        paragraph_tokens=1000,
        completion_tokens=200,
        reasoning_tokens=50,
        calls=10,
        prefix="my-prefix",
        calibration_file=calibration_file,
    )

    assert result == ModelCalibration(
        completion_token_factor=0.2,
        reasoning_token_factor=0.25,
        sample_calls=10,
        last_updated=result.last_updated,
        last_prefix="my-prefix",
    )
    loaded = load_calibration(calibration_file=calibration_file)
    assert loaded["test/model"]["medium"] == result


def test_update_calibration_keeps_separate_entries_per_effort(tmp_path):
    calibration_file = tmp_path / "calibration.json"
    update_calibration(
        "test/model",
        "none",
        paragraph_tokens=1000,
        completion_tokens=200,
        reasoning_tokens=0,
        calls=10,
        prefix="first-prefix",
        calibration_file=calibration_file,
    )

    update_calibration(
        "test/model",
        "high",
        paragraph_tokens=500,
        completion_tokens=400,
        reasoning_tokens=300,
        calls=5,
        prefix="second-prefix",
        calibration_file=calibration_file,
    )

    loaded = load_calibration(calibration_file=calibration_file)
    assert set(loaded["test/model"].keys()) == {"none", "high"}
    assert loaded["test/model"]["none"].completion_token_factor == 0.2
    assert loaded["test/model"]["high"].completion_token_factor == 0.8


def test_update_calibration_overwrites_previous_entry_for_same_model_and_effort(tmp_path):
    calibration_file = tmp_path / "calibration.json"
    update_calibration(
        "test/model",
        "medium",
        paragraph_tokens=1000,
        completion_tokens=200,
        reasoning_tokens=50,
        calls=10,
        prefix="first-prefix",
        calibration_file=calibration_file,
    )

    update_calibration(
        "test/model",
        "medium",
        paragraph_tokens=500,
        completion_tokens=100,
        reasoning_tokens=10,
        calls=5,
        prefix="second-prefix",
        calibration_file=calibration_file,
    )

    loaded = load_calibration(calibration_file=calibration_file)
    assert set(loaded["test/model"].keys()) == {"medium"}
    assert loaded["test/model"]["medium"].completion_token_factor == 0.2
    assert loaded["test/model"]["medium"].last_prefix == "second-prefix"


def test_update_calibration_returns_none_and_skips_write_when_no_calls(tmp_path):
    calibration_file = tmp_path / "calibration.json"

    result = update_calibration(
        "test/model",
        "none",
        paragraph_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        calls=0,
        prefix="my-prefix",
        calibration_file=calibration_file,
    )

    assert result is None
    assert not calibration_file.exists()


def test_update_calibration_keeps_other_models_entries(tmp_path):
    calibration_file = tmp_path / "calibration.json"
    update_calibration(
        "model/a",
        "none",
        paragraph_tokens=100,
        completion_tokens=10,
        reasoning_tokens=0,
        calls=1,
        prefix="p",
        calibration_file=calibration_file,
    )

    update_calibration(
        "model/b",
        "none",
        paragraph_tokens=200,
        completion_tokens=20,
        reasoning_tokens=0,
        calls=1,
        prefix="p",
        calibration_file=calibration_file,
    )

    loaded = load_calibration(calibration_file=calibration_file)
    assert set(loaded.keys()) == {"model/a", "model/b"}
