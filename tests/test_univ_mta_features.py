"""Shape-contract tests for the Stage 2 UNIV-to-MTA boundary."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter
from tools.inspect_univ_mta_features import validate_checkpoint_load_report


def checkpoint_report(load_fraction, loaded_key_count, loaded_parameter_count=1):
    return SimpleNamespace(
        load_fraction=load_fraction,
        loaded_key_count=loaded_key_count,
        loaded_parameter_count=loaded_parameter_count,
        model_state_key_count=313,
    )


def test_checkpoint_validation_accepts_fully_loaded_checkpoint():
    validation = validate_checkpoint_load_report(
        checkpoint_report(1.0, 313, 1_000), "student", 0.5
    )

    assert validation == {
        "passed": True,
        "min_load_fraction": 0.5,
        "load_fraction": 1.0,
        "loaded_key_count": 313,
        "model_state_key_count": 313,
        "loaded_parameter_count": 1_000,
    }


def test_checkpoint_validation_rejects_empty_load():
    with pytest.raises(RuntimeError, match="no usable weights.*randomly initialized"):
        validate_checkpoint_load_report(checkpoint_report(0.0, 0, 0), "student", 0.5)


def test_checkpoint_validation_rejects_load_below_threshold():
    with pytest.raises(RuntimeError, match="below the required minimum"):
        validate_checkpoint_load_report(checkpoint_report(0.49, 10), "teacher", 0.5)


def test_zero_checkpoint_threshold_still_rejects_empty_load():
    with pytest.raises(RuntimeError, match="no usable weights"):
        validate_checkpoint_load_report(checkpoint_report(0.0, 0, 0), "student", 0.0)


def test_univ_features_produce_finite_p3_p4_p5():
    adapter = MultiScaleTaskAdapter(384, 768, 256)
    outputs = adapter(
        torch.randn(1, 384, 28, 28),
        torch.randn(1, 196, 768),
        grid_size=(14, 14),
    )

    assert {name: list(value.shape) for name, value in outputs.items()} == {
        "P3": [1, 256, 28, 28],
        "P4": [1, 256, 14, 14],
        "P5": [1, 256, 7, 7],
    }
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_univ_semantic_tokens_require_grid_metadata():
    adapter = MultiScaleTaskAdapter(4, 8, 3)

    with pytest.raises(ValueError, match="require spatial_shape"):
        adapter(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 8))


def test_square_grid_size_metadata_works():
    adapter = MultiScaleTaskAdapter(4, 8, 3)
    outputs = adapter(
        torch.randn(1, 4, 4, 4), torch.randn(1, 4, 8), grid_size=(2, 2)
    )

    assert outputs["P4"].shape == (1, 3, 2, 2)
    assert outputs["P5"].shape == (1, 3, 1, 1)


def test_rectangular_grid_metadata_uses_existing_adapter_path():
    adapter = MultiScaleTaskAdapter(4, 8, 3)
    outputs = adapter(
        torch.randn(2, 4, 6, 10),
        torch.randn(2, 15, 8),
        spatial_shape=(3, 5),
    )

    assert outputs["P3"].shape == (2, 3, 6, 10)
    assert outputs["P4"].shape == (2, 3, 3, 5)
    assert outputs["P5"].shape == (2, 3, 1, 2)
    assert all(torch.isfinite(value).all() for value in outputs.values())
