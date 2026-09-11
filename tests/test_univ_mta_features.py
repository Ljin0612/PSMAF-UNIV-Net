"""Shape-contract tests for the Stage 2 UNIV-to-MTA boundary."""

import pytest

torch = pytest.importorskip("torch")

from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter


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
