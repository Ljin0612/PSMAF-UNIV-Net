"""Focused regression tests for Stage 6.1 resolution handling."""

import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import build_arg_parser as eval_parser
from detection.scripts.train_univ_mta_fasterrcnn_m3fd import training_configuration_summary
from psmaf_univ.checkpoint_loader import resize_pos_embed_if_needed
from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter


def test_pos_embed_224_is_unchanged_and_reports_grid():
    original = torch.randn(1, 196, 768)
    state = {"pos_embed": original.clone()}
    info = resize_pos_embed_if_needed(state, {"pos_embed": torch.empty_like(original)})
    assert not info["resized"]
    assert info["source_grid_size"] == info["target_grid_size"] == (14, 14)
    assert torch.equal(state["pos_embed"], original)


def test_pos_embed_320_is_bicubically_resized():
    state = {"pos_embed": torch.randn(1, 196, 768)}
    info = resize_pos_embed_if_needed(state, {"pos_embed": torch.empty(1, 400, 768)})
    assert info["resized"]
    assert info["source_grid_size"] == (14, 14)
    assert info["target_grid_size"] == (20, 20)
    assert state["pos_embed"].shape == (1, 400, 768)


@pytest.mark.parametrize("grid,p3", [((14, 14), 28), ((20, 20), 40)])
def test_mta_shapes_at_supported_resolutions(grid, p3):
    adapter = MultiScaleTaskAdapter(4, 8, 256)
    result = adapter(torch.randn(1, 4, p3, p3), torch.randn(1, grid[0] * grid[1], 8), grid_size=grid)
    assert [tuple(result[name].shape) for name in ("P3", "P4", "P5")] == [
        (1, 256, p3, p3), (1, 256, *grid), (1, 256, grid[0] // 2, grid[1] // 2)
    ]
    assert all(torch.isfinite(value).all() for value in result.values())


def test_eval_parser_accepts_320():
    args = eval_parser().parse_args(["--checkpoint", "detector.pth", "--image-size", "320"])
    assert args.image_size == 320


def test_training_summary_records_resolution_fields():
    from argparse import Namespace
    model = torch.nn.Linear(1, 1)
    backbone = Namespace(encoder=torch.nn.Linear(1, 1), adapter=torch.nn.Linear(1, 1))
    args = Namespace(image_size=320, freeze_univ=True, unfreeze_last_n_blocks=0,
                     unfreeze_norm=False, univ_lr=1e-5)
    summary = training_configuration_summary(args, model, backbone, [])
    assert summary["input_token_grid_size"] == [20, 20]
    assert summary["adapter_output_shapes"]["P3"] == [256, 40, 40]
