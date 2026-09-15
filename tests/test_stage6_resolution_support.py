"""Focused regression tests for Stage 6 resolution handling."""

import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import build_arg_parser as eval_parser
from detection.scripts.train_univ_mta_fasterrcnn_m3fd import (
    build_arg_parser as train_parser,
    training_configuration_summary,
)
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


def test_pos_embed_640_is_bicubically_resized_without_changing_dtype():
    state = {"pos_embed": torch.randn(1, 196, 768, dtype=torch.float64)}
    info = resize_pos_embed_if_needed(state, {"pos_embed": torch.empty(1, 1600, 768)})
    assert info["resized"]
    assert info["source_grid_size"] == (14, 14)
    assert info["target_grid_size"] == (40, 40)
    assert state["pos_embed"].shape == (1, 1600, 768)
    assert state["pos_embed"].dtype == torch.float64


def test_pos_embed_1024_is_resized_to_64_square_grid():
    state = {"pos_embed": torch.randn(1, 196, 768)}
    info = resize_pos_embed_if_needed(state, {"pos_embed": torch.empty(1, 4096, 768)})
    assert info["source_grid_size"] == (14, 14)
    assert info["target_grid_size"] == (64, 64)
    assert state["pos_embed"].shape == (1, 4096, 768)


@pytest.mark.parametrize(
    "grid,p3", [((14, 14), 28), ((20, 20), 40), ((40, 40), 80), ((64, 64), 128)]
)
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


@pytest.mark.parametrize("parser_factory", [train_parser, eval_parser])
def test_train_and_eval_parsers_accept_640(parser_factory):
    required = ["--checkpoint", "detector.pth"] if parser_factory is eval_parser else []
    args = parser_factory().parse_args([*required, "--image-size", "640"])
    assert args.image_size == 640


@pytest.mark.parametrize("parser_factory", [train_parser, eval_parser])
def test_train_and_eval_parsers_accept_224_320_and_1024(parser_factory):
    required = ["--checkpoint", "detector.pth"] if parser_factory is eval_parser else []
    for size in (224, 320, 1024):
        assert parser_factory().parse_args([*required, "--image-size", str(size)]).image_size == size


@pytest.mark.parametrize("parser_factory", [train_parser, eval_parser])
def test_train_and_eval_parsers_reject_unsupported_size(parser_factory, capsys):
    required = ["--checkpoint", "detector.pth"] if parser_factory is eval_parser else []
    with pytest.raises(SystemExit):
        parser_factory().parse_args([*required, "--image-size", "512"])
    assert "supported image sizes are 224, 320, 640, and 1024" in capsys.readouterr().err


def test_resume_argument_and_zero_evaluation_threshold_are_parsed():
    assert train_parser().parse_args(["--resume", "last.pth"]).resume.name == "last.pth"
    args = eval_parser().parse_args(["--checkpoint", "detector.pth", "--score-threshold", "0.0"])
    assert args.score_threshold == 0.0


def test_training_summary_records_resolution_fields():
    from argparse import Namespace
    model = torch.nn.Linear(1, 1)
    backbone = Namespace(encoder=torch.nn.Linear(1, 1), adapter=torch.nn.Linear(1, 1))
    args = Namespace(image_size=320, freeze_univ=True, unfreeze_last_n_blocks=0,
                     unfreeze_norm=False, univ_lr=1e-5)
    summary = training_configuration_summary(args, model, backbone, [])
    assert summary["input_token_grid_size"] == [20, 20]
    assert summary["adapter_output_shapes"]["P3"] == [256, 40, 40]


def test_training_summary_records_640_resolution_fields():
    from argparse import Namespace
    model = torch.nn.Linear(1, 1)
    backbone = Namespace(encoder=torch.nn.Linear(1, 1), adapter=torch.nn.Linear(1, 1))
    args = Namespace(image_size=640, freeze_univ=True, unfreeze_last_n_blocks=0,
                     unfreeze_norm=False, univ_lr=1e-5)
    summary = training_configuration_summary(args, model, backbone, [])
    assert summary["input_token_grid_size"] == [40, 40]
    assert summary["adapter_output_shapes"] == {
        "P3": [256, 80, 80], "P4": [256, 40, 40], "P5": [256, 20, 20]
    }


def test_training_summary_records_1024_shapes_and_parameter_fields():
    from argparse import Namespace
    model = torch.nn.Linear(1, 1)
    backbone = Namespace(encoder=torch.nn.Linear(1, 1), adapter=torch.nn.Linear(1, 1))
    args = Namespace(image_size=1024, freeze_univ=True, unfreeze_last_n_blocks=0,
                     unfreeze_norm=False, univ_lr=1e-5)
    summary = training_configuration_summary(args, model, backbone, [])
    assert summary["input_token_grid_size"] == [64, 64]
    assert summary["adapter_output_shapes"] == {
        "P3": [256, 128, 128], "P4": [256, 64, 64], "P5": [256, 32, 32]
    }
    for field in ("trainable_parameter_count", "trainable_univ_parameter_count",
                  "trainable_adapter_parameter_count", "trainable_detector_parameter_count"):
        assert field in summary
