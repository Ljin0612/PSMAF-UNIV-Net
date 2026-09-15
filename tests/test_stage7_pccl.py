"""Stage 7 paired loading, PCCL, sampling, and evaluation regressions."""

from argparse import Namespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_pccl_univ_mta_fasterrcnn_m3fd import build_arg_parser
from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import prediction_to_evaluation
from psmaf_univ.m3fd_detection import PairedM3FDDetectionDataset
from detection.scripts.train_pccl_univ_mta_fasterrcnn_m3fd import (
    build_arg_parser as build_train_parser, build_pccl_optimizer, pccl_mta_tokens,
    validate_pccl_configuration,
)
from psmaf_univ.pccl import (attention_pseudo_labels, pccl_objective, sample_patch_indices,
                             sampled_attention_pseudo_labels)


def _paired_tree(tmp_path):
    for name in ("ir", "vi", "labels", "meta"):
        (tmp_path / name).mkdir()
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(tmp_path / "ir" / "one.png")
    Image.fromarray(np.full((12, 16, 3), 127, dtype=np.uint8)).save(tmp_path / "vi" / "one.png")
    (tmp_path / "labels" / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "meta" / "train.txt").write_text("one\n")


def test_paired_loader_returns_ir_rgb_and_labels(tmp_path):
    _paired_tree(tmp_path)
    ir, rgb, target = PairedM3FDDetectionDataset(tmp_path, "train", 640)[0]
    assert ir.shape == rgb.shape == (3, 640, 640)
    assert target["labels"].tolist() == [0]
    assert not torch.equal(ir, rgb)


def test_attention_pseudo_matrix_shape_and_positive_diagonal():
    matrix = attention_pseudo_labels(torch.rand(2, 4, 17, 17), gamma=0.6)
    assert matrix.shape == (2, 17, 17)
    assert torch.all(matrix.diagonal(dim1=-2, dim2=-1) == 1)
    assert torch.all(matrix.sum(-1) >= 1)


def test_attention_labels_use_per_row_cumulative_mass_not_global_range():
    attention = torch.tensor([[[0.40, 0.30, 0.20, 0.10],
                               [90.0, 5.0, 3.0, 2.0],
                               [0.01, 0.01, 0.01, 0.97],
                               [0.25, 0.25, 0.25, 0.25]]])
    labels = attention_pseudo_labels(attention, gamma=0.6)
    # Row zero needs its two highest keys to cross 60%; its values are tiny
    # compared with row one, proving there is no image-global range cutoff.
    assert labels[0, 0].tolist() == [1, 1, 0, 0]
    assert labels[0, 1, 0] == 1


def test_sampled_labels_are_gathered_from_full_attention_semantics():
    attention = torch.tensor([[[.4, .3, .2, .1], [.1, .4, .3, .2],
                               [.2, .1, .4, .3], [.3, .2, .1, .4]]])
    indices = torch.tensor([0, 2])
    expected = attention_pseudo_labels(attention, .6)[:, indices][:, :, indices]
    assert torch.equal(sampled_attention_pseudo_labels(attention, indices, .6), expected)


def test_pccl_loss_is_finite_and_backpropagates():
    anchors = torch.randn(2, 8, 16)
    infrared = torch.randn(2, 8, 16, requires_grad=True)
    visible = torch.randn(2, 8, 16, requires_grad=True)
    labels = attention_pseudo_labels(torch.rand(2, 3, 8, 8))
    loss, loss_ia, loss_va = pccl_objective(anchors, infrared, visible, labels)
    assert all(torch.isfinite(value) for value in (loss, loss_ia, loss_va))
    loss.backward()
    assert infrared.grad is not None and visible.grad is not None


def test_patch_sampling_caps_640_token_grid_without_duplicates():
    indices = sample_patch_indices(40 * 40, 256)
    assert indices.shape == (256,)
    assert indices.unique().numel() == 256
    assert int(indices.max()) < 1600


def test_stage7_eval_defaults_to_640_and_ir_only_conversion():
    args = build_arg_parser().parse_args(["--checkpoint", "stage7.pth"])
    assert args.image_size == 640
    assert args.output_json.as_posix() == "outputs/stage7_eval/metrics.json"
    prediction = {"boxes": torch.zeros(1, 4), "scores": torch.ones(1),
                  "labels": torch.ones(1, dtype=torch.long)}
    converted = prediction_to_evaluation(prediction)
    assert converted["labels"].tolist() == [0]


class _TinyBackbone(torch.nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.adapter = torch.nn.Linear(2, 2)
        if freeze:
            self.encoder.requires_grad_(False)


def test_pccl_optimizer_assigns_univ_lr_and_excludes_frozen_univ():
    backbone = _TinyBackbone()
    projector = torch.nn.Linear(2, 2)
    optimizer = build_pccl_optimizer(backbone, backbone, (projector,), lr=.1, univ_lr=.002,
                                     pccl_lr=.03)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["univ"]["lr"] == .002
    assert groups["detector_mta"]["lr"] == .1
    assert groups["pccl_projectors"]["lr"] == .03
    backbone.encoder.requires_grad_(False)
    optimizer = build_pccl_optimizer(backbone, backbone, (projector,), lr=.1, univ_lr=.002)
    assert "univ" not in {group["name"] for group in optimizer.param_groups}


def test_frozen_norm_pccl_is_rejected_with_actionable_message():
    args = build_train_parser().parse_args(["--freeze-univ", "true"])
    with pytest.raises(ValueError, match="does not update inference-time parameters"):
        validate_pccl_configuration(args, _TinyBackbone(freeze=True))


def test_p4_pccl_backpropagates_to_mta_when_encoder_frozen():
    args = build_train_parser().parse_args(
        ["--freeze-univ", "true", "--pccl-feature-level", "p4"])
    backbone = _TinyBackbone(freeze=True)
    validate_pccl_configuration(args, backbone)
    source = torch.randn(1, 2, requires_grad=False)
    p4 = backbone.adapter(source).reshape(1, 2, 1, 1)
    tokens = pccl_mta_tokens({"P4": p4}, "p4")
    tokens.square().sum().backward()
    assert backbone.adapter.weight.grad is not None
    assert torch.count_nonzero(backbone.adapter.weight.grad)
