"""Stage 7 paired loading, PCCL, sampling, and evaluation regressions."""

from argparse import Namespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_pccl_univ_mta_fasterrcnn_m3fd import build_arg_parser
from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import prediction_to_evaluation
from psmaf_univ.m3fd_detection import PairedM3FDDetectionDataset
from psmaf_univ.pccl import attention_pseudo_labels, pccl_objective, sample_patch_indices


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
