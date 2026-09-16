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
    shared_encoder_tokens, validate_pccl_configuration,
)
from psmaf_univ.lora_adapter import (LoRALinear, attach_lora, is_lora_parameter,
                                     load_lora_state_dict, lora_state_dict)
from psmaf_univ.pccl import (PCCLLoss, attention_pseudo_labels, pccl_objective, sample_patch_indices,
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
    # Upstream UNIV excludes the key which crosses 60% cumulative mass.  Row
    # zero's values are tiny beside row one, proving there is no global cutoff.
    assert labels[0, 0].tolist() == [1, 0, 0, 0]
    assert labels[0, 1].tolist() == [0, 1, 0, 0]


def test_attention_labels_force_diagonal_and_define_zero_mass_rows():
    attention = torch.tensor([[[0., 1., 0.], [0., 0., 0.], [1., 0., 0.]]])
    labels = attention_pseudo_labels(attention, gamma=.6)
    assert torch.all(labels.diagonal(dim1=-2, dim2=-1) == 1)
    # cumsum <= 0 follows upstream UNIV and selects every tied zero entry.
    assert labels[0, 1].tolist() == [1, 1, 1]
    assert torch.all(labels.sum(-1) >= 1)


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


def test_pccl_logits_and_asymmetric_labels_share_query_key_direction(monkeypatch):
    anchors = torch.tensor([[[1., 0.], [0., 1.]]])
    patches = torch.tensor([[[1., 1.], [.5, -1.]]])
    labels = torch.tensor([[[1., 0.], [1., 1.]]])
    captured = {}

    def capture(logits, targets):
        captured["logits"] = logits
        captured["targets"] = targets
        return logits.sum() * 0

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", capture)
    PCCLLoss(temperature=1)(anchors, patches, labels)
    expected = torch.bmm(torch.nn.functional.normalize(patches, dim=-1),
                         torch.nn.functional.normalize(anchors, dim=-1).transpose(1, 2))
    assert torch.equal(captured["targets"], labels)
    assert torch.allclose(captured["logits"], expected)
    assert not torch.allclose(captured["logits"], expected.transpose(1, 2))


def test_pccl_loss_rejects_labels_outside_query_patch_key_anchor_shape():
    anchors = torch.randn(2, 4, 5)
    patches = torch.randn(2, 3, 5)
    with pytest.raises(ValueError, match=r"shape \(2, 3, 4\)"):
        PCCLLoss()(anchors, patches, torch.zeros(2, 4, 3))
    assert torch.isfinite(PCCLLoss()(anchors, patches, torch.zeros(2, 3, 4)))


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


class _TokenBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv = torch.nn.Linear(2, 6)
        self.attn.proj = torch.nn.Linear(2, 2)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(2, 4)
        self.mlp.fc2 = torch.nn.Linear(4, 2)

    def forward(self, value):
        return self.mlp.fc2(torch.relu(self.mlp.fc1(value)))


class _SharedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Conv2d(3, 2, 1)
        self.blocks3 = torch.nn.ModuleList([_TokenBlock()])
        self.norm = torch.nn.LayerNorm(2)
        self.calls = []

    def forward(self, image, **_kwargs):
        self.calls.append(image)
        value = self.convolution(image).mean((2, 3)).unsqueeze(1)
        value = self.blocks3[0](value)
        value = self.norm(value)
        return value, None


def test_lora_targets_only_transformer_linears_and_freezes_base():
    encoder = _SharedEncoder()
    names = attach_lora(encoder, rank=2, alpha=4, dropout=0)
    assert set(names) == {"blocks3.0.attn.qkv", "blocks3.0.attn.proj",
                          "blocks3.0.mlp.fc1", "blocks3.0.mlp.fc2"}
    assert isinstance(encoder.blocks3[0].mlp.fc1, LoRALinear)
    assert not encoder.convolution.weight.requires_grad
    assert all(parameter.requires_grad == is_lora_parameter(name)
               for name, parameter in encoder.named_parameters())


def test_visible_and_infrared_pccl_both_gradient_shared_lora():
    encoder = _SharedEncoder()
    attach_lora(encoder, rank=2, alpha=4, dropout=0)
    # Give B a nonzero value so a standalone test has immediate input gradients.
    for name, parameter in encoder.named_parameters():
        if ".lora_B." in name:
            torch.nn.init.constant_(parameter, .1)
    backbone = type("Backbone", (), {"encoder": encoder})()
    ir = encoder(torch.randn(1, 3, 4, 4))[0]
    visible = shared_encoder_tokens(backbone, torch.rand(1, 3, 4, 4))
    anchor = torch.randn_like(ir)
    labels = torch.ones(1, 1, 1)
    loss, loss_ia, loss_va = pccl_objective(anchor, ir, visible, labels)
    assert loss_ia.requires_grad and loss_va.requires_grad
    loss.backward()
    assert len(encoder.calls) == 2
    assert any(parameter.grad is not None for name, parameter in encoder.named_parameters()
               if is_lora_parameter(name))
    assert all(parameter.grad is None for name, parameter in encoder.named_parameters()
               if not is_lora_parameter(name))


def test_lora_optimizer_lr_and_explicit_state_round_trip():
    backbone = _TinyBackbone(freeze=True)
    # Install a transformer-shaped module on the tiny encoder container.
    backbone.encoder = _SharedEncoder()
    attach_lora(backbone.encoder, rank=2, alpha=4, dropout=0)
    projector = torch.nn.Linear(2, 2)
    optimizer = build_pccl_optimizer(backbone, backbone, (projector,), lr=.1,
                                     univ_lr=.002, lora_lr=1e-4)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["lora"]["lr"] == 1e-4
    saved = lora_state_dict(backbone.encoder)
    clone = _SharedEncoder()
    attach_lora(clone, rank=2, alpha=4, dropout=0)
    load_lora_state_dict(clone, saved)
    assert saved
    assert all(torch.equal(value, clone.state_dict()[name]) for name, value in saved.items())
