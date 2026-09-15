"""Synthetic tests for the Stage 4 single-stream detector scaffold."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")

from psmaf_univ.m3fd_detection import M3FDDetectionDataset
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone
from detection.scripts.train_univ_mta_fasterrcnn_m3fd import (
    UNIV_IR_IMAGE_MEAN,
    UNIV_IR_IMAGE_STD,
    build_arg_parser,
    build_detector,
    build_optimizer,
    configure_univ_trainability,
    make_checkpoint_payload,
    resolve_resume_progress,
    training_configuration_summary,
    validate_detection_losses,
)


class TrainabilityUNIV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(2, 2)
        self.blocks2 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Conv2d(3, 4, 1))
        self.blocks3 = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(11)])
        self.norm = torch.nn.LayerNorm(8)


def make_trainability_backbone():
    return UNIVMTADetectionBackbone(TrainabilityUNIV(), freeze_univ=True)


def test_partial_unfreeze_default_keeps_all_univ_frozen():
    backbone = make_trainability_backbone()
    assert configure_univ_trainability(backbone, True) == []
    assert not any(parameter.requires_grad for parameter in backbone.encoder.parameters())


@pytest.mark.parametrize(("count", "expected"), [
    (1, ["blocks3.10"]),
    (2, ["blocks3.9", "blocks3.10"]),
])
def test_partial_unfreeze_selects_only_last_blocks(count, expected):
    backbone = make_trainability_backbone()
    assert configure_univ_trainability(backbone, True, count) == expected
    trainable = [name for name, parameter in backbone.encoder.named_parameters() if parameter.requires_grad]
    assert all(any(name.startswith(module + ".") for module in expected) for name in trainable)
    assert not any(parameter.requires_grad for parameter in backbone.encoder.patch_embed.parameters())


def test_partial_unfreeze_optionally_includes_final_norm():
    backbone = make_trainability_backbone()
    assert configure_univ_trainability(backbone, True, 0, True) == ["norm"]
    assert all(parameter.requires_grad for parameter in backbone.encoder.norm.parameters())


def test_optimizer_and_summary_report_partial_unfreeze_separately():
    from argparse import Namespace

    backbone = make_trainability_backbone()
    names = configure_univ_trainability(backbone, True, 1, True)
    model = torch.nn.Module()
    model.backbone = backbone
    model.head = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, backbone, 0.005, 1e-5)

    assert [group["lr"] for group in optimizer.param_groups] == [0.005, 1e-5]
    assert optimizer.param_groups[1]["name"] == "univ"
    summary = training_configuration_summary(
        Namespace(freeze_univ=True, unfreeze_last_n_blocks=1, unfreeze_norm=True, univ_lr=1e-5),
        model, backbone, names,
    )
    assert set((
        "freeze_univ", "unfreeze_last_n_blocks", "unfreeze_norm", "univ_lr",
        "total_parameter_count", "trainable_parameter_count", "trainable_univ_parameter_count",
        "trainable_adapter_parameter_count", "trainable_detector_parameter_count",
        "unfrozen_univ_module_names",
    )).issubset(summary)
    assert summary["unfrozen_univ_module_names"] == ["blocks3.10", "norm"]


def make_sample(root: Path, label: str, split: str = "smoke_train") -> Path:
    for directory in ("ir", "labels", "meta"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    PIL.fromarray(np.zeros((20, 40, 3), dtype=np.uint8)).save(root / "ir/sample.png")
    (root / "labels/sample.txt").write_text(label, encoding="utf-8")
    (root / "meta" / f"{split}.txt").write_text("sample\n", encoding="utf-8")
    return root


def test_dataset_reads_meta_stem_and_converts_yolo_box(tmp_path):
    dataset = M3FDDetectionDataset(make_sample(tmp_path, "1 0.5 0.5 0.5 0.25\n"))
    image, target = dataset[0]
    assert dataset.stems == ["sample"]
    assert image.shape == (3, 224, 224)
    assert target["boxes"][0].tolist() == pytest.approx([56, 84, 168, 140])
    assert target["labels"].tolist() == [1]
    assert target["area"].tolist() == pytest.approx([6272])


def test_dataset_accepts_m3fd_02639_border_touching_box(tmp_path):
    label = "5 0.8589743589743589 0.5035714285714287 0.2794871794871795 0.992857142857143\n"

    _, target = M3FDDetectionDataset(make_sample(tmp_path, label))[0]

    assert target["labels"].tolist() == [5]
    assert target["boxes"][0].tolist() == pytest.approx(
        [161.1076923076923, 1.6000000000000192, 223.7128205128205, 224.0]
    )


def test_dataset_clamps_tiny_floating_point_boundary_overflow(tmp_path):
    label = "0 0.5 0.5 1.0000002 1.0000002\n"

    _, target = M3FDDetectionDataset(make_sample(tmp_path, label))[0]

    assert target["boxes"][0].tolist() == pytest.approx([0, 0, 224, 224])


def test_dataset_accepts_tiny_boundary_overflow_with_positive_float32_width(tmp_path):
    label = "0 1.00000005 0.5 0.0000002 0.2\n"

    _, target = M3FDDetectionDataset(make_sample(tmp_path, label))[0]

    assert target["boxes"][0, 2] > target["boxes"][0, 0]


@pytest.mark.parametrize(
    "label",
    [
        "0 -0.0000005 0.5 0.0000002 0.2",
        "0 1.0000005 0.5 0.0000002 0.2",
        "0 0.5 -0.0000005 0.2 0.0000002",
        "0 0.5 1.0000005 0.2 0.0000002",
    ],
)
def test_dataset_rejects_box_collapsed_after_clamping(tmp_path, label):
    label_path = tmp_path / "labels" / "sample.txt"

    with pytest.raises(ValueError) as error:
        M3FDDetectionDataset(make_sample(tmp_path, label))[0]

    message = str(error.value)
    assert str(label_path) in message
    assert "label row 1" in message
    assert "original YOLO values=" in message
    assert "computed normalized xyxy before clamping=" in message
    assert "clamped normalized xyxy=" in message
    assert "scaled float32 xyxy=" in message
    assert "degenerate box after float32 scaling/conversion" in message


@pytest.mark.parametrize(
    "label",
    ["0 1.0 0.5 1e-12 0.2", "0 0.5 1.0 0.2 1e-12"],
)
def test_dataset_rejects_box_collapsed_by_float32_conversion(tmp_path, label):
    with pytest.raises(ValueError, match="degenerate box after float32 scaling/conversion") as error:
        M3FDDetectionDataset(make_sample(tmp_path, label))[0]

    message = str(error.value)
    assert "label row 1" in message
    assert "original YOLO values=" in message
    assert "computed normalized xyxy before clamping=" in message
    assert "clamped normalized xyxy=" in message
    assert "scaled float32 xyxy=" in message


@pytest.mark.parametrize(
    "label",
    ["0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 0", "0 0.5 0.5 -0.1 0.2", "0 0.5 0.5 0.2 -0.1"],
)
def test_dataset_rejects_zero_sized_box(tmp_path, label):
    with pytest.raises(ValueError, match="invalid normalized box"):
        M3FDDetectionDataset(make_sample(tmp_path, label))[0]


def test_dataset_rejects_invalid_class_id(tmp_path):
    with pytest.raises(ValueError, match="invalid class ID"):
        M3FDDetectionDataset(make_sample(tmp_path, "6 0.5 0.5 0.2 0.2"))[0]


def test_dataset_handles_empty_label(tmp_path):
    _, target = M3FDDetectionDataset(make_sample(tmp_path, ""))[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["area"].shape == (0,)


@pytest.mark.parametrize(
    ("label", "message"),
    [("6 0.5 0.5 0.2 0.2", "invalid class ID"), ("0 0.05 0.5 0.2 0.2", "invalid normalized box")],
)
def test_dataset_rejects_invalid_annotations(tmp_path, label, message):
    with pytest.raises(ValueError, match=message):
        M3FDDetectionDataset(make_sample(tmp_path, label))[0]


class TinyUNIV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks2 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Conv2d(3, 4, 1))
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, image, mask_ratio=0, return_last_attention=True):
        spatial = self.blocks2(torch.nn.functional.interpolate(image, size=(28, 28)))
        tokens = self.norm(torch.zeros(image.shape[0], 196, 8, device=image.device) + spatial.mean())
        return tokens, None


def test_backbone_returns_ordered_pyramid():
    from collections import OrderedDict

    outputs = UNIVMTADetectionBackbone(TinyUNIV())(torch.zeros(1, 3, 224, 224))
    assert isinstance(outputs, OrderedDict)
    assert list(outputs) == ["P3", "P4", "P5"]
    assert [tuple(value.shape) for value in outputs.values()] == [
        (1, 256, 28, 28), (1, 256, 14, 14), (1, 256, 7, 7)
    ]


def test_detection_loss_validator_accepts_finite_required_losses():
    losses = {name: torch.tensor(1.0, requires_grad=True) for name in (
        "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"
    )}
    total = validate_detection_losses(losses)
    total.backward()
    assert total.item() == 4


def test_detection_loss_validator_rejects_nonfinite_loss():
    losses = {name: torch.tensor(1.0) for name in (
        "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"
    )}
    losses["loss_classifier"] = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="finite scalar"):
        validate_detection_losses(losses)


def test_training_defaults_use_released_splits_and_univ_ir_normalization():
    args = build_arg_parser().parse_args([])

    assert args.split == "train"
    assert args.val_split == "val"
    assert args.image_mean == list(UNIV_IR_IMAGE_MEAN)
    assert args.image_std == list(UNIV_IR_IMAGE_STD)
    assert args.save_interval_steps == 1000


def test_save_interval_steps_parser():
    assert build_arg_parser().parse_args(["--save-interval-steps", "17"]).save_interval_steps == 17


def test_recovery_checkpoint_contains_new_and_legacy_keys():
    from argparse import Namespace

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = Namespace(image_size=320, checkpoint_key="teacher", freeze_univ=False,
                     output_dir=Path("out"))
    payload = make_checkpoint_payload(
        model, optimizer, args, epoch=2, completed_epochs=1, global_step=12,
        optimizer_steps=11, steps_in_current_epoch=4,
    )

    assert payload["model_state_dict"] is payload["model"]
    assert payload["optimizer_state_dict"] is payload["optimizer"]
    assert payload["batch_index_in_epoch"] == payload["steps_in_current_epoch"] == 4
    assert payload["global_step"] == 12
    assert payload["optimizer_steps"] == 11
    assert payload["image_size"] == 320
    assert payload["checkpoint_key"] == "teacher"
    assert payload["freeze_univ"] is False
    assert payload["config"]["output_dir"] == "out"


def test_legacy_epoch_boundary_checkpoint_can_resume():
    progress = resolve_resume_progress(
        {"completed_epochs": 2, "optimizer_steps": 10, "global_step": 10},
        steps_per_epoch=5,
    )

    assert progress["steps_in_current_epoch"] == 0
    assert progress["resume_mode"] == "legacy_epoch_boundary_resume"
    assert progress["legacy_resume_allowed"] is True
    assert progress["resume_warning"]


def test_counterless_model_only_checkpoint_is_rejected_for_resume():
    with pytest.raises(ValueError, match=(
        "Cannot safely resume checkpoint without progress counters. Please use a checkpoint "
        "saved by the training script with progress metadata, or start a new training run."
    )):
        resolve_resume_progress({"model": {"weight": torch.ones(1)}}, steps_per_epoch=5)


def test_legacy_completed_epoch_counter_alone_can_resume():
    progress = resolve_resume_progress({"completed_epochs": 2}, steps_per_epoch=5)

    assert progress["global_step"] == 10
    assert progress["optimizer_steps"] == 10
    assert progress["completed_epochs"] == 2
    assert progress["resume_mode"] == "legacy_epoch_boundary_resume"


def test_legacy_partial_epoch_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="Cannot safely resume legacy partial-epoch checkpoint"):
        resolve_resume_progress(
            {"completed_epochs": 2, "optimizer_steps": 12, "global_step": 12},
            steps_per_epoch=5,
        )


def test_new_partial_epoch_checkpoint_can_resume():
    progress = resolve_resume_progress(
        {"completed_epochs": 2, "optimizer_steps": 12, "steps_in_current_epoch": 2},
        steps_per_epoch=5,
    )

    assert progress["steps_in_current_epoch"] == 2
    assert progress["resume_mode"] == "partial_epoch_resume"
    assert progress["legacy_resume_allowed"] is None


def test_fasterrcnn_receives_univ_ir_normalization():
    pytest.importorskip("torchvision")
    backbone = torch.nn.Conv2d(3, 256, 1)
    backbone.out_channels = 256

    detector = build_detector(backbone)

    assert detector.transform.image_mean == list(UNIV_IR_IMAGE_MEAN)
    assert detector.transform.image_std == list(UNIV_IR_IMAGE_STD)
