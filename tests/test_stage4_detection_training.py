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
    validate_detection_losses,
)


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
    assert "degenerate box after clamping" in message


@pytest.mark.parametrize("label", ["0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 0"])
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


def test_fasterrcnn_receives_univ_ir_normalization():
    pytest.importorskip("torchvision")
    backbone = torch.nn.Conv2d(3, 256, 1)
    backbone.out_channels = 256

    detector = build_detector(backbone)

    assert detector.transform.image_mean == list(UNIV_IR_IMAGE_MEAN)
    assert detector.transform.image_std == list(UNIV_IR_IMAGE_STD)
