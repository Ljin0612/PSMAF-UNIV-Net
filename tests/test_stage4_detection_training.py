"""Synthetic tests for the Stage 4 single-stream detector scaffold."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")

from psmaf_univ.m3fd_detection import M3FDDetectionDataset
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone
from detection.scripts.train_univ_mta_fasterrcnn_m3fd import validate_detection_losses


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
