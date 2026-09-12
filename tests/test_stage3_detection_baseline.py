"""Tests for the Stage 3 read-only dataset and detector boundary scaffolds."""
from pathlib import Path

import pytest

from psmaf_univ.m3fd_detection import M3FD_CLASS_NAMES, inspect_m3fd_detection


def make_dataset(root: Path) -> Path:
    (root / "Ir").mkdir(parents=True)
    (root / "Vis").mkdir()
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("\n".join(M3FD_CLASS_NAMES) + "\n")
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text(f"{split}_sample\n")
        (root / "Ir" / f"{split}_sample.png").write_bytes(b"synthetic")
        (root / "labels" / f"{split}_sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def test_m3fd_path_checker_accepts_synthetic_tree(tmp_path):
    report = inspect_m3fd_detection(make_dataset(tmp_path / "M3FD"))
    assert report["passed"] is True
    assert report["class_names"] == list(M3FD_CLASS_NAMES)
    assert report["visible_available"] is True
    assert all(report["splits"][split]["matched_pairs"] == 1 for split in ("train", "val", "test"))


def test_m3fd_checker_accepts_server_layout_without_classes_file(tmp_path):
    root = tmp_path / "M3FD_Detection"
    for directory in ("Annotation", "ir", "labels", "meta", "vi"):
        (root / directory).mkdir(parents=True)

    for split, stem in (("train", "03905"), ("val", "02579"), ("test", "04107")):
        (root / "meta" / f"{split}.txt").write_text(f"{stem}\n", encoding="utf-8")
        (root / "ir" / f"{stem}.png").write_bytes(b"synthetic")
        (root / "vi" / f"{stem}.png").write_bytes(b"synthetic")
        (root / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    report = inspect_m3fd_detection(root)

    assert report["passed"] is True
    assert report["class_names"] == list(M3FD_CLASS_NAMES)
    assert report["class_names_file"] is None
    assert report["class_names_correct"] is True
    assert report["visible_image_directory"] == str((root / "vi").resolve())
    assert report["split_files"]["train"] == str((root / "meta/train.txt").resolve())
    assert all(report["splits"][split]["matched_pairs"] == 1 for split in ("train", "val", "test"))


def test_m3fd_checker_reports_empty_and_missing_labels(tmp_path):
    root = make_dataset(tmp_path / "M3FD")
    (root / "labels/train_sample.txt").write_text("")
    (root / "labels/val_sample.txt").unlink()
    report = inspect_m3fd_detection(root)
    assert report["splits"]["train"]["empty_labels"] == ["train_sample"]
    assert report["splits"]["val"]["missing_labels"] == ["val_sample"]
    assert report["passed"] is False


def test_missing_dataset_path_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="M3FD dataset root does not exist"):
        inspect_m3fd_detection(tmp_path / "missing")


torch = pytest.importorskip("torch")
from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter
from tools.inspect_univ_mta_detection_smoke import require_file, validate_detection_features


def test_detection_smoke_shape_validator_and_feature_dictionary():
    adapter = MultiScaleTaskAdapter(4, 8, 256)
    features = adapter(torch.randn(1, 4, 28, 28), torch.randn(1, 196, 8), grid_size=(14, 14))
    assert list(features) == ["P3", "P4", "P5"]
    assert validate_detection_features(features)["compatible"] is True


def test_detection_shape_validator_rejects_wrong_shape():
    features = {"P3": torch.zeros(1, 256, 27, 28), "P4": torch.zeros(1, 256, 14, 14), "P5": torch.zeros(1, 256, 7, 7)}
    with pytest.raises(ValueError, match="P3 shape"):
        validate_detection_features(features)


def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="UNIV checkpoint does not exist"):
        require_file(tmp_path / "missing.pth", "UNIV checkpoint")
