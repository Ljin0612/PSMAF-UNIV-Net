"""Validation helpers and a torchvision-style M3FD detection dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any


M3FD_CLASS_NAMES = ("people", "car", "bus", "motorcycle", "lamp", "truck")
SPLITS = ("train", "val", "test")
DETECTION_SPLITS = (*SPLITS, "smoke_train", "smoke_val", "smoke_test")
YOLO_BOUNDARY_TOLERANCE = 1e-6


def yolo_labels_to_target(rows: list[list[float]], width: int, height: int):
    """Convert validated normalized YOLO rows to a torchvision target payload."""
    import torch

    boxes, labels = [], []
    for line_number, row in enumerate(rows, 1):
        if len(row) != 5:
            raise ValueError(f"label row {line_number} must contain 5 values; got {len(row)}")
        class_value, cx, cy, box_width, box_height = row
        if not class_value.is_integer() or not 0 <= int(class_value) < len(M3FD_CLASS_NAMES):
            raise ValueError(f"invalid class ID {class_value:g} on label row {line_number}")
        values = (cx, cy, box_width, box_height)
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
            raise ValueError(f"invalid non-finite box coordinates on label row {line_number}")
        x1, y1 = cx - box_width / 2, cy - box_height / 2
        x2, y2 = cx + box_width / 2, cy + box_height / 2
        if (
            box_width <= 0
            or box_height <= 0
            or x1 < -YOLO_BOUNDARY_TOLERANCE
            or y1 < -YOLO_BOUNDARY_TOLERANCE
            or x2 > 1 + YOLO_BOUNDARY_TOLERANCE
            or y2 > 1 + YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError(
                f"invalid normalized box coordinates on label row {line_number}: "
                f"cx={cx:g}, cy={cy:g}, w={box_width:g}, h={box_height:g}"
            )
        normalized_xyxy = (x1, y1, x2, y2)
        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))
        clamped_xyxy = (x1, y1, x2, y2)
        scaled_xyxy = torch.tensor(
            [x1 * width, y1 * height, x2 * width, y2 * height], dtype=torch.float32
        )
        if scaled_xyxy[2] <= scaled_xyxy[0] or scaled_xyxy[3] <= scaled_xyxy[1]:
            raise ValueError(
                f"degenerate box after float32 scaling/conversion on label row {line_number}: "
                f"original YOLO values=(class_id={class_value:g}, cx={cx:g}, cy={cy:g}, "
                f"w={box_width:g}, h={box_height:g}), "
                f"computed normalized xyxy before clamping={normalized_xyxy}, "
                f"clamped normalized xyxy={clamped_xyxy}, "
                f"scaled float32 xyxy={scaled_xyxy.tolist()}"
            )
        boxes.append(scaled_xyxy)
        labels.append(int(class_value))
    box_tensor = torch.stack(boxes) if boxes else torch.empty((0, 4), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    return box_tensor, label_tensor


class M3FDDetectionDataset:
    """Load one IR stream and YOLO annotations from the released server layout."""

    def __init__(self, root: str | Path, split: str = "smoke_train", image_size: int = 224) -> None:
        if split not in DETECTION_SPLITS:
            raise ValueError(f"unsupported M3FD split {split!r}; expected one of {DETECTION_SPLITS}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = image_size
        split_file = self.root / "meta" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"M3FD split file does not exist: {split_file}")
        self.stems = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.stems:
            raise ValueError(f"M3FD split contains no samples: {split_file}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image

        entry = Path(self.stems[index])
        stem = entry.stem if entry.suffix else entry.name
        image_path = self.root / "ir" / f"{stem}.png"
        label_path = self.root / "labels" / f"{stem}.txt"
        if not image_path.is_file():
            raise FileNotFoundError(f"M3FD IR image does not exist: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"M3FD label does not exist: {label_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((self.image_size, self.image_size))
            image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)
        rows = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError as error:
                raise ValueError(f"invalid numeric label at {label_path}:{line_number}") from error
        try:
            boxes, labels = yolo_labels_to_target(rows, self.image_size, self.image_size)
        except ValueError as error:
            raise ValueError(f"invalid M3FD label {label_path}: {error}") from error
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(labels.shape[0], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    """Keep variable-length detection targets as a list."""
    return tuple(zip(*batch))


def _find_directory(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _read_classes(root: Path) -> tuple[list[str] | None, Path | None]:
    for name in ("classes.txt", "class_names.txt", "labels/classes.txt"):
        path = root / name
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], path
    return None, None


def inspect_m3fd_detection(root: str | Path) -> dict[str, Any]:
    """Inspect M3FD without modifying it and return a JSON-serializable report.

    Split entries may be image paths or bare sample identifiers. Labels are
    expected to use the same relative stem and a ``.txt`` extension.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"M3FD dataset root does not exist or is not a directory: {root}")

    ir_dir = _find_directory(root, ("Ir", "IR", "ir", "images/ir", "images/IR"))
    visible_dir = _find_directory(root, ("vi", "Vis", "VIS", "visible", "images/visible", "images/vis"))
    label_dir = _find_directory(root, ("labels", "Labels", "annotations"))
    missing_required = []
    if ir_dir is None:
        missing_required.append("IR image directory")
    if label_dir is None:
        missing_required.append("label directory")

    split_files: dict[str, Path | None] = {}
    for split in SPLITS:
        split_files[split] = next(
            (
                path
                for path in (
                    root / "meta" / f"{split}.txt",
                    root / f"{split}.txt",
                    root / "splits" / f"{split}.txt",
                    root / "ImageSets" / f"{split}.txt",
                )
                if path.is_file()
            ),
            None,
        )
        if split_files[split] is None:
            missing_required.append(f"{split} split file")

    classes, class_file = _read_classes(root)
    if classes is None:
        # The released M3FD_Detection tree does not necessarily include a
        # classes.txt file. Its YOLO labels use this canonical class order.
        classes = list(M3FD_CLASS_NAMES)
    class_names_correct = classes == list(M3FD_CLASS_NAMES)
    if not class_names_correct:
        missing_required.append("canonical class names/order")

    split_reports: dict[str, Any] = {}
    if ir_dir is not None and label_dir is not None:
        image_suffixes = (".png", ".jpg", ".jpeg", ".bmp")
        for split, split_file in split_files.items():
            if split_file is None:
                continue
            missing_images, missing_labels, empty_labels, matched = [], [], [], 0
            entries = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            for entry in entries:
                relative = Path(entry)
                stem = relative.stem if relative.suffix.lower() in image_suffixes else relative.name
                candidates = ([root / relative] if relative.suffix.lower() in image_suffixes else []) + [
                    ir_dir / f"{stem}{suffix}" for suffix in image_suffixes
                ]
                image = next((path for path in candidates if path.is_file()), None)
                label = label_dir / f"{stem}.txt"
                if image is None:
                    missing_images.append(entry)
                if not label.is_file():
                    missing_labels.append(entry)
                elif label.stat().st_size == 0 or not label.read_text(encoding="utf-8").strip():
                    empty_labels.append(entry)
                if image is not None and label.is_file():
                    matched += 1
            split_reports[split] = {
                "entries": len(entries), "matched_pairs": matched,
                "missing_images": missing_images, "missing_labels": missing_labels,
                "empty_labels": empty_labels,
            }

    return {
        "dataset_root": str(root), "stage3_modality": "IR",
        "ir_image_directory": str(ir_dir) if ir_dir else None,
        "visible_image_directory": str(visible_dir) if visible_dir else None,
        "visible_available": visible_dir is not None,
        "label_directory": str(label_dir) if label_dir else None,
        "split_files": {key: str(value) if value else None for key, value in split_files.items()},
        "expected_class_names": list(M3FD_CLASS_NAMES), "class_names": classes,
        "class_names_file": str(class_file) if class_file else None,
        "class_names_correct": class_names_correct, "splits": split_reports,
        "errors": missing_required, "passed": not missing_required and all(
            not data[issue] for data in split_reports.values()
            for issue in ("missing_images", "missing_labels", "empty_labels")
        ),
    }
