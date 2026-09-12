"""Read-only validation helpers for the M3FD detection dataset layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any


M3FD_CLASS_NAMES = ("people", "car", "bus", "motorcycle", "lamp", "truck")
SPLITS = ("train", "val", "test")


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
