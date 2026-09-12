#!/usr/bin/env python3
"""Evaluate a trained Stage 4 M3FD-IR Faster R-CNN checkpoint with COCO-style AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from detection.scripts.train_univ_mta_fasterrcnn_m3fd import build_detector
from psmaf_univ.m3fd_detection import M3FD_CLASS_NAMES, M3FDDetectionDataset, detection_collate_fn
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone

IOU_THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))


def prediction_to_evaluation(
    prediction: Mapping[str, torch.Tensor], score_threshold: float = 0.0
) -> dict:
    """Detach a torchvision prediction and change foreground labels 1..6 to M3FD 0..5."""
    boxes = prediction["boxes"].detach().cpu().to(torch.float32)
    scores = prediction["scores"].detach().cpu().to(torch.float32)
    labels = prediction["labels"].detach().cpu().to(torch.int64) - 1
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if labels.numel() and (labels.min() < 0 or labels.max() >= len(M3FD_CLASS_NAMES)):
        raise ValueError("detector prediction contains a label outside foreground IDs 1..6")
    return {"boxes": boxes, "scores": scores, "labels": labels}


def target_to_evaluation(target: Mapping[str, torch.Tensor]) -> dict:
    """Extract CPU xyxy boxes and the loader's native M3FD class IDs."""
    return {
        "boxes": target["boxes"].detach().cpu().to(torch.float32),
        "labels": target["labels"].detach().cpu().to(torch.int64),
    }


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (box_area + areas - intersection).clamp(min=torch.finfo(torch.float32).eps)


def _class_ap(predictions: Sequence[dict], targets: Sequence[dict], class_id: int, iou: float):
    ground_truth = {}
    count = 0
    for image_id, target in enumerate(targets):
        boxes = target["boxes"][target["labels"] == class_id]
        ground_truth[image_id] = boxes
        count += len(boxes)
    if count == 0:
        return None

    detections = []
    for image_id, prediction in enumerate(predictions):
        selected = prediction["labels"] == class_id
        for box, score in zip(prediction["boxes"][selected], prediction["scores"][selected]):
            detections.append((float(score), image_id, box))
    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in ground_truth.items()
    }
    true_positives, false_positives = [], []
    for _score, image_id, box in detections:
        overlaps = _box_iou(box, ground_truth[image_id])
        if overlaps.numel():
            best_overlap, best_index = overlaps.max(dim=0)
            is_match = float(best_overlap) >= iou and not matched[image_id][best_index]
        else:
            is_match = False
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))
        if is_match:
            matched[image_id][best_index] = True
    if not detections:
        return 0.0
    tp = torch.tensor(true_positives).cumsum(0)
    fp = torch.tensor(false_positives).cumsum(0)
    recall = tp / count
    precision = tp / (tp + fp)
    # COCO's 101-point interpolated precision (area/max-detection ranges omitted).
    samples = torch.linspace(0, 1, 101)
    interpolated = [
        precision[recall >= level].max() if torch.any(recall >= level) else 0.0
        for level in samples
    ]
    return float(torch.as_tensor(interpolated).mean())


def compute_map(predictions: Sequence[dict], targets: Sequence[dict]) -> dict:
    """Compute COCO-style IoU-averaged AP for M3FD classes, including empty inputs."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of images")
    table = {
        threshold: [
            _class_ap(predictions, targets, class_id, threshold)
            for class_id in range(len(M3FD_CLASS_NAMES))
        ]
        for threshold in IOU_THRESHOLDS
    }
    def mean_available(values):
        available = [value for value in values if value is not None]
        return sum(available) / len(available) if available else 0.0

    ap50 = table[0.5]
    return {
        "mAP50": mean_available(ap50),
        "mAP50_95": mean_available([value for values in table.values() for value in values]),
        "AP75": mean_available(table[0.75]),
        "AP50_per_class": {name: ap50[index] for index, name in enumerate(M3FD_CLASS_NAMES)},
        "num_images": len(targets),
        "num_ground_truth_boxes": sum(len(target["boxes"]) for target in targets),
        "num_predicted_boxes": sum(len(prediction["boxes"]) for prediction in predictions),
    }


def _checkpoint_state(payload, checkpoint_key: str):
    """Accept standard Stage 4 payloads and optional student/teacher detector branches."""
    candidate = payload
    if (
        isinstance(candidate, Mapping)
        and checkpoint_key in candidate
        and isinstance(candidate[checkpoint_key], Mapping)
    ):
        candidate = candidate[checkpoint_key]
    if isinstance(candidate, Mapping) and "model" in candidate and isinstance(candidate["model"], Mapping):
        candidate = candidate["model"]
    if (
        not isinstance(candidate, Mapping)
        or not candidate
        or not all(torch.is_tensor(value) for value in candidate.values())
    ):
        raise ValueError("trained checkpoint does not contain a model state dictionary")
    return candidate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jinlei/database/M3FD_Detection"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained Stage 4 detector checkpoint")
    parser.add_argument(
        "--univ-checkpoint",
        type=Path,
        default=Path("/home/jinlei/checkpoints/UNIV/checkpoint0400.pth"),
    )
    parser.add_argument("--checkpoint-key", choices=("student", "teacher"), default="student")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/stage4_eval/metrics.json"))
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.image_size != 224:
        raise SystemExit("original UNIV currently requires --image-size 224")
    if args.batch_size < 1 or not 0 <= args.score_threshold <= 1:
        raise SystemExit("batch-size must be positive and score-threshold must be in [0, 1]")
    device = torch.device(args.device)
    dataset = M3FDDetectionDataset(args.data_root, args.split, args.image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn)
    backbone, _ = UNIVMTADetectionBackbone.from_checkpoint(
        args.univ_checkpoint, checkpoint_key=args.checkpoint_key, source_root=args.source_root, freeze_univ=True
    )
    model = build_detector(backbone, args.image_size).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_checkpoint_state(payload, args.checkpoint_key), strict=True)
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(prediction_to_evaluation(item, args.score_threshold) for item in outputs)
            targets.extend(target_to_evaluation(item) for item in batch_targets)
    results = compute_map(predictions, targets)
    results.update({"stage": "4.3", "dataset": "M3FD-IR", "split": args.split,
                    "checkpoint_key": args.checkpoint_key, "score_threshold": args.score_threshold})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
