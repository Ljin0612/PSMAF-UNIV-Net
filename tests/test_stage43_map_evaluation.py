"""Unit tests for Stage 4.3 prediction conversion and internal COCO-style AP."""

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import (
    build_arg_parser,
    build_evaluation_detector,
    compute_map,
    evaluation_metadata,
    prediction_to_evaluation,
    target_to_evaluation,
)
from detection.scripts.train_univ_mta_fasterrcnn_m3fd import (
    UNIV_IR_IMAGE_MEAN,
    UNIV_IR_IMAGE_STD,
)
from psmaf_univ.m3fd_detection import M3FD_CLASS_NAMES


def prediction(boxes=(), scores=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def target(boxes=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def test_prediction_and_target_conversion_use_m3fd_zero_based_labels():
    converted = prediction_to_evaluation(
        prediction([[0, 0, 10, 10], [1, 1, 2, 2]], [0.9, 0.1], [1, 6]), 0.5)
    converted_target = target_to_evaluation(target([[0, 0, 10, 10]], [0]))
    assert converted["labels"].tolist() == [0]
    assert converted["boxes"].dtype == torch.float32
    assert converted_target["labels"].tolist() == [0]


def test_postprocess_honors_requested_score_threshold_including_below_point_zero_five():
    raw = prediction([[0, 0, 1, 1], [1, 1, 2, 2]], [0.01, 0.1], [1, 1])
    assert prediction_to_evaluation(raw, 0.0)["scores"].tolist() == pytest.approx([0.01, 0.1])
    assert prediction_to_evaluation(raw, 0.05)["scores"].tolist() == pytest.approx([0.1])


def test_evaluator_defaults_to_univ_ir_normalization():
    args = build_arg_parser().parse_args(["--checkpoint", "detector.pth"])
    assert args.image_mean == list(UNIV_IR_IMAGE_MEAN)
    assert args.image_std == list(UNIV_IR_IMAGE_STD)


def test_evaluation_detector_passes_custom_normalization_and_zero_internal_threshold(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_build_detector(backbone, image_size, **kwargs):
        captured.update(backbone=backbone, image_size=image_size, **kwargs)
        return sentinel

    monkeypatch.setattr(
        "detection.scripts.eval_univ_mta_fasterrcnn_m3fd.build_detector",
        fake_build_detector,
    )
    args = build_arg_parser().parse_args([
        "--checkpoint", "detector.pth",
        "--image-mean", "0.1", "0.2", "0.3",
        "--image-std", "0.4", "0.5", "0.6",
    ])
    assert build_evaluation_detector("backbone", args) is sentinel
    assert captured == {
        "backbone": "backbone",
        "image_size": 224,
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.4, 0.5, 0.6],
        "box_score_thresh": 0.0,
    }


def test_evaluation_json_metadata_describes_explicit_postprocess_filtering():
    args = Namespace(score_threshold=0.0, image_mean=[0.1] * 3, image_std=[0.2] * 3)
    model = Namespace(roi_heads=Namespace(score_thresh=0.0))
    metadata = evaluation_metadata(args, model)
    assert metadata == {
        "requested_score_threshold": 0.0,
        "internal_box_score_thresh": 0.0,
        "score_threshold_applied_stage": "evaluation_script",
        "image_mean": [0.1, 0.1, 0.1],
        "image_std": [0.2, 0.2, 0.2],
    }


def test_perfect_prediction_metric_schema_and_class_names():
    metrics = compute_map(
        [prediction_to_evaluation(prediction([[0, 0, 10, 10]], [0.9], [1]))],
        [target([[0, 0, 10, 10]], [0])])
    assert set(metrics) == {"mAP50", "mAP50_95", "AP75", "AP50_per_class", "num_images",
                            "num_ground_truth_boxes", "num_predicted_boxes"}
    assert list(metrics["AP50_per_class"]) == list(M3FD_CLASS_NAMES)
    assert metrics["mAP50"] == pytest.approx(1)
    assert metrics["mAP50_95"] == pytest.approx(1)
    assert metrics["AP75"] == pytest.approx(1)
    assert metrics["AP50_per_class"]["people"] == pytest.approx(1)
    assert metrics["AP50_per_class"]["car"] is None


def test_empty_predictions_score_zero_when_ground_truth_exists():
    metrics = compute_map([prediction()], [target([[0, 0, 10, 10]], [0])])
    assert metrics["mAP50"] == 0
    assert metrics["num_predicted_boxes"] == 0
    assert metrics["num_ground_truth_boxes"] == 1


def test_empty_targets_are_supported_and_not_counted_as_a_class():
    metrics = compute_map([prediction([[0, 0, 10, 10]], [0.9], [1])], [target()])
    assert metrics["mAP50"] == 0
    assert metrics["mAP50_95"] == 0
    assert metrics["num_images"] == 1
    assert metrics["num_ground_truth_boxes"] == 0
    assert metrics["num_predicted_boxes"] == 1
    assert all(value is None for value in metrics["AP50_per_class"].values())
