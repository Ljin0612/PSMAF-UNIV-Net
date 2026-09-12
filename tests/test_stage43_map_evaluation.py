"""Unit tests for Stage 4.3 prediction conversion and internal COCO-style AP."""

import pytest

torch = pytest.importorskip("torch")

from detection.scripts.eval_univ_mta_fasterrcnn_m3fd import (
    compute_map,
    prediction_to_evaluation,
    target_to_evaluation,
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
