#!/usr/bin/env python3
"""Evaluate a trained Stage 7 checkpoint using the IR stream only.

``--checkpoint`` is the trained Stage 7 detector while ``--univ-checkpoint``
is only the original UNIV initialization checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from detection.scripts import eval_univ_mta_fasterrcnn_m3fd as base
from psmaf_univ.lora_adapter import attach_lora, load_lora_state_dict, lora_state_dict
from psmaf_univ.m3fd_detection import M3FDDetectionDataset, detection_collate_fn
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(image_size=640, output_json=Path("outputs/stage7_eval/metrics.json"),
                        stage_label="7")
    for action in parser._actions:
        if action.dest == "checkpoint":
            action.help = "Trained Stage 7 detector/LoRA checkpoint"
        elif action.dest == "univ_checkpoint":
            action.help = "Original UNIV pretrained initialization checkpoint"
    return parser


def load_stage7_payload(checkpoint: Path) -> tuple[Mapping, Mapping]:
    """Load the trained payload first and return its required saved configuration."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage 7 checkpoint must contain a mapping payload")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage 7 checkpoint is missing its config mapping")
    missing = [name for name in ("lora_enabled", "checkpoint_key", "image_size")
               if name not in config]
    if missing:
        raise ValueError(f"Stage 7 checkpoint config is missing: {', '.join(missing)}")
    return payload, config


def build_stage7_evaluation_detector(args):
    """Reconstruct the saved Stage 7 topology, then strictly restore its state."""
    payload, config = load_stage7_payload(args.checkpoint)

    # Saved architecture choices win over evaluator defaults.  The original
    # UNIV checkpoint is deliberately never consulted for Stage 7 metadata.
    args.checkpoint_key = config["checkpoint_key"]
    args.image_size = config["image_size"]
    backbone, load_report = UNIVMTADetectionBackbone.from_checkpoint(
        args.univ_checkpoint, checkpoint_key=args.checkpoint_key,
        source_root=args.source_root, freeze_univ=True, image_size=args.image_size,
    )

    if config["lora_enabled"]:
        required = ("lora_rank", "lora_alpha", "lora_dropout")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"Stage 7 LoRA config is missing: {', '.join(missing)}")
        explicit = payload.get("lora_state_dict")
        if not isinstance(explicit, Mapping) or not explicit:
            raise RuntimeError(
                "Stage 7 checkpoint has lora_enabled=true but is missing lora_state_dict"
            )
        names = attach_lora(backbone.encoder, rank=int(config["lora_rank"]),
                            alpha=float(config["lora_alpha"]),
                            dropout=float(config["lora_dropout"]))
        backbone.univ_grad_enabled = True
        backbone.unfrozen_univ_module_names = tuple(names)
        expected_keys = set(lora_state_dict(backbone.encoder))
        if set(explicit) != expected_keys:
            missing_keys = sorted(expected_keys - set(explicit))
            unexpected_keys = sorted(set(explicit) - expected_keys)
            raise RuntimeError(
                "Stage 7 lora_state_dict does not match the reconstructed topology; "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )

    model = base.build_evaluation_detector(backbone, args)
    model.load_state_dict(base._checkpoint_state(payload, args.checkpoint_key), strict=True)
    if config["lora_enabled"]:
        # Restore the separately audited adapter copy as well as the copy in the
        # complete detector state.  This also validates every explicit key.
        load_lora_state_dict(backbone.encoder, payload["lora_state_dict"])
    return model, load_report, payload, config


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        base.validate_normalization_args(args.image_mean, args.image_std)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or not 0 <= args.score_threshold <= 1:
        raise SystemExit("batch-size must be positive and score-threshold must be in [0, 1]")

    # Load/build before creating the dataset so checkpoint architecture metadata
    # (notably image_size) controls the IR-only evaluation input pipeline.
    model, load_report, _payload, _config = build_stage7_evaluation_detector(args)
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = M3FDDetectionDataset(args.data_root, args.split, args.image_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
    )
    predictions, targets = [], []
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(base.prediction_to_evaluation(item, args.score_threshold)
                               for item in outputs)
            targets.extend(base.target_to_evaluation(item) for item in batch_targets)
    results = base.compute_map(predictions, targets)
    results.update({"stage": "7", "dataset": "M3FD-IR", "split": args.split,
                    "checkpoint_key": args.checkpoint_key,
                    "score_threshold": args.score_threshold})
    results.update(base.evaluation_metadata(args, model, load_report))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
