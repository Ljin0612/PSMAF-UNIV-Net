#!/usr/bin/env python3
"""Evaluate a Stage 7 checkpoint using the IR stream only."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.scripts import eval_univ_mta_fasterrcnn_m3fd as base
import torch
from psmaf_univ.lora_adapter import attach_lora, load_lora_state_dict

_BASE_BUILD_ARG_PARSER = base.build_arg_parser


def build_arg_parser():
    parser = _BASE_BUILD_ARG_PARSER()
    parser.description = __doc__
    parser.set_defaults(image_size=640, output_json=Path("outputs/stage7_eval/metrics.json"), stage_label="7")
    return parser


def main() -> None:
    # Teach the base evaluator how to reconstruct the adapter topology before
    # strict model loading.  Data loading remains the established IR-only path.
    original = base.UNIVMTADetectionBackbone.from_checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        backbone, report = original(checkpoint, **kwargs)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config", payload.get("args", {})) if isinstance(payload, Mapping) else {}
        if config.get("lora_enabled", False):
            names = attach_lora(backbone.encoder, rank=int(config["lora_rank"]),
                                alpha=float(config["lora_alpha"]),
                                dropout=float(config["lora_dropout"]))
            backbone.univ_grad_enabled = True
            backbone.unfrozen_univ_module_names = tuple(names)
            explicit = payload.get("lora_state_dict")
            if explicit is None:
                raise RuntimeError("LoRA checkpoint is missing explicit lora_state_dict")
            load_lora_state_dict(backbone.encoder, explicit)
        return backbone, report

    base.UNIVMTADetectionBackbone.from_checkpoint = from_checkpoint
    base.build_arg_parser = build_arg_parser
    try:
        base.main()
    finally:
        base.UNIVMTADetectionBackbone.from_checkpoint = original


if __name__ == "__main__":
    main()
