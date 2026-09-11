#!/usr/bin/env python3
"""Run the Stage 3 single-stream UNIV/MTA detector-boundary smoke test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor

from psmaf_univ.checkpoint_loader import load_univ_checkpoint
from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter
from psmaf_univ.univ_diagnostics import build_original_univ, tensor_summary
from tools.inspect_univ_mta_features import validate_checkpoint_load_report


EXPECTED_SHAPES = {"P3": (1, 256, 28, 28), "P4": (1, 256, 14, 14), "P5": (1, 256, 7, 7)}


def require_file(path: str | Path, description: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {path}")
    return path


def validate_detection_features(features: dict[str, Tensor], expected=EXPECTED_SHAPES) -> dict:
    """Minimal head placeholder: enforce pyramid keys, shapes, and finite values."""
    if set(features) != set(expected):
        raise ValueError(f"detection features must have exactly {list(expected)}; got {list(features)}")
    for name, shape in expected.items():
        if tuple(features[name].shape) != tuple(shape):
            raise ValueError(f"{name} shape must be {tuple(shape)}; got {tuple(features[name].shape)}")
        if not torch.isfinite(features[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    return {"type": "shape_validator", "compatible": True, "features": {k: list(v.shape) for k, v in features.items()}}


def _load_image(path: Path | None, device: str) -> tuple[Tensor, str]:
    if path is None:
        return torch.zeros(1, 3, 224, 224, device=device), "dummy"
    require_file(path, "input image")
    from PIL import Image
    import numpy as np
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device), str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", choices=("student", "teacher"), default="student")
    parser.add_argument("--image", type=Path, help="Optional real M3FD IR image")
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-load-fraction", type=float, default=0.5)
    args = parser.parse_args()
    require_file(args.checkpoint, "UNIV checkpoint")
    model = build_original_univ(args.source_root).to(args.device).eval()
    checkpoint = load_univ_checkpoint(model, args.checkpoint, checkpoint_key=args.checkpoint_key)
    checkpoint_validation = validate_checkpoint_load_report(checkpoint, args.checkpoint_key, args.min_load_fraction)
    image, image_source = _load_image(args.image, args.device)
    captured = {}
    modules = dict(model.named_modules())
    missing = [name for name in ("blocks2.1", "norm") if name not in modules]
    if missing:
        raise ValueError(f"UNIV model is missing required feature modules: {', '.join(missing)}")
    handles = [modules[name].register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out)) for name in ("blocks2.1", "norm")]
    try:
        with torch.inference_mode():
            model(image, mask_ratio=0, return_last_attention=True)
    finally:
        for handle in handles:
            handle.remove()
    adapter = MultiScaleTaskAdapter(captured["blocks2.1"].shape[1], captured["norm"].shape[2], 256).to(args.device).eval()
    with torch.inference_mode():
        features = adapter(captured["blocks2.1"], captured["norm"], grid_size=(14, 14))
    report = {"stage": 3, "stream": "IR", "input_source": image_source,
              "checkpoint": asdict(checkpoint), "checkpoint_validation": checkpoint_validation,
              "selected_features": {k: tensor_summary(v) for k, v in captured.items()},
              "adapter_outputs": {k: tensor_summary(v) for k, v in features.items()},
              "detection_head": validate_detection_features(features)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
