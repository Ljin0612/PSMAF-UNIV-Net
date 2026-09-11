#!/usr/bin/env python3
"""Validate the UNIV-to-Multi-scale Task Adapter P3/P4/P5 boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from psmaf_univ.checkpoint_loader import load_univ_checkpoint
from psmaf_univ.multiscale_task_adapter import MultiScaleTaskAdapter
from psmaf_univ.univ_diagnostics import build_original_univ, tensor_summary


def validate_checkpoint_load_report(
    report: Any, checkpoint_key: str | None, min_load_fraction: float
) -> dict[str, Any]:
    """Reject checkpoint reports that cannot meaningfully initialize UNIV."""
    if not 0.0 <= min_load_fraction <= 1.0:
        raise ValueError("min_load_fraction must be between 0 and 1")

    def value(name: str, default: Any = None) -> Any:
        if isinstance(report, Mapping):
            return report.get(name, default)
        return getattr(report, name, default)

    load_fraction = float(value("load_fraction", 0.0))
    loaded_key_count = int(value("loaded_key_count", 0))
    loaded_parameter_count = value("loaded_parameter_count")
    reasons = []
    if loaded_key_count == 0:
        reasons.append("loaded_key_count is zero")
    if load_fraction <= 0:
        reasons.append("load_fraction is zero")
    if loaded_parameter_count is not None and int(loaded_parameter_count) == 0:
        reasons.append("loaded_parameter_count is zero")
    if load_fraction < min_load_fraction:
        reasons.append(
            f"load_fraction {load_fraction:.6g} is below the required minimum "
            f"{min_load_fraction:.6g}"
        )
    if reasons:
        branch = checkpoint_key if checkpoint_key is not None else "<root>"
        raise RuntimeError(
            f"Checkpoint load failed: no usable weights were loaded from branch {branch!r} "
            f"({'; '.join(reasons)}). Refusing to run MTA smoke test with randomly "
            "initialized UNIV weights."
        )

    validation = {
        "passed": True,
        "min_load_fraction": min_load_fraction,
        "load_fraction": load_fraction,
        "loaded_key_count": loaded_key_count,
        "model_state_key_count": int(value("model_state_key_count", 0)),
    }
    if loaded_parameter_count is not None:
        validation["loaded_parameter_count"] = int(loaded_parameter_count)
    return validation


def inspect_univ_mta_features(
    model: torch.nn.Module,
    *,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 1,
    device: str = "cpu",
    adapter_out_channels: int = 256,
) -> tuple[dict, list[str]]:
    """Run UNIV and its first-version adapter policy with explicit token layout."""
    if image_size != (224, 224):
        raise ValueError("the original UNIV Stage 2 smoke test currently supports --image-size 224 224 only")
    modules = dict(model.named_modules())
    required = ("blocks2.1", "norm")
    missing = [name for name in required if name not in modules]
    if missing:
        raise ValueError(f"UNIV model is missing required feature modules: {', '.join(missing)}")

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name in (*required, "blocks1.1"):
        if name in modules:
            handles.append(modules[name].register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.__setitem__(name, output)
            ))
    model = model.to(device).eval()
    image = torch.zeros(batch_size, 3, *image_size, device=device)
    try:
        with torch.inference_mode():
            latent, _attention = model(image, mask_ratio=0, return_last_attention=True)
    finally:
        for handle in handles:
            handle.remove()

    semantic = captured.get("norm", latent)
    grid_size = (14, 14)  # Explicit by policy; never inferred from token count.
    if semantic.ndim != 3 or semantic.shape[1] != grid_size[0] * grid_size[1]:
        raise ValueError(
            f"norm/output.latent must be BNC with grid_size={grid_size}; got {tuple(semantic.shape)}"
        )
    adapter = MultiScaleTaskAdapter(
        spatial_channels=captured["blocks2.1"].shape[1],
        semantic_channels=semantic.shape[2],
        out_channels=adapter_out_channels,
    ).to(device).eval()
    with torch.inference_mode():
        outputs = adapter(captured["blocks2.1"], semantic, grid_size=grid_size)

    selected = {
        "blocks2.1": tensor_summary(captured["blocks2.1"]),
        "norm": {**tensor_summary(semantic), "grid_size": list(grid_size)},
    }
    diagnostics = []
    if "blocks1.1" in captured:
        diagnostics.append({"module": "blocks1.1", **tensor_summary(captured["blocks1.1"])})
    return {
        "input_shape": list(image.shape),
        "selected_features": selected,
        "adapter_outputs": {name: tensor_summary(value) for name, value in outputs.items()},
        "diagnostic_features": diagnostics,
    }, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint0400.pth"))
    parser.add_argument("--checkpoint-key", choices=("student", "teacher"), default="student")
    parser.add_argument("--min-load-fraction", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--adapter-out-channels", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    model = build_original_univ(args.source_root)
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        checkpoint = load_univ_checkpoint(
            model, args.checkpoint, checkpoint_key=args.checkpoint_key
        )
    checkpoint_validation = validate_checkpoint_load_report(
        checkpoint, args.checkpoint_key, args.min_load_fraction
    )
    caught.extend(str(record.message) for record in records)
    report, runtime_warnings = inspect_univ_mta_features(
        model,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        device=args.device,
        adapter_out_channels=args.adapter_out_channels,
    )
    report.update(
        checkpoint_key=checkpoint.checkpoint_key,
        checkpoint_load_fraction=checkpoint.load_fraction,
        checkpoint_load_report=asdict(checkpoint),
        checkpoint_validation=checkpoint_validation,
        warnings=caught + runtime_warnings,
    )
    rendered = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
