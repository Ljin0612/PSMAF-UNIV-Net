#!/usr/bin/env python3
"""Verify a UNIV checkpoint against the original encoder architecture."""
import argparse
import json
from pathlib import Path
import sys

import torch

# Direct script execution puts tools/, rather than the repository root, first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.checkpoint_loader import extract_state_dict
from psmaf_univ.univ_diagnostics import build_original_univ, inspect_checkpoint_against_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--summary-only", action="store_true", help="inspect contents without constructing the model")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    summary = {
        "checkpoint": str(args.checkpoint),
        "top_level_keys": sorted(str(key) for key in checkpoint) if isinstance(checkpoint, dict) else [],
        "state_tensor_count": len(state),
        "state_parameter_count": sum(tensor.numel() for tensor in state.values()),
        "dtype_counts": {},
    }
    for tensor in state.values():
        dtype = str(tensor.dtype).removeprefix("torch.")
        summary["dtype_counts"][dtype] = summary["dtype_counts"].get(dtype, 0) + 1
    if not args.summary_only:
        summary["model_compatibility"] = inspect_checkpoint_against_model(build_original_univ(args.source_root), args.checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
