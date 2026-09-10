#!/usr/bin/env python3
"""Probe UNIV feature stages, final semantic tokens, and attention shapes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.univ_diagnostics import DEFAULT_PROBE_MODULES, build_original_univ, inspect_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modules", nargs="+", default=DEFAULT_PROBE_MODULES)
    args = parser.parse_args()
    model = build_original_univ(args.source_root)
    checkpoint_report = None
    if args.checkpoint:
        from psmaf_univ.univ_diagnostics import inspect_checkpoint_against_model

        checkpoint_report = inspect_checkpoint_against_model(model, args.checkpoint)
    report = inspect_features(
        model, batch_size=args.batch_size, image_size=tuple(args.image_size), module_names=args.modules, device=args.device
    )
    report["checkpoint"] = checkpoint_report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
