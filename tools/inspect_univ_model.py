#!/usr/bin/env python3
"""Construct the original UNIV encoder and print a machine-readable inventory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.univ_diagnostics import build_original_univ, model_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    args = parser.parse_args()
    print(json.dumps(model_inventory(build_original_univ(args.source_root)), indent=2))


if __name__ == "__main__":
    main()
