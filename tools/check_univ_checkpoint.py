#!/usr/bin/env python3
"""Report the branch and tensor count in a UNIV checkpoint."""
import argparse
from pathlib import Path
import sys

import torch

# Direct script execution puts tools/, rather than the repository root, first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.checkpoint_loader import extract_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    state = extract_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    print(f"state tensors: {len(state)}")


if __name__ == "__main__":
    main()
