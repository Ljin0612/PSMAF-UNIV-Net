#!/usr/bin/env python3
"""Evaluate a Stage 7 checkpoint using the IR stream only."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.scripts import eval_univ_mta_fasterrcnn_m3fd as base

_BASE_BUILD_ARG_PARSER = base.build_arg_parser


def build_arg_parser():
    parser = _BASE_BUILD_ARG_PARSER()
    parser.description = __doc__
    parser.set_defaults(image_size=640, output_json=Path("outputs/stage7_eval/metrics.json"), stage_label="7")
    return parser


def main() -> None:
    # Reuse the established IR-only evaluation implementation; replace only its
    # parser defaults so paired RGB data and the PCCL branch cannot be consulted.
    base.build_arg_parser = build_arg_parser
    base.main()


if __name__ == "__main__":
    main()
