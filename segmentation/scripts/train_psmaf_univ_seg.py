#!/usr/bin/env python3
"""Stage-3 command boundary for train psmaf univ seg."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to an experiment configuration")
    args = parser.parse_args()
    if args.config:
        raise SystemExit("Execution is reserved for the implementation stage; structure is ready.")


if __name__ == "__main__":
    main()
