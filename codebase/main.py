"""CLI entry point for running the default Syniscopy simulation."""

from __future__ import annotations

import argparse

from config import default_params


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Syniscopy simulation using concept-owned defaults.",
    )
    parser.add_argument(
        "--return-frames",
        action="store_true",
        help="Return frames from the orchestration layer instead of only writing configured outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    from simulation import run_simulation

    return run_simulation(default_params(), return_frames=bool(args.return_frames))


if __name__ == "__main__":
    main()
