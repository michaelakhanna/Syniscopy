#!/usr/bin/env python3
"""Run the fresh Syniscopy verification suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VERIFY_ROOT = Path(__file__).resolve().parent


PROFILE_MARKERS = {
    "quick": "quick and not artifacts and not renderer and not full",
    "full": "(quick or full) and not artifacts and not renderer",
    "artifacts": "artifacts",
    "adversarial": "quick or full or artifacts or renderer",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_MARKERS),
        default="quick",
        help="Which verification profile to run.",
    )
    parser.add_argument("--lab-report", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--packet-root", type=Path, default=None)
    parser.add_argument("--monte-carlo-samples", type=int, default=None)
    parser.add_argument("--include-renderer", action="store_true")
    parser.add_argument("--junit-xml", type=Path, default=None)
    args, extra_pytest_args = parser.parse_known_args()

    try:
        import pytest
    except Exception:
        print("pytest is required. Install with: python3 -m pip install -r verification/requirements-verification.txt")
        return 2

    os.environ["SYNISCOPY_VERIFY_PROFILE"] = args.profile
    if args.lab_report is not None:
        os.environ["SYNISCOPY_VERIFY_LAB_REPORT"] = str(args.lab_report)
    if args.mask_root is not None:
        os.environ["SYNISCOPY_VERIFY_MASK_ROOT"] = str(args.mask_root)
    if args.packet_root is not None:
        os.environ["SYNISCOPY_VERIFY_PACKET_ROOT"] = str(args.packet_root)
    if args.monte_carlo_samples is not None:
        os.environ["SYNISCOPY_VERIFY_MONTE_CARLO_SAMPLES"] = str(args.monte_carlo_samples)
    if args.include_renderer or args.profile == "adversarial":
        os.environ["SYNISCOPY_VERIFY_RUN_RENDERER"] = "1"

    pytest_cmd = [
        str(VERIFY_ROOT / "tests"),
        "-m",
        PROFILE_MARKERS[args.profile],
    ]
    if args.junit_xml is not None:
        pytest_cmd.extend(["--junitxml", str(args.junit_xml)])
    pytest_cmd.extend(extra_pytest_args)

    print(f"Running Syniscopy verification profile={args.profile}")
    print("pytest " + " ".join(pytest_cmd))
    return int(pytest.main(pytest_cmd))


if __name__ == "__main__":
    raise SystemExit(main())
