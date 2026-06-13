from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def pytest_addoption(parser):
    parser.addoption("--lab-report", action="store", default=None)
    parser.addoption("--mask-root", action="store", default=None)
    parser.addoption("--packet-root", action="store", default=None)
    parser.addoption("--monte-carlo-samples", action="store", default=None)
    parser.addoption("--include-renderer", action="store_true", default=False)


def pytest_configure(config):
    option_to_env = {
        "lab_report": "SYNISCOPY_VERIFY_LAB_REPORT",
        "mask_root": "SYNISCOPY_VERIFY_MASK_ROOT",
        "packet_root": "SYNISCOPY_VERIFY_PACKET_ROOT",
        "monte_carlo_samples": "SYNISCOPY_VERIFY_MONTE_CARLO_SAMPLES",
    }
    for option_name, env_name in option_to_env.items():
        value = config.getoption(option_name)
        if value:
            os.environ[env_name] = str(value)
    if config.getoption("include_renderer"):
        os.environ["SYNISCOPY_VERIFY_RUN_RENDERER"] = "1"

