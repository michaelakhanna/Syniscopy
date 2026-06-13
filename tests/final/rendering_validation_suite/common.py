"""Compatibility import target for generated large-section validation scripts."""

from __future__ import annotations

from pathlib import Path
import sys

GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

from common import *  # noqa: F401,F403
