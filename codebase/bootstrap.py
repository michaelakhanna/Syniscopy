"""Entry-point bootstrap helpers for scripts inside ``codebase``."""

from __future__ import annotations

import sys
from pathlib import Path


CODEBASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODEBASE_ROOT.parent


def ensure_codebase_on_path() -> Path:
    """Ensure direct script execution can import sibling codebase modules."""
    root = str(CODEBASE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return CODEBASE_ROOT
