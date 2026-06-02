"""Small shared utilities used across Syniscopy modules."""

from __future__ import annotations

import os
from collections.abc import Iterable


def relative_path(base_dir: str, path: str) -> str:
    """
    Return ``path`` relative to ``base_dir`` when both paths share a root.

    On platforms where relative paths cannot be computed across roots or
    drives, return the absolute input path.
    """
    base_dir_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    try:
        return os.path.relpath(path_abs, base_dir_abs)
    except ValueError:
        return path_abs


def init_infinite_dict(keys: Iterable[str]) -> dict[str, float]:
    """Return a mapping from each key to positive infinity."""
    return {key: float("inf") for key in keys}
