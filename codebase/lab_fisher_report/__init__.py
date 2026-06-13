"""Lab Fisher report package."""

from __future__ import annotations

__all__ = ["main", "run_report"]


def main(*args, **kwargs):
    from .coordinator import main as _main

    return _main(*args, **kwargs)


def run_report(*args, **kwargs):
    from .coordinator import run_report as _run_report

    return _run_report(*args, **kwargs)
