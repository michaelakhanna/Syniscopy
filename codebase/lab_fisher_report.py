#!/usr/bin/env python3
"""CLI wrapper for the lab Fisher report package."""

from __future__ import annotations

from lab_fisher_report import main, run_report

__all__ = ["main", "run_report"]


if __name__ == "__main__":
    raise SystemExit(main())
