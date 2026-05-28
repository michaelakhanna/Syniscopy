#!/usr/bin/env python3
"""Check optional generated release assets for local/runtime path leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

BLOCKED_MARKERS = {
    "/" + "Users/",
    "/" + "content/",
    "drive/" + "MyDrive",
    "runtime" + "/",
    "COLAB_" + "NOTEBOOK_ID",
    "COLAB_" + "NOTEBOOK_REDACTED",
    "localhost" + ".run",
    "FileNot" + "FoundError",
    "File-" + "not-found redacted",
    "Traceback (" + "most recent call last)",
    "Traceback " + "redacted",
}

LOCAL_SCRATCH_ROOT = "throw" + "_out"


def _is_private_scratch_path(path: Path) -> bool:
    return any(part.lower().startswith(LOCAL_SCRATCH_ROOT) for part in path.parts)


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify optional generated Syniscopy release assets before archival upload. "
            "Run this on staged supplemental/outputs or artifact folders, not on the "
            "source tree itself."
        )
    )
    parser.add_argument("paths", nargs="+", help="Generated asset path(s) to check.")
    args = parser.parse_args()

    failures: list[str] = []
    for raw_path in args.paths:
        root = Path(raw_path)
        if not root.exists():
            failures.append(f"path does not exist: {root}")
            continue
        if _is_private_scratch_path(root):
            failures.append(f"{root}: generated asset path must not be under local scratch")
            continue
        paths = _iter_text_files(root) if root.is_dir() else [root]
        for path in paths:
            if _is_private_scratch_path(path):
                failures.append(f"{path}: generated asset path must not be under local scratch")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in BLOCKED_MARKERS:
                if marker in text:
                    failures.append(f"{path}: blocked marker {marker!r}")
                    break

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("generated release asset check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
