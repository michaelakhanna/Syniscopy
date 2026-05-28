#!/usr/bin/env python3
"""Fail fast when a public source tree contains local/generated artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LOCAL_REVIEW_ROOT = "throw" + "out"
LOCAL_SCRATCH_ROOT = "throw" + "_out"
LOCAL_HYPHEN_ROOT = "throw" + "-out"
STALE_TRANSFER_DIR = "/".join(["supplemental", "artifacts", "sam2_transfer"])

BLOCKED_DIR_NAMES = {
    "__MACOSX",
    "__pycache__",
    ".ipynb_checkpoints",
    "release_uploads",
    LOCAL_SCRATCH_ROOT,
    LOCAL_SCRATCH_ROOT + "_2",
}
BLOCKED_FILE_NAMES = {
    ".DS_Store",
    "paper_submission_wording_review.md",
}
BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
}
BLOCKED_TEXT_MARKERS = {
    "/" + "Users/",
    "COLAB_" + "NOTEBOOK_ID",
    "localhost" + ".run",
    "Traceback (" + "most recent call last)",
    "private development " + "conversation",
}
BLOCKED_LOCAL_REVIEW_REFERENCE_MARKERS = {
    LOCAL_REVIEW_ROOT + "/",
    LOCAL_SCRATCH_ROOT,
    LOCAL_HYPHEN_ROOT,
}
BLOCKED_RELATIVE_PATHS = {
    "supplemental/data/liverpool_caustic_50nm",
    STALE_TRANSFER_DIR + "/base_sam2_overlay.avi",
    STALE_TRANSFER_DIR + "/finetuned_sam2_overlay.avi",
}
BLOCKED_RELATIVE_PREFIXES = {
    "supplemental/data/liverpool_caustic_50nm/",
    "supplemental/data/liverpool_caustic_50nm_review/clips/",
}


def iter_paths(root: Path):
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == ".git":
            continue
        yield path, rel


def has_blocked_local_review_reference(text: str) -> bool:
    normalized = text.lower().replace("\\", "/")
    return any(marker in normalized for marker in BLOCKED_LOCAL_REVIEW_REFERENCE_MARKERS)


def has_blocked_local_review_path(path: Path, rel: Path) -> bool:
    rel_posix = rel.as_posix().lower()
    if LOCAL_REVIEW_ROOT + "/" in rel_posix or (
        path.is_dir() and rel.name.lower() == LOCAL_REVIEW_ROOT
    ):
        return True
    return any(marker in rel_posix for marker in {LOCAL_SCRATCH_ROOT, LOCAL_HYPHEN_ROOT})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a public Syniscopy source tree excludes local build/release artifacts."
    )
    parser.add_argument("root", nargs="?", default=".", help="Source tree root to verify.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures: list[str] = []

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"release tree root does not exist or is not a directory: {root}")

    for path, rel in iter_paths(root):
        parts = set(rel.parts)
        name = path.name
        rel_posix = rel.as_posix()
        if rel_posix in BLOCKED_RELATIVE_PATHS or any(
            rel_posix.startswith(prefix) for prefix in BLOCKED_RELATIVE_PREFIXES
        ):
            failures.append(f"blocked local/stale caustic data path present: {rel_posix}")
            continue
        if parts & BLOCKED_DIR_NAMES:
            failures.append(f"blocked directory path present: {rel_posix}")
            continue
        if has_blocked_local_review_path(path, rel):
            failures.append(f"blocked local review/staging path present: {rel_posix}")
            continue
        if name in BLOCKED_FILE_NAMES or name.endswith(tuple(BLOCKED_SUFFIXES)):
            failures.append(f"blocked file present: {rel_posix}")
            continue
        if path.is_dir() and name == ".git":
            failures.append(f"nested .git directory present: {rel_posix}")
            continue
        if path.is_file() and path.suffix.lower() in {
            ".csv",
            ".ipynb",
            ".json",
            ".md",
            ".py",
            ".tex",
            ".txt",
            ".yaml",
            ".yml",
        }:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in BLOCKED_TEXT_MARKERS:
                if marker in text:
                    failures.append(f"blocked local/private marker {marker!r} in {rel_posix}")
                    break
            else:
                if has_blocked_local_review_reference(text):
                    failures.append(f"blocked local review/staging path reference in {rel_posix}")

    if (root / "supplemental" / "outputs").exists():
        failures.append("generated supplemental/outputs directory present")
    if (root / "release_uploads").exists():
        failures.append("release_uploads directory present")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"release tree check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
