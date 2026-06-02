#!/usr/bin/env python3
"""Check optional generated release assets for local/runtime path leakage."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
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
    "runtime" + "/",
    "COLAB_" + "NOTEBOOK_ID",
    "COLAB_" + "NOTEBOOK_REDACTED",
    "localhost" + ".run",
    "File-" + "not-found redacted",
    "Traceback (" + "most recent call last)",
    "Traceback " + "redacted",
}

LOCAL_SCRATCH_ROOT = "throw" + "_out"
LOCAL_REVIEW_ROOT = "throw" + "out"
LOCAL_HYPHEN_ROOT = "throw" + "-out"
BLOCKED_PACKAGED_PREFIXES = (
    LOCAL_REVIEW_ROOT + "/",
    LOCAL_SCRATCH_ROOT + "/",
    LOCAL_HYPHEN_ROOT + "/",
    "release_uploads/",
)


def _is_private_scratch_path(path: Path) -> bool:
    return any(part.lower().startswith(LOCAL_SCRATCH_ROOT) for part in path.parts)


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path



def _check_text(text: str, context: str, failures: list[str]) -> None:
    for marker in BLOCKED_MARKERS:
        if marker in text:
            failures.append(f"{context}: blocked marker {marker!r}")
            return


def _inspect_zip_bytes(data: bytes, context: str, failures: list[str], *, depth: int = 0) -> None:
    if depth > 3:
        failures.append(f"{context}: nested zip depth exceeds verifier limit")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                member_context = f"{context}!{name}"
                normalized = name.lower().replace("\\", "/")
                if normalized.startswith(BLOCKED_PACKAGED_PREFIXES):
                    failures.append(f"{member_context}: blocked packaged path")
                if normalized.startswith("supplemental/outputs/") or "/supplemental/outputs/" in normalized:
                    failures.append(f"{member_context}: generated supplemental output included")
                suffix = Path(name).suffix.lower()
                if suffix == ".zip":
                    _inspect_zip_bytes(zf.read(name), member_context, failures, depth=depth + 1)
                elif suffix in TEXT_SUFFIXES:
                    try:
                        _check_text(zf.read(name).decode("utf-8"), member_context, failures)
                    except UnicodeDecodeError:
                        continue
    except zipfile.BadZipFile as exc:
        failures.append(f"{context}: bad zip file: {exc}")


def _inspect_path(path: Path, failures: list[str]) -> None:
    if path.suffix.lower() == ".zip":
        _inspect_zip_bytes(path.read_bytes(), str(path), failures)
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    _check_text(text, str(path), failures)

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
        paths = list(_iter_text_files(root)) if root.is_dir() else [root]
        if root.is_dir():
            paths.extend(path for path in root.rglob("*.zip") if path.is_file())
        for path in paths:
            if _is_private_scratch_path(path):
                failures.append(f"{path}: generated asset path must not be under local scratch")
                continue
            _inspect_path(path, failures)

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("generated release asset check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
