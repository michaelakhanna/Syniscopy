#!/usr/bin/env python3
"""Fail fast when a public source tree contains local/generated artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
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
BLOCKED_PACKAGED_PREFIXES = (
    LOCAL_REVIEW_ROOT + "/",
    LOCAL_SCRATCH_ROOT + "/",
    LOCAL_HYPHEN_ROOT + "/",
    "release_uploads/",
)
BLOCKED_PACKAGED_INFIXES = (
    "/" + LOCAL_REVIEW_ROOT + "/",
    "/" + LOCAL_SCRATCH_ROOT + "/",
    "/" + LOCAL_HYPHEN_ROOT + "/",
)
BLOCKED_RELATIVE_PATHS = {
    "supplemental/data/liverpool_caustic_50nm",
    STALE_TRANSFER_DIR + "/base_sam2_overlay.avi",
    STALE_TRANSFER_DIR + "/finetuned_sam2_overlay.avi",
}
TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".jsonl", ".md", ".py", ".tex", ".txt", ".yaml", ".yml"}

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



def check_text_for_blocked_markers(text: str, context: str, failures: list[str]) -> None:
    for marker in BLOCKED_TEXT_MARKERS:
        if marker in text:
            failures.append(f"blocked local/private marker {marker!r} in {context}")
            return
    if has_blocked_local_review_reference(text):
        failures.append(f"blocked local review/staging path reference in {context}")


def inspect_zip_file(path: Path, rel_posix: str, failures: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            for name in names:
                normalized = name.lower().replace("\\", "/")
                if normalized.startswith(BLOCKED_PACKAGED_PREFIXES):
                    failures.append(f"blocked path inside {rel_posix}: {name}")
                if any(marker in normalized for marker in BLOCKED_PACKAGED_INFIXES):
                    failures.append(f"blocked local review path inside {rel_posix}: {name}")
                if normalized.startswith("supplemental/outputs/") or "/supplemental/outputs/" in normalized:
                    failures.append(f"generated supplemental outputs inside {rel_posix}: {name}")
                if normalized.endswith(".zip"):
                    failures.append(f"nested zip inside {rel_posix}: {name}")
                if Path(name).suffix.lower() in TEXT_SUFFIXES:
                    try:
                        data = zf.read(name).decode("utf-8")
                    except Exception:
                        continue
                    check_text_for_blocked_markers(data, f"{rel_posix}!{name}", failures)
            required_by_zip = {
                "supplemental/syniscopy_source.zip": {
                    "manifest": "syniscopy_source_manifest.json",
                    "members": {
                        "codebase/config/__init__.py",
                        "pyproject.toml",
                        "recipes/default.py",
                        "supplemental/notebook_source.py",
                        "sam2_starter/sam2training.ipynb",
                        "syniscopy_source_manifest.json",
                    },
                },
                "sam2_starter/syniscopy_codebase.zip": {
                    "manifest": "SYNISCOPY/sam2_starter_source_manifest.json",
                    "members": {
                        "SYNISCOPY/codebase/config/__init__.py",
                        "SYNISCOPY/pyproject.toml",
                        "SYNISCOPY/recipes/default.py",
                        "SYNISCOPY/sam2_starter/sam2training.ipynb",
                        "SYNISCOPY/sam2_starter_source_manifest.json",
                    },
                },
            }
            if rel_posix in required_by_zip:
                contract = required_by_zip[rel_posix]
                missing = sorted(set(contract["members"]) - set(names))
                if missing:
                    failures.append(f"{rel_posix} missing required members: {missing}")
                try:
                    manifest = json.loads(zf.read(contract["manifest"]).decode("utf-8"))
                except Exception as exc:
                    failures.append(f"{rel_posix} missing/read-failed internal manifest: {exc}")
                else:
                    source = manifest.get("source_provenance") if isinstance(manifest.get("source_provenance"), dict) else {}
                    if manifest.get("schema_version") != "syniscopy-source-zip-manifest-v1" or not source.get("fingerprint"):
                        failures.append(f"{rel_posix} internal manifest lacks source fingerprint")
    except zipfile.BadZipFile as exc:
        failures.append(f"bad zip file {rel_posix}: {exc}")

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
        if path.is_file() and path.suffix.lower() == ".zip":
            inspect_zip_file(path, rel_posix, failures)
        elif path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            check_text_for_blocked_markers(text, rel_posix, failures)

    for required_rel in (
        "codebase/dataset_schema.py",
        "codebase/supervision_policy.py",
        "codebase/metadata.py",
        "supplemental/notebook_source.py",
        "supplemental/package_experiments_for_colab.py",
        "supplemental/syniscopy_source.zip",
        "sam2_starter/syniscopy_codebase.zip",
        "sam2_starter/sam2training.ipynb",
        "sam2_starter/sam2inference.ipynb",
    ):
        if not (root / required_rel).exists():
            failures.append(f"required W14/W15 release path missing: {required_rel}")

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
