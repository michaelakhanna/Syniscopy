#!/usr/bin/env python3
"""Build the cross-platform source ZIP used by supplemental Colab notebooks."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from metadata import build_source_provenance  # noqa: E402
SUPPLEMENTAL_ROOT = ROOT / "supplemental"
ZIP_OUT = SUPPLEMENTAL_ROOT / "syniscopy_source.zip"
INCLUDE_PATHS = [
    "codebase",
    "docs",
    "examples",
    "recipes",
    "scripts",
    "sam2_starter",
    "supplemental/package_experiments_for_colab.py",
    "supplemental/package_experiments_for_colab.sh",
    "supplemental/rebuild_liverpool_review_clips.py",
    "supplemental/notebook_source.py",
    "supplemental/README.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements.txt",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
]
EXCLUDE_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", "__MACOSX"}
EXCLUDE_FILE_NAMES = {".DS_Store"}
EXCLUDE_FILE_PREFIXES = ("._",)
EXCLUDE_FILE_SUFFIXES = (".pyc", ".pyo")


def should_skip(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    name = rel_path.name
    if bool(parts & EXCLUDE_DIR_NAMES):
        return True
    if name in EXCLUDE_FILE_NAMES or name.startswith(EXCLUDE_FILE_PREFIXES):
        return True
    if name.endswith(EXCLUDE_FILE_SUFFIXES):
        return True
    return rel_path.parts[:1] == ("sam2_starter",) and name.endswith(".zip")


def iter_files() -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_PATHS:
        src = ROOT / rel
        if not src.exists():
            continue
        if src.is_dir():
            files.extend(
                p
                for p in src.rglob("*")
                if p.is_file() and not should_skip(p.relative_to(ROOT))
            )
        elif src.is_file() and not should_skip(src.relative_to(ROOT)):
            files.append(src)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip_manifest(files: list[Path], *, archive_prefix: str = "") -> dict:
    records = []
    for src in files:
        rel = src.relative_to(ROOT).as_posix()
        arcname = f"{archive_prefix}/{rel}" if archive_prefix else rel
        records.append({"path": arcname, "source_path": rel, "sha256": sha256_file(src)})
    source_provenance = build_source_provenance(str(ROOT))
    source_provenance["repo_root"] = "."
    source_provenance["git_root"] = "."
    return {
        "schema_version": "syniscopy-source-zip-manifest-v1",
        "packager": Path(__file__).relative_to(ROOT).as_posix(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_provenance": source_provenance,
        "file_count": len(records),
        "files": records,
    }

def main() -> None:
    SUPPLEMENTAL_ROOT.mkdir(parents=True, exist_ok=True)
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    files = iter_files()
    manifest = build_zip_manifest(files)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in files:
            zf.write(src, src.relative_to(ROOT).as_posix())
        zf.writestr("syniscopy_source_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("Supplemental source zip:")
    print(f"  {ZIP_OUT}")
    print()
    print("Upload this folder to Google Drive as:")
    print(f"  {SUPPLEMENTAL_ROOT}")
    print("  MyDrive/supplemental")
    print()
    print("Notebook outputs will be written under:")
    print("  MyDrive/supplemental/outputs")


if __name__ == "__main__":
    main()
