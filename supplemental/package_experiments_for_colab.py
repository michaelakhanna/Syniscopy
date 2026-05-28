#!/usr/bin/env python3
"""Build the cross-platform source ZIP used by supplemental Colab notebooks."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    "supplemental/README.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
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


def main() -> None:
    SUPPLEMENTAL_ROOT.mkdir(parents=True, exist_ok=True)
    (SUPPLEMENTAL_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in iter_files():
            zf.write(src, src.relative_to(ROOT).as_posix())

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
