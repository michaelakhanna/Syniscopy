#!/usr/bin/env python3
"""Build the cross-platform source ZIP used by Segment Anything Model 2 starter notebooks."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sam2_starter" / "syniscopy_codebase.zip"
INCLUDE_PATHS = [
    "codebase",
    "docs",
    "examples",
    "recipes",
    "scripts",
    "sam2_starter/package_source_zip.py",
    "sam2_starter/package_source_zip.sh",
    "sam2_starter/README.md",
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


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    return (
        bool(parts & EXCLUDE_DIR_NAMES)
        or name in EXCLUDE_FILE_NAMES
        or name.startswith(EXCLUDE_FILE_PREFIXES)
        or name.endswith(EXCLUDE_FILE_SUFFIXES)
    )


def iter_files() -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_PATHS:
        src = ROOT / rel
        if not src.exists():
            continue
        if src.is_dir():
            files.extend(p for p in src.rglob("*") if p.is_file() and not should_skip(p.relative_to(ROOT)))
        elif src.is_file() and not should_skip(src.relative_to(ROOT)):
            files.append(src)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in iter_files():
            arcname = Path("SYNISCOPY") / src.relative_to(ROOT)
            zf.write(src, arcname.as_posix())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
