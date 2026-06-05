"""Shared source-resolution helpers for supplemental Colab notebooks."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


SOURCE_ZIP_NAME = "syniscopy_source.zip"


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    explicit = globals().get("SYNISCOPY_SUPPLEMENTAL_ROOT", None)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    for drive_root in (
        Path("/content/drive/MyDrive"),
        Path("/content/drive/My Drive"),
        Path("/content/drive/Shareddrives"),
    ):
        if drive_root.exists():
            candidates.append(drive_root / "supplemental")
            candidates.append(drive_root / "SYNISCOPY" / "supplemental")
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            out.append(resolved)
            seen.add(key)
    return out


def find_supplemental_root() -> Path:
    """Locate the uploaded supplemental folder without relying on cwd."""

    for resolved in _candidate_roots():
        nested = resolved / "supplemental"
        if (
            (nested / SOURCE_ZIP_NAME).exists()
            or (nested / "source" / "codebase").is_dir()
            or (nested / "E01.ipynb").exists()
            or (nested / "E07.ipynb").exists()
        ):
            return nested
        if (
            (resolved / SOURCE_ZIP_NAME).exists()
            or (resolved / "source" / "codebase").is_dir()
            or (resolved / "outputs").is_dir()
            or (resolved / "E01.ipynb").exists()
            or (resolved / "E07.ipynb").exists()
        ):
            return resolved
    raise RuntimeError(
        "Syniscopy supplemental folder not found. Upload the supplemental folder "
        "to Drive as MyDrive/supplemental, including syniscopy_source.zip."
    )


def _extracted_root(extract_dir: Path) -> Path:
    if (extract_dir / "codebase").is_dir():
        return extract_dir
    candidates = [p for p in extract_dir.iterdir() if p.is_dir() and (p / "codebase").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError("Could not find a single codebase/ root after extracting syniscopy_source.zip.")
    return candidates[0]


def prepare_syniscopy_source(supplemental_root: Path) -> Path:
    """Extract source zip when present and reject stale extracted source trees."""

    supplemental_root = Path(supplemental_root).expanduser().resolve()
    source_root = supplemental_root / "source"
    zip_path = supplemental_root / SOURCE_ZIP_NAME
    if zip_path.exists():
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        stamp = source_root / ".syniscopy_source_sha256"
        if (
            source_root.exists()
            and (source_root / "codebase").is_dir()
            and stamp.exists()
            and stamp.read_text(encoding="utf-8").strip() == digest
        ):
            return source_root
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(source_root)
        root = _extracted_root(source_root)
        stamp.write_text(digest + "\n", encoding="utf-8")
        return root
    if (source_root / "codebase").is_dir():
        return source_root
    if (supplemental_root.parent / "codebase").is_dir():
        return supplemental_root.parent
    raise FileNotFoundError(
        f"Missing {SOURCE_ZIP_NAME}; expected it under {supplemental_root}."
    )
