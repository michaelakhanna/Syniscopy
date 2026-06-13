from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _float_field(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return float("nan")
    return float(raw)


def read_lab_ranking_fishers(lab_report_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(lab_report_dir) / "microscope_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, np.ndarray] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            microscope = str(row["microscope"])
            fxx = _float_field(row, "fisher_xx")
            fxy = _float_field(row, "fisher_xy")
            fyy = _float_field(row, "fisher_yy")
            out[microscope] = np.asarray([[fxx, fxy], [fxy, fyy]], dtype=float)
    if not out:
        raise ValueError(f"No microscope rows found in {path}")
    return out


def iter_fusion_rows(lab_report_dir: str | Path) -> Iterable[dict[str, str]]:
    path = Path(lab_report_dir) / "fusion_crlb.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def sigma_xy_from_fisher(F: np.ndarray) -> float:
    from fisher import sigma_xy_from_fisher as _sigma_xy_from_fisher

    return float(_sigma_xy_from_fisher(np.asarray(F, dtype=float)))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=float)


def validate_mask_group(mask_root: str | Path) -> int:
    root = Path(mask_root)
    roles = ["mask_geometry", "mask_supported", "ignore_mask", "loss_weight"]
    for role in roles:
        if not (root / role).exists():
            raise FileNotFoundError(root / role)

    checked = 0
    for geometry_path in sorted((root / "mask_geometry").glob("particle_*/*")):
        if geometry_path.is_dir():
            continue
        rel = geometry_path.relative_to(root / "mask_geometry")
        paths = {role: root / role / rel for role in roles}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise AssertionError(f"Missing paired masks for {rel}: {missing}")
        geometry = load_mask(paths["mask_geometry"]) > 0
        supported = load_mask(paths["mask_supported"]) > 0
        ignored = load_mask(paths["ignore_mask"]) > 0
        loss = load_mask(paths["loss_weight"])
        if not np.any(geometry):
            checked += 1
            continue
        if np.any(supported & ~geometry):
            raise AssertionError(f"{rel}: mask_supported is not subset of mask_geometry")
        if np.any(ignored & ~geometry):
            raise AssertionError(f"{rel}: ignore_mask is not subset of mask_geometry")
        if not np.array_equal(ignored, geometry & ~supported):
            raise AssertionError(f"{rel}: ignore_mask != mask_geometry & ~mask_supported")
        if np.any(loss[~supported] > 0.0):
            raise AssertionError(f"{rel}: loss_weight positive outside supported mask")
        if np.any(supported) and not np.any(loss[supported] > 0.0):
            raise AssertionError(f"{rel}: supported pixels have no positive loss_weight")
        checked += 1
    if checked == 0:
        raise AssertionError(f"No mask files found under {root}")
    return checked


def load_supervision_audit(mask_root: str | Path) -> dict:
    path = Path(mask_root) / "supervision_audit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_packet_paths(packet_root: str | Path) -> list[Path]:
    root = Path(packet_root)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    return sorted(root.glob("*.npz"))
