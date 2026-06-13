"""Shared helpers for generated large-section validation scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"


def add_paths() -> None:
    """Expose the Syniscopy codebase and generated-script directory."""

    for path in (CODEBASE_DIR, Path(__file__).resolve().parent):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def banner(title: str) -> None:
    line = "=" * max(12, len(str(title)))
    print(f"\n{line}\n{title}\n{line}")


def verdict(ok: bool, detail: str = "") -> int:
    status = "PASS" if bool(ok) else "FAIL"
    suffix = f" {detail}" if detail else ""
    print(f"\n>>> RESULT: {status}{suffix}\n")
    return 0 if bool(ok) else 1


def relative_error(a: Any, b: Any, *, floor: float = 1.0e-30) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), float(floor))
    return float(np.max(np.abs(aa - bb) / denom))


def install_scipy_trapz_shim() -> None:
    """Restore scipy.integrate.trapz for older oracle packages on SciPy 1.14+."""

    import scipy.integrate as integrate

    if not hasattr(integrate, "trapz") and hasattr(integrate, "trapezoid"):
        integrate.trapz = integrate.trapezoid


def tiny_render_overrides(
    *,
    modality: str,
    image_size: int = 32,
    num_frames: int = 1,
    matched_microscopes: Any = None,
    raw_camera_sequence: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a small current-API parameter payload for validation renders."""

    add_paths()
    from config import default_params

    params = default_params()
    params.update(
        {
            "imaging_model": str(modality),
            "image_size_pixels": int(image_size),
            "num_frames": int(num_frames),
            "duration_seconds": max(float(num_frames), 1.0) / 24.0,
            "pixel_size_nm": 50.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "max_psf_z_slices": 64,
            "random_seed": 12345,
            "mask_generation_enabled": False,
            "return_ideal_float_frames": True,
            "save_frame_sequence": False,
            "save_raw_camera_video": bool(raw_camera_sequence),
            "save_raw_camera_frame_sequence": False,
            "save_raw_frame_views": False,
            "matched_microscopes": matched_microscopes,
            "background_subtraction_method": "reference_frame",
            "shot_noise_enabled": False,
            "gaussian_noise_enabled": False,
            "fluorescence_backend": "parametric_psf",
            "fluorescence_source_representation": "projected_2d",
        }
    )
    params.update(overrides)
    return params


def set_particle_scene(
    params: dict[str, Any],
    *,
    pixel_size_nm: float,
    diameter_nm: float,
    center_pixel_xy: tuple[float, float] | None = None,
    material: str | None = None,
    z_nm: float = 0.0,
) -> dict[str, Any]:
    """Install a single spherical particle scene into a parameter payload."""

    add_paths()
    from config import default_param_value

    image_size = int(params.get("image_size_pixels", 32))
    cx, cy = (
        (0.5 * image_size, 0.5 * image_size)
        if center_pixel_xy is None
        else (float(center_pixel_xy[0]), float(center_pixel_xy[1]))
    )
    px = float(pixel_size_nm)
    material_name = str(
        material
        or (
            "fluorescent_polystyrene"
            if "fluorescence" in str(params.get("imaging_model", "")).lower()
            else "gold"
        )
    )
    particle = deepcopy(default_param_value("particles")[0])
    particle["name"] = "validation_particle_0"
    particle["motion"]["hydrodynamic_diameter_nm"] = float(diameter_nm)
    particle["motion"]["initial_position_nm"] = [cx * px, cy * px, float(z_nm)]
    component = particle["components"][0]
    component["shape"] = "sphere"
    component["offset_nm"] = [0.0, 0.0, 0.0]
    component["diameter_nm"] = float(diameter_nm)
    component["material"] = material_name
    component["refractive_index"] = None
    params["pixel_size_nm"] = px
    params["particles"] = [particle]
    return params


__all__ = [
    "add_paths",
    "banner",
    "install_scipy_trapz_shim",
    "relative_error",
    "set_particle_scene",
    "tiny_render_overrides",
    "verdict",
]
