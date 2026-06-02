"""Optional volumetric particle-scene helpers.

These functions deliberately sit beside the default single-frame renderer.  They
provide configured z-stack, confocal, light-sheet, and holotomography-style
volume reductions without changing the ordinary single-plane simulation path.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def resolve_volume_z_planes_nm(params: dict) -> np.ndarray:
    explicit = params.get("volumetric_z_planes_nm", None)
    if explicit is not None:
        planes = np.asarray(explicit, dtype=float).reshape(-1)
    else:
        count = int(params.get("volumetric_z_count", 5))
        if count <= 0:
            raise ValueError("volumetric_z_count must be positive.")
        span = float(params.get("volumetric_z_range_nm", 1000.0))
        if span < 0.0 or not np.isfinite(span):
            raise ValueError("volumetric_z_range_nm must be finite and non-negative.")
        if count == 1:
            planes = np.asarray([0.0], dtype=float)
        else:
            planes = np.linspace(-0.5 * span, 0.5 * span, count, dtype=float)
    if planes.size == 0 or not np.all(np.isfinite(planes)):
        raise ValueError("Volumetric z planes must be a non-empty finite sequence.")
    return planes


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Volume weights must be finite and non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return arr / total


def volume_plane_weights(params: dict, z_planes_nm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(params.get("volumetric_imaging_mode", "single_plane")).strip().lower()
    z = np.asarray(z_planes_nm, dtype=float).reshape(-1)
    if mode in {"single_plane", "z_stack", "holotomography_projection"}:
        weights = np.ones_like(z, dtype=float)
        model = "uniform_projection"
    elif mode == "confocal":
        sigma = float(params.get("confocal_pinhole_sigma_nm", 350.0))
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("confocal_pinhole_sigma_nm must be positive.")
        weights = np.exp(-0.5 * (z / sigma) ** 2)
        model = "gaussian_confocal_axial_detection"
    elif mode == "light_sheet":
        center = float(params.get("light_sheet_center_z_nm", 0.0))
        sigma = float(params.get("light_sheet_sigma_nm", 500.0))
        if sigma <= 0.0 or not np.isfinite(sigma) or not np.isfinite(center):
            raise ValueError("light-sheet center and sigma must be finite, with sigma positive.")
        weights = np.exp(-0.5 * ((z - center) / sigma) ** 2)
        model = "gaussian_light_sheet_excitation"
    else:
        raise ValueError(f"Unsupported volumetric_imaging_mode {mode!r}.")
    normalized = _normalized_weights(weights)
    metadata = {
        "volumetric_imaging_mode": mode,
        "volume_weight_model": model,
        "z_planes_nm": z.astype(float).tolist(),
        "z_plane_weights": normalized.astype(float).tolist(),
        "volume_output_mode": str(params.get("volume_output_mode", "integrated_projection")),
    }
    return normalized, metadata


def combine_volume_stack(
    stack: np.ndarray,
    z_planes_nm: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(stack, dtype=float)
    if arr.ndim < 3:
        raise ValueError("Volume stack must have shape (Z, ...).")
    if arr.shape[0] != len(z_planes_nm):
        raise ValueError("Volume stack first axis must match z plane count.")
    weights, metadata = volume_plane_weights(params, np.asarray(z_planes_nm, dtype=float))
    output_mode = str(params.get("volume_output_mode", "integrated_projection")).strip().lower()
    if output_mode == "z_stack":
        return arr.copy(), {**metadata, "volume_combination": "z_stack"}
    if output_mode != "integrated_projection":
        raise ValueError("volume_output_mode must be 'integrated_projection' or 'z_stack'.")
    shape = (weights.size,) + (1,) * (arr.ndim - 1)
    combined = np.sum(arr * weights.reshape(shape), axis=0)
    return combined, {**metadata, "volume_combination": "weighted_integrated_projection"}


def params_for_focus_plane(params: dict, z_plane_nm: float) -> dict:
    """Return params whose particle initial z positions are shifted by a focus plane."""
    out = deepcopy(params)
    out["volumetric_imaging_mode"] = "single_plane"
    out["scene_dimensionality"] = "single_plane_particle_scene"
    out["num_frames"] = 1
    out["duration_seconds"] = max(float(out.get("duration_seconds", 1.0)), 1.0 / float(out.get("fps", 1.0)))
    for particle in out.get("particles", []) or []:
        motion = particle.setdefault("motion", {})
        pos = motion.get("initial_position_nm")
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        coords = list(np.asarray(pos, dtype=float).reshape(-1))
        if len(coords) < 3:
            coords = (coords + [0.0, 0.0, 0.0])[:3]
        coords[2] = float(coords[2]) - float(z_plane_nm)
        motion["initial_position_nm"] = coords
    return out


def holotomography_phase_projection_stack(
    volume: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a bounded projection-stack proxy for configured phase volumes."""
    arr = np.asarray(volume, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Holotomography projection stack expects a 3D volume.")
    angles = np.asarray(params.get("holotomography_projection_angles_deg", [0.0]), dtype=float).reshape(-1)
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("holotomography_projection_angles_deg must be finite and non-empty.")
    projections = []
    for angle in angles:
        # Bounded proxy: rotate only by nearest quadrant using array transforms
        # available without optional dependencies, then project along z.
        k = int(np.round(float(angle) / 90.0)) % 4
        rotated = np.rot90(arr, k=k, axes=(1, 2))
        projections.append(np.sum(rotated, axis=0))
    stack = np.asarray(projections, dtype=float)
    metadata = {
        "holotomography_model": "phase_projection_stack_quadrant_proxy",
        "holotomography_projection_angles_deg": angles.astype(float).tolist(),
        "holotomography_output_mode": str(params.get("holotomography_output_mode", "phase_projection_stack")),
        "projection_axis": "z_after_quadrant_rotation",
    }
    if str(params.get("holotomography_output_mode", "phase_projection_stack")).strip().lower() == "reconstruction_volume":
        metadata["reconstruction_volume_status"] = "not_requested_by_default_projection_stack_available"
    return stack, metadata

