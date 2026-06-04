"""Shared imports, constants, and scalar helpers for substrate patterns."""
from __future__ import annotations
import math
import os
import numpy as np
from threading import RLock
from typing import Dict, Tuple, Optional

from config import param_value
from config.runtime import (
    resolved_sample_environment_pattern_dimension,
    resolved_sample_environment_pattern_dimensions,
)
from param_schema.sample_environment import BAR_ORIENTATION_CHOICES, PATTERN_DEFAULT_PRESETS

_MAX_SHAPE_AXIS_DISTORTION_FRAC = 0.25
_MIN_SHAPE_RADIUS_FACTOR = 0.5
_MIN_EDGE_RADIUS_FACTOR = 0.05
_REFLECTION_BOUNDARY_BISECTION_STEPS = 20
_CIRCULAR_VOID_PATTERNS = {"gold_holes", "holey_carbon"}
_CIRCULAR_MATERIAL_PATTERNS = {"nanopillars", "fiducial_dots", "patterned_coverslip"}
_BAR_MATERIAL_PATTERNS = {"grid_bars", "microfluidic_walls"}
_LAYOUT_CACHE: Dict[Tuple, object] = {}
_LAYOUT_CACHE_LOCK = RLock()

try:
    import cv2 as cv2
except ImportError:
    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                "OpenCV (cv2) is required for substrate roughness/image IO "
                f"operation {name!r}."
            )

    cv2 = _MissingCV2()

def _pattern_dimensions(params: dict) -> dict:
    return resolved_sample_environment_pattern_dimensions(params)

def _substrate_pattern_is_enabled(params: dict) -> bool:
    return (
        bool(param_value(params, 'sample_environment_enabled'))
        and bool(param_value(params, 'sample_environment_pattern_enabled'))
    )

def canonical_sample_environment_pattern_and_preset(pattern: object, preset: object) -> tuple[str, str]:
    """Return stripped canonical sample-environment pattern and preset values."""
    p = str(pattern).strip().lower()
    q = str(preset).strip().lower()
    return p, q

def _read_positive_pattern_dimension(params: dict, key: str, default: float | None = None) -> float:
    if default is None:
        value = resolved_sample_environment_pattern_dimension(params, key)
    else:
        dims = _pattern_dimensions(params)
        value = dims[key] if key in dims and dims[key] is not None else default
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"sample_environment_pattern_dimensions[{key!r}] must be finite and positive.")
    return value

def _dimension_um(params: dict, key: str, default: float | None = None) -> float:
    return _read_positive_pattern_dimension(params, key, default)

def _dimension_um_from_keys(
    params: dict,
    um_key: str,
    nm_key: str,
    default_um: float,
) -> float:
    dims = _pattern_dimensions(params)
    if um_key in dims and dims[um_key] is not None:
        return _dimension_um(params, um_key, default_um)
    if nm_key in dims and dims[nm_key] is not None:
        value = float(dims[nm_key]) * 1.0e-3
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"sample_environment_pattern_dimensions[{nm_key!r}] must be finite and positive.")
        return value
    return float(default_um)

def _dimension_factor(params: dict, key: str, default: float | None = None) -> float:
    return _read_positive_pattern_dimension(params, key, default)

def _centered_pattern_grid(shape: tuple, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    pixel_size_um = float(pixel_size_nm) * 1e-3
    x_um = (np.arange(width, dtype=float) - width / 2.0 + 0.5) * pixel_size_um
    y_um = (np.arange(height, dtype=float) - height / 2.0 + 0.5) * pixel_size_um
    return np.meshgrid(x_um, y_um)

def _canonical_bar_orientation(orientation: object) -> str:
    normalized = str(orientation).strip().lower()
    if normalized in BAR_ORIENTATION_CHOICES:
        return normalized
    raise ValueError(f"Bar/wall orientation must be one of {BAR_ORIENTATION_CHOICES!r}.")

def _bar_solid_mask_from_coordinates(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    pitch_um: float,
    width_um: float,
    orientation: str,
    clearance_um: float = 0.0,
) -> np.ndarray:
    half_pitch = float(pitch_um) / 2.0
    half_width = 0.5 * float(width_um) + max(float(clearance_um), 0.0)
    if half_width >= half_pitch:
        return np.ones_like(np.asarray(x_um, dtype=float), dtype=bool)
    x_mod = (np.asarray(x_um, dtype=float) + half_pitch) % float(pitch_um) - half_pitch
    y_mod = (np.asarray(y_um, dtype=float) + half_pitch) % float(pitch_um) - half_pitch
    orientation = _canonical_bar_orientation(orientation)
    if orientation == "vertical":
        return np.abs(x_mod) <= half_width
    if orientation == "horizontal":
        return np.abs(y_mod) <= half_width
    if orientation == "both":
        return (np.abs(x_mod) <= half_width) | (np.abs(y_mod) <= half_width)
    raise ValueError(f"Bar/wall orientation must be one of {BAR_ORIENTATION_CHOICES!r}.")

def _pattern_intensity_from_material_fraction(
    material_fraction: np.ndarray,
    *,
    material_factor: float,
    background_factor: float,
) -> np.ndarray:
    fraction = np.asarray(material_fraction, dtype=float)
    pattern = (
        fraction * float(material_factor)
        + (1.0 - fraction) * float(background_factor)
    )
    mean_val = float(pattern.mean())
    if mean_val > 0.0:
        pattern /= mean_val
    return pattern.astype(float)

__all__ = [
    "Dict",
    "Optional",
    "PATTERN_DEFAULT_PRESETS",
    "RLock",
    "Tuple",
    "_BAR_MATERIAL_PATTERNS",
    "_CIRCULAR_MATERIAL_PATTERNS",
    "_CIRCULAR_VOID_PATTERNS",
    "_LAYOUT_CACHE",
    "_LAYOUT_CACHE_LOCK",
    "_MAX_SHAPE_AXIS_DISTORTION_FRAC",
    "_MIN_EDGE_RADIUS_FACTOR",
    "_MIN_SHAPE_RADIUS_FACTOR",
    "_REFLECTION_BOUNDARY_BISECTION_STEPS",
    "_bar_solid_mask_from_coordinates",
    "_canonical_bar_orientation",
    "_centered_pattern_grid",
    "_dimension_factor",
    "_dimension_um",
    "_dimension_um_from_keys",
    "_pattern_dimensions",
    "_pattern_intensity_from_material_fraction",
    "_read_positive_pattern_dimension",
    "_substrate_pattern_is_enabled",
    "canonical_sample_environment_pattern_and_preset",
    "cv2",
    "math",
    "np",
    "os",
]
