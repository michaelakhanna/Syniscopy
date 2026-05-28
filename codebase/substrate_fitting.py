"""
Reference-background calibration helpers for sample-environment patterns.

The fitter summarizes background-only or particle-sparse real frames as a
``sample_environment_pattern_*`` parameter dictionary for the structured
sample-environment model.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# Image-processing heuristics for background-only reference frames. Lag
# fractions bound the autocorrelation search window; the percentile and minimum
# area select dark-hole candidates after mild smoothing.
_FLAT_PROFILE_STD_EPS = 1e-12
_AUTOCORR_MIN_LAG_FRACTION = 0.03
_AUTOCORR_MAX_LAG_FRACTION = 0.75
_AUTOCORR_MIN_LAG_PX = 2
_BACKGROUND_SMOOTH_SIGMA_PX = 2.0
_MIN_HOLE_COMPONENT_AREA_PX = 4
_MIN_BACKGROUND_CONTRAST_EPS = 1e-9


def _as_background_image(real_background_frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(real_background_frames, dtype=float)
    if arr.ndim == 2:
        image = arr
    elif arr.ndim == 3:
        image = np.median(arr, axis=0)
    else:
        raise ValueError(
            "real_background_frames must be a 2D image or a 3D frame stack."
        )
    if image.size == 0:
        raise ValueError("real_background_frames cannot be empty.")
    return image


def _estimate_period_from_autocorrelation(profile: np.ndarray) -> float | None:
    profile = np.asarray(profile, dtype=float)
    profile = profile - float(np.mean(profile))
    if float(np.std(profile)) <= _FLAT_PROFILE_STD_EPS:
        return None

    corr = np.correlate(profile, profile, mode="full")
    corr = corr[corr.size // 2:]
    corr[0] = 0.0
    if corr.size < 5:
        return None

    # Ignore near-zero lags and the noisiest tail when searching for the first
    # repeat distance in the one-dimensional autocorrelation profile.
    lo = max(_AUTOCORR_MIN_LAG_PX, int(_AUTOCORR_MIN_LAG_FRACTION * profile.size))
    hi = max(lo + 1, int(_AUTOCORR_MAX_LAG_FRACTION * profile.size))
    window = corr[lo:hi]
    if window.size == 0 or float(np.max(window)) <= 0.0:
        return None
    return float(lo + int(np.argmax(window)))


def fit_substrate_pattern(
    real_background_frames: np.ndarray,
    *,
    pixel_size_nm: float,
    model: str = "gold_holes",
) -> dict[str, Any]:
    """
    Estimate substrate-pattern parameters from reference-background frames.

    The supported model is ``gold_holes``. The returned dictionary estimates
    lattice period from 1D autocorrelation of the median background, estimates
    feature radius from low-intensity connected components, and reports
    jitter/edge perturbation as unresolved. The low-intensity component rule
    assumes holes are darker than the surrounding film in the supplied reference
    frames.
    """
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive.")
    model = str(model).strip().lower()
    if model != "gold_holes":
        raise ValueError("fit_substrate_pattern supports only gold_holes.")

    image = _as_background_image(real_background_frames)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("real_background_frames contains no finite pixels.")

    fill = float(np.nanmedian(image[finite]))
    image = np.where(finite, image, fill)
    smoothed = cv2.GaussianBlur(
        image.astype(np.float32),
        (0, 0),
        _BACKGROUND_SMOOTH_SIGMA_PX,
    )

    row_period = _estimate_period_from_autocorrelation(np.mean(smoothed, axis=0))
    col_period = _estimate_period_from_autocorrelation(np.mean(smoothed, axis=1))
    periods = [p for p in (row_period, col_period) if p is not None and p > 0.0]
    period_px = float(np.median(periods)) if periods else None

    p05, p95 = np.percentile(smoothed, [5.0, 95.0])
    if float(p95 - p05) <= _MIN_BACKGROUND_CONTRAST_EPS:
        low_mask = np.zeros(smoothed.shape, dtype=np.uint8)
    else:
        norm = cv2.normalize(
            smoothed,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)
        _threshold, low_mask_u8 = cv2.threshold(
            norm,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        low_mask = (low_mask_u8 > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask, 8)
    areas = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= _MIN_HOLE_COMPONENT_AREA_PX:
            areas.append(area)
    radius_px = float(np.sqrt(np.median(areas) / np.pi)) if areas else None

    pixel_size_um = float(pixel_size_nm) * 1e-3
    if period_px is not None:
        pitch_um = period_px * pixel_size_um
    else:
        pitch_um = None
    if radius_px is not None:
        hole_diameter_um = 2.0 * radius_px * pixel_size_um
    else:
        hole_diameter_um = None

    if pitch_um is not None and hole_diameter_um is not None:
        spacing_um = max(0.0, pitch_um - hole_diameter_um)
    else:
        spacing_um = None

    low_values = smoothed[low_mask > 0]
    high_values = smoothed[low_mask == 0]
    if low_values.size and high_values.size and float(np.mean(high_values)) > 0.0:
        hole_intensity_factor = float(np.mean(low_values) / np.mean(high_values))
    else:
        hole_intensity_factor = 1.0

    return {
        "calibration_type": "reference-background calibration",
        "sample_environment_pattern": "gold_holes",
        "sample_environment_pattern_preset": "default_gold_holes",
        "sample_environment_pattern_dimensions": {
            "hole_diameter_um": hole_diameter_um,
            "hole_edge_to_edge_spacing_um": spacing_um,
            "hole_intensity_factor": hole_intensity_factor,
            "gold_intensity_factor": 1.0,
        },
        "sample_environment_pattern_randomization_enabled": False,
        "sample_environment_pattern_position_jitter_std_nm": None,
        "sample_environment_pattern_shape_regularity": None,
        "sample_environment_pattern_edge_perturbation_max_rel_radius": None,
        "fit_quality": {
            "period_px": period_px,
            "radius_px": radius_px,
            "row_period_px": row_period,
            "col_period_px": col_period,
            "method": "median_background_autocorrelation_otsu_connected_components",
            "threshold_rule": "otsu_dark_features_after_gaussian_smoothing",
            "background_contrast_p05_p95": float(p95 - p05),
        },
    }
