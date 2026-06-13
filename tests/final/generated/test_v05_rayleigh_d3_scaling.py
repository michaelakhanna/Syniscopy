from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import builtins
from typing import Sequence

import numpy as np
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for substrate-dependent bootstrap; missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _particle_frame(diameter_nm: float, params: dict) -> float:
    from simulation import generate_single_frame_views

    p = deepcopy(params)
    p["particles"] = [deepcopy(params["particles"][0])]
    p["particles"][0]["components"][0]["diameter_nm"] = float(diameter_nm)
    p["particles"][0]["motion"]["hydrodynamic_diameter_nm"] = float(diameter_nm)

    out = generate_single_frame_views(p)
    contrast = np.asarray(out["contrast_frame"], dtype=float)
    return float(np.max(np.abs(contrast)))


def _fit_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    assert np.count_nonzero(finite) >= 2
    x_fit = x_arr[finite]
    y_fit = y_arr[finite]
    return float(np.polyfit(np.log(x_fit), np.log(y_fit), 1)[0])


def test_rayleigh_fisher_scaling_control_matches_d_cubed_expectation() -> None:
    from fisher import compute_rayleigh_amplitude_scaling_control

    diameters = np.array([20.0, 30.0, 40.0, 60.0, 80.0], dtype=float)
    control = compute_rayleigh_amplitude_scaling_control(diameters, reference_diameter_nm=50.0)

    slope = _fit_slope(control["diameter_nm"], control["sigma_xy_nm"])
    assert slope == pytest.approx(-3.0, abs=0.20)


def test_rayleigh_rendering_peak_scales_as_d_cubed_in_small_particle_regime() -> None:
    _require_cv2_for_bootstrap()
    from config import default_params

    params = deepcopy(default_params())
    params.update(
        {
            "imaging_model": "interferometric",
            "background_subtraction_method": "reference_frame",
            "shot_noise_enabled": False,
            "gaussian_noise_enabled": False,
            "random_seed": 7,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "num_frames": 1,
            "fps": 1.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "image_size_pixels": 64,
        }
    )

    diameters = np.array([10.0, 14.0, 20.0, 28.0, 40.0], dtype=float)
    peaks = np.array([_particle_frame(float(d), params) for d in diameters], dtype=float)
    assert np.all(np.isfinite(peaks))
    assert np.all(peaks > 0.0)

    slope = _fit_slope(diameters, peaks)
    assert slope == pytest.approx(3.0, abs=0.25)
