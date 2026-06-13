from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _test_params() -> dict:
    from config import default_params

    params = default_params()
    params.update(
        {
            "imaging_model": "bright_field",
            "noise_model": {},
            "modality_noise": {},
            "detector_noise_input_domain": "camera_counts",
            "shot_noise_enabled": True,
            "gaussian_noise_enabled": True,
            "detector_qe": 1.0,
            "fluorescence_detector_qe": 1.0,
            "camera_gain_e_per_count": 1.0,
            "read_noise_counts": 2.5,
            "background_offset_counts": 0.0,
            "dark_offset_counts": 0.0,
            "dark_current_e_per_pixel_per_s": 0.0,
            "exposure_time_s": 1.0,
            "dark_frame_map": None,
            "random_seed": 1234,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "psf_oversampling_factor": 1,
            "image_size_pixels": 16,
            "pixel_size_nm": 50.0,
            "max_psf_z_slices": 1,
        }
    )
    return params


def _mean_empirical_pixel_variance(clean_signal: np.ndarray, params: dict, reps: int) -> float:
    from camera_noise import apply_camera_noise_counts

    clean_signal = np.asarray(clean_signal, dtype=float)
    rng = np.random.default_rng(20260)
    draws = np.stack(
        [apply_camera_noise_counts(clean_signal, params, rng=rng) for _ in range(int(reps))],
        axis=0,
    )
    return float(np.var(draws, axis=0, ddof=0).mean())


def test_total_noise_variance_matches_empirical_sampling() -> None:
    from camera_noise import total_noise_variance_counts

    params = _test_params()
    levels = np.array([20.0, 100.0, 400.0, 1600.0], dtype=float)
    reps = 200

    empirical = []
    predicted = []
    for level in levels:
        frame = np.full((28, 28), level, dtype=float)
        empirical.append(_mean_empirical_pixel_variance(frame, params, reps))
        predicted.append(float(np.asarray(total_noise_variance_counts(frame, params)).mean()))

    empirical = np.asarray(empirical, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(empirical - predicted) / np.where(predicted > 0.0, predicted, 1.0)
    assert np.all(np.isfinite(rel))
    assert np.nanmax(rel) < 0.18


def test_variance_is_affine_in_signal_under_model() -> None:
    from camera_noise import total_noise_variance_counts

    params = _test_params()
    levels = np.array([0.0, 50.0, 200.0, 800.0], dtype=float)
    predicted = [float(np.asarray(total_noise_variance_counts(np.full((20, 20), L), params)).mean()) for L in levels]
    predicted = np.asarray(predicted, dtype=float)

    slope, intercept = np.polyfit(levels, predicted, 1)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)
    assert slope > 0.0
    assert intercept >= 0.0
    assert np.mean(predicted) > 0.0


def test_qpi_reference_frame_phase_likelihood_does_not_add_independent_reference_variance() -> None:
    from camera_noise import (
        analysis_contrast_noise_model,
        analysis_contrast_noise_variance,
        camera_noise_metadata,
        qpi_phase_noise_variance_rad2,
    )

    params = _test_params()
    params.update(
        {
            "imaging_model": "quantitative_phase",
            "background_subtraction_method": "reference_frame",
            "shot_noise_enabled": True,
            "gaussian_noise_enabled": False,
            "qpi_visibility": 1.0,
            "qpi_detected_quanta_per_pixel": 100.0,
            "qpi_phase_noise_std_rad": None,
            "qpi_phase_to_count_scale": 100.0,
        }
    )
    signal = np.full((8, 8), 1000.0, dtype=float)
    reference = np.full((8, 8), 1000.0, dtype=float)

    single_phase_variance = qpi_phase_noise_variance_rad2(signal, params)
    analysis_variance = analysis_contrast_noise_variance(signal, reference, params)
    noise_model = analysis_contrast_noise_model(signal, reference, params)
    metadata = camera_noise_metadata(params)

    assert np.allclose(single_phase_variance, 0.01)
    assert np.allclose(analysis_variance, single_phase_variance)
    assert not np.allclose(analysis_variance, 2.0 * single_phase_variance)
    assert noise_model.measurement_domain == "phase"
    assert noise_model.signal_units == "radian"
    assert noise_model.noise_variance_units == "radian_squared"
    assert noise_model.covariance_kind == "independent_pixels"
    assert (
        metadata["analysis_reference_frame_noise_contract"]
        == "deterministic_reference_centering"
    )
    assert (
        metadata["analysis_reference_frame_variance_basis"]
        == "single_reference_normalized_phase_map_variance"
    )
