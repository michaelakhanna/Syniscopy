from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _first_frame(result: dict, key: str) -> np.ndarray | None:
    metadata = result.get("metadata", {})
    frames = metadata.get(key) or []
    if not frames:
        return None
    return np.asarray(frames[0], dtype=float)


def _build_base_case(modality: str) -> dict:
    from calibration_profiles import CALIBRATION_PROFILES, native_params
    from modality_registry import canonical_modality_name

    canon = canonical_modality_name(modality)
    case = dict(CALIBRATION_PROFILES.get(canon, {"modality": canon}))
    case["modality"] = canon
    case.update(
        {
            "image_size_pixels": 32,
            "pixel_size_nm": 22.0,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "psf_oversampling_factor": 1,
            "num_frames": 1,
            "background_subtraction_method": "reference_frame",
        }
    )
    return native_params(case)


def _roi_mask(frame: np.ndarray) -> np.ndarray:
    mean_delta = frame - np.median(frame)
    scale = float(np.abs(mean_delta).max())
    if not np.isfinite(scale) or scale <= 0:
        return np.ones_like(frame, dtype=bool)
    mask = np.abs(mean_delta) > 0.2 * scale
    return mask if np.any(mask) else np.ones_like(frame, dtype=bool)


def _ratio_for_modality(modality: str, n_frames: int = 12) -> float:
    from camera_noise import analysis_contrast_noise_variance
    from modality_registry import modality_uses_relative_reference_contrast
    from simulation import run_simulation

    base = _build_base_case(modality)

    clean = deepcopy(base)
    clean["random_seed"] = 1000
    baseline = run_simulation(clean, return_frames=True)

    ideal_signal = _first_frame(baseline, "ideal_signal_frames")
    ideal_reference = _first_frame(baseline, "ideal_reference_frames")
    assert ideal_signal is not None, "baseline ideal signal missing"

    relative_reference = modality_uses_relative_reference_contrast(modality)
    analytic_variance = np.asarray(
        analysis_contrast_noise_variance(
            ideal_signal,
            ideal_reference,
            clean,
            relative_reference=relative_reference,
        ),
        dtype=float,
    )

    noisy_frames = []
    for idx in range(n_frames):
        params = deepcopy(base)
        params["random_seed"] = 3000 + idx * 17
        result = run_simulation(params, return_frames=True)
        frame = _first_frame(result, "contrast_frames_float")
        if frame is not None:
            noisy_frames.append(frame)

    assert noisy_frames, f"no noisy contrast frames for {modality}"
    stacked = np.stack(noisy_frames, axis=0)
    empirical_var = np.var(stacked.astype(float), axis=0, ddof=1)

    mask = _roi_mask(np.mean(stacked.astype(float), axis=0))
    emp = float(np.mean(empirical_var[mask]))
    pred = float(np.mean(analytic_variance[mask]))

    assert np.isfinite(emp)
    assert np.isfinite(pred)
    assert emp > 0.0
    assert pred > 0.0
    return emp / pred


def test_l09_noise_prediction_matches_empirical_contrast_variance() -> None:
    for modality in ("interferometric", "dark_field", "quantitative_phase"):
        ratio = _ratio_for_modality(modality, n_frames=14)
        # Broad tolerance allows for Monte Carlo noise while still detecting drift
        # by >~2x mismatch in variance transfer, which has repeatedly been
        # the bug pattern for stale or wrong noise contracts.
        assert 0.35 <= ratio <= 3.0, f"{modality} ratio={ratio:.3f} outside expected range"
