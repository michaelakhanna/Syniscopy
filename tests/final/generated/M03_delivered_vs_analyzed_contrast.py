"""M03 [provenance guard] Delivered frames vs analyzed contrast.

This checks that the floating analysis contrast used by the CRLB path is kept
separate from the public delivered display frames, and that each public frame
surface is internally self-consistent. It compares:

  1. metadata["analysis_contrast_frames"], the quantitative Fisher-safe contrast
  2. metadata["contrast_frames_float"], the raw-observation contrast preview basis
  3. result["frames"] and metadata["background_subtracted_frames"], which are the
     public dataset/video frames after normalize_contrast_frames(...)

Run:
    python M03_delivered_vs_analyzed_contrast.py
Writes:
    _runs/M03/M03_results.json
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import add_paths, banner, verdict

add_paths()

from calibration_profiles import CALIBRATION_PROFILES, native_params
from simulation import run_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_runs", "M03")
os.makedirs(OUT, exist_ok=True)


def _scene_params() -> dict:
    case = dict(CALIBRATION_PROFILES["interferometric"])
    case.update(
        {
            "modality": "interferometric",
            "particle_material": "gold",
            "diameter_nm": 60.0,
            "image_size_pixels": 32,
            "pixel_size_nm": 20.0,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "psf_oversampling_factor": 1,
            "numerical_aperture": 1.0,
            "wavelength_nm": 532.0,
            "background_intensity": 20000.0,
        }
    )
    params = native_params(case)
    params.update(
        {
            "num_frames": 3,
            "fps": 24.0,
            "duration_seconds": 3.0 / 24.0,
            "random_seed": 2025,
            "output_filename": os.path.join(OUT, "M03_preview.avi"),
            "mask_generation_enabled": False,
            "save_frame_sequence": False,
            "save_raw_camera_video": False,
            "save_raw_camera_frame_sequence": False,
            "save_raw_frame_views": False,
            "return_ideal_float_frames": True,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "background_subtraction_method": "reference_frame",
        }
    )
    return params


def _stack(name: str, frames) -> np.ndarray:
    if frames is None or len(frames) == 0:
        raise ValueError(f"{name} is empty")
    arr = np.asarray([np.asarray(frame) for frame in frames])
    if arr.ndim != 3:
        raise ValueError(f"{name} must stack to (T,H,W); got {arr.shape}")
    return arr


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _max_relative(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1.0e-30)
    return float(np.max(np.abs(aa - bb) / denom))


banner("M03  delivered-vs-analyzed contrast provenance")

params = _scene_params()
result = run_simulation(params, return_frames=True)
metadata = dict(result.get("metadata", {}) or {})

analysis = _stack("analysis_contrast_frames", metadata.get("analysis_contrast_frames"))
contrast_float = _stack("contrast_frames_float", metadata.get("contrast_frames_float"))
metadata_delivered = _stack("background_subtracted_frames", metadata.get("background_subtracted_frames"))

frames = np.asarray(result.get("frames"))
if frames.ndim != 4 or frames.shape[1] != 1:
    raise ValueError(f"result['frames'] must have shape (T,1,H,W); got {frames.shape}")
returned_delivered = np.asarray(frames[:, 0, :, :])

same_domain_rel = _max_relative(analysis, contrast_float)
returned_vs_metadata_abs = float(
    np.max(np.abs(returned_delivered.astype(float) - metadata_delivered.astype(float)))
)
same_shape = analysis.shape == contrast_float.shape == returned_delivered.shape
analysis_quantitative = bool(metadata.get("analysis_contrast_frame_quantitative"))
analysis_safe = bool(metadata.get("analysis_contrast_frame_safe_for_fisher"))
raw_preview_display_only = metadata.get("raw_observation_contrast_frame_units") == "display_only"

ok_surfaces_present = same_shape and np.isfinite(same_domain_rel)
ok_returned_metadata = returned_vs_metadata_abs == 0.0
ok_analysis_contract = analysis_quantitative and analysis_safe and raw_preview_display_only
all_ok = bool(ok_surfaces_present and ok_returned_metadata and ok_analysis_contract)

payload = {
    "modality": "interferometric",
    "frame_count": int(analysis.shape[0]),
    "analysis_shape": list(analysis.shape),
    "delivered_shape": list(returned_delivered.shape),
    "analysis_vs_raw_observation_contrast_frames_max_relative_difference": same_domain_rel,
    "returned_frames_vs_metadata_background_subtracted_max_abs_difference": returned_vs_metadata_abs,
    "same_shape": bool(same_shape),
    "analysis_contrast_frame_quantitative": bool(analysis_quantitative),
    "analysis_contrast_frame_safe_for_fisher": bool(analysis_safe),
    "raw_observation_contrast_frame_units": metadata.get("raw_observation_contrast_frame_units"),
    "surfaces_present_ok": bool(ok_surfaces_present),
    "returned_metadata_ok": bool(ok_returned_metadata),
    "analysis_contract_ok": bool(ok_analysis_contract),
}

print(
    f"  [{'OK' if ok_surfaces_present else 'FAIL'}] analysis/raw-observation "
    f"contrast surfaces are present with same shape; max_rel={same_domain_rel:.3e}"
)
print(
    f"  [{'OK' if ok_returned_metadata else 'FAIL'}] result['frames'] == "
    f"metadata background_subtracted_frames  max_abs={returned_vs_metadata_abs:.3g}"
)
print(
    f"  [{'OK' if ok_analysis_contract else 'FAIL'}] analysis contrast metadata is "
    f"quantitative/Fisher-safe and raw observation is display-only"
)

out_path = os.path.join(OUT, "M03_results.json")
json.dump(payload, open(out_path, "w"), indent=2)
print(f"\nWROTE {out_path}")
print("Any FAIL means the CRLB/provenance path or public delivered-frame contract is inconsistent.")
raise SystemExit(verdict(all_ok, "(delivered and analyzed contrast surfaces are explicitly separated and consistent)"))
