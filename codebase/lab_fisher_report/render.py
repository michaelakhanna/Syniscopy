"""Rendering and image-conversion helpers for lab Fisher reports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from camera_noise import analysis_contrast_noise_variance, camera_noise_metadata
from fisher import compute_localization_crlb
from imaging_models import get_imaging_model
from postprocessing import compute_single_frame_contrast
from simulation.latent_scene import _simulate_latent_scene
from simulation.scene_render import _render_scene_with_params
from simulation.units import _canonical_measurement_domain_and_signal_units

__all__ = ["_density_uint8", "_display_uint8", "_render_modality"]


def _render_modality(
    base_params: dict[str, Any],
    modality: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    params = deepcopy(base_params)
    params["imaging_model"] = modality
    latent_scene = _simulate_latent_scene(params)
    rendered = _render_scene_with_params(
        params,
        latent_scene,
        save_video_output=False,
        return_frames=True,
    )
    if rendered is None or not rendered.get("frames", np.empty((0, 1, 0, 0))).size:
        raise RuntimeError(f"no simulation frames returned for modality {modality!r}")

    metadata = rendered.get("metadata", {})
    ideal_signal_frames = [np.asarray(frame, dtype=float) for frame in metadata.get("ideal_signal_frames", [])]
    ideal_reference_frames = [np.asarray(frame, dtype=float) for frame in metadata.get("ideal_reference_frames", [])]

    if not ideal_signal_frames or not ideal_reference_frames:
        raise RuntimeError(
            "rendering pipeline did not return ideal signal/reference frames; "
            "set return_ideal_float_frames=True."
        )

    if len(ideal_signal_frames) != len(ideal_reference_frames):
        raise RuntimeError(
            "ideal_signal_frames and ideal_reference_frames length mismatch "
            f"({len(ideal_signal_frames)} vs {len(ideal_reference_frames)})."
        )

    model = get_imaging_model(params)
    render_metadata = dict(metadata.get("render_metadata", {}) or {})
    noise_params = dict(params)
    effective_exposure_time_s = render_metadata.get("effective_exposure_time_s")
    if effective_exposure_time_s is not None:
        noise_params["exposure_time_s"] = float(effective_exposure_time_s)
    response_function = dict(render_metadata.get("response_function", {}) or {})
    if not response_function:
        response_function = model.compute_response_function(ideal_signal_frames[0].shape, params)
    measurement_domain, signal_units = _canonical_measurement_domain_and_signal_units(
        params,
        model,
        modality,
        response_function=response_function,
    )
    detector_meta = camera_noise_metadata(noise_params)

    pixel_size_nm = float(params["pixel_size_nm"])
    per_frame: list[dict[str, Any]] = []
    fisher_matrices: list[np.ndarray] = []
    for frame_index, (signal, reference) in enumerate(zip(ideal_signal_frames, ideal_reference_frames)):
        contrast = compute_single_frame_contrast(signal, reference, params)
        if contrast is None:
            raise RuntimeError("contrast generation returned no frame.")
        contrast = np.asarray(contrast, dtype=float)
        noise_var = analysis_contrast_noise_variance(
            signal,
            reference,
            noise_params,
        )
        noise_var = np.asarray(noise_var, dtype=float)
        if noise_var.shape != contrast.shape and noise_var.size != 1:
            raise RuntimeError(
                "Noise-variance shape does not match contrast frame shape "
                f"for modality {modality!r}, frame {frame_index}."
            )
        crlb = compute_localization_crlb(
            contrast,
            noise_var,
            pixel_size_nm,
            signal_units=signal_units,
            measurement_domain=measurement_domain,
        )
        fisher = np.asarray(crlb["fisher_matrix"], dtype=float)
        fisher_matrices.append(fisher)
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "contrast": contrast,
                "noise_variance": noise_var,
                "crlb": crlb,
                "fisher_matrix": fisher,
                "measurement_domain": measurement_domain,
                "signal_units": signal_units,
                "noise_variance_units": crlb.get("noise_variance_units"),
                "detector_noise_input_domain": detector_meta.get("detector_noise_input_domain", ""),
                "nonlinear_detector_effects_active": bool(detector_meta.get("nonlinear_detector_effects_active", False)),
                "deterministic_detector_transfer_active": bool(detector_meta.get("deterministic_detector_transfer_active", False)),
                "safe_for_linear_fisher_variance": bool(detector_meta.get("safe_for_linear_fisher_variance", True)),
                "fisher_variance_model_scope": detector_meta.get("fisher_variance_model_scope", ""),
                "detector_likelihood_status": detector_meta.get("detector_likelihood_status", ""),
            }
        )

    return {
        "per_frame": per_frame,
        "fisher_matrices": fisher_matrices,
    }, {
        "resolved_params": params,
        "num_frames": len(per_frame),
    }


def _display_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    center = float(np.median(finite))
    spread = float(np.percentile(np.abs(finite - center), 99.5))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 1.0
    out = 0.5 + 0.42 * (arr - center) / spread
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def _density_uint8(density: np.ndarray) -> np.ndarray:
    arr = np.asarray(density, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    if float(arr.max(initial=0.0)) <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    logged = np.log1p(arr)
    hi = float(np.percentile(logged[logged > 0.0], 99.0)) if np.any(logged > 0.0) else 1.0
    if not np.isfinite(hi) or hi <= 0.0:
        hi = float(logged.max())
    return np.clip(255.0 * logged / hi, 0.0, 255.0).astype(np.uint8)
