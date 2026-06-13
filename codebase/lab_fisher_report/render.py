"""Rendering and image-conversion helpers for lab Fisher reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from camera_noise import (
    analysis_contrast_noise_model,
    analysis_noise_params_for_frame,
    camera_noise_metadata,
    detector_contrast_frames_for_analysis,
)
from fisher import (
    compute_off_axis_demodulated_localization_crlb,
    compute_localization_crlb,
    is_off_axis_holography_modality,
    lateral_derivative_plan_metadata,
    require_array_only_spectral_lateral_derivative_ready,
)
from imaging_models import get_imaging_model
from config import SamplingGeometry
from experiment_contracts import ConvergenceStatus
from noise_contracts import (
    resolve_fisher_likelihood_eligibility,
    summarize_analysis_noise_model,
)
from postprocessing import compute_single_frame_contrast
from simulation.latent_scene import _simulate_latent_scene
from simulation.scene_render import (
    _ensure_run_scope_detector_static_seed,
    _render_scene_with_params,
)
from simulation.units import _canonical_measurement_domain_and_signal_units

from .microscopes import MicroscopeSpec, resolve_microscope_params
from .scene_view import scene_provenance_from_params

__all__ = ["_density_uint8", "_display_uint8", "_render_microscope"]


def _render_microscope(
    base_params: dict[str, Any],
    microscope: MicroscopeSpec,
    *,
    shared_params: dict[str, Any] | None = None,
    resolved_params: dict[str, Any] | None = None,
    latent_scene_view: dict[str, Any] | None = None,
    shared_latent_metadata: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Render one microscope candidate and compute its Fisher/CRLB records.

    The comparison identity is the microscope name; the modality is backend
    metadata installed into effective params immediately before normalization.
    Keeping this seam microscope-owned makes same-modality microscopes with
    different overlays possible once the coordinator is rekeyed.
    """

    params = (
        dict(resolved_params)
        if resolved_params is not None
        else resolve_microscope_params(
            microscope,
            base_params,
            shared_params=shared_params,
        )
    )
    modality = microscope.modality
    microscope_name = microscope.name
    _ensure_run_scope_detector_static_seed(params)
    latent_scene = (
        dict(latent_scene_view)
        if latent_scene_view is not None
        else _simulate_latent_scene(params)
    )
    rendered = _render_scene_with_params(
        params,
        latent_scene,
        save_video_output=False,
        return_frames=True,
    )
    if rendered is None or not rendered.get("frames", np.empty((0, 1, 0, 0))).size:
        raise RuntimeError(f"no simulation frames returned for microscope {microscope_name!r} (modality {modality!r})")

    metadata = rendered.get("metadata", {})
    ideal_signal_frames = [np.asarray(frame, dtype=float) for frame in metadata.get("ideal_signal_frames", [])]
    ideal_reference_frames = [np.asarray(frame, dtype=float) for frame in metadata.get("ideal_reference_frames", [])]
    detector_input_signal_frames = [
        np.asarray(frame, dtype=float)
        for frame in metadata.get("detector_input_signal_frames", [])
    ] or ideal_signal_frames
    detector_input_reference_frames = [
        np.asarray(frame, dtype=float)
        for frame in metadata.get("detector_input_reference_frames", [])
    ] or ideal_reference_frames
    detector_object_field_frames = [
        np.asarray(frame, dtype=np.complex128)
        for frame in metadata.get("detector_object_field_frames", [])
    ]

    if not detector_input_signal_frames or not detector_input_reference_frames:
        raise RuntimeError(
            "rendering pipeline did not return detector-input signal/reference frames; "
            "set return_ideal_float_frames=True."
        )

    if len(detector_input_signal_frames) != len(detector_input_reference_frames):
        raise RuntimeError(
            "detector_input_signal_frames and detector_input_reference_frames length mismatch "
            f"({len(detector_input_signal_frames)} vs {len(detector_input_reference_frames)})."
        )
    if is_off_axis_holography_modality(modality) and len(detector_object_field_frames) != len(detector_input_signal_frames):
        raise RuntimeError(
            "off-axis DHM Fisher requires detector_object_field_frames aligned "
            "with detector_input_signal_frames; got "
            f"{len(detector_object_field_frames)} and {len(detector_input_signal_frames)}."
        )

    model = get_imaging_model(params)
    render_metadata = dict(metadata.get("render_metadata", {}) or {})
    # Frame-aware likelihood params are mandatory for phase-domain QPI when the
    # renderer has produced per-frame coherent-reference quanta.  A global
    # params+exposure merge would drop that map and make Fisher depend on an
    # absent or silently scalar photon basis.
    metadata_noise_params = analysis_noise_params_for_frame(
        params,
        render_metadata,
        frame_index=0,
    )
    response_function = dict(render_metadata.get("response_function", {}) or {})
    if not response_function:
        response_function = model.compute_response_function(detector_input_signal_frames[0].shape, params)
    measurement_domain, signal_units = _canonical_measurement_domain_and_signal_units(
        params,
        model,
        modality,
        response_function=response_function,
    )
    detector_meta = camera_noise_metadata(metadata_noise_params)
    output_type = getattr(model, "output_type", "intensity")

    pixel_size_nm = SamplingGeometry.from_params(params).detector_pixel_size_nm

    def _crlb_for_frame(
        frame_index: int,
        contrast: np.ndarray,
        noise_input,
        *,
        signal: np.ndarray,
        reference: np.ndarray | None,
        object_field: np.ndarray | None,
    ) -> tuple[dict[str, Any], Any]:
        if is_off_axis_holography_modality(modality):
            crlb_result, observation = compute_off_axis_demodulated_localization_crlb(
                signal,
                reference,
                params,
                pixel_size_nm,
                response_function=response_function,
                object_field_detector=object_field,
            )
            return crlb_result, observation.noise_model
        require_array_only_spectral_lateral_derivative_ready(
            modality=modality,
            params=params,
            model=model,
            response_function=response_function,
            num_particles=len(latent_scene.particle_instances),
            structured_environment_active=bool(
                render_metadata.get("sample_environment_active", False)
                or render_metadata.get("sample_environment_pattern_enabled", False)
            ),
            context=(
                f"lab Fisher report microscope {microscope_name!r} "
                f"(modality {modality!r}), frame {frame_index}"
            ),
        )
        return compute_localization_crlb(
            contrast,
            noise_input,
            pixel_size_nm,
            signal_units=signal_units,
            measurement_domain=measurement_domain,
        ), noise_input

    per_frame: list[dict[str, Any]] = []
    fisher_matrices: list[np.ndarray] = []
    for frame_index, (signal, reference) in enumerate(zip(detector_input_signal_frames, detector_input_reference_frames)):
        frame_noise_params = analysis_noise_params_for_frame(
            params,
            render_metadata,
            frame_index=frame_index,
        )
        if output_type == "phase":
            contrast_signal, contrast_reference = signal, reference
        else:
            contrast_signal, contrast_reference = detector_contrast_frames_for_analysis(
                signal,
                reference,
                frame_noise_params,
            )
        contrast = compute_single_frame_contrast(contrast_signal, contrast_reference, params)
        if contrast is None:
            raise RuntimeError("contrast generation returned no frame.")
        contrast = np.asarray(contrast, dtype=float)
        noise_model = analysis_contrast_noise_model(
            signal,
            reference,
            frame_noise_params,
        )
        noise_summary = summarize_analysis_noise_model(
            noise_model,
            expected_shape=contrast.shape,
            context=f"contrast frame shape for microscope {microscope_name!r} (modality {modality!r}), frame {frame_index}",
        )
        crlb, fisher_noise_model = _crlb_for_frame(
            frame_index,
            contrast,
            noise_model,
            signal=signal,
            reference=reference,
            object_field=(
                detector_object_field_frames[frame_index]
                if is_off_axis_holography_modality(modality)
                else None
            ),
        )
        fisher_noise_summary = (
            summarize_analysis_noise_model(
                fisher_noise_model,
                expected_shape=contrast.shape,
                context=(
                    f"Fisher likelihood shape for microscope {microscope_name!r} "
                    f"(modality {modality!r}), frame {frame_index}"
                ),
            )
            if fisher_noise_model is not noise_model
            else noise_summary
        )
        noise_var = fisher_noise_summary.diagonal_variance
        crlb.update(lateral_derivative_plan_metadata())
        # Fisher/CRLB production eligibility is a cross-seam contract: the
        # contrast-domain noise model and the detector-transfer metadata must
        # both be safe for the Fisher variance model actually used.  A finite
        # CRLB alone is allowed as a diagnostic, but must not enter ranking or
        # fusion when detector transfer remains diagnostic-only.
        eligibility = resolve_fisher_likelihood_eligibility(
            fisher_noise_model,
            detector_meta,
            crlb,
            context=f"lab Fisher report microscope {microscope_name!r} (modality {modality!r}), frame {frame_index}",
        )
        fisher = np.asarray(crlb["fisher_matrix"], dtype=float)
        if not np.all(np.isfinite(fisher)):
            convergence_status = ConvergenceStatus.NONFINITE.value
        elif bool(crlb.get("singular", False)):
            convergence_status = ConvergenceStatus.STABLE_SINGULAR.value
        elif bool(eligibility.safe_for_ordering) and bool(eligibility.safe_for_fusion):
            convergence_status = ConvergenceStatus.FINITE_CONVERGED.value
        else:
            convergence_status = ConvergenceStatus.PRODUCTION_GRID_ONLY.value
        fisher_matrices.append(fisher)
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "latent_scene_id": (
                    ""
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("latent_scene_id", "")
                ),
                "latent_schedule_id": (
                    ""
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("latent_schedule_id", "")
                ),
                "observation_time_s": (
                    ""
                    if shared_latent_metadata is None
                    else (
                        shared_latent_metadata.get("observation_times_s", [""])[frame_index]
                        if frame_index < len(shared_latent_metadata.get("observation_times_s", []))
                        else ""
                    )
                ),
                "state_time_policy": (
                    ""
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("state_time_policy", "")
                ),
                "fusion_time_alignment": (
                    ""
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("fusion_time_alignment", "")
                ),
                "shared_coordinate_frame": (
                    ""
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("shared_coordinate_frame", "")
                ),
                "same_latent_scene": bool(
                    False
                    if shared_latent_metadata is None
                    else shared_latent_metadata.get("same_latent_scene", False)
                ),
                "contrast": contrast,
                "noise_variance": noise_var,
                # Keep the full likelihood object for downstream Fisher-density
                # artifacts. The diagonal summary remains for tables, but it is
                # not a sufficient scientific basis when scan-line covariance is present.
                "analysis_noise_model": fisher_noise_model,
                "analysis_noise_covariance_kind": fisher_noise_summary.covariance_kind,
                "crlb": crlb,
                "fisher_matrix": fisher,
                "measurement_domain": measurement_domain,
                "signal_units": signal_units,
                "noise_variance_units": crlb.get("noise_variance_units"),
                "detector_noise_input_domain": detector_meta.get("detector_noise_input_domain", ""),
                "nonlinear_detector_effects_active": bool(detector_meta.get("nonlinear_detector_effects_active", False)),
                "deterministic_detector_transfer_active": bool(detector_meta.get("deterministic_detector_transfer_active", False)),
                "safe_for_linear_fisher_variance": bool(detector_meta.get("safe_for_linear_fisher_variance", False)),
                "safe_for_covariance_fisher_variance": bool(detector_meta.get("safe_for_covariance_fisher_variance", False)),
                "safe_for_ordering": bool(eligibility.safe_for_ordering),
                "safe_for_fusion": bool(eligibility.safe_for_fusion),
                "detector_safe_for_report_fisher": bool(eligibility.detector_safe_for_report_fisher),
                "fisher_likelihood_uses_covariance": bool(eligibility.used_covariance_fisher),
                "fisher_likelihood_eligibility_contract_id": eligibility.contract_id,
                "fisher_variance_model_scope": detector_meta.get("fisher_variance_model_scope", ""),
                "covariance_fisher_variance_model_scope": detector_meta.get("covariance_fisher_variance_model_scope", ""),
                "detector_likelihood_status": detector_meta.get("detector_likelihood_status", ""),
                "fisher_noise_covariance_model": crlb.get("fisher_noise_covariance_model", ""),
                "derivative_basis": crlb.get("derivative_basis", ""),
                "nyquist_band_fraction": crlb.get("nyquist_band_fraction", ""),
                "boundary_energy_fraction": crlb.get("boundary_energy_fraction", ""),
                "convergence_status": convergence_status,
                "status_reason": eligibility.status_reason,
            }
        )

    return {
        "per_frame": per_frame,
        "fisher_matrices": fisher_matrices,
    }, {
        "resolved_params": params,
        "microscope_name": microscope_name,
        "modality": modality,
        "num_frames": len(per_frame),
        "scene_provenance": scene_provenance_from_params(params),
        "shared_latent_scene": dict(shared_latent_metadata or {}),
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
