from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from camera_noise import analysis_contrast_noise_variance
from config import normalize_params
from config.runtime import internal_param_value, param_value
from fisher import (
    build_brownian_process_covariance,
    compute_localization_crlb,
    compute_localization_crlb_from_lateral_rerenders,
    summarize_fisher_sequence,
)
from imaging_models import get_imaging_model
from json_utils import json_safe
from modality_profiles import profile_card_for_model
from modality_registry import canonical_modality_name
from particle_specs import get_particle_specs
from postprocessing import (
    apply_background_subtraction,
    compute_single_frame_contrast,
    normalize_raw_camera_frames,
    save_video,
)
from rendering import generate_video_and_masks
from trajectory import (
    resolve_translational_diameters_nm,
    stokes_einstein_diffusion_coefficient,
)

logger = logging.getLogger(__name__)

from .latent_scene import _build_particle_instances_for_scene, _simulate_latent_scene
from .output import (
    _RUNTIME_PARAM_KEYS,
    _ensure_run_scope_layout_extent,
    _ensure_run_scope_layout_token,
    _jsonable_crlb_summary,
    _packet_sample_environment_metadata,
    _raw_signal_video_filename,
    _resolve_public_num_frames,
    _setup_output_dirs,
    _simulation_result,
    _source_map_provenance,
)
from .units import (
    _canonical_contrast_frame_units,
    _canonical_measurement_domain_and_signal_units,
)

def _render_scene_with_params(
    params: dict,
    latent_scene: dict,
    *,
    save_video_output: bool,
    return_frames: bool,
) -> dict | None:
    _setup_output_dirs(params)

    particle_instances = _build_particle_instances_for_scene(params, latent_scene)

    rendered = generate_video_and_masks(
        params,
        particle_instances,
    )
    render_metadata = dict(getattr(rendered, "render_metadata", {}) or {})
    raw_signal_frames = rendered.signal_frames
    raw_reference_frames = rendered.reference_frames
    ideal_signal_frames = rendered.ideal_signal_frames
    ideal_reference_frames = rendered.ideal_reference_frames

    final_frames = apply_background_subtraction(
        raw_signal_frames,
        raw_reference_frames,
        params,
    )

    if not final_frames:
        logger.info("Video generation failed or produced no frames.")
        if return_frames:
            return _simulation_result([], ["default"], {
                "raw_signal_frames": list(raw_signal_frames),
                "raw_reference_frames": list(raw_reference_frames),
                "ideal_signal_frames": list(ideal_signal_frames),
                "ideal_reference_frames": list(ideal_reference_frames),
                "background_subtracted_frames": [],
                "mask_arrays": list(getattr(rendered, "mask_arrays", [])),
                "supervision_records": list(getattr(rendered, "supervision_records", [])),
                "supervision_audit_summary": getattr(rendered, "supervision_audit_summary", None),
                "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
                "source_map_provenance": _source_map_provenance(params, render_metadata),
                "render_metadata": render_metadata,
            })
        return None

    analysis_video_path = None
    raw_signal_video_path = None
    if save_video_output:
        img_size = (params["image_size_pixels"], params["image_size_pixels"])
        analysis_video_path = str(params["output_filename"])
        save_video(analysis_video_path, final_frames, params["fps"], img_size)
        if bool(param_value(params, "save_raw_camera_video")):
            raw_signal_video_path = _raw_signal_video_filename(params)
            raw_camera_frames = normalize_raw_camera_frames(raw_signal_frames, params)
            save_video(raw_signal_video_path, raw_camera_frames, params["fps"], img_size)

    if return_frames:
        return _simulation_result(final_frames, ["default"], {
            "analysis_video_path": analysis_video_path,
            "raw_signal_video_path": raw_signal_video_path,
            "analysis_video_semantics": "background_subtracted_contrast_normalized_uint8",
            "raw_signal_video_semantics": "windowed_raw_detector_count_preview_uint8",
            "raw_signal_frames": list(raw_signal_frames),
            "raw_reference_frames": list(raw_reference_frames),
            "ideal_signal_frames": list(ideal_signal_frames),
            "ideal_reference_frames": list(ideal_reference_frames),
            "background_subtracted_frames": list(final_frames),
            "mask_arrays": list(getattr(rendered, "mask_arrays", [])),
            "supervision_records": list(getattr(rendered, "supervision_records", [])),
            "supervision_audit_summary": getattr(rendered, "supervision_audit_summary", None),
            "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
            "source_map_provenance": _source_map_provenance(params, render_metadata),
            "render_metadata": render_metadata,
        })

    return None


def _latent_scene_with_lateral_perturbation(
    latent_scene: dict,
    *,
    particle_index: int,
    frame_index: int,
    axis: int,
    delta_nm: float,
) -> dict:
    perturbed = deepcopy(latent_scene)
    trajectories = np.asarray(latent_scene.get("trajectories_nm", []), dtype=float).copy()
    if trajectories.ndim != 3 or trajectories.shape[2] < 2:
        raise ValueError(
            "latent_scene['trajectories_nm'] must have shape (particles, frames, 3) "
            "for rerendered lateral Fisher derivatives."
        )
    if particle_index < 0 or particle_index >= trajectories.shape[0]:
        raise ValueError(
            f"fisher_particle_index={particle_index} is outside the latent scene "
            f"particle range 0..{trajectories.shape[0] - 1}."
        )
    if frame_index < 0 or frame_index >= trajectories.shape[1]:
        raise ValueError(
            f"frame_index={frame_index} is outside the latent scene frame range "
            f"0..{trajectories.shape[1] - 1}."
        )
    trajectories[int(particle_index), int(frame_index), int(axis)] += float(delta_nm)
    perturbed["trajectories_nm"] = trajectories
    return perturbed


def _analysis_contrast_for_latent_scene(
    params: dict,
    latent_scene: dict,
    *,
    frame_index: int,
) -> np.ndarray:
    render_params = deepcopy(params)
    render_params["mask_generation_enabled"] = False
    render_params["_return_mask_arrays"] = False
    render_params["_write_mask_files"] = False
    render_params["return_ideal_float_frames"] = True
    render_params["background_subtraction_method"] = "reference_frame"
    result = _render_scene_with_params(
        render_params,
        latent_scene,
        save_video_output=False,
        return_frames=True,
    ) or {}
    metadata = dict(result.get("metadata", {}) or {})
    signal_frames = metadata.get("ideal_signal_frames") or []
    reference_frames = metadata.get("ideal_reference_frames") or []
    if frame_index < 0 or frame_index >= len(signal_frames):
        raise ValueError(
            f"frame_index={frame_index} is outside rerendered signal frame range "
            f"0..{len(signal_frames) - 1}."
        )
    if frame_index >= len(reference_frames):
        raise ValueError(
            f"frame_index={frame_index} is outside rerendered reference frame range "
            f"0..{len(reference_frames) - 1}."
        )
    return compute_single_frame_contrast(
        np.asarray(signal_frames[frame_index], dtype=float),
        np.asarray(reference_frames[frame_index], dtype=float),
        render_params,
    )


def render_matched_modality_observations(
    params: dict,
    modalities,
    *,
    frame_index: int = 0,
) -> dict:
    """
    Render one latent scene through multiple imaging models for packet output.

    The returned packet payload contains analysis contrast images, supervision
    masks, lateral Fisher matrices, and CRLB summaries on a shared detector
    coordinate frame.
    """
    if isinstance(modalities, (str, bytes)) or not isinstance(modalities, (list, tuple)):
        raise ValueError("modalities must be a list/tuple of at least two imaging model names.")
    requested_modality_names = [str(modality).strip() for modality in modalities]
    if len(requested_modality_names) < 2 or any(not name for name in requested_modality_names):
        raise ValueError("modalities must contain at least two non-empty names.")
    modality_names = [canonical_modality_name(name) for name in requested_modality_names]
    if len(set(modality_names)) != len(modality_names):
        raise ValueError(
            "matched modality packets require unique modality names after canonicalization; "
            f"got {requested_modality_names!r} -> {modality_names!r}."
        )

    base_params = deepcopy(params)
    if param_value(base_params, "channels") is not None:
        raise ValueError("matched modality packets cannot be combined with PARAMS['channels'].")
    base_params["channels"] = None
    _resolve_public_num_frames(base_params)
    _ensure_run_scope_layout_token(base_params)
    base_params = normalize_params(base_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    max_layout_extent = None
    for modality in modality_names:
        extent_params = deepcopy(base_params)
        extent_params["imaging_model"] = modality
        _ensure_run_scope_layout_extent(extent_params)
        extent = internal_param_value(extent_params, "_substrate_pattern_layout_extent_nm")
        if extent is not None:
            max_layout_extent = (
                float(extent)
                if max_layout_extent is None
                else max(float(max_layout_extent), float(extent))
            )
    if max_layout_extent is not None:
        base_params["_substrate_pattern_layout_extent_nm"] = float(max_layout_extent)
        base_params = normalize_params(base_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    latent_scene = _simulate_latent_scene(base_params)

    images_by_modality: dict[str, np.ndarray] = {}
    rendered_signal_frame_by_modality: dict[str, np.ndarray] = {}
    reference_frame_by_modality: dict[str, np.ndarray] = {}
    noise_variance_by_modality: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    fisher_by_modality: dict[str, np.ndarray] = {}
    crlb_by_modality: dict[str, dict] = {}
    modality_metadata: dict[str, dict] = {}
    for modality in modality_names:
        modality_params = deepcopy(base_params)
        modality_params["imaging_model"] = modality
        modality_params["mask_generation_enabled"] = True
        modality_params["_return_mask_arrays"] = True
        modality_params["_write_mask_files"] = False
        modality_params["return_ideal_float_frames"] = True
        modality_params["background_subtraction_method"] = "reference_frame"
        _ensure_run_scope_layout_extent(modality_params)
        modality_params = normalize_params(modality_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
        result = _render_scene_with_params(
            modality_params,
            latent_scene,
            save_video_output=False,
            return_frames=True,
        ) or {}
        metadata = dict(result.get("metadata", {}) or {})
        signal_frames = metadata.get("ideal_signal_frames") or metadata.get("raw_signal_frames", [])
        reference_frames = metadata.get("ideal_reference_frames") or metadata.get("raw_reference_frames", [])
        if frame_index < 0 or frame_index >= len(signal_frames):
            raise ValueError(
                f"frame_index={frame_index} is outside rendered frame range 0..{len(signal_frames) - 1}."
            )
        if frame_index >= len(reference_frames):
            raise ValueError(
                f"frame_index={frame_index} is outside rendered reference-frame range "
                f"0..{len(reference_frames) - 1}."
            )
        signal_frame = np.asarray(signal_frames[frame_index], dtype=float)
        reference_frame = np.asarray(reference_frames[frame_index], dtype=float)
        contrast_frame = compute_single_frame_contrast(
            signal_frame,
            reference_frame,
            modality_params,
        )
        render_metadata = dict(metadata.get("render_metadata", {}) or {})
        noise_params = dict(modality_params)
        effective_exposure_time_s = render_metadata.get("effective_exposure_time_s")
        if effective_exposure_time_s is not None:
            noise_params["exposure_time_s"] = float(effective_exposure_time_s)
        noise_variance = analysis_contrast_noise_variance(
            signal_frame,
            reference_frame,
            noise_params,
        )
        model = get_imaging_model(modality_params)
        output_type = getattr(model, "output_type", "intensity")
        render_metadata = dict(metadata.get("render_metadata", {}) or {})
        response_function = dict(render_metadata.get("response_function", {}) or {})
        if not response_function:
            response_function = model.compute_response_function(signal_frame.shape, modality_params)
        measurement_domain, signal_units = _canonical_measurement_domain_and_signal_units(
            modality_params,
            model,
            modality,
            response_function=response_function,
        )
        lateral_derivative_mode = str(modality_params["fisher_lateral_derivative_mode"]).strip().lower()
        structured_environment_active = bool(
            modality_params["sample_environment_enabled"]
            and (
                modality_params["sample_environment_pattern_enabled"]
                or str(modality_params["sample_environment_pattern"]).strip().lower() != "none"
            )
        )
        if lateral_derivative_mode == "stationary_shift" and structured_environment_active:
            raise ValueError(
                "Matched-modality Fisher diagnostics with structured sample "
                "environments require PARAMS['fisher_lateral_derivative_mode']="
                "'rerendered_xy'."
            )
        detected_target = str(modality_params["detected_quanta_derivative_target"]).strip().lower()
        if detected_target != "signed_contrast_scaled":
            raise ValueError(
                "render_matched_modality_observations currently computes Fisher on signed "
                "analysis contrast frames. Set PARAMS['detected_quanta_derivative_target']="
                "'signed_contrast_scaled' for matched packets, or route count-mean derivative "
                "comparisons through the detected-quanta comparator."
            )
        lateral_step_nm = float(modality_params["fisher_lateral_step_nm"])
        fisher_particle_index = int(modality_params["fisher_particle_index"])

        def _frame_crlb(local_frame_index: int, local_contrast, local_noise_variance) -> dict:
            if lateral_derivative_mode == "rerendered_xy":
                x_minus = _analysis_contrast_for_latent_scene(
                    modality_params,
                    _latent_scene_with_lateral_perturbation(
                        latent_scene,
                        particle_index=fisher_particle_index,
                        frame_index=local_frame_index,
                        axis=0,
                        delta_nm=-lateral_step_nm,
                    ),
                    frame_index=local_frame_index,
                )
                x_plus = _analysis_contrast_for_latent_scene(
                    modality_params,
                    _latent_scene_with_lateral_perturbation(
                        latent_scene,
                        particle_index=fisher_particle_index,
                        frame_index=local_frame_index,
                        axis=0,
                        delta_nm=lateral_step_nm,
                    ),
                    frame_index=local_frame_index,
                )
                y_minus = _analysis_contrast_for_latent_scene(
                    modality_params,
                    _latent_scene_with_lateral_perturbation(
                        latent_scene,
                        particle_index=fisher_particle_index,
                        frame_index=local_frame_index,
                        axis=1,
                        delta_nm=-lateral_step_nm,
                    ),
                    frame_index=local_frame_index,
                )
                y_plus = _analysis_contrast_for_latent_scene(
                    modality_params,
                    _latent_scene_with_lateral_perturbation(
                        latent_scene,
                        particle_index=fisher_particle_index,
                        frame_index=local_frame_index,
                        axis=1,
                        delta_nm=lateral_step_nm,
                    ),
                    frame_index=local_frame_index,
                )
                return compute_localization_crlb_from_lateral_rerenders(
                    x_minus,
                    x_plus,
                    y_minus,
                    y_plus,
                    np.asarray(local_noise_variance, dtype=float),
                    pixel_size_nm=float(modality_params["pixel_size_nm"]),
                    lateral_step_nm=lateral_step_nm,
                    signal_units=signal_units,
                    measurement_domain=measurement_domain,
                )
            if lateral_derivative_mode == "stationary_shift":
                return compute_localization_crlb(
                    np.asarray(local_contrast, dtype=float),
                    np.asarray(local_noise_variance, dtype=float),
                    pixel_size_nm=float(modality_params["pixel_size_nm"]),
                    signal_units=signal_units,
                    measurement_domain=measurement_domain,
                )
            raise ValueError(
                "fisher_lateral_derivative_mode must be 'stationary_shift' or "
                f"'rerendered_xy'; got {lateral_derivative_mode!r}."
            )

        crlb = _frame_crlb(frame_index, contrast_frame, noise_variance)
        images_by_modality[modality] = np.asarray(contrast_frame, dtype=float)
        rendered_signal_frame_by_modality[modality] = np.asarray(signal_frame, dtype=float)
        reference_frame_by_modality[modality] = np.asarray(reference_frame, dtype=float)
        noise_variance_by_modality[modality] = np.asarray(noise_variance, dtype=float)
        fisher_by_modality[modality] = np.asarray(crlb["fisher_matrix"], dtype=float)
        crlb_summary = _jsonable_crlb_summary(crlb)
        crlb_summary.setdefault("convergence_status", "production_grid_only")
        crlb_summary.setdefault("validation_status", "diagnostic_only")
        crlb_summary.setdefault("production_grid_diagnostic", True)
        crlb_summary.setdefault("safe_for_ordering", False)
        crlb_summary.setdefault("safe_for_fusion", False)
        crlb_summary.setdefault("safe_for_time_allocation", False)
        crlb_summary.setdefault("safe_for_registration", False)
        crlb_summary.setdefault("safe_for_detected_quanta_ranking", False)
        crlb_summary.setdefault("derivative_step_policy", "single_step_no_convergence_sweep")
        crlb_summary.setdefault("derivative_target", "analysis_contrast_frame")
        crlb_by_modality[modality] = crlb_summary
        crlb_noise_variance_units = str(
            crlb_summary.get("noise_variance_units")
            or (crlb_summary.get("derivative_metadata") or {}).get("noise_variance_units")
            or f"{signal_units}^2"
        )
        sequence_requested = bool(
            modality_params["sequence_fisher_enabled"]
            or modality_params["dynamic_bayesian_enabled"]
        )
        sequence_fisher_summary: dict[str, Any] = {
            "sequence_enabled": False,
            "sequence_requested": bool(sequence_requested),
            "sequence_crlb_model": "per_frame",
            "frame_count": int(len(signal_frames)),
            "selected_frame_index": int(frame_index),
            "measurement_domain": measurement_domain,
            "signal_units": signal_units,
            "noise_variance_units": crlb_noise_variance_units,
        }
        if sequence_requested and len(signal_frames) > 1:
            per_frame_fishers = []
            for seq_frame_index in range(len(signal_frames)):
                seq_signal = np.asarray(signal_frames[seq_frame_index], dtype=float)
                seq_reference = np.asarray(reference_frames[seq_frame_index], dtype=float)
                seq_contrast = compute_single_frame_contrast(
                    seq_signal,
                    seq_reference,
                    modality_params,
                )
                seq_noise = analysis_contrast_noise_variance(
                    seq_signal,
                    seq_reference,
                    modality_params,
                )
                seq_crlb = _frame_crlb(seq_frame_index, seq_contrast, seq_noise)
                per_frame_fishers.append(np.asarray(seq_crlb["fisher_matrix"], dtype=float))

            dynamic_covariance = None
            initial_covariance = None
            if bool(modality_params["dynamic_bayesian_enabled"]):
                diameters = resolve_translational_diameters_nm(modality_params)
                if len(diameters) != 1:
                    raise ValueError(
                        "Matched-packet dynamic Bayesian CRLB currently requires "
                        "exactly one hydrodynamic diameter."
                    )
                diffusion = stokes_einstein_diffusion_coefficient(
                    float(diameters[0]),
                    float(modality_params["temperature_K"]),
                    float(modality_params["viscosity_Pa_s"]),
                )
                dynamic_covariance = build_brownian_process_covariance(
                    ("x", "y"),
                    fps=float(modality_params["fps"]),
                    translational_diffusion_coeff_m2_s=float(diffusion)
                    * float(modality_params["dynamic_process_noise_scale"]),
                )
                initial_covariance = np.eye(2, dtype=float) * float(
                    modality_params["dynamic_initial_variance_nm2"]
                )
            sequence_fisher_summary = summarize_fisher_sequence(
                per_frame_fishers,
                state_axes=("x", "y"),
                measurement_domain=measurement_domain,
                signal_units=signal_units,
                noise_variance_units=crlb_noise_variance_units,
                state_axis_units={"x": "nm", "y": "nm"},
                dynamic_process_noise_covariance=dynamic_covariance,
                dynamic_bayesian_enabled=bool(modality_params["dynamic_bayesian_enabled"]),
                fps=float(modality_params["fps"]),
                initial_covariance=initial_covariance,
                include_smoothing=bool(modality_params["dynamic_include_smoothing"]),
            )
            sequence_fisher_summary["sequence_requested"] = True
        for mask_entry in metadata.get("mask_arrays", []) or []:
            if int(mask_entry.get("frame_index", -1)) != int(frame_index):
                continue
            particle_number = int(mask_entry.get("particle_index", 0)) + 1
            for mask_name, mask_arr in dict(mask_entry.get("masks", {}) or {}).items():
                masks[f"{modality}__{mask_name}__particle_{particle_number}"] = np.asarray(mask_arr)
        contrast_units = _canonical_contrast_frame_units(
            modality_params,
            model,
            modality,
            response_function=response_function,
        )
        profile_card = dict(render_metadata.get("modality_profile_card", {}) or {})
        required_profile_fields = {
            "safe_for_linear_fisher_variance",
            "detector_likelihood_status",
            "detector_noise_input_domain",
            "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active",
            "fisher_variance_model_scope",
        }
        if not profile_card or not required_profile_fields.issubset(profile_card):
            profile_card = profile_card_for_model(
                modality_params,
                model,
                modality_name=modality,
                response_function=response_function,
                model_canvas_shape=signal_frame.shape,
            )
        detector_safe_for_linear_fisher = bool(profile_card["safe_for_linear_fisher_variance"])
        detector_likelihood_status = str(profile_card["detector_likelihood_status"])
        crlb_by_modality[modality]["detector_noise_input_domain"] = profile_card[
            "detector_noise_input_domain"
        ]
        crlb_by_modality[modality]["nonlinear_detector_effects_active"] = bool(
            profile_card["nonlinear_detector_effects_active"]
        )
        crlb_by_modality[modality]["deterministic_detector_transfer_active"] = bool(
            profile_card["deterministic_detector_transfer_active"]
        )
        crlb_by_modality[modality]["safe_for_linear_fisher_variance"] = (
            detector_safe_for_linear_fisher
        )
        crlb_by_modality[modality]["fisher_variance_model_scope"] = profile_card[
            "fisher_variance_model_scope"
        ]
        crlb_by_modality[modality]["detector_likelihood_status"] = (
            detector_likelihood_status
        )
        if not detector_safe_for_linear_fisher:
            crlb_by_modality[modality]["validation_status"] = "diagnostic_only"
            crlb_by_modality[modality]["safe_for_ordering"] = False
            crlb_by_modality[modality]["safe_for_fusion"] = False
            crlb_by_modality[modality]["safe_for_detected_quanta_ranking"] = False
        modality_metadata[modality] = {
            "imaging_model": modality,
            "configured_wavelength_nm": float(modality_params["wavelength_nm"]),
            "probe_wavelength_nm": response_function.get(
                "probe_wavelength_nm",
                model.probe_wavelength_nm(modality_params),
            ),
            "output_type": output_type,
            "measurement_domain": measurement_domain,
            "signal_units": signal_units,
            "contrast_frame_units": contrast_units,
            "backend_fidelity_metadata": json_safe(
                profile_card.get("backend_fidelity_metadata")
                or response_function.get("backend_fidelity_metadata", {})
            ),
            "fisher_source": "analysis_contrast_frame",
            "detected_quanta_derivative_target": detected_target,
            "sequence_fisher_summary": json_safe(sequence_fisher_summary),
            "detector_noise_input_domain": profile_card["detector_noise_input_domain"],
            "nonlinear_detector_effects_active": bool(
                profile_card["nonlinear_detector_effects_active"]
            ),
            "deterministic_detector_transfer_active": bool(
                profile_card["deterministic_detector_transfer_active"]
            ),
            "safe_for_linear_fisher_variance": detector_safe_for_linear_fisher,
            "fisher_variance_model_scope": profile_card["fisher_variance_model_scope"],
            "detector_likelihood_status": detector_likelihood_status,
            "modality_profile_card": json_safe(profile_card),
            "response_function": json_safe(response_function),
            "render_metadata": json_safe(render_metadata),
        }

    trajectories = np.asarray(latent_scene.get("trajectories_nm", []), dtype=float)
    latent_state = {
        "frame_index": int(frame_index),
        "num_frames": int(latent_scene.get("num_frames", 0)),
            "random_seed": base_params["random_seed"],
        "trajectories_nm": trajectories.tolist(),
        "orientations": json_safe(latent_scene.get("orientations")),
        "particles": json_safe(get_particle_specs(base_params)),
        "sample_environment": _packet_sample_environment_metadata(base_params),
    }
    return {
        "latent_state": latent_state,
        "images_by_modality": images_by_modality,
        "rendered_signal_frame_by_modality": rendered_signal_frame_by_modality,
        "reference_frame_by_modality": reference_frame_by_modality,
        "noise_variance_by_modality": noise_variance_by_modality,
        "masks": masks,
        "fisher_by_modality": fisher_by_modality,
        "crlb_by_modality": crlb_by_modality,
        "metadata": {
            "schema_version": "syniscopy-matched-modality-packet-v1",
            "image_kind": "analysis_contrast_frame",
            "modalities": modality_names,
            "modality_metadata": modality_metadata,
            "shared_coordinate_frame": {
                "frame_index": int(frame_index),
                "pixel_size_nm": float(base_params["pixel_size_nm"]),
                "image_size_pixels": int(base_params["image_size_pixels"]),
                "world_origin": "upper_left_pixel_center_nm",
                "axes": ["x_nm", "y_nm"],
                "fisher_frame": "shared_xy_detector_frame",
            },
        },
    }


__all__ = [
    "_analysis_contrast_for_latent_scene",
    "_latent_scene_with_lateral_perturbation",
    "_render_scene_with_params",
    "render_matched_modality_observations",
]
