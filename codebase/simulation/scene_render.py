from __future__ import annotations
from configured_parameters import configured_assign

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from camera_noise import (
    DetectorNoiseRuntime,
    analysis_contrast_noise_model,
    analysis_noise_params_for_frame,
    detector_contrast_frames_for_analysis,
    deterministic_detector_transfer_counts,
)
from array_representation import (
    ArrayRepresentation,
    COORD_DETECTOR_XY,
    DOMAIN_CAMERA_COUNT,
    DOMAIN_ELECTRON_COUNT,
    DOMAIN_PHASE,
    DOMAIN_RELATIVE_REFERENCE,
    STAGE_ANALYSIS_CONTRAST,
    STAGE_RAW_CAMERA_NOISY,
    UNKNOWN_ARRAY_REPRESENTATION,
    VALUE_ABSOLUTE,
    VALUE_DELTA,
    VALUE_DISPLAY,
    VALUE_RELATIVE,
)
from config.runtime import (
    AcquisitionProfile,
    BackgroundSubtractionSettings,
    FisherAnalysisSettings,
    MicroscopeRuntimeSettings,
    ModalitySettings,
    MotionDynamicsSettings,
    OpticalInstrumentSettings,
    SamplingGeometry,
    SampleEnvironmentSettings,
    SimulationOutputSettings,
    SpectralIntegrationSettings,
)
from contrast_contracts import AnalysisContrastProduct
from fisher import (
    build_brownian_process_covariance,
    compute_localization_crlb,
    compute_off_axis_demodulated_localization_crlb,
    is_off_axis_holography_modality,
    lateral_derivative_plan_metadata,
    require_array_only_spectral_lateral_derivative_ready,
    summarize_fisher_sequence,
)
from imaging_models import get_imaging_model
from json_utils import json_safe
from modality_profiles import profile_card_for_model
from modality_registry import (
    canonical_modality_name,
    is_electron_modality,
    modality_uses_relative_reference_contrast,
)
from microscope_specs import (
    OVERLAY_SURFACE_COMMON_GRID_PACKET,
    normalize_microscope_specs,
)
from particle_specs import get_particle_specs
from noise_contracts import (
    resolve_fisher_likelihood_eligibility,
    summarize_analysis_noise_model,
)
from postprocessing import (
    compute_contrast_frames,
    compute_single_frame_contrast,
    normalize_contrast_frames,
    normalize_raw_camera_frames,
    save_video,
)
from rendering import generate_video_and_masks
from simulation_runtime_state import runtime_state
from trajectory import (
    resolve_translational_diameters_nm,
    stokes_einstein_diffusion_coefficient,
)
from shared_constants import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)

logger = logging.getLogger(__name__)

from .latent_scene import _build_particle_instances_for_scene, _simulate_latent_scene
from .output import (
    _ensure_run_scope_detector_static_seed,
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

def _analysis_contrast_metadata(params: dict, render_metadata: dict) -> dict[str, Any]:
    model = get_imaging_model(params)
    modality = ModalitySettings.from_params(params).modality
    response_function = dict(render_metadata.get("response_function", {}) or {})
    if not response_function:
        sampling_shape = render_metadata.get("render_geometry", {}).get("model_canvas_shape")
        if sampling_shape is not None:
            response_function = model.compute_response_function(tuple(sampling_shape), params)
    method = BackgroundSubtractionSettings.from_params(params).method
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        contrast_units = "electron_count" if is_electron_modality(modality) else "detector_count"
        contrast_semantics = "raw_signal_frame_before_background_subtraction"
    elif method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        contrast_units = (
            "electron_count_difference"
            if is_electron_modality(modality)
            else "detector_count_difference"
        )
        contrast_semantics = "video_median_subtracted_count_difference"
    elif method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        contrast_units = _canonical_contrast_frame_units(
            params,
            model,
            modality,
            response_function=response_function,
        )
        contrast_semantics = "reference_frame_contrast_before_display_windowing"
    else:
        contrast_units = _canonical_contrast_frame_units(
            params,
            model,
            modality,
            response_function=response_function,
        )
        contrast_semantics = "floating_point_contrast_before_display_windowing"
    return {
        "analysis_contrast_frame_semantics": contrast_semantics,
        "analysis_contrast_frame_units": contrast_units,
    }


def _run_scope_layout_metadata(params: dict) -> dict[str, Any]:
    state = runtime_state(params)
    return {
        "substrate_pattern_layout_cache_token": state.substrate_pattern_layout_cache_token,
        "substrate_pattern_layout_extent_nm": state.substrate_pattern_layout_extent_nm,
    }


def _noise_params_for_render_metadata(
    params: dict,
    render_metadata: dict,
    *,
    frame_index: int = 0,
) -> dict:
    # Centralize render-metadata-to-likelihood reconstruction.  QPI phase
    # Fisher needs the renderer-owned per-frame detected-quanta sidecar; a
    # plain params+exposure merge silently drops that physical support.
    return analysis_noise_params_for_frame(
        params,
        render_metadata,
        frame_index=frame_index,
    )


def _analysis_contrast_input_frames(
    signal_frame,
    reference_frame,
    params: dict,
    *,
    noise_params: dict | None = None,
    output_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    signal_arr = np.asarray(signal_frame, dtype=float)
    reference_arr = None if reference_frame is None else np.asarray(reference_frame, dtype=float)
    if output_type is None:
        output_type = getattr(get_imaging_model(params), "output_type", "intensity")
    if output_type == "phase":
        return signal_arr, reference_arr
    return detector_contrast_frames_for_analysis(
        signal_arr,
        reference_arr,
        noise_params or params,
    )


def _quantitative_contrast_contract(
    params: dict,
    *,
    output_type: str,
    source: str,
) -> ArrayRepresentation:
    if not source:
        return UNKNOWN_ARRAY_REPRESENTATION
    modality = ModalitySettings.from_params(params).modality
    output_key = str(output_type or "").strip().lower()
    if output_key == "phase":
        return ArrayRepresentation(
            domain=DOMAIN_PHASE,
            value_form=VALUE_ABSOLUTE,
            units="radian",
            coordinate_frame=COORD_DETECTOR_XY,
            pipeline_stage=STAGE_ANALYSIS_CONTRAST,
            semantic_label="phase_radian",
        )
    if is_electron_modality(modality):
        return ArrayRepresentation(
            domain=DOMAIN_ELECTRON_COUNT,
            value_form=VALUE_DELTA,
            units="electron_count_difference",
            coordinate_frame=COORD_DETECTOR_XY,
            pipeline_stage=STAGE_ANALYSIS_CONTRAST,
            semantic_label="electron_count_difference",
        )
    if modality_uses_relative_reference_contrast(modality):
        return ArrayRepresentation(
            domain=DOMAIN_RELATIVE_REFERENCE,
            value_form=VALUE_RELATIVE,
            units="relative_reference",
            coordinate_frame=COORD_DETECTOR_XY,
            pipeline_stage=STAGE_ANALYSIS_CONTRAST,
            semantic_label="relative_photoresponse",
        )
    return ArrayRepresentation(
        domain=DOMAIN_CAMERA_COUNT,
        value_form=VALUE_DELTA,
        units="detector_count_difference",
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=STAGE_ANALYSIS_CONTRAST,
        semantic_label="additive_count_difference",
    )


def _metadata_frame_sequence(metadata: dict, *keys: str) -> tuple[list[Any], str]:
    for key in keys:
        frames = metadata.get(key)
        if frames is not None and len(frames) > 0:
            return list(frames), key
    return [], ""


def _frame_at_index(frames: list[Any], frame_index: int, *, label: str) -> np.ndarray:
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(
            f"frame_index={frame_index} is outside {label} frame range "
            f"0..{len(frames) - 1}."
        )
    return np.asarray(frames[frame_index], dtype=float)


def _matched_packet_analysis_and_sidecar_frames(
    metadata: dict,
    *,
    frame_index: int,
    noise_params: dict,
    output_type: str,
) -> dict[str, Any]:
    """
    Select separate frames for analysis and persisted packet sidecars.

    Analysis uses detector-input frames because that path owns deterministic
    detector transfer exactly once.  Count-domain packet ``signal__`` and
    ``reference__`` sidecars use detector means before stochastic noise because
    downstream count-budget comparators require real detector-count mean images.
    """
    signal_frames, signal_source = _metadata_frame_sequence(
        metadata,
        "detector_input_signal_frames",
        "ideal_signal_frames",
        "raw_signal_frames",
    )
    reference_frames, reference_source = _metadata_frame_sequence(
        metadata,
        "detector_input_reference_frames",
        "ideal_reference_frames",
        "raw_reference_frames",
    )
    signal_frame = _frame_at_index(signal_frames, frame_index, label="rendered signal")
    reference_frame = _frame_at_index(reference_frames, frame_index, label="rendered reference")

    if str(output_type).strip().lower() == "phase":
        return {
            "analysis_signal_frame": signal_frame,
            "analysis_reference_frame": reference_frame,
            "packet_signal_frame": signal_frame,
            "packet_reference_frame": reference_frame,
            "analysis_signal_frame_source": signal_source,
            "analysis_reference_frame_source": reference_source,
            "rendered_signal_frame_basis": "phase_or_display_signal_before_detector_count_mean_transfer",
            "reference_frame_basis": "phase_or_display_signal_before_detector_count_mean_transfer",
            "packet_sidecar_frame_source": signal_source,
        }

    mean_signal_frames, mean_signal_source = _metadata_frame_sequence(
        metadata,
        "detector_mean_signal_frames",
    )
    mean_reference_frames, mean_reference_source = _metadata_frame_sequence(
        metadata,
        "detector_mean_reference_frames",
    )
    if mean_signal_frames and mean_reference_frames:
        packet_signal_frame = _frame_at_index(
            mean_signal_frames,
            frame_index,
            label="detector-mean signal",
        )
        packet_reference_frame = _frame_at_index(
            mean_reference_frames,
            frame_index,
            label="detector-mean reference",
        )
        packet_source = (
            mean_signal_source
            if mean_signal_source == mean_reference_source
            else f"{mean_signal_source}+{mean_reference_source}"
        )
    else:
        fallback_runtime = DetectorNoiseRuntime()
        packet_signal_frame = deterministic_detector_transfer_counts(
            signal_frame,
            noise_params,
            runtime=fallback_runtime,
        )
        packet_reference_frame = deterministic_detector_transfer_counts(
            reference_frame,
            noise_params,
            runtime=fallback_runtime,
        )
        packet_source = "computed_from_detector_input_by_deterministic_detector_transfer_counts"

    return {
        "analysis_signal_frame": signal_frame,
        "analysis_reference_frame": reference_frame,
        "packet_signal_frame": np.asarray(packet_signal_frame, dtype=float),
        "packet_reference_frame": np.asarray(packet_reference_frame, dtype=float),
        "analysis_signal_frame_source": signal_source,
        "analysis_reference_frame_source": reference_source,
        "rendered_signal_frame_basis": "detector_mean_before_stochastic_noise",
        "reference_frame_basis": "detector_mean_before_stochastic_noise",
        "packet_sidecar_frame_source": packet_source,
    }


def _empty_analysis_contrast_product(reason: str) -> AnalysisContrastProduct:
    return AnalysisContrastProduct(
        frames=[],
        source="",
        representation=UNKNOWN_ARRAY_REPRESENTATION,
        semantics="unavailable_quantitative_analysis_contrast",
        quantitative=False,
        safe_for_fisher=False,
        background_subtraction_method=None,
        provenance_warning=str(reason),
    )


def _analysis_contrast_product_from_frames(
    frames: list[np.ndarray],
    *,
    params: dict,
    output_type: str,
    source: str,
) -> AnalysisContrastProduct:
    representation = _quantitative_contrast_contract(
        params,
        output_type=output_type,
        source=source,
    )
    has_frames = bool(frames)
    safe_for_fisher = bool(
        has_frames
        and representation is not UNKNOWN_ARRAY_REPRESENTATION
        and representation.value_form != VALUE_DISPLAY
        and representation.units not in {None, "display_only"}
    )
    warning = (
        ""
        if safe_for_fisher
        else "No deterministic quantitative analysis contrast frames were available."
    )
    return AnalysisContrastProduct(
        frames=tuple(np.asarray(frame, dtype=float) for frame in frames),
        source=source,
        representation=representation,
        semantics="reference_frame_quantitative_analysis_contrast_before_display_windowing",
        quantitative=bool(has_frames and representation.value_form != VALUE_DISPLAY),
        safe_for_fisher=safe_for_fisher,
        background_subtraction_method="reference_frame" if has_frames else None,
        display_background_subtraction_applied=False,
        provenance_warning=warning,
    )


def _quantitative_contrast_product_for_render(
    params: dict,
    render_metadata: dict,
    *,
    detector_mean_signal_frames,
    detector_mean_reference_frames,
    detector_input_signal_frames,
    detector_input_reference_frames,
    ideal_signal_frames,
    ideal_reference_frames,
) -> AnalysisContrastProduct:
    output_type = getattr(get_imaging_model(params), "output_type", "intensity")
    method = BackgroundSubtractionSettings.from_params(params).method
    quantitative_params = dict(params)
    # The public background_subtraction_method controls displayed/raw views.
    # Quantitative sidecars are a separate analysis contract: they must use the
    # reference-frame basis so raw detector counts, QPI display counts, or a
    # video-median display transform cannot be promoted to Fisher-safe contrast.
    configured_assign(quantitative_params, 'background_subtraction_method', "reference_frame")
    source_suffix = (
        "reference_frame_quantitative_policy"
        if method in RAW_BACKGROUND_SUBTRACTION_METHODS or method in VIDEO_BACKGROUND_SUBTRACTION_METHODS
        else "reference_frame_quantitative_policy"
    )

    if output_type != "phase":
        mean_signal_frames = list(detector_mean_signal_frames or [])
        mean_reference_frames = list(detector_mean_reference_frames or [])
        if mean_signal_frames and mean_reference_frames:
            from modality_registry import modality_uses_relative_reference_contrast

            if not modality_uses_relative_reference_contrast(ModalitySettings.from_params(params).modality):
                n_frames = min(len(mean_signal_frames), len(mean_reference_frames))
                frames = [
                    np.asarray(
                        compute_single_frame_contrast(
                            np.asarray(mean_signal_frames[idx], dtype=float),
                            np.asarray(mean_reference_frames[idx], dtype=float),
                            quantitative_params,
                        ),
                        dtype=float,
                    )
                    for idx in range(n_frames)
                ]
                return _analysis_contrast_product_from_frames(
                    frames,
                    params=params,
                    output_type=output_type,
                    source=(
                        "detector_mean_float_from_render_runtime_before_stochastic_noise_"
                        f"{source_suffix}"
                    ),
                )

    signal_frames = list(detector_input_signal_frames or [])
    reference_frames = list(detector_input_reference_frames or [])
    source = "detector_input_float_before_stochastic_noise"
    if not signal_frames or not reference_frames:
        signal_frames = list(ideal_signal_frames or [])
        reference_frames = list(ideal_reference_frames or [])
        source = "ideal_float_before_stochastic_noise"
    if not signal_frames:
        return _empty_analysis_contrast_product("Renderer returned no deterministic signal frames.")
    if not reference_frames:
        return _empty_analysis_contrast_product(
            "Reference-frame quantitative analysis contrast requires deterministic reference frames."
        )
    n_frames = min(len(signal_frames), len(reference_frames))
    quantitative_frames: list[np.ndarray] = []
    for idx in range(n_frames):
        noise_params = _noise_params_for_render_metadata(params, render_metadata, frame_index=idx)
        contrast_signal, contrast_reference = _analysis_contrast_input_frames(
            signal_frames[idx],
            reference_frames[idx],
            params,
            noise_params=noise_params,
            output_type=output_type,
        )
        contrast = compute_single_frame_contrast(
            contrast_signal,
            contrast_reference,
            quantitative_params,
        )
        quantitative_frames.append(np.asarray(contrast, dtype=float))
    return _analysis_contrast_product_from_frames(
        quantitative_frames,
        params=params,
        output_type=output_type,
        source=f"{source}_{source_suffix}",
    )


def _render_scene_with_params(
    params: dict,
    latent_scene: dict,
    *,
    save_video_output: bool,
    return_frames: bool,
) -> dict | None:
    _setup_output_dirs(params)
    _ensure_run_scope_detector_static_seed(params)

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
    detector_input_signal_frames = rendered.detector_input_signal_frames
    detector_input_reference_frames = rendered.detector_input_reference_frames
    detector_mean_signal_frames = rendered.detector_mean_signal_frames
    detector_mean_reference_frames = rendered.detector_mean_reference_frames
    detector_object_field_frames = rendered.detector_object_field_frames

    contrast_frames_float = compute_contrast_frames(
        raw_signal_frames,
        raw_reference_frames,
        params,
    )
    final_frames = normalize_contrast_frames(
        contrast_frames_float,
        raw_signal_frames[0].shape if raw_signal_frames else (0, 0),
    )
    contrast_metadata = _analysis_contrast_metadata(params, render_metadata)
    quantitative_contrast_product = _quantitative_contrast_product_for_render(
        params,
        render_metadata,
        detector_mean_signal_frames=detector_mean_signal_frames,
        detector_mean_reference_frames=detector_mean_reference_frames,
        detector_input_signal_frames=detector_input_signal_frames,
        detector_input_reference_frames=detector_input_reference_frames,
        ideal_signal_frames=ideal_signal_frames,
        ideal_reference_frames=ideal_reference_frames,
    )
    quantitative_contrast_frames = list(quantitative_contrast_product.frames)
    analysis_contrast_frames = list(quantitative_contrast_product.frames)
    analysis_metadata = quantitative_contrast_product.metadata(prefix="analysis_contrast")
    quantitative_metadata = quantitative_contrast_product.metadata(prefix="quantitative_contrast")
    contrast_metadata.update(
        {
            "analysis_contrast_frame_source": analysis_metadata["analysis_contrast_source"],
            "analysis_contrast_frame_basis": analysis_metadata["analysis_contrast_frame_basis"],
            "analysis_contrast_frame_contrast_basis": analysis_metadata["analysis_contrast_contrast_basis"],
            "analysis_contrast_frame_units": analysis_metadata["analysis_contrast_units"],
            "analysis_contrast_frame_semantics": analysis_metadata["analysis_contrast_semantics"],
            "analysis_contrast_frame_quantitative": analysis_metadata["analysis_contrast_quantitative"],
            "analysis_contrast_frame_safe_for_fisher": analysis_metadata["analysis_contrast_safe_for_fisher"],
            "analysis_contrast_frame_provenance_warning": analysis_metadata["analysis_contrast_provenance_warning"],
            "analysis_contrast_frame_contract_id": analysis_metadata["analysis_contrast_contract_id"],
            "raw_observation_contrast_frame_source": "raw_uint16_noisy_clipped",
            "raw_observation_contrast_frame_basis": STAGE_RAW_CAMERA_NOISY,
            "raw_observation_contrast_frame_contrast_basis": "display_only",
            "raw_observation_contrast_frame_units": "display_only",
            "raw_observation_contrast_frame_quantitative": False,
            "quantitative_contrast_frame_key": (
                "quantitative_contrast_frames"
                if quantitative_contrast_frames
                else None
            ),
            "quantitative_contrast_frame_source": quantitative_metadata["quantitative_contrast_source"],
            "quantitative_contrast_frame_quantitative": quantitative_metadata["quantitative_contrast_quantitative"],
            "quantitative_contrast_frame_basis": quantitative_metadata["quantitative_contrast_frame_basis"],
            "quantitative_contrast_frame_contrast_basis": quantitative_metadata["quantitative_contrast_contrast_basis"],
            "quantitative_contrast_frame_units": quantitative_metadata["quantitative_contrast_units"],
            "quantitative_contrast_frame_semantics": quantitative_metadata["quantitative_contrast_semantics"],
            "quantitative_contrast_frame_safe_for_fisher": quantitative_metadata["quantitative_contrast_safe_for_fisher"],
            "quantitative_contrast_frame_provenance_warning": quantitative_metadata["quantitative_contrast_provenance_warning"],
            "quantitative_contrast_frame_contract_id": quantitative_metadata["quantitative_contrast_contract_id"],
            "quantitative_contrast_background_subtraction_method": quantitative_metadata["quantitative_contrast_background_subtraction_method"],
            "quantitative_contrast_display_background_subtraction_applied": quantitative_metadata["quantitative_contrast_display_background_subtraction_applied"],
        }
    )

    if not final_frames:
        logger.info("Video generation failed or produced no frames.")
        if return_frames:
            return _simulation_result([], ["default"], {
                "raw_signal_frames": list(raw_signal_frames),
                "raw_reference_frames": list(raw_reference_frames),
                "ideal_signal_frames": list(ideal_signal_frames),
                "ideal_reference_frames": list(ideal_reference_frames),
                "detector_input_signal_frames": list(detector_input_signal_frames),
                "detector_input_reference_frames": list(detector_input_reference_frames),
                "detector_mean_signal_frames": list(detector_mean_signal_frames),
                "detector_mean_reference_frames": list(detector_mean_reference_frames),
                "detector_object_field_frames": list(detector_object_field_frames),
                "analysis_contrast_frames": list(analysis_contrast_frames),
                "contrast_frames_float": list(contrast_frames_float),
                "raw_observation_contrast_frames": list(contrast_frames_float),
                "quantitative_contrast_frames": list(quantitative_contrast_frames),
                **contrast_metadata,
                "background_subtracted_frames": [],
                "mask_arrays": list(getattr(rendered, "mask_arrays", [])),
                "supervision_records": list(getattr(rendered, "supervision_records", [])),
                "supervision_audit_summary": getattr(rendered, "supervision_audit_summary", None),
                "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
                "rendered_trajectories_nm": np.asarray(
                    getattr(rendered, "rendered_trajectories_nm", [])
                ),
                "trajectory_semantics": {
                    "trajectories_nm": (
                        "latent Brownian particle centers before render-time "
                        "rigid drift/vibration"
                    ),
                    "rendered_trajectories_nm": (
                        "exposure-averaged particle centers actually rendered "
                        "into output frames"
                    ),
                },
                "latent_scene": deepcopy(latent_scene),
                "run_scope_layout": _run_scope_layout_metadata(params),
                "source_map_provenance": _source_map_provenance(params, render_metadata),
                "render_metadata": render_metadata,
            })
        return None

    analysis_video_path = None
    raw_signal_video_path = None
    if save_video_output:
        sampling = SamplingGeometry.from_params(params)
        acquisition = AcquisitionProfile.from_params(params)
        output_settings = SimulationOutputSettings.from_params(params)
        img_size = sampling.detector_shape
        analysis_video_path = output_settings.output_filename
        save_video(analysis_video_path, final_frames, acquisition.fps, img_size)
        if output_settings.save_raw_camera_video:
            raw_signal_video_path = _raw_signal_video_filename(params)
            raw_camera_frames = normalize_raw_camera_frames(raw_signal_frames, params)
            save_video(raw_signal_video_path, raw_camera_frames, acquisition.fps, img_size)

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
            "detector_input_signal_frames": list(detector_input_signal_frames),
            "detector_input_reference_frames": list(detector_input_reference_frames),
            "detector_mean_signal_frames": list(detector_mean_signal_frames),
            "detector_mean_reference_frames": list(detector_mean_reference_frames),
            "detector_object_field_frames": list(detector_object_field_frames),
            "analysis_contrast_frames": list(analysis_contrast_frames),
            "contrast_frames_float": list(contrast_frames_float),
            "raw_observation_contrast_frames": list(contrast_frames_float),
            "quantitative_contrast_frames": list(quantitative_contrast_frames),
            **contrast_metadata,
            "background_subtracted_frames": list(final_frames),
            "mask_arrays": list(getattr(rendered, "mask_arrays", [])),
            "supervision_records": list(getattr(rendered, "supervision_records", [])),
            "supervision_audit_summary": getattr(rendered, "supervision_audit_summary", None),
            "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
            "rendered_trajectories_nm": np.asarray(
                getattr(rendered, "rendered_trajectories_nm", [])
            ),
            "trajectory_semantics": {
                "trajectories_nm": (
                    "latent Brownian particle centers before render-time "
                    "rigid drift/vibration"
                ),
                "rendered_trajectories_nm": (
                    "exposure-averaged particle centers actually rendered into "
                    "output frames"
                ),
            },
            "latent_scene": deepcopy(latent_scene),
            "run_scope_layout": _run_scope_layout_metadata(params),
            "source_map_provenance": _source_map_provenance(params, render_metadata),
            "render_metadata": render_metadata,
        })

    return None


def render_matched_microscope_observations(
    params: dict,
    microscopes,
    *,
    frame_index: int = 0,
    latent_scene: dict | None = None,
) -> dict:
    """
    Render one latent scene through multiple configured microscopes for packet output.

    The returned packet payload contains analysis contrast images, supervision
    masks, lateral Fisher matrices, and CRLB summaries on a shared detector
    coordinate frame.
    """
    microscope_specs = normalize_microscope_specs(
        microscopes,
        context="matched_microscopes",
        allow_modality_strings=False,
        overlay_surface=OVERLAY_SURFACE_COMMON_GRID_PACKET,
    )
    microscope_names = [spec.name for spec in microscope_specs]
    modality_by_microscope = {
        spec.name: spec.modality
        for spec in microscope_specs
    }

    base_params = deepcopy(params)
    if SpectralIntegrationSettings.from_params(base_params).channels is not None:
        raise ValueError("matched microscope packets cannot be combined with parameters['channels'].")
    configured_assign(base_params, 'channels', None)
    _resolve_public_num_frames(base_params)
    _ensure_run_scope_layout_token(base_params)
    MotionDynamicsSettings.from_params(base_params)
    SampleEnvironmentSettings.from_params(base_params)
    BackgroundSubtractionSettings.from_params(base_params)
    _ensure_run_scope_detector_static_seed(base_params)
    base_state = runtime_state(base_params)
    if base_state.substrate_pattern_layout_extent_nm is None:
        max_layout_extent = None
        for spec in microscope_specs:
            extent_params = deepcopy(base_params)
            extent_params.update(deepcopy(spec.params_overlay))
            configured_assign(extent_params, 'imaging_model', spec.modality)
            _ensure_run_scope_layout_extent(extent_params)
            extent = runtime_state(extent_params).substrate_pattern_layout_extent_nm
            if extent is not None:
                max_layout_extent = (
                    float(extent)
                    if max_layout_extent is None
                    else max(float(max_layout_extent), float(extent))
                )
        if max_layout_extent is not None:
            base_state.substrate_pattern_layout_extent_nm = float(max_layout_extent)
    if latent_scene is None:
        latent_scene = _simulate_latent_scene(base_params)
    else:
        latent_scene = deepcopy(latent_scene)

    images_by_microscope: dict[str, np.ndarray] = {}
    rendered_signal_frame_by_microscope: dict[str, np.ndarray] = {}
    reference_frame_by_microscope: dict[str, np.ndarray] = {}
    noise_variance_by_microscope: dict[str, np.ndarray] = {}
    analysis_contrast_noise_model_by_microscope: dict[str, dict] = {}
    masks: dict[str, np.ndarray] = {}
    fisher_by_microscope: dict[str, np.ndarray] = {}
    crlb_by_microscope: dict[str, dict] = {}
    analysis_contrast_noise_model_sequence_by_microscope: dict[str, list[dict]] = {}
    microscope_metadata: dict[str, dict] = {}
    for spec in microscope_specs:
        microscope_name = spec.name
        modality = spec.modality
        modality_params = deepcopy(base_params)
        modality_params.update(deepcopy(spec.params_overlay))
        configured_assign(modality_params, 'imaging_model', modality)
        configured_assign(modality_params, 'mask_generation_enabled', True)
        modality_state = runtime_state(modality_params)
        modality_state.return_mask_arrays = True
        modality_state.write_mask_files = False
        configured_assign(modality_params, 'return_ideal_float_frames', True)
        configured_assign(modality_params, 'background_subtraction_method', "reference_frame")
        _ensure_run_scope_layout_extent(modality_params)
        MicroscopeRuntimeSettings.from_params(modality_params)
        result = _render_scene_with_params(
            modality_params,
            latent_scene,
            save_video_output=False,
            return_frames=True,
        ) or {}
        metadata = dict(result.get("metadata", {}) or {})
        render_metadata = dict(metadata.get("render_metadata", {}) or {})
        noise_params = _noise_params_for_render_metadata(modality_params, render_metadata, frame_index=frame_index)
        model = get_imaging_model(modality_params)
        output_type = getattr(model, "output_type", "intensity")
        selected_frames = _matched_packet_analysis_and_sidecar_frames(
            metadata,
            frame_index=frame_index,
            noise_params=noise_params,
            output_type=output_type,
        )
        signal_frames, _signal_source = _metadata_frame_sequence(
            metadata,
            "detector_input_signal_frames",
            "ideal_signal_frames",
            "raw_signal_frames",
        )
        reference_frames, _reference_source = _metadata_frame_sequence(
            metadata,
            "detector_input_reference_frames",
            "ideal_reference_frames",
            "raw_reference_frames",
        )
        object_field_frames = list(metadata.get("detector_object_field_frames", []) or [])
        if is_off_axis_holography_modality(modality) and len(object_field_frames) != len(signal_frames):
            raise RuntimeError(
                "off-axis DHM matched-packet Fisher requires "
                "detector_object_field_frames aligned with detector-input "
                f"signal frames; got {len(object_field_frames)} and {len(signal_frames)}."
            )
        signal_frame = selected_frames["analysis_signal_frame"]
        reference_frame = selected_frames["analysis_reference_frame"]
        packet_signal_frame = selected_frames["packet_signal_frame"]
        packet_reference_frame = selected_frames["packet_reference_frame"]
        contrast_signal_frame, contrast_reference_frame = _analysis_contrast_input_frames(
            signal_frame,
            reference_frame,
            modality_params,
            noise_params=noise_params,
            output_type=output_type,
        )
        contrast_frame = compute_single_frame_contrast(
            contrast_signal_frame,
            contrast_reference_frame,
            modality_params,
        )
        noise_model = analysis_contrast_noise_model(
            signal_frame,
            reference_frame,
            noise_params,
        )
        analysis_contrast_noise_model_by_microscope[microscope_name] = json_safe(noise_model)
        noise_summary = summarize_analysis_noise_model(
            noise_model,
            expected_shape=np.asarray(contrast_frame).shape,
            context=f"matched-packet contrast frame shape for modality {modality!r}",
        )
        # ``noise_variance_by_microscope`` is a diagonal summary for report
        # consumers. The durable Fisher likelihood is the full
        # AnalysisNoiseModel stored in analysis_contrast_noise_model_by_microscope;
        # otherwise row-correlated scan-line covariance would be diagonalized.
        noise_variance = noise_summary.diagonal_variance
        response_function = dict(render_metadata.get("response_function", {}) or {})
        if not response_function:
            response_function = model.compute_response_function(signal_frame.shape, modality_params)
        measurement_domain, signal_units = _canonical_measurement_domain_and_signal_units(
            modality_params,
            model,
            modality,
            response_function=response_function,
        )
        fisher_settings = FisherAnalysisSettings.from_params(modality_params)
        fisher_settings.require_signed_contrast_detected_quanta_target(
            context="render_matched_microscope_observations"
        )
        detected_target = fisher_settings.detected_quanta_derivative_target
        trajectories = np.asarray(latent_scene.get("trajectories_nm", []), dtype=float)
        if trajectories.ndim != 3 or trajectories.shape[2] < 2:
            raise ValueError(
                "latent_scene['trajectories_nm'] must have shape (particles, frames, 3) "
                "for matched-microscope Fisher diagnostics."
            )

        def _frame_crlb(
            local_frame_index: int,
            local_contrast,
            local_noise_variance,
            *,
            local_signal,
            local_reference,
            local_object_field,
        ) -> tuple[dict, Any]:
            sampling = SamplingGeometry.from_params(modality_params)
            if is_off_axis_holography_modality(modality):
                crlb_result, observation = compute_off_axis_demodulated_localization_crlb(
                    np.asarray(local_signal, dtype=float),
                    None if local_reference is None else np.asarray(local_reference, dtype=float),
                    modality_params,
                    sampling.detector_pixel_size_nm,
                    response_function=response_function,
                    object_field_detector=local_object_field,
                )
                return crlb_result, observation.noise_model
            require_array_only_spectral_lateral_derivative_ready(
                modality=modality,
                params=modality_params,
                model=model,
                response_function=response_function,
                num_particles=int(trajectories.shape[0]),
                structured_environment_active=bool(
                    SampleEnvironmentSettings.from_params(modality_params).enabled
                ),
                context=(
                    "render_matched_microscope_observations "
                    f"microscope {microscope_name!r} (modality {modality!r}), "
                    f"frame {local_frame_index}"
                ),
            )
            return compute_localization_crlb(
                np.asarray(local_contrast, dtype=float),
                local_noise_variance,
                pixel_size_nm=sampling.detector_pixel_size_nm,
                signal_units=signal_units,
                measurement_domain=measurement_domain,
            ), local_noise_variance

        crlb, fisher_noise_model = _frame_crlb(
            frame_index,
            contrast_frame,
            noise_model,
            local_signal=signal_frame,
            local_reference=reference_frame,
            local_object_field=(
                np.asarray(object_field_frames[frame_index], dtype=np.complex128)
                if is_off_axis_holography_modality(modality)
                else None
            ),
        )
        if fisher_noise_model is not noise_model:
            analysis_contrast_noise_model_by_microscope[microscope_name] = json_safe(fisher_noise_model)
            fisher_noise_summary = summarize_analysis_noise_model(
                fisher_noise_model,
                expected_shape=np.asarray(contrast_frame).shape,
                context=f"matched-packet Fisher likelihood shape for modality {modality!r}",
            )
            noise_variance = fisher_noise_summary.diagonal_variance
        images_by_microscope[microscope_name] = np.asarray(contrast_frame, dtype=float)
        rendered_signal_frame_by_microscope[microscope_name] = np.asarray(packet_signal_frame, dtype=float)
        reference_frame_by_microscope[microscope_name] = np.asarray(packet_reference_frame, dtype=float)
        noise_variance_by_microscope[microscope_name] = np.asarray(noise_variance, dtype=float)
        fisher_by_microscope[microscope_name] = np.asarray(crlb["fisher_matrix"], dtype=float)
        crlb_summary = _jsonable_crlb_summary(crlb)
        crlb_summary.setdefault("convergence_status", "production_grid_only")
        crlb_summary.setdefault("validation_status", "diagnostic_only")
        crlb_summary.setdefault("production_grid_diagnostic", True)
        crlb_summary.setdefault("safe_for_ordering", False)
        crlb_summary.setdefault("safe_for_fusion", False)
        crlb_summary.setdefault("safe_for_time_allocation", False)
        crlb_summary.setdefault("safe_for_registration", False)
        crlb_summary.setdefault("safe_for_detected_quanta_ranking", False)
        crlb_summary.setdefault("derivative_step_policy", "spectral_no_derivative_step")
        crlb_summary.setdefault("derivative_target", "analysis_contrast_frame")
        crlb_summary.update(lateral_derivative_plan_metadata())
        crlb_by_microscope[microscope_name] = crlb_summary
        crlb_noise_variance_units = str(
            crlb_summary.get("noise_variance_units")
            or (crlb_summary.get("derivative_metadata") or {}).get("noise_variance_units")
            or f"{signal_units}^2"
        )
        crlb_derivative_metadata = dict(crlb_summary.get("derivative_metadata", {}) or {})
        fisher_measurement_domain = str(
            crlb_derivative_metadata.get("measurement_domain") or measurement_domain
        )
        fisher_signal_units = str(
            crlb_derivative_metadata.get("signal_units") or signal_units
        )
        dynamics = MotionDynamicsSettings.from_params(modality_params)
        sequence_requested = bool(
            dynamics.sequence_fisher_enabled
            or dynamics.dynamic_bayesian_enabled
        )
        sequence_fisher_summary: dict[str, Any] = {
            "sequence_enabled": False,
            "sequence_requested": bool(sequence_requested),
            "sequence_crlb_model": "per_frame",
            "frame_count": int(len(signal_frames)),
            "selected_frame_index": int(frame_index),
            "measurement_domain": fisher_measurement_domain,
            "signal_units": fisher_signal_units,
            "noise_variance_units": crlb_noise_variance_units,
        }
        sequence_noise_models: list[dict] = []
        if sequence_requested and len(signal_frames) > 1:
            per_frame_fishers = []
            for seq_frame_index in range(len(signal_frames)):
                seq_signal = np.asarray(signal_frames[seq_frame_index], dtype=float)
                seq_reference = np.asarray(reference_frames[seq_frame_index], dtype=float)
                seq_noise_params = _noise_params_for_render_metadata(
                    modality_params,
                    render_metadata,
                    frame_index=seq_frame_index,
                )
                seq_contrast_signal, seq_contrast_reference = _analysis_contrast_input_frames(
                    seq_signal,
                    seq_reference,
                    modality_params,
                    noise_params=seq_noise_params,
                    output_type=output_type,
                )
                seq_contrast = compute_single_frame_contrast(
                    seq_contrast_signal,
                    seq_contrast_reference,
                    modality_params,
                )
                seq_noise_model = analysis_contrast_noise_model(
                    seq_signal,
                    seq_reference,
                    seq_noise_params,
                )
                seq_noise_summary = summarize_analysis_noise_model(
                    seq_noise_model,
                    expected_shape=np.asarray(seq_contrast).shape,
                    context=(
                        "matched-packet sequence contrast frame shape for "
                        f"modality {modality!r}, frame {seq_frame_index}"
                    ),
                )
                seq_noise = seq_noise_summary.diagonal_variance
                seq_crlb, seq_fisher_noise_model = _frame_crlb(
                    seq_frame_index,
                    seq_contrast,
                    seq_noise_model,
                    local_signal=seq_signal,
                    local_reference=seq_reference,
                    local_object_field=(
                        np.asarray(object_field_frames[seq_frame_index], dtype=np.complex128)
                        if is_off_axis_holography_modality(modality)
                        else None
                    ),
                )
                per_frame_fishers.append(np.asarray(seq_crlb["fisher_matrix"], dtype=float))
                sequence_noise_models.append(json_safe(seq_fisher_noise_model))

            dynamic_covariance = None
            initial_covariance = None
            acquisition = AcquisitionProfile.from_params(modality_params)
            if dynamics.dynamic_bayesian_enabled:
                diameters = resolve_translational_diameters_nm(modality_params)
                if len(diameters) != 1:
                    raise ValueError(
                        "Matched-packet dynamic Bayesian CRLB currently requires "
                        "exactly one hydrodynamic diameter."
                    )
                diffusion = stokes_einstein_diffusion_coefficient(
                    float(diameters[0]),
                    dynamics.temperature_K,
                    dynamics.viscosity_Pa_s,
                )
                dynamic_covariance = build_brownian_process_covariance(
                    ("x", "y"),
                    fps=acquisition.fps,
                    translational_diffusion_coeff_m2_s=float(diffusion)
                    * dynamics.dynamic_process_noise_scale,
                )
                initial_covariance = (
                    np.eye(2, dtype=float) * dynamics.dynamic_initial_variance_nm2
                )
            sequence_fisher_summary = summarize_fisher_sequence(
                per_frame_fishers,
                state_axes=("x", "y"),
                measurement_domain=measurement_domain,
                signal_units=signal_units,
                noise_variance_units=crlb_noise_variance_units,
                state_axis_units={"x": "nm", "y": "nm"},
                dynamic_process_noise_covariance=dynamic_covariance,
                dynamic_bayesian_enabled=dynamics.dynamic_bayesian_enabled,
                fps=acquisition.fps,
                initial_covariance=initial_covariance,
                include_smoothing=dynamics.dynamic_include_smoothing,
            )
            sequence_fisher_summary["sequence_requested"] = True
            sequence_fisher_summary["analysis_contrast_noise_model_sequence"] = (
                [json_safe(item) for item in sequence_noise_models]
            )
        analysis_contrast_noise_model_sequence_by_microscope[microscope_name] = [
            dict(item) for item in sequence_noise_models
        ]
        for mask_entry in metadata.get("mask_arrays", []) or []:
            if int(mask_entry.get("frame_index", -1)) != int(frame_index):
                continue
            particle_number = int(mask_entry.get("particle_index", 0)) + 1
            for mask_name, mask_arr in dict(mask_entry.get("masks", {}) or {}).items():
                masks[f"{microscope_name}__{mask_name}__particle_{particle_number}"] = np.asarray(mask_arr)
        contrast_units = _canonical_contrast_frame_units(
            modality_params,
            model,
            modality,
            response_function=response_function,
        )
        profile_card = dict(render_metadata.get("modality_profile_card", {}) or {})
        required_profile_fields = {
            "safe_for_linear_fisher_variance",
            "safe_for_covariance_fisher_variance",
            "detector_likelihood_status",
            "detector_noise_input_domain",
            "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active",
            "fisher_variance_model_scope",
            "covariance_fisher_variance_model_scope",
        }
        if not profile_card or not required_profile_fields.issubset(profile_card):
            profile_card = profile_card_for_model(
                modality_params,
                model,
                modality_name=modality,
                response_function=response_function,
                model_canvas_shape=signal_frame.shape,
            )
        # Matched-packet metadata uses the same eligibility resolver as lab
        # reports, so covariance-aware row-noise handling and diagnostic-only
        # detector transfer cannot diverge across public output paths.
        eligibility = resolve_fisher_likelihood_eligibility(
            fisher_noise_model,
            profile_card,
            crlb_by_microscope[microscope_name],
            context=f"matched-packet microscope {microscope_name!r} ({modality!r})",
        )
        detector_safe_for_linear_fisher = bool(eligibility.safe_for_linear_fisher_variance)
        detector_safe_for_covariance_fisher = bool(
            eligibility.safe_for_covariance_fisher_variance
        )
        detector_safe_for_report_fisher = bool(eligibility.detector_safe_for_report_fisher)
        crlb_used_covariance_fisher = bool(eligibility.used_covariance_fisher)
        detector_likelihood_status = str(eligibility.detector_likelihood_status)
        crlb_by_microscope[microscope_name]["detector_noise_input_domain"] = profile_card[
            "detector_noise_input_domain"
        ]
        crlb_by_microscope[microscope_name]["nonlinear_detector_effects_active"] = bool(
            profile_card["nonlinear_detector_effects_active"]
        )
        crlb_by_microscope[microscope_name]["deterministic_detector_transfer_active"] = bool(
            profile_card["deterministic_detector_transfer_active"]
        )
        crlb_by_microscope[microscope_name]["safe_for_linear_fisher_variance"] = (
            detector_safe_for_linear_fisher
        )
        crlb_by_microscope[microscope_name]["safe_for_covariance_fisher_variance"] = (
            detector_safe_for_covariance_fisher
        )
        crlb_by_microscope[microscope_name]["covariance_fisher_variance_model_scope"] = (
            profile_card.get("covariance_fisher_variance_model_scope", "")
        )
        crlb_by_microscope[microscope_name]["fisher_variance_model_scope"] = profile_card[
            "fisher_variance_model_scope"
        ]
        crlb_by_microscope[microscope_name]["detector_likelihood_status"] = (
            detector_likelihood_status
        )
        crlb_by_microscope[microscope_name]["detector_safe_for_report_fisher"] = (
            detector_safe_for_report_fisher
        )
        crlb_by_microscope[microscope_name]["fisher_likelihood_uses_covariance"] = (
            crlb_used_covariance_fisher
        )
        crlb_by_microscope[microscope_name]["fisher_likelihood_eligibility_contract_id"] = (
            eligibility.contract_id
        )
        if not eligibility.safe_for_ordering:
            crlb_by_microscope[microscope_name]["validation_status"] = "diagnostic_only"
            crlb_by_microscope[microscope_name]["safe_for_ordering"] = False
            crlb_by_microscope[microscope_name]["safe_for_fusion"] = False
            crlb_by_microscope[microscope_name]["safe_for_detected_quanta_ranking"] = False
            crlb_by_microscope[microscope_name]["status_reason"] = eligibility.status_reason
        instrument = OpticalInstrumentSettings.from_params(modality_params)
        microscope_metadata[microscope_name] = {
            "microscope": microscope_name,
            "imaging_model": modality,
            "configured_wavelength_nm": float(instrument.wavelength_nm),
            "probe_wavelength_nm": response_function.get(
                "probe_wavelength_nm",
                instrument.probe_wavelength_nm,
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
            "analysis_signal_frame_source": selected_frames["analysis_signal_frame_source"],
            "analysis_reference_frame_source": selected_frames["analysis_reference_frame_source"],
            "detector_input_frame_basis": "incident_or_model_input_before_detector_transfer",
            "packet_sidecar_frame_source": selected_frames["packet_sidecar_frame_source"],
            "rendered_signal_frame_basis": selected_frames["rendered_signal_frame_basis"],
            "reference_frame_basis": selected_frames["reference_frame_basis"],
            "detected_quanta_derivative_target": detected_target,
            "fisher_lateral_derivative_basis": "spectral_band_limited",
            "fisher_lateral_derivative_basis_resolution": "single_center_render_fft_spectral_gradient",
            "analysis_contrast_noise_model": json_safe(noise_model),
            "analysis_contrast_noise_model_sequence": json_safe(
                analysis_contrast_noise_model_sequence_by_microscope.get(microscope_name, [])
            ),
            "sequence_fisher_summary": json_safe(sequence_fisher_summary),
            "detector_noise_input_domain": profile_card["detector_noise_input_domain"],
            "nonlinear_detector_effects_active": bool(
                profile_card["nonlinear_detector_effects_active"]
            ),
            "deterministic_detector_transfer_active": bool(
                profile_card["deterministic_detector_transfer_active"]
            ),
            "safe_for_linear_fisher_variance": detector_safe_for_linear_fisher,
            "safe_for_covariance_fisher_variance": detector_safe_for_covariance_fisher,
            "detector_safe_for_report_fisher": detector_safe_for_report_fisher,
            "fisher_likelihood_uses_covariance": crlb_used_covariance_fisher,
            "fisher_likelihood_eligibility_contract_id": eligibility.contract_id,
            "fisher_variance_model_scope": profile_card["fisher_variance_model_scope"],
            "covariance_fisher_variance_model_scope": profile_card.get(
                "covariance_fisher_variance_model_scope",
                "",
            ),
            "detector_likelihood_status": detector_likelihood_status,
            "modality_profile_card": json_safe(profile_card),
            "response_function": json_safe(response_function),
            "render_metadata": json_safe(render_metadata),
        }

    trajectories = np.asarray(latent_scene.get("trajectories_nm", []), dtype=float)
    latent_state = {
        "frame_index": int(frame_index),
        "num_frames": int(latent_scene.get("num_frames", 0)),
        "random_seed": AcquisitionProfile.from_params(base_params).random_seed,
        "trajectories_nm": trajectories.tolist(),
        "orientations": json_safe(latent_scene.get("orientations")),
        "particles": json_safe(get_particle_specs(base_params)),
        "sample_environment": _packet_sample_environment_metadata(base_params),
    }
    return {
        "latent_state": latent_state,
        "images_by_microscope": images_by_microscope,
        "rendered_signal_frame_by_microscope": rendered_signal_frame_by_microscope,
        "reference_frame_by_microscope": reference_frame_by_microscope,
        "noise_variance_by_microscope": noise_variance_by_microscope,
        "analysis_contrast_noise_model_by_microscope": json_safe(
            analysis_contrast_noise_model_by_microscope
        ),
        "analysis_noise_model_by_microscope": json_safe(
            analysis_contrast_noise_model_by_microscope
        ),
        "analysis_contrast_noise_model_sequence_by_microscope": json_safe(
            analysis_contrast_noise_model_sequence_by_microscope
        ),
        "masks": masks,
        "fisher_by_microscope": fisher_by_microscope,
        "crlb_by_microscope": crlb_by_microscope,
        "metadata": {
            "schema_version": "syniscopy-matched-microscope-payload-v1",
            "image_kind": "analysis_contrast_frame",
            "rendered_signal_frame_basis": "per_microscope_declared_in_rendered_signal_frame_basis_by_microscope",
            "reference_frame_basis": "per_microscope_declared_in_reference_frame_basis_by_microscope",
            "detector_input_frame_basis": "incident_or_model_input_before_detector_transfer",
            "rendered_signal_frame_basis_by_microscope": {
                microscope: microscope_metadata[microscope]["rendered_signal_frame_basis"]
                for microscope in microscope_names
            },
            "reference_frame_basis_by_microscope": {
                microscope: microscope_metadata[microscope]["reference_frame_basis"]
                for microscope in microscope_names
            },
            "fisher_lateral_derivative_basis_by_microscope": {
                microscope: microscope_metadata[microscope]["fisher_lateral_derivative_basis"]
                for microscope in microscope_names
            },
            "analysis_contrast_noise_model_by_microscope": {
                microscope: microscope_metadata[microscope]["analysis_contrast_noise_model"]
                for microscope in microscope_names
            },
            "analysis_contrast_noise_model_sequence_by_microscope": {
                microscope: microscope_metadata[microscope]["analysis_contrast_noise_model_sequence"]
                for microscope in microscope_names
            },
            "microscopes": microscope_names,
            "modality_by_microscope": modality_by_microscope,
            "microscope_metadata": microscope_metadata,
            "shared_coordinate_frame": {
                "frame_index": int(frame_index),
                "pixel_size_nm": SamplingGeometry.from_params(base_params).detector_pixel_size_nm,
                "image_size_pixels": SamplingGeometry.from_params(base_params).image_size_pixels,
                "world_origin": "upper_left_pixel_center_nm",
                "axes": ["x_nm", "y_nm"],
                "fisher_frame": "shared_xy_detector_frame",
            },
        },
    }


__all__ = [
    "_render_scene_with_params",
    "render_matched_microscope_observations",
]
