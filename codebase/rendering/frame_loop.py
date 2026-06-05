"""Per-frame stamping, composition, noise, and supervision loop."""

from __future__ import annotations

import logging
import numbers
import os

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from backend_fidelity import extract_backend_fidelity_metadata
from camera_noise import (
    DetectorNoiseRuntime,
    analysis_contrast_noise_variance,
)
from config import OpticalModeSettings, RenderRuntimeConfig, param_value
from config.runtime import internal_param_value, resolved_modality
from imaging_models import get_imaging_model
from imaging_models.sem_source import (
    source_like_numeric_array,
    source_like_projected_array,
    source_like_scaled,
    source_like_sum,
)
from json_utils import json_safe
from mask_generation import generate_central_lobe_mask
from modality_profiles import profile_card_for_model
from modality_registry import is_vectorial_full_field_modality
from particle_model import ParticleInstance
from particle_specs import particle_count
from substrate import sample_environment_from_params
from substrate.patterns import (
    compute_contrast_scale_for_frame,
    generate_empirical_background_field,
    generate_reference_and_background_maps,
    generate_sample_environment_roughness_field,
    resize_empirical_background_field,
)
from supervision_policy import (
    SupervisionAudit,
    SupervisionPolicy,
)
from trajectory import resolve_num_frames

from .canvas import resolve_render_canvas_geometry
from .frame_io import save_supervision_masks, write_supervision_sidecars
from .frame_set import RenderedFrameSet
from .diagnostics import _array_diagnostics, _clip_diagnostics
from .orientation_interpolation import _interpolate_orientation_for_instance
from .per_particle_state import (
    _ParticleFrameRenderState,
    _accumulate_projected_geometry_disk,
    _iter_subparticle_render_info,
)


logger = logging.getLogger(__name__)


def _resize_complex_area(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Area-resize a complex field by resizing real and imaginary parts."""
    field = np.asarray(arr)
    return (
        cv2.resize(field.real.astype(float), size, interpolation=cv2.INTER_AREA)
        + 1j * cv2.resize(field.imag.astype(float), size, interpolation=cv2.INTER_AREA)
    ).astype(np.complex128)


def generate_video_and_masks(params: dict, particle_instances: list[ParticleInstance]) -> RenderedFrameSet:
    """
    Generate video frames and, when enabled, segmentation masks.

    Particles are placed according to their trajectories, optical fields are
    accumulated with the configured motion-blur sampling, and the selected
    imaging model converts the scene into signal/reference frame sequences.

    The mask generation step uses detector-domain, noise-free particle
    contributions:

        - For each particle and frame, every rendered sub-exposure is evaluated
          with all particles and with that particle removed; the difference in
          detector counts is averaged over the exposure.
        - The saved geometry mask preserves the latent projected object
          footprint and, for non-composite particles, can include contrast lobes
          from the final detector-count contribution.
        - Composite particles use flood-fill contrast support around the rendered
          object geometry.

    Supervision policy operates on the same contrast images. The renderer emits
    a supervision annotation schema:

        mask_geometry   projected object/support mask before support gating
        mask_supported  geometry mask after configured support-factor gating
        ignore_mask     object pixels unsupported for the selected target
        loss_weight     per-pixel soft weight encoded as 0..255

    Side effects:
        When ``params["mask_generation_enabled"]`` is true, this function writes
        mask PNGs under ``params["mask_output_directory"]`` and writes the
        sidecar files ``supervision_records.jsonl``, ``supervision_audit.json``,
        and ``annotation_schema.json`` in that directory.

    Returns:
        ``RenderedFrameSet`` containing noisy signal/reference frames and,
        when ``params["return_ideal_float_frames"]`` is true, pre-noise float
        detector-count frames. The ideal-frame lists are empty when that option
        is false.

    """
    render_config = RenderRuntimeConfig.from_params(params)
    fps = render_config.fps
    num_frames = resolve_num_frames(params)
    if bool(param_value(params, 'supervision_stop_when_all_temporally_unsupported')) and render_config.mask_generation_enabled:
        raise ValueError(
            "supervision_stop_when_all_temporally_unsupported is incompatible "
            "with fixed-length video and dataset manifests. Set it to False and "
            "use the emitted mask_supported/ignore_mask sidecars to exclude "
            "unsupported frames."
        )

    frame_interval_s = 1.0 / fps
    rng = np.random.default_rng(render_config.random_seed)
    detector_noise_runtime = DetectorNoiseRuntime(rng=rng)

    exposure_time_ms = (
        1000.0 * frame_interval_s
        if render_config.exposure_time_ms is None
        else render_config.exposure_time_ms
    )
    exposure_time_s = exposure_time_ms / 1000.0

    if exposure_time_s <= 0.0:
        raise ValueError("PARAMS['exposure_time_ms'] must be positive.")
    if exposure_time_s > frame_interval_s + 1e-12:
        raise ValueError(
            "PARAMS['exposure_time_ms'] must satisfy exposure_time_ms <= 1000 / fps "
            "so that the exposure window is contained within a single frame interval."
        )
    signal_exposure_scale = float(exposure_time_s / frame_interval_s)
    params = dict(params)
    params["_exposure_signal_scale"] = signal_exposure_scale
    noise_params = dict(params)
    noise_params["exposure_time_s"] = float(exposure_time_s)

    num_particles = len(particle_instances)
    expected_particle_count = particle_count(params)
    if num_particles != expected_particle_count:
        raise ValueError(
            "Number of ParticleInstance objects (%d) does not match "
            "the number of PARAMS['particles'] entries (%d)."
            % (num_particles, expected_particle_count)
        )

    img_size = render_config.image_size_pixels
    pixel_size_nm = render_config.pixel_size_nm
    os_factor = render_config.psf_oversampling_factor
    final_shape_hw = (img_size, img_size)
    final_dsize_wh = (img_size, img_size)

    # Instantiate before sizing the render canvas: Fourier-domain models need
    # a wider guard band than point-placement alone.
    imaging_model = get_imaging_model(params)
    model_output_type = getattr(imaging_model, "output_type", "intensity")
    scale_signal_counts_for_exposure = model_output_type != "phase"

    def _apply_signal_exposure_scale(counts: np.ndarray) -> np.ndarray:
        if not scale_signal_counts_for_exposure or signal_exposure_scale == 1.0:
            return counts
        return np.asarray(counts, dtype=float) * signal_exposure_scale

    requires_optical_scattered_field = bool(
        getattr(imaging_model, "requires_optical_scattered_field", True)
    )
    pre_crop_optical_filtering = bool(
        getattr(imaging_model, "requires_pre_crop_optical_filtering", False)
    )
    uses_particle_sources = bool(
        getattr(imaging_model, "uses_particle_material_sources", False)
    )
    geometry = resolve_render_canvas_geometry(
        params,
        particle_instances=particle_instances,
        imaging_model=imaging_model,
    )
    os_size = int(geometry["os_size_pixels"])
    render_guard_radius = int(geometry["render_guard_radius_pixels"])
    os_canvas_size = int(geometry["os_canvas_size_pixels"])
    crop_start = int(geometry["crop_start"])
    crop_end = int(geometry["crop_end"])

    # Full-canvas coordinate grids for sub-pixel particle placement. Native
    # scalar-paraxial PSFs use continuous radial placement; vectorial and
    # vectorial-derived scalar fields use direct 2-D field sampling.
    if requires_optical_scattered_field:
        _yy_canvas_grid, _xx_canvas_grid = np.indices(
            (os_canvas_size, os_canvas_size), dtype=np.int32,
        )
        xx_canvas_full = _xx_canvas_grid.astype(np.float64)
        yy_canvas_full = _yy_canvas_grid.astype(np.float64)
        del _xx_canvas_grid, _yy_canvas_grid
    else:
        xx_canvas_full = None
        yy_canvas_full = None

    # Build reference maps after selecting the imaging model so modalities that
    # do not consume optical substrate patterns can explicitly bypass them while
    # leaving the caller's PARAMS untouched.
    pattern_requested = (
        bool(param_value(params, 'sample_environment_enabled'))
        and bool(param_value(params, 'sample_environment_pattern_enabled'))
    )
    pattern_active = (
        pattern_requested
        and bool(getattr(imaging_model, "uses_sample_environment_pattern", False))
    )
    reference_map_params = dict(params)
    if pattern_requested and not pattern_active:
        reference_map_params["sample_environment_pattern_enabled"] = False

    fov_shape_os = (os_size, os_size)
    model_shape_os = (
        (os_canvas_size, os_canvas_size)
        if pre_crop_optical_filtering
        else fov_shape_os
    )
    layout_extent_nm = (
        float(os_canvas_size) * float(pixel_size_nm) / float(os_factor)
        if pre_crop_optical_filtering
        else None
    )
    if pattern_active and layout_extent_nm is not None:
        current_layout_extent_nm = internal_param_value(
            reference_map_params,
            "_substrate_pattern_layout_extent_nm",
        )
        reference_map_params["_substrate_pattern_layout_extent_nm"] = max(
            float(layout_extent_nm),
            float(current_layout_extent_nm)
            if current_layout_extent_nm is not None
            else float(layout_extent_nm),
        )
    sample_environment_model = (
        sample_environment_from_params(
            reference_map_params,
            model_shape_os,
            pixel_size_nm=float(pixel_size_nm) / float(os_factor),
        )
        if bool(param_value(params, 'sample_environment_enabled'))
        else None
    )
    if pre_crop_optical_filtering:
        E_ref_model_base, _, background_model_base = generate_reference_and_background_maps(
            reference_map_params,
            fov_shape_os=model_shape_os,
            final_fov_shape=model_shape_os,
            layout_extent_nm=layout_extent_nm,
        )
        E_ref_os_base = E_ref_model_base[crop_start:crop_end, crop_start:crop_end]
        background_os_base = background_model_base[crop_start:crop_end, crop_start:crop_end]
        E_ref_final_base = _resize_complex_area(E_ref_os_base, final_dsize_wh)
        background_final_base = cv2.resize(
            background_os_base,
            final_dsize_wh,
            interpolation=cv2.INTER_AREA,
        ).astype(float)
    else:
        E_ref_os_base, E_ref_final_base, background_final_base = generate_reference_and_background_maps(
            reference_map_params,
            fov_shape_os=fov_shape_os,
            final_fov_shape=final_shape_hw,
            layout_extent_nm=layout_extent_nm,
        )
        E_ref_model_base = E_ref_os_base
    E_ref_intensity_os_base = np.abs(E_ref_os_base) ** 2
    E_ref_intensity_final_base = np.abs(E_ref_final_base) ** 2
    E_ref_intensity_model_base = np.abs(E_ref_model_base) ** 2

    roughness_model_raw = param_value(params, 'sample_environment_pattern_roughness_model')
    roughness_model = str(roughness_model_raw).strip().lower()
    roughness_amplitude = float(
        param_value(params, 'sample_environment_pattern_roughness_amplitude')
    )
    roughness_correlation_pixels = float(
        param_value(params, 'sample_environment_pattern_roughness_correlation_pixels')
    )
    roughness_phase_std = float(
        param_value(params, 'sample_environment_pattern_roughness_phase_std')
    )
    roughness_source = param_value(params, 'sample_environment_pattern_roughness_source')
    roughness_source_tag = (
        "none"
        if roughness_source is None
        else (
            "path"
            if isinstance(roughness_source, (str, bytes, os.PathLike))
            else "array_like"
        )
    )
    roughness_source_coupling_mode = str(
        param_value(params, 'sample_environment_pattern_roughness_source_coupling')
    ).strip().lower()
    if roughness_source_coupling_mode not in {
        "independent",
        "coherent_amplitude",
        "field_weighted",
        "scene_weighted",
        "channel_weighted",
    }:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_source_coupling'] must "
            "be 'independent', 'coherent_amplitude', 'field_weighted', "
            "'scene_weighted', or 'channel_weighted'."
        )
    use_roughness = (
        bool(param_value(params, 'sample_environment_enabled'))
        and roughness_model != "none"
        and (
            roughness_amplitude > 0.0
            or roughness_phase_std > 0.0
            or roughness_model == "source_matched"
        )
    )
    roughness_dynamic = use_roughness and roughness_model == "flicker"

    roughness_model_base = np.ones((model_shape_os[0], model_shape_os[1]), dtype=np.complex128)
    roughness_os_base = np.ones(fov_shape_os, dtype=np.complex128)
    roughness_final_base = np.ones(final_shape_hw, dtype=np.complex128)
    if use_roughness:
        roughness_model_base = generate_sample_environment_roughness_field(
            reference_map_params,
            model_shape_os,
            rng=rng,
        )
        roughness_model_base = np.asarray(roughness_model_base, dtype=np.complex128)
        roughness_os_base = (
            roughness_model_base[crop_start:crop_end, crop_start:crop_end]
            if pre_crop_optical_filtering
            else roughness_model_base
        )
        roughness_final_base = _resize_complex_area(roughness_os_base, final_dsize_wh)

    empirical_background_enabled = bool(
        param_value(params, 'empirical_background_enabled')
    )
    if empirical_background_enabled:
        if pre_crop_optical_filtering:
            empirical_background_model = generate_empirical_background_field(
                params,
                model_shape_os,
                rng=rng,
            )
            empirical_background_os = empirical_background_model[
                crop_start:crop_end, crop_start:crop_end
            ]
            empirical_background_final = cv2.resize(
                empirical_background_os,
                final_dsize_wh,
                interpolation=cv2.INTER_AREA,
            )
            empirical_background_final = np.clip(
                empirical_background_final, 1e-6, None
            ).astype(float)
        else:
            empirical_background_final = generate_empirical_background_field(
                params,
                final_shape_hw,
                rng=rng,
            )
            empirical_background_os = resize_empirical_background_field(
                empirical_background_final,
                fov_shape_os,
            )
            empirical_background_model = empirical_background_os
        empirical_background_sqrt_os = np.sqrt(empirical_background_os)
        empirical_background_sqrt_model = np.sqrt(empirical_background_model)
    else:
        empirical_background_final = None
        empirical_background_os = None
        empirical_background_model = None
        empirical_background_sqrt_os = None
        empirical_background_sqrt_model = None

    contrast_model_raw = param_value(params, 'sample_environment_pattern_contrast_model')
    contrast_model = str(contrast_model_raw).strip().lower()
    if contrast_model not in ("static", "time_dependent"):
        raise ValueError(
            "Unsupported sample_environment_pattern_contrast_model "
            f"'{contrast_model_raw}'. Supported values are 'static' and 'time_dependent'."
        )
    use_dynamic_contrast = (contrast_model == "time_dependent") and pattern_active

    E_ref_amplitude = render_config.reference_field_amplitude
    background_intensity = render_config.background_intensity

    if use_dynamic_contrast:
        if E_ref_amplitude > 0.0:
            pattern_os_base = E_ref_intensity_os_base / (E_ref_amplitude ** 2)
            pattern_model_base = E_ref_intensity_model_base / (E_ref_amplitude ** 2)
        else:
            pattern_os_base = np.ones_like(E_ref_intensity_os_base, dtype=float)
            pattern_model_base = np.ones_like(E_ref_intensity_model_base, dtype=float)

        if background_intensity > 0.0:
            pattern_final_base = background_final_base / background_intensity
        else:
            pattern_final_base = np.ones_like(background_final_base, dtype=float)

        mean_os = float(pattern_os_base.mean())
        if mean_os > 0.0:
            pattern_os_base /= mean_os

        mean_model = float(pattern_model_base.mean())
        if mean_model > 0.0:
            pattern_model_base /= mean_model

        mean_final = float(pattern_final_base.mean())
        if mean_final > 0.0:
            pattern_final_base /= mean_final

    bit_depth = render_config.bit_depth
    if (
        isinstance(bit_depth, bool)
        or not isinstance(bit_depth, numbers.Integral)
        or bit_depth <= 0
    ):
        raise ValueError("PARAMS['bit_depth'] must be a positive integer.")
    bit_depth = int(bit_depth)

    max_supported_bit_depth = 16
    if bit_depth > max_supported_bit_depth:
        raise ValueError(
            f"PARAMS['bit_depth']={bit_depth} exceeds the maximum supported bit depth "
            f"of {max_supported_bit_depth} for uint16 storage."
        )

    max_camera_count = (1 << bit_depth) - 1

    num_subsamples = render_config.motion_blur_subsamples if render_config.motion_blur_enabled else 1
    if not isinstance(num_subsamples, int) or num_subsamples <= 0:
        raise ValueError(
            "PARAMS['motion_blur_subsamples'] must be a positive integer."
        )
    sub_dt = exposure_time_s / num_subsamples

    all_signal_frames = []
    all_reference_frames = []
    return_ideal_float_frames = render_config.return_ideal_float_frames
    all_signal_ideal_frames = []
    all_reference_ideal_frames = []
    response_function = json_safe(
        imaging_model.compute_response_function(model_shape_os, params)
    )
    backend_fidelity_metadata = json_safe(
        extract_backend_fidelity_metadata(response_function, backend_contract=None)
    )
    profile_card = json_safe(
        profile_card_for_model(
            params,
            imaging_model,
            response_function=response_function,
            model_canvas_shape=model_shape_os,
        )
    )
    render_metadata = {
        "imaging_model": resolved_modality(params),
        "model_class": imaging_model.__class__.__name__,
        "output_type": model_output_type,
        "uses_particle_material_sources": uses_particle_sources,
        "requires_optical_scattered_field": requires_optical_scattered_field,
        "sample_environment_roughness": {
            "enabled": bool(use_roughness),
            "model": roughness_model,
            "source_mode": roughness_source_tag,
            "source_coupling": roughness_source_coupling_mode,
            "amplitude": roughness_amplitude,
            "correlation_pixels": roughness_correlation_pixels,
            "phase_std_rad": roughness_phase_std,
        },
        "render_geometry": {
            key: json_safe(value)
            for key, value in geometry.items()
        },
        "response_function": response_function,
        "backend_fidelity_metadata": backend_fidelity_metadata,
        "modality_profile_card": profile_card,
        "effective_exposure_time_ms": float(exposure_time_ms),
        "effective_exposure_time_s": float(exposure_time_s),
        "frame_interval_s": float(frame_interval_s),
        "signal_exposure_scale": float(signal_exposure_scale),
        "signal_exposure_scaling": (
            "detector_count_budget_scaled_by_exposure_time_over_frame_interval"
            if scale_signal_counts_for_exposure
            else "phase_output_signal_not_display_scaled; noise_quanta_scaled_by_exposure"
        ),
        "clip_diagnostics": [],
        "source_map_diagnostics": [],
    }
    vectorial_dpc_enabled = bool(
        getattr(imaging_model, "_vectorial_dpc_enabled", lambda *_: False)(params)
    )
    optical_settings = OpticalModeSettings.from_params(params)
    vectorial_detection_mode = optical_settings.vectorial_detection_mode
    if vectorial_detection_mode not in {
        "analyzer_x",
        "analyzer_y",
        "incoherent_sum",
        "unpolarized",
        "full_vector",
    }:
        raise ValueError(
            "PARAMS['vectorial_detection_mode'] must be 'analyzer_x', "
            "'analyzer_y', 'incoherent_sum', 'unpolarized', or "
            "'full_vector'; got "
            f"{vectorial_detection_mode!r}."
        )
    vectorial_full_field_enabled = (
        optical_settings.uses_full_vector_field
        and is_vectorial_full_field_modality(resolved_modality(params))
    )
    vectorial_field_enabled = bool(vectorial_dpc_enabled or vectorial_full_field_enabled)
    render_metadata["vectorial_field_pipeline"] = {
        "enabled": vectorial_field_enabled,
        "dpc_vectorial_channel_enabled": bool(vectorial_dpc_enabled),
        "coherent_full_vector_enabled": bool(vectorial_full_field_enabled),
        "vectorial_detection_mode": vectorial_detection_mode,
        "field_canvas_shape": (
            [3, os_canvas_size, os_canvas_size]
            if vectorial_field_enabled
            else [os_canvas_size, os_canvas_size]
        ),
    }
    optical_field_representations = []
    seen_field_representation_keys = set()
    for instance in particle_instances:
        for _world_pos_nm, sub_interp, _mult, _source_mult, _diameter_nm, _material in _iter_subparticle_render_info(
            instance=instance,
            base_position_nm=np.zeros(3, dtype=float),
            orientation_matrix=None,
        ):
            metadata = getattr(sub_interp, "metadata", {}) or {}
            summary = {
                "interpolator_class": type(sub_interp).__name__,
                "backend": metadata.get("backend"),
                "field_representation": metadata.get("field_representation"),
                "scalar_compatibility_reduction": metadata.get("scalar_compatibility_reduction"),
                "vectorial_detection_requested": metadata.get("vectorial_detection_requested"),
                "vectorial_field_reason": metadata.get("vectorial_field_reason"),
            }
            key = tuple(summary.items())
            if key not in seen_field_representation_keys:
                seen_field_representation_keys.add(key)
                optical_field_representations.append(json_safe(summary))
    render_metadata["optical_field_representations"] = optical_field_representations
    vectorial_field_shape = (
        (3, os_canvas_size, os_canvas_size)
        if vectorial_field_enabled
        else (os_canvas_size, os_canvas_size)
    )

    def _new_particle_frame_states() -> list[_ParticleFrameRenderState]:
        states: list[_ParticleFrameRenderState] = []
        for particle_index, _instance in enumerate(particle_instances):
            source_canvas = (
                imaging_model.initialize_particle_source_canvas(
                    (os_canvas_size, os_canvas_size), params
                )
                if uses_particle_sources
                else None
            )
            states.append(
                _ParticleFrameRenderState(
                    field_canvas=np.zeros(vectorial_field_shape, dtype=np.complex128),
                    source_canvas=source_canvas,
                    geometry_canvas=np.zeros(
                        (os_canvas_size, os_canvas_size), dtype=np.float32
                    ),
                    rendered_position_sum_nm=np.zeros(3, dtype=float),
                    particle_index=int(particle_index),
                )
            )
        return states

    def _model_arrays_from_states(
        states: list[_ParticleFrameRenderState],
    ) -> tuple[np.ndarray, list[np.ndarray], list[object] | None]:
        particle_field_canvases = [state.field_canvas for state in states]
        if particle_field_canvases:
            E_sca_total_canvas = np.sum(particle_field_canvases, axis=0)
        else:
            E_sca_total_canvas = np.zeros(
                vectorial_field_shape,
                dtype=np.complex128,
            )
        if pre_crop_optical_filtering:
            E_sca_total_for_model = E_sca_total_canvas
            particle_fields_for_model = particle_field_canvases
            particle_source_maps_for_model = None
            if uses_particle_sources:
                particle_source_maps_for_model = [
                    state.source_canvas for state in states
                ]
        else:
            if np.asarray(E_sca_total_canvas).ndim == 3:
                E_sca_total_for_model = E_sca_total_canvas[
                    :, crop_start:crop_end, crop_start:crop_end
                ]
            else:
                E_sca_total_for_model = E_sca_total_canvas[
                    crop_start:crop_end, crop_start:crop_end
                ]
            particle_fields_for_model = [
                state.field_fov(crop_start, crop_end) for state in states
            ]
            particle_source_maps_for_model = None
            if uses_particle_sources:
                particle_source_maps_for_model = [
                    state.source_fov(crop_start, crop_end) for state in states
                ]
        return (
            E_sca_total_for_model,
            particle_fields_for_model,
            particle_source_maps_for_model,
        )

    def _roughness_intensity_exact(
        roughness_field: np.ndarray,
    ) -> np.ndarray:
        roughness_field = np.asarray(roughness_field, dtype=np.complex128)
        return np.clip(np.abs(roughness_field) ** 2, 1e-6, None)

    def _roughness_scene_envelope(scattered_field: np.ndarray) -> np.ndarray:
        field = np.asarray(scattered_field, dtype=np.complex128)
        if field.ndim == 2:
            field_intensity = np.abs(field) ** 2
        elif field.ndim == 3:
            field_intensity = np.sum(
                np.abs(field[: min(3, field.shape[0])]) ** 2, axis=0
            )
        else:
            raise ValueError(
                "Cannot build roughness scene envelope from scattered field "
                f"with ndim={field.ndim}; expected 2 or 3."
            )

        finite = np.isfinite(field_intensity)
        if not np.any(finite):
            return np.ones_like(field_intensity, dtype=float)
        mean_intensity = float(np.mean(field_intensity[finite]))
        if mean_intensity <= 1e-12:
            return np.ones_like(field_intensity, dtype=float)
        return np.asarray(field_intensity / mean_intensity, dtype=float)

    def _roughness_channel_envelope(
        scattered_field: np.ndarray,
        detection_mode: str,
    ) -> np.ndarray:
        field = np.asarray(scattered_field, dtype=np.complex128)
        if field.ndim == 2:
            return _roughness_scene_envelope(field)
        if field.ndim != 3:
            raise ValueError(
                "Cannot build channel-aware roughness envelope from scattered field "
                f"with ndim={field.ndim}; expected 2 or 3."
            )
        if field.shape[0] < 2:
            return _roughness_scene_envelope(field)

        detection_mode = str(detection_mode).strip().lower()
        if detection_mode == "analyzer_x":
            field_intensity = np.abs(field[0]) ** 2
        elif detection_mode == "analyzer_y":
            field_intensity = np.abs(field[1]) ** 2
        elif detection_mode in {"incoherent_sum", "unpolarized"}:
            field_intensity = 0.5 * (
                np.abs(field[0]) ** 2 + np.abs(field[1]) ** 2
            )
        elif detection_mode == "full_vector":
            num_channels = min(3, field.shape[0])
            field_intensity = np.sum(np.abs(field[:num_channels]) ** 2, axis=0)
        else:
            num_channels = min(3, field.shape[0])
            field_intensity = np.sum(np.abs(field[:num_channels]) ** 2, axis=0)

        finite = np.isfinite(field_intensity)
        if not np.any(finite):
            return np.ones_like(field_intensity, dtype=float)
        mean_intensity = float(np.mean(field_intensity[finite]))
        if mean_intensity <= 1e-12:
            return np.ones_like(field_intensity, dtype=float)
        return np.asarray(field_intensity / mean_intensity, dtype=float)

    def _apply_roughness_to_model_arrays(
        E_sca_total_for_model: np.ndarray,
        particle_fields_for_model: list[np.ndarray],
        particle_source_maps_for_model: list[object] | None,
        roughness_field_for_model: np.ndarray | None,
        roughness_intensity_gain_for_model: np.ndarray | None,
        roughness_source_coupling_mode: str,
        vectorial_detection_mode: str = "full_vector",
    ) -> tuple[
        np.ndarray,
        list[np.ndarray],
        list[object] | None,
    ]:
        if roughness_field_for_model is None:
            return (
                np.asarray(E_sca_total_for_model, dtype=np.complex128),
                [np.asarray(field, dtype=np.complex128) for field in particle_fields_for_model],
                None if particle_source_maps_for_model is None else list(particle_source_maps_for_model),
            )

        roughness_field = np.asarray(roughness_field_for_model, dtype=np.complex128)
        E_sca_unmodified_for_model = np.asarray(
            E_sca_total_for_model,
            dtype=np.complex128,
        )
        has_scattered_signal = bool(
            np.any(np.isfinite(E_sca_unmodified_for_model))
            and np.mean(np.abs(E_sca_unmodified_for_model) ** 2) > 1e-18
        )
        if has_scattered_signal:
            scene_envelope = _roughness_scene_envelope(E_sca_unmodified_for_model)
            source_envelopes = [
                _roughness_scene_envelope(np.asarray(field, dtype=np.complex128))
                for field in particle_fields_for_model
            ] if particle_fields_for_model else []
            channel_envelopes = [
                _roughness_channel_envelope(np.asarray(field, dtype=np.complex128), vectorial_detection_mode)
                for field in particle_fields_for_model
            ]
        else:
            if particle_source_maps_for_model:
                total_source = np.zeros_like(
                    source_like_projected_array(particle_source_maps_for_model[0]),
                    dtype=np.complex128,
                )
                for source_map in particle_source_maps_for_model:
                    total_source += np.asarray(
                        source_like_projected_array(source_map),
                        dtype=np.complex128,
                    )
                scene_envelope = _roughness_scene_envelope(total_source)
                source_envelopes = [
                    _roughness_scene_envelope(
                        np.asarray(source_like_projected_array(source_map), dtype=np.complex128)
                    )
                    for source_map in particle_source_maps_for_model
                ]
                channel_envelopes = list(source_envelopes)
            else:
                scene_envelope = np.ones_like(E_sca_unmodified_for_model.real, dtype=float)
                source_envelopes = []
                channel_envelopes = []

        E_sca_total_for_model = E_sca_unmodified_for_model * roughness_field
        particle_fields_for_model = [
            np.asarray(field, dtype=np.complex128) * roughness_field
            for field in particle_fields_for_model
        ]

        if particle_source_maps_for_model is None:
            source_maps_coupled = None
        else:
            if roughness_source_coupling_mode == "coherent_amplitude":
                if roughness_intensity_gain_for_model is None:
                    roughness_intensity_gain_for_model = _roughness_intensity_exact(
                        roughness_field
                    )
                intensity_gain = np.asarray(roughness_intensity_gain_for_model, dtype=float)
                source_maps_coupled = [
                    source_like_scaled(source_map, intensity_gain)
                    for source_map in particle_source_maps_for_model
                ]
            elif roughness_source_coupling_mode == "field_weighted":
                if roughness_intensity_gain_for_model is None:
                    roughness_intensity_gain_for_model = _roughness_intensity_exact(
                        roughness_field
                    )
                intensity_gain = np.asarray(roughness_intensity_gain_for_model, dtype=float)
                source_maps_coupled = [
                    source_like_scaled(
                        source_map,
                        intensity_gain
                        * np.asarray(
                            source_envelopes[idx]
                            if idx < len(source_envelopes)
                            else scene_envelope,
                            dtype=float,
                        ),
                    )
                    for idx, source_map in enumerate(particle_source_maps_for_model)
                ]
            elif roughness_source_coupling_mode == "scene_weighted":
                if roughness_intensity_gain_for_model is None:
                    roughness_intensity_gain_for_model = _roughness_intensity_exact(
                        roughness_field
                    )
                intensity_gain = np.asarray(roughness_intensity_gain_for_model, dtype=float)
                source_maps_coupled = [
                    source_like_scaled(
                        source_map,
                        intensity_gain * np.asarray(scene_envelope, dtype=float),
                    )
                    for source_map in particle_source_maps_for_model
                ]
            elif roughness_source_coupling_mode == "channel_weighted":
                if roughness_intensity_gain_for_model is None:
                    roughness_intensity_gain_for_model = _roughness_intensity_exact(
                        roughness_field
                    )
                intensity_gain = np.asarray(roughness_intensity_gain_for_model, dtype=float)
                source_maps_coupled = [
                    source_like_scaled(
                        source_map,
                        intensity_gain
                        * np.asarray(
                            channel_envelopes[idx]
                            if idx < len(channel_envelopes)
                            else scene_envelope,
                            dtype=float,
                        ),
                    )
                    for idx, source_map in enumerate(particle_source_maps_for_model)
                ]
            else:
                if roughness_intensity_gain_for_model is None:
                    roughness_intensity_gain_for_model = _roughness_intensity_exact(
                        roughness_field
                    )
                intensity_gain = np.asarray(roughness_intensity_gain_for_model, dtype=float)
                source_maps_coupled = [
                    source_like_scaled(source_map, intensity_gain)
                    for source_map in particle_source_maps_for_model
                ]

        return (
            E_sca_total_for_model,
            particle_fields_for_model,
            source_maps_coupled,
        )

    def _detector_intensity_os_from_states(
        states: list[_ParticleFrameRenderState],
        E_ref_model_current: np.ndarray,
        frame_index: int,
        roughness_field_for_model: np.ndarray | None = None,
        roughness_intensity_gain_for_model: np.ndarray | None = None,
        roughness_source_coupling_mode: str = "independent",
        vectorial_detection_mode: str = "full_vector",
    ) -> np.ndarray:
        (
            E_sca_total_for_model,
            particle_fields_for_model,
            particle_source_maps_for_model,
        ) = _model_arrays_from_states(states)
        (
            E_sca_total_for_model,
            particle_fields_for_model,
            particle_source_maps_for_model,
        ) = _apply_roughness_to_model_arrays(
            E_sca_total_for_model=E_sca_total_for_model,
            particle_fields_for_model=particle_fields_for_model,
            particle_source_maps_for_model=particle_source_maps_for_model,
            roughness_field_for_model=roughness_field_for_model,
            roughness_intensity_gain_for_model=roughness_intensity_gain_for_model,
            roughness_source_coupling_mode=roughness_source_coupling_mode,
            vectorial_detection_mode=vectorial_detection_mode,
        )
        scene_with_environment = getattr(
            imaging_model,
            "compute_scene_intensity_with_sample_environment",
            None,
        )
        if callable(scene_with_environment):
            intensity_for_model = scene_with_environment(
                particle_fields_for_model,
                [particle_instances[state.particle_index] for state in states],
                E_sca_total_for_model,
                E_ref_model_current,
                params,
                particle_source_maps=particle_source_maps_for_model,
                frame_index=frame_index,
                sample_environment=sample_environment_model,
            )
        else:
            scene_from_render_states = getattr(
                imaging_model,
                "compute_scene_intensity_from_render_states",
                None,
            )
            if callable(scene_from_render_states):
                intensity_for_model = scene_from_render_states(
                    particle_fields_for_model,
                    states,
                    E_sca_total_for_model,
                    E_ref_model_current,
                    params,
                    particle_source_maps=particle_source_maps_for_model,
                    frame_index=frame_index,
                )
            else:
                intensity_for_model = imaging_model.compute_scene_intensity(
                    particle_fields_for_model,
                    [particle_instances[state.particle_index] for state in states],
                    E_sca_total_for_model,
                    E_ref_model_current,
                    params,
                    particle_source_maps=particle_source_maps_for_model,
                    frame_index=frame_index,
                )
            intensity_for_model = imaging_model.apply_sample_environment(
                intensity=intensity_for_model,
                E_sca_total=E_sca_total_for_model,
                background_field=E_ref_model_current,
                params=params,
                sample_environment=sample_environment_model,
            )
        if pre_crop_optical_filtering:
            return intensity_for_model[crop_start:crop_end, crop_start:crop_end]
        return intensity_for_model

    def _detector_counts_from_states(
        states: list[_ParticleFrameRenderState],
        E_ref_model_current: np.ndarray,
        background_final_current: np.ndarray,
        E_ref_intensity_final_current: np.ndarray,
        frame_index: int,
        roughness_field_for_model: np.ndarray | None = None,
        roughness_intensity_gain_for_model: np.ndarray | None = None,
        roughness_source_coupling_mode: str = "independent",
        vectorial_detection_mode: str = "full_vector",
    ) -> np.ndarray:
        intensity_os_current = _detector_intensity_os_from_states(
            states,
            E_ref_model_current,
            frame_index,
            roughness_field_for_model=roughness_field_for_model,
            roughness_intensity_gain_for_model=roughness_intensity_gain_for_model,
            roughness_source_coupling_mode=roughness_source_coupling_mode,
            vectorial_detection_mode=vectorial_detection_mode,
        )
        intensity_current = cv2.resize(
            intensity_os_current,
            final_dsize_wh,
            interpolation=cv2.INTER_AREA,
        )
        return _apply_signal_exposure_scale(
            imaging_model.scale_intensity_to_counts(
                intensity=intensity_current,
                background_final=background_final_current,
                E_ref_intensity_final=E_ref_intensity_final_current,
                params=params,
            )
        )

    return_mask_arrays = bool(internal_param_value(params, "_return_mask_arrays"))
    write_mask_files = bool(internal_param_value(params, "_write_mask_files"))
    returned_mask_arrays: list[dict] = []

    if render_config.mask_generation_enabled:
        supervision_policy = SupervisionPolicy(params, num_particles)
        supervision_audit = SupervisionAudit()
        supervision_records: list[dict] = []
        mask_root_dir = render_config.mask_output_directory
        if write_mask_files:
            for schema_name in (
                "mask_geometry",
                "mask_supported",
                "ignore_mask",
                "loss_weight",
            ):
                for i in range(num_particles):
                    os.makedirs(
                        os.path.join(mask_root_dir, schema_name, f"particle_{i + 1}"),
                        exist_ok=True,
                    )
    else:
        supervision_policy = None
        supervision_audit = None
        supervision_records = []

    logger.info("Generating video frames and masks...")
    for f in tqdm(range(num_frames), disable=not logger.isEnabledFor(logging.INFO)):
        if use_dynamic_contrast:
            alpha_f = compute_contrast_scale_for_frame(params, f, num_frames)

            pattern_os_f = 1.0 + alpha_f * (pattern_os_base - 1.0)
            pattern_model_f = 1.0 + alpha_f * (pattern_model_base - 1.0)
            pattern_final_f = 1.0 + alpha_f * (pattern_final_base - 1.0)

            pattern_os_f = np.maximum(pattern_os_f, 1e-8)
            pattern_model_f = np.maximum(pattern_model_f, 1e-8)
            pattern_final_f = np.maximum(pattern_final_f, 1e-8)

            E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os_f)).astype(np.complex128)
            E_ref_model = (E_ref_amplitude * np.sqrt(pattern_model_f)).astype(np.complex128)

            E_ref_intensity_final = (E_ref_amplitude ** 2) * pattern_final_f
            background_final = background_intensity * pattern_final_f
        else:
            E_ref_os = E_ref_os_base
            E_ref_model = E_ref_model_base
            E_ref_intensity_final = E_ref_intensity_final_base
            background_final = background_final_base

        roughness_model_f = None
        roughness_intensity_gain_model_f = None
        if use_roughness:
            if roughness_dynamic:
                roughness_model_f = generate_sample_environment_roughness_field(
                    reference_map_params,
                    model_shape_os,
                    rng=rng,
                ).astype(np.complex128)
                roughness_os_f = (
                    roughness_model_f[crop_start:crop_end, crop_start:crop_end]
                    if pre_crop_optical_filtering
                    else roughness_model_f
                )
                roughness_final_f = _resize_complex_area(roughness_os_f, final_dsize_wh)
            else:
                roughness_model_f = roughness_model_base
                roughness_os_f = roughness_os_base
                roughness_final_f = roughness_final_base

            roughness_intensity_gain_model_f = _roughness_intensity_exact(
                roughness_model_f
            )
            roughness_intensity_gain_final_f = _roughness_intensity_exact(
                roughness_final_f
            )

            E_ref_os = (E_ref_os * roughness_os_f).astype(np.complex128)
            E_ref_model = (E_ref_model * roughness_model_f).astype(np.complex128)
            E_ref_intensity_final = (
                E_ref_intensity_final
                * roughness_intensity_gain_final_f
            )
            background_final = background_final * roughness_intensity_gain_final_f

        if empirical_background_enabled:
            E_ref_os = (E_ref_os * empirical_background_sqrt_os).astype(np.complex128)
            E_ref_model = (E_ref_model * empirical_background_sqrt_model).astype(np.complex128)
            E_ref_intensity_final = E_ref_intensity_final * empirical_background_final
            background_final = background_final * empirical_background_final

        particle_frame_states = _new_particle_frame_states()

        drift_velocity = np.asarray(
            param_value(params, 'drift_velocity_nm_per_s'),
            dtype=float,
        )
        if drift_velocity.size == 1:
            drift_velocity = np.array([float(drift_velocity), 0.0, 0.0])
        if drift_velocity.size != 3:
            raise ValueError("drift_velocity_nm_per_s must be a scalar or length-3 sequence.")

        frame_vibration_nm = np.zeros(3, dtype=float)
        vibration_std_nm = float(param_value(params, 'vibration_jitter_std_nm'))
        if vibration_std_nm > 0.0:
            frame_vibration_nm = rng.normal(scale=vibration_std_nm, size=3)
            if not bool(param_value(params, 'vibration_include_axial')):
                frame_vibration_nm[2] = 0.0

        intensity_os_sum = None
        subsample_states_by_exposure: list[list[_ParticleFrameRenderState]] = []
        for s in range(num_subsamples):
            subsample_states = _new_particle_frame_states()
            frame_center_time = (f + 0.5) * frame_interval_s
            start_time = frame_center_time - 0.5 * exposure_time_s
            current_time = start_time + (s + 0.5) * sub_dt
            global_motion_shift_nm = drift_velocity * current_time + frame_vibration_nm

            # Trajectory samples are indexed by output frame number: trajectory[f]
            # is the particle state represented by frame f and by the masks saved
            # for frame f. Physical time starts at the beginning of frame 0, so
            # the center of frame f is at (f + 0.5) * frame_interval_s. Convert
            # physical sub-exposure times back to this frame-centered index.
            time_index_float = (current_time / frame_interval_s) - 0.5

            if num_frames == 1:
                frame_idx_floor = 0
                frame_idx_ceil = 0
                interp_factor = 0.0
            elif time_index_float < 0.0:
                frame_idx_floor = 0
                frame_idx_ceil = 1
                interp_factor = time_index_float
            elif time_index_float >= num_frames - 1:
                frame_idx_floor = num_frames - 2
                frame_idx_ceil = num_frames - 1
                interp_factor = time_index_float - frame_idx_floor
            else:
                frame_idx_floor = int(np.floor(time_index_float))
                frame_idx_ceil = frame_idx_floor + 1
                interp_factor = time_index_float - frame_idx_floor

            for i, instance in enumerate(particle_instances):
                frame_state = particle_frame_states[i]
                subsample_state = subsample_states[i]
                traj = instance.trajectory_nm
                if traj.shape[0] != num_frames or traj.shape[1] != 3:
                    raise ValueError(
                        "ParticleInstance %d has trajectory shape %s, expected (%d, 3)."
                        % (i, traj.shape, num_frames)
                    )

                pos_floor = traj[frame_idx_floor]
                pos_ceil = traj[frame_idx_ceil]
                current_pos_nm = (1.0 - interp_factor) * pos_floor + interp_factor * pos_ceil
                current_pos_nm = current_pos_nm + global_motion_shift_nm
                frame_state.add_rendered_position(current_pos_nm)

                orientation_matrix = _interpolate_orientation_for_instance(
                    instance=instance,
                    time_index_float=time_index_float,
                )

                sub_infos = _iter_subparticle_render_info(
                    instance=instance,
                    base_position_nm=current_pos_nm,
                    orientation_matrix=orientation_matrix,
                )

                for world_pos_nm, sub_interp, local_multiplier, local_source_multiplier, sub_diameter_nm, sub_material in sub_infos:
                    px, py, pz = world_pos_nm
                    center_x_canvas = crop_start + px / pixel_size_nm * os_factor
                    center_y_canvas = crop_start + py / pixel_size_nm * os_factor
                    _accumulate_projected_geometry_disk(
                        frame_state.geometry_canvas,
                        center_x_canvas=center_x_canvas,
                        center_y_canvas=center_y_canvas,
                        diameter_nm=float(sub_diameter_nm),
                        pixel_size_nm=float(pixel_size_nm),
                        os_factor=int(os_factor),
                    )

                    if requires_optical_scattered_field:
                        E_sca_2D = sub_interp([pz])[0]

                        # Sub-pixel particle placement. Scalar PSFs use the
                        # established radial profile path. Full-vector PSFs
                        # keep their 2-D component maps because Ex/Ey/Ez can
                        # carry polarization-dependent azimuthal structure.
                        is_vector_field = False
                        if E_sca_2D.ndim == 2:
                            field_components = [E_sca_2D]
                        elif E_sca_2D.ndim == 3 and E_sca_2D.shape[0] == 3:
                            is_vector_field = True
                            field_components = [E_sca_2D[comp] for comp in range(3)]
                        else:
                            raise ValueError(
                                "Particle PSF must be 2D scalar or 3xHxW vectorial."
                            )
                        field_metadata = getattr(sub_interp, "metadata", {}) or {}
                        use_radial_scalar_placement = (
                            not is_vector_field
                            and field_metadata.get("backend") == "scalar_paraxial"
                            and field_metadata.get("scalar_compatibility_reduction") == "native_scalar_paraxial"
                        )

                        for component_index, E_component in enumerate(field_components):
                            pupil_samples = E_component.shape[0]
                            center_psf = pupil_samples // 2
                            particle_x_canvas = float(center_x_canvas)
                            particle_y_canvas = float(center_y_canvas)
                            nm_per_pixel_canvas = pixel_size_nm / os_factor
                            nm_per_pixel_psf = nm_per_pixel_canvas
                            amplitude_scale = (
                                instance.signal_multiplier
                                * local_multiplier
                                * float(field_metadata.get("field_amplitude_scale", 1.0))
                            )

                            if is_vector_field:
                                dx_canvas = xx_canvas_full - particle_x_canvas
                                dy_canvas = yy_canvas_full - particle_y_canvas
                                x_psf = center_psf + dx_canvas * nm_per_pixel_canvas / nm_per_pixel_psf
                                y_psf = center_psf + dy_canvas * nm_per_pixel_canvas / nm_per_pixel_psf
                                E_real = map_coordinates(
                                    E_component.real,
                                    [y_psf, x_psf],
                                    order=1,
                                    mode="constant",
                                    cval=0.0,
                                )
                                E_imag = map_coordinates(
                                    E_component.imag,
                                    [y_psf, x_psf],
                                    order=1,
                                    mode="constant",
                                    cval=0.0,
                                )
                                field_contribution = (E_real + 1j * E_imag) * amplitude_scale
                                for target_state in (frame_state, subsample_state):
                                    target_state.field_canvas[component_index] += field_contribution
                            elif use_radial_scalar_placement:
                                E_radial_line = E_component[center_psf, center_psf:]
                                max_bin_psf = E_radial_line.size - 1
                                if max_bin_psf <= 0:
                                    continue
                                r_bins_nm = (
                                    np.arange(max_bin_psf + 1, dtype=float)
                                    * nm_per_pixel_psf
                                )
                                dx_canvas = xx_canvas_full - particle_x_canvas
                                dy_canvas = yy_canvas_full - particle_y_canvas
                                r_canvas_nm = np.sqrt(
                                    dx_canvas * dx_canvas + dy_canvas * dy_canvas
                                ) * nm_per_pixel_canvas
                                E_real = np.interp(
                                    r_canvas_nm.ravel(),
                                    r_bins_nm,
                                    E_radial_line.real,
                                    right=0.0,
                                ).reshape(r_canvas_nm.shape)
                                E_imag = np.interp(
                                    r_canvas_nm.ravel(),
                                    r_bins_nm,
                                    E_radial_line.imag,
                                    right=0.0,
                                ).reshape(r_canvas_nm.shape)
                                field_contribution = (E_real + 1j * E_imag) * amplitude_scale
                                for target_state in (frame_state, subsample_state):
                                    target_state.field_canvas += field_contribution
                            else:
                                dx_canvas = xx_canvas_full - particle_x_canvas
                                dy_canvas = yy_canvas_full - particle_y_canvas
                                x_psf = center_psf + dx_canvas * nm_per_pixel_canvas / nm_per_pixel_psf
                                y_psf = center_psf + dy_canvas * nm_per_pixel_canvas / nm_per_pixel_psf
                                E_real = map_coordinates(
                                    E_component.real,
                                    [y_psf, x_psf],
                                    order=1,
                                    mode="constant",
                                    cval=0.0,
                                )
                                E_imag = map_coordinates(
                                    E_component.imag,
                                    [y_psf, x_psf],
                                    order=1,
                                    mode="constant",
                                    cval=0.0,
                                )
                                field_contribution = (E_real + 1j * E_imag) * amplitude_scale
                                for target_state in (frame_state, subsample_state):
                                    target_state.field_canvas += field_contribution

                    # Source-canvas accumulation uses the same sub-pixel canvas
                    # coordinate as the scattered-field placement.
                    if frame_state.source_canvas is not None:
                        for target_state in (frame_state, subsample_state):
                            imaging_model.accumulate_particle_source(
                                target_state.source_canvas,
                                center_x_canvas=center_x_canvas,
                                center_y_canvas=center_y_canvas,
                                diameter_nm=float(sub_diameter_nm),
                                pixel_size_nm=float(pixel_size_nm),
                                os_factor=int(os_factor),
                                material_properties=sub_material,
                                params=params,
                                particle_z_nm=float(pz),
                                source_multiplier=float(instance.source_multiplier) * float(local_source_multiplier),
                            )

            subsample_intensity_os = _detector_intensity_os_from_states(
                subsample_states,
                E_ref_model,
                f,
                roughness_field_for_model=roughness_model_f,
                roughness_intensity_gain_for_model=roughness_intensity_gain_model_f,
                roughness_source_coupling_mode=roughness_source_coupling_mode,
                vectorial_detection_mode=vectorial_detection_mode,
            )
            subsample_states_by_exposure.append(subsample_states)
            if intensity_os_sum is None:
                intensity_os_sum = np.asarray(subsample_intensity_os, dtype=float)
            else:
                intensity_os_sum += np.asarray(subsample_intensity_os, dtype=float)

        for frame_state in particle_frame_states:
            frame_state.normalize_exposure(num_subsamples)

        if uses_particle_sources:
            source_maps = [
                state.source_canvas
                for state in particle_frame_states
                if state.source_canvas is not None
            ]
            if source_maps:
                source_sum = source_like_sum(source_maps)
                source_sum_array = source_like_numeric_array(source_sum)
                if np.asarray(source_sum_array).ndim == 3:
                    source_fov = source_sum_array[:, crop_start:crop_end, crop_start:crop_end]
                else:
                    source_fov = source_sum_array[crop_start:crop_end, crop_start:crop_end]
                diag = {"frame_index": int(f)}
                diag.update(_array_diagnostics(source_fov, prefix="source_map"))
                diag["source_map_ndim"] = int(np.asarray(source_fov).ndim)
                diag["source_axis_order"] = response_function.get(
                    "source_axis_order",
                    "zyx" if np.asarray(source_fov).ndim == 3 else "yx",
                )
                diag["source_input_kind"] = response_function.get(
                    "source_input_kind",
                    "sliced_sem_source_volume"
                    if np.asarray(source_fov).ndim == 3
                    else "projected_2d_source_map",
                )
                diag["source_projection_policy"] = response_function.get(
                    "source_projection_policy",
                    None,
                )
                for source_key in (
                    "source_z_origin",
                    "source_z_offset_nm",
                    "source_slice_thickness_nm",
                    "source_z_planes_nm",
                    "source_z_uses_particle_world_z",
                    "source_representation_request_satisfied",
                ):
                    if source_key in response_function:
                        diag[source_key] = response_function.get(source_key)
                render_metadata["source_map_diagnostics"].append(diag)

        if intensity_os_sum is None:
            raise RuntimeError("No detector-domain subexposure intensity was rendered.")
        intensity_os = intensity_os_sum / float(num_subsamples)
        intensity = cv2.resize(intensity_os, final_dsize_wh, interpolation=cv2.INTER_AREA)

        intensity_scaled = _apply_signal_exposure_scale(
            imaging_model.scale_intensity_to_counts(
                intensity=intensity,
                background_final=background_final,
                E_ref_intensity_final=E_ref_intensity_final,
                params=params,
            )
        )

        # Render a no-particle reference through the same model and sample
        # environment so background subtraction compares like with like.
        if vectorial_field_enabled:
            zero_field_for_model = np.zeros((3,) + E_ref_model.shape, dtype=np.complex128)
        else:
            zero_field_for_model = np.zeros_like(E_ref_model, dtype=np.complex128)
        scene_with_environment = getattr(
            imaging_model,
            "compute_scene_intensity_with_sample_environment",
            None,
        )
        if callable(scene_with_environment):
            reference_intensity_for_model = scene_with_environment(
                [],
                [],
                zero_field_for_model,
                E_ref_model,
                params,
                particle_source_maps=[] if uses_particle_sources else None,
                frame_index=f,
                sample_environment=sample_environment_model,
            )
        else:
            reference_intensity_for_model = imaging_model.compute_scene_intensity(
                [],
                [],
                zero_field_for_model,
                E_ref_model,
                params,
                particle_source_maps=[] if uses_particle_sources else None,
                frame_index=f,
            )
            reference_intensity_for_model = imaging_model.apply_sample_environment(
                intensity=reference_intensity_for_model,
                E_sca_total=zero_field_for_model,
                background_field=E_ref_model,
                params=params,
                sample_environment=sample_environment_model,
            )
        if pre_crop_optical_filtering:
            reference_intensity_os = reference_intensity_for_model[
                crop_start:crop_end, crop_start:crop_end
            ]
        else:
            reference_intensity_os = reference_intensity_for_model
        reference_intensity = cv2.resize(
            reference_intensity_os,
            final_dsize_wh,
            interpolation=cv2.INTER_AREA,
        )
        reference_frame_ideal = _apply_signal_exposure_scale(
            imaging_model.scale_intensity_to_counts(
                intensity=reference_intensity,
                background_final=background_final,
                E_ref_intensity_final=E_ref_intensity_final,
                params=params,
            )
        )
        supervision_noise_variance = analysis_contrast_noise_variance(
            intensity_scaled,
            reference_frame_ideal,
            noise_params,
            relative_reference=False,
            runtime=detector_noise_runtime,
        )
        finite_variance = supervision_noise_variance[
            np.isfinite(supervision_noise_variance)
        ]
        supervision_noise_std = (
            float(np.sqrt(np.median(finite_variance)))
            if finite_variance.size
            else float("inf")
        )

        if render_config.mask_generation_enabled:
            subsample_all_counts = [
                _detector_counts_from_states(
                    states,
                    E_ref_model,
                    background_final,
                    E_ref_intensity_final,
                    f,
                    roughness_field_for_model=roughness_model_f,
                    roughness_intensity_gain_for_model=roughness_intensity_gain_model_f,
                    roughness_source_coupling_mode=roughness_source_coupling_mode,
                    vectorial_detection_mode=vectorial_detection_mode,
                )
                for states in subsample_states_by_exposure
            ]
            mask_inputs = []
            for i, instance in enumerate(particle_instances):
                frame_state = particle_frame_states[i]
                contribution_sum = None
                for states, counts_all in zip(
                    subsample_states_by_exposure,
                    subsample_all_counts,
                ):
                    states_without_particle = [
                        state for idx, state in enumerate(states) if idx != i
                    ]
                    counts_without_particle = _detector_counts_from_states(
                        states_without_particle,
                        E_ref_model,
                        background_final,
                        E_ref_intensity_final,
                        f,
                        roughness_field_for_model=roughness_model_f,
                        roughness_intensity_gain_for_model=roughness_intensity_gain_model_f,
                        roughness_source_coupling_mode=roughness_source_coupling_mode,
                        vectorial_detection_mode=vectorial_detection_mode,
                    )
                    contribution = counts_all - counts_without_particle
                    if contribution_sum is None:
                        contribution_sum = np.asarray(contribution, dtype=float)
                    else:
                        contribution_sum += np.asarray(contribution, dtype=float)
                if contribution_sum is None:
                    contrast_final_counts = np.zeros(final_shape_hw, dtype=float)
                else:
                    contrast_final_counts = contribution_sum / float(
                        len(subsample_states_by_exposure)
                    )
                if getattr(imaging_model, "output_type", "intensity") == "phase":
                    phase_to_count = float(param_value(params, "qpi_phase_to_count_scale"))
                    if not np.isfinite(phase_to_count) or phase_to_count <= 0.0:
                        raise ValueError(
                            "PARAMS['qpi_phase_to_count_scale'] must be positive "
                            "for phase-domain supervision support."
                        )
                    contrast_for_support = contrast_final_counts / phase_to_count
                    support_signal_units = "radian"
                    support_measurement_domain = "phase"
                    support_noise_variance_units = "radian_squared"
                else:
                    contrast_for_support = contrast_final_counts
                    support_signal_units = "detector_count"
                    support_measurement_domain = "detector_count"
                    support_noise_variance_units = "detector_count_squared"

                H, W = contrast_for_support.shape
                geometry_os = frame_state.geometry_canvas[
                    crop_start:crop_end, crop_start:crop_end
                ]
                geometry_final = cv2.resize(
                    geometry_os, final_dsize_wh, interpolation=cv2.INTER_AREA
                )
                projected_geometry_mask = (geometry_final > 0.0).astype(np.uint8) * 255
                if projected_geometry_mask.shape != (H, W):
                    raise RuntimeError(
                        "Latent geometry mask shape does not match contrast image shape."
                    )
                position_nm = frame_state.rendered_position_nm(instance.trajectory_nm[f, :])
                center_yx = (
                    int(round(float(position_nm[1]) / float(pixel_size_nm))),
                    int(round(float(position_nm[0]) / float(pixel_size_nm))),
                )
                is_composite = bool(
                    getattr(getattr(instance, "particle_type", None), "is_composite", False)
                )
                lobe_mask, lobe_status = generate_central_lobe_mask(
                    contrast_final_counts,
                    center_yx=center_yx,
                    outer_ring_count=0 if is_composite else int(param_value(params, 'mask_outer_ring_count')),
                    use_floodfill=is_composite,
                    max_area_fraction=float(param_value(params, 'mask_max_area_fraction')),
                    return_status=True,
                )
                geometry_mask = np.maximum(projected_geometry_mask, lobe_mask).astype(np.uint8)
                mask_geometry_metadata = {
                    "mask_geometry_composition": "projected_object_or_contrast_support",
                    "mask_geometry_units": "final_detector_grid_pixels",
                    "mask_geometry_algorithm": "floodfill_contrast_support" if is_composite else "radial_sign_change_contrast_support",
                    "projected_geometry_pixels": int(np.count_nonzero(projected_geometry_mask)),
                    "contrast_support_pixels": int(np.count_nonzero(lobe_mask)),
                    "mask_outer_ring_count": 0 if is_composite else int(param_value(params, 'mask_outer_ring_count')),
                    "mask_max_area_fraction": float(param_value(params, 'mask_max_area_fraction')),
                }
                mask_geometry_metadata.update(lobe_status)

                mask_inputs.append(
                    {
                        "particle_index": i,
                        "frame_index": f,
                        "position_nm": position_nm,
                        "contrast_image": contrast_for_support,
                        "geometry_mask": geometry_mask,
                        "mask_geometry_metadata": mask_geometry_metadata,
                    }
                )

            all_positions_nm = np.asarray(
                [item["position_nm"] for item in mask_inputs],
                dtype=float,
            )
            all_geometry_masks = [
                item["geometry_mask"] for item in mask_inputs
            ]
            for item in mask_inputs:
                policy_result = supervision_policy.evaluate(
                    particle_index=item["particle_index"],
                    frame_index=item["frame_index"],
                    position_nm=item["position_nm"],
                    contrast_image=item["contrast_image"],
                    geometry_mask=item["geometry_mask"],
                    all_positions_nm=all_positions_nm,
                    all_geometry_masks=all_geometry_masks,
                    noise_std=supervision_noise_std,
                    noise_variance_map=supervision_noise_variance,
                    signal_units=support_signal_units,
                    measurement_domain=support_measurement_domain,
                    noise_variance_units=support_noise_variance_units,
                )

                masks = policy_result["masks"]
                record = policy_result["record"]
                record.update(item["mask_geometry_metadata"])
                supervision_records.append(record)
                supervision_audit.add(record)
                if return_mask_arrays:
                    returned_mask_arrays.append(
                        {
                            "particle_index": int(item["particle_index"]),
                            "frame_index": int(item["frame_index"]),
                            "masks": {
                                str(schema_name): np.asarray(mask_arr).copy()
                                for schema_name, mask_arr in masks.items()
                            },
                        }
                    )

                if write_mask_files:
                    save_supervision_masks(
                        masks,
                        mask_root_dir,
                        particle_index=item["particle_index"],
                        frame_index=item["frame_index"],
                    )

        if return_ideal_float_frames:
            all_signal_ideal_frames.append(
                np.where(np.isfinite(intensity_scaled), intensity_scaled, 0.0).astype(float, copy=False)
            )

        signal_frame_noisy = imaging_model.compute_noise(
            intensity_scaled,
            noise_params,
            rng=rng,
            detector_noise_runtime=detector_noise_runtime,
        )
        clip_diag = {"frame_index": int(f)}
        clip_diag.update(
            _clip_diagnostics(
                signal_frame_noisy,
                max_camera_count=max_camera_count,
                prefix="signal",
            )
        )
        all_signal_frames.append(
            np.clip(signal_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        if return_ideal_float_frames:
            all_reference_ideal_frames.append(
                np.where(np.isfinite(reference_frame_ideal), reference_frame_ideal, 0.0).astype(float, copy=False)
            )
        reference_frame_noisy = imaging_model.compute_noise(
            reference_frame_ideal,
            noise_params,
            rng=rng,
            detector_noise_runtime=detector_noise_runtime,
        )
        clip_diag.update(
            _clip_diagnostics(
                reference_frame_noisy,
                max_camera_count=max_camera_count,
                prefix="reference",
            )
        )
        render_metadata["clip_diagnostics"].append(clip_diag)
        all_reference_frames.append(
            np.clip(reference_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

    logger.info("Frame and mask generation complete.")
    supervision_audit_summary = (
        json_safe(supervision_audit.summary())
        if supervision_audit is not None else None
    )
    if render_config.mask_generation_enabled and write_mask_files:
        write_supervision_sidecars(
            params=params,
            supervision_records=supervision_records,
            supervision_audit_summary=supervision_audit_summary or {},
        )
    response_function = json_safe(
        imaging_model.compute_response_function(model_shape_os, params)
    )
    render_metadata["response_function"] = response_function
    render_metadata["backend_fidelity_metadata"] = json_safe(
        extract_backend_fidelity_metadata(response_function, backend_contract=None)
    )
    return RenderedFrameSet(
        signal_frames=all_signal_frames,
        reference_frames=all_reference_frames,
        ideal_signal_frames=all_signal_ideal_frames,
        ideal_reference_frames=all_reference_ideal_frames,
        mask_arrays=returned_mask_arrays,
        supervision_records=list(supervision_records),
        supervision_audit_summary=supervision_audit_summary,
        render_metadata=json_safe(render_metadata),
    )
