import os
import logging
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from config import PARAMS, validate_params
from camera_noise import analysis_contrast_noise_variance, apply_camera_noise_counts
from fisher_diagnostic import compute_localization_crlb
from materials import (
    resolve_component_refractive_index,
    resolve_primary_component_refractive_indices,
    resolve_particle_material_properties,
)
from particle_specs import get_particle_specs, normalize_particle_specs, particle_count
from trajectory import resolve_num_frames, simulate_trajectories, simulate_orientations
from optics import compute_complex_psf_stack
from rendering import (
    estimate_optical_filter_guard_radius_pixels,
    estimate_psf_padding_radius_pixels,
    generate_video_and_masks,
)
from postprocessing import (
    apply_background_subtraction,
    save_video,
    compute_single_frame_contrast,
    normalize_contrast_frames,
)
from particle_model import build_particle_types_and_instances
from imaging_model import get_imaging_model, modality_uses_relative_reference_contrast

logger = logging.getLogger(__name__)

# Hard upper bound (in nm) on the per-particle-type z-stack range, used as
# a global safety cap in run_simulation's trajectory-based sizing code. The
# production pipeline derives each particle type's z-range from its realized
# Brownian trajectory; this constant is only applied to protect against
# pathological configurations that would otherwise produce extremely expensive
# iPSF stacks.
_MAX_AUTO_Z_STACK_RANGE_NM = 200000.0
# Trajectory-derived iPSF stacks need slack for axial render perturbations and
# interpolation edges: 5 z-steps absolute margin, 10% relative margin, and at
# least 100 z-steps of half-span for nearly stationary trajectories.
_Z_GRID_ABSOLUTE_MARGIN_STEPS = 5.0
_Z_GRID_RELATIVE_MARGIN_FRACTION = 0.10
_Z_GRID_MIN_HALF_SPAN_STEPS = 100.0
_NUM_FRAME_DURATION_SEARCH_STEPS = 32
_VISIBLE_WAVELENGTH_MIN_NM = 380.0
_VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM = 420.0
_VISIBLE_WAVELENGTH_VIOLET_BLUE_NM = 440.0
_VISIBLE_WAVELENGTH_BLUE_GREEN_NM = 490.0
_VISIBLE_WAVELENGTH_GREEN_CYAN_NM = 510.0
_VISIBLE_WAVELENGTH_YELLOW_RED_NM = 580.0
_VISIBLE_WAVELENGTH_RED_EDGE_NM = 645.0
_VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM = 700.0
_VISIBLE_WAVELENGTH_MAX_NM = 780.0
_WAVELENGTH_RGB_EDGE_FACTOR = 0.3
_WAVELENGTH_RGB_GAMMA = 0.8
_RUNTIME_PARAM_KEYS = {
    "_particle_specs",
    "_resolved_particles",
    "_resolved_primary_component_refractive_indices",
    "_resolved_particle_material_properties",
    "_resolved_particle_material_properties_metadata",
    "_camera_noise_static_seed",
    "_return_mask_arrays",
    "_write_mask_files",
    "_substrate_pattern_layout_cache_token",
    "_substrate_pattern_layout_extent_nm",
}



def _channel_spec_to_params(base_params: dict, channel, channel_index: int) -> tuple[str, dict, np.ndarray, float]:
    """
    Resolve one spectral sample.

    Returns:
        channel_name:
            Human-readable channel label.
        channel_params:
            PARAMS clone with wavelength/probe overrides for this spectral sample.
        detector_weights_rgb:
            Length-3 detector response weights [R, G, B] for this sample.
        spectral_weight:
            Scalar spectral source/integration weight for this sample.
    """
    channel_params = deepcopy(base_params)
    channel_params.pop("channels", None)

    if isinstance(channel, dict):
        channel_name = str(channel.get("name", f"ch{channel_index + 1}"))
        channel_params.update(
            {k: v for k, v in channel.items()
             if k not in {"name", "rgb", "detector_weights_rgb", "detector_weights", "weight", "spectral_weight"}}
        )
        wavelength_nm = float(channel_params.get("wavelength_nm", base_params.get("wavelength_nm", 532.0)))
        spectral_weight = float(channel.get("spectral_weight", channel.get("weight", 1.0)))
        weights = None
        for weights_key in ("detector_weights_rgb", "detector_weights", "rgb"):
            if weights_key in channel and channel[weights_key] is not None:
                weights = channel[weights_key]
                break
        if weights is None:
            detector_weights_rgb = _wavelength_to_rgb_weights(wavelength_nm)
        else:
            detector_weights_rgb = np.asarray(weights, dtype=float)
    else:
        wavelength_nm = float(channel)
        channel_name = f"{wavelength_nm:.0f}nm"
        channel_params["wavelength_nm"] = wavelength_nm
        channel_params["probe_wavelength_nm"] = wavelength_nm
        spectral_weight = 1.0
        detector_weights_rgb = _wavelength_to_rgb_weights(wavelength_nm)

    if not np.isfinite(spectral_weight) or spectral_weight < 0.0:
        raise ValueError(
            "Each channel spectral_weight/weight must be finite and non-negative; "
            f"got {spectral_weight!r} for channel {channel_name!r}."
        )

    if detector_weights_rgb.shape != (3,):
        raise ValueError(
            "Each channel detector weight must be length 3, ordered [R, G, B]. "
            f"Got shape {detector_weights_rgb.shape} for channel {channel_name!r}."
        )
    if not np.all(np.isfinite(detector_weights_rgb)):
        raise ValueError(
            "Each channel detector weight must contain only finite values; "
            f"got {detector_weights_rgb!r} for channel {channel_name!r}."
        )
    if np.any(detector_weights_rgb < 0.0):
        raise ValueError(
            "Each channel detector weight must be non-negative; "
            f"got {detector_weights_rgb!r} for channel {channel_name!r}."
        )

    channel_params["wavelength_nm"] = wavelength_nm
    probe_wavelength_nm = channel_params.get("probe_wavelength_nm", None)
    if probe_wavelength_nm is None:
        probe_wavelength_nm = wavelength_nm
    channel_params["probe_wavelength_nm"] = float(probe_wavelength_nm)
    validate_params(channel_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    return channel_name, channel_params, detector_weights_rgb.astype(float), spectral_weight


def _wavelength_to_rgb_weights(wavelength_nm: float) -> np.ndarray:
    """
    Approximate visible-wavelength display/detector response.

    This piecewise display approximation is deterministic and preserves
    visible-wavelength ordering for RGB sidecars. Scientific spectral work
    should pass explicit
    detector_weights_rgb and spectral_weight per sample.
    """
    wl = float(wavelength_nm)
    if wl < _VISIBLE_WAVELENGTH_MIN_NM or wl > _VISIBLE_WAVELENGTH_MAX_NM:
        return np.zeros(3, dtype=float)

    if wl < _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM:
        r = -(
            wl - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM
        ) / (_VISIBLE_WAVELENGTH_VIOLET_BLUE_NM - _VISIBLE_WAVELENGTH_MIN_NM)
        g = 0.0
        b = 1.0
    elif wl < _VISIBLE_WAVELENGTH_BLUE_GREEN_NM:
        r = 0.0
        g = (
            wl - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM
        ) / (_VISIBLE_WAVELENGTH_BLUE_GREEN_NM - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM)
        b = 1.0
    elif wl < _VISIBLE_WAVELENGTH_GREEN_CYAN_NM:
        r = 0.0
        g = 1.0
        b = -(
            wl - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM
        ) / (_VISIBLE_WAVELENGTH_GREEN_CYAN_NM - _VISIBLE_WAVELENGTH_BLUE_GREEN_NM)
    elif wl < _VISIBLE_WAVELENGTH_YELLOW_RED_NM:
        r = (
            wl - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM
        ) / (_VISIBLE_WAVELENGTH_YELLOW_RED_NM - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM)
        g = 1.0
        b = 0.0
    elif wl < _VISIBLE_WAVELENGTH_RED_EDGE_NM:
        r = 1.0
        g = -(
            wl - _VISIBLE_WAVELENGTH_RED_EDGE_NM
        ) / (_VISIBLE_WAVELENGTH_RED_EDGE_NM - _VISIBLE_WAVELENGTH_YELLOW_RED_NM)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if wl < _VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM:
        factor = _WAVELENGTH_RGB_EDGE_FACTOR + (
            1.0 - _WAVELENGTH_RGB_EDGE_FACTOR
        ) * (wl - _VISIBLE_WAVELENGTH_MIN_NM) / (
            _VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM - _VISIBLE_WAVELENGTH_MIN_NM
        )
    elif wl <= _VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM:
        factor = 1.0
    else:
        factor = _WAVELENGTH_RGB_EDGE_FACTOR + (
            1.0 - _WAVELENGTH_RGB_EDGE_FACTOR
        ) * (_VISIBLE_WAVELENGTH_MAX_NM - wl) / (
            _VISIBLE_WAVELENGTH_MAX_NM - _VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM
        )

    return np.asarray([
        (max(r, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
        (max(g, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
        (max(b, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
    ], dtype=float)


def _disable_detector_noise_for_spectral_component(params: dict) -> dict:
    """
    Return a params clone for deterministic spectral rendering.

    For broadband/RGB rendering, noise belongs after spectral integration into
    detector channels. Rendering each wavelength with independent shot/readout
    noise and then summing channels is not the correct measurement model.

    This disables all noise by setting the canonical camera_noise.py toggles
    and per-pixel artefact controls to their "off" states.
    """
    p = deepcopy(params)
    # Disable via canonical camera_noise.py parameter names.
    p["shot_noise_enabled"] = False
    p["gaussian_noise_enabled"] = False
    p["fixed_pattern_gain_std"] = 0.0
    p["fixed_pattern_offset_counts"] = 0.0
    p["hot_pixel_fraction"] = 0.0
    p["scan_line_noise_counts"] = 0.0
    p["dark_offset_counts"] = 0.0
    # Also clear override containers so no per-modality detector noise can be
    # reintroduced before the post-integration RGB noise pass.
    noise_model = dict(p.get("noise_model", {}) or {})
    noise_model.update(
        shot_noise_enabled=False,
        gaussian_noise_enabled=False,
        fixed_pattern_gain_std=0.0,
        fixed_pattern_offset_counts=0.0,
        hot_pixel_fraction=0.0,
        scan_line_noise_counts=0.0,
        dark_offset_counts=0.0,
    )
    p["noise_model"] = noise_model
    p["modality_noise"] = {}
    return p


def _setup_output_dirs(params: dict) -> None:
    if params["mask_generation_enabled"]:
        base_mask_dir = params["mask_output_directory"]
        logger.info("Checking for mask output directories at %s...", base_mask_dir)
        os.makedirs(base_mask_dir, exist_ok=True)

    output_dir = os.path.dirname(params["output_filename"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _build_safe_z_grid_for_type(
    z_min_realized_nm: float,
    z_max_realized_nm: float,
    z_step_nm: float,
) -> np.ndarray:
    z_min_realized_nm = float(z_min_realized_nm)
    z_max_realized_nm = float(z_max_realized_nm)
    z_step_nm = float(z_step_nm)

    if z_step_nm <= 0.0:
        raise ValueError("PARAMS['z_stack_step_nm'] must be positive.")

    if z_max_realized_nm < z_min_realized_nm:
        z_min_realized_nm, z_max_realized_nm = z_max_realized_nm, z_min_realized_nm

    z_center = 0.5 * (z_min_realized_nm + z_max_realized_nm)
    realized_half_span = 0.5 * (z_max_realized_nm - z_min_realized_nm)

    absolute_margin_nm = _Z_GRID_ABSOLUTE_MARGIN_STEPS * z_step_nm
    relative_margin_factor = _Z_GRID_RELATIVE_MARGIN_FRACTION
    min_half_span_nm = _Z_GRID_MIN_HALF_SPAN_STEPS * z_step_nm

    safe_half_span = realized_half_span
    safe_half_span += absolute_margin_nm
    safe_half_span *= (1.0 + relative_margin_factor)
    safe_half_span = max(safe_half_span, min_half_span_nm)

    max_half_span_allowed = 0.5 * _MAX_AUTO_Z_STACK_RANGE_NM
    if safe_half_span > max_half_span_allowed:
        safe_half_span = max_half_span_allowed

    z_min_safe = z_center - safe_half_span
    z_max_safe = z_center + safe_half_span

    z_values = np.arange(z_min_safe, z_max_safe + z_step_nm, z_step_nm, dtype=float)
    if z_values.size < 2:
        z_values = np.array(
            [z_center - z_step_nm * 0.5, z_center + z_step_nm * 0.5],
            dtype=float,
        )
    return z_values


def _include_axial_render_perturbation(
    params: dict,
    z_positions_nm: np.ndarray,
    duration_seconds: float,
) -> tuple[float, float]:
    """
    Return z min/max after deterministic render-time axial perturbations.

    Brownian trajectories are the physical particle path. Rendering can add
    bench motion (drift/vibration) on top of that path before the PSF lookup.
    The PSF cache must cover the rendered path, not only the raw Brownian path.
    """
    z = np.asarray(z_positions_nm, dtype=float)
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    drift_velocity = np.asarray(
        params.get("drift_velocity_nm_per_s", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    if drift_velocity.size == 1:
        drift_z_nm_per_s = 0.0
    elif drift_velocity.size == 3:
        drift_z_nm_per_s = float(drift_velocity[2])
    else:
        raise ValueError("drift_velocity_nm_per_s must be a scalar or length-3 sequence.")

    drift_end_nm = drift_z_nm_per_s * max(float(duration_seconds), 0.0)
    z_min += min(0.0, drift_end_nm)
    z_max += max(0.0, drift_end_nm)

    if bool(params.get("vibration_include_axial", False)):
        vibration_std_nm = float(params.get("vibration_jitter_std_nm", 0.0))
        if vibration_std_nm > 0.0:
            vibration_margin_nm = 4.0 * vibration_std_nm
            z_min -= vibration_margin_nm
            z_max += vibration_margin_nm

    return z_min, z_max


def _simulate_latent_scene(params: dict) -> dict:
    """
    Simulate wavelength-independent latent scene state once.

    This includes trajectories and rigid-body orientations. Wavelength-dependent
    optical constants and PSFs are intentionally not built here; those are built
    per spectral sample against this same latent scene.
    """
    if params.get("random_seed", None) is not None:
        np.random.seed(int(params["random_seed"]))

    normalize_particle_specs(params, mutate=True)
    trajectories_nm = simulate_trajectories(params)

    num_particles = particle_count(params)
    num_frames = resolve_num_frames(params)
    orientations = simulate_orientations(params, num_particles, num_frames)

    return {
        "trajectories_nm": trajectories_nm,
        "orientations": orientations,
        "num_frames": num_frames,
    }


def _collect_type_keys_required(params: dict) -> tuple[dict, set]:
    particle_specs = get_particle_specs(params)
    type_to_component_refs = {}
    type_keys_required = set()

    for i, spec in enumerate(particle_specs):
        for component in spec.components:
            n_component = resolve_component_refractive_index(params, component)
            key = (
                float(component.diameter_nm),
                float(n_component.real),
                float(n_component.imag),
            )
            type_keys_required.add(key)
            type_to_component_refs.setdefault(key, []).append(
                (i, float(component.offset_nm[2]))
            )

    return type_to_component_refs, type_keys_required


def _build_particle_instances_for_scene(params: dict, latent_scene: dict):
    """
    Build wavelength-dependent particle instances for a fixed latent scene.
    """
    resolve_primary_component_refractive_indices(params)
    resolve_particle_material_properties(params)
    type_to_component_refs, type_keys_required = _collect_type_keys_required(params)

    logger.info("Pre-computing unique particle complex PSF stacks with trajectory-based Z-ranges...")
    z_step_nm = float(params["z_stack_step_nm"])
    if z_step_nm <= 0.0:
        raise ValueError("PARAMS['z_stack_step_nm'] must be positive.")

    psf_interpolators_by_type = {}
    default_z_range_nm = float(params.get("z_stack_range_nm", 30500.0))
    default_half_span_nm = 0.5 * default_z_range_nm
    use_shared_psf_z_grid = bool(params.get("shared_psf_z_grid_enabled", False))
    trajectories_nm = latent_scene["trajectories_nm"]
    duration_seconds = float(params["duration_seconds"])

    for type_key in sorted(type_keys_required):
        diam_nm_type, n_real, n_imag = type_key
        component_refs = type_to_component_refs.get(type_key, None)

        if use_shared_psf_z_grid:
            z_min_type = -default_half_span_nm
            z_max_type = default_half_span_nm
            if component_refs is not None and len(component_refs) > 0:
                indices_array = np.asarray([ref[0] for ref in component_refs], dtype=int)
                z_offsets = np.asarray([ref[1] for ref in component_refs], dtype=float)[:, None]
                z_positions_type = trajectories_nm[indices_array, :, 2] + z_offsets
                z_min_rendered, z_max_rendered = _include_axial_render_perturbation(
                    params,
                    z_positions_type,
                    duration_seconds,
                )
                z_values_safe = _build_safe_z_grid_for_type(
                    z_min_realized_nm=z_min_rendered,
                    z_max_realized_nm=z_max_rendered,
                    z_step_nm=z_step_nm,
                )
                z_min_type = min(z_min_type, float(z_values_safe[0]))
                z_max_type = max(z_max_type, float(z_values_safe[-1]))
            z_values_type = np.arange(
                z_min_type,
                z_max_type + z_step_nm,
                z_step_nm,
                dtype=float,
            )
        elif component_refs is not None and len(component_refs) > 0:
            indices_array = np.asarray([ref[0] for ref in component_refs], dtype=int)
            z_offsets = np.asarray([ref[1] for ref in component_refs], dtype=float)[:, None]
            z_positions_type = trajectories_nm[indices_array, :, 2] + z_offsets
            z_min_rendered, z_max_rendered = _include_axial_render_perturbation(
                params,
                z_positions_type,
                duration_seconds,
            )
            z_values_type = _build_safe_z_grid_for_type(
                z_min_realized_nm=z_min_rendered,
                z_max_realized_nm=z_max_rendered,
                z_step_nm=z_step_nm,
            )
        else:
            z_center = 0.0
            z_values_type = np.arange(
                z_center - default_half_span_nm,
                z_center + default_half_span_nm + z_step_nm,
                z_step_nm,
                dtype=float,
            )

        logger.info(
            "  Particle type (diameter = %.1f nm, n = %.4f + %.4fi): "
            "z-range [% .1f, % .1f] nm with %d slices.",
            float(diam_nm_type),
            float(n_real),
            float(n_imag),
            float(z_values_type[0]),
            float(z_values_type[-1]),
            int(z_values_type.size),
        )

        psf_interpolators_by_type[type_key] = compute_complex_psf_stack(
            params,
            float(diam_nm_type),
            complex(float(n_real), float(n_imag)),
            z_values_type,
        )

    _, particle_instances = build_particle_types_and_instances(
        params=params,
        trajectories_nm=latent_scene["trajectories_nm"],
        psf_interpolators_by_type=psf_interpolators_by_type,
        orientations=latent_scene["orientations"],
    )
    return particle_instances


def _frames_to_channel_first(frames, *, channel_count: int) -> np.ndarray:
    """Convert returned display frames to the public (T, C, H, W) schema."""
    arr = np.asarray(frames)
    if arr.size == 0:
        return np.empty((0, int(channel_count), 0, 0), dtype=np.uint8)
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 4 and arr.shape[-1] in {1, 3, 4}:
        return np.moveaxis(arr, -1, 1)
    if arr.ndim == 4:
        return arr
    raise ValueError(
        "Returned frames must have shape (T, H, W), (T, H, W, C), or "
        f"(T, C, H, W); got {arr.shape}."
    )


def _simulation_result(frames, channels: list[str], metadata: dict) -> dict:
    channels = [str(ch) for ch in channels]
    if not channels:
        raise ValueError("Simulation result channels must be non-empty.")
    frame_array = _frames_to_channel_first(frames, channel_count=len(channels))
    if frame_array.ndim != 4:
        raise ValueError(
            "Simulation result frames must use the public (T, C, H, W) schema; "
            f"got {frame_array.shape}."
        )
    if frame_array.shape[1] != len(channels):
        raise ValueError(
            "Simulation result channel count mismatch: "
            f"frames have C={frame_array.shape[1]} but channels={channels!r}."
        )
    return {
        "frames": frame_array,
        "channels": channels,
        "metadata": dict(metadata),
    }


def _jsonable_crlb_summary(crlb: dict) -> dict:
    """Return the packet metadata subset of a localization CRLB result."""
    summary_keys = (
        "sigma_x_nm",
        "sigma_y_nm",
        "sigma_xy_nm",
        "fisher_det",
        "singular",
        "rank",
        "axes_singular",
    )
    out = {key: crlb.get(key) for key in summary_keys if key in crlb}
    if "fisher_matrix" in crlb:
        out["fisher_shape"] = list(np.asarray(crlb["fisher_matrix"]).shape)
    return out


def _packet_jsonable(value: Any) -> Any:
    """Convert selected packet scene metadata to JSON-safe Python objects."""
    if is_dataclass(value):
        return _packet_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _packet_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_packet_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _packet_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _packet_jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _packet_sample_environment_metadata(params: dict) -> dict:
    """Return public scene-environment fields relevant to a matched packet."""
    keys = (
        "sample_environment_enabled",
        "sample_environment",
        "sample_environment_pattern_enabled",
        "sample_environment_pattern",
        "sample_environment_pattern_pitch_um",
        "sample_environment_pattern_hole_diameter_um",
        "sample_environment_pattern_bar_width_um",
        "sample_environment_pattern_material",
        "sample_environment_pattern_height_nm",
        "mounting_interface",
        "refractive_index_medium",
        "refractive_index_immersion",
    )
    return {key: _packet_jsonable(params.get(key)) for key in keys if key in params}


def _source_map_provenance(params: dict) -> dict:
    model = get_imaging_model(params)
    return {
        "imaging_model": str(params.get("imaging_model", "bright_field")),
        "uses_particle_material_sources": bool(
            getattr(model, "uses_particle_material_sources", False)
        ),
        "source_response_applied_before_fov_crop": bool(
            getattr(model, "requires_pre_crop_optical_filtering", False)
        ),
        "source_map_coordinate_frame": (
            "oversampled_render_canvas"
            if getattr(model, "uses_particle_material_sources", False)
            else None
        ),
    }


def _resolve_public_num_frames(params: dict) -> None:
    """Apply PARAMS['num_frames'] by resolving duration_seconds."""
    raw_num_frames = params.get("num_frames", None)
    if raw_num_frames is None:
        return
    if isinstance(raw_num_frames, bool):
        raise ValueError("PARAMS['num_frames'] must be a positive integer, not bool.")
    if isinstance(raw_num_frames, (float, np.floating)) and not float(raw_num_frames).is_integer():
        raise ValueError("PARAMS['num_frames'] must be an integer frame count.")
    try:
        requested_num_frames = int(raw_num_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("PARAMS['num_frames'] must be a positive integer.") from exc
    if requested_num_frames <= 0:
        raise ValueError("PARAMS['num_frames'] must be positive.")

    fps = float(params.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError("PARAMS['fps'] must be positive when num_frames is set.")

    duration_seconds = requested_num_frames / fps
    for _ in range(_NUM_FRAME_DURATION_SEARCH_STEPS):
        if int(fps * duration_seconds) == requested_num_frames:
            params["duration_seconds"] = float(duration_seconds)
            return
        duration_seconds = float(np.nextafter(duration_seconds, np.inf))

    raise RuntimeError(
        "Could not choose duration_seconds that reproduces "
        f"num_frames={requested_num_frames} at fps={fps}."
    )


def _ensure_run_scope_layout_token(params: dict) -> None:
    """
    Give unseeded sample-environment layouts a per-run cache token.

    This keeps optical backgrounds and Brownian exclusion geometry consistent
    within one run without reusing the first unseeded layout across later runs
    in the same Python process.
    """
    if not bool(params.get("sample_environment_pattern_enabled", False)):
        return
    if not bool(params.get("sample_environment_enabled", True)):
        return
    if params.get("_substrate_pattern_layout_cache_token", None) is not None:
        return
    if params.get("random_seed", None) is not None:
        return
    params["_substrate_pattern_layout_cache_token"] = (
        f"run:{int(np.random.SeedSequence().entropy)}"
    )


def _ensure_run_scope_layout_extent(params: dict) -> None:
    """Use one substrate layout extent for trajectory and optical consumers."""
    if not bool(params.get("sample_environment_pattern_enabled", False)):
        return
    if not bool(params.get("sample_environment_enabled", True)):
        return

    img_size = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = float(params["psf_oversampling_factor"])
    if img_size <= 0 or pixel_size_nm <= 0.0 or os_factor <= 0.0:
        return

    imaging_model = get_imaging_model(params)
    pre_crop_optical_filtering = bool(
        getattr(imaging_model, "requires_pre_crop_optical_filtering", False)
    )
    layout_extent_nm = float(img_size) * pixel_size_nm
    if pre_crop_optical_filtering:
        os_size = int(img_size * os_factor)
        render_guard_radius = max(
            estimate_psf_padding_radius_pixels(params),
            estimate_optical_filter_guard_radius_pixels(params),
        )
        layout_extent_nm = (
            float(os_size + 2 * int(render_guard_radius))
            * pixel_size_nm
            / os_factor
        )
    current_extent = params.get("_substrate_pattern_layout_extent_nm", None)
    if current_extent is not None:
        layout_extent_nm = max(float(layout_extent_nm), float(current_extent))
    params["_substrate_pattern_layout_extent_nm"] = float(layout_extent_nm)


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
                "source_map_provenance": _source_map_provenance(params),
            })
        return None

    if save_video_output:
        img_size = (params["image_size_pixels"], params["image_size_pixels"])
        save_video(params["output_filename"], final_frames, params["fps"], img_size)

    if return_frames:
        return _simulation_result(final_frames, ["default"], {
            "raw_signal_frames": list(raw_signal_frames),
            "raw_reference_frames": list(raw_reference_frames),
            "ideal_signal_frames": list(ideal_signal_frames),
            "ideal_reference_frames": list(ideal_reference_frames),
            "background_subtracted_frames": list(final_frames),
            "mask_arrays": list(getattr(rendered, "mask_arrays", [])),
            "supervision_records": list(getattr(rendered, "supervision_records", [])),
            "supervision_audit_summary": getattr(rendered, "supervision_audit_summary", None),
            "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
            "source_map_provenance": _source_map_provenance(params),
        })

    return None


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
    modality_names = [str(modality).strip() for modality in modalities]
    if len(modality_names) < 2 or any(not name for name in modality_names):
        raise ValueError("modalities must contain at least two non-empty names.")

    base_params = deepcopy(params)
    if base_params.get("channels") is not None:
        raise ValueError("matched modality packets cannot be combined with PARAMS['channels'].")
    base_params["channels"] = None
    _resolve_public_num_frames(base_params)
    _ensure_run_scope_layout_token(base_params)
    validate_params(base_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    max_layout_extent = None
    for modality in modality_names:
        extent_params = deepcopy(base_params)
        extent_params["imaging_model"] = modality
        _ensure_run_scope_layout_extent(extent_params)
        extent = extent_params.get("_substrate_pattern_layout_extent_nm", None)
        if extent is not None:
            max_layout_extent = (
                float(extent)
                if max_layout_extent is None
                else max(float(max_layout_extent), float(extent))
            )
    if max_layout_extent is not None:
        base_params["_substrate_pattern_layout_extent_nm"] = float(max_layout_extent)
        validate_params(base_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    latent_scene = _simulate_latent_scene(base_params)

    images_by_modality: dict[str, np.ndarray] = {}
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
        validate_params(modality_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
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
        noise_variance = analysis_contrast_noise_variance(
            signal_frame,
            reference_frame,
            modality_params,
        )
        crlb = compute_localization_crlb(
            np.asarray(contrast_frame, dtype=float),
            np.asarray(noise_variance, dtype=float),
            pixel_size_nm=float(modality_params["pixel_size_nm"]),
        )
        images_by_modality[modality] = np.asarray(contrast_frame, dtype=float)
        fisher_by_modality[modality] = np.asarray(crlb["fisher_matrix"], dtype=float)
        crlb_by_modality[modality] = _jsonable_crlb_summary(crlb)
        for mask_entry in metadata.get("mask_arrays", []) or []:
            if int(mask_entry.get("frame_index", -1)) != int(frame_index):
                continue
            particle_number = int(mask_entry.get("particle_index", 0)) + 1
            for mask_name, mask_arr in dict(mask_entry.get("masks", {}) or {}).items():
                masks[f"{modality}__{mask_name}__particle_{particle_number}"] = np.asarray(mask_arr)
        output_type = getattr(get_imaging_model(modality_params), "output_type", "intensity")
        if output_type == "phase":
            contrast_units = "radians"
        elif modality_uses_relative_reference_contrast(modality):
            contrast_units = "relative_reference"
        else:
            contrast_units = "detector_count_difference"
        modality_metadata[modality] = {
            "imaging_model": modality,
            "wavelength_nm": float(modality_params.get("wavelength_nm", 0.0)),
            "probe_wavelength_nm": modality_params.get("probe_wavelength_nm"),
            "output_type": output_type,
            "contrast_frame_units": contrast_units,
            "fisher_source": "analysis_contrast_frame",
        }

    trajectories = np.asarray(latent_scene.get("trajectories_nm", []), dtype=float)
    latent_state = {
        "frame_index": int(frame_index),
        "num_frames": int(latent_scene.get("num_frames", 0)),
        "random_seed": base_params.get("random_seed"),
        "trajectories_nm": trajectories.tolist(),
        "orientations": _packet_jsonable(latent_scene.get("orientations")),
        "particles": _packet_jsonable(get_particle_specs(base_params)),
        "sample_environment": _packet_sample_environment_metadata(base_params),
    }
    return {
        "latent_state": latent_state,
        "images_by_modality": images_by_modality,
        "masks": masks,
        "fisher_by_modality": fisher_by_modality,
        "crlb_by_modality": crlb_by_modality,
        "metadata": {
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


def _save_rgb_video(path: str, frames_rgb: list[np.ndarray], fps: float) -> None:
    """Save RGB frames through the canonical video writer."""
    if not frames_rgb:
        return
    save_video(path, frames_rgb, fps, color_order="rgb")


def _multichannel_output_mode(params: dict) -> str:
    """
    Resolve multichannel video output mode.

    Single-channel simulations do not use this. For multichannel/spectral
    simulations, this prevents forced RGB output and lets callers request
    RGB visualization, per-channel grayscale sidecars, both, or returned arrays
    only.
    """
    raw = str(params.get("multichannel_output_mode", "rgb")).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    allowed = {"rgb", "channels", "both", "none"}
    if raw not in allowed:
        raise ValueError(
            "PARAMS['multichannel_output_mode'] must be one of "
            "{'rgb', 'channels', 'both', 'none'}; got "
            f"{params.get('multichannel_output_mode')!r}."
        )

    return raw


def _safe_channel_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))
    safe = safe.strip("._")
    return safe or "channel"


def _save_channel_videos(params: dict, spectral_items: list[dict], fps: float) -> list[str]:
    """
    Save per-channel grayscale sidecar videos.

    These sidecars preserve individual channel renderings. They are separate
    from the RGB visualization composite and are only written when
    ``multichannel_output_mode`` is ``"channels"`` or ``"both"``.
    """
    sidecar_dir = params.get("multichannel_sidecar_directory", None)
    if not sidecar_dir:
        stem, _ = os.path.splitext(params["output_filename"])
        sidecar_dir = stem + "_channels"

    os.makedirs(sidecar_dir, exist_ok=True)

    written = []
    used_names: set[str] = set()
    for channel_index, item in enumerate(spectral_items):
        base_name = _safe_channel_filename(item.get("name", "channel"))
        name = base_name
        if name in used_names:
            suffix = f"_ch{channel_index + 1}"
            name = f"{base_name}{suffix}"
            suffix_count = 2
            while name in used_names:
                name = f"{base_name}{suffix}_{suffix_count}"
                suffix_count += 1
        used_names.add(name)
        frame_meta = dict(item.get("frames", {}).get("metadata", {}) or {})
        frames = list(
            frame_meta.get("background_subtracted_frames", [])
            or []
        )
        if not frames:
            continue

        first = np.asarray(frames[0])
        if first.ndim != 2:
            raise ValueError(
                "Channel sidecar frames must be grayscale arrays with shape (H, W); "
                f"got {first.shape} for channel {name!r}."
            )

        h, w = first.shape
        out_path = os.path.join(sidecar_dir, f"{name}.avi")
        save_video(out_path, frames, fps, (w, h))
        written.append(out_path)

    return written



def _apply_detector_noise_to_rgb_raw_frames(
    frames_rgb_float: list[np.ndarray],
    params: dict,
) -> list[np.ndarray]:
    out = []
    for frame_rgb in frames_rgb_float:
        frame_rgb = np.asarray(frame_rgb, dtype=float)
        noisy_channels = []
        for c in range(3):
            noisy_channels.append(apply_camera_noise_counts(frame_rgb[:, :, c], params))
        out.append(np.stack(noisy_channels, axis=-1))
    return out


def _background_subtract_rgb(
    signal_rgb_frames: list[np.ndarray],
    reference_rgb_frames: list[np.ndarray],
    params: dict,
) -> list[np.ndarray]:
    final_channels = []
    for c in range(3):
        sig_c = [np.asarray(f, dtype=float)[:, :, c] for f in signal_rgb_frames]
        ref_c = [np.asarray(f, dtype=float)[:, :, c] for f in reference_rgb_frames]
        final_c = apply_background_subtraction(sig_c, ref_c, params)
        final_channels.append([np.asarray(x, dtype=np.uint8) for x in final_c])

    n_frames = min(len(final_channels[0]), len(final_channels[1]), len(final_channels[2]))
    return [
        np.stack(
            [final_channels[0][i], final_channels[1][i], final_channels[2][i]],
            axis=-1,
        ).astype(np.uint8)
        for i in range(n_frames)
    ]


def _validate_multichannel_frame_arrays(
    raw_signal_arrays: list[np.ndarray],
    raw_reference_arrays: list[np.ndarray],
) -> tuple[int, int, int]:
    if not raw_signal_arrays:
        raise ValueError("Multichannel rendering produced no channel frame arrays.")
    if len(raw_signal_arrays) != len(raw_reference_arrays):
        raise ValueError(
            "Multichannel signal/reference channel counts differ: "
            f"{len(raw_signal_arrays)} signal vs {len(raw_reference_arrays)} reference."
        )

    expected_shape: tuple[int, int, int] | None = None
    for channel_index, (sig_arr, ref_arr) in enumerate(zip(raw_signal_arrays, raw_reference_arrays)):
        if sig_arr.size == 0 or ref_arr.size == 0:
            raise ValueError(
                f"Multichannel channel {channel_index} produced empty signal/reference frames."
            )
        if sig_arr.ndim != 3 or ref_arr.ndim != 3:
            raise ValueError(
                "Multichannel channel arrays must have shape (T, H, W); "
                f"channel {channel_index} has signal {sig_arr.shape} and reference {ref_arr.shape}."
            )
        if sig_arr.shape != ref_arr.shape:
            raise ValueError(
                f"Multichannel channel {channel_index} signal/reference shapes differ: "
                f"{sig_arr.shape} vs {ref_arr.shape}."
            )
        if expected_shape is None:
            expected_shape = tuple(int(v) for v in sig_arr.shape)
        elif sig_arr.shape != expected_shape:
            raise ValueError(
                "All multichannel channel arrays must have the same shape; "
                f"channel 0 has {expected_shape}, channel {channel_index} has {sig_arr.shape}."
            )
        if not np.all(np.isfinite(sig_arr)) or not np.all(np.isfinite(ref_arr)):
            raise ValueError(
                f"Multichannel channel {channel_index} contains non-finite signal/reference values."
            )

    assert expected_shape is not None
    return expected_shape


def _run_multichannel_simulation(
    params: dict,
    channels,
    *,
    return_frames: bool = False,
):
    """
    Render one latent scene through multiple spectral samples and integrate into RGB.

    Important invariant:
        trajectories, orientations, particle identities, masks, and sample geometry
        are simulated once. Each spectral sample rebuilds wavelength-dependent
        materials/PSFs against that same latent scene. Detector noise is applied
        only after RGB channel integration.
    """
    if not isinstance(channels, (list, tuple)) or len(channels) == 0:
        raise ValueError("PARAMS['channels'] must be a non-empty list when set.")

    latent_scene = _simulate_latent_scene(params)

    spectral_items = []
    for channel_index, channel in enumerate(channels):
        channel_name, channel_params, detector_weights_rgb, spectral_weight = _channel_spec_to_params(
            params,
            channel,
            channel_index,
        )
        deterministic_params = _disable_detector_noise_for_spectral_component(channel_params)
        deterministic_params["mask_generation_enabled"] = bool(channel_index == 0 and params.get("mask_generation_enabled", True))

        frames = _render_scene_with_params(
            deterministic_params,
            latent_scene,
            save_video_output=False,
            return_frames=True,
        ) or {}

        spectral_items.append(
            {
                "name": channel_name,
                "params": channel_params,
                "detector_weights_rgb": detector_weights_rgb,
                "spectral_weight": float(spectral_weight),
                "frames": frames,
            }
        )

    raw_signal_arrays = [
        np.asarray(item["frames"].get("metadata", {}).get("raw_signal_frames", []), dtype=float)
        for item in spectral_items
    ]
    raw_reference_arrays = [
        np.asarray(item["frames"].get("metadata", {}).get("raw_reference_frames", []), dtype=float)
        for item in spectral_items
    ]

    n_frames, h, w = _validate_multichannel_frame_arrays(
        raw_signal_arrays,
        raw_reference_arrays,
    )

    signal_rgb_float = []
    reference_rgb_float = []
    for t in range(n_frames):
        sig_rgb = np.zeros((h, w, 3), dtype=float)
        ref_rgb = np.zeros((h, w, 3), dtype=float)
        for item, sig_arr, ref_arr in zip(spectral_items, raw_signal_arrays, raw_reference_arrays):
            weights = item["detector_weights_rgb"] * item["spectral_weight"]
            for c in range(3):
                sig_rgb[:, :, c] += sig_arr[t] * weights[c]
                ref_rgb[:, :, c] += ref_arr[t] * weights[c]
        signal_rgb_float.append(sig_rgb)
        reference_rgb_float.append(ref_rgb)

    signal_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(signal_rgb_float, params)
    reference_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(reference_rgb_float, params)

    final_rgb = _background_subtract_rgb(signal_rgb_noisy, reference_rgb_noisy, params)
    output_mode = _multichannel_output_mode(params)
    written_channel_sidecars = []
    if output_mode in {"rgb", "both"}:
        _save_rgb_video(params["output_filename"], final_rgb, float(params["fps"]))
    if output_mode in {"channels", "both"}:
        written_channel_sidecars = _save_channel_videos(
            params,
            spectral_items,
            float(params["fps"]),
        )


    if return_frames:
        return _simulation_result(final_rgb, ["red", "green", "blue"], {
            "spectral_channels": [item["name"] for item in spectral_items],
            "spectral_items": spectral_items,
            "raw_signal_frames_by_spectral_sample": raw_signal_arrays,
            "raw_reference_frames_by_spectral_sample": raw_reference_arrays,
            "raw_signal_frames_rgb": signal_rgb_noisy,
            "raw_reference_frames_rgb": reference_rgb_noisy,
            "background_subtracted_frames_rgb": final_rgb,
            "multichannel_output_mode": output_mode,
            "channel_sidecar_videos": written_channel_sidecars,
        })

    return None


def run_simulation(params: dict, return_frames: bool = False):
    """
    Run the complete Syniscopy simulation and video generation pipeline.

    If PARAMS['channels'] is set, the simulator
    uses a same-scene spectral path: one latent scene is generated, each
    wavelength is rendered against that scene, detector channels are integrated,
    and noise is applied after integration.

    Without channels, this is the ordinary single-channel path.

    The input dictionary is copied before run-scoped state is resolved. Derived
    particle specs, material metadata, layout extents, and static detector seeds
    remain local to this simulation while signal/reference rendering share the
    same resolved state.
    """
    run_params = deepcopy(params)
    _resolve_public_num_frames(run_params)
    _ensure_run_scope_layout_token(run_params)
    validate_params(run_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    _ensure_run_scope_layout_extent(run_params)
    validate_params(run_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)

    channels = run_params.get("channels", None)
    if channels is not None:
        return _run_multichannel_simulation(run_params, channels, return_frames=return_frames)

    latent_scene = _simulate_latent_scene(run_params)
    return _render_scene_with_params(
        run_params,
        latent_scene,
        save_video_output=True,
        return_frames=return_frames,
    )


def generate_single_frame_views(params: dict) -> dict:
    """
    Generate all relevant single-frame views for the current parameter set.

    Assumptions:
      - params is a full PARAMS-like dictionary.
      - The caller has already configured params for the desired scenario
        (e.g., single frame, single particle) if needed.
      - This function does NOT write any files (no video, no masks).

    Returns:
      A dict containing:
        - "params_resolved": a run-scoped PARAMS copy with resolved particle
          objects and material metadata.
        - "raw_signal_frame": 2D uint16 array of the signal frame.
        - "raw_reference_frame": 2D uint16 array of the reference frame.
        - "ideal_signal_frame": optional pre-noise float signal frame.
        - "ideal_reference_frame": optional pre-noise float reference frame.
        - "detector_difference_frame": optional pre-noise detector-count
          difference image, ``ideal_signal_frame - ideal_reference_frame``.
        - "contrast_frame": 2D floating-point array of the single-frame
          contrast view in the modality's analysis convention. Relative
          reference modalities use ``(S-R)/R``; phase-output modalities such as
          QPI use radians; additive count-domain modalities use ``S-R``. When
          ideal float frames are requested, this uses the pre-noise
          signal/reference pair; otherwise it uses the rendered raw frames.
        - "contrast_frame_units": a short label describing the analysis
          convention of "contrast_frame".
        - "final_frame_8bit": 2D uint8 display-normalized version of
          "contrast_frame" for previewing this single frame. It is not a
          temporal video-median product and should not be interpreted as a
          byte-for-byte frame from the multi-frame video writer.
    """
    # Resolve derived particle/material fields into a run-local copy so returned
    # params match the rendered frame without mutating the caller's dictionary.
    params_local = deepcopy(params)
    _resolve_public_num_frames(params_local)
    _ensure_run_scope_layout_token(params_local)
    validate_params(params_local, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    _ensure_run_scope_layout_extent(params_local)
    validate_params(params_local, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    method = str(params_local.get("background_subtraction_method", "video_median")).strip().lower()
    if method != "reference_frame":
        raise ValueError(
            "generate_single_frame_views requires "
            "background_subtraction_method='reference_frame' because it returns "
            "analysis contrast with explicit physical units. Use run_simulation "
            "or lower-level postprocessing helpers for raw/video-median previews."
        )

    latent_scene = _simulate_latent_scene(params_local)
    particle_instances = _build_particle_instances_for_scene(params_local, latent_scene)

    original_mask_generation_enabled = params_local.get("mask_generation_enabled")
    params_local["mask_generation_enabled"] = False
    try:
        rendered = generate_video_and_masks(
            params_local,
            particle_instances,
        )
    finally:
        if original_mask_generation_enabled is None:
            params_local.pop("mask_generation_enabled", None)
        else:
            params_local["mask_generation_enabled"] = original_mask_generation_enabled
    raw_signal_frames = rendered.signal_frames
    raw_reference_frames = rendered.reference_frames
    ideal_signal_frames = rendered.ideal_signal_frames
    ideal_reference_frames = rendered.ideal_reference_frames

    raw_signal_frame = raw_signal_frames[0] if raw_signal_frames else None
    raw_reference_frame = raw_reference_frames[0] if raw_reference_frames else None
    ideal_signal_frame = ideal_signal_frames[0] if ideal_signal_frames else None
    ideal_reference_frame = ideal_reference_frames[0] if ideal_reference_frames else None

    contrast_signal_frame = ideal_signal_frame if ideal_signal_frame is not None else raw_signal_frame
    contrast_reference_frame = ideal_reference_frame if ideal_reference_frame is not None else raw_reference_frame
    if contrast_signal_frame is not None and contrast_reference_frame is not None:
        detector_difference_frame = (
            np.asarray(contrast_signal_frame, dtype=float)
            - np.asarray(contrast_reference_frame, dtype=float)
        )
    else:
        detector_difference_frame = None
    if contrast_signal_frame is not None and contrast_reference_frame is not None:
        contrast_frame = compute_single_frame_contrast(
            contrast_signal_frame,
            contrast_reference_frame,
            params_local,
        )
    else:
        contrast_frame = None

    if contrast_frame is not None:
        final_8bit_list = normalize_contrast_frames(
            [contrast_frame],
            contrast_frame.shape,
        )
        final_frame_8bit = final_8bit_list[0] if final_8bit_list else None
    else:
        final_frame_8bit = None

    model = get_imaging_model(params_local)
    output_type = getattr(model, "output_type", "intensity")
    if output_type == "phase":
        contrast_frame_units = "radians"
    elif modality_uses_relative_reference_contrast(params_local.get("imaging_model", "bright_field")):
        contrast_frame_units = "relative_reference"
    else:
        contrast_frame_units = "detector_count_difference"

    return {
        "params_resolved": params_local,
        "raw_signal_frame": raw_signal_frame,
        "raw_reference_frame": raw_reference_frame,
        "ideal_signal_frame": ideal_signal_frame,
        "ideal_reference_frame": ideal_reference_frame,
        "detector_difference_frame": detector_difference_frame,
        "contrast_frame": contrast_frame,
        "contrast_frame_units": contrast_frame_units,
        "final_frame_8bit": final_frame_8bit,
    }


def main():
    """
    Script entry point: run the simulation using the global config.PARAMS.

    This path performs one simulation configured by config.PARAMS. The
    programmatic dataset entry point is dataset_generator.generate_dataset.
    """
    run_simulation(PARAMS)


if __name__ == '__main__':
    main()
