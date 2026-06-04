from __future__ import annotations

import logging
import os

import numpy as np

from imaging_models import get_imaging_model
from json_utils import json_safe
from rendering import resolve_render_canvas_geometry
from shared_constants import KNOWN_INTERNAL_PARAM_KEYS
from config.runtime import SamplingGeometry, internal_param_value, param_value, resolved_modality
from trajectory import resolve_public_num_frames as _resolve_public_num_frames

logger = logging.getLogger(__name__)

_RUNTIME_PARAM_KEYS = set(KNOWN_INTERNAL_PARAM_KEYS)

def _setup_output_dirs(params: dict) -> None:
    if params["mask_generation_enabled"]:
        base_mask_dir = params["mask_output_directory"]
        logger.info("Checking for mask output directories at %s...", base_mask_dir)
        os.makedirs(base_mask_dir, exist_ok=True)

    output_dir = os.path.dirname(params["output_filename"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _frames_to_channel_first(frames, *, channel_count: int) -> np.ndarray:
    """Convert returned display frames to the public (T, C, H, W) schema."""
    arr = np.asarray(frames)
    channel_count = int(channel_count)
    if arr.size == 0:
        return np.empty((0, channel_count, 0, 0), dtype=np.uint8)
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 4:
        if arr.shape[1] == channel_count:
            return arr
        if arr.shape[-1] == channel_count:
            return np.moveaxis(arr, -1, 1)
        if arr.shape[-1] in {1, 3, 4} and arr.shape[1] not in {1, 3, 4}:
            return np.moveaxis(arr, -1, 1)
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
        "state_axes",
        "derivative_units_by_axis",
        "sigma_units_by_axis",
        "pixel_size_nm",
        "lateral_derivative_mode",
        "lateral_step_nm",
        "lateral_step_note",
        "axial_derivative_mode",
        "orientation_derivative_mode",
        "z_step_nm",
        "rotation_step_rad",
        "noise_variance_units",
        "fisher_units",
        "rank_tolerance",
        "eigenvalues",
        "numerical_fisher_rank",
        "condition_number",
        "singular_axes",
        "measurement_domain",
        "signal_units",
        "derivative_metadata",
    )
    out = {
        key: json_safe(crlb.get(key))
        for key in summary_keys
        if key in crlb
    }
    derivative_metadata = crlb.get("derivative_metadata")
    if isinstance(derivative_metadata, dict):
        for key in ("measurement_domain", "signal_units", "noise_variance_units"):
            if key in derivative_metadata and key not in out:
                out[key] = json_safe(derivative_metadata[key])
    if "fisher_matrix" in crlb:
        out["fisher_shape"] = list(np.asarray(crlb["fisher_matrix"]).shape)
    return out


def _packet_sample_environment_metadata(params: dict) -> dict:
    """Return public scene-environment fields relevant to a matched packet."""
    keys = (
        "sample_environment_enabled",
        "sample_environment",
        "sample_environment_pattern_enabled",
        "sample_environment_pattern",
        "sample_environment_pattern_preset",
        "sample_environment_pattern_dimensions",
        "sample_environment_pattern_pitch_um",
        "sample_environment_pattern_hole_diameter_um",
        "sample_environment_pattern_bar_width_um",
        "sample_environment_pattern_material",
        "sample_environment_pattern_height_nm",
        "sample_environment_pattern_roughness_model",
        "sample_environment_pattern_roughness_source_coupling",
        "sample_environment_pattern_roughness_amplitude",
        "sample_environment_pattern_roughness_phase_std",
        "medium_material",
        "mounting_interface_material",
        "bulk_substrate_material",
        "mounting_interface_thickness_nm",
        "mounting_interface",
        "refractive_index_medium",
        "refractive_index_immersion",
    )
    return {key: json_safe(params[key]) for key in keys if key in params}


def _source_map_provenance(params: dict, render_metadata: dict | None = None) -> dict:
    model = get_imaging_model(params)
    render_metadata = dict(render_metadata or {})
    response = dict(render_metadata.get("response_function", {}) or {})
    if not response:
        sampling = SamplingGeometry.from_params(params)
        response = model.compute_response_function(
            sampling.model_canvas_shape,
            params,
        )
    render_geometry = dict(render_metadata.get("render_geometry", {}) or {})
    source_diagnostics = list(render_metadata.get("source_map_diagnostics", []) or [])
    return {
        "imaging_model": resolved_modality(params),
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
        "source_input_kind": response.get("source_input_kind"),
        "source_map_ndim": response.get("source_map_ndim"),
        "source_axis_order": response.get("source_axis_order"),
        "source_projection_policy": response.get("source_projection_policy"),
        "backend_consumes_volume_source": response.get("backend_consumes_volume_source"),
        "source_slice_thickness_nm": response.get("source_slice_thickness_nm"),
        "source_z_planes_nm": response.get("source_z_planes_nm"),
        "source_z_origin": response.get("source_z_origin"),
        "source_z_uses_particle_world_z": response.get("source_z_uses_particle_world_z"),
        "actual_render_geometry": json_safe(render_geometry),
        "actual_source_map_diagnostics": json_safe(source_diagnostics),
    }


def _ensure_run_scope_layout_token(params: dict) -> None:
    """
    Give unseeded sample-environment layouts a per-run cache token.

    This keeps optical backgrounds and Brownian exclusion geometry consistent
    within one run without reusing the first unseeded layout across later runs
    in the same Python process.
    """
    if not bool(param_value(params, 'sample_environment_pattern_enabled')):
        return
    if not bool(param_value(params, 'sample_environment_enabled')):
        return
    if internal_param_value(params, "_substrate_pattern_layout_cache_token") is not None:
        return
    if param_value(params, 'random_seed') is not None:
        return
    params["_substrate_pattern_layout_cache_token"] = (
        f"run:{int(np.random.SeedSequence().entropy)}"
    )


def _ensure_run_scope_layout_extent(params: dict) -> None:
    """Use one substrate layout extent for trajectory and optical consumers."""
    if not bool(param_value(params, 'sample_environment_pattern_enabled')):
        return
    if not bool(param_value(params, 'sample_environment_enabled')):
        return

    img_size = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = float(params["psf_oversampling_factor"])
    if img_size <= 0 or pixel_size_nm <= 0.0 or os_factor <= 0.0:
        return

    imaging_model = get_imaging_model(params)
    geometry = resolve_render_canvas_geometry(params, particle_instances=None, imaging_model=imaging_model)
    layout_extent_nm = float(geometry["layout_extent_nm"])
    current_extent = internal_param_value(params, "_substrate_pattern_layout_extent_nm")
    if current_extent is not None:
        layout_extent_nm = max(float(layout_extent_nm), float(current_extent))
    params["_substrate_pattern_layout_extent_nm"] = float(layout_extent_nm)


def _multichannel_output_mode(params: dict) -> str:
    """
    Resolve multichannel video output mode.

    Single-channel simulations do not use this. For multichannel/spectral
    simulations, this prevents forced RGB output and lets callers request
    RGB visualization, per-channel grayscale sidecars, both, or returned arrays
    only.
    """
    raw = str(param_value(params, 'multichannel_output_mode')).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    allowed = {"rgb", "channels", "both", "none"}
    if raw not in allowed:
        raise ValueError(
            "PARAMS['multichannel_output_mode'] must be one of "
            "{'rgb', 'channels', 'both', 'none'}; got "
            f"{param_value(params, 'multichannel_output_mode')!r}."
        )

    return raw


def _safe_channel_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))
    safe = safe.strip("._")
    return safe or "channel"


__all__ = [
    "_RUNTIME_PARAM_KEYS",
    "_ensure_run_scope_layout_extent",
    "_ensure_run_scope_layout_token",
    "_frames_to_channel_first",
    "_jsonable_crlb_summary",
    "_multichannel_output_mode",
    "_packet_sample_environment_metadata",
    "_resolve_public_num_frames",
    "_safe_channel_filename",
    "_setup_output_dirs",
    "_simulation_result",
    "_source_map_provenance",
]
