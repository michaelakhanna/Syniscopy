"""Public simulation orchestration entry points."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from config import normalize_params
from optical_extensions import expand_broadband_quadrature
from postprocessing import compute_single_frame_contrast, normalize_contrast_frames
from rendering import generate_video_and_masks
from imaging_models import get_imaging_model
from volume_rendering import (
    combine_volume_stack,
    holotomography_phase_projection_stack,
    params_for_focus_plane,
    resolve_volume_z_planes_nm,
)

from .latent_scene import _build_particle_instances_for_scene, _simulate_latent_scene
from .output import (
    _RUNTIME_PARAM_KEYS,
    _ensure_run_scope_layout_extent,
    _ensure_run_scope_layout_token,
    _resolve_public_num_frames,
)
from .scene_render import _render_scene_with_params
from .spectral_channels import _run_multichannel_simulation
from .units import _canonical_contrast_frame_units


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
    run_params = expand_broadband_quadrature(run_params)
    _resolve_public_num_frames(run_params)
    _ensure_run_scope_layout_token(run_params)
    run_params = normalize_params(run_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    _ensure_run_scope_layout_extent(run_params)
    run_params = normalize_params(run_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)

    volume_mode = str(run_params["volumetric_imaging_mode"]).strip().lower()
    if volume_mode != "single_plane":
        if return_frames:
            return generate_volumetric_views(run_params)
        raise ValueError(
            "PARAMS['volumetric_imaging_mode'] is not 'single_plane'. "
            "Use generate_volumetric_views(params) or run_simulation(..., return_frames=True) "
            "for configured volumetric outputs."
        )

    channels = run_params["channels"]
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
          QPI use radian units; additive count-domain modalities use ``S-R``. This
          function forces ideal float rendering so analysis contrast cannot
          silently come from clipped uint16 preview frames.
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
    spectral_model = str(params_local["spectral_integration_model"]).strip().lower()
    if spectral_model != "single_wavelength" or params_local["channels"] is not None:
        raise ValueError(
            "generate_single_frame_views is a single-analysis-frame helper. "
            "Use run_simulation(return_frames=True) for configured spectral/broadband channels."
        )
    params_local["return_ideal_float_frames"] = True
    _resolve_public_num_frames(params_local)
    _ensure_run_scope_layout_token(params_local)
    params_local = normalize_params(params_local, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    _ensure_run_scope_layout_extent(params_local)
    params_local = normalize_params(params_local, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    method = str(params_local["background_subtraction_method"]).strip().lower()
    if method != "reference_frame":
        raise ValueError(
            "generate_single_frame_views requires "
            "background_subtraction_method='reference_frame' because it returns "
            "analysis contrast with explicit physical units. Use run_simulation "
            "or lower-level postprocessing helpers for raw/video-median previews."
        )

    latent_scene = _simulate_latent_scene(params_local)
    particle_instances = _build_particle_instances_for_scene(params_local, latent_scene)

    original_mask_generation_enabled = params_local["mask_generation_enabled"]
    params_local["mask_generation_enabled"] = False
    try:
        rendered = generate_video_and_masks(
            params_local,
            particle_instances,
        )
    finally:
        params_local["mask_generation_enabled"] = original_mask_generation_enabled
    raw_signal_frames = rendered.signal_frames
    raw_reference_frames = rendered.reference_frames
    ideal_signal_frames = rendered.ideal_signal_frames
    ideal_reference_frames = rendered.ideal_reference_frames

    raw_signal_frame = raw_signal_frames[0] if raw_signal_frames else None
    raw_reference_frame = raw_reference_frames[0] if raw_reference_frames else None
    ideal_signal_frame = ideal_signal_frames[0] if ideal_signal_frames else None
    ideal_reference_frame = ideal_reference_frames[0] if ideal_reference_frames else None

    if ideal_signal_frame is None or ideal_reference_frame is None:
        raise RuntimeError(
            "generate_single_frame_views requires ideal float signal/reference frames; "
            "the renderer did not return them despite return_ideal_float_frames=True."
        )
    contrast_signal_frame = ideal_signal_frame
    contrast_reference_frame = ideal_reference_frame
    detector_difference_frame = (
        np.asarray(contrast_signal_frame, dtype=float)
        - np.asarray(contrast_reference_frame, dtype=float)
    )
    contrast_frame = compute_single_frame_contrast(
        contrast_signal_frame,
        contrast_reference_frame,
        params_local,
    )

    final_8bit_list = normalize_contrast_frames(
        [contrast_frame],
        contrast_frame.shape,
    )
    final_frame_8bit = final_8bit_list[0] if final_8bit_list else None

    model = get_imaging_model(params_local)
    render_metadata = dict(getattr(rendered, "render_metadata", {}) or {})
    response_function = dict(render_metadata.get("response_function", {}) or {})
    if not response_function:
        response_function = model.compute_response_function(
            np.asarray(contrast_signal_frame).shape,
            params_local,
        )
    contrast_frame_units = _canonical_contrast_frame_units(
        params_local,
        model,
        params_local["imaging_model"],
        response_function=response_function,
    )

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
        "render_metadata": render_metadata,
    }


def generate_volumetric_views(params: dict) -> dict:
    """Generate optional configured z-stack/sectioned views from one particle scene.

    The ordinary renderer remains single-plane by default. This helper makes
    the z contract explicit: each requested plane is rendered as a single-frame
    view with the focal plane shifted relative to the configured particle state,
    then a configured volume reducer combines or returns the stack.
    """
    params_local = deepcopy(params)
    params_local = expand_broadband_quadrature(params_local)
    params_local = normalize_params(params_local, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    mode = str(params_local["volumetric_imaging_mode"]).strip().lower()
    if mode == "single_plane":
        params_local["background_subtraction_method"] = "reference_frame"
        view = generate_single_frame_views(params_local)
        volume_output_mode = str(params_local["volume_output_mode"]).strip().lower()
        return {
            "mode": "single_plane",
            "z_planes_nm": [0.0],
            "contrast_volume": np.asarray([view["contrast_frame"]], dtype=float),
            "combined_contrast_frame": np.asarray(view["contrast_frame"], dtype=float),
            "plane_views": [view],
            "volume_metadata": {
                "scene_dimensionality": str(params_local["scene_dimensionality"]),
                "volumetric_imaging_mode": "single_plane",
                "volume_output_mode": volume_output_mode,
            },
        }

    z_planes = resolve_volume_z_planes_nm(params_local)
    plane_views = []
    contrast_frames = []
    signal_frames = []
    reference_frames = []
    for z_plane in z_planes:
        plane_params = params_for_focus_plane(params_local, float(z_plane))
        plane_params["background_subtraction_method"] = "reference_frame"
        plane_view = generate_single_frame_views(plane_params)
        plane_views.append(plane_view)
        contrast_frames.append(np.asarray(plane_view["contrast_frame"], dtype=float))
        signal_frames.append(np.asarray(plane_view["ideal_signal_frame"], dtype=float))
        reference_frames.append(np.asarray(plane_view["ideal_reference_frame"], dtype=float))

    contrast_volume = np.asarray(contrast_frames, dtype=float)
    combined, volume_metadata = combine_volume_stack(contrast_volume, z_planes, params_local)
    volume_metadata.update(
        {
            "scene_dimensionality": str(params_local["scene_dimensionality"]),
            "volumetric_plane_count": int(len(z_planes)),
            "volumetric_source": "rerendered_particle_focus_planes",
        }
    )
    output: dict[str, Any] = {
        "mode": mode,
        "z_planes_nm": z_planes.astype(float).tolist(),
        "contrast_volume": contrast_volume,
        "signal_volume": np.asarray(signal_frames, dtype=float),
        "reference_volume": np.asarray(reference_frames, dtype=float),
        "combined_contrast_frame": combined,
        "plane_views": plane_views,
        "volume_metadata": volume_metadata,
    }
    if mode == "holotomography_projection":
        projection_stack, holo_metadata = holotomography_phase_projection_stack(
            contrast_volume,
            params_local,
        )
        output["holotomography_projection_stack"] = projection_stack
        output["volume_metadata"] = {**volume_metadata, **holo_metadata}
    return output
