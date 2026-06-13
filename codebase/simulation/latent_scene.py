from __future__ import annotations

from config.runtime import (
    AcquisitionProfile,
    FocusPlaneState,
    MotionDynamicsSettings,
    OpticalPsfGridSettings,
)
import logging

import numpy as np

from imaging_models import get_imaging_model
from particle_material_resolution import (
    resolve_component_refractive_index,
    resolve_primary_component_refractive_indices,
    resolve_particle_material_properties,
)
from optics import compute_complex_psf_stack
from optical_scattering import (
    optical_scattering_key_for_component,
    optical_scattering_model_from_key,
    optical_scattering_reference_diameter_from_key,
    optical_scattering_refractive_index_from_key,
    optical_scattering_shape_from_key,
)
from particle_model import build_particle_types_and_instances
from particle_specs import get_particle_specs, normalize_particle_specs, particle_count
from trajectory import (
    resolve_num_frames,
    simulate_orientations_at_times,
    simulate_orientations,
    simulate_trajectories_at_times,
    simulate_trajectories,
)

logger = logging.getLogger(__name__)

_MAX_AUTO_Z_STACK_RANGE_NM = 200000.0
_Z_GRID_ABSOLUTE_MARGIN_STEPS = 5.0
_Z_GRID_RELATIVE_MARGIN_FRACTION = 0.10
_Z_GRID_MIN_HALF_SPAN_STEPS = 10.0

def _build_safe_z_grid_for_type(
    z_min_realized_nm: float,
    z_max_realized_nm: float,
    z_step_nm: float,
    *,
    max_slices: int | None = None,
    min_half_span_steps: float = _Z_GRID_MIN_HALF_SPAN_STEPS,
) -> np.ndarray:
    z_min_realized_nm = float(z_min_realized_nm)
    z_max_realized_nm = float(z_max_realized_nm)
    z_step_nm = float(z_step_nm)

    if z_step_nm <= 0.0:
        raise ValueError("parameters['z_stack_step_nm'] must be positive.")

    if z_max_realized_nm < z_min_realized_nm:
        z_min_realized_nm, z_max_realized_nm = z_max_realized_nm, z_min_realized_nm

    z_center = 0.5 * (z_min_realized_nm + z_max_realized_nm)
    realized_half_span = 0.5 * (z_max_realized_nm - z_min_realized_nm)

    absolute_margin_nm = _Z_GRID_ABSOLUTE_MARGIN_STEPS * z_step_nm
    relative_margin_factor = _Z_GRID_RELATIVE_MARGIN_FRACTION
    min_half_span_nm = float(min_half_span_steps) * z_step_nm

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
    if max_slices is not None and z_values.size > int(max_slices):
        raise ValueError(
            "Resolved optical PSF z-grid would require "
            f"{z_values.size} slices, exceeding max_psf_z_slices={int(max_slices)}. "
            "Increase z_stack_step_nm, reduce the axial trajectory range, or raise "
            "max_psf_z_slices explicitly."
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

    motion = MotionDynamicsSettings.from_params(params)
    drift_min_nm, drift_max_nm = motion.axial_drift_extent_nm(duration_seconds)
    z_min += drift_min_nm
    z_max += drift_max_nm

    vibration_margin_nm = motion.axial_vibration_margin_nm
    if vibration_margin_nm > 0.0:
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


def _simulate_latent_scene_at_times(params: dict, times_s) -> dict:
    """
    Simulate wavelength-independent latent scene state on explicit times.

    Report-level microscope comparisons use this to own one physical Brownian
    realization before microscope-local sampling, detector, and timing overlays
    are applied.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size <= 0:
        raise ValueError("times_s must be a non-empty 1D array.")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain only finite values.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")

    normalize_particle_specs(params, mutate=True)
    trajectories_nm = simulate_trajectories_at_times(params, times)

    num_particles = particle_count(params)
    num_frames = int(times.size)
    orientations = simulate_orientations_at_times(params, num_particles, times)

    return {
        "trajectories_nm": trajectories_nm,
        "orientations": orientations,
        "num_frames": num_frames,
        "latent_times_s": times.astype(float),
    }


def _component_axial_offset_bounds(component, *, orientation_active: bool) -> tuple[float, float]:
    offset = np.asarray(component.offset_nm, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("Particle component offset_nm must be a finite length-3 vector.")
    if bool(orientation_active):
        radius_nm = float(np.linalg.norm(offset)) + float(component.bounding_radius_nm)
        return -radius_nm, radius_nm
    z_offset = float(offset[2])
    half_extent_nm = float(component.axial_half_extent_nm(None))
    return z_offset - half_extent_nm, z_offset + half_extent_nm


def _collect_type_keys_required(
    params: dict,
    *,
    orientations: np.ndarray | None = None,
) -> tuple[dict, set, dict]:
    particle_specs = get_particle_specs(params)
    type_to_component_refs = {}
    type_keys_required = set()
    component_by_type_key = {}
    orientation_active = orientations is not None

    for i, spec in enumerate(particle_specs):
        component_orientation_active = bool(
            orientation_active and not getattr(spec, "is_single_sphere", False)
        )
        for component in spec.components:
            n_component = resolve_component_refractive_index(params, component)
            key = optical_scattering_key_for_component(params, component, n_component)
            type_keys_required.add(key)
            component_by_type_key.setdefault(key, component)
            z_offset_min, z_offset_max = _component_axial_offset_bounds(
                component,
                orientation_active=component_orientation_active,
            )
            type_to_component_refs.setdefault(key, []).append(
                (i, float(z_offset_min), float(z_offset_max))
            )

    return type_to_component_refs, type_keys_required, component_by_type_key


def _build_particle_instances_for_scene(params: dict, latent_scene: dict):
    """
    Build wavelength-dependent particle instances for a fixed latent scene.
    """
    imaging_model = get_imaging_model(params)
    requires_complex_optical_psf = bool(
        getattr(imaging_model, "requires_complex_optical_psf", True)
    )
    if requires_complex_optical_psf:
        resolve_primary_component_refractive_indices(params)
        resolve_particle_material_properties(params, require_optical_refractive_index=True)
        type_to_component_refs, type_keys_required, component_by_type_key = _collect_type_keys_required(
            params,
            orientations=latent_scene.get("orientations"),
        )
    else:
        resolve_particle_material_properties(params, require_optical_refractive_index=False)
        type_to_component_refs, type_keys_required, component_by_type_key = {}, set(), {}

    if not requires_complex_optical_psf:
        _, particle_instances = build_particle_types_and_instances(
            params=params,
            trajectories_nm=latent_scene["trajectories_nm"],
            psf_interpolators_by_type={},
            orientations=latent_scene["orientations"],
            require_optical_psf=False,
        )
        return particle_instances

    logger.info("Pre-computing unique particle complex PSF stacks with trajectory-based Z-ranges...")
    psf_grid = OpticalPsfGridSettings.from_params(params)
    z_step_nm = psf_grid.z_stack_step_nm
    max_psf_z_slices = psf_grid.max_z_slices
    min_half_span_steps = psf_grid.min_half_span_steps

    psf_interpolators_by_type = {}
    default_half_span_nm = psf_grid.default_half_span_nm
    use_shared_psf_z_grid = psf_grid.shared_z_grid_enabled
    trajectories_nm = latent_scene["trajectories_nm"]
    duration_seconds = AcquisitionProfile.from_params(params).duration_seconds
    focus_plane_z_nm = FocusPlaneState.from_params(params).z_nm

    for type_key in sorted(type_keys_required):
        diam_nm_type = optical_scattering_reference_diameter_from_key(type_key)
        n_complex = optical_scattering_refractive_index_from_key(type_key)
        scattering_model = optical_scattering_model_from_key(type_key)
        geometry_shape = optical_scattering_shape_from_key(type_key)
        component_refs = type_to_component_refs.get(type_key, None)

        if use_shared_psf_z_grid:
            z_min_type = -default_half_span_nm
            z_max_type = default_half_span_nm
            if component_refs is not None and len(component_refs) > 0:
                indices_array = np.asarray([ref[0] for ref in component_refs], dtype=int)
                z_offset_bounds = np.asarray(
                    [[ref[1], ref[2]] for ref in component_refs],
                    dtype=float,
                )
                z_positions_type = (
                    trajectories_nm[indices_array, :, 2][:, :, None]
                    - focus_plane_z_nm
                    + z_offset_bounds[:, None, :]
                )
                z_min_rendered, z_max_rendered = _include_axial_render_perturbation(
                    params,
                    z_positions_type,
                    duration_seconds,
                )
                z_values_safe = _build_safe_z_grid_for_type(
                    z_min_realized_nm=z_min_rendered,
                    z_max_realized_nm=z_max_rendered,
                    z_step_nm=z_step_nm,
                    max_slices=max_psf_z_slices,
                    min_half_span_steps=min_half_span_steps,
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
            z_offset_bounds = np.asarray(
                [[ref[1], ref[2]] for ref in component_refs],
                dtype=float,
            )
            z_positions_type = (
                trajectories_nm[indices_array, :, 2][:, :, None]
                - focus_plane_z_nm
                + z_offset_bounds[:, None, :]
            )
            z_min_rendered, z_max_rendered = _include_axial_render_perturbation(
                params,
                z_positions_type,
                duration_seconds,
            )
            z_values_type = _build_safe_z_grid_for_type(
                z_min_realized_nm=z_min_rendered,
                z_max_realized_nm=z_max_rendered,
                z_step_nm=z_step_nm,
                max_slices=max_psf_z_slices,
                min_half_span_steps=min_half_span_steps,
            )
        else:
            z_center = 0.0
            z_values_type = np.arange(
                z_center - default_half_span_nm,
                z_center + default_half_span_nm + z_step_nm,
                z_step_nm,
                dtype=float,
            )
        if max_psf_z_slices is not None and z_values_type.size > max_psf_z_slices:
            raise ValueError(
                "Resolved optical PSF z-grid would require "
                f"{z_values_type.size} slices for particle type {type_key}, "
                f"exceeding max_psf_z_slices={max_psf_z_slices}."
            )

        logger.info(
            "  Optical scattering type (%s/%s, reference diameter = %.1f nm, n = %.4f + %.4fi): "
            "z-range [% .1f, % .1f] nm with %d slices.",
            scattering_model,
            geometry_shape,
            float(diam_nm_type),
            float(n_complex.real),
            float(n_complex.imag),
            float(z_values_type[0]),
            float(z_values_type[-1]),
            int(z_values_type.size),
        )

        psf_interpolators_by_type[type_key] = compute_complex_psf_stack(
            params,
            float(diam_nm_type),
            n_complex,
            z_values_type,
            optical_scattering_model=scattering_model,
            component_geometry=component_by_type_key.get(type_key),
        )

    _, particle_instances = build_particle_types_and_instances(
        params=params,
        trajectories_nm=latent_scene["trajectories_nm"],
        psf_interpolators_by_type=psf_interpolators_by_type,
        orientations=latent_scene["orientations"],
        require_optical_psf=True,
    )
    return particle_instances


__all__ = [
    "_build_particle_instances_for_scene",
    "_build_safe_z_grid_for_type",
    "_component_axial_offset_bounds",
    "_collect_type_keys_required",
    "_include_axial_render_perturbation",
    "_simulate_latent_scene",
    "_simulate_latent_scene_at_times",
]
