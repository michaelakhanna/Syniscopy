from __future__ import annotations

from config import param_value
import logging

import numpy as np

from imaging_models import get_imaging_model
from materials import (
    resolve_component_refractive_index,
    resolve_primary_component_refractive_indices,
    resolve_particle_material_properties,
)
from optics import compute_complex_psf_stack
from particle_model import build_particle_types_and_instances
from particle_specs import get_particle_specs, normalize_particle_specs, particle_count
from trajectory import (
    resolve_num_frames,
    simulate_orientations,
    simulate_trajectories,
)

logger = logging.getLogger(__name__)

_MAX_AUTO_Z_STACK_RANGE_NM = 200000.0
_Z_GRID_ABSOLUTE_MARGIN_STEPS = 5.0
_Z_GRID_RELATIVE_MARGIN_FRACTION = 0.10
_Z_GRID_MIN_HALF_SPAN_STEPS = 100.0

def _build_safe_z_grid_for_type(
    z_min_realized_nm: float,
    z_max_realized_nm: float,
    z_step_nm: float,
    *,
    max_slices: int | None = None,
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

    drift_velocity = np.asarray(
        param_value(params, 'drift_velocity_nm_per_s'),
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

    if bool(param_value(params, 'vibration_include_axial')):
        vibration_std_nm = float(param_value(params, 'vibration_jitter_std_nm'))
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
    imaging_model = get_imaging_model(params)
    requires_complex_optical_psf = bool(
        getattr(imaging_model, "requires_complex_optical_psf", True)
    )
    if requires_complex_optical_psf:
        resolve_primary_component_refractive_indices(params)
        resolve_particle_material_properties(params, require_optical_refractive_index=True)
        type_to_component_refs, type_keys_required = _collect_type_keys_required(params)
    else:
        resolve_particle_material_properties(params, require_optical_refractive_index=False)
        type_to_component_refs, type_keys_required = {}, set()

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
    z_step_nm = float(params["z_stack_step_nm"])
    if z_step_nm <= 0.0:
        raise ValueError("PARAMS['z_stack_step_nm'] must be positive.")
    max_psf_z_slices_raw = param_value(params, "max_psf_z_slices")
    max_psf_z_slices = None if max_psf_z_slices_raw is None else int(max_psf_z_slices_raw)

    psf_interpolators_by_type = {}
    default_z_range_nm = float(param_value(params, 'z_stack_range_nm'))
    default_half_span_nm = 0.5 * default_z_range_nm
    use_shared_psf_z_grid = bool(param_value(params, 'shared_psf_z_grid_enabled'))
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
                    max_slices=max_psf_z_slices,
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
                max_slices=max_psf_z_slices,
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
        require_optical_psf=True,
    )
    return particle_instances


__all__ = [
    "_build_particle_instances_for_scene",
    "_build_safe_z_grid_for_type",
    "_collect_type_keys_required",
    "_include_axial_render_perturbation",
    "_simulate_latent_scene",
]
