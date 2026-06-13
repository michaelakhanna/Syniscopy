"""Optional volumetric particle-scene helpers.

These functions deliberately sit beside the default single-frame renderer.  They
provide configured z-stack, confocal, light-sheet, and holotomography-style
volume reductions without changing the ordinary single-plane simulation path.
"""

from __future__ import annotations
from configured_parameters import configured_assign
from config import AcquisitionProfile, VolumeRenderingSettings

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from array_representation import ArrayRepresentation, representation_from_volume_basis
from simulation_runtime_state import runtime_state


VOLUME_BASIS_FOCUS_STACK_CONTRAST = "focus_stack_contrast"
VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM = "phase_density_rad_per_nm"
VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST = "refractive_index_contrast"
VOLUME_BASIS_EMITTER_DENSITY_PER_NM = "emitter_density_per_nm"
VOLUME_BASIS_ELECTRON_POTENTIAL_DENSITY_PER_NM = "electron_potential_density_per_nm"
VOLUME_BASIS_DETECTOR_COUNT_DENSITY_PER_NM = "detector_count_density_per_nm"

VOLUME_COORDINATE_FOCUS_PLANE_Z_NM = "focus_plane_z_nm"
VOLUME_COORDINATE_PHYSICAL_SAMPLE_Z_NM = "physical_sample_z_nm"
VOLUME_COORDINATE_FOCUS_RELATIVE_Z_NM = "focus_relative_z_nm"

VOLUME_REDUCTION_Z_STACK = "z_stack"
VOLUME_REDUCTION_FOCUS_WEIGHTED_AVERAGE = "focus_weighted_average"
VOLUME_REDUCTION_INTEGRATED_PROJECTION = "integrated_projection"
VOLUME_REDUCTION_HOLOTOMOGRAPHY_PHASE_PROJECTION = "holotomography_phase_projection"

FOCUS_STACK_VOLUME_BASES = {VOLUME_BASIS_FOCUS_STACK_CONTRAST}
PHYSICAL_LINE_INTEGRAL_BASES = {
    VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM,
    VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST,
    VOLUME_BASIS_EMITTER_DENSITY_PER_NM,
    VOLUME_BASIS_ELECTRON_POTENTIAL_DENSITY_PER_NM,
    VOLUME_BASIS_DETECTOR_COUNT_DENSITY_PER_NM,
}
SUPPORTED_VOLUME_REDUCTIONS = {
    VOLUME_REDUCTION_Z_STACK,
    VOLUME_REDUCTION_FOCUS_WEIGHTED_AVERAGE,
    VOLUME_REDUCTION_INTEGRATED_PROJECTION,
}
LINE_INTEGRAL_VOLUME_OPERATIONS = {
    VOLUME_REDUCTION_INTEGRATED_PROJECTION,
    VOLUME_REDUCTION_HOLOTOMOGRAPHY_PHASE_PROJECTION,
}
_PHASE_DENSITY_VALUE_UNITS = {"radian_per_nm", "rad_per_nm"}
_REFRACTIVE_INDEX_CONTRAST_VALUE_UNITS = {
    "refractive_index_contrast",
    "unitless",
    "index",
}


@dataclass(frozen=True)
class VolumeFieldContract:
    """Resolved unit/basis contract for volume reductions and projections."""

    volume_basis: str
    coordinate_role: str
    value_units: str
    operation: str
    output_units: str
    output_basis: str
    physical_line_integral_performed: bool
    contract_id: str
    representation: ArrayRepresentation | None = None

    def __post_init__(self) -> None:
        if self.representation is None:
            object.__setattr__(
                self,
                "representation",
                representation_from_volume_basis(
                    volume_basis=self.volume_basis,
                    coordinate_role=self.coordinate_role,
                    value_units=self.value_units,
                    output_basis=self.output_basis,
                    output_units=self.output_units,
                    physical_line_integral_performed=self.physical_line_integral_performed,
                ),
            )


def _normalized_contract_label(value: Any, *, field_name: str) -> str:
    label = str(value or "").strip().lower()
    if not label:
        raise ValueError(f"{field_name} must be declared before reducing a volume stack.")
    return label


def _projection_output_units_for_basis(volume_basis: str, value_units: str) -> str:
    units = str(value_units or "plane_units").strip() or "plane_units"
    if volume_basis == VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM:
        return "radian"
    if volume_basis == VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST:
        return "refractive_index_contrast_nm_integral"
    if units.endswith("_per_nm"):
        return units[: -len("_per_nm")]
    if units.endswith("/nm"):
        return units[: -len("/nm")]
    return f"{units}_nm_integral"


def _projection_output_basis_for_basis(volume_basis: str) -> str:
    if volume_basis == VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM:
        return "phase_radian_projection"
    if volume_basis == VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST:
        return "refractive_index_contrast_line_integral"
    if volume_basis == VOLUME_BASIS_EMITTER_DENSITY_PER_NM:
        return "emitter_column_density"
    if volume_basis == VOLUME_BASIS_ELECTRON_POTENTIAL_DENSITY_PER_NM:
        return "electron_potential_column_density"
    if volume_basis == VOLUME_BASIS_DETECTOR_COUNT_DENSITY_PER_NM:
        return "detector_count_column_density"
    return f"{volume_basis}_line_integral"


def _validate_physical_line_integral_units(volume_basis: str, value_units: str) -> tuple[str, str]:
    if volume_basis == VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM:
        if value_units not in _PHASE_DENSITY_VALUE_UNITS:
            raise ValueError(
                "Volume basis 'phase_density_rad_per_nm' represents d(phase)/dz and "
                "therefore requires input units 'radian_per_nm' or 'rad_per_nm'. "
                "A phase frame in 'radian' is already z-integrated; keep it as a "
                "z_stack/focus_weighted_average or convert it to a per-nm density "
                "before requesting a physical line integral."
            )
        return "radian", "phase_radian_projection"
    if volume_basis == VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST:
        if value_units not in _REFRACTIVE_INDEX_CONTRAST_VALUE_UNITS:
            raise ValueError(
                "Volume basis 'refractive_index_contrast' requires unitless RI-contrast "
                "input units ('refractive_index_contrast', 'unitless', or 'index')."
            )
        return "refractive_index_contrast_nm_integral", "refractive_index_contrast_line_integral"
    return _projection_output_units_for_basis(volume_basis, value_units), _projection_output_basis_for_basis(volume_basis)


def _resolve_volume_field_contract(
    *,
    volume_basis: str,
    coordinate_role: str,
    value_units: str,
    operation: str,
) -> VolumeFieldContract:
    basis = _normalized_contract_label(volume_basis, field_name="volume_basis")
    coordinate = _normalized_contract_label(coordinate_role, field_name="coordinate_role")
    units = str(value_units or "plane_units").strip().lower() or "plane_units"
    op = _normalized_contract_label(operation, field_name="volume_operation")

    if op in LINE_INTEGRAL_VOLUME_OPERATIONS:
        # Physical z integrals are a shared volume-field contract, not a local
        # holotomography policy. This prevents phase frames in radians, focus
        # stacks, or detector-display values from being multiplied by nanometres
        # and later mistaken for physical phase projections.
        if basis not in PHYSICAL_LINE_INTEGRAL_BASES:
            raise ValueError(
                f"{op} requires a physical per-z volume basis such as "
                "'phase_density_rad_per_nm'. The supplied basis "
                f"{basis!r} is not physically integrable over z. Use "
                "volume_output_mode='focus_weighted_average' for rerendered "
                "focus-stack contrast, or volume_output_mode='z_stack' to retain "
                "all planes."
            )
        if basis in FOCUS_STACK_VOLUME_BASES or coordinate != VOLUME_COORDINATE_PHYSICAL_SAMPLE_Z_NM:
            raise ValueError(
                f"{op} requires physical_sample_z_nm coordinates and a per-z "
                f"density basis; got coordinate_role={coordinate!r}, "
                f"volume_basis={basis!r}."
            )
        output_units, output_basis = _validate_physical_line_integral_units(basis, units)
        return VolumeFieldContract(
            volume_basis=basis,
            coordinate_role=coordinate,
            value_units=units,
            operation=op,
            output_units=output_units,
            output_basis=output_basis,
            physical_line_integral_performed=True,
            contract_id="physical_per_z_density_line_integral_v1",
        )

    if op == VOLUME_REDUCTION_FOCUS_WEIGHTED_AVERAGE:
        return VolumeFieldContract(
            volume_basis=basis,
            coordinate_role=coordinate,
            value_units=units,
            operation=op,
            output_units=units,
            output_basis=basis,
            physical_line_integral_performed=False,
            contract_id="focus_weighted_observable_average_v1",
        )
    if op == VOLUME_REDUCTION_Z_STACK:
        return VolumeFieldContract(
            volume_basis=basis,
            coordinate_role=coordinate,
            value_units=units,
            operation=op,
            output_units=units,
            output_basis=basis,
            physical_line_integral_performed=False,
            contract_id="volume_stack_no_reduction_v1",
        )
    raise ValueError(f"Unsupported volume operation {operation!r}.")


def resolve_volume_z_planes_nm(params: dict) -> np.ndarray:
    settings = VolumeRenderingSettings.from_params(params)
    explicit = settings.z_planes_nm
    if explicit is not None:
        planes = np.asarray(explicit, dtype=float).reshape(-1)
    else:
        if settings.z_range_nm == 0.0 or settings.z_count == 1:
            planes = np.asarray([0.0], dtype=float)
        else:
            step_limited_count = int(np.floor(settings.z_range_nm / settings.z_step_nm + 1.0e-12)) + 1
            plane_count = max(settings.z_count, step_limited_count)
            planes = np.linspace(-0.5 * settings.z_range_nm, 0.5 * settings.z_range_nm, plane_count, dtype=float)
    if planes.size == 0 or not np.all(np.isfinite(planes)):
        raise ValueError("Volumetric z planes must be a non-empty finite sequence.")
    return planes


def _validated_response_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Volume weights must be finite and non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("Volume weights must have positive sum.")
    return arr


def _z_quadrature_weights_nm(z_planes_nm: np.ndarray, params: dict) -> np.ndarray:
    z = np.asarray(z_planes_nm, dtype=float).reshape(-1)
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("Volumetric z planes must be a non-empty finite sequence.")
    if z.size == 1:
        return np.asarray([_z_spacing_nm(z, params)], dtype=float)

    order = np.argsort(z)
    sorted_z = z[order]
    widths = np.empty_like(sorted_z, dtype=float)
    widths[0] = 0.5 * (sorted_z[1] - sorted_z[0])
    widths[-1] = 0.5 * (sorted_z[-1] - sorted_z[-2])
    widths[1:-1] = 0.5 * (sorted_z[2:] - sorted_z[:-2])
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError("Volumetric z quadrature weights must be finite and non-negative.")

    out = np.empty_like(widths, dtype=float)
    out[order] = widths
    return out


def volume_plane_weights(params: dict, z_planes_nm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    settings = VolumeRenderingSettings.from_params(params)
    mode = settings.imaging_mode
    z = np.asarray(z_planes_nm, dtype=float).reshape(-1)
    if mode in {"single_plane", "z_stack", "holotomography_projection"}:
        weights = np.ones_like(z, dtype=float)
        model = "uniform_projection"
    elif mode == "confocal":
        weights = np.exp(-0.5 * (z / settings.confocal_pinhole_sigma_nm) ** 2)
        model = "gaussian_confocal_axial_detection"
    elif mode == "light_sheet":
        weights = np.exp(-0.5 * ((z - settings.light_sheet_center_z_nm) / settings.light_sheet_sigma_nm) ** 2)
        model = "gaussian_light_sheet_excitation"
    else:
        raise ValueError(f"Unsupported volumetric_imaging_mode {mode!r}.")
    response_weights = _validated_response_weights(weights)
    metadata = {
        "volumetric_imaging_mode": mode,
        "volume_weight_model": model,
        "z_planes_nm": z.astype(float).tolist(),
        "z_plane_response_weights": response_weights.astype(float).tolist(),
        "z_plane_spacing_nm": (
            float(np.median(np.diff(np.sort(z)))) if z.size >= 2 else None
        ),
        "volume_output_mode": settings.output_mode,
    }
    return response_weights, metadata


def combine_volume_stack(
    stack: np.ndarray,
    z_planes_nm: np.ndarray,
    params: dict,
    *,
    volume_basis: str,
    coordinate_role: str,
    value_units: str = "plane_units",
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(stack, dtype=float)
    if arr.ndim < 3:
        raise ValueError("Volume stack must have shape (Z, ...).")
    if arr.shape[0] != len(z_planes_nm):
        raise ValueError("Volume stack first axis must match z plane count.")
    basis = _normalized_contract_label(volume_basis, field_name="volume_basis")
    coordinate = _normalized_contract_label(coordinate_role, field_name="coordinate_role")
    units = str(value_units or "plane_units").strip() or "plane_units"
    weights, metadata = volume_plane_weights(params, np.asarray(z_planes_nm, dtype=float))
    output_mode = VolumeRenderingSettings.from_params(params).output_mode
    if output_mode not in SUPPORTED_VOLUME_REDUCTIONS:
        allowed = ", ".join(sorted(SUPPORTED_VOLUME_REDUCTIONS))
        raise ValueError(f"volume_output_mode must be one of {allowed}; got {output_mode!r}.")
    contract = _resolve_volume_field_contract(
        volume_basis=basis,
        coordinate_role=coordinate,
        value_units=units,
        operation=output_mode,
    )
    metadata.update(
        {
            "volume_basis": contract.volume_basis,
            "volume_coordinate_role": contract.coordinate_role,
            "projection_input_units": contract.value_units,
            "projection_output_units": contract.output_units,
            "projection_output_basis": contract.output_basis,
            "volume_field_contract_id": contract.contract_id,
            **contract.representation.metadata(prefix="volume_array"),
        }
    )
    if output_mode == VOLUME_REDUCTION_Z_STACK:
        return arr.copy(), {**metadata, "volume_combination": VOLUME_REDUCTION_Z_STACK}

    quadrature_weights_nm = _z_quadrature_weights_nm(np.asarray(z_planes_nm, dtype=float), params)
    raw_weights = weights * quadrature_weights_nm
    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0) or float(np.sum(raw_weights)) <= 0.0:
        raise ValueError("Volume reduction weights must be finite, non-negative, and have positive sum.")

    if output_mode == VOLUME_REDUCTION_FOCUS_WEIGHTED_AVERAGE:
        # Focus-stack contrast frames are already rendered analysis observables.
        # They may be averaged over optical focus settings, but multiplying them
        # by nanometres as a physical line integral would create contrast*nm or
        # radian*nm units and corrupt downstream scientific amplitudes.
        average_weights = raw_weights / float(np.sum(raw_weights))
        shape = (average_weights.size,) + (1,) * (arr.ndim - 1)
        combined = np.sum(arr * average_weights.reshape(shape), axis=0)
        metadata.update(
            {
                "z_plane_quadrature_weights_nm": quadrature_weights_nm.astype(float).tolist(),
                "z_plane_focus_average_weights": average_weights.astype(float).tolist(),
                "volume_combination": VOLUME_REDUCTION_FOCUS_WEIGHTED_AVERAGE,
                "projection_output_units": contract.output_units,
                "projection_output_basis": contract.output_basis,
                "physical_line_integral_performed": contract.physical_line_integral_performed,
            }
        )
        return combined, metadata

    integration_weights_nm = raw_weights
    shape = (integration_weights_nm.size,) + (1,) * (arr.ndim - 1)
    combined = np.sum(arr * integration_weights_nm.reshape(shape), axis=0)
    metadata.update(
        {
            "z_plane_quadrature_weights_nm": quadrature_weights_nm.astype(float).tolist(),
            "z_plane_integration_weights_nm": integration_weights_nm.astype(float).tolist(),
            "volume_combination": "weighted_line_integral_projection",
            "projection_output_units": contract.output_units,
            "projection_output_basis": contract.output_basis,
            "physical_line_integral_performed": contract.physical_line_integral_performed,
        }
    )
    return combined, metadata


def _z_spacing_nm(z_planes_nm: np.ndarray, params: dict) -> float:
    z = np.asarray(z_planes_nm, dtype=float).reshape(-1)
    if z.size >= 2:
        diffs = np.diff(np.sort(z))
        finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if finite.size:
            return float(np.median(finite))
    return VolumeRenderingSettings.from_params(params).z_step_nm


def params_for_focus_plane(params: dict, z_plane_nm: float) -> dict:
    """Return single-plane params for an explicit focus-plane render.

    Particle z coordinates remain physical sample-world coordinates.  The
    optical defocus plane is carried separately in runtime state so
    source-map physics such as TIRF evanescent height does not accidentally
    consume focus-relative z.
    """
    out = deepcopy(params)
    acquisition = AcquisitionProfile.from_params(out)
    configured_assign(out, "volumetric_imaging_mode", "single_plane")
    configured_assign(out, "scene_dimensionality", "single_plane_particle_scene")
    runtime_state(out).focus_plane_z_nm = float(z_plane_nm)
    configured_assign(out, "num_frames", 1)
    configured_assign(
        out,
        "duration_seconds",
        max(
            acquisition.duration_seconds,
            acquisition.frame_interval_s,
        ),
    )
    return out


def holotomography_phase_projection_stack(
    volume: np.ndarray,
    params: dict,
    *,
    z_planes_nm: np.ndarray | None = None,
    input_units: str = "radian_per_nm",
    input_basis: str = "focus_stack_contrast",
    coordinate_role: str = VOLUME_COORDINATE_PHYSICAL_SAMPLE_Z_NM,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """
    Return phase projection stack and optional unfiltered backprojection volume.

    The projection model expects a physical phase-volume basis (phase-density or
    refractive-index contrast), not a rendered focus-stack contrast stack.
    """
    arr = np.asarray(volume, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Holotomography projection stack expects a 3D volume.")
    basis = str(input_basis).strip().lower()
    if not basis or basis == "focus_stack_contrast":
        raise ValueError(
            "Holotomography projection requires a physical phase-volume basis "
            "('phase_density_rad_per_nm' or 'refractive_index_contrast'), not a "
            "focus-stack contrast stack. Set input_basis explicitly from a physical "
            "volume source."
        )
    units = str(input_units).strip().lower()
    if basis not in {VOLUME_BASIS_PHASE_DENSITY_RAD_PER_NM, VOLUME_BASIS_REFRACTIVE_INDEX_CONTRAST}:
        raise ValueError(
            "Holotomography projection supports input_basis='phase_density_rad_per_nm' or "
            "'refractive_index_contrast'. Got "
            f"{input_basis!r}."
        )
    contract = _resolve_volume_field_contract(
        volume_basis=basis,
        coordinate_role=coordinate_role,
        value_units=units,
        operation=VOLUME_REDUCTION_HOLOTOMOGRAPHY_PHASE_PROJECTION,
    )
    settings = VolumeRenderingSettings.from_params(params)
    angles = np.asarray(settings.holotomography_projection_angles_deg, dtype=float).reshape(-1)
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("holotomography_projection_angles_deg must be finite and non-empty.")
    if z_planes_nm is None:
        z_planes = resolve_volume_z_planes_nm(params)
    else:
        z_planes = np.asarray(z_planes_nm, dtype=float).reshape(-1)
    dz_nm = _z_spacing_nm(z_planes, params)
    quadrature_weights_nm = _z_quadrature_weights_nm(z_planes, params)
    weight_shape = (quadrature_weights_nm.size,) + (1,) * (arr.ndim - 1)
    projections = []
    for angle in angles:
        rotated = _rotate_yx(arr, float(angle))
        projections.append(
            np.sum(rotated * quadrature_weights_nm.reshape(weight_shape), axis=0)
        )
    stack = np.asarray(projections, dtype=float)
    output_mode = settings.holotomography_output_mode
    reconstruction = None
    reconstruction_status = None
    if output_mode == "reconstruction_volume":
        reconstruction = _unfiltered_backprojection_volume(stack, angles, arr.shape)
        reconstruction_status = "unfiltered_backprojection_volume_returned"
    metadata = {
        "holotomography_model": "physical_line_integral_projection_stack_interpolated_rotation",
        "holotomography_input_basis": contract.volume_basis,
        "holotomography_input_units": contract.value_units,
        "holotomography_projection_units": contract.output_units,
        "holotomography_projection_angles_deg": angles.astype(float).tolist(),
        "holotomography_output_mode": output_mode,
        "projection_axis": "z_after_yx_rotation",
        "projection_input_units": contract.value_units,
        "projection_output_units": contract.output_units,
        "projection_output_basis": contract.output_basis,
        "volume_basis": contract.volume_basis,
        "volume_coordinate_role": contract.coordinate_role,
        "volume_field_contract_id": contract.contract_id,
        **contract.representation.metadata(prefix="volume_array"),
        "physical_line_integral_performed": contract.physical_line_integral_performed,
        "projection_z_spacing_nm": dz_nm,
        "projection_z_quadrature_weights_nm": quadrature_weights_nm.astype(float).tolist(),
        "projection_integration_rule": "per_plane_quadrature_weights",
        "rotation_interpolation_order": 1,
    }
    if reconstruction_status is not None:
        metadata["reconstruction_volume_status"] = reconstruction_status
        metadata["reconstruction_model"] = "unfiltered_backprojection_diagnostic"
    return stack, metadata, reconstruction


def _rotate_yx(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each z plane in y/x by the requested angle without changing shape."""
    try:
        from scipy.ndimage import rotate

        return rotate(
            np.asarray(arr, dtype=float),
            float(angle_deg),
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    except ImportError:
        nearest_quadrant = round(float(angle_deg) / 90.0)
        if abs(float(angle_deg) - 90.0 * nearest_quadrant) > 1.0e-9:
            raise RuntimeError(
                "scipy is required for non-quadrant holotomography projection angles."
            )
        return np.rot90(np.asarray(arr, dtype=float), k=int(nearest_quadrant) % 4, axes=(1, 2))


def _unfiltered_backprojection_volume(
    projection_stack: np.ndarray,
    angles_deg: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Simple diagnostic backprojection volume for reconstruction-output mode."""
    projections = np.asarray(projection_stack, dtype=float)
    if projections.ndim != 3 or projections.shape[0] != len(angles_deg):
        raise ValueError("Projection stack must have shape (angle, y, x).")
    z_count = int(target_shape[0])
    recon = np.zeros(tuple(int(v) for v in target_shape), dtype=float)
    for projection, angle in zip(projections, np.asarray(angles_deg, dtype=float)):
        slab = np.repeat(np.asarray(projection, dtype=float)[None, :, :] / max(z_count, 1), z_count, axis=0)
        recon += _rotate_yx(slab, -float(angle))
    if projections.shape[0] > 0:
        recon /= float(projections.shape[0])
    return recon
