"""Geometry-aware optical scattering dispatch and analytic primitive amplitudes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import j1

from config.runtime import (
    OpticalInstrumentSettings,
    OpticalModeSettings,
    OpticalScatteringSettings,
    SamplingGeometry,
)
from particle_specs import ParticleComponentSpec


OPTICAL_SCATTERING_AUTO = "auto"
OPTICAL_SCATTERING_MIE = "mie"
OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY = "analytic_polarizability"
OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS = "born_rayleigh_gans"
OPTICAL_SCATTERING_MODELS = {
    OPTICAL_SCATTERING_AUTO,
    OPTICAL_SCATTERING_MIE,
    OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
    OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
}


@dataclass(frozen=True)
class OpticalScatteringSpec:
    model: str
    geometry_shape: str
    dimensions_key: tuple[Any, ...]
    reference_diameter_nm: float
    refractive_index: complex

    @property
    def cache_key(self) -> tuple[Any, ...]:
        n = complex(self.refractive_index)
        return (
            "optical_scattering",
            self.model,
            self.geometry_shape,
            self.dimensions_key,
            float(self.reference_diameter_nm),
            float(n.real),
            float(n.imag),
        )


def resolve_optical_scattering_model(
    params: dict,
    component: ParticleComponentSpec,
) -> str:
    requested = OpticalScatteringSettings.from_params(params).model
    shape = component.shape_kind
    if requested == OPTICAL_SCATTERING_AUTO:
        return component.default_optical_scattering_model
    if not component.supports_optical_scattering_model(requested):
        valid = component.valid_optical_scattering_models
        if requested == OPTICAL_SCATTERING_MIE and not component.supports_exact_mie:
            raise ValueError(
                "optical_scattering_model='mie' is valid only for sphere components; "
                f"got shape={shape!r}. Use optical_scattering_model='auto' or "
                "'analytic_polarizability' for anisotropic analytic primitives."
            )
        raise ValueError(
            f"optical_scattering_model={requested!r} is not valid for shape={shape!r}; "
            f"valid models are {tuple(valid)!r}."
        )
    return requested


def component_dimensions_key(component: ParticleComponentSpec) -> tuple[Any, ...]:
    return component.geometry_dimensions_key


def optical_scattering_spec_for_component(
    params: dict,
    component: ParticleComponentSpec,
    refractive_index: complex,
) -> OpticalScatteringSpec:
    return OpticalScatteringSpec(
        model=resolve_optical_scattering_model(params, component),
        geometry_shape=component.shape_kind,
        dimensions_key=component_dimensions_key(component),
        reference_diameter_nm=float(component.diameter_nm),
        refractive_index=complex(refractive_index),
    )


def optical_scattering_key_for_component(
    params: dict,
    component: ParticleComponentSpec,
    refractive_index: complex,
) -> tuple[Any, ...]:
    return optical_scattering_spec_for_component(params, component, refractive_index).cache_key


def optical_scattering_model_from_key(key: tuple[Any, ...]) -> str:
    if len(key) < 2 or key[0] != "optical_scattering":
        raise ValueError(f"Invalid optical scattering key: {key!r}.")
    return str(key[1])


def optical_scattering_shape_from_key(key: tuple[Any, ...]) -> str:
    if len(key) < 3 or key[0] != "optical_scattering":
        raise ValueError(f"Invalid optical scattering key: {key!r}.")
    return str(key[2])


def optical_scattering_reference_diameter_from_key(key: tuple[Any, ...]) -> float:
    if len(key) < 5 or key[0] != "optical_scattering":
        raise ValueError(f"Invalid optical scattering key: {key!r}.")
    return float(key[4])


def optical_scattering_refractive_index_from_key(key: tuple[Any, ...]) -> complex:
    if len(key) < 7 or key[0] != "optical_scattering":
        raise ValueError(f"Invalid optical scattering key: {key!r}.")
    return complex(float(key[5]), float(key[6]))


def scattering_metadata_for_key(key: tuple[Any, ...]) -> dict[str, Any]:
    model = optical_scattering_model_from_key(key)
    shape = optical_scattering_shape_from_key(key)
    dimensions_key = tuple(key[3]) if len(key) > 3 and isinstance(key[3], tuple) else ()
    if model == OPTICAL_SCATTERING_MIE:
        return {
            "optical_scattering_model": model,
            "geometry_shape": shape,
            "geometry_dimensions_key": dimensions_key,
            "backend_fidelity_level": "reference_validated",
            "equations_or_model_family": "mie_lorenz_sphere_scattering",
            "implemented_approximation_level": "exact_homogeneous_sphere",
            "native_operating_assumptions": "homogeneous sphere in a homogeneous medium",
            "known_omissions": (),
        }
    if model == OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY:
        return {
            "optical_scattering_model": model,
            "geometry_shape": shape,
            "geometry_dimensions_key": dimensions_key,
            "backend_fidelity_level": "physics_based",
            "equations_or_model_family": "quasi_static_anisotropic_polarizability",
            "implemented_approximation_level": (
                "analytic primitive point scatterer with orientation-dependent "
                "polarizability tensor"
            ),
            "native_operating_assumptions": (
                "small-particle/quasi-static regime; weak retardation across the "
                "particle; no coupled multipole, DDA, T-matrix, or full Born volume scattering"
            ),
            "absolute_scale_calibration_status": "internally_normalized_not_reference_validated",
            "known_omissions": (
                "retardation across extended rods",
                "material dispersion beyond resolved refractive index",
                "near-field substrate coupling",
                "shape-resolved diffraction lobes beyond point-scatterer PSF",
            ),
        }
    if model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS:
        return {
            "optical_scattering_model": model,
            "geometry_shape": shape,
            "geometry_dimensions_key": dimensions_key,
            "backend_fidelity_level": "physics_based",
            "equations_or_model_family": "rayleigh_gans_born_form_factor",
            "implemented_approximation_level": (
                "weak-scatterer primitive Fourier form factor in the objective pupil"
            ),
            "validation_status": "unchecked",
            "absolute_scale_calibration_status": "internally_normalized_not_reference_validated",
            "native_operating_assumptions": (
                "weak refractive-index contrast; small accumulated phase; single scattering; "
                "no internal multiple scattering, DDA, T-matrix, or substrate near-field coupling"
            ),
            "known_omissions": (
                "spherocylinder form factor uses a volume-matched prolate-spheroid approximation",
                "voxelized geometry uses the configured occupancy resolution",
            ),
        }
    raise ValueError(f"Unsupported optical scattering model {model!r}.")


def _sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1.0e-10
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def _sphere_or_ellipsoid_form_factor(
    q_body: np.ndarray,
    semi_axes_nm: tuple[float, float, float],
) -> np.ndarray:
    axes = np.asarray(semi_axes_nm, dtype=float)
    u = np.sqrt(np.sum((q_body * axes[:, None, None]) ** 2, axis=0))
    out = np.ones_like(u, dtype=float)
    mask = np.abs(u) > 1.0e-6
    um = u[mask]
    out[mask] = 3.0 * (np.sin(um) - um * np.cos(um)) / (um ** 3)
    return out


def _cylinder_form_factor(
    q_body: np.ndarray,
    *,
    length_nm: float,
    diameter_nm: float,
) -> np.ndarray:
    half_length = 0.5 * float(length_nm)
    radius = 0.5 * float(diameter_nm)
    axial = _sinc(q_body[0] * half_length)
    q_radial = np.sqrt(q_body[1] * q_body[1] + q_body[2] * q_body[2])
    x = q_radial * radius
    radial = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1.0e-6
    radial[mask] = 2.0 * j1(x[mask]) / x[mask]
    return axial * radial


def born_rayleigh_gans_form_factor(
    *,
    component_geometry: ParticleComponentSpec,
    qx_nm_inv: np.ndarray,
    qy_nm_inv: np.ndarray,
    qz_nm_inv: np.ndarray,
    orientation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized Fourier form factor for analytic primitive Born/RG scattering."""
    shape = component_geometry.shape_kind
    q_world = np.stack(
        [
            np.asarray(qx_nm_inv, dtype=float),
            np.asarray(qy_nm_inv, dtype=float),
            np.asarray(qz_nm_inv, dtype=float),
        ],
        axis=0,
    )
    R = np.eye(3, dtype=float) if orientation_matrix is None else np.asarray(orientation_matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("orientation_matrix must be 3x3 for Born/Rayleigh-Gans scattering.")
    q_body = np.einsum("ji,j...->i...", R, q_world)
    if shape in {"sphere", "ellipsoid"}:
        form = _sphere_or_ellipsoid_form_factor(q_body, component_geometry.semi_axes_nm)
    elif shape == "cylinder":
        form = _cylinder_form_factor(
            q_body,
            length_nm=float(component_geometry.length_nm),
            diameter_nm=float(component_geometry.diameter_nm),
        )
    elif shape == "spherocylinder":
        # Closed-form capsule scattering is not the same as a cylinder. Use an
        # explicit prolate-spheroid approximation rather than pretending the
        # capped rod is an exact cylinder.
        form = _sphere_or_ellipsoid_form_factor(q_body, component_geometry.semi_axes_nm)
    elif shape == "voxel_volume":
        if component_geometry.voxel_geometry is None:
            raise ValueError("voxel_volume Born/Rayleigh-Gans scattering requires voxel_geometry.")
        form = component_geometry.voxel_geometry.normalized_form_factor(q_body)
    else:
        raise ValueError(
            "Born/Rayleigh-Gans form factor is valid only for sphere, ellipsoid, "
            f"cylinder, spherocylinder, and voxel_volume components; got {shape!r}."
        )
    form = np.asarray(form, dtype=np.complex128)
    invalid = ~np.isfinite(form.real) | ~np.isfinite(form.imag)
    if np.any(invalid):
        form = form.copy()
        form[invalid] = 0.0 + 0.0j
    return form


def born_rayleigh_gans_render_multiplier(
    params: dict,
    *,
    component_geometry: ParticleComponentSpec,
    material_properties: Any,
    fallback_refractive_index: complex,
) -> complex:
    instrument = OpticalInstrumentSettings.from_params(params)
    sampling = SamplingGeometry.from_params(params)
    n_particle = _as_complex_index(material_properties, params, fallback_refractive_index)
    eps_particle = complex(n_particle) ** 2
    eps_medium = complex(float(instrument.refractive_index_medium)) ** 2
    delta_eps = eps_particle - eps_medium
    k_medium = (
        2.0
        * np.pi
        * float(instrument.refractive_index_medium)
        / float(instrument.probe_wavelength_nm)
    )
    pixel_area_nm2 = float(sampling.model_canvas_pixel_size_nm) ** 2
    if not np.isfinite(pixel_area_nm2) or pixel_area_nm2 <= 0.0:
        raise ValueError("model canvas pixel area must be positive for Born/Rayleigh-Gans scattering.")
    return complex((k_medium * k_medium) / np.sqrt(pixel_area_nm2)) * delta_eps * float(component_geometry.volume_nm3)


def _as_complex_index(material_properties: Any, params: dict, fallback: complex) -> complex:
    if material_properties is not None and hasattr(material_properties, "n_complex"):
        return complex(
            material_properties.n_complex(
                OpticalInstrumentSettings.from_params(params).probe_wavelength_nm
            )
        )
    return complex(fallback)


def _primitive_ellipsoid_semi_axes_nm(component: ParticleComponentSpec) -> tuple[float, float, float]:
    shape = component.shape_kind
    if shape in {"sphere", "ellipsoid", "cylinder", "spherocylinder"}:
        return component.semi_axes_nm
    raise ValueError(f"Unsupported analytic polarizability shape {shape!r}.")


def _depolarization_factors(semi_axes_nm: tuple[float, float, float]) -> np.ndarray:
    axes = np.asarray(semi_axes_nm, dtype=float)
    if axes.shape != (3,) or np.any(~np.isfinite(axes)) or np.any(axes <= 0.0):
        raise ValueError(f"semi_axes_nm must be finite positive values; got {semi_axes_nm!r}.")
    if np.allclose(axes, axes[0], rtol=1e-10, atol=1e-10):
        return np.full(3, 1.0 / 3.0, dtype=float)

    nodes, weights = np.polynomial.legendre.leggauss(96)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    one_minus_u = np.maximum(1.0 - u, 1.0e-15)
    s = u / one_minus_u
    ds_du = 1.0 / (one_minus_u * one_minus_u)
    a2 = axes * axes
    delta = np.sqrt((s[:, None] + a2[None, :]).prod(axis=1))
    factors = []
    abc = float(np.prod(axes))
    for axis_index in range(3):
        denom = (s + a2[axis_index]) * delta
        factors.append(0.5 * abc * float(np.sum(w * ds_du / denom)))
    out = np.asarray(factors, dtype=float)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Could not compute depolarization factors for axes {semi_axes_nm!r}.")
    return out / total


def analytic_polarizability_tensor_body_nm3(
    *,
    component_geometry: ParticleComponentSpec,
    refractive_index: complex,
    medium_refractive_index: float,
) -> np.ndarray:
    axes = _primitive_ellipsoid_semi_axes_nm(component_geometry)
    depol = _depolarization_factors(axes)
    eps_particle = complex(refractive_index) ** 2
    eps_medium = complex(float(medium_refractive_index)) ** 2
    delta = eps_particle - eps_medium
    volume_nm3 = float(component_geometry.volume_nm3)
    alpha = volume_nm3 * delta / (eps_medium + depol * delta)
    if not np.all(np.isfinite(alpha.real)) or not np.all(np.isfinite(alpha.imag)):
        raise ValueError("Analytic polarizability tensor produced nonfinite values.")
    return np.diag(alpha.astype(np.complex128))


def _coherent_incident_polarization(params: dict) -> np.ndarray:
    optical = OpticalModeSettings.from_params(params)
    model = optical.polarization_model
    if model == "scalar":
        model = "linear_x"
    if model == "unpolarized":
        model = "linear_x"
    theta = np.deg2rad(optical.vectorial_polarization_rotation_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    if model in {"linear_x", "x"}:
        return np.array([c, s, 0.0], dtype=float)
    if model in {"linear_y", "y"}:
        return np.array([-s, c, 0.0], dtype=float)
    raise ValueError(f"Unsupported coherent polarization model {model!r}.")


def _detection_projection_vector(params: dict) -> np.ndarray | None:
    optical = OpticalModeSettings.from_params(params)
    if optical.vectorial_detection_mode == "analyzer_x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if optical.vectorial_detection_mode == "analyzer_y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if optical.vectorial_detection_mode in {"full_vector", "incoherent_sum", "unpolarized"}:
        return None
    return _coherent_incident_polarization(params)


def _field_metadata_accepts_full_vector_multiplier(field_metadata: dict[str, Any] | None) -> bool:
    metadata = dict(field_metadata or {})
    return (
        str(metadata.get("field_representation", "")).strip().lower()
        == "vectorial_coherent_field"
        and str(metadata.get("scalar_compatibility_reduction", "")).strip().lower()
        == "full_vector_field"
    )


def analytic_polarizability_render_multiplier(
    params: dict,
    *,
    component_geometry: ParticleComponentSpec,
    material_properties: Any,
    orientation_matrix: np.ndarray | None,
    fallback_refractive_index: complex,
) -> complex | np.ndarray:
    projected = analytic_polarizability_dipole_vector(
        params,
        component_geometry=component_geometry,
        material_properties=material_properties,
        orientation_matrix=orientation_matrix,
        fallback_refractive_index=fallback_refractive_index,
    )
    detector = _detection_projection_vector(params)
    detection_mode = OpticalModeSettings.from_params(params).vectorial_detection_mode
    if detector is None and detection_mode == "full_vector":
        return projected
    if detector is None:
        return complex(np.linalg.norm(projected))
    return complex(np.dot(detector.astype(np.complex128), projected))


def analytic_polarizability_dipole_vector(
    params: dict,
    *,
    component_geometry: ParticleComponentSpec,
    material_properties: Any,
    orientation_matrix: np.ndarray | None,
    fallback_refractive_index: complex,
) -> np.ndarray:
    """Return the scaled world-frame dipole vector for analytic optical scattering."""

    instrument = OpticalInstrumentSettings.from_params(params)
    sampling = SamplingGeometry.from_params(params)
    n_particle = _as_complex_index(material_properties, params, fallback_refractive_index)
    alpha_body = analytic_polarizability_tensor_body_nm3(
        component_geometry=component_geometry,
        refractive_index=n_particle,
        medium_refractive_index=instrument.refractive_index_medium,
    )
    R = np.eye(3, dtype=float) if orientation_matrix is None else np.asarray(orientation_matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("orientation_matrix must be 3x3 for analytic optical scattering.")
    alpha_world = R @ alpha_body @ R.T
    incident = _coherent_incident_polarization(params)
    dipole = alpha_world @ incident.astype(np.complex128)
    k_medium = (
        2.0
        * np.pi
        * float(instrument.refractive_index_medium)
        / float(instrument.probe_wavelength_nm)
    )
    pixel_area_nm2 = float(sampling.model_canvas_pixel_size_nm) ** 2
    if not np.isfinite(pixel_area_nm2) or pixel_area_nm2 <= 0.0:
        raise ValueError("model canvas pixel area must be positive for analytic optical scattering.")
    scale = (k_medium * k_medium) / np.sqrt(6.0 * np.pi * pixel_area_nm2)
    return complex(scale) * np.asarray(dipole, dtype=np.complex128)


def optical_scattering_render_multiplier(
    params: dict,
    *,
    component_geometry: ParticleComponentSpec,
    material_properties: Any,
    orientation_matrix: np.ndarray | None,
    field_metadata: dict[str, Any] | None,
) -> complex | np.ndarray:
    metadata = dict(field_metadata or {})
    model = str(metadata.get("optical_scattering_model", OPTICAL_SCATTERING_MIE)).strip().lower()
    if model == OPTICAL_SCATTERING_MIE:
        return 1.0 + 0.0j
    if (
        model == OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY
        and str(metadata.get("analytic_polarizability_vectorial_transport", "")).strip().lower()
        == "debye_operator_applied_to_dipole"
    ):
        return 1.0 + 0.0j
    if model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS:
        fallback_n = complex(
            float(metadata.get("particle_refractive_index", {}).get("real", 1.0))
            if isinstance(metadata.get("particle_refractive_index"), dict)
            else 1.0,
            float(metadata.get("particle_refractive_index", {}).get("imag", 0.0))
            if isinstance(metadata.get("particle_refractive_index"), dict)
            else 0.0,
        )
        return born_rayleigh_gans_render_multiplier(
            params,
            component_geometry=component_geometry,
            material_properties=material_properties,
            fallback_refractive_index=fallback_n,
        )
    if model != OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY:
        raise ValueError(f"Unsupported optical_scattering_model in field metadata: {model!r}.")
    fallback_n = complex(
        float(metadata.get("particle_refractive_index", {}).get("real", 1.0))
        if isinstance(metadata.get("particle_refractive_index"), dict)
        else 1.0,
        float(metadata.get("particle_refractive_index", {}).get("imag", 0.0))
        if isinstance(metadata.get("particle_refractive_index"), dict)
        else 0.0,
    )
    multiplier = analytic_polarizability_render_multiplier(
        params,
        component_geometry=component_geometry,
        material_properties=material_properties,
        orientation_matrix=orientation_matrix,
        fallback_refractive_index=fallback_n,
    )
    if isinstance(multiplier, np.ndarray):
        if _field_metadata_accepts_full_vector_multiplier(field_metadata):
            if multiplier.shape != (3,):
                raise ValueError(
                    "analytic full-vector polarizability multiplier must have shape (3,); "
                    f"got {multiplier.shape!r}."
                )
            return multiplier.astype(np.complex128)
        return complex(np.linalg.norm(multiplier))
    return complex(multiplier)


__all__ = [
    "OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY",
    "OPTICAL_SCATTERING_AUTO",
    "OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS",
    "OPTICAL_SCATTERING_MIE",
    "OPTICAL_SCATTERING_MODELS",
    "OpticalScatteringSpec",
    "analytic_polarizability_render_multiplier",
    "analytic_polarizability_dipole_vector",
    "born_rayleigh_gans_form_factor",
    "born_rayleigh_gans_render_multiplier",
    "component_dimensions_key",
    "optical_scattering_key_for_component",
    "optical_scattering_model_from_key",
    "optical_scattering_reference_diameter_from_key",
    "optical_scattering_refractive_index_from_key",
    "optical_scattering_render_multiplier",
    "optical_scattering_shape_from_key",
    "optical_scattering_spec_for_component",
    "resolve_optical_scattering_model",
    "scattering_metadata_for_key",
]
