"""Resolve canonical particle components into optical/electron material values."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from material_optical_catalog import (
    MATERIAL_ELECTRON_DEFAULTS,
    MATERIAL_FLUORESCENCE_DEFAULTS,
    lookup_refractive_index,
    normalize_material_name,
)
from material_serialization import material_properties_to_dict
from material_types import MaterialProperties
from config.runtime import OpticalInstrumentSettings
from particle_specs import ParticleComponentSpec, get_particle_specs
from shared_constants import NONNEGATIVE_MATERIAL_PROPERTY_FIELDS, SOURCE_MATERIAL_PROPERTY_FIELDS
from simulation_runtime_state import runtime_state


def _probe_wavelength_nm(params: dict) -> float:
    return OpticalInstrumentSettings.from_params(params).probe_wavelength_nm


def _coerce_complex(value: Any) -> complex:
    """Accept Python complex values plus JSON-friendly complex encodings."""
    if isinstance(value, dict):
        if "real" in value or "imag" in value:
            return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


def _override_n_complex_visible(override: Any, wavelength_nm: float) -> complex | None:
    """Return an explicit material-property refractive index override if present."""
    if isinstance(override, MaterialProperties):
        return override.n_complex(wavelength_nm)
    if not isinstance(override, dict) or "n_complex_visible" not in override:
        return None
    return _coerce_complex(override["n_complex_visible"])


def _validate_nonnegative_material_fields(values: dict[str, Any]) -> None:
    for key in NONNEGATIVE_MATERIAL_PROPERTY_FIELDS:
        if key not in values or values[key] is None:
            continue
        value = float(values[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"material_properties[{key!r}] must be finite and non-negative; got {values[key]!r}.")


def _validate_positive_material_fields(values: dict[str, Any]) -> None:
    for key in ("atomic_number", "atomic_weight_g_mol"):
        if key not in values or values[key] is None:
            continue
        value = float(values[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"material_properties[{key!r}] must be finite and positive; got {values[key]!r}.")


def _has_explicit_source_material_fields(value: Any) -> bool:
    if isinstance(value, MaterialProperties):
        return True
    if not isinstance(value, dict):
        return False
    return any(key in value and value[key] is not None for key in SOURCE_MATERIAL_PROPERTY_FIELDS)


def resolve_component_refractive_index(
    params: dict,
    component: ParticleComponentSpec,
) -> complex:
    """Resolve optical refractive index for one particle component."""
    wavelength_nm = _probe_wavelength_nm(params)
    if component.refractive_index is not None:
        return _coerce_complex(component.refractive_index)
    material_override_n = _override_n_complex_visible(component.material_properties, wavelength_nm)
    if material_override_n is not None:
        return complex(material_override_n)
    if component.material is None:
        raise ValueError(
            "Particle component refractive index is undefined. Provide either "
            "component.refractive_index, component.material, or "
            "component.material_properties.n_complex_visible."
        )
    return lookup_refractive_index(
        material_name=str(component.material),
        wavelength_nm=wavelength_nm,
        diameter_nm=float(component.diameter_nm),
    )


def _material_properties_from_name(
    material_name: str | None,
    wavelength_nm: float,
    diameter_nm: float | None,
    refractive_index: complex | None = None,
) -> MaterialProperties:
    """Resolve a particle MaterialProperties object from a material label."""
    if material_name is None:
        n_visible = complex(refractive_index) if refractive_index is not None else 1.0 + 0.0j
        return MaterialProperties(name="custom_particle", n_complex_visible=n_visible)

    try:
        canonical = normalize_material_name(str(material_name))
    except ValueError:
        if refractive_index is None:
            raise
        return MaterialProperties(
            name=str(material_name),
            n_complex_visible=complex(refractive_index),
        )
    n_visible = complex(
        refractive_index
        if refractive_index is not None
        else lookup_refractive_index(canonical, wavelength_nm=wavelength_nm, diameter_nm=diameter_nm)
    )
    fluorescence = MATERIAL_FLUORESCENCE_DEFAULTS.get(canonical, {})
    electron = MATERIAL_ELECTRON_DEFAULTS.get(canonical, {})
    return MaterialProperties(
        name=canonical,
        n_complex_visible=n_visible,
        mean_inner_potential_V=float(electron.get("mean_inner_potential_V", 0.0) or 0.0),
        density_g_cm3=float(electron.get("density_g_cm3", 0.0) or 0.0),
        se_yield_coefficient=float(electron.get("se_yield_coefficient", 0.0) or 0.0),
        atomic_number=(
            None if electron.get("atomic_number") is None else float(electron["atomic_number"])
        ),
        atomic_weight_g_mol=(
            None if electron.get("atomic_weight_g_mol") is None else float(electron["atomic_weight_g_mol"])
        ),
        fluorophore_density=float(fluorescence.get("fluorophore_density", 0.0) or 0.0),
        autofluorescence_per_nm=float(fluorescence.get("autofluorescence_per_nm", 0.0) or 0.0),
        excitation_peak_nm=fluorescence.get("excitation_peak_nm"),
        emission_peak_nm=fluorescence.get("emission_peak_nm"),
    )


def _apply_material_override(base: MaterialProperties, override: Any) -> MaterialProperties:
    """Apply one per-particle material-property override dictionary."""
    if override is None:
        return base
    if isinstance(override, MaterialProperties):
        return override
    if not isinstance(override, dict):
        raise TypeError("component material_properties entries must be dictionaries or MaterialProperties objects.")

    allowed = {
        "name",
        "n_complex_visible",
        "mean_inner_potential_V",
        "density_g_cm3",
        "se_yield_coefficient",
        "atomic_number",
        "atomic_weight_g_mol",
        "autofluorescence_per_nm",
        "fluorophore_density",
        "emission_peak_nm",
        "excitation_peak_nm",
        "polarizability_tensor",
    }
    unknown = sorted(set(override) - allowed)
    if unknown:
        raise ValueError(f"Unsupported material-property override key(s): {unknown}.")
    _validate_nonnegative_material_fields(override)
    _validate_positive_material_fields(override)

    n_complex_visible = override.get("n_complex_visible", base.n_complex_visible)
    if isinstance(n_complex_visible, dict):
        n_complex_visible = complex(
            float(n_complex_visible.get("real", 0.0)),
            float(n_complex_visible.get("imag", 0.0)),
        )

    return MaterialProperties(
        name=str(override.get("name", base.name)),
        n_complex_visible=n_complex_visible,
        mean_inner_potential_V=float(override.get("mean_inner_potential_V", base.mean_inner_potential_V)),
        density_g_cm3=float(override.get("density_g_cm3", base.density_g_cm3)),
        se_yield_coefficient=float(override.get("se_yield_coefficient", base.se_yield_coefficient)),
        atomic_number=(
            None if override.get("atomic_number", base.atomic_number) is None
            else float(override.get("atomic_number", base.atomic_number))
        ),
        atomic_weight_g_mol=(
            None if override.get("atomic_weight_g_mol", base.atomic_weight_g_mol) is None
            else float(override.get("atomic_weight_g_mol", base.atomic_weight_g_mol))
        ),
        autofluorescence_per_nm=float(override.get("autofluorescence_per_nm", base.autofluorescence_per_nm)),
        fluorophore_density=float(override.get("fluorophore_density", base.fluorophore_density)),
        emission_peak_nm=(
            None if override.get("emission_peak_nm", base.emission_peak_nm) is None
            else float(override.get("emission_peak_nm", base.emission_peak_nm))
        ),
        excitation_peak_nm=(
            None if override.get("excitation_peak_nm", base.excitation_peak_nm) is None
            else float(override.get("excitation_peak_nm", base.excitation_peak_nm))
        ),
        polarizability_tensor=override.get("polarizability_tensor", base.polarizability_tensor),
    )


def resolve_component_material_properties(
    params: dict,
    component: ParticleComponentSpec,
    *,
    require_optical_refractive_index: bool = True,
) -> MaterialProperties:
    """Resolve full modality material properties for one particle component."""
    if require_optical_refractive_index:
        n_complex = resolve_component_refractive_index(params, component)
    else:
        n_complex = _override_n_complex_visible(
            component.material_properties,
            _probe_wavelength_nm(params),
        )
        if n_complex is None and component.refractive_index is not None:
            n_complex = _coerce_complex(component.refractive_index)
    material_name = None if component.material is None else str(component.material)
    source_requires_explicit_properties = material_name is None
    try:
        base = _material_properties_from_name(
            material_name,
            wavelength_nm=_probe_wavelength_nm(params),
            diameter_nm=float(component.diameter_nm),
            refractive_index=n_complex,
        )
    except ValueError:
        if require_optical_refractive_index or component.material_properties is None:
            raise
        source_requires_explicit_properties = True
        base = MaterialProperties(
            name=str(material_name or "custom_particle"),
            n_complex_visible=complex(n_complex if n_complex is not None else 1.0 + 0.0j),
        )
    resolved = _apply_material_override(base, component.material_properties)
    if (
        not require_optical_refractive_index
        and source_requires_explicit_properties
        and not _has_explicit_source_material_fields(component.material_properties)
    ):
        raise ValueError(
            "Source-map modalities require each custom or refractive-index-only "
            "particle component to define material source properties. Set a "
            "recognized component material or provide material_properties with "
            "at least one of mean_inner_potential_V, density_g_cm3, "
            "se_yield_coefficient, fluorophore_density, or "
            "autofluorescence_per_nm."
        )
    return resolved


def resolve_primary_component_refractive_indices(params: dict) -> np.ndarray:
    """Resolve one primary-component complex refractive index per logical particle."""
    state = runtime_state(params)
    cached = state.resolved_primary_component_refractive_indices
    specs = get_particle_specs(params)
    if (
        isinstance(cached, np.ndarray)
        and state.resolved_primary_component_refractive_indices_fingerprint is not None
        and state.resolved_primary_component_refractive_indices_fingerprint == state.particle_specs_fingerprint
        and cached.size == len(specs)
    ):
        return cached

    resolved = [
        resolve_component_refractive_index(params, spec.primary_component)
        for spec in specs
    ]
    resolved_array = np.asarray(resolved, dtype=np.complex128)
    state.resolved_primary_component_refractive_indices = resolved_array
    state.resolved_primary_component_refractive_indices_fingerprint = state.particle_specs_fingerprint
    return resolved_array


def resolve_particle_material_properties(
    params: dict,
    *,
    require_optical_refractive_index: bool = True,
) -> List[MaterialProperties]:
    """
    Resolve per-particle MaterialProperties for modality-specific physics.

    Component ``material_properties`` is the canonical way to provide
    fluorescence/electron/material fields. Explicit component refractive-index
    overrides still affect the optical n used by the resolved MaterialProperties.
    """
    state = runtime_state(params)
    specs = get_particle_specs(params)
    if (
        state.resolved_particle_material_properties is not None
        and state.resolved_particle_material_properties_fingerprint is not None
        and state.resolved_particle_material_properties_fingerprint == state.particle_specs_fingerprint
        and len(state.resolved_particle_material_properties) == len(specs)
    ):
        return state.resolved_particle_material_properties

    resolved = [
        resolve_component_material_properties(
            params,
            spec.primary_component,
            require_optical_refractive_index=require_optical_refractive_index,
        )
        for spec in specs
    ]

    state.resolved_particle_material_properties = resolved
    state.resolved_particle_material_properties_fingerprint = state.particle_specs_fingerprint
    wavelength_nm = _probe_wavelength_nm(params)
    state.resolved_particle_material_properties_metadata = [
        material_properties_to_dict(material, wavelength_nm=wavelength_nm) for material in resolved
    ]
    return resolved


__all__ = [
    "resolve_component_material_properties",
    "resolve_component_refractive_index",
    "resolve_particle_material_properties",
    "resolve_primary_component_refractive_indices",
]
