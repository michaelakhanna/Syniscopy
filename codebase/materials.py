"""
Material library and refractive index resolution utilities.

This module provides:
    - A library of common metallic/plasmonic and dielectric materials,
      each associated with a (possibly wavelength-dependent) complex
      refractive index model.
    - A lookup function that maps a material name (and optionally wavelength
      and particle size) to a complex refractive index suitable for use in
      Mie scattering.
    - Helpers that resolve component and particle refractive indices from the
      canonical PARAMS["particles"] object list.

Semantics:
    - Each particle component can specify `material`.
    - Each component can also specify `refractive_index`; when provided, that
      value overrides material-based lookup for that component.

For metals (e.g., gold, silver), this module uses simple tabulated,
wavelength-dependent optical constants (n + i k) with linear interpolation
in wavelength. For common dielectrics and biological-like materials,
it uses wavelength-independent constants, which is a good approximation
for narrow-band visible illumination.

The lookup interface accepts wavelength and diameter so the same call site can
serve constant-index, tabulated, and size-dependent material models.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from particle_specs import ParticleComponentSpec, get_particle_specs
from substrate import MaterialProperties


def _coerce_complex(value: Any) -> complex:
    """Accept Python complex values plus JSON-friendly complex encodings."""
    if isinstance(value, dict):
        if "real" in value or "imag" in value:
            return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


# --- Material property tables -------------------------------------------------
# Canonical material names mapped to approximate *constant* complex
# refractive indices. These are used for dielectric / weakly dispersive
# materials where treating n as wavelength-independent over the visible is
# acceptable for narrow-band visible simulations.
#
# Values are dimensionless refractive indices n + i k.
_MATERIAL_REFRACTIVE_INDEX: Dict[str, complex] = {
    # Common dielectrics / lab-relevant materials (mostly lossless in this model)
    "air": 1.00 + 0.0j,
    "carbon": 2.42 + 0.0j,
    "pet": 1.57 + 0.0j,         # Polyethylene terephthalate
    "polyethylene": 1.51 + 0.0j,
    "polypropylene": 1.49 + 0.0j,
    "polystyrene": 1.59 + 0.0j,
    "fluorescent_polystyrene": 1.59 + 0.0j,
    "silica": 1.46 + 0.0j,      # SiO2
    "water": 1.33 + 0.0j,
    "protein": 1.45 + 0.0j,     # Representative protein-rich material
    "lipid": 1.47 + 0.0j,       # Representative lipid-rich material
    "glass": 1.52 + 0.0j,       # Generic microscope glass
}

_MATERIAL_FLUORESCENCE_DEFAULTS: Dict[str, Dict[str, float | None]] = {
    "fluorescent_polystyrene": {
        "fluorophore_density": 1.0,
        "excitation_peak_nm": 488.0,
        "emission_peak_nm": 520.0,
        "autofluorescence_per_nm": 0.0,
    },
    "protein": {
        "fluorophore_density": 0.0,
        "excitation_peak_nm": 280.0,
        "emission_peak_nm": 340.0,
        "autofluorescence_per_nm": 0.0,
    },
}

# Nominal electron-material defaults used by the simplified TEM/SEM paths.
# These values make recognized material labels populate the electron-facing
# fields already present in MaterialProperties. Calibrated studies should
# override these fields explicitly through component material_properties.
_MATERIAL_ELECTRON_DEFAULTS: Dict[str, Dict[str, float]] = {
    "gold": {
        "mean_inner_potential_V": 25.0,
        "density_g_cm3": 19.3,
        "se_yield_coefficient": 0.18,
    },
    "silver": {
        "mean_inner_potential_V": 22.0,
        "density_g_cm3": 10.5,
        "se_yield_coefficient": 0.16,
    },
    "pet": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.38,
        "se_yield_coefficient": 0.05,
    },
    "polyethylene": {
        "mean_inner_potential_V": 7.5,
        "density_g_cm3": 0.94,
        "se_yield_coefficient": 0.05,
    },
    "polypropylene": {
        "mean_inner_potential_V": 7.5,
        "density_g_cm3": 0.90,
        "se_yield_coefficient": 0.05,
    },
    "polystyrene": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.05,
        "se_yield_coefficient": 0.05,
    },
    "fluorescent_polystyrene": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.05,
        "se_yield_coefficient": 0.05,
    },
    "silica": {
        "mean_inner_potential_V": 10.1,
        "density_g_cm3": 2.20,
        "se_yield_coefficient": 0.10,
    },
    "water": {
        "mean_inner_potential_V": 4.0,
        "density_g_cm3": 1.00,
        "se_yield_coefficient": 0.02,
    },
    "protein": {
        "mean_inner_potential_V": 6.0,
        "density_g_cm3": 1.35,
        "se_yield_coefficient": 0.04,
    },
    "lipid": {
        "mean_inner_potential_V": 4.5,
        "density_g_cm3": 0.92,
        "se_yield_coefficient": 0.04,
    },
    "glass": {
        "mean_inner_potential_V": 9.5,
        "density_g_cm3": 2.50,
        "se_yield_coefficient": 0.10,
    },
    "air": {
        "mean_inner_potential_V": 0.0,
        "density_g_cm3": 0.0,
        "se_yield_coefficient": 0.0,
    },
    "carbon": {
        "mean_inner_potential_V": 8.7,
        "density_g_cm3": 2.0,
        "se_yield_coefficient": 0.08,
    },
}

# Recognized user-facing material labels. Each label maps to a canonical key
# in the internal material tables.
_MATERIAL_NAME_VARIANTS: Dict[str, List[str]] = {
    "gold": [
        "gold",
        "au",
        "gold nanoparticle",
        "nanogold",
    ],
    "silver": [
        "silver",
        "ag",
        "silver nanoparticle",
        "nanosilver",
    ],
    "pet": [
        "pet",
        "polyethylene terephthalate",
        "pet plastic",
    ],
    "polyethylene": [
        "polyethylene",
        "pe",
    ],
    "polypropylene": [
        "polypropylene",
        "pp",
    ],
    "polystyrene": [
        "polystyrene",
        "ps",
    ],
    "fluorescent_polystyrene": [
        "fluorescent_polystyrene",
        "fluorescent polystyrene",
        "fluorescent_ps",
        "fluorescent ps",
    ],
    "air": [
        "air",
    ],
    "carbon": [
        "carbon",
        "amorphous carbon",
        "holey carbon",
    ],
    "silica": [
        "silica",
        "sio2",
        "silicon dioxide",
    ],
    "water": [
        "water",
        "h2o",
    ],
    "protein": [
        "protein",
        "proteins",
    ],
    "lipid": [
        "lipid",
        "lipids",
    ],
    "glass": [
        "glass",
        "bk7",
        "borosilicate glass",
    ],
}

# Build a mapping from lowercase material label to canonical material key.
_MATERIAL_NAME_MAP: Dict[str, str] = {}
for canonical, variants in _MATERIAL_NAME_VARIANTS.items():
    for variant in variants:
        _MATERIAL_NAME_MAP[variant.lower()] = canonical
    # Also allow the canonical name itself.
    _MATERIAL_NAME_MAP[canonical.lower()] = canonical


# --- Wavelength-dependent data for plasmonic metals ---------------------------
# The following optical constants (n and k) are approximate values in the
# visible range for bulk materials (e.g., Johnson & Christy-type data).
# They are sufficient for coherent single-particle simulations.

# Gold (Au): wavelengths in nm and corresponding n, k values.
_GOLD_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
_GOLD_N = np.array([1.46, 0.97, 0.57, 0.27, 0.17], dtype=float)
_GOLD_K = np.array([1.94, 1.87, 2.37, 3.06, 3.76], dtype=float)

# Silver (Ag): wavelengths in nm and corresponding n, k values.
_SILVER_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
_SILVER_N = np.array([0.13, 0.13, 0.14, 0.14, 0.15], dtype=float)
_SILVER_K = np.array([2.98, 3.15, 3.35, 3.54, 3.70], dtype=float)


def _normalize_material_name(name: str) -> str:
    """
    Normalize a user-provided material name to a canonical key.

    Args:
        name (str): User-provided material name (case-insensitive).

    Returns:
        str: Canonical material key used in the internal tables.

    Raises:
        ValueError: If the material name is not recognized.
    """
    key = name.strip().lower()
    if key in _MATERIAL_NAME_MAP:
        return _MATERIAL_NAME_MAP[key]

    supported = sorted(set(_MATERIAL_NAME_MAP.values()))
    raise ValueError(
        f"Unknown particle material '{name}'. Supported materials include: {supported}"
    )


def _interp_complex_from_table(
    wavelengths_nm: np.ndarray,
    n_values: np.ndarray,
    k_values: np.ndarray,
    wavelength_nm: float,
) -> complex:
    """
    Linearly interpolate complex refractive index n + i k from tabulated data.

    For wavelengths outside the tabulated range, the nearest endpoint value
    is used (clamping).

    Args:
        wavelengths_nm (np.ndarray): 1D array of wavelengths in nanometers.
        n_values (np.ndarray): 1D array of real refractive indices at those wavelengths.
        k_values (np.ndarray): 1D array of extinction coefficients at those wavelengths.
        wavelength_nm (float): Query wavelength in nanometers.

    Returns:
        complex: Interpolated complex refractive index n + i k.
    """
    wl = float(wavelength_nm)
    w = wavelengths_nm
    n = n_values
    k = k_values

    if wl <= w[0]:
        n_interp = n[0]
        k_interp = k[0]
    elif wl >= w[-1]:
        n_interp = n[-1]
        k_interp = k[-1]
    else:
        idx = int(np.searchsorted(w, wl) - 1)
        idx = max(0, min(idx, len(w) - 2))
        wl0 = w[idx]
        wl1 = w[idx + 1]
        t = (wl - wl0) / (wl1 - wl0) if wl1 != wl0 else 0.0
        n_interp = (1.0 - t) * n[idx] + t * n[idx + 1]
        k_interp = (1.0 - t) * k[idx] + t * k[idx + 1]

    return complex(float(n_interp), float(k_interp))


def lookup_refractive_index(
    material_name: str,
    wavelength_nm: float,
    diameter_nm: Optional[float] = None,
) -> complex:
    """
    Look up the complex refractive index for a given material.

    Metals (e.g., gold, silver) are modeled with wavelength-dependent
    optical constants using small tabulated datasets with linear
    interpolation. Common dielectrics and biological-like materials are
    modeled as wavelength-independent over the visible range, which is an
    appropriate approximation for narrow-band illumination.

    Args:
        material_name (str): The name of the material, e.g., "Gold", "PET",
            "Polyethylene", "Protein". Case-insensitive material labels are
            accepted.
        wavelength_nm (float): Illumination wavelength in nanometers.
        diameter_nm (Optional[float]): Particle diameter in nanometers. The
            bundled constant-index and tabulated models do not depend on it.

    Returns:
        complex: Complex refractive index n + i k for the requested material.
    """
    canonical = _normalize_material_name(material_name)

    if canonical == "gold":
        return _interp_complex_from_table(
            _GOLD_WAVELENGTHS_NM,
            _GOLD_N,
            _GOLD_K,
            wavelength_nm,
        )
    if canonical == "silver":
        return _interp_complex_from_table(
            _SILVER_WAVELENGTHS_NM,
            _SILVER_N,
            _SILVER_K,
            wavelength_nm,
        )

    if canonical in _MATERIAL_REFRACTIVE_INDEX:
        # Dielectrics / weakly dispersive materials: treat n as constant
        # over the visible; variations are small compared to metals for
        # our purposes.
        return complex(_MATERIAL_REFRACTIVE_INDEX[canonical])

    # If we ever get here, the name map and constant table are inconsistent.
    raise ValueError(
        f"Material '{material_name}' normalized to '{canonical}', "
        "but no refractive index model is defined for this key."
    )


def resolve_component_refractive_index(
    params: dict,
    component: ParticleComponentSpec,
) -> complex:
    """Resolve optical refractive index for one particle component."""
    if component.refractive_index is not None:
        return _coerce_complex(component.refractive_index)
    if component.material is None:
        raise ValueError(
            "Particle component refractive index is undefined. Provide either "
            "component.refractive_index or component.material."
        )
    return lookup_refractive_index(
        material_name=str(component.material),
        wavelength_nm=float(params["wavelength_nm"]),
        diameter_nm=float(component.diameter_nm),
    )


def resolve_component_material_properties(
    params: dict,
    component: ParticleComponentSpec,
) -> MaterialProperties:
    """Resolve full modality material properties for one particle component."""
    n_complex = resolve_component_refractive_index(params, component)
    base = _material_properties_from_name(
        None if component.material is None else str(component.material),
        wavelength_nm=float(params.get("wavelength_nm", 532.0)),
        diameter_nm=float(component.diameter_nm),
        refractive_index=n_complex,
    )
    return _apply_material_override(base, component.material_properties)


def resolve_primary_component_refractive_indices(params: dict) -> np.ndarray:
    """
    Resolve one primary-component complex refractive index per logical particle.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain:
                - "wavelength_nm"
                - "particles"

    Returns:
        np.ndarray: 1D array of complex refractive indices with shape
            (num_particles,), dtype=np.complex128.
    """
    specs = get_particle_specs(params)
    resolved = [
        resolve_component_refractive_index(params, spec.primary_component)
        for spec in specs
    ]
    resolved_array = np.asarray(resolved, dtype=np.complex128)
    params["_resolved_primary_component_refractive_indices"] = resolved_array
    return resolved_array



def material_properties_to_dict(
    material: MaterialProperties,
    *,
    wavelength_nm: float = 532.0,
) -> Dict[str, Any]:
    """Return a JSON-serializable representation of a MaterialProperties object."""
    n_value = material.n_complex(float(wavelength_nm))
    return {
        "name": material.name,
        "n_complex_visible": {
            "real": float(n_value.real),
            "imag": float(n_value.imag),
        },
        "mean_inner_potential_V": float(material.mean_inner_potential_V),
        "density_g_cm3": float(material.density_g_cm3),
        "se_yield_coefficient": float(material.se_yield_coefficient),
        "autofluorescence_per_nm": float(material.autofluorescence_per_nm),
        "fluorophore_density": float(material.fluorophore_density),
        "emission_peak_nm": None if material.emission_peak_nm is None else float(material.emission_peak_nm),
        "excitation_peak_nm": None if material.excitation_peak_nm is None else float(material.excitation_peak_nm),
        "polarizability_tensor": material.polarizability_tensor,
    }


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
        canonical = _normalize_material_name(str(material_name))
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
    fluorescence = _MATERIAL_FLUORESCENCE_DEFAULTS.get(canonical, {})
    electron = _MATERIAL_ELECTRON_DEFAULTS.get(canonical, {})
    return MaterialProperties(
        name=canonical,
        n_complex_visible=n_visible,
        mean_inner_potential_V=float(electron.get("mean_inner_potential_V", 0.0) or 0.0),
        density_g_cm3=float(electron.get("density_g_cm3", 0.0) or 0.0),
        se_yield_coefficient=float(electron.get("se_yield_coefficient", 0.0) or 0.0),
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
        "autofluorescence_per_nm",
        "fluorophore_density",
        "emission_peak_nm",
        "excitation_peak_nm",
        "polarizability_tensor",
    }
    unknown = sorted(set(override) - allowed)
    if unknown:
        raise ValueError(f"Unsupported material-property override key(s): {unknown}.")

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


def resolve_particle_material_properties(
    params: dict,
) -> List[MaterialProperties]:
    """
    Resolve per-particle MaterialProperties for modality-specific physics.

    Component `material_properties` is the canonical way to provide
    fluorescence/electron/material fields. Explicit component refractive-index
    overrides still affect the optical n used by the resolved
    MaterialProperties.
    """
    specs = get_particle_specs(params)
    resolved = [
        resolve_component_material_properties(params, spec.primary_component)
        for spec in specs
    ]

    params["_resolved_particle_material_properties"] = resolved
    wavelength_nm = float(params.get("wavelength_nm", 532.0))
    params["_resolved_particle_material_properties_metadata"] = [
        material_properties_to_dict(material, wavelength_nm=wavelength_nm) for material in resolved
    ]
    return resolved
