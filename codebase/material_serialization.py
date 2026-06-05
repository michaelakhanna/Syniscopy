"""Serialization helpers for material-property runtime objects."""

from __future__ import annotations

from typing import Any, Dict

from material_types import MaterialProperties


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
        "atomic_number": None if material.atomic_number is None else float(material.atomic_number),
        "atomic_weight_g_mol": None if material.atomic_weight_g_mol is None else float(material.atomic_weight_g_mol),
        "autofluorescence_per_nm": float(material.autofluorescence_per_nm),
        "fluorophore_density": float(material.fluorophore_density),
        "emission_peak_nm": None if material.emission_peak_nm is None else float(material.emission_peak_nm),
        "excitation_peak_nm": None if material.excitation_peak_nm is None else float(material.excitation_peak_nm),
        "polarizability_tensor": material.polarizability_tensor,
    }


__all__ = ["material_properties_to_dict"]
