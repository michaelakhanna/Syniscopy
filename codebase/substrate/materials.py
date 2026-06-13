"""Substrate material properties and optical helper formulas."""
from __future__ import annotations
from configured_parameters import configured_value

from dataclasses import replace

import numpy as np

from material_optical_catalog import (
    material_electron_defaults,
    material_fluorescence_defaults,
    optical_index_model_for_material,
)
from material_types import ComplexIndexModel, MaterialProperties


def _catalog_material(material_name: str, *, display_name: str | None = None) -> MaterialProperties:
    electron = material_electron_defaults(material_name)
    fluorescence = material_fluorescence_defaults(material_name)
    return MaterialProperties(
        display_name or material_name,
        optical_index_model_for_material(material_name),
        mean_inner_potential_V=float(electron.get("mean_inner_potential_V", 0.0)),
        density_g_cm3=float(electron.get("density_g_cm3", 0.0)),
        se_yield_coefficient=float(electron.get("se_yield_coefficient", 0.0)),
        atomic_number=None if electron.get("atomic_number") is None else float(electron["atomic_number"]),
        atomic_weight_g_mol=(
            None
            if electron.get("atomic_weight_g_mol") is None
            else float(electron["atomic_weight_g_mol"])
        ),
        autofluorescence_per_nm=float(fluorescence.get("autofluorescence_per_nm", 0.0) or 0.0),
        fluorophore_density=float(fluorescence.get("fluorophore_density", 0.0) or 0.0),
        emission_peak_nm=fluorescence.get("emission_peak_nm"),
        excitation_peak_nm=fluorescence.get("excitation_peak_nm"),
    )



AIR = _catalog_material("air")
VACUUM = MaterialProperties("vacuum", 1.00 + 0.0j)
WATER = _catalog_material("water")
SIO2 = replace(_catalog_material("silica", display_name="SiO2"), autofluorescence_per_nm=0.02)
SI = _catalog_material("silicon", display_name="Si")
CARBON = _catalog_material("carbon")
GOLD = _catalog_material("gold")
SILVER = _catalog_material("silver")
GLASS = replace(_catalog_material("glass"), density_g_cm3=2.5)
PET = _catalog_material("pet", display_name="PET")
POLYETHYLENE = _catalog_material("polyethylene")
POLYPROPYLENE = _catalog_material("polypropylene")
POLYSTYRENE = _catalog_material("polystyrene")
FLUORESCENT_POLYSTYRENE = _catalog_material("fluorescent_polystyrene")
PROTEIN = _catalog_material("protein")
LIPID = _catalog_material("lipid")


_MATERIALS = {
    "air": AIR,
    "vacuum": VACUUM,
    "water": WATER,
    "silica": SIO2,
    "silicon": SI,
    "carbon": CARBON,
    "gold": GOLD,
    "silver": SILVER,
    "glass": GLASS,
    "pet": PET,
    "polyethylene": POLYETHYLENE,
    "polypropylene": POLYPROPYLENE,
    "polystyrene": POLYSTYRENE,
    "fluorescent_polystyrene": FLUORESCENT_POLYSTYRENE,
    "protein": PROTEIN,
    "lipid": LIPID,
}


def material_from_name(
    name: str | MaterialProperties | None,
    default: MaterialProperties | None = None,
) -> MaterialProperties:
    if isinstance(name, MaterialProperties):
        return name
    if name is None:
        if default is None:
            raise ValueError("material name is required when no default is provided.")
        return default
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _MATERIALS:
        known = ", ".join(sorted(_MATERIALS))
        raise ValueError(
            f"Unknown sample-environment material {name!r}. "
            f"Known materials are: {known}."
        )
    return _MATERIALS[key]

def fresnel_reflection_amplitude(
    material_top: str | MaterialProperties | None,
    material_bottom: str | MaterialProperties | None,
    wavelength_nm: float,
    *,
    default_top: MaterialProperties = WATER,
    default_bottom: MaterialProperties = GLASS,
) -> complex:
    """Normal-incidence Fresnel reflection amplitude from top to bottom medium."""
    top = material_from_name(material_top, default_top)
    bottom = material_from_name(material_bottom, default_bottom)
    n_top = top.n_complex(float(wavelength_nm))
    n_bottom = bottom.n_complex(float(wavelength_nm))
    denom = n_top + n_bottom
    if abs(denom) <= 1e-12:
        raise ValueError(
            "Fresnel reflection denominator is near zero for "
            f"n_top={n_top!r}, n_bottom={n_bottom!r}."
        )
    return (n_top - n_bottom) / denom

def _material_with_param_overrides(material: MaterialProperties, params: dict, prefix: str) -> MaterialProperties:
    updates = {}
    for field_name in (
        "autofluorescence_per_nm",
        "fluorophore_density",
        "emission_peak_nm",
        "excitation_peak_nm",
        "se_yield_coefficient",
        "mean_inner_potential_V",
        "density_g_cm3",
    ):
        key = f"{prefix}_{field_name}"
        if key in params:
            value = configured_value(params, key)
            if value is None:
                updates[field_name] = None
            else:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"{key} must be finite when provided; got {value}."
                    )
                updates[field_name] = value
    if not updates:
        return material
    return replace(material, **updates)

__all__ = [
    "AIR",
    "CARBON",
    "ComplexIndexModel",
    "FLUORESCENT_POLYSTYRENE",
    "GLASS",
    "GOLD",
    "LIPID",
    "MaterialProperties",
    "PET",
    "POLYETHYLENE",
    "POLYPROPYLENE",
    "POLYSTYRENE",
    "PROTEIN",
    "SILVER",
    "SI",
    "SIO2",
    "VACUUM",
    "WATER",
    "fresnel_reflection_amplitude",
    "material_from_name",
]
