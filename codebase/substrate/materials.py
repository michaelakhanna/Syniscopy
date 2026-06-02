"""Substrate material properties and optical helper formulas."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


ComplexIndexModel = complex | Callable[[float], complex]


def _validate_wavelength_nm(wavelength_nm: float) -> float:
    wavelength_nm = float(wavelength_nm)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(
            "wavelength_nm must be finite and positive; "
            f"got {wavelength_nm}."
        )
    return wavelength_nm


@dataclass(frozen=True)

class MaterialProperties:
    """
    Material properties used by optical, fluorescence, and electron models.

    Optical indices are dimensionless. Mean inner potential is in volts,
    density is g/cm^3, fluorescence/autofluorescence scales are relative source
    terms, and spectral peaks are wavelengths in nanometres.
    """

    name: str
    n_complex_visible: ComplexIndexModel = 1.0 + 0.0j
    mean_inner_potential_V: float = 0.0
    density_g_cm3: float = 0.0
    se_yield_coefficient: float = 0.0
    autofluorescence_per_nm: float = 0.0
    fluorophore_density: float = 0.0
    emission_peak_nm: float | None = None
    excitation_peak_nm: float | None = None
    polarizability_tensor: tuple[tuple[float, float, float], ...] | None = None

    def n_complex(self, wavelength_nm: float) -> complex:
        """Return the complex refractive index at ``wavelength_nm``."""
        wavelength_nm = _validate_wavelength_nm(wavelength_nm)
        model = self.n_complex_visible
        if callable(model):
            n_complex = complex(model(wavelength_nm))
        else:
            n_complex = complex(model)
        if not np.isfinite(n_complex):
            raise ValueError(
                "Material refractive-index model returned a nonfinite value "
                f"for {self.name!r} at wavelength_nm={wavelength_nm}."
            )
        return n_complex



AIR = MaterialProperties("air", 1.00 + 0.0j)
VACUUM = MaterialProperties("vacuum", 1.00 + 0.0j)
WATER = MaterialProperties("water", 1.33 + 0.0j, mean_inner_potential_V=4.0, density_g_cm3=1.00)
SIO2 = MaterialProperties(
    "SiO2",
    1.46 + 0.0j,
    mean_inner_potential_V=10.1,
    density_g_cm3=2.20,
    se_yield_coefficient=0.10,
    autofluorescence_per_nm=0.02,
)
SI = MaterialProperties(
    "Si",
    3.88 + 0.02j,
    mean_inner_potential_V=11.7,
    density_g_cm3=2.33,
    se_yield_coefficient=0.13,
)
CARBON = MaterialProperties(
    "carbon",
    2.42 + 0.0j,
    mean_inner_potential_V=8.7,
    density_g_cm3=2.0,
    se_yield_coefficient=0.08,
)
GOLD = MaterialProperties(
    "gold",
    0.47 + 2.41j,
    mean_inner_potential_V=25.0,
    density_g_cm3=19.3,
    se_yield_coefficient=0.18,
)
SILVER = MaterialProperties(
    "silver",
    0.14 + 3.98j,
    mean_inner_potential_V=22.0,
    density_g_cm3=10.5,
    se_yield_coefficient=0.16,
)
GLASS = MaterialProperties("glass", 1.52 + 0.0j, mean_inner_potential_V=9.5, density_g_cm3=2.5)
PET = MaterialProperties(
    "PET",
    1.57 + 0.0j,
    mean_inner_potential_V=8.0,
    density_g_cm3=1.38,
    se_yield_coefficient=0.05,
)
POLYETHYLENE = MaterialProperties(
    "polyethylene",
    1.51 + 0.0j,
    mean_inner_potential_V=7.5,
    density_g_cm3=0.94,
    se_yield_coefficient=0.05,
)
POLYPROPYLENE = MaterialProperties(
    "polypropylene",
    1.49 + 0.0j,
    mean_inner_potential_V=7.5,
    density_g_cm3=0.90,
    se_yield_coefficient=0.05,
)
POLYSTYRENE = MaterialProperties(
    "polystyrene",
    1.59 + 0.0j,
    mean_inner_potential_V=8.0,
    density_g_cm3=1.05,
    se_yield_coefficient=0.05,
)
FLUORESCENT_POLYSTYRENE = replace(
    POLYSTYRENE,
    name="fluorescent_polystyrene",
    fluorophore_density=1.0,
    excitation_peak_nm=488.0,
    emission_peak_nm=520.0,
)
PROTEIN = MaterialProperties(
    "protein",
    1.45 + 0.0j,
    mean_inner_potential_V=6.0,
    density_g_cm3=1.35,
    se_yield_coefficient=0.04,
)
LIPID = MaterialProperties(
    "lipid",
    1.47 + 0.0j,
    mean_inner_potential_V=4.5,
    density_g_cm3=0.92,
    se_yield_coefficient=0.04,
)


_MATERIALS = {
    "air": AIR,
    "vacuum": VACUUM,
    "water": WATER,
    "buffer": WATER,
    "sio2": SIO2,
    "silica": SIO2,
    "silicon_dioxide": SIO2,
    "si": SI,
    "silicon": SI,
    "carbon": CARBON,
    "holey_carbon": CARBON,
    "gold": GOLD,
    "au": GOLD,
    "silver": SILVER,
    "ag": SILVER,
    "glass": GLASS,
    "bk7": GLASS,
    "borosilicate_glass": GLASS,
    "pet": PET,
    "polyethylene_terephthalate": PET,
    "polyethylene": POLYETHYLENE,
    "polypropylene": POLYPROPYLENE,
    "polystyrene": POLYSTYRENE,
    "ps": POLYSTYRENE,
    "fluorescent_polystyrene": FLUORESCENT_POLYSTYRENE,
    "fluorescent_polystyrene_bead": FLUORESCENT_POLYSTYRENE,
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
            f"Known materials/aliases are: {known}."
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
            value = params[key]
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
