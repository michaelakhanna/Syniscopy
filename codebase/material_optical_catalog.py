"""Single-source optical and modality material defaults."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


MATERIAL_REFRACTIVE_INDEX: Dict[str, complex] = {
    "air": 1.00 + 0.0j,
    "carbon": 2.42 + 0.0j,
    "pet": 1.57 + 0.0j,
    "polyethylene": 1.51 + 0.0j,
    "polypropylene": 1.49 + 0.0j,
    "polystyrene": 1.59 + 0.0j,
    "fluorescent_polystyrene": 1.59 + 0.0j,
    "silica": 1.46 + 0.0j,
    "water": 1.33 + 0.0j,
    "protein": 1.45 + 0.0j,
    "lipid": 1.47 + 0.0j,
    "glass": 1.52 + 0.0j,
}

MATERIAL_FLUORESCENCE_DEFAULTS: Dict[str, Dict[str, float | None]] = {
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

MATERIAL_ELECTRON_DEFAULTS: Dict[str, Dict[str, float]] = {
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

MATERIAL_NAME_VARIANTS: Dict[str, List[str]] = {
    "gold": ["gold", "au", "gold nanoparticle", "nanogold"],
    "silver": ["silver", "ag", "silver nanoparticle", "nanosilver"],
    "pet": ["pet", "polyethylene terephthalate", "pet plastic"],
    "polyethylene": ["polyethylene", "pe"],
    "polypropylene": ["polypropylene", "pp"],
    "polystyrene": ["polystyrene", "ps"],
    "fluorescent_polystyrene": [
        "fluorescent_polystyrene",
        "fluorescent polystyrene",
        "fluorescent_ps",
        "fluorescent ps",
    ],
    "air": ["air"],
    "carbon": ["carbon", "amorphous carbon", "holey carbon"],
    "silica": ["silica", "sio2", "silicon dioxide"],
    "water": ["water", "h2o"],
    "protein": ["protein", "proteins"],
    "lipid": ["lipid", "lipids"],
    "glass": ["glass", "bk7", "borosilicate glass"],
}

MATERIAL_NAME_MAP: Dict[str, str] = {}
for canonical, variants in MATERIAL_NAME_VARIANTS.items():
    for variant in variants:
        MATERIAL_NAME_MAP[variant.lower()] = canonical
    MATERIAL_NAME_MAP[canonical.lower()] = canonical

GOLD_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
GOLD_N = np.array([1.46, 0.97, 0.57, 0.27, 0.17], dtype=float)
GOLD_K = np.array([1.94, 1.87, 2.37, 3.06, 3.76], dtype=float)

SILVER_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
SILVER_N = np.array([0.13, 0.13, 0.14, 0.14, 0.15], dtype=float)
SILVER_K = np.array([2.98, 3.15, 3.35, 3.54, 3.70], dtype=float)


def normalize_material_name(name: str) -> str:
    key = name.strip().lower()
    if key in MATERIAL_NAME_MAP:
        return MATERIAL_NAME_MAP[key]
    supported = sorted(set(MATERIAL_NAME_MAP.values()))
    raise ValueError(
        f"Unknown particle material '{name}'. Supported materials include: {supported}"
    )


def _interp_complex_from_table(
    wavelengths_nm: np.ndarray,
    n_values: np.ndarray,
    k_values: np.ndarray,
    wavelength_nm: float,
) -> complex:
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
    del diameter_nm
    canonical = normalize_material_name(material_name)
    if canonical == "gold":
        return _interp_complex_from_table(GOLD_WAVELENGTHS_NM, GOLD_N, GOLD_K, wavelength_nm)
    if canonical == "silver":
        return _interp_complex_from_table(
            SILVER_WAVELENGTHS_NM,
            SILVER_N,
            SILVER_K,
            wavelength_nm,
        )
    if canonical in MATERIAL_REFRACTIVE_INDEX:
        return complex(MATERIAL_REFRACTIVE_INDEX[canonical])
    raise ValueError(
        f"Material '{material_name}' normalized to '{canonical}', "
        "but no refractive index model is defined for this key."
    )


def optical_index_model_for_material(material_name: str) -> complex | Callable[[float], complex]:
    canonical = normalize_material_name(material_name)
    if canonical in {"gold", "silver"}:
        return lambda wavelength_nm, material=canonical: lookup_refractive_index(
            material,
            wavelength_nm,
        )
    return lookup_refractive_index(canonical, 532.0)


def material_electron_defaults(material_name: str) -> dict[str, float]:
    return dict(MATERIAL_ELECTRON_DEFAULTS.get(normalize_material_name(material_name), {}))


def material_fluorescence_defaults(material_name: str) -> dict[str, float | None]:
    return dict(MATERIAL_FLUORESCENCE_DEFAULTS.get(normalize_material_name(material_name), {}))


__all__ = [
    "MATERIAL_ELECTRON_DEFAULTS",
    "MATERIAL_FLUORESCENCE_DEFAULTS",
    "MATERIAL_NAME_MAP",
    "MATERIAL_NAME_VARIANTS",
    "MATERIAL_REFRACTIVE_INDEX",
    "lookup_refractive_index",
    "material_electron_defaults",
    "material_fluorescence_defaults",
    "normalize_material_name",
    "optical_index_model_for_material",
]
