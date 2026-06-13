"""Single-source optical, fluorescence, electron, and SEM material constants."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional

import numpy as np


MATERIAL_REFRACTIVE_INDEX: Dict[str, complex] = {
    "air": 1.00 + 0.0j,
    "aluminum": 1.44 + 7.38j,
    "carbon": 2.42 + 0.0j,
    "copper": 0.64 + 2.62j,
    "pet": 1.57 + 0.0j,
    "polyethylene": 1.51 + 0.0j,
    "polypropylene": 1.49 + 0.0j,
    "polystyrene": 1.59 + 0.0j,
    "fluorescent_polystyrene": 1.59 + 0.0j,
    "silica": 1.46 + 0.0j,
    "silicon": 4.14 + 0.04j,
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
        "atomic_number": 79.0,
        "atomic_weight_g_mol": 196.96657,
    },
    "silver": {
        "mean_inner_potential_V": 22.0,
        "density_g_cm3": 10.5,
        "se_yield_coefficient": 0.16,
        "atomic_number": 47.0,
        "atomic_weight_g_mol": 107.8682,
    },
    "copper": {
        "mean_inner_potential_V": 17.0,
        "density_g_cm3": 8.96,
        "se_yield_coefficient": 0.12,
        "atomic_number": 29.0,
        "atomic_weight_g_mol": 63.546,
    },
    "aluminum": {
        "mean_inner_potential_V": 12.0,
        "density_g_cm3": 2.70,
        "se_yield_coefficient": 0.08,
        "atomic_number": 13.0,
        "atomic_weight_g_mol": 26.9815385,
    },
    "silicon": {
        "mean_inner_potential_V": 12.1,
        "density_g_cm3": 2.329,
        "se_yield_coefficient": 0.10,
        "atomic_number": 14.0,
        "atomic_weight_g_mol": 28.085,
    },
    "pet": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.38,
        "se_yield_coefficient": 0.05,
        "atomic_number": 5.7,
        "atomic_weight_g_mol": 7.08,
    },
    "polyethylene": {
        "mean_inner_potential_V": 7.5,
        "density_g_cm3": 0.94,
        "se_yield_coefficient": 0.05,
        "atomic_number": 4.0,
        "atomic_weight_g_mol": 4.68,
    },
    "polypropylene": {
        "mean_inner_potential_V": 7.5,
        "density_g_cm3": 0.90,
        "se_yield_coefficient": 0.05,
        "atomic_number": 4.0,
        "atomic_weight_g_mol": 4.68,
    },
    "polystyrene": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.05,
        "se_yield_coefficient": 0.05,
        "atomic_number": 5.7,
        "atomic_weight_g_mol": 6.51,
    },
    "fluorescent_polystyrene": {
        "mean_inner_potential_V": 8.0,
        "density_g_cm3": 1.05,
        "se_yield_coefficient": 0.05,
        "atomic_number": 5.7,
        "atomic_weight_g_mol": 6.51,
    },
    "silica": {
        "mean_inner_potential_V": 10.1,
        "density_g_cm3": 2.20,
        "se_yield_coefficient": 0.10,
        "atomic_number": 10.8,
        "atomic_weight_g_mol": 20.03,
    },
    "water": {
        "mean_inner_potential_V": 4.0,
        "density_g_cm3": 1.00,
        "se_yield_coefficient": 0.02,
        "atomic_number": 7.42,
        "atomic_weight_g_mol": 6.01,
    },
    "protein": {
        "mean_inner_potential_V": 6.0,
        "density_g_cm3": 1.35,
        "se_yield_coefficient": 0.04,
        "atomic_number": 7.0,
        "atomic_weight_g_mol": 6.9,
    },
    "lipid": {
        "mean_inner_potential_V": 4.5,
        "density_g_cm3": 0.92,
        "se_yield_coefficient": 0.04,
        "atomic_number": 5.9,
        "atomic_weight_g_mol": 6.5,
    },
    "glass": {
        "mean_inner_potential_V": 9.5,
        "density_g_cm3": 2.50,
        "se_yield_coefficient": 0.10,
        "atomic_number": 10.8,
        "atomic_weight_g_mol": 20.03,
    },
    "air": {
        "mean_inner_potential_V": 0.0,
        "density_g_cm3": 0.0,
        "se_yield_coefficient": 0.0,
        "atomic_number": 7.3,
        "atomic_weight_g_mol": 14.6,
    },
    "carbon": {
        "mean_inner_potential_V": 8.7,
        "density_g_cm3": 2.0,
        "se_yield_coefficient": 0.08,
        "atomic_number": 6.0,
        "atomic_weight_g_mol": 12.011,
    },
}

MATERIAL_NAME_MAP: Dict[str, str] = {
    canonical: canonical
    for canonical in (
        set(MATERIAL_REFRACTIVE_INDEX)
        | set(MATERIAL_FLUORESCENCE_DEFAULTS)
        | set(MATERIAL_ELECTRON_DEFAULTS)
    )
}

GOLD_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
GOLD_N = np.array([1.46, 0.97, 0.57, 0.27, 0.17], dtype=float)
GOLD_K = np.array([1.94, 1.87, 2.37, 3.06, 3.76], dtype=float)

SILVER_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
SILVER_N = np.array([0.13, 0.13, 0.14, 0.14, 0.15], dtype=float)
SILVER_K = np.array([2.98, 3.15, 3.35, 3.54, 3.70], dtype=float)


@dataclass(frozen=True)
class SEMTransportMaterial:
    """Material constants required by screened-Rutherford SEM transport."""

    name: str
    atomic_number: float
    atomic_weight_g_mol: float
    density_g_cm3: float
    se_yield_coefficient: float


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
    return _lookup_refractive_index_cached(canonical, float(wavelength_nm))


@lru_cache(maxsize=1024)
def _lookup_refractive_index_cached(canonical: str, wavelength_nm: float) -> complex:
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
        f"Material '{canonical}' has no refractive index model defined."
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
    canonical = normalize_material_name(material_name)
    return dict(_material_electron_defaults_cached(canonical))


def material_fluorescence_defaults(material_name: str) -> dict[str, float | None]:
    canonical = normalize_material_name(material_name)
    return dict(_material_fluorescence_defaults_cached(canonical))


def sem_transport_material(material_name: str) -> SEMTransportMaterial:
    canonical = normalize_material_name(material_name)
    return _sem_transport_material_cached(canonical)


@lru_cache(maxsize=256)
def _material_electron_defaults_cached(canonical: str) -> tuple[tuple[str, float], ...]:
    return tuple(MATERIAL_ELECTRON_DEFAULTS.get(canonical, {}).items())


@lru_cache(maxsize=256)
def _material_fluorescence_defaults_cached(canonical: str) -> tuple[tuple[str, float | None], ...]:
    return tuple(MATERIAL_FLUORESCENCE_DEFAULTS.get(canonical, {}).items())


@lru_cache(maxsize=256)
def _sem_transport_material_cached(canonical: str) -> SEMTransportMaterial:
    values = MATERIAL_ELECTRON_DEFAULTS[canonical]
    required = (
        "atomic_number",
        "atomic_weight_g_mol",
        "density_g_cm3",
        "se_yield_coefficient",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            f"Material {material_name!r} has no complete SEM transport constants: {missing}."
        )
    atomic_number = float(values["atomic_number"])
    atomic_weight = float(values["atomic_weight_g_mol"])
    density = float(values["density_g_cm3"])
    se_yield = float(values["se_yield_coefficient"])
    if (
        not np.isfinite(atomic_number)
        or not np.isfinite(atomic_weight)
        or not np.isfinite(density)
        or atomic_number <= 0.0
        or atomic_weight <= 0.0
        or density <= 0.0
    ):
        raise ValueError(
            f"Material {material_name!r} is not valid for SEM transport: "
            "atomic_number, atomic_weight_g_mol, and density_g_cm3 must be positive."
        )
    return SEMTransportMaterial(
        name=canonical,
        atomic_number=atomic_number,
        atomic_weight_g_mol=atomic_weight,
        density_g_cm3=density,
        se_yield_coefficient=se_yield,
    )


__all__ = [
    "MATERIAL_ELECTRON_DEFAULTS",
    "MATERIAL_FLUORESCENCE_DEFAULTS",
    "MATERIAL_NAME_MAP",
    "MATERIAL_REFRACTIVE_INDEX",
    "SEMTransportMaterial",
    "lookup_refractive_index",
    "material_electron_defaults",
    "material_fluorescence_defaults",
    "normalize_material_name",
    "optical_index_model_for_material",
    "sem_transport_material",
]
