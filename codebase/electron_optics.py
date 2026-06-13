"""Electron-optics constants and formulas."""

from __future__ import annotations

import numpy as np

# Physical constants in SI, used only by electron-optics formulas.
_PLANCK_H_J_S = 6.62607015e-34
_REDUCED_PLANCK_HBAR_J_S = _PLANCK_H_J_S / (2.0 * np.pi)
_ELECTRON_MASS_KG = 9.1093837015e-31
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_SPEED_OF_LIGHT_M_S = 2.99792458e8


def electron_wavelength_m(acceleration_kV: float) -> float:
    """
    Relativistic de Broglie wavelength of an electron accelerated through
    ``acceleration_kV`` kilovolts. Returns wavelength in metres.
    """
    voltage = float(acceleration_kV) * 1.0e3
    if voltage <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    kinetic_j = _ELEMENTARY_CHARGE_C * voltage
    momentum = np.sqrt(
        (kinetic_j / _SPEED_OF_LIGHT_M_S) ** 2
        + 2.0 * _ELECTRON_MASS_KG * kinetic_j
    )
    return _PLANCK_H_J_S / momentum


def electron_interaction_parameter_rad_per_V_nm(acceleration_kV: float) -> float:
    """Relativistic weak-phase interaction parameter in rad/(V nm)."""
    voltage = float(acceleration_kV) * 1.0e3
    if voltage <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    kinetic_j = _ELEMENTARY_CHARGE_C * voltage
    momentum = np.sqrt(
        (kinetic_j / _SPEED_OF_LIGHT_M_S) ** 2
        + 2.0 * _ELECTRON_MASS_KG * kinetic_j
    )
    sigma_per_v_m = (
        _ELEMENTARY_CHARGE_C
        * (kinetic_j + _ELECTRON_MASS_KG * _SPEED_OF_LIGHT_M_S**2)
        / (_REDUCED_PLANCK_HBAR_J_S * _SPEED_OF_LIGHT_M_S**2 * momentum)
    )
    return float(sigma_per_v_m * 1.0e-9)


def scherzer_defocus_m(acceleration_kV: float, Cs_mm: float) -> float:
    """
    Scherzer defocus in metres for acceleration voltage and spherical
    aberration. Positive output follows the underfocus convention.
    """
    cs_m = 1.0e-3 * float(Cs_mm)
    if cs_m < 0.0:
        raise ValueError(f"C_s must be non-negative; got {Cs_mm} mm.")
    wavelength_m = electron_wavelength_m(acceleration_kV)
    return np.sqrt(1.5 * wavelength_m * cs_m)


__all__ = [
    "_ELEMENTARY_CHARGE_C",
    "electron_interaction_parameter_rad_per_V_nm",
    "electron_wavelength_m",
    "scherzer_defocus_m",
]
