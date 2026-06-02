"""Electron-optics constants and formulas."""

from __future__ import annotations

import numpy as np

# Physical constants in SI, used only by electron-optics formulas.
_PLANCK_H_J_S            = 6.62607015e-34
_REDUCED_PLANCK_HBAR_J_S = _PLANCK_H_J_S / (2.0 * np.pi)
_ELECTRON_MASS_KG        = 9.1093837015e-31
_ELEMENTARY_CHARGE_C     = 1.602176634e-19
_SPEED_OF_LIGHT_M_S      = 2.99792458e8

def electron_wavelength_m(acceleration_kV: float) -> float:
    """
    Relativistic de Broglie wavelength of an electron accelerated through
    ``acceleration_kV`` kilovolts. Returns wavelength in metres.

    The non-relativistic form would be lambda = h / sqrt(2 m_e e V), but
    at 100-300 kV the relativistic correction factor (1 + eV/(2 m c^2))
    under the square root reduces the apparent wavelength by ~5-20%.
    We use the relativistic expression to remain accurate across the
    full range of modern TEM operating voltages (80-300 kV).
    """
    V = float(acceleration_kV) * 1.0e3  # volts
    if V <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    eV = _ELEMENTARY_CHARGE_C * V
    m = _ELECTRON_MASS_KG
    c = _SPEED_OF_LIGHT_M_S
    h = _PLANCK_H_J_S
    # Relativistic momentum from E_kin = eV: p = sqrt( (eV)^2 / c^2 + 2 m eV ).
    p = np.sqrt((eV / c) ** 2 + 2.0 * m * eV)
    return h / p

def electron_interaction_parameter_rad_per_V_nm(acceleration_kV: float) -> float:
    """
    Relativistic weak-phase interaction parameter in rad/(V nm).

    The TEM source map uses ``phi = sigma * V_mip * thickness`` where
    ``V_mip`` is the material mean inner potential in volts and thickness is
    in nanometres.
    """
    V = float(acceleration_kV) * 1.0e3
    if V <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    kinetic_J = _ELEMENTARY_CHARGE_C * V
    m = _ELECTRON_MASS_KG
    c = _SPEED_OF_LIGHT_M_S
    p = np.sqrt((kinetic_J / c) ** 2 + 2.0 * m * kinetic_J)
    sigma_per_V_m = (
        _ELEMENTARY_CHARGE_C
        * (kinetic_J + m * c ** 2)
        / (_REDUCED_PLANCK_HBAR_J_S * c ** 2 * p)
    )
    return float(sigma_per_V_m * 1.0e-9)

def scherzer_defocus_m(acceleration_kV: float, Cs_mm: float) -> float:
    """
    Scherzer defocus in metres given the electron wavelength (from the
    accelerating voltage) and the objective spherical aberration Cs (in mm).
    Convention: positive output means underfocus (Scherzer condition).
    The Scherzer defocus is the value that produces the broadest pass-band
    in the phase-contrast transfer function:

        Delta f_Sch = sqrt( 1.5 · lambda · C_s ).
    """
    Cs_m = 1.0e-3 * float(Cs_mm)
    if Cs_m < 0.0:
        raise ValueError(f"C_s must be non-negative; got {Cs_mm} mm.")
    wavelength_m = electron_wavelength_m(acceleration_kV)
    return np.sqrt(1.5 * wavelength_m * Cs_m)

__all__ = [name for name in globals() if not name.startswith("__")]
