"""Shared material datatypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


ComplexIndexModel = Union[complex, Callable[[float], complex]]


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
    atomic_number: float | None = None
    atomic_weight_g_mol: float | None = None
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


__all__ = ["ComplexIndexModel", "MaterialProperties"]
