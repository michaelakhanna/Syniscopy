"""Normal-incidence thin-film transfer-matrix utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from substrate import GLASS, WATER, MaterialProperties, material_from_name


def _layer_index(layer: dict[str, Any], wavelength_nm: float) -> complex:
    if "n_complex" in layer:
        raw = layer["n_complex"]
        if isinstance(raw, dict):
            return complex(float(raw.get("real", 0.0)), float(raw.get("imag", 0.0)))
        return complex(raw)
    material = layer.get("material", None)
    if isinstance(material, MaterialProperties):
        return material.n_complex(wavelength_nm)
    return material_from_name(material).n_complex(wavelength_nm)


def _material_index(
    material: str | MaterialProperties | None,
    wavelength_nm: float,
    *,
    default: MaterialProperties,
) -> complex:
    return material_from_name(material, default).n_complex(wavelength_nm)


def normal_incidence_thinfilm_reflection(
    incident_material: str | MaterialProperties | None,
    substrate_material: str | MaterialProperties | None,
    layers: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    wavelength_nm: float,
) -> complex:
    """Return normal-incidence amplitude reflection for an ordered layer stack.

    Layers are ordered from the incident medium toward the substrate. With no
    layers this reduces exactly to the normal-incidence Fresnel coefficient.
    """
    wavelength = float(wavelength_nm)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError(f"wavelength_nm must be positive and finite; got {wavelength_nm!r}.")
    n0 = _material_index(incident_material, wavelength, default=WATER)
    ns = _material_index(substrate_material, wavelength, default=GLASS)
    stack = [] if layers is None else list(layers)
    if not stack:
        denom = n0 + ns
        if abs(denom) <= 1e-12:
            raise ValueError("Thin-film Fresnel denominator is near zero.")
        return (n0 - ns) / denom

    M = np.eye(2, dtype=complex)
    for layer in stack:
        if not isinstance(layer, dict):
            raise TypeError("Each thin-film layer must be a dictionary.")
        n_j = _layer_index(layer, wavelength)
        d_j = float(layer.get("thickness_nm", 0.0))
        if not np.isfinite(d_j) or d_j < 0.0:
            raise ValueError(f"Layer thickness_nm must be finite and non-negative; got {d_j!r}.")
        delta = 2.0 * np.pi * n_j * d_j / wavelength
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        eta = n_j
        if abs(eta) <= 1e-12:
            raise ValueError(f"Layer refractive index is near zero: {n_j!r}.")
        # Material indices in Syniscopy use the common extinction convention
        # n_complex = n + i k.  With that convention the propagation matrix
        # carries the opposite sign from the n - i k characteristic-matrix form.
        M_j = np.array(
            [
                [cos_d, -1j * sin_d / eta],
                [-1j * eta * sin_d, cos_d],
            ],
            dtype=complex,
        )
        M = M @ M_j

    numerator = n0 * M[0, 0] + n0 * ns * M[0, 1] - M[1, 0] - ns * M[1, 1]
    denominator = n0 * M[0, 0] + n0 * ns * M[0, 1] + M[1, 0] + ns * M[1, 1]
    if abs(denominator) <= 1e-12:
        raise ValueError("Thin-film reflection denominator is near zero.")
    return numerator / denominator
