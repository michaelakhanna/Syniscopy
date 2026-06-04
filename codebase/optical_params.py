"""Shared optical-parameter resolution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from config.runtime import param_value


def resolve_probe_wavelength_nm(params: dict[str, Any]) -> float:
    """Return the optical probe wavelength, with probe-specific override support."""
    value = param_value(params, "probe_wavelength_nm")
    if value is None:
        value = param_value(params, "wavelength_nm")
    wavelength_nm = float(value)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(
            "Optical probe wavelength must be finite and positive; "
            f"got {value!r}."
        )
    return wavelength_nm


__all__ = ["resolve_probe_wavelength_nm"]
