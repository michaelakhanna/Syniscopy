"""Shared optical-parameter resolution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from config.runtime import param_value
from modality_registry import canonical_modality_name


def resolve_probe_wavelength_nm(params: dict[str, Any]) -> float:
    """Return the optical probe wavelength, with probe-specific override support."""
    value = param_value(params, "probe_wavelength_nm")
    if value is None:
        try:
            active_modality = canonical_modality_name(param_value(params, "imaging_model"))
        except Exception:
            active_modality = ""
        if active_modality == "ricm":
            value = param_value(params, "ricm_wavelength_nm")
        else:
            value = param_value(params, "wavelength_nm")
    wavelength_nm = float(value)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(
            "Optical probe wavelength must be finite and positive; "
            f"got {value!r}."
        )
    return wavelength_nm


__all__ = ["resolve_probe_wavelength_nm"]
