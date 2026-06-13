"""Metadata helpers shared by Fisher diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._constants import (
    _FISHER_EIGENVALUE_UNDERFLOW_FLOOR,
    _FISHER_RANK_RELATIVE_TOL,
    _RELATIVE_DET_SINGULAR_TOL,
)


def _derivative_unit(signal_units: str, coordinate_unit: str) -> str:
    unit = str(signal_units or "contrast").strip() or "contrast"
    return f"{unit}_per_{coordinate_unit}"


def _variance_units(signal_units: str) -> str:
    unit = str(signal_units or "contrast").strip() or "contrast"
    return f"{unit}_squared"


def _localization_derivative_metadata(
    pixel_size_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """Metadata for the lateral spectral band-limited derivative convention."""
    step_note = (
        "FFT spectral derivative of the sampled band-limited observable; "
        "no finite-difference step"
    )
    return {
        "state_axes": ["x", "y"],
        "measurement_domain": str(measurement_domain),
        "signal_units": str(signal_units),
        "derivative_units": [
            _derivative_unit(signal_units, "nm"),
            _derivative_unit(signal_units, "nm"),
        ],
        "derivative_basis": "spectral_band_limited",
        "step_size_free": True,
        "lateral_derivative_step_size_free": True,
        "lateral_step_note": step_note,
        "pixel_size_nm": float(pixel_size_nm),
        "noise_variance_units": (
            str(noise_variance_units)
            if noise_variance_units is not None
            else _variance_units(signal_units)
        ),
        "fisher_axis_units": ["1/nm^2", "1/nm^2"],
        "relative_determinant_singularity_tolerance": float(_RELATIVE_DET_SINGULAR_TOL),
    }


def _fisher_rank_metadata(F: np.ndarray) -> dict[str, Any]:
    """Return eigenvalue/rank metadata for an already assembled Fisher matrix."""
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] != F_arr.shape[1]:
        return {
            "fisher_eigenvalues": [],
            "fisher_rank_tolerance": float("nan"),
            "numerical_fisher_rank": 0,
            "condition_number": float("inf"),
        }
    F_sym = 0.5 * (F_arr + F_arr.T)
    try:
        evals = np.linalg.eigvalsh(F_sym)
    except np.linalg.LinAlgError:
        evals = np.asarray([], dtype=float)
    if evals.size == 0 or not np.all(np.isfinite(evals)):
        return {
            "fisher_eigenvalues": evals.astype(float).tolist(),
            "fisher_rank_tolerance": float("nan"),
            "numerical_fisher_rank": 0,
            "condition_number": float("inf"),
        }
    scale = max(float(np.max(np.abs(evals))), 0.0)
    rank_tol = max(_FISHER_EIGENVALUE_UNDERFLOW_FLOOR, scale * _FISHER_RANK_RELATIVE_TOL)
    positive = evals > rank_tol
    positive_evals = evals[positive]
    condition = (
        float(np.max(positive_evals) / np.min(positive_evals))
        if positive_evals.size
        and np.min(positive_evals) > 0.0
        else float("inf")
    )
    return {
        "fisher_eigenvalues": evals.astype(float).tolist(),
        "fisher_rank_tolerance": float(rank_tol),
        "numerical_fisher_rank": int(np.count_nonzero(positive)),
        "condition_number": condition,
    }


def _diagnostic_metadata_aliases(
    derivative_metadata: dict[str, Any],
    rank_metadata: dict[str, Any],
    *,
    axes_singular: list[str] | tuple[str, ...],
    sigma_units: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return public top-level metadata aliases for generated CRLB tables."""
    state_axes = [str(axis) for axis in derivative_metadata.get("state_axes", [])]
    derivative_units = list(derivative_metadata.get("derivative_units", []))
    fisher_axis_units = list(derivative_metadata.get("fisher_axis_units", []))
    sigma_units_list = [str(unit) for unit in sigma_units]
    return {
        "state_axes": state_axes,
        "derivative_units_by_axis": {
            axis: derivative_units[idx]
            for idx, axis in enumerate(state_axes)
            if idx < len(derivative_units)
        },
        "sigma_units_by_axis": {
            axis: sigma_units_list[idx]
            for idx, axis in enumerate(state_axes)
            if idx < len(sigma_units_list)
        },
        "pixel_size_nm": derivative_metadata.get("pixel_size_nm"),
        "derivative_basis": derivative_metadata.get("derivative_basis"),
        "step_size_free": derivative_metadata.get("step_size_free"),
        "boundary_energy_fraction": derivative_metadata.get("boundary_energy_fraction"),
        "nyquist_band_fraction": derivative_metadata.get("nyquist_band_fraction"),
        "lateral_step_note": derivative_metadata.get("lateral_step_note"),
        "axial_derivative_mode": derivative_metadata.get("axial_derivative_mode"),
        "orientation_derivative_mode": derivative_metadata.get("orientation_derivative_mode"),
        "z_step_nm": derivative_metadata.get("z_step_nm"),
        "rotation_step_rad": derivative_metadata.get("rotation_step_rad"),
        "noise_variance_units": derivative_metadata.get("noise_variance_units"),
        "fisher_units": {
            axis: fisher_axis_units[idx]
            for idx, axis in enumerate(state_axes)
            if idx < len(fisher_axis_units)
        },
        "rank_tolerance": rank_metadata.get("fisher_rank_tolerance"),
        "eigenvalues": list(rank_metadata.get("fisher_eigenvalues", [])),
        "numerical_fisher_rank": rank_metadata.get("numerical_fisher_rank"),
        "condition_number": rank_metadata.get("condition_number"),
        "singular_axes": list(axes_singular),
    }
