"""Information-density and nuisance-adjusted Fisher diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from noise_contracts import AnalysisNoiseModel, IndependentPixelNoiseModel

from ._metadata_helpers import _derivative_unit, _variance_units
from .precision import apply_analysis_noise_precision, compute_fisher_density_maps_from_gradients
from .spectral_fisher import lateral_information_density_continuous

def _stack_named_maps(
    maps: dict[str, np.ndarray],
    *,
    expected_shape: tuple[int, int] | None = None,
    kind: str,
) -> tuple[list[str], np.ndarray, tuple[int, int]]:
    if not isinstance(maps, dict) or not maps:
        raise ValueError(f"{kind} must be a non-empty dict of image-shaped arrays.")
    names = list(maps.keys())
    arrays = []
    shape = expected_shape
    for name in names:
        arr = np.asarray(maps[name], dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{kind} map {name!r} must be 2D; got {arr.shape}.")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"All {kind} maps must have the same shape {shape}; "
                f"{name!r} has {arr.shape}."
            )
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{kind} map {name!r} must contain only finite values.")
        arrays.append(arr.reshape(-1))
    assert shape is not None
    return names, np.stack(arrays, axis=1), shape

def compute_nuisance_adjusted_fisher(
    parameter_derivative_maps: dict[str, np.ndarray],
    nuisance_basis_maps: dict[str, np.ndarray],
    noise_variance_map: IndependentPixelNoiseModel | AnalysisNoiseModel,
    *,
    rcond: float = 1e-12,
    signal_units: str = "model_signal",
    measurement_domain: str = "model_signal",
    noise_variance_units: str | None = None,
    parameter_units_by_axis: dict[str, str] | None = None,
    nuisance_units_by_basis: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Project structured-background nuisance directions out of Fisher information.

    The signed derivative maps are required. Squared Fisher-density maps cannot
    recover cross terms and are therefore not accepted as substitutes.

    Parameters
    ----------
    parameter_derivative_maps : dict[str, ndarray]
        Signed image derivatives for the parameters of interest. All maps must
        be finite 2D arrays with the same shape.
    nuisance_basis_maps : dict[str, ndarray]
        Signed image maps spanning the nuisance-background subspace. An empty
        dict leaves the raw Fisher matrix unchanged.
    noise_variance_map : ndarray or float
        Positive per-pixel variance, either scalar or shaped like the maps.
    rcond : float
        Relative cutoff passed to the pseudoinverse of the nuisance Fisher
        block.

    Returns
    -------
    dict
        Contains parameter and nuisance names, the raw Fisher matrix,
        nuisance Fisher matrix, cross Fisher block, information-loss matrix,
        adjusted Fisher matrix, per-parameter information-loss fractions, mean
        loss fraction, and nuisance rank.
    """
    parameter_names, G, shape = _stack_named_maps(
        parameter_derivative_maps,
        kind="parameter_derivative_maps",
    )
    if nuisance_basis_maps:
        nuisance_basis_names, B, _ = _stack_named_maps(
            nuisance_basis_maps,
            expected_shape=shape,
            kind="nuisance_basis_maps",
        )
    else:
        nuisance_basis_names = []
        B = np.zeros((G.shape[0], 0), dtype=float)

    if not isinstance(noise_variance_map, (IndependentPixelNoiseModel, AnalysisNoiseModel)):
        raise TypeError(
            "compute_nuisance_adjusted_fisher requires a typed Fisher noise "
            "likelihood. Use IndependentPixelNoiseModel for the current diagonal "
            "nuisance projection; structured AnalysisNoiseModel covariance needs a "
            "future covariance-aware nuisance projector."
        )
    noise_variance_map.require_safe_for_fisher(context="compute_nuisance_adjusted_fisher")

    parameter_arrays = tuple(
        np.asarray(parameter_derivative_maps[name], dtype=float)
        for name in parameter_names
    )
    weighted_parameter_arrays, _parameter_precision_metadata = apply_analysis_noise_precision(
        parameter_arrays,
        noise_variance_map,
        context="compute_nuisance_adjusted_fisher(parameter)",
    )
    WG = np.stack([arr.reshape(-1) for arr in weighted_parameter_arrays], axis=1)
    raw_fisher = G.T @ WG

    # Nuisance projection is also a Fisher quadratic form. Using a local
    # 1/variance vector here would reintroduce the same floor-support bug fixed
    # for localization and orientation CRLBs, and would silently diagonalize any
    # covariance model that fisher.precision already knows how to apply.
    if B.shape[1] == 0:
        nuisance_fisher = np.zeros((0, 0), dtype=float)
        cross_fisher = np.zeros((G.shape[1], 0), dtype=float)
        information_loss = np.zeros_like(raw_fisher)
        nuisance_rank = 0
    else:
        nuisance_arrays = tuple(
            np.asarray(nuisance_basis_maps[name], dtype=float)
            for name in nuisance_basis_names
        )
        weighted_nuisance_arrays, _nuisance_precision_metadata = apply_analysis_noise_precision(
            nuisance_arrays,
            noise_variance_map,
            context="compute_nuisance_adjusted_fisher(nuisance)",
        )
        WB = np.stack([arr.reshape(-1) for arr in weighted_nuisance_arrays], axis=1)
        nuisance_fisher = B.T @ WB
        cross_fisher = G.T @ WB
        singular_values = np.linalg.svd(nuisance_fisher, compute_uv=False)
        if singular_values.size == 0:
            nuisance_rank = 0
        else:
            rank_cutoff = float(rcond) * float(np.max(singular_values))
            nuisance_rank = int(np.count_nonzero(singular_values > rank_cutoff))
        nuisance_inverse = np.linalg.pinv(nuisance_fisher, rcond=float(rcond))
        information_loss = cross_fisher @ nuisance_inverse @ cross_fisher.T

    adjusted = raw_fisher - information_loss
    raw_fisher = 0.5 * (raw_fisher + raw_fisher.T)
    information_loss = 0.5 * (information_loss + information_loss.T)
    adjusted = 0.5 * (adjusted + adjusted.T)

    loss_fraction: dict[str, float] = {}
    for idx, name in enumerate(parameter_names):
        raw_diag = float(raw_fisher[idx, idx])
        loss_diag = float(information_loss[idx, idx])
        if raw_diag > 0.0 and np.isfinite(raw_diag):
            value = loss_diag / raw_diag
            loss_fraction[name] = float(min(1.0, max(0.0, value)))
        else:
            loss_fraction[name] = 0.0

    mean_loss = float(np.mean(list(loss_fraction.values()))) if loss_fraction else 0.0
    variance_units = str(noise_variance_units or _variance_units(str(signal_units)))
    parameter_units = {
        name: str((parameter_units_by_axis or {}).get(name, "parameter"))
        for name in parameter_names
    }
    nuisance_units = {
        name: str((nuisance_units_by_basis or {}).get(name, "nuisance_amplitude"))
        for name in nuisance_basis_names
    }
    parameter_derivative_units = {
        name: _derivative_unit(str(signal_units), parameter_units[name])
        for name in parameter_names
    }
    nuisance_basis_units = {
        name: _derivative_unit(str(signal_units), nuisance_units[name])
        for name in nuisance_basis_names
    }
    parameter_fisher_units = {
        name_i: {
            name_j: (
                f"{parameter_derivative_units[name_i]}*"
                f"{parameter_derivative_units[name_j]}/{variance_units}"
            )
            for name_j in parameter_names
        }
        for name_i in parameter_names
    }
    return {
        "parameter_names": parameter_names,
        "nuisance_basis_names": nuisance_basis_names,
        "raw_fisher": raw_fisher,
        "nuisance_fisher": nuisance_fisher,
        "cross_fisher": cross_fisher,
        "information_loss_matrix": information_loss,
        "adjusted_fisher": adjusted,
        "information_loss_fraction": loss_fraction,
        "mean_information_loss_fraction": mean_loss,
        "nuisance_rank": nuisance_rank,
        "unit_metadata": {
            "measurement_domain": str(measurement_domain),
            "signal_units": str(signal_units),
            "noise_variance_units": variance_units,
            "parameter_units_by_axis": parameter_units,
            "nuisance_units_by_basis": nuisance_units,
            "parameter_derivative_units_by_axis": parameter_derivative_units,
            "nuisance_basis_units_by_basis": nuisance_basis_units,
            "raw_fisher_units_by_entry": parameter_fisher_units,
            "adjusted_fisher_units_by_entry": parameter_fisher_units,
            "information_loss_units_by_entry": parameter_fisher_units,
            "nuisance_fisher_units_by_entry": {
                name_i: {
                    name_j: (
                        f"{nuisance_basis_units[name_i]}*"
                        f"{nuisance_basis_units[name_j]}/{variance_units}"
                    )
                    for name_j in nuisance_basis_names
                }
                for name_i in nuisance_basis_names
            },
            "cross_fisher_units_by_entry": {
                name_i: {
                    name_j: (
                        f"{parameter_derivative_units[name_i]}*"
                        f"{nuisance_basis_units[name_j]}/{variance_units}"
                    )
                    for name_j in nuisance_basis_names
                }
                for name_i in parameter_names
            },
        },
    }

def _positive_variance_map(
    noise_variance_map: np.ndarray | float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a strictly-positive variance map broadcast to ``shape``."""
    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        return np.full(shape, float(noise_variance_map), dtype=float)
    var = np.asarray(noise_variance_map, dtype=float)
    if var.shape != shape:
        raise ValueError(
            f"noise_variance_map shape {var.shape} does not match expected shape {shape}."
        )
    if np.any(~np.isfinite(var)):
        raise ValueError("noise_variance_map must contain only finite values.")
    if np.any(var <= 0.0):
        raise ValueError("noise_variance_map must contain only positive values.")
    return var

def compute_information_density_maps(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel,
    pixel_size_nm: float,
    *,
    mask_support: np.ndarray | None = None,
    substrate_background_contribution: np.ndarray | None = None,
    z_step_nm: float | None = None,
    rotation_renders: dict[str, np.ndarray] | None = None,
    rotation_step_rad: float | None = None,
    row_correlated_line_variance: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Expose the per-pixel Fisher-information summands used by the CRLB code.

    Returned maps are image-shaped tensors in the same pixel grid as the
    supplied central contrast image:

        Ix_info_map = (dI/dx0)^2 / sigma^2
        Iy_info_map = (dI/dy0)^2 / sigma^2
        Iz_info_map = (dC/dz)^2 / sigma^2              [when z_step_nm is set]
        Iomega_*_info_map = (dC/domega_*)^2 / sigma^2  [when renders supplied]

    The output also echoes ``noise_variance_map`` and ``mask_support``. When a
    mask support map is not supplied, support defaults to absolute contrast
    normalized to [0, 1] so downstream dataset audits can write observability
    maps without reimplementing the Fisher internals.
    """
    c = np.asarray(per_particle_contrast, dtype=float)
    if z_step_nm is None:
        if c.ndim != 2:
            raise ValueError(
                f"2D information maps expect (H, W) contrast; got shape {c.shape}."
            )
        centre = c
        dC_dz = None
    else:
        if c.ndim != 3 or c.shape[0] != 3:
            raise ValueError(
                f"3D information maps expect (3, H, W) stack; got shape {c.shape}."
            )
        if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
            raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")
        centre = c[1]
        dC_dz = (c[2] - c[0]) / (2.0 * z_step_nm)

    if not isinstance(noise_variance_map, (AnalysisNoiseModel, IndependentPixelNoiseModel)):
        raise TypeError(
            "compute_information_density_maps requires a typed Fisher noise "
            "likelihood. Raw diagonal arrays are ambiguous: they may be complete "
            "independent-pixel covariance or only the diagonal summary of a "
            "structured likelihood."
        )
    var = _positive_variance_map(noise_variance_map.diagonal_variance, centre.shape)

    # Fisher density is defined in the same likelihood basis as the CRLB:
    # g * (Sigma^-1 g). The independent-pixel case reduces to the historical
    # g^2 / variance maps, while row-correlated scan-line noise keeps the
    # off-diagonal covariance instead of silently diagonalizing it.
    maps: dict[str, np.ndarray] = {
        "noise_variance_map": var.astype(float),
    }
    density_grads: dict[str, np.ndarray] = {}

    abs_c = np.abs(centre)
    peak = float(abs_c.max()) if abs_c.size else 0.0
    if mask_support is None:
        maps["mask_support"] = (
            (abs_c / peak).astype(float)
            if peak > 0.0 else np.zeros_like(centre, dtype=float)
        )
    else:
        support = np.asarray(mask_support)
        if support.shape != centre.shape:
            raise ValueError(
                f"mask_support shape {support.shape} does not match image shape {centre.shape}."
            )
        maps["mask_support"] = support.astype(np.uint8 if support.dtype == bool else float)
    maps["contrast_contribution"] = centre.astype(float)
    if substrate_background_contribution is None:
        maps["substrate_background_contribution"] = np.zeros_like(centre, dtype=float)
    else:
        substrate = np.asarray(substrate_background_contribution, dtype=float)
        if substrate.shape != centre.shape:
            raise ValueError(
                "substrate_background_contribution shape "
                f"{substrate.shape} does not match image shape {centre.shape}."
            )
        maps["substrate_background_contribution"] = substrate

    if dC_dz is not None:
        density_grads["Iz_info_map"] = dC_dz

    if rotation_renders is not None:
        if (
            rotation_step_rad is None
            or not np.isfinite(rotation_step_rad)
            or rotation_step_rad <= 0.0
        ):
            raise ValueError(
                "rotation_step_rad must be positive when rotation_renders are supplied."
            )
        required = {
            "rx_minus", "rx_plus",
            "ry_minus", "ry_plus",
            "rz_minus", "rz_plus",
        }
        missing = required - set(rotation_renders)
        if missing:
            raise ValueError(
                f"rotation_renders missing required keys: {sorted(missing)}."
            )
        for axis, minus_key, plus_key in (
            ("x", "rx_minus", "rx_plus"),
            ("y", "ry_minus", "ry_plus"),
            ("z", "rz_minus", "rz_plus"),
        ):
            minus = np.asarray(rotation_renders[minus_key], dtype=float)
            plus = np.asarray(rotation_renders[plus_key], dtype=float)
            if minus.shape != centre.shape or plus.shape != centre.shape:
                raise ValueError(
                    "rotation render shapes must match the central contrast shape."
                )
            grad = (plus - minus) / (2.0 * rotation_step_rad)
            density_grads[f"Iomega_{axis}_info_map"] = grad

    if noise_variance_map.covariance_kind == "independent_pixels" and not density_grads:
        density_maps = lateral_information_density_continuous(
            centre,
            noise_variance_map.diagonal_variance,
            pixel_size_nm,
        )
    else:
        from .lateral import _lateral_coordinate_derivatives

        dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(centre, pixel_size_nm)
        density_grads = {
            "Ix_info_map": dI_dx0,
            "Iy_info_map": dI_dy0,
            **density_grads,
        }
        # Route covariance-bearing density channels through the shared Fisher
        # precision operator. This keeps saved density maps algebraically
        # consistent with scalar CRLBs on covariance structure and on numerical
        # variance-floor support.
        density_maps, _density_metadata = compute_fisher_density_maps_from_gradients(
            density_grads,
            noise_variance_map,
            row_correlated_line_variance=0.0,
            context="compute_information_density_maps",
        )
    maps.update(density_maps)
    return maps

__all__ = ['compute_nuisance_adjusted_fisher', 'compute_information_density_maps']
