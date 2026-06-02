"""Information-density and nuisance-adjusted Fisher diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._metadata_helpers import _derivative_unit, _variance_units

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
    noise_variance_map: np.ndarray | float,
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

    var = _positive_variance_map(noise_variance_map, shape).reshape(-1)
    weights = 1.0 / var

    WG = G * weights[:, None]
    raw_fisher = G.T @ WG

    if B.shape[1] == 0:
        nuisance_fisher = np.zeros((0, 0), dtype=float)
        cross_fisher = np.zeros((G.shape[1], 0), dtype=float)
        information_loss = np.zeros_like(raw_fisher)
        nuisance_rank = 0
    else:
        WB = B * weights[:, None]
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
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    *,
    mask_support: np.ndarray | None = None,
    substrate_background_contribution: np.ndarray | None = None,
    z_step_nm: float | None = None,
    rotation_renders: dict[str, np.ndarray] | None = None,
    rotation_step_rad: float | None = None,
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
    from .lateral import _lateral_coordinate_derivatives

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

    var = _positive_variance_map(noise_variance_map, centre.shape)
    inv_var = 1.0 / var
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(centre, pixel_size_nm)

    maps: dict[str, np.ndarray] = {
        "Ix_info_map": (dI_dx0 * dI_dx0 * inv_var).astype(float),
        "Iy_info_map": (dI_dy0 * dI_dy0 * inv_var).astype(float),
        "noise_variance_map": var.astype(float),
    }

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
        maps["Iz_info_map"] = (dC_dz * dC_dz * inv_var).astype(float)

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
            maps[f"Iomega_{axis}_info_map"] = (grad * grad * inv_var).astype(float)

    return maps

__all__ = ['compute_nuisance_adjusted_fisher', 'compute_information_density_maps']
