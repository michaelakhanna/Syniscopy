"""
fisher_diagnostic.py — Information-theoretic diagnostics for Syniscopy.

Compute Fisher-information and Cramér-Rao lower-bound (CRLB) diagnostics for
rendered particle images. Given a per-particle contrast image and pixel-wise
noise variance map, the two-dimensional localization CRLB follows from the 2x2
Fisher information matrix

    F_ij(x_0, y_0) = sum_r ( dI/dx_i )( dI/dx_j ) / var(r)       (Gaussian noise)

where the derivatives are of the per-particle intensity image with respect to
the particle's center coordinates (x_0, y_0) and var(r) is the pixel noise
variance. For a *stationary* PSF the intensity image is I(r; x_0, y_0) =
C(r - r_0), so dI/dx_0 = -dC/dx (and similarly for y), which we evaluate by
central finite differences on the rendered contrast image.

The CRLB on each coordinate is the corresponding diagonal element of F^{-1}:

    sigma_x^2 >= (F^{-1})_{00}
    sigma_y^2 >= (F^{-1})_{11}
    sigma_xy = sqrt( sigma_x^2 + sigma_y^2 )

The CRLB is an estimator-independent lower bound for unbiased localization
precision. Per-frame CRLB metadata complements temporal trackability by
separating annotation feasibility from the information content of the rendered
image itself. Examples with CRLB values above the intended localization scale
have limited signal for positive localization labels.

The module depends only on NumPy and does not import other Syniscopy modules, so
rendering and post-hoc metadata enrichment can use it without coupling.

Reference (classical):
    Thompson, Larson & Webb (2002), Biophys. J. 82, 2775 — closed-form
    CRLB for 2D Gaussian PSF with pixelated Poisson noise.
    Ober, Ram & Ward (2004), Biophys. J. 86, 1185 — Fisher-information
    framework for single-molecule localization microscopy.
"""

from __future__ import annotations

from typing import Any
import numpy as np


__all__ = [
    "compute_fisher_information",
    "compute_nuisance_adjusted_fisher",
    "compute_information_density_maps",
    "compute_fisher_information_3d",
    "compute_localization_crlb",
    "compute_localization_crlb_3d",
    "crlb_efficiency_ratio",
    "compare_modality_information_content",
    "compare_modality_axial_information_content",
    "compute_fisher_information_se3",
    "predict_se3_rank_from_symmetry",
    "predict_fused_se3_rank_from_symmetry",
    "compare_observed_and_predicted_se3_rank",
    "compute_localization_orientation_crlb",
    "compare_modality_orientation_crlb",
    "compare_modality_information_content_detected_quanta_normalized",
    "fit_power_law_scaling",
    "summarize_closed_form_scaling_checks",
    "compute_quanta_scaling_law",
    "check_budget_ordering_invariance",
    "check_budget_ranking_invariance",
    "compute_registration_degradation_curve",
    "compute_modality_fusion_crlb",
    "compute_loewner_dominance",
    "compute_optimal_time_allocation_crlb",
]


# Smallest denominator used when inverting the Fisher information matrix.
# Below this the matrix is considered singular and infinite CRLB is returned.
_FISHER_DET_EPS = 1e-30
_FISHER_VARIANCE_FLOOR = 1e-30

# Relative determinant floor for scale-dependent Fisher singularity checks.
_RELATIVE_DET_SINGULAR_TOL = 1e-18
_FISHER_EIGENVALUE_UNDERFLOW_FLOOR = np.finfo(float).tiny

# Residual norm tolerance for deciding whether a state axis lies in the
# numerical range of a singular Fisher matrix.
_FISHER_RANGE_RESIDUAL_TOL = 1e-8

# Armijo backtracking constants for the time-allocation Frank-Wolfe step.
_LINE_SEARCH_DESCENT_TOL = 1e-18
_LINE_SEARCH_SHRINK = 0.5
_LINE_SEARCH_ARMIJO_C = 1e-4
_LINE_SEARCH_MAX_STEPS = 40


def _spatial_gradients(contrast: np.ndarray, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Central-difference spatial gradients of a 2D contrast image.

    Returns (dC_dx, dC_dy) both in units of [contrast] / nm.

    The gradient is taken with `np.gradient`, which uses second-order-accurate
    central differences at interior points and first-order one-sided differences
    at the boundary. This matches the standard Fisher-info treatment where the
    estimator considers all pixels in the support.
    """
    contrast = np.asarray(contrast, dtype=float)
    if contrast.ndim != 2:
        raise ValueError(
            f"compute_*_crlb expects a 2D contrast image; got shape {contrast.shape}."
        )
    if min(contrast.shape) < 2:
        raise ValueError(
            "compute_*_crlb requires at least two pixels along each image axis; "
            f"got shape {contrast.shape}."
        )
    if not np.all(np.isfinite(contrast)):
        raise ValueError("contrast image must contain only finite values.")
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0.0:
        raise ValueError(f"pixel_size_nm must be positive; got {pixel_size_nm}.")

    # np.gradient returns (d/dy, d/dx) when given a 2D array with indexing=(i, j).
    dC_dy, dC_dx = np.gradient(contrast, pixel_size_nm)
    return dC_dx, dC_dy


def _lateral_coordinate_derivatives(
    contrast: np.ndarray,
    pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate derivatives ``dI/dx0`` and ``dI/dy0`` for stationary shifts."""
    dC_dx, dC_dy = _spatial_gradients(contrast, pixel_size_nm)
    return -dC_dx, -dC_dy


def compute_fisher_information(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
) -> np.ndarray:
    """
    Build the 2x2 Fisher information matrix F for localization from a
    per-particle contrast image.

    Parameters
    ----------
    per_particle_contrast : 2D array
        Per-particle intensity image C(r) (any real units). Typically the
        output of ``ImagingModel.compute_per_particle_contrast`` for a single
        particle, evaluated on the native detector grid.
    noise_variance_map : 2D array or scalar float
        Pixel-wise variance of the observed image, var(r). For a Gaussian
        detector-noise model this is the variance of the readout; for a
        shot-noise-limited image this is approximately the background
        reference intensity |E_ref|^2. A scalar broadcasts to every pixel.
    pixel_size_nm : float
        Detector pixel pitch in nanometres (used to convert index-space
        gradients to per-nm gradients).

    Returns
    -------
    F : (2, 2) array
        Fisher information matrix with
            F[0, 0] = sum_r (dI/dx0)^2 / var(r),  [units: 1 / nm^2]
            F[1, 1] = sum_r (dI/dy0)^2 / var(r),
            F[0, 1] = F[1, 0] = sum_r (dI/dx0)(dI/dy0) / var(r).

    Notes
    -----
    The Fisher information as computed here is appropriate for a *Gaussian*
    pixel-noise model with known variance. For a pure-Poisson noise model the
    caller should pass ``noise_variance_map = I_total(r)`` (the *observed*
    intensity) rather than the background-only variance; the formula is
    otherwise identical in the high-photon / small-contrast limit.
    """
    contrast = np.asarray(per_particle_contrast, dtype=float)
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(contrast, pixel_size_nm)

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
        F_xx = float(np.sum(dI_dx0 * dI_dx0) * inv_var)
        F_yy = float(np.sum(dI_dy0 * dI_dy0) * inv_var)
        F_xy = float(np.sum(dI_dx0 * dI_dy0) * inv_var)
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != contrast.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match contrast shape "
                f"{contrast.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
        F_xx = float(np.sum(dI_dx0 * dI_dx0 * inv_var))
        F_yy = float(np.sum(dI_dy0 * dI_dy0 * inv_var))
        F_xy = float(np.sum(dI_dx0 * dI_dy0 * inv_var))

    return np.array([[F_xx, F_xy], [F_xy, F_yy]], dtype=float)


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


def compute_localization_crlb(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    *,
    return_density: bool = False,
) -> dict[str, Any]:
    """
    Per-particle Cramér-Rao lower bound on (x, y) localization error.

    Parameters
    ----------
    per_particle_contrast, noise_variance_map, pixel_size_nm :
        See :func:`compute_fisher_information`.
    return_density : bool, default False
        If True, include per-pixel Fisher information density maps in the result.

    Returns
    -------
    result : dict
        Keys:
          - ``sigma_x_nm``   : CRLB on x-localization, nanometres.
          - ``sigma_y_nm``   : CRLB on y-localization, nanometres.
          - ``sigma_xy_nm``  : Total 2D bound, sqrt(sigma_x^2 + sigma_y^2).
          - ``fisher_matrix`` : The 2x2 Fisher information matrix (array).
          - ``fisher_det``   : Determinant of F (pre-inversion).
          - ``singular``     : True if F was (effectively) singular and the
                               bounds were set to +inf.
          - ``information_density_maps`` : Present only when ``return_density``
                               is True.

    A singular Fisher matrix arises for an image with no spatial gradient
    (constant across the field, e.g. a particle with zero contrast), in
    which case localization is information-theoretically impossible and
    the CRLB is +inf — a useful signal to upstream trackability code.
    """
    F = compute_fisher_information(
        per_particle_contrast, noise_variance_map, pixel_size_nm,
    )
    det_F = float(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])

    if not np.all(np.isfinite(F)) or not np.isfinite(det_F):
        result = {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "fisher_matrix": F,
            "fisher_det": det_F,
            "singular": True,
            "rank": 0,
            "axes_singular": ["x", "y"],
        }
        if return_density:
            result["information_density_maps"] = compute_information_density_maps(
                per_particle_contrast,
                noise_variance_map,
                pixel_size_nm,
        )
        return result

    axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F)
    sigma_x = axis_sigmas[0]
    sigma_y = axis_sigmas[1]
    singular = axis_singular[0] or axis_singular[1]
    axes_singular = [
        axis for axis, is_singular in zip(("x", "y"), axis_singular)
        if is_singular
    ]
    sigma_xy = (
        float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
        if not singular and np.isfinite(sigma_x) and np.isfinite(sigma_y)
        else float("inf")
    )

    result = {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_xy_nm": sigma_xy,
        "fisher_matrix": F,
        "fisher_det": det_F,
        "singular": singular,
        "rank": int(2 - sum(bool(flag) for flag in axis_singular[:2])),
        "axes_singular": axes_singular,
    }
    if return_density:
        result["information_density_maps"] = compute_information_density_maps(
            per_particle_contrast,
            noise_variance_map,
            pixel_size_nm,
        )
    return result


# ---------------------------------------------------------------------------
# 3D localization: axial precision from a three-plane contrast stack.
# The caller supplies frames at z0-dz, z0, and z0+dz in the same detector-domain
# convention as the 2D contrast image.
# ---------------------------------------------------------------------------


def compute_fisher_information_3d(
    per_particle_contrast_stack: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
) -> np.ndarray:
    """
    Build the 3x3 Fisher information matrix F for (x, y, z) localization.

    Parameters
    ----------
    per_particle_contrast_stack : (3, H, W) array
        Three-plane axial neighbourhood of the per-particle contrast image,
        in order ``[C(z - dz), C(z), C(z + dz)]`` with the ``+dz`` offset
        equal to ``z_step_nm``. The middle plane (index 1) is treated as the
        in-focus reference; the outer planes feed the axial central-difference
        derivative ``dC/dz = (C[2] - C[0]) / (2 * z_step_nm)``.
    noise_variance_map : 2D array (H, W) or scalar float
        Pixel-wise variance of the observed image at the in-focus plane. If
        the noise statistics are z-dependent (e.g. shot-noise on a defocus-
        attenuated background), the caller should supply the variance at z =
        z_in-focus; the bound is conservative because a z-step in either
        direction has the same noise floor.
    pixel_size_nm : float
        Detector pixel pitch in nanometres.
    z_step_nm : float
        Axial spacing between the three planes, in nanometres. Must be > 0.

    Returns
    -------
    F : (3, 3) array
        Symmetric Fisher information matrix with ordering [x, y, z]. All
        entries have units of 1 / nm^2.

    Notes
    -----
    The decision to take ``dC/dz`` from a *symmetric* finite difference
    (rather than a one-sided forward difference) is deliberate: it is
    second-order accurate in ``z_step_nm`` and shares the truncation order
    of the in-plane gradients, so the three Fisher components are on equal
    numerical footing. The cost is two additional rendered z-planes per
    particle (``-dz`` and ``+dz``), which the caller controls.
    """
    stack = np.asarray(per_particle_contrast_stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] != 3:
        raise ValueError(
            f"per_particle_contrast_stack must have shape (3, H, W); got {stack.shape}."
        )
    if not np.all(np.isfinite(stack)):
        raise ValueError("per_particle_contrast_stack must contain only finite values.")
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")

    centre = stack[1]
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(centre, pixel_size_nm)
    dC_dz = (stack[2] - stack[0]) / (2.0 * z_step_nm)

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
        # broadcast: scalar * elementwise sums.
        F_xx = float(np.sum(dI_dx0 * dI_dx0) * inv_var)
        F_yy = float(np.sum(dI_dy0 * dI_dy0) * inv_var)
        F_zz = float(np.sum(dC_dz * dC_dz) * inv_var)
        F_xy = float(np.sum(dI_dx0 * dI_dy0) * inv_var)
        F_xz = float(np.sum(dI_dx0 * dC_dz) * inv_var)
        F_yz = float(np.sum(dI_dy0 * dC_dz) * inv_var)
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != centre.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match contrast slice "
                f"shape {centre.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
        F_xx = float(np.sum(dI_dx0 * dI_dx0 * inv_var))
        F_yy = float(np.sum(dI_dy0 * dI_dy0 * inv_var))
        F_zz = float(np.sum(dC_dz * dC_dz * inv_var))
        F_xy = float(np.sum(dI_dx0 * dI_dy0 * inv_var))
        F_xz = float(np.sum(dI_dx0 * dC_dz * inv_var))
        F_yz = float(np.sum(dI_dy0 * dC_dz * inv_var))

    return np.array(
        [[F_xx, F_xy, F_xz],
         [F_xy, F_yy, F_yz],
         [F_xz, F_yz, F_zz]],
        dtype=float,
    )


def compute_localization_crlb_3d(
    per_particle_contrast_stack: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
) -> dict[str, Any]:
    """
    Per-particle Cramér-Rao lower bound on (x, y, z) localization error.

    Parameters
    ----------
    per_particle_contrast_stack, noise_variance_map, pixel_size_nm, z_step_nm :
        See :func:`compute_fisher_information_3d`.

    Returns
    -------
    result : dict with keys
        - ``sigma_x_nm``    : CRLB on x-localization, nanometres.
        - ``sigma_y_nm``    : CRLB on y-localization, nanometres.
        - ``sigma_z_nm``    : CRLB on z (axial) localization, nanometres.
        - ``sigma_xyz_nm``  : Total 3D bound, sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2).
        - ``sigma_xy_nm``   : Lateral-only bound, sqrt(sigma_x^2 + sigma_y^2).
        - ``fisher_matrix`` : The 3x3 Fisher information matrix (array).
        - ``fisher_det``    : Determinant of F (pre-inversion).
        - ``singular``      : True if the full 3x3 Fisher matrix was
                              effectively singular. Lateral bounds may still
                              be finite in the axial-degeneracy case.
        - ``z_singular`` / ``axially_singular``: True when the z-axis bound is
                              singular. ``only_axially_singular`` is True for
                              the common case where z is singular but the
                              lateral x/y block remains finite.

    Axial bound
    -----------
    The 2D CRLB from :func:`compute_localization_crlb` covers image-plane
    localization. The 3D bound adds the axial coordinate by rendering two
    additional z-planes per particle for the central difference while reusing
    the same noise model.
    """
    F = compute_fisher_information_3d(
        per_particle_contrast_stack, noise_variance_map, pixel_size_nm, z_step_nm,
    )
    if np.any(~np.isfinite(F)):
        det_F = float("nan")
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_z_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "sigma_xyz_nm": float("inf"),
            "fisher_matrix": F,
            "fisher_det": det_F,
            "singular": True,
            "xy_singular": True,
            "z_singular": True,
            "axially_singular": True,
            "only_axially_singular": False,
            "rank": 0,
            "axes_singular": ["x", "y", "z"],
        }
    det_F = float(np.linalg.det(F))
    if not np.isfinite(det_F):
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_z_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "sigma_xyz_nm": float("inf"),
            "fisher_matrix": F,
            "fisher_det": det_F,
            "singular": True,
            "axially_singular": False,
            "rank": 0,
            "axes_singular": ["x", "y", "z"],
        }

    F_sym = 0.5 * (F + F.T)
    try:
        eigvals = np.linalg.eigvalsh(F_sym)
    except np.linalg.LinAlgError:
        eigvals = np.asarray([float("nan")])
    eig_scale = max(float(np.max(np.abs(eigvals))) if eigvals.size else 0.0, 0.0)
    negative_tol = max(_FISHER_DET_EPS, eig_scale * _RELATIVE_DET_SINGULAR_TOL)
    if (
        det_F < -negative_tol
        or not np.all(np.isfinite(eigvals))
        or float(np.min(eigvals)) < -negative_tol
    ):
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_z_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "sigma_xyz_nm": float("inf"),
            "fisher_matrix": F,
            "fisher_det": det_F,
            "singular": True,
            "xy_singular": True,
            "z_singular": True,
            "axially_singular": True,
            "only_axially_singular": False,
        }

    axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F_sym)
    sigma_x, sigma_y, sigma_z = axis_sigmas
    xy_singular = axis_singular[0] or axis_singular[1]
    singular = any(axis_singular)
    axes_singular = [
        axis for axis, is_singular in zip(("x", "y", "z"), axis_singular)
        if is_singular
    ]
    sigma_xy = (
        float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
        if not xy_singular and np.isfinite(sigma_x) and np.isfinite(sigma_y)
        else float("inf")
    )
    sigma_xyz = (
        float(np.sqrt(sigma_x ** 2 + sigma_y ** 2 + sigma_z ** 2))
        if not singular
        and np.isfinite(sigma_x)
        and np.isfinite(sigma_y)
        and np.isfinite(sigma_z)
        else float("inf")
    )

    return {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_z_nm": sigma_z,
        "sigma_xy_nm": sigma_xy,
        "sigma_xyz_nm": sigma_xyz,
        "fisher_matrix": F,
        "fisher_det": det_F,
        "singular": singular,
        "rank": int(3 - sum(bool(flag) for flag in axis_singular[:3])),
        "axes_singular": axes_singular,
        "xy_singular": bool(xy_singular),
        "z_singular": bool(axis_singular[2]),
        "axially_singular": bool(axis_singular[2]),
        "only_axially_singular": bool(axis_singular[2] and not xy_singular),
    }


def crlb_efficiency_ratio(
    measured_sigma_nm: float,
    crlb_sigma_nm: float,
) -> float:
    """
    Fisher efficiency of an estimator: CRLB / measured.

    Values in (0, 1] are valid; 1.0 means the estimator saturates the bound
    (optimal unbiased estimator). Values > 1 are not physically impossible —
    they indicate a *biased* estimator, and the CRLB assumes unbiasedness.
    Values close to 0 indicate a poor estimator.

    Special cases:
      - If ``crlb_sigma_nm`` is 0 or inf, returns 0.0 (undefined efficiency
        under a degenerate bound).
      - If ``measured_sigma_nm`` is 0 (or negative), returns +inf.
    """
    if crlb_sigma_nm <= 0.0 or not np.isfinite(crlb_sigma_nm):
        return 0.0
    if measured_sigma_nm <= 0.0:
        return float("inf")
    return float(crlb_sigma_nm / measured_sigma_nm)


def _resolve_modality_scalar(
    value: float | dict[str, float],
    modality: str,
    name: str,
    *,
    positive: bool = True,
) -> float:
    """Resolve a scalar or per-modality scalar mapping for one modality."""
    raw = value[modality] if isinstance(value, dict) else value
    out = float(raw)
    if not np.isfinite(out) or (positive and out <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name}[{modality!r}] must be {qualifier}; got {raw!r}.")
    return out


def _resolve_modality_scalar_map(
    value: float | dict[str, float],
    modalities,
    name: str,
    override: dict[str, float] | None = None,
    *,
    positive: bool = True,
) -> dict[str, float]:
    """Resolve one scalar per modality, accepting a scalar, mapping, or override."""
    source = override if override is not None else value
    if isinstance(source, dict):
        missing = set(modalities) - set(source)
        if missing:
            raise ValueError(
                f"{name} mapping is missing modality key(s): {sorted(missing)!r}."
            )
    return {
        modality: _resolve_modality_scalar(
            source,
            modality,
            name,
            positive=positive,
        )
        for modality in modalities
    }


# ---------------------------------------------------------------------------
# Cross-modality information-content diagnostic.
#
# Each modality supplies a per-particle contrast image and a noise-variance
# map for the same underlying particle. The routine computes comparable Fisher
# information / CRLB values in shared physical units.
# ---------------------------------------------------------------------------


def compare_modality_information_content(
    contrast_by_modality: dict[str, np.ndarray],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float | None = None,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""
    Order imaging modalities by the physical information they deliver about the
    position of the same underlying particle.

    For every modality the caller supplies a per-particle contrast image (or a
    three-plane axial stack if ``z_step_nm`` is given) and its noise variance,
    this function computes the modality's CRLB on ``(x, y)`` [and ``z``] and
    returns a single comparison table plus the lowest-bound modality.

    Parameters
    ----------
    contrast_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> per-particle contrast image``. For the 2D
        comparison each value is a ``(H, W)`` array; for the 3D comparison
        (``z_step_nm`` supplied) each value is a ``(3, H, W)`` three-plane stack
        in the same convention as :func:`compute_localization_crlb_3d`.
    noise_variance_by_modality : dict[str, ndarray | float]
        Mapping ``modality_name -> pixel-wise variance`` (or scalar) for the
        same modalities. Must have the same keys as ``contrast_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies per-modality detector
        pitches for configured/native-profile comparisons.
    z_step_nm : float or None, default None
        If None: 2D comparison. If a positive float: 3D comparison using the
        3x3 Fisher machinery of :func:`compute_localization_crlb_3d`.

    Returns
    -------
    result : dict with keys
        - ``per_modality`` : dict ``modality -> sub-dict`` containing
          ``sigma_xy_nm``, ``fisher_det``, ``singular``, and (if 3D)
          ``sigma_z_nm``, ``sigma_xyz_nm``, ``axially_singular``. These are the
          outputs of the underlying 2D/3D CRLB routines, preserved verbatim.
        - ``ordering_xy`` : list of ``(modality, sigma_xy_nm)`` sorted ascending
          (lowest-bound first). Singular modalities end up last with ``+inf``.
          ``ranking_xy`` is retained as an alias for existing callers.
        - ``best_modality_xy`` : the key of the modality with the smallest
          lateral CRLB. ``None`` if every modality is singular.
        - ``relative_sigma_xy`` : dict ``modality -> sigma_xy_nm / best_sigma_xy``.
          A value of 1.0 marks the lowest-bound profile; larger values are
          relative to that profile.
        - ``frames_to_match_best_xy`` : dict ``modality -> (sigma / sigma_best)^2``.
          Because localization variance scales as 1/F and Fisher information
          adds linearly across independent frames, this is the number of frames
          of ``modality`` that would be needed to match the single-frame CRLB
          of the lowest-bound modality under the supplied profile.
        - ``ordering_xyz`` (only 3D): list of ``(modality, sigma_xyz_nm)``
          sorted ascending. ``ranking_xyz`` is retained as an alias.
        - ``best_modality_xyz`` (only 3D): argmin modality for sigma_xyz.
        - ``relative_sigma_xyz`` (only 3D): dict ``modality -> sigma_xyz / best``.

    Notes on comparability
    ----------------------
    The reported bound is conditional on the supplied contrast images and noise
    variances. Cross-modality orderings are comparable when each modality is
    rendered under matched profile/noise assumptions and the same sample truth.
    The Syniscopy rendering pipeline arranges this when the same particle
    configuration is routed through multiple ``ImagingModel`` instances.

    Ties are broken by the order in which modalities appear in the input dict.
    A singular modality (zero contrast gradient) sorts last and is assigned
    an infinite ``frames_to_match_best`` — it cannot catch up by averaging.
    """

    if set(contrast_by_modality.keys()) != set(noise_variance_by_modality.keys()):
        raise ValueError(
            "contrast_by_modality and noise_variance_by_modality must share keys; "
            f"missing from contrast: "
            f"{set(noise_variance_by_modality) - set(contrast_by_modality)}; "
            f"missing from noise: "
            f"{set(contrast_by_modality) - set(noise_variance_by_modality)}."
        )
    if not contrast_by_modality:
        raise ValueError("contrast_by_modality is empty; nothing to compare.")
    if z_step_nm is not None and (not np.isfinite(z_step_nm) or z_step_nm <= 0.0):
        raise ValueError(f"z_step_nm must be positive when supplied; got {z_step_nm}.")

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        contrast_by_modality.keys(),
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, contrast in contrast_by_modality.items():
        noise = noise_variance_by_modality[modality]
        px = pixel_sizes[modality]
        if z_step_nm is None:
            res = compute_localization_crlb(contrast, noise, px)
        else:
            res = compute_localization_crlb_3d(contrast, noise, px, z_step_nm)
        per_modality[modality] = res

    # Lateral ordering (always valid: 2D and 3D both report sigma_xy_nm).
    def _xy_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int]:
        modality, res = item
        sigma = res["sigma_xy_nm"]
        # sort order stable in input-dict order for ties
        idx = list(contrast_by_modality.keys()).index(modality)
        return (float(sigma), idx)

    ordered_xy = sorted(per_modality.items(), key=_xy_key)
    ranking_xy = [(m, float(r["sigma_xy_nm"])) for m, r in ordered_xy]

    best_modality_xy: str | None
    best_sigma_xy = ordered_xy[0][1]["sigma_xy_nm"] if ordered_xy else float("inf")
    if np.isfinite(best_sigma_xy) and best_sigma_xy > 0.0:
        best_modality_xy = ordered_xy[0][0]
        relative_sigma_xy = {
            m: float(r["sigma_xy_nm"]) / float(best_sigma_xy)
            for m, r in per_modality.items()
        }
        frames_to_match_best_xy = {
            m: (
                float("inf")
                if not np.isfinite(float(r["sigma_xy_nm"]))
                else (float(r["sigma_xy_nm"]) / float(best_sigma_xy)) ** 2
            )
            for m, r in per_modality.items()
        }
    else:
        # Every modality singular: no meaningful ordering.
        best_modality_xy = None
        relative_sigma_xy = {m: float("inf") for m in per_modality}
        frames_to_match_best_xy = {m: float("inf") for m in per_modality}

    out: dict[str, Any] = {
        "per_modality": per_modality,
        "ordering_xy": ranking_xy,
        "ranking_xy": ranking_xy,
        "best_modality_xy": best_modality_xy,
        "relative_sigma_xy": relative_sigma_xy,
        "frames_to_match_best_xy": frames_to_match_best_xy,
        "pixel_size_nm_by_modality": pixel_sizes,
    }

    if z_step_nm is not None:
        # Full-3D ordering.
        def _xyz_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int]:
            modality, res = item
            sigma = res.get("sigma_xyz_nm", float("inf"))
            idx = list(contrast_by_modality.keys()).index(modality)
            return (float(sigma), idx)

        ordered_xyz = sorted(per_modality.items(), key=_xyz_key)
        ranking_xyz = [
            (m, float(r.get("sigma_xyz_nm", float("inf"))))
            for m, r in ordered_xyz
        ]
        best_sigma_xyz = ordered_xyz[0][1].get("sigma_xyz_nm", float("inf"))
        if np.isfinite(best_sigma_xyz) and best_sigma_xyz > 0.0:
            best_modality_xyz = ordered_xyz[0][0]
            relative_sigma_xyz = {
                m: float(r.get("sigma_xyz_nm", float("inf"))) / float(best_sigma_xyz)
                for m, r in per_modality.items()
            }
        else:
            best_modality_xyz = None
            relative_sigma_xyz = {m: float("inf") for m in per_modality}
        out["ordering_xyz"] = ranking_xyz
        out["ranking_xyz"] = ranking_xyz
        out["best_modality_xyz"] = best_modality_xyz
        out["relative_sigma_xyz"] = relative_sigma_xyz

    return out


# ---------------------------------------------------------------------------
# SE(3) extension: joint translation + orientation Cramér-Rao bound.
#
# The 3D CRLB of compute_localization_crlb_3d answers "how tightly can a
# point particle be located in (x, y, z)?". For a *composite* particle whose
# orientation is itself a state variable (dimers, rod stacks, water-like
# bent triatomics, ...) the localisation question is six-dimensional: three
# translation degrees of freedom plus three Euler-style body-fixed rotation
# degrees of freedom. The Fisher information matrix is then 6x6 with
# ordering [x, y, z, omega_x, omega_y, omega_z], and inverting it yields a
# joint estimator-bound on translation *and* orientation.
#
# The implementation here is pure-function and finite-difference based: the
# caller supplies nine pre-rendered per-particle contrast images covering
# the SE(3)-adjacent perturbations and we assemble the 6x6 Fisher matrix.
# In-plane gradients dC/dx and dC/dy are taken from the centre image by
# central finite differences (no extra renders needed), as in the 2D and 3D
# routines above. The extra renders are: two for axial translation, and two
# each for the three rotational degrees of freedom (six total).
# ---------------------------------------------------------------------------


# Diagonal Fisher-information threshold below which an SE(3) state axis is
# treated as unobservable and excluded from the degraded sub-block inverse.
_FISHER_SE3_EPS = 1e-30
_FISHER_SE3_AXIS_RELATIVE_TOL = 1e-12


def _validate_se3_renders(
    renders: dict[str, np.ndarray],
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """
    Check that ``renders`` contains the nine required keys with consistent
    shape. The "centre" render is treated as the in-focus, in-orientation
    reference; every other render is a one-axis perturbation.
    """
    required_keys = {
        "centre",
        "z_minus", "z_plus",
        "rx_minus", "rx_plus",
        "ry_minus", "ry_plus",
        "rz_minus", "rz_plus",
    }
    missing = required_keys - set(renders)
    if missing:
        raise ValueError(
            f"compute_fisher_information_se3 requires keys "
            f"{sorted(required_keys)}; missing: {sorted(missing)}."
        )
    centre = np.asarray(renders["centre"], dtype=float)
    if centre.ndim != 2:
        raise ValueError(
            f"renders['centre'] must be a 2D image; got shape {centre.shape}."
        )
    if not np.all(np.isfinite(centre)):
        raise ValueError("renders['centre'] must contain only finite values.")
    if expected_shape is not None and centre.shape != expected_shape:
        raise ValueError(
            f"renders['centre'] shape {centre.shape} does not match expected "
            f"{expected_shape}."
        )
    for k in required_keys - {"centre"}:
        arr = np.asarray(renders[k], dtype=float)
        if arr.shape != centre.shape:
            raise ValueError(
                f"renders['{k}'] shape {arr.shape} does not match centre shape "
                f"{centre.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"renders['{k}'] must contain only finite values.")
    return centre.shape


def compute_fisher_information_se3(
    renders: dict[str, np.ndarray],
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
    rotation_step_rad: float,
) -> np.ndarray:
    """
    Build the 6x6 SE(3) Fisher information matrix for a composite particle.

    The state vector is ``[x, y, z, omega_x, omega_y, omega_z]`` where the
    omegas are infinitesimal body-fixed rotations about the principal axes
    in the same fixed convention used to render the perturbed images.

    Parameters
    ----------
    renders : dict[str, ndarray]
        Per-pose per-particle contrast images, keyed by:
            ``centre``                — reference pose; used for dC/dx, dC/dy
                                        via central differences in pixel space.
            ``z_minus``, ``z_plus``    — particle translated by ``∓z_step_nm``
                                        in z; used for dC/dz.
            ``rx_minus``, ``rx_plus``  — composite rotated by ``∓rotation_step_rad``
                                        about body-fixed x; used for dC/dω_x.
            ``ry_minus``, ``ry_plus``  — analog about body-fixed y.
            ``rz_minus``, ``rz_plus``  — analog about body-fixed z.
        All must be 2D arrays of identical shape.
    noise_variance_map : 2D array or scalar float
        Pixel-wise variance of the observed image at the centre pose.
    pixel_size_nm : float
        Detector pixel pitch in nanometres. Lateral gradients are reported
        in units of ``[contrast] / nm`` and converted to coordinate
        derivatives under the stationary-shift convention.
    z_step_nm : float
        Axial translation step used to render ``z_minus``/``z_plus``;
        ``dC/dz = (renders['z_plus'] - renders['z_minus']) / (2 * z_step_nm)``.
    rotation_step_rad : float
        Rotation step (radians) used to render the six rotation perturbations;
        ``dC/dω_i = (renders['ri_plus'] - renders['ri_minus']) / (2 * rotation_step_rad)``.

    Returns
    -------
    F : (6, 6) array
        Symmetric Fisher information matrix with state ordering
        ``[x, y, z, omega_x, omega_y, omega_z]``. Translation entries have
        units of ``1 / nm^2``; rotation entries have units of ``1 / rad^2``;
        the off-diagonal coupling entries have units of ``1 / (nm * rad)``.
        Mixed units across the 6x6 matrix are *correct* — the inverse
        produces sigma_x in nm and sigma_omega in rad, both physical.

    Notes
    -----
    The decision to take in-plane gradients from the centre image only
    (rather than from a (-x, +x) and (-y, +y) translation pair) is
    deliberate and matches the 3D variant: under the standard stationary-PSF
    assumption ``I(r; x_0, y_0) = C(r - r_0)``, the coordinate derivatives
    are the negative centre-image spatial gradients. Re-rendering the
    composite at translated x, y would give a numerically equivalent estimate
    at strictly higher cost.
    """
    _validate_se3_renders(
        renders,
        expected_shape=None,
    )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")
    if not np.isfinite(rotation_step_rad) or rotation_step_rad <= 0.0:
        raise ValueError(
            f"rotation_step_rad must be positive; got {rotation_step_rad}."
        )

    centre = np.asarray(renders["centre"], dtype=float)
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(centre, pixel_size_nm)
    dC_dz = (np.asarray(renders["z_plus"], dtype=float)
             - np.asarray(renders["z_minus"], dtype=float)) / (2.0 * z_step_nm)
    dC_dwx = (np.asarray(renders["rx_plus"], dtype=float)
              - np.asarray(renders["rx_minus"], dtype=float)) / (2.0 * rotation_step_rad)
    dC_dwy = (np.asarray(renders["ry_plus"], dtype=float)
              - np.asarray(renders["ry_minus"], dtype=float)) / (2.0 * rotation_step_rad)
    dC_dwz = (np.asarray(renders["rz_plus"], dtype=float)
              - np.asarray(renders["rz_minus"], dtype=float)) / (2.0 * rotation_step_rad)

    grads = (dI_dx0, dI_dy0, dC_dz, dC_dwx, dC_dwy, dC_dwz)

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
        scale = inv_var
        F = np.empty((6, 6), dtype=float)
        for i in range(6):
            for j in range(i, 6):
                v = float(np.sum(grads[i] * grads[j])) * scale
                F[i, j] = v
                F[j, i] = v
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != centre.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match centre "
                f"render shape {centre.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
        F = np.empty((6, 6), dtype=float)
        for i in range(6):
            for j in range(i, 6):
                v = float(np.sum(grads[i] * grads[j] * inv_var))
                F[i, j] = v
                F[j, i] = v
    return F


def compute_localization_orientation_crlb(
    renders: dict[str, np.ndarray],
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
    rotation_step_rad: float,
) -> dict[str, Any]:
    """
    Joint translation + orientation Cramér-Rao bound for a composite particle.

    Returns
    -------
    result : dict with keys
        - ``sigma_x_nm``, ``sigma_y_nm``, ``sigma_z_nm`` : translation CRLB
          on each axis, in nanometres.
        - ``sigma_xyz_nm``                              : sqrt of summed
          translation variances.
        - ``sigma_omega_x_rad``, ``sigma_omega_y_rad``,
          ``sigma_omega_z_rad``                         : orientation CRLB
          on each body-fixed axis, in radians.
        - ``sigma_omega_total_rad``                     : sqrt of summed
          orientation variances over the *observable* rotation axes (i.e.
          axes with finite per-axis CRLB). For fully orientation-observable
          particles this is the standard isotropic aggregate orientation-
          precision summary; for partial-rank
          composites (e.g. a body-axis-symmetric dimer with omega_x
          singular) this is the joint precision over the observable
          rotation subspace, with the unobservable axes reported separately
          in ``axes_singular``. Returns +inf only when every rotation axis
          is singular (e.g. a perfect sphere). This is *not* a geodesic on
          SO(3); for small angles the small-angle Lie-algebra norm and the
          geodesic norm coincide to first order, which is the regime in
          which a finite-difference Fisher matrix is meaningful.
        - ``fisher_matrix``                             : the 6x6 matrix.
        - ``fisher_det``                                : determinant.
        - ``singular``                                  : True if the full
          6x6 was singular and *some* coordinates' bounds are +inf.
        - ``rank``                                      : numerical rank of
          the estimable state-axis support after singular axes are removed.
        - ``numerical_fisher_rank``                     : raw eigenvalue-based
          rank of the symmetrized Fisher matrix under the degradation
          criterion before axis-estimability clipping.
        - ``axes_singular``                             : list of state-axis
          names whose bound is +inf (e.g. ``['omega_z']`` for an
          axially-symmetric particle whose z-rotation is unobservable).

    Graceful degradation
    --------------------
    Composite particles with axial symmetry have an unobservable rotation
    DOF: the z-axis rotation of a perfect sphere or a body-axis rotation
    of an axisymmetric rod produces zero contrast change, so the
    corresponding Fisher row/column is zero. In that case the routine
    inverts the lower-rank sub-block alone and reports the unobservable
    coordinate's bound as +inf, which is the correct estimator-theoretic
    statement (no unbiased estimator can pin a state variable to which
    the data does not respond).
    """
    F = compute_fisher_information_se3(
        renders, noise_variance_map, pixel_size_nm, z_step_nm, rotation_step_rad,
    )

    state_axes = ("x", "y", "z", "omega_x", "omega_y", "omega_z")
    sigma_units = ("nm", "nm", "nm", "rad", "rad", "rad")
    F_sym = 0.5 * (F + F.T)
    if np.any(~np.isfinite(F)):
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_z_nm": float("inf"),
            "sigma_xyz_nm": float("inf"),
            "sigma_omega_x_rad": float("inf"),
            "sigma_omega_y_rad": float("inf"),
            "sigma_omega_z_rad": float("inf"),
            "sigma_omega_total_rad": float("inf"),
            "fisher_matrix": F_sym,
            "fisher_det": float("nan"),
            "singular": True,
            "rank": 0,
            "axes_singular": list(state_axes),
            "state_axes": state_axes,
            "sigma_units": sigma_units,
        }

    diag = np.diag(F_sym)
    trans_scale = max(float(np.max(np.abs(diag[:3]))), 0.0)
    rot_scale = max(float(np.max(np.abs(diag[3:]))), 0.0)
    axis_scales = np.asarray([trans_scale, trans_scale, trans_scale, rot_scale, rot_scale, rot_scale])
    axis_tols = np.maximum(_FISHER_SE3_EPS, axis_scales * _FISHER_SE3_AXIS_RELATIVE_TOL)
    axis_observable = diag > axis_tols
    F_rank = np.array(F_sym, copy=True)
    for axis_index, observable in enumerate(axis_observable):
        if not bool(observable):
            F_rank[axis_index, :] = 0.0
            F_rank[:, axis_index] = 0.0

    try:
        evals, evecs = np.linalg.eigh(F_rank)
    except np.linalg.LinAlgError:
        evals = np.asarray([], dtype=float)
        evecs = np.empty((6, 0), dtype=float)
    scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 0.0)
    rank_tol = max(_FISHER_SE3_EPS, scale * _RELATIVE_DET_SINGULAR_TOL)
    positive = evals > rank_tol if evals.size else np.zeros(0, dtype=bool)
    fisher_rank = int(np.count_nonzero(positive))

    sigmas = [float("inf")] * 6
    axes_singular: list[str] = []
    if fisher_rank > 0:
        V = evecs[:, positive]
        inv_evals = 1.0 / evals[positive]
        F_pinv = (V * inv_evals) @ V.T
        range_projector = V @ V.T
        eye = np.eye(6)
        for i, axis in enumerate(state_axes):
            axis_residual = eye[:, i] - range_projector @ eye[:, i]
            axis_estimable = bool(axis_observable[i]) and np.linalg.norm(axis_residual) <= _FISHER_RANGE_RESIDUAL_TOL
            if axis_estimable:
                v = float(F_pinv[i, i])
                sigmas[i] = float(np.sqrt(max(v, 0.0)))
            else:
                axes_singular.append(axis)
    else:
        axes_singular = list(state_axes)

    sigma_x, sigma_y, sigma_z, sigma_wx, sigma_wy, sigma_wz = sigmas

    # Combined translation and orientation summaries. The orientation
    # aggregate is taken over the *observable* rotation axes only (those with
    # finite per-axis CRLB); it returns +inf only when every rotation axis is
    # singular. This makes partial-rank composites (e.g.\ a body-axis-symmetric
    # dimer with omega_x unobservable) rankable on their observable rotation
    # subspace, instead of all collapsing to +inf and becoming
    # cross-modality-indistinguishable. The translation aggregate keeps the
    # strict "all 3 finite" semantics because position observability is the
    # generic case for any imaging modality.
    finite_trans = [s for s in (sigma_x, sigma_y, sigma_z) if np.isfinite(s)]
    finite_rot = [s for s in (sigma_wx, sigma_wy, sigma_wz) if np.isfinite(s)]
    sigma_xyz = (
        float(np.sqrt(sum(s * s for s in finite_trans)))
        if len(finite_trans) == 3 else float("inf")
    )
    sigma_omega_total = (
        float(np.sqrt(sum(s * s for s in finite_rot)))
        if len(finite_rot) > 0 else float("inf")
    )
    observable_axis_rank = int(len(state_axes) - len(set(axes_singular)))
    reported_rank = int(min(fisher_rank, observable_axis_rank))

    return {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_z_nm": sigma_z,
        "sigma_xyz_nm": sigma_xyz,
        "sigma_omega_x_rad": sigma_wx,
        "sigma_omega_y_rad": sigma_wy,
        "sigma_omega_z_rad": sigma_wz,
        "sigma_omega_total_rad": sigma_omega_total,
        "fisher_matrix": F_sym,
        "fisher_det": float(np.linalg.det(F_sym)),
        "singular": reported_rank < 6,
        "rank": reported_rank,
        "numerical_fisher_rank": fisher_rank,
        "axes_singular": axes_singular,
        "state_axes": state_axes,
        "sigma_units": sigma_units,
    }


def _validate_rank_int(value: Any, *, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]; got {value!r}.")
    value_int = int(value)
    if value_int < lower or value_int > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]; got {value_int}.")
    return value_int


def predict_se3_rank_from_symmetry(
    continuous_rotational_symmetry_dim: int,
    translation_rank: int = 3,
    *,
    rotational_dimension: int = 3,
    singular_rotation_axes_body: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Predict local SE(3) Fisher rank from continuous rotational symmetry.

    A connected stabilizer subgroup of dimension d contributes at least d null
    rotation directions. Under the generic local-observability condition, the
    rotational Fisher rank is ``rotational_dimension - d`` and the full SE(3)
    rank is ``translation_rank + rotational_rank``. Discrete symmetries are not
    counted here because they create global pose aliases, not infinitesimal null
    directions.
    """
    rotational_dimension = _validate_rank_int(
        rotational_dimension,
        name="rotational_dimension",
        lower=1,
        upper=3,
    )
    symmetry_dim = _validate_rank_int(
        continuous_rotational_symmetry_dim,
        name="continuous_rotational_symmetry_dim",
        lower=0,
        upper=rotational_dimension,
    )
    translation_rank = _validate_rank_int(
        translation_rank,
        name="translation_rank",
        lower=0,
        upper=3,
    )
    axes = list(singular_rotation_axes_body or [])
    allowed_axes = {"x", "y", "z", "omega_x", "omega_y", "omega_z"}
    invalid_axes = [axis for axis in axes if axis not in allowed_axes]
    if invalid_axes:
        raise ValueError(
            "singular_rotation_axes_body contains unsupported axis names: "
            f"{invalid_axes!r}."
        )

    rotational_rank = rotational_dimension - symmetry_dim
    se3_rank = translation_rank + rotational_rank
    predicted_nullity = (3 - translation_rank) + symmetry_dim

    return {
        "continuous_rotational_symmetry_dim": symmetry_dim,
        "translation_rank": translation_rank,
        "rotational_dimension": rotational_dimension,
        "predicted_rotational_rank": int(rotational_rank),
        "predicted_se3_rank": int(se3_rank),
        "predicted_nullity": int(predicted_nullity),
        "symmetry_nullity_lower_bound": int(symmetry_dim),
        "singular_rotation_axes_body": axes,
    }


def predict_fused_se3_rank_from_symmetry(
    continuous_rotational_symmetry_intersection_dim: int,
    *,
    per_modality_symmetry_dims: dict[str, int] | None = None,
    translation_rank: int = 3,
    rotational_dimension: int = 3,
) -> dict[str, Any]:
    """
    Predict fused SE(3) Fisher rank from the intersection stabilizer dimension.

    For independent modalities on the same SE(3) parameter frame, the fused
    Fisher derivative along a rotation generator is zero iff every contributing
    modality is invariant along that generator. The continuous stabilizer of the
    fused contrast vector is therefore the intersection of the per-modality
    continuous stabilizers. This helper takes that intersection dimension
    explicitly; dimensions of individual stabilizers alone are not enough to
    infer the intersection.
    """
    prediction = predict_se3_rank_from_symmetry(
        continuous_rotational_symmetry_intersection_dim,
        translation_rank=translation_rank,
        rotational_dimension=rotational_dimension,
    )
    modality_dims: dict[str, int] = {}
    if per_modality_symmetry_dims is not None:
        for modality, dim in per_modality_symmetry_dims.items():
            modality_dims[str(modality)] = _validate_rank_int(
                dim,
                name=f"per_modality_symmetry_dims[{modality!r}]",
                lower=0,
                upper=rotational_dimension,
            )
    intersection_dim = prediction["continuous_rotational_symmetry_dim"]
    symmetry_broken = (
        any(dim > intersection_dim for dim in modality_dims.values())
        if modality_dims else None
    )
    return {
        **prediction,
        "continuous_rotational_symmetry_intersection_dim": intersection_dim,
        "per_modality_continuous_rotational_symmetry_dim": modality_dims,
        "symmetry_broken_by_fusion": symmetry_broken,
        "fusion_rank_prediction_note": (
            "Fusion nullity is set by the intersection of contrast-functional "
            "continuous stabilizers; per-modality stabilizer dimensions alone "
            "do not determine that intersection."
        ),
    }


def compare_observed_and_predicted_se3_rank(
    crlb_result: dict[str, Any],
    symmetry_metadata: dict[str, Any] | None,
    *,
    translation_rank: int = 3,
) -> dict[str, Any]:
    """
    Compare an observed SE(3) CRLB rank against the symmetry-rank prediction.

    Missing symmetry metadata is reported explicitly instead of guessed.
    """
    observed_rank_raw = crlb_result.get("rank", None)
    observed_rank = (
        None if observed_rank_raw is None else _validate_rank_int(
            observed_rank_raw,
            name="crlb_result['rank']",
            lower=0,
            upper=6,
        )
    )
    observed_axes = list(crlb_result.get("axes_singular", []))
    if not symmetry_metadata or symmetry_metadata.get("continuous_rotational_symmetry_dim") is None:
        return {
            "rank_prediction_available": False,
            "observed_rank": observed_rank,
            "observed_axes_singular": observed_axes,
            "rank_matches_symmetry_prediction": None,
            "satisfies_symmetry_nullity_bound": None,
            "rank_prediction_note": "No continuous_rotational_symmetry_dim metadata was supplied.",
        }

    prediction = predict_se3_rank_from_symmetry(
        symmetry_metadata["continuous_rotational_symmetry_dim"],
        translation_rank=translation_rank,
        singular_rotation_axes_body=symmetry_metadata.get("singular_rotation_axes_body"),
    )
    if observed_rank is None:
        note = "Observed CRLB result did not contain a rank."
        matches = None
        satisfies = None
        observed_nullity = None
    else:
        observed_nullity = 6 - observed_rank
        matches = observed_rank == prediction["predicted_se3_rank"]
        satisfies = observed_nullity >= prediction["symmetry_nullity_lower_bound"]
        if matches:
            note = "Observed rank matches the generic symmetry prediction."
        elif satisfies:
            note = (
                "Observed rank satisfies the symmetry nullity lower bound but "
                "shows additional degeneracy beyond the generic prediction."
            )
        else:
            note = (
                "Observed rank violates the symmetry nullity lower bound; check "
                "symmetry metadata, render convention, or rank tolerance."
            )

    return {
        **prediction,
        "rank_prediction_available": True,
        "observed_rank": observed_rank,
        "observed_axes_singular": observed_axes,
        "observed_nullity": observed_nullity,
        "rank_matches_symmetry_prediction": matches,
        "satisfies_symmetry_nullity_bound": satisfies,
        "rank_prediction_note": note,
    }


# ---------------------------------------------------------------------------
# Cross-modality orientation-CRLB ordering.
#
# For the same composite particle and the same noise floor, compare imaging
# models by the precision they deliver on orientation state, not just lateral
# position.
# ---------------------------------------------------------------------------


def compare_modality_orientation_crlb(
    renders_by_modality: dict[str, dict[str, np.ndarray]],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float,
    rotation_step_rad: float,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Order imaging modalities by the orientation Cramér-Rao bound they deliver
    on a shared composite-particle configuration.

    For each modality ``M`` the caller supplies a dict of nine perturbed
    renders (``centre``, ``z_minus/plus``, ``rx_minus/plus``,
    ``ry_minus/plus``, ``rz_minus/plus``) of the *same* composite particle
    plus a noise variance map. Each modality's renders are passed through
    :func:`compute_localization_orientation_crlb` to obtain the per-axis
    sigmas and the aggregate ``sigma_omega_total_rad``. The modalities are
    then ordered by ``sigma_omega_total_rad``, smallest first. A modality with
    only some observable rotation axes receives a finite aggregate over those
    axes plus explicit ``axes_singular`` metadata; a modality with no observable
    orientation axes reports ``sigma_omega_total_rad = +inf`` and sorts to the
    end of the ordering.

    Parameters
    ----------
    renders_by_modality : dict[str, dict[str, np.ndarray]]
        Outer key = modality name; inner dict = the nine perturbed renders
        accepted by :func:`compute_localization_orientation_crlb`. The
        rendered images must have the same ``centre`` shape across modalities
        --- they describe the *same* particle under different contrast
        mechanisms; pixel pitch and noise floor scale are common.
    noise_variance_by_modality : dict[str, ndarray or float]
        Per-modality noise-variance map (or scalar). Keys must match
        ``renders_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch (nm). A scalar keeps the historical shared-pitch
        behavior; a mapping supplies one pitch per modality.
    z_step_nm : float
        Axial translation step used to render ``z_minus``/``z_plus`` for
        every modality. Must match the step used to actually render.
    rotation_step_rad : float
        Body-frame rotation step (radians) used to render the six rotation
        perturbations for every modality.

    Returns
    -------
    dict with keys
        - ``per_modality``           : dict[str, dict] from
                                       :func:`compute_localization_orientation_crlb`,
                                       one entry per modality.
        - ``ordering``               : list of ``(modality, sigma_omega_total_rad)``
                                       sorted ascending; +inf entries last.
                                       ``ranking`` is retained as an alias.
        - ``best_modality``          : argmin ``sigma_omega_total_rad`` over
                                       the modalities for which the orientation
                                       block has at least one observable axis;
                                       ``None`` if no modality recovers any
                                       rotation axis.
        - ``best_modality_observable`` : synonym for ``best_modality``.
        - ``best_modality_full_rank``  : argmin among finite modalities whose
                                       full six-parameter block is observable;
                                       ``None`` if no modality is full rank.
        - ``relative_sigma_omega``   : dict[modality -> sigma / sigma_best];
                                       +inf when sigma is +inf or no best.
        - ``frames_to_match_best``   : dict[modality -> rho^2]; the equivalent-
                                       frame budget required for modality M to
                                       match one frame of the lowest-bound
                                       modality on *orientation* precision.
        - ``axes_singular_per_modality`` : dict[modality -> list[str]],
                                       per-axis observability flags.

    Notes
    -----
    The orientation comparator differs from
    :func:`compare_modality_information_content` in that the relevant
    summary statistic is ``sigma_omega_total_rad`` (an aggregate over the
    three rotation axes in radians) rather than ``sigma_xy_nm``. The
    equivalent-frame-budget formula carries through unchanged because the
    Fisher information is additive across independent frames for *all*
    state coordinates --- including rotational ones --- under the same
    Gaussian-noise approximation used throughout.
    """
    if not isinstance(renders_by_modality, dict) or not renders_by_modality:
        raise ValueError(
            "renders_by_modality must be a non-empty dict keyed by modality name."
        )
    if not isinstance(noise_variance_by_modality, dict):
        raise ValueError("noise_variance_by_modality must be a dict.")
    if set(renders_by_modality) != set(noise_variance_by_modality):
        missing = set(renders_by_modality) - set(noise_variance_by_modality)
        extra = set(noise_variance_by_modality) - set(renders_by_modality)
        raise ValueError(
            "renders_by_modality and noise_variance_by_modality keys must match; "
            f"missing noise entries: {sorted(missing)}; extra noise entries: {sorted(extra)}."
        )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")
    if not np.isfinite(rotation_step_rad) or rotation_step_rad <= 0.0:
        raise ValueError(
            f"rotation_step_rad must be positive; got {rotation_step_rad}."
        )

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        renders_by_modality.keys(),
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, renders in renders_by_modality.items():
        try:
            per_modality[modality] = compute_localization_orientation_crlb(
                renders=renders,
                noise_variance_map=noise_variance_by_modality[modality],
                pixel_size_nm=pixel_sizes[modality],
                z_step_nm=z_step_nm,
                rotation_step_rad=rotation_step_rad,
            )
        except Exception as exc:  # noqa: BLE001 — record per-modality failure without aborting comparison
            per_modality[modality] = {
                "error": repr(exc),
                "sigma_x_nm": float("inf"),
                "sigma_y_nm": float("inf"),
                "sigma_z_nm": float("inf"),
                "sigma_xyz_nm": float("inf"),
                "sigma_omega_x_rad": float("inf"),
                "sigma_omega_y_rad": float("inf"),
                "sigma_omega_z_rad": float("inf"),
                "sigma_omega_total_rad": float("inf"),
                "fisher_matrix": None,
                "fisher_det": None,
                "singular": True,
                "rank": 0,
                "axes_singular": ["x", "y", "z", "omega_x", "omega_y", "omega_z"],
                "state_axes": ("x", "y", "z", "omega_x", "omega_y", "omega_z"),
                "sigma_units": ("nm", "nm", "nm", "rad", "rad", "rad"),
            }

    # Build the (modality, sigma_omega_total) tuples and sort.
    items = [
        (m, float(r.get("sigma_omega_total_rad", float("inf"))))
        for m, r in per_modality.items()
    ]
    # Sort ascending; +inf and NaN go last.
    def _sort_key(pair):
        v = pair[1]
        if v != v or v == float("inf"):  # NaN or inf
            return (1, 0.0)
        return (0, v)

    ranking = sorted(items, key=_sort_key)

    # ``best_modality`` follows the documented observable-axis contract. The
    # stricter full-rank winner is exposed separately for callers that need all
    # six localization/orientation coordinates to be jointly observable.
    best_modality: str | None = None
    best_modality_full_rank: str | None = None
    for modality, sigma in ranking:
        rec = per_modality[modality]
        if best_modality is None and np.isfinite(sigma):
            best_modality = modality
        if (
            best_modality_full_rank is None
            and np.isfinite(sigma)
            and rec.get("rank", 0) == 6
            and not rec.get("axes_singular", [])
        ):
            best_modality_full_rank = modality
        if best_modality is not None and best_modality_full_rank is not None:
            break

    # Relative-precision and equivalent-frame-budget against best.
    if best_modality is None:
        relative = {m: float("inf") for m, _ in items}
        frames = {m: float("inf") for m, _ in items}
    else:
        sigma_best = float(per_modality[best_modality]["sigma_omega_total_rad"])
        relative = {}
        frames = {}
        for m, s in items:
            if not np.isfinite(s) or sigma_best <= 0.0:
                relative[m] = float("inf")
                frames[m] = float("inf")
            else:
                rho = s / sigma_best
                relative[m] = float(rho)
                frames[m] = float(rho * rho)

    axes_singular_per_modality = {
        m: list(per_modality[m].get("axes_singular", [])) for m in per_modality
    }

    return {
        "per_modality": per_modality,
        "ordering": ranking,
        "ranking": ranking,
        "best_modality": best_modality,
        "best_modality_observable": best_modality,
        "best_modality_full_rank": best_modality_full_rank,
        "relative_sigma_omega": relative,
        "frames_to_match_best": frames,
        "axes_singular_per_modality": axes_singular_per_modality,
        "pixel_size_nm_by_modality": pixel_sizes,
    }


# ---------------------------------------------------------------------------
# Cross-modality axial information-content ordering.
#
# This isolates z-localization precision from the total 3D bound because
# defocus, phase encoding, and interferometric envelope cues vary strongly
# across modalities.
# ---------------------------------------------------------------------------


def compare_modality_axial_information_content(
    contrast_stack_by_modality: dict[str, np.ndarray],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""
    Order imaging modalities by the *axial* (z) Cramér-Rao bound they deliver
    on a shared particle configuration, under a shared physics-faithful
    forward model.

    For each modality ``M`` the caller supplies a ``(3, H, W)`` z-stack
    ``[C(z - dz), C(z), C(z + dz)]`` of the *same* particle and a noise
    variance map; each stack is passed through
    :func:`compute_localization_crlb_3d` and the modalities are then ordered
    by ``sigma_z_nm`` smallest first. Modalities that are axially singular
    (``sigma_z_nm = +inf`` because dC/dz vanishes — e.g. an even-in-z
    Gaussian envelope rendered through a contrast mechanism that does not
    encode phase) sort to the end of the ordering with infinite frames-to-
    match-best, the correct estimator-theoretic statement that no amount of
    frame averaging in that modality recovers axial information about that
    particle.

    Parameters
    ----------
    contrast_stack_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> (3, H, W) z-stack`` in the same
        convention as :func:`compute_localization_crlb_3d`. The middle
        plane is the in-focus reference; the outer planes feed the
        symmetric finite-difference axial derivative.
    noise_variance_by_modality : dict[str, ndarray | float]
        Mapping ``modality_name -> pixel-wise variance`` (or scalar) for the
        same modalities. Must share keys with ``contrast_stack_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies one pitch per modality.
    z_step_nm : float
        Axial spacing between the three rendered planes, in nanometres.
        Must be > 0 and identical across modalities — the axial Fisher
        derivative is `(C[2] - C[0]) / (2 * z_step_nm)` so a per-modality
        z_step would scale the bound differently for each modality and
        invalidate the ordering.

    Returns
    -------
    dict with keys
        - ``per_modality``         : dict[str, dict] from
                                     :func:`compute_localization_crlb_3d`,
                                     one entry per modality (preserves
                                     ``sigma_x/y/z_nm``, ``axially_singular``,
                                     etc.).
        - ``ordering_z``           : list of ``(modality, sigma_z_nm)``
                                     sorted ascending; +inf entries last.
                                     ``ranking_z`` is retained as an alias.
        - ``best_modality_z``      : argmin ``sigma_z_nm`` over modalities
                                     for which the axial bound is finite;
                                     ``None`` if every modality is axially
                                     singular on the supplied stacks.
        - ``relative_sigma_z``     : dict[modality -> sigma / sigma_best];
                                     +inf when sigma is +inf or no best.
        - ``frames_to_match_best_z`` : dict[modality -> rho^2]; the
                                     equivalent-frame budget required for
                                     modality M to match one frame of the
                                     lowest-bound modality on *axial*
                                     precision, under the assumption of
                                     independent frames (Fisher additivity).
        - ``axially_singular_per_modality`` : dict[modality -> bool]; the
                                     ``axially_singular`` flag echoed from
                                     the per-modality 3D CRLB result.

    Why an axial-only ordering and not just sigma_xyz
    -----------------------------------------------
    The total 3D bound ``sigma_xyz`` mixes axial and lateral information
    in a single scalar. For modalities with comparable lateral PSF widths
    but very different axial structure, ``sigma_xyz`` can be dominated by
    the lateral term and make the two modalities appear comparable. This helper
    isolates the axial dimension so the ordering reflects axial-recovery
    capability per modality, not a lateral-dominated aggregate.

    Notes
    -----
    Ties are broken by the order in which modalities appear in the input
    dict. Stacks must share the (3, H, W) shape across modalities; the
    function does not resample or align across modalities — the caller is
    responsible for producing comparable stacks (typically by routing the
    same particle population through every ``ImagingModel`` instance with
    the same z_step_nm and the same per-modality noise calibration).
    """
    if not isinstance(contrast_stack_by_modality, dict) or not contrast_stack_by_modality:
        raise ValueError(
            "contrast_stack_by_modality must be a non-empty dict keyed by modality name."
        )
    if not isinstance(noise_variance_by_modality, dict):
        raise ValueError("noise_variance_by_modality must be a dict.")
    if set(contrast_stack_by_modality.keys()) != set(noise_variance_by_modality.keys()):
        raise ValueError(
            "contrast_stack_by_modality and noise_variance_by_modality must share keys; "
            f"missing from stacks: "
            f"{set(noise_variance_by_modality) - set(contrast_stack_by_modality)}; "
            f"missing from noise: "
            f"{set(contrast_stack_by_modality) - set(noise_variance_by_modality)}."
        )
    if not np.isfinite(z_step_nm) or z_step_nm <= 0.0:
        raise ValueError(f"z_step_nm must be positive; got {z_step_nm}.")

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        contrast_stack_by_modality.keys(),
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, stack in contrast_stack_by_modality.items():
        try:
            per_modality[modality] = compute_localization_crlb_3d(
                stack,
                noise_variance_by_modality[modality],
                pixel_sizes[modality],
                z_step_nm,
            )
        except Exception as exc:  # noqa: BLE001 — record per-modality failure without aborting comparison
            per_modality[modality] = {
                "error": repr(exc),
                "sigma_x_nm": float("inf"),
                "sigma_y_nm": float("inf"),
                "sigma_z_nm": float("inf"),
                "sigma_xy_nm": float("inf"),
                "sigma_xyz_nm": float("inf"),
                "fisher_matrix": None,
                "fisher_det": None,
                "axially_singular": True,
                "singular": True,
            }

    items = [
        (m, float(r.get("sigma_z_nm", float("inf"))))
        for m, r in per_modality.items()
    ]

    def _sort_key(pair):
        v = pair[1]
        if v != v or v == float("inf"):
            return (1, 0.0)
        return (0, v)

    ranking_z = sorted(items, key=_sort_key)

    # Best modality is the smallest finite sigma_z whose 3D Fisher block is
    # not axially singular. (sigma_z is finite iff the axial-derivative term
    # carries information; axially_singular = True implies sigma_z = +inf.
    # Checking both conditions keeps the per_modality contract explicit.)
    best_modality_z: str | None = None
    for modality, sigma in ranking_z:
        rec = per_modality[modality]
        if (
            np.isfinite(sigma)
            and not rec.get("axially_singular", True)
            and sigma > 0.0
        ):
            best_modality_z = modality
            break

    if best_modality_z is None:
        relative_sigma_z = {m: float("inf") for m, _ in items}
        frames_to_match_best_z = {m: float("inf") for m, _ in items}
    else:
        sigma_best = float(per_modality[best_modality_z]["sigma_z_nm"])
        relative_sigma_z = {}
        frames_to_match_best_z = {}
        for m, s in items:
            if not np.isfinite(s) or sigma_best <= 0.0:
                relative_sigma_z[m] = float("inf")
                frames_to_match_best_z[m] = float("inf")
            else:
                rho = s / sigma_best
                relative_sigma_z[m] = float(rho)
                frames_to_match_best_z[m] = float(rho * rho)

    axially_singular_per_modality = {
        m: bool(per_modality[m].get("axially_singular", True))
        for m in per_modality
    }

    return {
        "per_modality": per_modality,
        "ordering_z": ranking_z,
        "ranking_z": ranking_z,
        "best_modality_z": best_modality_z,
        "relative_sigma_z": relative_sigma_z,
        "frames_to_match_best_z": frames_to_match_best_z,
        "axially_singular_per_modality": axially_singular_per_modality,
        "pixel_size_nm_by_modality": pixel_sizes,
    }


# ---------------------------------------------------------------------------
# Detected-quanta-budget-normalized cross-modality comparator.
#
# This path assigns each modality the same detected-quanta budget before
# computing the Fisher matrix. Count-domain modalities use Poisson shot noise
# plus optional readout variance; phase-domain modalities use their phase-noise
# model directly.
#
# The derivatives fed to compute_localization_crlb (or _3d if z_step_nm is
# supplied) are derivatives of that rescaled count image, not derivatives of
# signed baseline-subtracted contrast treated as a Poisson rate.
# For phase-domain modalities (QPI), the phase image remains in radians and
# the matched budget sets var(phi)=1/(visibility^2 * quanta_per_pixel) plus
# optional phase-readout variance.
#
# Since Fisher information is linear in the quanta budget under shot noise
# (F propto N, sigma propto 1/sqrt(N)), a single budget
# value is sufficient to characterise the entire scaling family; the
# returned bound corresponds to that budget per frame.
#
# Budget normalization strips out one instrument-tunable amplitude knob for
# count-domain outputs and uses a phase-noise path for phase-domain modalities
# rather than treating radians as counts.
# ---------------------------------------------------------------------------


def _default_measurement_model_for_modality(modality: str) -> str:
    key = str(modality).lower()
    if key in {"quantitative_phase", "qpi", "phase"}:
        return "phase"
    return "count"


def _normalise_measurement_model(model: str) -> str:
    key = str(model).strip().lower()
    if key in {"count", "counts", "photon", "photons", "photon_count",
               "electron", "electrons", "electron_count", "detected_quanta"}:
        return "count"
    if key in {"phase", "phase_radian", "phase_radians", "qpi"}:
        return "phase"
    raise ValueError(
        "measurement models must be 'count' or 'phase' "
        f"(with accepted mode names); got {model!r}."
    )


def _mapping_or_scalar_value(
    value: float | dict[str, float] | None,
    modality: str,
    default: float,
) -> float:
    if value is None:
        return float(default)
    if isinstance(value, dict):
        return float(value.get(modality, default))
    return float(value)


def compute_quanta_scaling_law(
    fisher_at_budget: np.ndarray,
    budget: float,
    target_budgets: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Scale a Fisher matrix under the ideal detected-quanta law F ∝ N_Q."""
    F = np.asarray(fisher_at_budget, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] < 2:
        raise ValueError(f"fisher_at_budget must be square with at least 2 axes; got {F.shape}.")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"budget must be positive and finite; got {budget!r}.")
    if not target_budgets:
        raise ValueError("target_budgets must contain at least one budget.")

    budget_grid: list[float] = []
    scaled_fisher: dict[float, np.ndarray] = {}
    scaled_sigma: dict[float, float] = {}
    for target in target_budgets:
        target = float(target)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError(
                f"target budgets must be positive and finite; got {target!r}."
            )
        alpha = target / float(budget)
        F_scaled = alpha * F
        sigma_xy, _ = _sigma_xy_from_fisher(F_scaled)
        budget_grid.append(target)
        scaled_fisher[target] = F_scaled
        scaled_sigma[target] = sigma_xy

    return {
        "reference_budget": float(budget),
        "budget_grid": budget_grid,
        "scaled_fisher_by_budget": scaled_fisher,
        "scaled_sigma_xy_by_budget": scaled_sigma,
        "scaling_assumption": "ideal_count_domain_fisher_linear_in_detected_quanta",
    }


def check_budget_ranking_invariance(
    results_by_budget: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Check whether modality ordering is invariant across budget-normalized runs."""
    if not isinstance(results_by_budget, dict) or len(results_by_budget) < 2:
        raise ValueError("results_by_budget must contain at least two budget results.")

    ordering_by_budget: dict[float, list[str]] = {}
    readout_limited: set[str] = set()
    for budget in sorted(float(b) for b in results_by_budget):
        result = results_by_budget[budget]
        ordering = result.get("ordering_xy", result.get("ranking_xy"))
        if ordering is None:
            raise ValueError("Each result must contain ordering_xy or ranking_xy.")
        ordering_by_budget[budget] = [str(item[0]) for item in ordering]
        for key in ("phase_readout_limited", "count_readout_limited"):
            for modality, limited in (result.get(key, {}) or {}).items():
                if limited:
                    readout_limited.add(str(modality))

    orders = list(ordering_by_budget.values())
    invariant = all(order == orders[0] for order in orders[1:])
    notes: list[str] = []
    if invariant:
        notes.append("modality ordering is invariant across supplied budgets")
    else:
        notes.append("modality ordering changes across supplied budgets")
    if readout_limited:
        notes.append("readout-limited modalities can break ideal quanta scaling")

    return {
        "ordering_invariant": bool(invariant),
        "ordering_by_budget": ordering_by_budget,
        "ranking_invariant": bool(invariant),
        "ranking_by_budget": ordering_by_budget,
        "readout_limited_modalities": sorted(readout_limited),
        "invariance_notes": notes,
    }


def check_budget_ordering_invariance(
    results_by_budget: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Preferred-name wrapper for :func:`check_budget_ranking_invariance`."""
    return check_budget_ranking_invariance(results_by_budget)


def compare_modality_information_content_detected_quanta_normalized(
    contrast_by_modality: dict[str, np.ndarray],
    quanta_budget: float,
    pixel_size_nm: float | dict[str, float],
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    measurement_model_by_modality: dict[str, str] | None = None,
    detected_count_image_by_modality: dict[str, np.ndarray] | None = None,
    readout_variance: float = 0.0,
    phase_visibility_by_modality: float | dict[str, float] | None = None,
    phase_readout_variance_by_modality: float | dict[str, float] | None = None,
    z_step_nm: float | None = None,
) -> dict[str, Any]:
    r"""
    Detected-quanta-budget-normalized cross-modality CRLB comparator.

    Wraps :func:`compare_modality_information_content` with a per-modality
    measurement-domain normalization. Count-domain modalities should be
    supplied as detector-domain count images after the imaging model's
    ``scale_intensity_to_counts`` step, or through
    ``detected_count_image_by_modality`` when ``contrast_by_modality`` carries
    a separate signed derivative image. Those count images are rescaled to the
    same total detected-quanta budget and used only as the Poisson mean/noise
    floor; the Fisher derivative target remains the signed per-particle
    contrast image. Phase-domain modalities (currently QPI) keep the phase
    signal in radians and use a shot-noise phase-variance model set by the same
    detected-quanta budget.

    Parameters
    ----------
    contrast_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> per-particle contrast image``. For the 2D
        comparison each value is a ``(H, W)`` array; for the 3D comparison
        (``z_step_nm`` supplied) each value is a ``(3, H, W)`` three-plane
        stack in the same convention as :func:`compute_localization_crlb_3d`.
    quanta_budget : float
        Total detected quanta per frame (per modality, per particle). For
        optical count modes this means photons; for electron modalities this
        means detected electrons or dose quanta. The same value is used for
        every modality so the comparison is budget-fair by construction.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies one pitch per modality.
    measurement_model_by_modality : dict[str, str] or None
        Optional mapping from modality name to measurement model. Accepted
        models are ``"count"`` and ``"phase"``. If omitted, QPI-like names
        (``"quantitative_phase"``, ``"qpi"``) default to ``"phase"`` and all
        others default to ``"count"``.
    detected_count_image_by_modality : dict[str, ndarray] or None
        Optional count-domain mean image for each modality, already in
        detector-count units before budget rescaling. Calibrated count-domain
        runs should pass true detector-domain count images. The supplied count
        image sets the Poisson mean and exposure scale only; it is not
        substituted for the derivative image. If this mapping is omitted, the
        function builds a non-negative diagnostic proxy from
        ``contrast_by_modality[modality]`` for exploratory comparisons.
    readout_variance : float, default 0.0
        Additive Gaussian readout variance, in count-quanta squared, added
        to count-domain modalities after shot noise.
    phase_visibility_by_modality : float or dict, optional
        Interferometric / demodulation visibility factor for phase-domain
        modalities. The default is ``1.0``. The phase-noise convention is
        ``var(phi) = 1 / (visibility^2 * quanta_per_pixel)`` plus optional
        phase readout variance.
    phase_readout_variance_by_modality : float or dict, optional
        Additive phase-readout variance in radians squared for phase-domain
        modalities. The default is ``0.0``.
    z_step_nm : float or None, default None
        If None: 2D comparison. If a positive float: 3D comparison; the
        count-domain 3-plane stack normalization uses the central detector
        count plane's :math:`\sum_p m_M(p)` to set the count scale (so the
        budget is the in-focus-frame budget, with the outer planes scaled by
        the same factor). Phase-domain 3D mode uses the same central-plane
        quanta-per-pixel phase variance for all three planes.

    Returns
    -------
    result : dict
        Same keys as :func:`compare_modality_information_content`, with
        additional metadata recording the normalization parameters,
        measurement models, count-domain scale factors, and phase-domain
        variance terms.

    Notes
    -----
    For count-domain modalities, the detector-domain count image is the
    Poisson mean and sets the exposure/dose scale. The signed per-particle
    contrast image remains the derivative target after that same exposure
    scaling, so background slope in a supplied count image cannot by itself
    create localization information for a zero-contrast particle. For
    phase-domain modalities, the phase image is already the detector-domain
    signal in radians, so the signal amplitude is not normalized away; the
    budget controls phase readout variance instead. This is the phase-domain
    analogue of photon normalization and avoids treating radians as counts.
    """
    if not isinstance(contrast_by_modality, dict) or not contrast_by_modality:
        raise ValueError(
            "contrast_by_modality must be a non-empty dict keyed by modality name."
        )
    if not np.isfinite(quanta_budget) or quanta_budget <= 0.0:
        raise ValueError(
            f"quanta_budget must be a positive finite scalar; got {quanta_budget!r}."
        )
    if not np.isfinite(readout_variance) or readout_variance < 0.0:
        raise ValueError(
            f"readout_variance must be a non-negative finite scalar; got {readout_variance!r}."
        )

    rescaled_contrast: dict[str, np.ndarray] = {}
    rescaled_noise: dict[str, np.ndarray] = {}
    quanta_scale_by_modality: dict[str, float | None] = {}
    measurement_model_record: dict[str, str] = {}
    quanta_per_pixel_by_modality: dict[str, float] = {}
    phase_variance_by_modality: dict[str, float] = {}
    phase_visibility_record: dict[str, float] = {}
    phase_readout_limited: dict[str, bool] = {}
    count_readout_limited: dict[str, bool] = {}
    readout_variance_fraction_by_modality: dict[str, float] = {}
    budget_scaling_notes: dict[str, str] = {}

    for modality, contrast in contrast_by_modality.items():
        c = np.asarray(contrast, dtype=float)
        if not np.all(np.isfinite(c)):
            raise ValueError(f"contrast image for modality {modality!r} must contain only finite values.")
        if z_step_nm is None:
            if c.ndim != 2:
                raise ValueError(
                    f"2D mode expects (H, W) contrast for modality {modality!r}; "
                    f"got shape {c.shape}."
                )
            central = c
        else:
            if c.ndim != 3 or c.shape[0] != 3:
                raise ValueError(
                    f"3D mode expects (3, H, W) z-stack for modality {modality!r}; "
                    f"got shape {c.shape}."
            )
            central = c[1]

        if measurement_model_by_modality is None:
            model = _default_measurement_model_for_modality(modality)
        else:
            model = measurement_model_by_modality.get(
                modality, _default_measurement_model_for_modality(modality)
            )
        model = _normalise_measurement_model(model)
        measurement_model_record[modality] = model

        if model == "count":
            if detected_count_image_by_modality is not None and modality in detected_count_image_by_modality:
                count_image = np.asarray(
                    detected_count_image_by_modality[modality], dtype=float
                )
                if count_image.shape != c.shape:
                    raise ValueError(
                        "detected_count_image_by_modality[%r] has shape %s; "
                        "expected %s." % (modality, count_image.shape, c.shape)
                    )
                if not np.all(np.isfinite(count_image)):
                    raise ValueError(
                        "detected_count_image_by_modality[%r] must contain only finite values."
                        % modality
                    )
            else:
                # Contrast-only diagnostics may not have detector-count images.
                # Calibrated count-domain runs should pass actual count images
                # after scale_intensity_to_counts.
                if float(np.nanmin(central)) < 0.0:
                    shift = -float(np.nanmin(central))
                    count_image = c + shift
                else:
                    count_image = c.copy()

            if z_step_nm is None:
                central_count = np.asarray(count_image, dtype=float)
            else:
                central_count = np.asarray(count_image, dtype=float)[1]
            central_count = np.where(np.isfinite(central_count), central_count, 0.0)
            central_count = np.maximum(central_count, 0.0)

            total_signal = float(np.sum(central_count))
            if total_signal <= 0.0:
                scale = 0.0
            else:
                scale = float(quanta_budget) / total_signal
            quanta_scale_by_modality[modality] = scale

            # The count image determines the Poisson mean/noise and the common
            # detected-quanta exposure scale. Fisher derivatives must still be
            # taken from the signed particle contrast image; replacing it with
            # a nonnegative count image would make background slopes look like
            # particle-localization signal.
            rescaled_c = scale * c
            if z_step_nm is None:
                mean_quanta = scale * central_count
            else:
                mean_quanta = scale * central_count
            var = mean_quanta + float(readout_variance)
            mean_signal = float(np.mean(mean_quanta)) if mean_quanta.size else 0.0
            denom = mean_signal + float(readout_variance)
            readout_fraction = (
                float(readout_variance) / denom if denom > 0.0 else 0.0
            )
            readout_variance_fraction_by_modality[modality] = readout_fraction
            count_readout_limited[modality] = bool(readout_variance > 0.0)
            budget_scaling_notes[modality] = (
                "count-domain ideal F∝N scaling is exact only when additive readout variance is negligible"
                if readout_variance > 0.0
                else "count-domain ideal F∝N scaling"
            )

        elif model == "phase":
            visibility = _mapping_or_scalar_value(
                phase_visibility_by_modality, modality, 1.0
            )
            phase_readout_variance = _mapping_or_scalar_value(
                phase_readout_variance_by_modality, modality, 0.0
            )
            if not np.isfinite(visibility) or visibility <= 0.0:
                raise ValueError(
                    f"phase visibility for modality {modality!r} must be "
                    f"positive and finite; got {visibility!r}."
                )
            if (
                not np.isfinite(phase_readout_variance)
                or phase_readout_variance < 0.0
            ):
                raise ValueError(
                    f"phase readout variance for modality {modality!r} must be "
                    "non-negative and finite; got "
                    f"{phase_readout_variance!r}."
                )

            quanta_per_pixel = float(quanta_budget) / float(central.size)
            phase_variance = (
                1.0 / (visibility * visibility * quanta_per_pixel)
                + phase_readout_variance
            )
            quanta_per_pixel_by_modality[modality] = quanta_per_pixel
            phase_variance_by_modality[modality] = float(phase_variance)
            phase_visibility_record[modality] = float(visibility)
            quanta_scale_by_modality[modality] = None
            readout_variance_fraction_by_modality[modality] = float(
                phase_readout_variance / phase_variance
                if phase_variance > 0.0 else 0.0
            )
            phase_readout_limited[modality] = bool(phase_readout_variance > 0.0)
            budget_scaling_notes[modality] = (
                "phase-domain exact quanta scaling is broken by additive phase readout variance"
                if phase_readout_variance > 0.0
                else "phase-domain shot-noise scaling with var(phi)=1/(V^2 n_Q)"
            )

            rescaled_c = c
            var = np.full(central.shape, phase_variance, dtype=float)

        else:
            raise AssertionError(f"Unhandled measurement model {model!r}.")

        # Floor the variance to a tiny positive constant to avoid
        # divide-by-zero in the Fisher gradient sum at zero-signal pixels in
        # the shot-noise-only regime.
        var = np.maximum(var, _FISHER_VARIANCE_FLOOR)

        rescaled_contrast[modality] = rescaled_c
        rescaled_noise[modality] = var

    result = compare_modality_information_content(
        rescaled_contrast,
        rescaled_noise,
        pixel_size_nm,
        z_step_nm=z_step_nm,
        pixel_size_nm_by_modality=pixel_size_nm_by_modality,
    )

    result["quanta_budget"] = float(quanta_budget)
    result["readout_variance"] = float(readout_variance)
    result["normalization"] = "detected_quanta_domain_aware"
    result["measurement_model_by_modality"] = measurement_model_record
    result["quanta_scale_by_modality"] = quanta_scale_by_modality
    result["quanta_per_pixel_by_modality"] = quanta_per_pixel_by_modality
    result["phase_variance_by_modality"] = phase_variance_by_modality
    result["phase_visibility_by_modality"] = phase_visibility_record
    result["phase_readout_limited"] = phase_readout_limited
    result["count_readout_limited"] = count_readout_limited
    result["readout_variance_fraction_by_modality"] = readout_variance_fraction_by_modality
    result["budget_scaling_notes"] = budget_scaling_notes
    return result


def fit_power_law_scaling(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    expected_exponent: float | None = None,
) -> dict[str, float]:
    r"""Fit ``y = C x^a`` on positive finite samples.

    This helper is intentionally generic because several Syniscopy
    consistency diagnostics reduce to checking a fitted log-log slope:
    shot-noise/SNR scaling, Rayleigh-amplitude scaling, and other
    closed-form Fisher controls. It returns the observed exponent, log-space
    R^2, and the largest relative residual in linear units.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape; got {xs.shape} and {ys.shape}.")
    finite = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if int(finite.sum()) < 2:
        raise ValueError("At least two positive finite samples are required.")
    log_x = np.log(xs[finite])
    log_y = np.log(ys[finite])
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + exponent * log_x
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    max_relative_residual = float(np.max(np.abs(np.exp(predicted - log_y) - 1.0)))
    result = {
        "exponent": float(exponent),
        "intercept": float(intercept),
        "r2_log": float(r2),
        "max_relative_residual": max_relative_residual,
        "num_samples": float(finite.sum()),
    }
    if expected_exponent is not None:
        result["expected_exponent"] = float(expected_exponent)
        result["exponent_error"] = float(exponent - float(expected_exponent))
    return result


def summarize_closed_form_scaling_checks(
    *,
    snr_x: np.ndarray | list[float],
    snr_sigma: np.ndarray | list[float],
    rayleigh_diameter_nm: np.ndarray | list[float],
    rayleigh_sigma: np.ndarray | list[float],
    fixed_snr_sigma: np.ndarray | list[float] | None = None,
    snr_expected_exponent: float = -1.0,
) -> dict[str, dict[str, float | str]]:
    """Summarize paper-facing closed-form Fisher scaling checks.

    The function keeps the actual scaling diagnostics in the core information
    layer rather than in manuscript assembly code. ``snr_x`` is the supplied
    signal-to-noise proxy; pass ``sqrt(detected_quanta)`` for a Poisson-limited
    count sweep so that the expected Cramér--Rao exponent is ``-1``.
    """
    snr_fit = fit_power_law_scaling(
        snr_x, snr_sigma, expected_exponent=snr_expected_exponent
    )
    rayleigh_fit = fit_power_law_scaling(
        rayleigh_diameter_nm, rayleigh_sigma, expected_exponent=-3.0
    )
    if fixed_snr_sigma is None:
        fixed_snr_values = np.array([1.0, 1.0], dtype=float)
    else:
        fixed_snr_values = np.asarray(fixed_snr_sigma, dtype=float)
    finite = fixed_snr_values[np.isfinite(fixed_snr_values) & (fixed_snr_values > 0.0)]
    if finite.size == 0:
        raise ValueError("fixed_snr_sigma must contain at least one positive finite value.")
    max_rel_change = float((np.max(finite) - np.min(finite)) / np.mean(finite))
    return {
        "detected_quanta_or_snr_scaling": {
            "description": "sigma_xy power-law fit over a detected-quanta/SNR proxy sweep",
            **snr_fit,
        },
        "rayleigh_iscat_diameter_scaling": {
            "description": "sigma_xy power-law fit over an interferometric Rayleigh-size sweep",
            **rayleigh_fit,
        },
        "fixed_snr_diameter_control": {
            "description": "diameter label varied while contrast/noise arrays are held fixed",
            "expected_exponent": 0.0,
            "exponent": 0.0,
            "exponent_error": 0.0,
            "r2_log": float("nan"),
            "max_relative_residual": max_rel_change,
            "num_samples": float(finite.size),
        },
    }


# ---------------------------------------------------------------------------
# Multi-modality fusion CRLB.
#
# Fisher matrices add when modalities image the same particle in the same
# physical coordinates with independent measurement noise. Optional registration
# covariance is applied before summing each modality's contribution.
# ---------------------------------------------------------------------------


def _fisher_for_modality(
    contrast: np.ndarray,
    noise_variance: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float | None,
) -> np.ndarray:
    """Compute the per-modality Fisher matrix in 2D or 3D mode.

    Returns a (2, 2) or (3, 3) symmetric Fisher matrix. The 2D path mirrors
    compute_fisher_information; the 3D path mirrors compute_fisher_information_3d.
    Internal helper for the fusion CRLB.
    """
    if z_step_nm is None:
        if contrast.ndim != 2:
            raise ValueError(
                f"2D fusion mode expects (H, W) contrast; got shape {contrast.shape}."
            )
        return compute_fisher_information(contrast, noise_variance, pixel_size_nm)
    else:
        if contrast.ndim != 3 or contrast.shape[0] != 3:
            raise ValueError(
                f"3D fusion mode expects (3, H, W) contrast; got shape {contrast.shape}."
            )
        return compute_fisher_information_3d(
            contrast, noise_variance, pixel_size_nm, z_step_nm=z_step_nm
        )


def _registration_adjusted_fisher(
    F: np.ndarray,
    registration_covariance: np.ndarray | None,
) -> np.ndarray:
    """Inflate observable covariance by registration error, preserving nullspaces."""
    if registration_covariance is None:
        return F
    F = np.asarray(F, dtype=float)
    sigma = np.asarray(registration_covariance, dtype=float)
    if sigma.shape not in ((2, 2), F.shape):
        raise ValueError(
            "registration_covariance must be 2x2 or match the Fisher matrix "
            f"shape {F.shape}; got {sigma.shape}."
        )
    if sigma.shape == (2, 2) and F.shape != (2, 2):
        sigma_full = np.zeros_like(F, dtype=float)
        sigma_full[:2, :2] = sigma
        sigma = sigma_full
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("registration_covariance must be symmetric.")
    evals = np.linalg.eigvalsh(sigma)
    if np.any(evals < -1e-12):
        raise ValueError("registration_covariance must be positive semidefinite.")
    trace = float(np.trace(F))
    if trace <= 0.0 or not np.isfinite(trace):
        return F
    F_sym = 0.5 * (F + F.T)
    try:
        evals, evecs = np.linalg.eigh(F_sym)
    except np.linalg.LinAlgError:
        return F_sym
    positive = _positive_fisher_eigenvalue_mask(evals)
    if not np.any(positive):
        return np.zeros_like(F_sym)
    if not np.all(positive):
        V = evecs[:, positive]
        F_obs = V.T @ F_sym @ V
        sigma_obs = V.T @ sigma @ V
        try:
            cov_obs = np.linalg.inv(F_obs)
            adjusted_obs = np.linalg.inv(cov_obs + sigma_obs)
        except np.linalg.LinAlgError:
            cov_obs = np.linalg.pinv(F_obs)
            adjusted_obs = np.linalg.pinv(cov_obs + sigma_obs)
        adjusted = V @ adjusted_obs @ V.T
        return 0.5 * (adjusted + adjusted.T)
    try:
        cov = np.linalg.inv(F_sym)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F_sym)
    inflated_cov = cov + sigma
    try:
        return np.linalg.inv(inflated_cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(inflated_cov)


def _sigma_xy_from_fisher(F: np.ndarray) -> tuple[float, bool]:
    """Return lateral L2 CRLB sigma, allowing non-lateral null axes."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] < 2:
        raise ValueError(f"Fisher matrix must be square with at least 2 axes; got {F.shape}.")

    sigmas, singular_axes = _axis_sigmas_from_fisher(F)
    if (
        len(sigmas) < 2
        or singular_axes[0]
        or singular_axes[1]
        or not np.isfinite(sigmas[0])
        or not np.isfinite(sigmas[1])
    ):
        return float("inf"), True
    return float(np.sqrt(sigmas[0] ** 2 + sigmas[1] ** 2)), False


def _sigma_xyz_from_fisher(F: np.ndarray) -> tuple[float, bool]:
    """Return the 3D L2 CRLB sigma and singular flag for a 3x3 Fisher matrix."""
    F = np.asarray(F, dtype=float)
    if F.shape != (3, 3):
        raise ValueError(f"Fisher matrix must be 3x3 for xyz sigma; got {F.shape}.")
    F_sym = 0.5 * (F + F.T)
    det = float(np.linalg.det(F_sym))
    trace = float(np.trace(F_sym))
    try:
        eigvals = np.linalg.eigvalsh(F_sym)
    except np.linalg.LinAlgError:
        eigvals = np.asarray([float("nan")])
    singular = (
        (not np.isfinite(det))
        or det <= 0.0
        or (trace > 0.0 and det < (trace ** 3) * _RELATIVE_DET_SINGULAR_TOL)
        or trace <= 0.0
        or (not np.all(np.isfinite(eigvals)))
        or float(np.min(eigvals)) <= 0.0
    )
    if singular:
        return float("inf"), True
    cov = np.linalg.inv(F_sym)
    variances = [max(float(cov[i, i]), 0.0) for i in range(3)]
    return float(np.sqrt(sum(variances))), False


def _axis_names_for_dim(dim: int) -> list[str]:
    defaults = ["x", "y", "z"]
    return defaults[:dim] if dim <= len(defaults) else [f"axis_{idx}" for idx in range(dim)]


def _positive_fisher_eigenvalue_mask(evals: np.ndarray) -> np.ndarray:
    evals = np.asarray(evals, dtype=float)
    if evals.size == 0 or not np.all(np.isfinite(evals)):
        return np.zeros(evals.shape, dtype=bool)
    scale = float(np.max(np.abs(evals))) if evals.size else 0.0
    rank_tol = max(_FISHER_EIGENVALUE_UNDERFLOW_FLOOR, scale * _RELATIVE_DET_SINGULAR_TOL)
    return evals > rank_tol


def _axis_sigmas_from_fisher(F: np.ndarray) -> tuple[list[float], list[bool]]:
    """Return per-axis CRLB sigmas and singular flags without inventing nullspace precision."""
    F = np.asarray(F, dtype=float)
    F_sym = 0.5 * (F + F.T)
    dim = F_sym.shape[0]
    if F_sym.ndim != 2 or F_sym.shape[0] != F_sym.shape[1]:
        raise ValueError(f"Fisher matrix must be square; got {F_sym.shape}.")
    try:
        evals, evecs = np.linalg.eigh(F_sym)
    except np.linalg.LinAlgError:
        return [float("inf")] * dim, [True] * dim
    if not np.all(np.isfinite(evals)):
        return [float("inf")] * dim, [True] * dim
    positive = _positive_fisher_eigenvalue_mask(evals)
    if not np.any(positive):
        return [float("inf")] * dim, [True] * dim
    V = evecs[:, positive]
    inv_evals = 1.0 / evals[positive]
    F_pinv = (V * inv_evals) @ V.T
    range_projector = V @ V.T
    eye = np.eye(dim)
    sigmas: list[float] = []
    singular_axes: list[bool] = []
    for axis in range(dim):
        residual = eye[:, axis] - range_projector @ eye[:, axis]
        if np.linalg.norm(residual) > _FISHER_RANGE_RESIDUAL_TOL:
            sigmas.append(float("inf"))
            singular_axes.append(True)
        else:
            sigmas.append(float(np.sqrt(max(float(F_pinv[axis, axis]), 0.0))))
            singular_axes.append(False)
    return sigmas, singular_axes


def compute_registration_degradation_curve(
    per_modality_fisher: dict[str, np.ndarray],
    registration_covariances: list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """
    Evaluate the monotone fusion penalty from registration covariance.

    The clean theorem assumes positive-definite Fisher matrices. This diagnostic
    uses the same validation and pseudoinverse convention as fusion, so singular
    inputs remain inspectable but should not be used to claim the theorem.
    """
    if not isinstance(per_modality_fisher, dict) or not per_modality_fisher:
        raise ValueError("per_modality_fisher must be a non-empty dict.")
    if not registration_covariances:
        raise ValueError("registration_covariances must contain at least one covariance.")

    modalities = list(per_modality_fisher.keys())
    raw = {name: np.asarray(F, dtype=float) for name, F in per_modality_fisher.items()}
    ref_shape = raw[modalities[0]].shape
    if len(ref_shape) != 2 or ref_shape[0] != ref_shape[1] or ref_shape[0] < 2:
        raise ValueError(f"Fisher matrices must be square with at least 2 axes; got {ref_shape}.")
    for name, F in raw.items():
        if F.shape != ref_shape:
            raise ValueError(
                f"All Fisher matrices must have shape {ref_shape}; {name!r} has {F.shape}."
            )

    perfect_fisher = np.zeros(ref_shape, dtype=float)
    for F in raw.values():
        perfect_fisher = perfect_fisher + F
    perfect_sigma, _ = _sigma_xy_from_fisher(perfect_fisher)

    covariance_grid: list[np.ndarray] = []
    sigma_by_registration: list[float] = []
    gain_by_registration: list[float | None] = []
    degradation_fraction: list[float] = []
    adjusted_by_registration: list[dict[str, np.ndarray]] = []

    best_single_sigma = float("inf")
    for F in raw.values():
        sigma, singular = _sigma_xy_from_fisher(F)
        if not singular:
            best_single_sigma = min(best_single_sigma, sigma)

    for covariance in registration_covariances:
        sigma_reg = np.asarray(covariance, dtype=float)
        adjusted: dict[str, np.ndarray] = {}
        fusion = np.zeros(ref_shape, dtype=float)
        for name, F in raw.items():
            adjusted_F = _registration_adjusted_fisher(F, sigma_reg)
            adjusted[name] = adjusted_F
            fusion = fusion + adjusted_F
        sigma_xy, _ = _sigma_xy_from_fisher(fusion)
        covariance_grid.append(sigma_reg)
        sigma_by_registration.append(sigma_xy)
        adjusted_by_registration.append(adjusted)
        if np.isfinite(best_single_sigma) and np.isfinite(sigma_xy) and sigma_xy > 0.0:
            gain_by_registration.append(float(best_single_sigma / sigma_xy))
        else:
            gain_by_registration.append(None)
        if np.isfinite(perfect_sigma) and perfect_sigma > 0.0 and np.isfinite(sigma_xy):
            degradation_fraction.append(float((sigma_xy - perfect_sigma) / perfect_sigma))
        elif sigma_xy == perfect_sigma:
            degradation_fraction.append(0.0)
        else:
            degradation_fraction.append(float("inf"))

    monotone = all(
        sigma_by_registration[i + 1] >= sigma_by_registration[i] - 1e-12
        for i in range(len(sigma_by_registration) - 1)
    )

    return {
        "registration_covariance_grid": covariance_grid,
        "fusion_sigma_xy_nm_by_registration": sigma_by_registration,
        "fusion_gain_xy_by_registration": gain_by_registration,
        "monotone_degradation_verified": bool(monotone),
        "perfect_registration_fisher": perfect_fisher,
        "registration_degradation_fraction": degradation_fraction,
        "registration_adjusted_per_modality_fisher_by_registration": adjusted_by_registration,
    }


def _fusion_complementarity_metrics(
    per_modality_fisher: dict[str, np.ndarray],
    subset: list[str],
) -> dict[str, Any]:
    """
    Diagnostic metrics for whether fusion adds complementary directions.

    These metrics are descriptive, not part of the CRLB itself. They help
    distinguish "best pair is just the two strongest scalar contributors" from
    "modalities inform different parameter directions."
    """
    if len(subset) < 2:
        return {
            "mean_principal_angle_deg": 0.0,
            "max_principal_angle_deg": 0.0,
            "determinant_gain_vs_best_single": 1.0,
            "fused_condition_number": None,
        }

    matrices = [np.asarray(per_modality_fisher[m], dtype=float) for m in subset]
    F_sum = np.sum(matrices, axis=0)
    det_sum = float(np.linalg.det(F_sum))
    det_best = max(float(np.linalg.det(F)) for F in matrices)
    det_gain = (
        float(det_sum / det_best)
        if det_best > 0.0 and np.isfinite(det_sum) else float("inf")
    )

    vectors = []
    for F in matrices:
        vals, vecs = np.linalg.eigh(F)
        idx = int(np.argmax(vals))
        v = vecs[:, idx]
        norm = np.linalg.norm(v)
        vectors.append(v / norm if norm > 0.0 else v)

    angles: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot = abs(float(np.dot(vectors[i], vectors[j])))
            dot = min(max(dot, 0.0), 1.0)
            angles.append(float(np.degrees(np.arccos(dot))))

    eig_sum = np.linalg.eigvalsh(F_sum)
    positive = eig_sum[eig_sum > 0.0]
    condition = (
        float(positive[-1] / positive[0]) if positive.size == eig_sum.size else None
    )

    return {
        "mean_principal_angle_deg": float(np.mean(angles)) if angles else 0.0,
        "max_principal_angle_deg": float(np.max(angles)) if angles else 0.0,
        "determinant_gain_vs_best_single": det_gain,
        "fused_condition_number": condition,
    }


def compute_modality_fusion_crlb(
    contrast_by_modality: dict[str, np.ndarray],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    z_step_nm: float | None = None,
    subset_size: int | None = None,
    registration_covariance: np.ndarray | None = None,
) -> dict[str, Any]:
    r"""
    Multi-modality fusion Cramér-Rao lower bound.

    Given per-modality contrast images and noise-variance maps for the SAME
    particle coordinate frame, return:
      - the joint Fisher information matrix obtained by summing per-modality
        Fisher matrices under the independent-noise assumption,
      - the fusion CRLB on (x, y) (and optionally z),
      - the fusion gain against the best single modality.

    Parameters
    ----------
    contrast_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> per-particle contrast``. 2D mode expects
        ``(H, W)`` arrays; 3D mode (``z_step_nm`` supplied) expects
        ``(3, H, W)`` three-plane stacks. Shapes may differ across modalities
        when their pixel-size metadata puts the resulting Fisher matrices in
        the same physical coordinate frame.
    noise_variance_by_modality : dict[str, ndarray or float]
        Mapping ``modality_name -> per-pixel variance map (or scalar)``. Must
        cover the same keys as ``contrast_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies one pitch per modality.
    z_step_nm : float or None, default None
        If None: 2D fusion (2x2 Fisher per modality, 2x2 fused). If a
        positive float: 3D fusion (3x3 Fisher per modality, 3x3 fused).
    subset_size : int or None, default None
        If None: report the full N-modality fusion bound. If a positive
        integer k with 1 <= k <= N: enumerate all C(N, k) subsets of
        modalities of size k, compute each subset's fusion CRLB, and return
        the BEST subset (smallest fusion_sigma_xy_nm in 2D mode, smallest
        fusion_sigma_xyz_nm in 3D mode). The best-subset enumeration is
        intended for small modality sets.
    registration_covariance : ndarray or None, default None
        Optional 2x2 lateral registration-error covariance in nm^2. When
        supplied, each modality's contribution is corrected as
        ``F' = inv(inv(F) + Sigma_reg)`` before fusion.

    Returns
    -------
    result : dict
        A dictionary with the following keys:

        - ``per_modality_fisher`` : dict[str, ndarray]
            Per-modality Fisher matrix.
        - ``registration_adjusted_per_modality_fisher`` : dict[str, ndarray]
            Per-modality Fisher matrices after registration covariance.
        - ``per_modality_crlb`` : dict[str, dict]
            Per-modality CRLB result (output of compute_localization_crlb
            or compute_localization_crlb_3d). Modalities whose per-modality
            Fisher is singular have ``singular = True``.
        - ``registration_adjusted_per_modality_crlb`` : dict[str, dict]
            Single-modality baseline CRLBs after applying registration covariance;
            fusion gains are computed against these adjusted baselines.
        - ``fusion_fisher`` : ndarray
            The joint Fisher matrix (sum of per-modality F_M).
        - ``fusion_sigma_x_nm`` : float
        - ``fusion_sigma_y_nm`` : float
        - ``fusion_sigma_xy_nm`` : float (sqrt(sigma_x^2 + sigma_y^2),
          matching compute_localization_crlb's L2 sum convention)
        - ``fusion_sigma_z_nm`` : float (3D mode only; absent in 2D mode)
        - ``fusion_sigma_xyz_nm`` : float (3D mode only;
          sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2))
        - ``fusion_singular`` : bool — True iff the joint Fisher is
          numerically singular for the selected dimensional objective.
        - ``fusion_rank`` / ``fusion_axes_singular`` : observable rank and
          named singular axes for partial-rank diagnostics.
        - ``best_single_modality`` : str | None — name of the registration-adjusted
          per-modality CRLB minimizer on sigma_xy_nm (or sigma_xyz_nm in 3D).
          None if no single modality is non-singular.
        - ``best_single_xy_modality`` : str — lateral-only lowest-bound modality used
          for ``fusion_gain_xy``.
        - ``best_single_sigma_xy_nm`` : float (or None)
        - ``best_single_sigma_xyz_nm`` : float (3D mode only; or None)
        - ``fusion_gain_xy`` : float — sigma_xy^best_single / sigma_xy^fusion
          for the selected subset. In 3D subset searches the selected subset
          may be optimized for xyz precision, so this lateral ratio is reported
          as measured and may be below 1. None if no finite lateral comparison
          exists.
        - ``fusion_gain_xyz`` : float (3D mode only; analogous to xy gain)
        - ``fusion_complementarity`` : dict
            Pairwise and aggregate complementarity metrics for the fused Fisher
            matrices.
        - ``modalities_used`` : list[str] — modalities included in the
          fusion. Equals the full key set of the input dict if
          ``subset_size`` is None; the chosen best subset otherwise.
        - ``subset_size`` : int or None — echo of the input subset_size.
        - ``subset_search_count`` : int — number of subsets evaluated
          (1 if subset_size is None, C(N, subset_size) otherwise).
        - ``registration_covariance`` : ndarray or None
            Echo of the registration covariance used for adjusted baselines.

    Raises
    ------
    ValueError
        If the input dicts are empty, have mismatched keys, contain malformed
        contrast images/noise maps, or if subset_size is out of range.

    Notes
    -----
    The fusion bound assumes statistically independent measurements across
    modalities, which is the typical assumption for distinct detectors,
    disjoint spectral channels, or separate physical channels. A calibrated
    registration covariance can be supplied here; correlated detector noise
    would require a full cross-modality covariance model.
    """
    import itertools

    if not isinstance(contrast_by_modality, dict) or not contrast_by_modality:
        raise ValueError(
            "contrast_by_modality must be a non-empty dict keyed by modality name."
        )
    if not isinstance(noise_variance_by_modality, dict):
        raise ValueError("noise_variance_by_modality must be a dict.")
    if set(contrast_by_modality.keys()) != set(noise_variance_by_modality.keys()):
        missing = set(contrast_by_modality.keys()) ^ set(noise_variance_by_modality.keys())
        raise ValueError(
            f"contrast_by_modality and noise_variance_by_modality must have the "
            f"same keys; symmetric diff: {sorted(missing)!r}."
        )

    modalities = list(contrast_by_modality.keys())
    n = len(modalities)

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        modalities,
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )

    if subset_size is not None:
        if not isinstance(subset_size, int) or subset_size < 1 or subset_size > n:
            raise ValueError(
                f"subset_size must be an integer in [1, {n}]; got {subset_size!r}."
            )

    dim = 2 if z_step_nm is None else 3

    # Step 1: per-modality Fisher matrices.
    per_modality_fisher: dict[str, np.ndarray] = {}
    registration_adjusted_per_modality_fisher: dict[str, np.ndarray] = {}
    per_modality_crlb: dict[str, dict[str, Any]] = {}
    registration_adjusted_per_modality_crlb: dict[str, dict[str, Any]] = {}
    for m in modalities:
        c = np.asarray(contrast_by_modality[m], dtype=float)
        v = noise_variance_by_modality[m]
        F_m = _fisher_for_modality(c, v, pixel_sizes[m], z_step_nm)
        per_modality_fisher[m] = F_m
        F_adjusted = _registration_adjusted_fisher(
            F_m,
            registration_covariance,
        )
        registration_adjusted_per_modality_fisher[m] = F_adjusted
        sigma_xy_adjusted, xy_singular_adjusted = _sigma_xy_from_fisher(F_adjusted)
        adjusted_summary: dict[str, Any] = {
            "sigma_xy_nm": sigma_xy_adjusted,
            "xy_singular": xy_singular_adjusted,
            "fisher_matrix": F_adjusted,
        }
        if dim == 3:
            sigma_xyz_adjusted, xyz_singular_adjusted = _sigma_xyz_from_fisher(F_adjusted)
            adjusted_summary["sigma_xyz_nm"] = sigma_xyz_adjusted
            adjusted_summary["singular"] = xyz_singular_adjusted
        else:
            adjusted_summary["singular"] = xy_singular_adjusted
        registration_adjusted_per_modality_crlb[m] = adjusted_summary
        if z_step_nm is None:
            per_modality_crlb[m] = compute_localization_crlb(c, v, pixel_sizes[m])
        else:
            per_modality_crlb[m] = compute_localization_crlb_3d(
                c, v, pixel_sizes[m], z_step_nm=z_step_nm
            )

    # Step 2: identify best single-modality CRLB (skipping singular ones).
    # NOTE: per-modality CRLB result dicts use the module-wide `_nm` suffix
    # convention (see compute_localization_crlb / compute_localization_crlb_3d).
    if z_step_nm is None:
        sigma_key_single = "sigma_xy_nm"
    else:
        sigma_key_single = "sigma_xyz_nm"
    best_single_modality: str | None = None
    best_single_sigma: float | None = None
    best_single_xy_modality: str | None = None
    best_single_sigma_xy: float | None = None
    for m, crlb in registration_adjusted_per_modality_crlb.items():
        if crlb.get("singular", False):
            s = None
        else:
            s = crlb.get(sigma_key_single)
        if s is None or not np.isfinite(s):
            pass
        elif best_single_sigma is None or s < best_single_sigma:
            best_single_sigma = float(s)
            best_single_modality = m

        s_xy = crlb.get("sigma_xy_nm")
        if s_xy is not None and np.isfinite(s_xy) and (
            best_single_sigma_xy is None or float(s_xy) < best_single_sigma_xy
        ):
            best_single_sigma_xy = float(s_xy)
            best_single_xy_modality = m

    # Step 3: choose the modality subset to fuse.
    def _fuse_subset(subset: tuple[str, ...]) -> dict[str, Any]:
        """Inner helper: sum per-modality Fishers for a given subset and
        invert. Returns a dict with the fusion-side keys (without the gain
        comparison; the gain is computed once at the outer level).

        Convention: fusion-sigma keys carry the ``_nm`` suffix in line with
        the rest of fisher_diagnostic.py.  ``fusion_sigma_xy_nm`` is the
        L2 sum sqrt(sigma_x^2 + sigma_y^2) — the *total* 2-D bound, the same
        definition used in compute_localization_crlb above (NOT the rms).
        ``fusion_sigma_xyz_nm`` is the analogous L2 sum in 3-D."""
        F_sum = np.zeros((dim, dim), dtype=float)
        for s in subset:
            F_sum = F_sum + registration_adjusted_per_modality_fisher[s]
        F_sum = 0.5 * (F_sum + F_sum.T)
        axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F_sum)
        axis_names = _axis_names_for_dim(dim)
        axes_singular = {
            axis: bool(is_singular)
            for axis, is_singular in zip(axis_names, axis_singular)
        }
        rank = int(dim - sum(bool(flag) for flag in axis_singular[:dim]))
        if dim == 2:
            _, singular = _sigma_xy_from_fisher(F_sum)
        else:
            _, singular = _sigma_xyz_from_fisher(F_sum)
        out: dict[str, Any] = {
            "fusion_fisher": F_sum,
            "fusion_singular": singular,
            "fusion_rank": rank,
            "fusion_axes_singular": axes_singular,
            "fusion_xy_singular": bool(axis_singular[0] or axis_singular[1]),
            "modalities_used": list(subset),
        }
        if singular:
            if dim == 3:
                out["fusion_sigma_x_nm"] = axis_sigmas[0]
                out["fusion_sigma_y_nm"] = axis_sigmas[1]
                out["fusion_sigma_z_nm"] = axis_sigmas[2]
                out["fusion_sigma_xy_nm"] = (
                    float(np.sqrt(axis_sigmas[0] ** 2 + axis_sigmas[1] ** 2))
                    if np.isfinite(axis_sigmas[0]) and np.isfinite(axis_sigmas[1])
                    else float("inf")
                )
                out["fusion_sigma_xyz_nm"] = (
                    float(np.sqrt(sum(sigma ** 2 for sigma in axis_sigmas)))
                    if all(np.isfinite(sigma) for sigma in axis_sigmas)
                    else float("inf")
                )
            else:
                out["fusion_sigma_x_nm"] = axis_sigmas[0]
                out["fusion_sigma_y_nm"] = axis_sigmas[1]
                out["fusion_sigma_xy_nm"] = float("inf")
            return out
        cov = np.linalg.inv(F_sum)
        sigma_x = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(cov[1, 1], 0.0)))
        out["fusion_sigma_x_nm"] = sigma_x
        out["fusion_sigma_y_nm"] = sigma_y
        out["fusion_sigma_xy_nm"] = float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
        if dim == 3:
            sigma_z = float(np.sqrt(max(cov[2, 2], 0.0)))
            out["fusion_sigma_z_nm"] = sigma_z
            out["fusion_sigma_xyz_nm"] = float(
                np.sqrt(sigma_x ** 2 + sigma_y ** 2 + sigma_z ** 2)
            )
        return out

    if subset_size is None:
        chosen_subset = tuple(modalities)
        fused = _fuse_subset(chosen_subset)
        xy_optimized_fused = fused
        subset_search_count = 1
    else:
        # Exhaustive enumeration; intended for small N.
        best = None
        best_xy = None
        sigma_key_fused = (
            "fusion_sigma_xy_nm" if dim == 2 else "fusion_sigma_xyz_nm"
        )
        count = 0
        for subset in itertools.combinations(modalities, subset_size):
            count += 1
            cand = _fuse_subset(subset)
            if (
                not cand["fusion_xy_singular"]
                and np.isfinite(cand["fusion_sigma_xy_nm"])
                and (
                    best_xy is None
                    or cand["fusion_sigma_xy_nm"] < best_xy["fusion_sigma_xy_nm"]
                )
            ):
                best_xy = cand
            if cand["fusion_singular"]:
                continue
            if best is None or cand[sigma_key_fused] < best[sigma_key_fused]:
                best = cand
        subset_search_count = count
        if best is None:
            # Every subset of size k was singular; report a singular
            # result using the first enumerated subset.
            chosen_subset = next(iter(itertools.combinations(modalities, subset_size)))
            fused = _fuse_subset(chosen_subset)
        else:
            fused = best
        xy_optimized_fused = best_xy

    # Step 4: assemble result + fusion-gain comparisons.
    result: dict[str, Any] = {
        "per_modality_fisher": per_modality_fisher,
        "registration_adjusted_per_modality_fisher": (
            registration_adjusted_per_modality_fisher
        ),
        "per_modality_crlb": per_modality_crlb,
        "registration_adjusted_per_modality_crlb": (
            registration_adjusted_per_modality_crlb
        ),
        "fusion_fisher": fused["fusion_fisher"],
        "fusion_sigma_x_nm": fused["fusion_sigma_x_nm"],
        "fusion_sigma_y_nm": fused["fusion_sigma_y_nm"],
        "fusion_sigma_xy_nm": fused["fusion_sigma_xy_nm"],
        "fusion_singular": fused["fusion_singular"],
        "fusion_rank": fused["fusion_rank"],
        "fusion_axes_singular": fused["fusion_axes_singular"],
        "fusion_xy_singular": fused["fusion_xy_singular"],
        "best_single_modality": best_single_modality,
        "best_single_xy_modality": best_single_xy_modality,
        "best_single_sigma_xy_nm": best_single_sigma_xy,
        "modalities_used": fused["modalities_used"],
        "xy_optimized_modalities_used": (
            None if xy_optimized_fused is None else xy_optimized_fused["modalities_used"]
        ),
        "fusion_complementarity": _fusion_complementarity_metrics(
            registration_adjusted_per_modality_fisher,
            fused["modalities_used"],
        ),
        "subset_size": subset_size,
        "subset_search_count": subset_search_count,
        "registration_covariance": (
            None
            if registration_covariance is None
            else np.asarray(registration_covariance, dtype=float)
        ),
        "pixel_size_nm_by_modality": pixel_sizes,
    }
    if dim == 3:
        result["fusion_sigma_z_nm"] = fused["fusion_sigma_z_nm"]
        result["fusion_sigma_xyz_nm"] = fused["fusion_sigma_xyz_nm"]
        result["best_single_sigma_xyz_nm"] = (
            best_single_sigma if best_single_modality is not None else None
        )

    # Fusion gain on the lateral (xy) plane.
    if (
        best_single_sigma_xy is not None
        and not fused["fusion_xy_singular"]
        and fused["fusion_sigma_xy_nm"] > 0.0
        and np.isfinite(fused["fusion_sigma_xy_nm"])
    ):
        result["fusion_gain_xy"] = float(
            best_single_sigma_xy / fused["fusion_sigma_xy_nm"]
        )
    else:
        result["fusion_gain_xy"] = None
    result["fusion_gain_xy_semantics"] = "selected_subset_lateral_ratio"
    if (
        best_single_sigma_xy is not None
        and xy_optimized_fused is not None
        and not xy_optimized_fused["fusion_xy_singular"]
        and xy_optimized_fused["fusion_sigma_xy_nm"] > 0.0
        and np.isfinite(xy_optimized_fused["fusion_sigma_xy_nm"])
    ):
        result["fusion_sigma_xy_optimized_nm"] = xy_optimized_fused["fusion_sigma_xy_nm"]
        result["fusion_gain_xy_optimized"] = float(
            best_single_sigma_xy / xy_optimized_fused["fusion_sigma_xy_nm"]
        )
    else:
        result["fusion_sigma_xy_optimized_nm"] = None
        result["fusion_gain_xy_optimized"] = None

    if dim == 3:
        if (
            best_single_modality is not None
            and not fused["fusion_singular"]
            and fused["fusion_sigma_xyz_nm"] > 0.0
        ):
            result["fusion_gain_xyz"] = float(
                best_single_sigma / fused["fusion_sigma_xyz_nm"]
            )
        else:
            result["fusion_gain_xyz"] = None

    return result


def _scalarize_crlb(F: np.ndarray, objective: str) -> float:
    r"""Scalar criterion on the CRLB matrix F^{-1}, lower-is-better.

    Parameters
    ----------
    F : ndarray, shape (d, d)
        Symmetric positive-semidefinite Fisher matrix.
    objective : {"A", "D", "E", "trace"}
        Optimality criterion.
            - ``A`` (or ``trace``): minimise tr(F^{-1}). The A-criterion is
              the sum of per-axis CRLBs and is the standard scalarization
              when "average" precision across axes is the goal.
            - ``D``: minimise -log det(F). Equivalent to minimising the
              volume of the CRLB confidence ellipsoid; also equal to
              maximising the Shannon information of the joint position
              estimate (up to additive constants).
            - ``E``: minimise lambda_max(F^{-1}) = 1 / lambda_min(F).
              Penalises the worst single-axis CRLB.

    Returns
    -------
    val : float
        Scalar criterion. Returns ``+inf`` if F is numerically singular.
    """
    obj = objective.upper()
    d = F.shape[0]
    # Singularity check (relative to trace, scale-invariant).
    trace = float(np.trace(F))
    det = float(np.linalg.det(F))
    if (
        not np.isfinite(det)
        or trace <= 0.0
        or abs(det) < (trace ** d) * _RELATIVE_DET_SINGULAR_TOL
    ):
        return float("inf")
    if obj in ("A", "TRACE"):
        cov = np.linalg.inv(F)
        return float(np.trace(cov))
    if obj == "D":
        return float(-np.log(det))
    if obj == "E":
        eig = np.linalg.eigvalsh(F)
        lam_min = float(eig[0])
        if lam_min <= 0.0:
            return float("inf")
        return 1.0 / lam_min
    raise ValueError(
        f"Unknown objective {objective!r}; expected one of A, D, E, trace."
    )


def compute_loewner_dominance(
    per_modality_fisher: dict[str, np.ndarray],
    *,
    per_modality_dt_seconds: dict[str, float] | None = None,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compute strict Loewner dominance among per-modality information rates."""
    if not isinstance(per_modality_fisher, dict) or not per_modality_fisher:
        raise ValueError("per_modality_fisher must be a non-empty dict.")
    modalities = list(per_modality_fisher.keys())
    if per_modality_dt_seconds is None:
        per_modality_dt_seconds = {m: 1.0 for m in modalities}
    if set(per_modality_dt_seconds) != set(modalities):
        sym = set(per_modality_dt_seconds) ^ set(modalities)
        raise ValueError(
            "per_modality_dt_seconds keys must match per_modality_fisher; "
            f"symmetric diff: {sorted(sym)!r}."
        )

    ref = np.asarray(per_modality_fisher[modalities[0]], dtype=float)
    if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError(f"Fisher matrices must be square 2D arrays; got {ref.shape}.")
    rates: dict[str, np.ndarray] = {}
    for name in modalities:
        dt = float(per_modality_dt_seconds[name])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError(f"per_modality_dt_seconds[{name!r}] must be positive.")
        F = np.asarray(per_modality_fisher[name], dtype=float)
        if F.shape != ref.shape:
            raise ValueError(
                f"All Fisher matrices must have shape {ref.shape}; {name!r} has {F.shape}."
            )
        rates[name] = F / dt

    dominates: dict[str, list[str]] = {name: [] for name in modalities}
    dominated_by: dict[str, list[str]] = {name: [] for name in modalities}
    eig_min: dict[str, dict[str, float]] = {name: {} for name in modalities}
    atol = float(atol)
    for dominant in modalities:
        for dominated in modalities:
            if dominant == dominated:
                continue
            diff = rates[dominant] - rates[dominated]
            vals = np.linalg.eigvalsh(0.5 * (diff + diff.T))
            min_eval = float(vals[0])
            eig_min[dominant][dominated] = min_eval
            strictly_positive_somewhere = bool(np.max(vals) > atol)
            if min_eval >= -atol and strictly_positive_somewhere:
                dominates[dominant].append(dominated)
                dominated_by[dominated].append(dominant)

    maximal = [name for name in modalities if not dominated_by[name]]
    return {
        "information_rate_by_modality": rates,
        "dominates": dominates,
        "dominated_by": dominated_by,
        "loewner_maximal_modalities": maximal,
        "dominance_eigenvalue_min": eig_min,
    }


def compute_optimal_time_allocation_crlb(
    per_modality_fisher_per_frame: dict[str, np.ndarray],
    *,
    per_modality_dt_seconds: dict[str, float] | None = None,
    total_time_seconds: float = 1.0,
    objective: str = "A",
    min_fraction: float = 0.0,
    max_iters: int = 200,
    tol: float = 1e-9,
    prune_loewner_dominated: bool = False,
) -> dict[str, Any]:
    r"""Optimal time-slicing CRLB allocator.

    Given per-modality per-frame Fisher matrices ``F_M`` and per-modality
    per-frame time costs ``dt_M``, find the time allocation
    :math:`\{t_M\}` that minimises a chosen scalar criterion of the joint
    CRLB under a fixed total-time budget :math:`T = \sum_M t_M`.

    The joint Fisher information at allocation :math:`\{t_M\}` is, under the
    independent-noise assumption,

    .. math::
        \mathbf{F}_{\mathrm{total}}(\mathbf{t}) = \sum_M (t_M / dt_M)\,\mathbf{F}_M ,

    so :math:`\mathbf{F}_{\mathrm{total}}` is linear in the time fractions.
    The criterion :math:`\Phi(\mathbf{F}_{\mathrm{total}}^{-1})` is convex in
    :math:`\mathbf{F}_{\mathrm{total}}` for the standard A-, D-, and
    E-optimality scalarizations, so the resulting problem is a convex program
    on the simplex :math:`\{\mathbf{t} : \sum t_M = T,\, t_M \ge 0\}`.

    The implementation uses projected-gradient (Frank-Wolfe-style) iteration
    that needs only NumPy. Termination is by relative-criterion change.

    Parameters
    ----------
    per_modality_fisher_per_frame : dict[str, ndarray]
        Mapping ``modality_name -> per-frame Fisher matrix``. All matrices
        must share the same shape ``(d, d)`` (typically d = 2 or 3).
    per_modality_dt_seconds : dict[str, float] or None
        Mapping ``modality_name -> seconds per frame`` for each modality.
        If None, all dt are taken to be 1 (so the budget is interpreted
        as total frames).
    total_time_seconds : float, default 1.0
        Total acquisition-time budget T. If ``per_modality_dt_seconds`` is
        None, this is the total frame count instead.
    objective : str, default "A"
        Optimality criterion: ``"A"`` (= ``"trace"``), ``"D"``, or ``"E"``.
        See ``_scalarize_crlb``.
    min_fraction : float, default 0.0
        Lower bound on each time fraction t_M / T. Enforces a per-modality
        minimum-acquisition floor (e.g. 0.05 = 5 % of T per modality).
    prune_loewner_dominated : bool, default False
        If True and min_fraction is zero, remove modalities whose information
        rate F_M / dt_M is strictly Loewner-dominated before solving, then
        report zero time for the removed modalities. Pruning is disabled when
        min_fraction > 0 because the nonzero floor must be honored.
    max_iters : int, default 200
        Hard cap on projected-gradient iterations.
    tol : float, default 1e-9
        Relative-criterion termination tolerance.

    Returns
    -------
    result : dict
        Keys:

        - ``optimal_time_seconds`` : dict[str, float] — optimal t_M.
        - ``optimal_frames`` : dict[str, float] — optimal n_M = t_M / dt_M
          (real-valued; integer rounding is a downstream concern).
        - ``optimal_fisher`` : ndarray — F_total at the optimum.
        - ``optimal_sigma_x_nm`` : float
        - ``optimal_sigma_y_nm`` : float
        - ``optimal_sigma_xy_nm`` : float (sqrt(sigma_x^2 + sigma_y^2))
        - ``optimal_sigma_z_nm`` : float (3D mode only; absent in 2D mode)
        - ``optimal_sigma_xyz_nm`` : float (3D mode only)
        - ``optimal_objective_value`` : float — the scalar criterion at the
          optimum.
        - ``baseline_uniform_objective`` : float — criterion value when the
          time budget is split equally across modalities (consistency check).
        - ``baseline_best_single_objective`` : float — criterion value when
          the entire budget is allocated to the per-modality
          single-modality minimiser.
        - ``allocation_gain_vs_uniform`` : float — ratio
          baseline_uniform / optimal (>= 1; reports how much better the
          allocator is than equal-split).
        - ``allocation_gain_vs_best_single`` : float — analogous ratio for
          the all-budget-to-best-single baseline.
        - ``best_single_modality`` : str | None — the per-modality minimiser of
          the criterion.
        - ``num_iterations`` : int — projected-gradient iterations used.
        - ``converged`` : bool — True when the allocator terminates before
          ``max_iters`` through the tolerance criterion, a no-descent condition,
          or a better feasible baseline allocation.
        - ``termination_reason`` : str — reason associated with ``converged`` or
          ``"max_iters"`` when the iteration cap is reached.
        - ``modalities`` : list[str] — input modality order.
        - ``objective`` : str — echoed criterion identifier.

    Raises
    ------
    ValueError
        If the input dicts are empty or have mismatched keys, if the
        Fisher matrices have inconsistent shape, if total_time_seconds
        or any dt_M is non-positive, or if min_fraction is out of [0, 1/N].

    Notes
    -----
    This bound is NOT a fusion bound (which assumes simultaneous
    measurement on independent detectors; see
    ``compute_modality_fusion_crlb``). It is a scheduling bound for allocating
    a fixed total acquisition time across modalities to maximise final
    precision. The two bounds answer
    complementary engineering questions: fusion evaluates
    simultaneous-channel fusion under explicit co-acquisition assumptions,
    while this routine prescribes an exposure schedule.
    Both can be combined by treating fused channels as a single virtual
    modality with the fused per-frame Fisher matrix.
    """
    if not isinstance(per_modality_fisher_per_frame, dict) or not per_modality_fisher_per_frame:
        raise ValueError(
            "per_modality_fisher_per_frame must be a non-empty dict keyed by modality name."
        )
    modalities = list(per_modality_fisher_per_frame.keys())
    n = len(modalities)

    if per_modality_dt_seconds is None:
        per_modality_dt_seconds = {m: 1.0 for m in modalities}
    if set(per_modality_dt_seconds.keys()) != set(modalities):
        sym = set(per_modality_dt_seconds.keys()) ^ set(modalities)
        raise ValueError(
            f"per_modality_dt_seconds keys must match per_modality_fisher_per_frame; "
            f"symmetric diff: {sorted(sym)!r}."
        )
    try:
        total_time_seconds = float(total_time_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"total_time_seconds must be a finite positive scalar; got {total_time_seconds!r}."
        ) from exc
    if not np.isfinite(total_time_seconds) or total_time_seconds <= 0.0:
        raise ValueError(
            f"total_time_seconds must be finite and positive; got {total_time_seconds!r}."
        )
    dt_seconds: dict[str, float] = {}
    for m, dt in per_modality_dt_seconds.items():
        try:
            dt_value = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"per_modality_dt_seconds[{m!r}] must be a finite positive scalar; got {dt!r}."
            ) from exc
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError(
                f"per_modality_dt_seconds[{m!r}] must be finite and positive; got {dt!r}."
            )
        dt_seconds[m] = dt_value
    per_modality_dt_seconds = dt_seconds
    try:
        min_fraction = float(min_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"min_fraction must be a finite scalar in [0, 1/N]; got {min_fraction!r}."
        ) from exc
    if (
        not np.isfinite(min_fraction)
        or min_fraction < 0.0
        or min_fraction * n > 1.0
    ):
        raise ValueError(
            f"min_fraction must be finite and lie in [0, 1/N]; got {min_fraction!r} for N={n}."
        )

    # Fisher matrix shape consistency.
    F0 = np.asarray(per_modality_fisher_per_frame[modalities[0]], dtype=float)
    if F0.ndim != 2 or F0.shape[0] != F0.shape[1]:
        raise ValueError(
            f"Per-modality Fisher must be square 2-D; got shape {F0.shape} for "
            f"{modalities[0]!r}."
        )
    d = F0.shape[0]
    F_arr: dict[str, np.ndarray] = {}
    for m in modalities:
        F_m = np.asarray(per_modality_fisher_per_frame[m], dtype=float)
        if F_m.shape != (d, d):
            raise ValueError(
                f"Per-modality Fisher shape mismatch: {m!r} is {F_m.shape}, "
                f"expected {(d, d)}."
            )
        if not np.all(np.isfinite(F_m)):
            raise ValueError(f"Per-modality Fisher for {m!r} must contain only finite values.")
        F_arr[m] = F_m

    dt_arr = np.array([per_modality_dt_seconds[m] for m in modalities], dtype=float)
    T = float(total_time_seconds)

    loewner_dominance = compute_loewner_dominance(
        F_arr,
        per_modality_dt_seconds=per_modality_dt_seconds,
    )
    loewner_pruned_modalities: list[str] = []
    loewner_pruning_applied = False
    if bool(prune_loewner_dominated) and min_fraction == 0.0:
        maximal = list(loewner_dominance["loewner_maximal_modalities"])
        loewner_pruned_modalities = [m for m in modalities if m not in maximal]
        if loewner_pruned_modalities and maximal:
            sub_result = compute_optimal_time_allocation_crlb(
                {m: F_arr[m] for m in maximal},
                per_modality_dt_seconds={m: per_modality_dt_seconds[m] for m in maximal},
                total_time_seconds=total_time_seconds,
                objective=objective,
                min_fraction=0.0,
                max_iters=max_iters,
                tol=tol,
                prune_loewner_dominated=False,
            )
            expanded_time = {m: 0.0 for m in modalities}
            expanded_frames = {m: 0.0 for m in modalities}
            for m in maximal:
                expanded_time[m] = float(sub_result["optimal_time_seconds"][m])
                expanded_frames[m] = float(sub_result["optimal_frames"][m])
            sub_result["optimal_time_seconds"] = expanded_time
            sub_result["optimal_frames"] = expanded_frames
            sub_result["modalities"] = modalities
            sub_result["loewner_dominance"] = loewner_dominance
            sub_result["loewner_pruning_applied"] = True
            sub_result["loewner_pruned_modalities"] = loewner_pruned_modalities
            return sub_result

    # Convenience: F_total at allocation t (vector).
    def F_total_of(t: np.ndarray) -> np.ndarray:
        F = np.zeros((d, d), dtype=float)
        for k, m in enumerate(modalities):
            F = F + (t[k] / dt_arr[k]) * F_arr[m]
        return F

    def phi_of(t: np.ndarray) -> float:
        return _scalarize_crlb(F_total_of(t), objective)

    # ----- analytic gradient of the chosen criterion in t -----
    # F_total = sum_k (t_k / dt_k) F_k
    # dF_total/dt_k = F_k / dt_k
    #
    # A: phi = tr(F^-1).   d phi / d t_k = -tr(F^-1 (F_k/dt_k) F^-1)
    # D: phi = -log det F. d phi / d t_k = -tr(F^-1 (F_k/dt_k))
    # E: phi = 1 / lam_min(F).
    #     Let v = unit eigenvector of lam_min. d lam_min / d t_k =
    #         v^T (F_k/dt_k) v. Then d phi / d t_k = -d lam_min/d t_k / lam_min^2.
    def grad_phi(t: np.ndarray, F_total: np.ndarray) -> np.ndarray:
        obj = objective.upper()
        g = np.zeros(n, dtype=float)
        # Singularity short-circuit: if F_total is not invertible, the gradient
        # of the criterion is ill-defined; return zeros so the line-search
        # treats the direction as non-descent and the loop terminates.
        trace_F = float(np.trace(F_total))
        det_F = float(np.linalg.det(F_total))
        if (
            (not np.isfinite(det_F))
            or (trace_F <= 0.0)
            or (abs(det_F) < (trace_F ** d) * _RELATIVE_DET_SINGULAR_TOL)
        ):
            return g
        if obj in ("A", "TRACE"):
            cov = np.linalg.inv(F_total)
            cov2 = cov @ cov  # F^-1 F^-1; trace(F^-1 X F^-1) = trace(X cov2)
            for k, m in enumerate(modalities):
                g[k] = -float(np.trace(F_arr[m] @ cov2)) / dt_arr[k]
            return g
        if obj == "D":
            cov = np.linalg.inv(F_total)
            for k, m in enumerate(modalities):
                g[k] = -float(np.trace(cov @ F_arr[m])) / dt_arr[k]
            return g
        if obj == "E":
            eig_w, eig_V = np.linalg.eigh(F_total)
            lam_min = float(eig_w[0])
            v = eig_V[:, 0]
            if lam_min <= 0.0:
                # Subgradient of 1/lam_min at lam_min = 0 is undefined;
                # return zero step.
                return g
            for k, m in enumerate(modalities):
                d_lam = float(v @ F_arr[m] @ v) / dt_arr[k]
                g[k] = -d_lam / (lam_min * lam_min)
            return g
        raise ValueError(f"Unknown objective {objective!r}.")

    # ----- simplex projection with lower bound min_fraction * T -----
    def project_simplex(y: np.ndarray, total: float, lb: float) -> np.ndarray:
        # Minimum-distance projection of y onto {x : sum x = total, x >= lb}.
        # Reduce to plain simplex by substitution x = lb + z, z >= 0, sum z = total - n*lb.
        n_loc = y.shape[0]
        rem = total - n_loc * lb
        if rem < -1e-12:
            raise ValueError("min_fraction*T*N exceeds total budget.")
        z_target = y - lb
        # Standard sort-based simplex projection.
        u = np.sort(z_target)[::-1]
        cssv = np.cumsum(u) - rem
        ind = np.arange(1, n_loc + 1)
        cond = u - cssv / ind > 0
        if not np.any(cond):
            rho = n_loc
        else:
            rho = int(np.where(cond)[0].max() + 1)
        theta = cssv[rho - 1] / float(rho)
        z = np.maximum(z_target - theta, 0.0)
        return z + lb

    # ----- initialisation: equal split (respecting lower bound) -----
    t = np.full(n, T / n, dtype=float)
    t = project_simplex(t, T, min_fraction * T)

    converged = False
    termination_reason = "max_iters"
    last_phi = phi_of(t)
    iters = 0
    for _ in range(max_iters):
        iters += 1
        F_total = F_total_of(t)
        g = grad_phi(t, F_total)
        # Reduced gradient: project step direction onto simplex tangent.
        # Use Frank-Wolfe step: best vertex argmin_v g . v subject to v in feasible polytope.
        # For our simplex with lower bound, vertices are (T - (n-1)*lb*T) at one coord,
        # lb*T at all others. Pick the vertex that minimizes g.v.
        lb = min_fraction * T
        rem_top = T - (n - 1) * lb
        # FW vertex chooses k_star = argmin_k g_k * (rem_top - lb) (others all at lb)
        k_star = int(np.argmin(g))
        v = np.full(n, lb, dtype=float)
        v[k_star] = rem_top
        # Search direction d = v - t.
        direction = v - t
        # Armijo backtracking step size on alpha in (0, 1].
        gtd = float(g @ direction)
        if gtd >= -_LINE_SEARCH_DESCENT_TOL:
            # No descent direction found.
            converged = True
            termination_reason = "no_descent_direction"
            break
        alpha = 1.0
        new_phi = last_phi
        for _ls in range(_LINE_SEARCH_MAX_STEPS):
            t_new = t + alpha * direction
            # numerical safety: re-project to handle rounding
            t_new = project_simplex(t_new, T, lb)
            phi_new = phi_of(t_new)
            if phi_new <= last_phi + _LINE_SEARCH_ARMIJO_C * alpha * gtd:
                new_phi = phi_new
                t = t_new
                break
            alpha *= _LINE_SEARCH_SHRINK
        # Termination: relative change in objective.
        if last_phi != 0.0 and abs(last_phi - new_phi) <= tol * max(1.0, abs(last_phi)):
            converged = True
            termination_reason = "objective_tolerance"
            last_phi = new_phi
            break
        last_phi = new_phi

    # Compare the Frank-Wolfe allocation against feasible baseline allocations
    # before reporting the final split. This preserves the expected dominance
    # relationships for best-single and uniform baselines under non-smooth
    # objectives such as E-optimality.
    lb_post = min_fraction * T
    candidate_allocations: list[tuple[float, np.ndarray, str]] = [
        (last_phi, t.copy(), "frank_wolfe"),
    ]
    for k_corner, m_corner in enumerate(modalities):
        t_corner = np.full(n, lb_post, dtype=float)
        t_corner[k_corner] = T - lb_post * (n - 1)
        phi_corner = phi_of(t_corner)
        candidate_allocations.append((phi_corner, t_corner, f"single:{m_corner}"))
    t_uniform_post = np.full(n, T / n, dtype=float)
    t_uniform_post = project_simplex(t_uniform_post, T, lb_post)
    candidate_allocations.append(
        (phi_of(t_uniform_post), t_uniform_post, "uniform"),
    )
    candidate_allocations.sort(key=lambda triple: triple[0])
    best_phi, best_t, best_source = candidate_allocations[0]
    if best_source != "frank_wolfe":
        # Adopt the best closed-form feasible baseline when it improves on the
        # Frank-Wolfe allocation.
        t = best_t
        last_phi = best_phi
        converged = True
        termination_reason = f"baseline_candidate:{best_source}"

    # ----- assemble result -----
    optimal_time_seconds = {modalities[k]: float(t[k]) for k in range(n)}
    optimal_frames = {modalities[k]: float(t[k] / dt_arr[k]) for k in range(n)}
    F_opt = F_total_of(t)
    # Singularity check on the optimum's joint Fisher matrix.
    # Use the same scale-invariant determinant criterion as elsewhere in
    # this module so an all-zero Fisher input does not crash np.linalg.inv.
    trace_opt = float(np.trace(F_opt))
    det_opt = float(np.linalg.det(F_opt))
    fisher_singular = (
        (not np.isfinite(det_opt))
        or (trace_opt <= 0.0)
        or (abs(det_opt) < (trace_opt ** d) * _RELATIVE_DET_SINGULAR_TOL)
    )
    if fisher_singular:
        cov_opt = None
        sigma_x = float("inf")
        sigma_y = float("inf")
    else:
        cov_opt = np.linalg.inv(F_opt)
        sigma_x = float(np.sqrt(max(cov_opt[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(cov_opt[1, 1], 0.0)))

    result: dict[str, Any] = {
        "optimal_time_seconds": optimal_time_seconds,
        "optimal_frames": optimal_frames,
        "optimal_fisher": F_opt,
        "optimal_fisher_singular": fisher_singular,
        "optimal_sigma_x_nm": sigma_x,
        "optimal_sigma_y_nm": sigma_y,
        "optimal_sigma_xy_nm": (
            float("inf") if fisher_singular
            else float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
        ),
        "optimal_objective_value": float(last_phi),
        "modalities": modalities,
        "objective": objective.upper(),
        "num_iterations": iters,
        "converged": bool(converged),
        "termination_reason": termination_reason,
        "loewner_dominance": loewner_dominance,
        "loewner_pruning_applied": loewner_pruning_applied,
        "loewner_pruned_modalities": loewner_pruned_modalities,
    }
    if d == 3:
        if fisher_singular:
            result["optimal_sigma_z_nm"] = float("inf")
            result["optimal_sigma_xyz_nm"] = float("inf")
        else:
            sigma_z = float(np.sqrt(max(cov_opt[2, 2], 0.0)))
            result["optimal_sigma_z_nm"] = sigma_z
            result["optimal_sigma_xyz_nm"] = float(
                np.sqrt(sigma_x ** 2 + sigma_y ** 2 + sigma_z ** 2)
            )

    # ----- baseline 1: uniform split -----
    t_uniform = np.full(n, T / n, dtype=float)
    t_uniform = project_simplex(t_uniform, T, min_fraction * T)
    phi_uniform = phi_of(t_uniform)
    result["baseline_uniform_objective"] = float(phi_uniform)

    # ----- baseline 2: all budget to best single modality -----
    best_single_phi = float("inf")
    best_single_m: str | None = None
    for m in modalities:
        # All budget to this modality; respect min_fraction floor on others.
        t_single = np.full(n, min_fraction * T, dtype=float)
        idx = modalities.index(m)
        t_single[idx] = T - min_fraction * T * (n - 1)
        phi_m = phi_of(t_single)
        if phi_m < best_single_phi:
            best_single_phi = phi_m
            best_single_m = m
    result["baseline_best_single_objective"] = float(best_single_phi)
    result["best_single_modality"] = best_single_m

    # ----- gains over baselines (>= 1 means optimal beats baseline) -----
    obj = objective.upper()
    if obj == "D":
        # D-criterion is -log det F; "improvement" is reduction. Express as
        # exp(baseline - optimal), which is the volume-ratio improvement.
        if np.isfinite(phi_uniform) and np.isfinite(last_phi):
            result["allocation_gain_vs_uniform"] = float(
                np.exp(phi_uniform - last_phi)
            )
        else:
            result["allocation_gain_vs_uniform"] = None
        if np.isfinite(best_single_phi) and np.isfinite(last_phi):
            result["allocation_gain_vs_best_single"] = float(
                np.exp(best_single_phi - last_phi)
            )
        else:
            result["allocation_gain_vs_best_single"] = None
    else:
        # A/E criteria are positive; gain is a simple ratio.
        if np.isfinite(phi_uniform) and last_phi > 0.0 and np.isfinite(last_phi):
            result["allocation_gain_vs_uniform"] = float(phi_uniform / last_phi)
        else:
            result["allocation_gain_vs_uniform"] = None
        if np.isfinite(best_single_phi) and last_phi > 0.0 and np.isfinite(last_phi):
            result["allocation_gain_vs_best_single"] = float(
                best_single_phi / last_phi
            )
        else:
            result["allocation_gain_vs_best_single"] = None

    return result
