"""Axial Fisher-information and z-localization CRLB implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from noise_contracts import (
    AnalysisNoiseModel,
    IndependentPixelNoiseModel,
    analysis_noise_model_from_likelihood,
)

from ._constants import _FISHER_DET_EPS, _RELATIVE_DET_SINGULAR_TOL
from .precision import compute_fisher_from_gradients_with_noise
from ._metadata_helpers import (
    _derivative_unit,
    _diagnostic_metadata_aliases,
    _fisher_rank_metadata,
    _localization_derivative_metadata,
    _variance_units,
)
from .lateral import (
    _lateral_coordinate_derivatives,
    _reject_correlated_noise_for_diagonal_fisher,
    _resolve_analysis_noise_input,
)
from unit_contracts import assert_compatible


def _adaptive_crlb_from_steps(
    steps_by_size: dict[float, Any],
    compute_result_for_step,
    metric_key: str,
    *,
    convergence_tolerance: float,
    min_stable_steps: int,
    source_contract: str,
    modality: str,
) -> dict[str, Any]:
    """Generic adaptive finite-difference sweep used by axial rerender stacks."""

    from .convergence import (
        _matrix_rank_condition,
        _relative_span,
        _select_convergence_status,
        annotate_fisher_result_status,
    )

    per_step: list[dict[str, Any]] = []
    for step in sorted((float(s) for s in steps_by_size), reverse=True):
        result = compute_result_for_step(step, steps_by_size[step])
        fisher = np.asarray(result["fisher_matrix"], dtype=float)
        rank, cond, axes = _matrix_rank_condition(fisher)
        item = dict(result)
        item.update(
            {
                "derivative_step": float(step),
                "fisher_rank": rank,
                "condition_number": cond,
                "singular_axes": axes,
            }
        )
        per_step.append(item)

    status, selected, reason = _select_convergence_status(
        per_step,
        metric_key,
        convergence_tolerance,
        min_stable_steps,
    )
    selected_result = next(
        (
            item
            for item in per_step
            if selected is not None and float(item["derivative_step"]) == float(selected)
        ),
        per_step[-1] if per_step else {},
    )
    final = annotate_fisher_result_status(
        selected_result,
        convergence_status=status,
        source_contract=source_contract,
        modality=modality,
    )
    return {
        "convergence_status": status,
        "selected_step_nm": selected,
        "candidate_steps_nm": [float(item["derivative_step"]) for item in per_step],
        "relative_crlb_span": _relative_span(
            [float(item.get(metric_key, float("nan"))) for item in per_step]
        ),
        "rank_range": [
            min([int(item.get("fisher_rank", 0)) for item in per_step], default=0),
            max([int(item.get("fisher_rank", 0)) for item in per_step], default=0),
        ],
        "reason": reason,
        "per_step_results": per_step,
        "final_result": final,
    }

def _localization_3d_derivative_metadata(
    pixel_size_nm: float,
    z_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata for the lateral plus axial derivative convention."""
    meta = _localization_derivative_metadata(
        pixel_size_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    meta.update(
        {
            "state_axes": ["x", "y", "z"],
            "axis_derivative_basis_by_axis": {
                "x": "stationary_spectral_band_limited_gradient",
                "y": "stationary_spectral_band_limited_gradient",
                "z": "symmetric_rerendered_z_pair",
            },
            "x_y_derivative_precondition": "stationary_template_identity",
            "derivative_units": [
                _derivative_unit(signal_units, "nm"),
                _derivative_unit(signal_units, "nm"),
                _derivative_unit(signal_units, "nm"),
            ],
            "axial_derivative_mode": "symmetric_rerendered_z_pair",
            "z_step_nm": float(z_step_nm),
            "fisher_axis_units": ["1/nm^2", "1/nm^2", "1/nm^2"],
        }
    )
    return meta

def compute_fisher_information_3d(
    per_particle_contrast_stack: np.ndarray,
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel,
    pixel_size_nm: float,
    z_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
    params: dict[str, Any] | None = None,
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

    Contract
    --------
    This low-level routine is a stationary-template 3D operator: the supplied
    z stack gives the axial derivative, while x/y derivatives are spectral
    band-limited gradients of the central plane. Public array-only callers must
    validate the x/y stationary-template precondition before calling this
    function; modalities with detector/world-fixed lateral structure require an
    explicit x/y rerendered derivative bundle instead.

    Notes
    -----
    The decision to take ``dC/dz`` from a *symmetric* finite difference
    (rather than a one-sided forward difference) is deliberate: it is
    second-order accurate in ``z_step_nm`` and shares the truncation order
    of the in-plane gradients, so the three Fisher components are on equal
    numerical footing. The cost is two additional rendered z-planes per
    particle (``-dz`` and ``+dz``), which the caller controls.
    """
    noise_variance_map, measurement_domain, signal_units, noise_variance_units, model_line_variance, noise_model = (
        _resolve_analysis_noise_input(
            noise_variance_map,
            context="compute_fisher_information_3d",
            measurement_domain=measurement_domain,
            signal_units=signal_units,
            noise_variance_units=noise_variance_units,
        )
    )
    assert_compatible(
        context="compute_fisher_information_3d",
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units or _variance_units(signal_units),
        params=params,
    )
    if noise_model is None or noise_model.covariance_kind == "independent_pixels":
        _reject_correlated_noise_for_diagonal_fisher(
            params,
            context="compute_fisher_information_3d",
            measurement_domain=measurement_domain,
            signal_units=signal_units,
        )

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
    grads = (dI_dx0, dI_dy0, dC_dz)

    if noise_model is None:
        raise TypeError(
            "compute_fisher_information_3d requires a typed Fisher noise likelihood; "
            "raw diagonal variances must be wrapped before this point."
        )

    # The 3D path must use the same typed likelihood-to-precision operator as
    # lateral CRLBs and density maps. A local 1/variance shortcut treats the
    # numerical Fisher variance floor as calibrated physical information and
    # makes z/3D rankings depend on the caller branch rather than the noise
    # model. Keep all covariance, floor-support, and row-correlation policy in
    # fisher.precision.
    return compute_fisher_from_gradients_with_noise(
        grads,
        noise_model,
        context="compute_fisher_information_3d",
    )

def compute_localization_crlb_3d(
    per_particle_contrast_stack: np.ndarray,
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel,
    pixel_size_nm: float,
    z_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
    params: dict[str, Any] | None = None,
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
    original_noise_model = (
        noise_variance_map
        if isinstance(noise_variance_map, (AnalysisNoiseModel, IndependentPixelNoiseModel))
        else None
    )
    if original_noise_model is not None:
        original_noise_model.require_safe_for_fisher(context="compute_localization_crlb_3d")
        measurement_domain = original_noise_model.measurement_domain
        signal_units = original_noise_model.signal_units
        noise_variance_units = original_noise_model.noise_variance_units
    assert_compatible(
        context="compute_localization_crlb_3d",
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units or _variance_units(signal_units),
        params=params,
    )
    from .fusion import _axis_sigmas_from_fisher

    F = compute_fisher_information_3d(
        per_particle_contrast_stack,
        noise_variance_map,
        pixel_size_nm,
        z_step_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
        params=params,
    )
    rank_metadata = _fisher_rank_metadata(F)
    derivative_metadata = _localization_3d_derivative_metadata(
        pixel_size_nm,
        z_step_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    noise_metadata: dict[str, Any] = {}
    if original_noise_model is not None:
        noise_metadata["analysis_noise_covariance_kind"] = original_noise_model.covariance_kind
        noise_metadata["analysis_noise_status_reason"] = original_noise_model.status_reason
        if original_noise_model.covariance_kind == "row_correlated_scan_lines":
            noise_metadata["fisher_noise_covariance_model"] = (
                "row_correlated_scan_line_covariance"
                if float(original_noise_model.row_correlated_variance) > 0.0
                else "transformed_row_correlated_scan_line_covariance"
            )
            noise_metadata["scan_line_noise_variance_counts2"] = float(
                original_noise_model.row_correlated_variance
            )
            noise_metadata["safe_for_covariance_fisher_variance"] = True
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
            "derivative_metadata": derivative_metadata,
            **rank_metadata,
            **noise_metadata,
            **_diagnostic_metadata_aliases(
                derivative_metadata,
                rank_metadata,
                axes_singular=["x", "y", "z"],
                sigma_units=["nm", "nm", "nm"],
            ),
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
            "xy_singular": True,
            "z_singular": True,
            "only_axially_singular": False,
            "derivative_metadata": derivative_metadata,
            **rank_metadata,
            **noise_metadata,
            **_diagnostic_metadata_aliases(
                derivative_metadata,
                rank_metadata,
                axes_singular=["x", "y", "z"],
                sigma_units=["nm", "nm", "nm"],
            ),
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
            "rank": 0,
            "axes_singular": ["x", "y", "z"],
            "derivative_metadata": derivative_metadata,
            **rank_metadata,
            **noise_metadata,
            **_diagnostic_metadata_aliases(
                derivative_metadata,
                rank_metadata,
                axes_singular=["x", "y", "z"],
                sigma_units=["nm", "nm", "nm"],
            ),
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
        "derivative_metadata": derivative_metadata,
        **rank_metadata,
        **noise_metadata,
        **_diagnostic_metadata_aliases(
            derivative_metadata,
            rank_metadata,
            axes_singular=axes_singular,
            sigma_units=["nm", "nm", "nm"],
        ),
    }

def adaptive_axial_crlb_from_stacks(stacks_by_z_step_nm: dict[float, np.ndarray], noise_variance_map: np.ndarray | float, pixel_size_nm: float, *, convergence_tolerance: float = 0.10, min_stable_steps: int = 2, source_contract: str = "Contract-LZ", modality: str = "unknown", signal_units: str = "contrast", measurement_domain: str = "contrast", noise_variance_units: str | None = None) -> dict[str, Any]:
    def _compute(step: float, stack: np.ndarray) -> dict[str, Any]:
        return compute_localization_crlb_3d(
            stack,
            noise_variance_map,
            pixel_size_nm,
            step,
            signal_units=signal_units,
            measurement_domain=measurement_domain,
            noise_variance_units=noise_variance_units,
        )

    return _adaptive_crlb_from_steps(
        stacks_by_z_step_nm,
        _compute,
        metric_key="sigma_xyz_nm",
        convergence_tolerance=convergence_tolerance,
        min_stable_steps=min_stable_steps,
        source_contract=source_contract,
        modality=modality,
    )

__all__ = ['compute_fisher_information_3d', 'compute_localization_crlb_3d', 'adaptive_axial_crlb_from_stacks']
