"""Axial Fisher-information and z-localization CRLB implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from common_utils import init_infinite_dict
from experiment_contracts import combine_parent_statuses, normalize_convergence_status

from ._constants import _FISHER_DET_EPS, _RELATIVE_DET_SINGULAR_TOL
from ._metadata_helpers import (
    _derivative_unit,
    _diagnostic_metadata_aliases,
    _fisher_rank_metadata,
    _localization_derivative_metadata,
    _variance_units,
)
from .lateral import _adaptive_crlb_from_steps

def _localization_3d_derivative_metadata(
    pixel_size_nm: float,
    z_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
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
    from .lateral import _build_symmetric_fisher_from_gradients, _lateral_coordinate_derivatives

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
    return _build_symmetric_fisher_from_gradients(
        (dI_dx0, dI_dy0, dC_dz),
        inv_var,
    )

def compute_localization_crlb_3d(
    per_particle_contrast_stack: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    z_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
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
    from .fusion import _axis_sigmas_from_fisher

    F = compute_fisher_information_3d(
        per_particle_contrast_stack, noise_variance_map, pixel_size_nm, z_step_nm,
    )
    rank_metadata = _fisher_rank_metadata(F)
    derivative_metadata = _localization_3d_derivative_metadata(
        pixel_size_nm,
        z_step_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
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
            "derivative_metadata": derivative_metadata,
            **rank_metadata,
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
        **_diagnostic_metadata_aliases(
            derivative_metadata,
            rank_metadata,
            axes_singular=axes_singular,
            sigma_units=["nm", "nm", "nm"],
        ),
    }

def compare_modality_axial_information_content(
    contrast_stack_by_modality: dict[str, np.ndarray],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
    measurement_domain_by_modality: str | dict[str, str] | None = None,
    signal_units_by_modality: str | dict[str, str] | None = None,
    noise_variance_units_by_modality: str | dict[str, str] | None = None,
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
    from .lateral import _resolve_modality_scalar_map, _resolve_modality_string_map, _sort_key_finite_then_value

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
    measurement_domains = _resolve_modality_string_map(
        measurement_domain_by_modality,
        contrast_stack_by_modality.keys(),
        "contrast",
    )
    signal_units = _resolve_modality_string_map(
        signal_units_by_modality,
        contrast_stack_by_modality.keys(),
        "contrast",
    )
    noise_variance_units = _resolve_modality_string_map(
        noise_variance_units_by_modality,
        contrast_stack_by_modality.keys(),
        "",
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, stack in contrast_stack_by_modality.items():
        try:
            per_modality[modality] = compute_localization_crlb_3d(
                stack,
                noise_variance_by_modality[modality],
                pixel_sizes[modality],
                z_step_nm,
                signal_units=signal_units[modality],
                measurement_domain=measurement_domains[modality],
                noise_variance_units=(
                    noise_variance_units[modality] or None
                ),
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
        if parent_result_metadata_by_modality is not None:
            per_modality[modality]["parent_convergence_status"] = normalize_convergence_status(
                dict(parent_result_metadata_by_modality).get(modality, {}).get(
                    "convergence_status",
                    "unchecked",
                )
            )

    items = [
        (m, float(r.get("sigma_z_nm", float("inf"))))
        for m, r in per_modality.items()
    ]

    ranking_z = sorted(items, key=_sort_key_finite_then_value)

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
        relative_sigma_z = init_infinite_dict(per_modality)
        frames_to_match_best_z = init_infinite_dict(per_modality)
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

    out: dict[str, Any] = {
        "per_modality": per_modality,
        "ordering_z": ranking_z,
        "ranking_z": ranking_z,
        "best_modality_z": best_modality_z,
        "relative_sigma_z": relative_sigma_z,
        "frames_to_match_best_z": frames_to_match_best_z,
        "axially_singular_per_modality": axially_singular_per_modality,
        "pixel_size_nm_by_modality": pixel_sizes,
        "measurement_domain_by_modality": measurement_domains,
        "signal_units_by_modality": signal_units,
        "noise_variance_units_by_modality": {
            modality: (
                noise_variance_units[modality]
                or _variance_units(signal_units[modality])
            )
            for modality in contrast_stack_by_modality
        },
    }
    parent_metadata = combine_parent_statuses(parent_result_metadata_by_modality)
    out["parent_status_metadata"] = parent_metadata
    out["parent_convergence_status_by_modality"] = parent_metadata[
        "parent_convergence_statuses"
    ]
    out["validation_status"] = parent_metadata["validation_status"]
    out["production_grid_diagnostic"] = parent_metadata["production_grid_diagnostic"]
    out["safe_for_ordering"] = parent_metadata["safe_for_ordering"]
    out["safe_for_fusion"] = parent_metadata["safe_for_fusion"]
    out["safe_for_time_allocation"] = parent_metadata["safe_for_time_allocation"]
    out["safe_for_registration"] = parent_metadata["safe_for_registration"]
    out["safe_for_detected_quanta_ranking"] = parent_metadata[
        "safe_for_detected_quanta_ranking"
    ]
    out["status_reason"] = parent_metadata["status_reason"]

    return out

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

__all__ = ['compute_fisher_information_3d', 'compute_localization_crlb_3d', 'compare_modality_axial_information_content', 'adaptive_axial_crlb_from_stacks']
