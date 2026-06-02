"""Cross-modality Fisher fusion and registration diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment_contracts import ValidationStatus, combine_parent_statuses

from ._constants import (
    _FISHER_EIGENVALUE_UNDERFLOW_FLOOR,
    _FISHER_RANK_RELATIVE_TOL,
    _FISHER_RANGE_RESIDUAL_TOL,
    _RELATIVE_DET_SINGULAR_TOL,
)
from ._metadata_helpers import _variance_units

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
    from .axial import compute_fisher_information_3d
    from .lateral import compute_fisher_information

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

def sigma_xy_from_fisher(F: np.ndarray) -> float:
    """Return lateral L2 CRLB sigma from a Fisher matrix.

    This public wrapper is used by notebooks that work directly with
    rerendered finite-difference Fisher matrices rather than image-domain
    gradients. The singular-axis convention matches the internal CRLB helpers:
    if either lateral axis is singular, the returned sigma is infinite.
    """
    sigma_xy, _ = _sigma_xy_from_fisher(F)
    return float(sigma_xy)

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
    rank_tol = max(_FISHER_EIGENVALUE_UNDERFLOW_FLOOR, scale * _FISHER_RANK_RELATIVE_TOL)
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
    *,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the monotone fusion penalty from registration covariance.

    The clean theorem assumes positive-definite Fisher matrices. This diagnostic
    uses the same validation and pseudoinverse convention as fusion, so singular
    inputs remain inspectable but should not be used to claim the theorem.
    """
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

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

    status_metadata = combine_parent_statuses(parent_result_metadata_by_modality)
    parent_convergence_statuses = _parent_convergence_statuses(parent_result_metadata_by_modality)
    parent_validation_statuses = _parent_validation_statuses(parent_result_metadata_by_modality)
    return {
        "registration_covariance_grid": covariance_grid,
        "fusion_sigma_xy_nm_by_registration": sigma_by_registration,
        "fusion_gain_xy_by_registration": gain_by_registration,
        "monotone_degradation_verified": bool(monotone),
        "perfect_registration_fisher": perfect_fisher,
        "registration_degradation_fraction": degradation_fraction,
        "registration_adjusted_per_modality_fisher_by_registration": adjusted_by_registration,
        "parent_status_metadata": status_metadata,
        "parent_convergence_statuses": parent_convergence_statuses,
        "parent_validation_statuses": parent_validation_statuses,
        "validation_status": status_metadata["validation_status"],
        "registration_validation_status": status_metadata["validation_status"],
        "production_grid_diagnostic": status_metadata["production_grid_diagnostic"],
        "safe_for_registration": status_metadata["safe_for_registration"],
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
    modality_profile_cards: dict[str, dict[str, Any]] | None = None,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
    measurement_domain_by_modality: str | dict[str, str] | None = None,
    signal_units_by_modality: str | dict[str, str] | None = None,
    noise_variance_units_by_modality: str | dict[str, str] | None = None,
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
    from .axial import compute_localization_crlb_3d
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses
    from .lateral import _resolve_modality_scalar_map, _resolve_modality_string_map, compute_localization_crlb

    import itertools
    from modality_compatibility import fusion_subset_metadata

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
    measurement_domains = _resolve_modality_string_map(
        measurement_domain_by_modality,
        modalities,
        "contrast",
    )
    signal_units = _resolve_modality_string_map(
        signal_units_by_modality,
        modalities,
        "contrast",
    )
    noise_variance_units = _resolve_modality_string_map(
        noise_variance_units_by_modality,
        modalities,
        "",
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
            per_modality_crlb[m] = compute_localization_crlb(
                c,
                v,
                pixel_sizes[m],
                signal_units=signal_units[m],
                measurement_domain=measurement_domains[m],
                noise_variance_units=noise_variance_units[m] or None,
            )
        else:
            per_modality_crlb[m] = compute_localization_crlb_3d(
                c,
                v,
                pixel_sizes[m],
                z_step_nm=z_step_nm,
                signal_units=signal_units[m],
                measurement_domain=measurement_domains[m],
                noise_variance_units=noise_variance_units[m] or None,
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
        the rest of the fisher package.  ``fusion_sigma_xy_nm`` is the
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
        "registration_covariance_units": "nm^2",
        "pixel_size_nm_by_modality": pixel_sizes,
        "measurement_domain_by_modality": measurement_domains,
        "signal_units_by_modality": signal_units,
        "noise_variance_units_by_modality": {
            modality: (
                noise_variance_units[modality]
                or _variance_units(signal_units[modality])
            )
            for modality in modalities
        },
    }
    result["fusion_physical_metadata"] = fusion_subset_metadata(
        fused["modalities_used"],
        modality_profile_cards,
    )
    selected_parent_metadata = (
        {m: parent_result_metadata_by_modality[m] for m in fused["modalities_used"]}
        if parent_result_metadata_by_modality else None
    )
    result["parent_status_metadata"] = combine_parent_statuses(selected_parent_metadata)
    result["parent_convergence_statuses"] = _parent_convergence_statuses(selected_parent_metadata)
    result["parent_validation_statuses"] = _parent_validation_statuses(selected_parent_metadata)
    result["validation_status"] = result["parent_status_metadata"]["validation_status"]
    result["fusion_validation_status"] = result["parent_status_metadata"]["validation_status"]
    result["production_grid_diagnostic"] = result["parent_status_metadata"]["production_grid_diagnostic"]
    physical_fusion_allowed = bool(
        result["fusion_physical_metadata"].get("physically_feasible_fusion_allowed", False)
    )
    result["safe_for_fusion"] = (
        bool(result["parent_status_metadata"]["safe_for_fusion"])
        and physical_fusion_allowed
    )
    result["fusion_interpretation"] = (
        "physically_feasible_fusion"
        if physical_fusion_allowed
        else "algebraic_diagnostic_only"
    )
    result["physical_compatibility_status"] = result["fusion_physical_metadata"].get(
        "physical_compatibility_status",
        "not_declared",
    )
    if not physical_fusion_allowed:
        result["fusion_validation_status"] = ValidationStatus.DIAGNOSTIC_ONLY.value
        result["production_grid_diagnostic"] = True
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

def compute_modality_fusion_crlb_from_fisher_matrices(
    per_modality_fisher: dict[str, np.ndarray],
    *,
    subset_size: int | None = None,
    registration_covariance: np.ndarray | None = None,
    modality_profile_cards: dict[str, dict[str, Any]] | None = None,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fusion CRLB directly from precomputed Fisher matrices.

    This is the core-owned route for workflows that already selected
    rerendered finite-difference Fisher matrices. It uses the same lateral L2
    sigma convention, registration-covariance model, physical-compatibility
    metadata, and parent-status gating as image-domain fusion.
    """
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

    import itertools
    from modality_compatibility import fusion_subset_metadata

    if not isinstance(per_modality_fisher, dict) or not per_modality_fisher:
        raise ValueError("per_modality_fisher must be a non-empty dict.")
    modalities = list(per_modality_fisher)
    matrices = {m: np.asarray(F, dtype=float) for m, F in per_modality_fisher.items()}
    shape = matrices[modalities[0]].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        raise ValueError(f"Fisher matrices must be square with at least two axes; got {shape}.")
    for modality, F in matrices.items():
        if F.shape != shape:
            raise ValueError(
                f"All Fisher matrices must have shape {shape}; {modality!r} has {F.shape}."
            )
        if not np.all(np.isfinite(F)):
            raise ValueError(f"Fisher matrix for {modality!r} contains non-finite values.")
    if subset_size is not None and (
        not isinstance(subset_size, int) or subset_size < 1 or subset_size > len(modalities)
    ):
        raise ValueError(
            f"subset_size must be an integer in [1, {len(modalities)}]; got {subset_size!r}."
        )

    dim = int(shape[0])
    adjusted = {
        modality: _registration_adjusted_fisher(F, registration_covariance)
        for modality, F in matrices.items()
    }
    adjusted_crlb: dict[str, dict[str, Any]] = {}
    best_single_modality: str | None = None
    best_single_sigma_xy: float | None = None
    for modality, F in adjusted.items():
        sigma_xy, xy_singular = _sigma_xy_from_fisher(F)
        axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F)
        adjusted_crlb[modality] = {
            "sigma_xy_nm": sigma_xy,
            "xy_singular": xy_singular,
            "singular": bool(any(axis_singular)),
            "fisher_matrix": F,
            "axis_sigmas_nm": axis_sigmas,
            "axes_singular": {
                axis: bool(flag)
                for axis, flag in zip(_axis_names_for_dim(dim), axis_singular)
            },
        }
        if (
            not xy_singular
            and np.isfinite(sigma_xy)
            and (best_single_sigma_xy is None or sigma_xy < best_single_sigma_xy)
        ):
            best_single_sigma_xy = float(sigma_xy)
            best_single_modality = modality

    def _fuse_subset(subset: tuple[str, ...]) -> dict[str, Any]:
        F_sum = np.zeros(shape, dtype=float)
        for modality in subset:
            F_sum = F_sum + adjusted[modality]
        F_sum = 0.5 * (F_sum + F_sum.T)
        sigma_xy, xy_singular = _sigma_xy_from_fisher(F_sum)
        axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F_sum)
        axes_singular = {
            axis: bool(flag)
            for axis, flag in zip(_axis_names_for_dim(dim), axis_singular)
        }
        return {
            "fusion_fisher": F_sum,
            "fusion_sigma_x_nm": axis_sigmas[0],
            "fusion_sigma_y_nm": axis_sigmas[1],
            "fusion_sigma_xy_nm": sigma_xy,
            "fusion_singular": bool(any(axis_singular)),
            "fusion_xy_singular": bool(xy_singular),
            "fusion_rank": int(dim - sum(bool(flag) for flag in axis_singular[:dim])),
            "fusion_axes_singular": axes_singular,
            "modalities_used": list(subset),
        }

    if subset_size is None:
        fused = _fuse_subset(tuple(modalities))
        subset_search_count = 1
    else:
        fused = None
        subset_search_count = 0
        for subset in itertools.combinations(modalities, subset_size):
            subset_search_count += 1
            candidate = _fuse_subset(subset)
            if candidate["fusion_xy_singular"] or not np.isfinite(candidate["fusion_sigma_xy_nm"]):
                if fused is None:
                    fused = candidate
                continue
            if (
                fused is None
                or fused["fusion_xy_singular"]
                or candidate["fusion_sigma_xy_nm"] < fused["fusion_sigma_xy_nm"]
            ):
                fused = candidate
        if fused is None:
            fused = _fuse_subset(next(iter(itertools.combinations(modalities, subset_size))))

    selected_parent_metadata = (
        {m: parent_result_metadata_by_modality[m] for m in fused["modalities_used"]}
        if parent_result_metadata_by_modality else None
    )
    parent_status_metadata = combine_parent_statuses(selected_parent_metadata)
    physical_metadata = fusion_subset_metadata(fused["modalities_used"], modality_profile_cards)
    physical_fusion_allowed = bool(
        physical_metadata.get("physically_feasible_fusion_allowed", False)
    )
    fusion_validation_status = parent_status_metadata["validation_status"]
    production_grid_diagnostic = parent_status_metadata["production_grid_diagnostic"]
    if not physical_fusion_allowed:
        fusion_validation_status = ValidationStatus.DIAGNOSTIC_ONLY.value
        production_grid_diagnostic = True

    result: dict[str, Any] = {
        "per_modality_fisher": matrices,
        "registration_adjusted_per_modality_fisher": adjusted,
        "registration_adjusted_per_modality_crlb": adjusted_crlb,
        "fusion_fisher": fused["fusion_fisher"],
        "fusion_sigma_x_nm": fused["fusion_sigma_x_nm"],
        "fusion_sigma_y_nm": fused["fusion_sigma_y_nm"],
        "fusion_sigma_xy_nm": fused["fusion_sigma_xy_nm"],
        "fusion_singular": fused["fusion_singular"],
        "fusion_xy_singular": fused["fusion_xy_singular"],
        "fusion_rank": fused["fusion_rank"],
        "fusion_axes_singular": fused["fusion_axes_singular"],
        "best_single_modality": best_single_modality,
        "best_single_xy_modality": best_single_modality,
        "best_single_sigma_xy_nm": best_single_sigma_xy,
        "modalities_used": fused["modalities_used"],
        "subset_size": subset_size,
        "subset_search_count": subset_search_count,
        "registration_covariance": (
            None
            if registration_covariance is None
            else np.asarray(registration_covariance, dtype=float)
        ),
        "registration_covariance_units": "nm^2",
        "fusion_complementarity": _fusion_complementarity_metrics(
            adjusted,
            fused["modalities_used"],
        ),
        "fusion_physical_metadata": physical_metadata,
        "parent_status_metadata": parent_status_metadata,
        "parent_convergence_statuses": _parent_convergence_statuses(selected_parent_metadata),
        "parent_validation_statuses": _parent_validation_statuses(selected_parent_metadata),
        "validation_status": parent_status_metadata["validation_status"],
        "fusion_validation_status": fusion_validation_status,
        "production_grid_diagnostic": production_grid_diagnostic,
        "safe_for_fusion": bool(parent_status_metadata["safe_for_fusion"]) and physical_fusion_allowed,
        "fusion_interpretation": (
            "physically_feasible_fusion"
            if physical_fusion_allowed
            else "algebraic_diagnostic_only"
        ),
        "physical_compatibility_status": physical_metadata.get(
            "physical_compatibility_status",
            "not_declared",
        ),
        "fusion_metric_convention": "L2_sigma_xy_not_rms",
    }
    if (
        best_single_sigma_xy is not None
        and not fused["fusion_xy_singular"]
        and np.isfinite(fused["fusion_sigma_xy_nm"])
        and fused["fusion_sigma_xy_nm"] > 0.0
    ):
        result["fusion_gain_xy"] = float(best_single_sigma_xy / fused["fusion_sigma_xy_nm"])
    else:
        result["fusion_gain_xy"] = None
    result["fusion_gain_xy_semantics"] = "selected_subset_lateral_ratio"
    return result

__all__ = ['sigma_xy_from_fisher', 'compute_registration_degradation_curve', 'compute_modality_fusion_crlb', 'compute_modality_fusion_crlb_from_fisher_matrices']
