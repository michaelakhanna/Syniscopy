"""Candidate Fisher fusion and registration diagnostics.

The algebra in this module is keyed by comparison candidate. A candidate may be
a microscope, acquisition profile, or virtual information object. Physical
modality identity is metadata consumed only by compatibility checks.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiment_contracts import ValidationStatus, combine_parent_statuses

from ._constants import (
    _FISHER_EIGENVALUE_UNDERFLOW_FLOOR,
    _FISHER_RANK_RELATIVE_TOL,
    _FISHER_RANGE_RESIDUAL_TOL,
    _RELATIVE_DET_SINGULAR_TOL,
)
from .candidates import (
    FisherCandidate,
    FisherMatrixCandidate,
    candidate_metadata_records,
    matrix_candidate_metadata_records,
)
from .comparison import (
    COMPARISON_TARGET_LATERAL_XY,
    COMPARISON_TARGET_LOCALIZATION_XYZ,
    fisher_derivative_basis_for_candidate,
    resolve_fisher_candidate_noise_inputs,
)


def _matrix_candidate_list(
    candidates: Sequence[FisherMatrixCandidate],
    *,
    context: str,
) -> list[FisherMatrixCandidate]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError(f"{context} requires at least one FisherMatrixCandidate.")
    keys = [candidate.key for candidate in candidate_list]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"{context} candidate keys must be unique; duplicates: {duplicates!r}.")
    return candidate_list


def _parent_metadata_from_matrix_candidates(
    candidates: Sequence[FisherMatrixCandidate],
) -> dict[str, dict[str, Any]] | None:
    metadata = {
        candidate.key: dict(candidate.parent_result_metadata)
        for candidate in candidates
        if candidate.parent_result_metadata
    }
    return metadata or None

def _fisher_for_candidate(
    contrast: np.ndarray,
    noise_variance: Any,
    pixel_size_nm: float,
    z_step_nm: float | None,
) -> np.ndarray:
    """Compute one candidate's Fisher matrix in 2D or 3D mode.

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

def compute_candidate_registration_degradation_curve(
    candidates: Sequence[FisherMatrixCandidate],
    registration_covariances: list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """
    Evaluate the monotone fusion penalty from registration covariance.

    ``candidates`` carries Fisher matrices and parent metadata as
    ``FisherMatrixCandidate`` objects; this API does not accept parallel
    candidate-keyed matrix/metadata columns.

    The clean theorem assumes positive-definite Fisher matrices. This diagnostic
    uses the same validation and pseudoinverse convention as fusion, so singular
    inputs remain inspectable but should not be used to claim the theorem.
    """
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

    candidate_list = _matrix_candidate_list(
        candidates,
        context="compute_candidate_registration_degradation_curve",
    )
    if not registration_covariances:
        raise ValueError("registration_covariances must contain at least one covariance.")

    candidates = [candidate.key for candidate in candidate_list]
    raw = {candidate.key: np.asarray(candidate.fisher_matrix, dtype=float) for candidate in candidate_list}
    candidate_parent_metadata = _parent_metadata_from_matrix_candidates(candidate_list)
    ref_shape = raw[candidates[0]].shape
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

    status_metadata = combine_parent_statuses(candidate_parent_metadata)
    parent_convergence_statuses = _parent_convergence_statuses(candidate_parent_metadata)
    parent_validation_statuses = _parent_validation_statuses(candidate_parent_metadata)
    return {
        "registration_covariance_grid": covariance_grid,
        "fusion_sigma_xy_nm_by_registration": sigma_by_registration,
        "fusion_gain_xy_by_registration": gain_by_registration,
        "monotone_degradation_verified": bool(monotone),
        "perfect_registration_fisher": perfect_fisher,
        "registration_degradation_fraction": degradation_fraction,
        "registration_adjusted_per_candidate_fisher_by_registration": adjusted_by_registration,
        "parent_status_metadata": status_metadata,
        "parent_convergence_statuses": parent_convergence_statuses,
        "parent_validation_statuses": parent_validation_statuses,
        "validation_status": status_metadata["validation_status"],
        "registration_validation_status": status_metadata["validation_status"],
        "production_grid_diagnostic": status_metadata["production_grid_diagnostic"],
        "safe_for_registration": status_metadata["safe_for_registration"],
    }

def _fusion_complementarity_metrics(
    per_candidate_fisher: dict[str, np.ndarray],
    subset: list[str],
) -> dict[str, Any]:
    """
    Diagnostic metrics for whether fusion adds complementary directions.

    These metrics are descriptive, not part of the CRLB itself. They help
    distinguish "best pair is just the two strongest scalar contributors" from
    "candidates inform different parameter directions."
    """
    if len(subset) < 2:
        return {
            "mean_principal_angle_deg": 0.0,
            "max_principal_angle_deg": 0.0,
            "determinant_gain_vs_best_single": 1.0,
            "fused_condition_number": None,
        }

    matrices = [np.asarray(per_candidate_fisher[m], dtype=float) for m in subset]
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

def compute_fisher_candidate_fusion_crlb(
    candidates: Sequence[FisherCandidate],
    *,
    z_step_nm: float | None = None,
    subset_size: int | None = None,
    registration_covariance: np.ndarray | None = None,
    candidate_profile_cards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    r"""
    Candidate fusion Cramér-Rao lower bound.

    Given complete ``FisherCandidate`` objects for the same particle coordinate
    frame, return:
      - the joint Fisher information matrix obtained by summing per-candidate
        Fisher matrices under the independent-noise assumption,
      - the fusion CRLB on (x, y) (and optionally z),
      - the fusion gain against the best single candidate.

    Parameters
    ----------
    candidates : sequence[FisherCandidate]
        Complete rendered analysis candidates. 2D mode expects each candidate
        signal to be an ``(H, W)`` array; 3D mode (``z_step_nm`` supplied)
        expects each signal to be a ``(3, H, W)`` three-plane stack.
    z_step_nm : float or None, default None
        If None: 2D fusion (2x2 Fisher per candidate, 2x2 fused). If a
        positive float: 3D fusion (3x3 Fisher per candidate, 3x3 fused).
    subset_size : int or None, default None
        If None: report the full N-candidate fusion bound. If a positive
        integer k with 1 <= k <= N: enumerate all C(N, k) subsets of
        candidates of size k, compute each subset's fusion CRLB, and return
        the BEST subset (smallest fusion_sigma_xy_nm in 2D mode, smallest
        fusion_sigma_xyz_nm in 3D mode). The best-subset enumeration is
        intended for small candidate sets.
    registration_covariance : ndarray or None, default None
        Optional 2x2 lateral registration-error covariance in nm^2. When
        supplied, each candidate's contribution is corrected as
        ``F' = inv(inv(F) + Sigma_reg)`` before fusion.

    Returns
    -------
    result : dict
        A dictionary with the following keys:

        - ``per_candidate_fisher`` : dict[str, ndarray]
            Per-candidate Fisher matrix.
        - ``registration_adjusted_per_candidate_fisher`` : dict[str, ndarray]
            Per-candidate Fisher matrices after registration covariance.
        - ``per_candidate_crlb`` : dict[str, dict]
            Per-candidate CRLB result (output of compute_localization_crlb
            or compute_localization_crlb_3d). Candidates whose per-candidate
            Fisher is singular have ``singular = True``.
        - ``registration_adjusted_per_candidate_crlb`` : dict[str, dict]
            Single-candidate baseline CRLBs after applying registration covariance;
            fusion gains are computed against these adjusted baselines.
        - ``fusion_fisher`` : ndarray
            The joint Fisher matrix (sum of per-candidate F_i).
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
        - ``best_single_candidate`` : str | None — name of the registration-adjusted
          per-candidate CRLB minimizer on sigma_xy_nm (or sigma_xyz_nm in 3D).
          None if no single candidate is non-singular.
        - ``best_single_xy_candidate`` : str — lateral-only lowest-bound candidate used
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
        - ``candidates_used`` : list[str] — candidates included in the
          fusion. Equals the full key set of the candidates if
          ``subset_size`` is None; the chosen best subset otherwise.
        - ``subset_size`` : int or None — echo of the input subset_size.
        - ``subset_search_count`` : int — number of subsets evaluated
          (1 if subset_size is None, C(N, subset_size) otherwise).
        - ``registration_covariance`` : ndarray or None
            Echo of the registration covariance used for adjusted baselines.

    Raises
    ------
    ValueError
        If the candidate list is empty, has duplicate keys, contains malformed
        signals/noise, or if subset_size is out of range.

    Notes
    -----
    The fusion bound assumes statistically independent measurements across
    candidates. Physical modality compatibility is metadata-gated separately,
    because candidate identity and backend modality identity are distinct.
    """
    from .axial import compute_localization_crlb_3d
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses
    from .lateral import compute_localization_crlb

    import itertools

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("compute_fisher_candidate_fusion_crlb requires at least one candidate.")
    candidate_keys = [candidate.key for candidate in candidate_list]
    duplicates = sorted({key for key in candidate_keys if candidate_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Fisher candidate keys must be unique; duplicates: {duplicates!r}.")
    n = len(candidate_list)

    target = (
        COMPARISON_TARGET_LATERAL_XY
        if z_step_nm is None
        else COMPARISON_TARGET_LOCALIZATION_XYZ
    )
    noise_inputs = resolve_fisher_candidate_noise_inputs(
        candidate_list,
        context="compute_fisher_candidate_fusion_crlb",
    )
    candidate_parent_metadata = {
        candidate.key: dict(candidate.parent_result_metadata)
        for candidate in candidate_list
        if candidate.parent_result_metadata
    } or None

    if subset_size is not None:
        if not isinstance(subset_size, int) or subset_size < 1 or subset_size > n:
            raise ValueError(
                f"subset_size must be an integer in [1, {n}]; got {subset_size!r}."
            )

    dim = 2 if z_step_nm is None else 3

    # Step 1: per-candidate Fisher matrices.
    per_candidate_fisher: dict[str, np.ndarray] = {}
    registration_adjusted_per_candidate_fisher: dict[str, np.ndarray] = {}
    per_candidate_crlb: dict[str, dict[str, Any]] = {}
    registration_adjusted_per_candidate_crlb: dict[str, dict[str, Any]] = {}
    derivative_basis: dict[str, dict[str, Any]] = {}
    for candidate in candidate_list:
        m = candidate.key
        c = np.asarray(candidate.signal)
        v = noise_inputs[m]
        derivative_basis[m] = fisher_derivative_basis_for_candidate(
            candidate,
            target=target,
            z_step_nm=z_step_nm,
        )
        from .dhm_demodulated import (
            compute_off_axis_demodulated_fisher_information,
            compute_off_axis_demodulated_localization_crlb_from_field,
            is_off_axis_demodulated_fisher_payload,
            is_off_axis_holography_modality,
        )

        off_axis_demodulated_2d = (
            z_step_nm is None
            and is_off_axis_holography_modality(candidate.modality)
            and is_off_axis_demodulated_fisher_payload(c, v)
        )
        if off_axis_demodulated_2d:
            F_m, _precision_metadata = compute_off_axis_demodulated_fisher_information(
                c,
                v,
                candidate.pixel_size_nm,
                context=f"fusion candidate {m!r} off-axis demodulated Fisher",
            )
        else:
            F_m = _fisher_for_candidate(np.asarray(c, dtype=float), v, candidate.pixel_size_nm, z_step_nm)
        per_candidate_fisher[m] = F_m
        F_adjusted = _registration_adjusted_fisher(
            F_m,
            registration_covariance,
        )
        registration_adjusted_per_candidate_fisher[m] = F_adjusted
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
        registration_adjusted_per_candidate_crlb[m] = adjusted_summary
        if z_step_nm is None:
            if off_axis_demodulated_2d:
                per_candidate_crlb[m] = compute_off_axis_demodulated_localization_crlb_from_field(
                    c,
                    v,
                    candidate.pixel_size_nm,
                )
            else:
                per_candidate_crlb[m] = compute_localization_crlb(
                    np.asarray(c, dtype=float),
                    v,
                    candidate.pixel_size_nm,
                    signal_units=candidate.signal_units,
                    measurement_domain=candidate.measurement_domain,
                    noise_variance_units=candidate.noise_variance_units,
                )
        else:
            per_candidate_crlb[m] = compute_localization_crlb_3d(
                np.asarray(c, dtype=float),
                v,
                candidate.pixel_size_nm,
                z_step_nm=z_step_nm,
                signal_units=candidate.signal_units,
                measurement_domain=candidate.measurement_domain,
                noise_variance_units=candidate.noise_variance_units,
            )

    # Step 2: identify best single-candidate CRLB (skipping singular ones).
    # NOTE: per-candidate CRLB result dicts use the module-wide `_nm` suffix
    # convention (see compute_localization_crlb / compute_localization_crlb_3d).
    if z_step_nm is None:
        sigma_key_single = "sigma_xy_nm"
    else:
        sigma_key_single = "sigma_xyz_nm"
    best_single_candidate: str | None = None
    best_single_sigma: float | None = None
    best_single_xy_candidate: str | None = None
    best_single_sigma_xy: float | None = None
    for m, crlb in registration_adjusted_per_candidate_crlb.items():
        if crlb.get("singular", False):
            s = None
        else:
            s = crlb.get(sigma_key_single)
        if s is None or not np.isfinite(s):
            pass
        elif best_single_sigma is None or s < best_single_sigma:
            best_single_sigma = float(s)
            best_single_candidate = m

        s_xy = crlb.get("sigma_xy_nm")
        if s_xy is not None and np.isfinite(s_xy) and (
            best_single_sigma_xy is None or float(s_xy) < best_single_sigma_xy
        ):
            best_single_sigma_xy = float(s_xy)
            best_single_xy_candidate = m

    # Step 3: choose the candidate subset to fuse.
    def _fuse_subset(subset: tuple[str, ...]) -> dict[str, Any]:
        """Inner helper: sum per-candidate Fishers for a given subset and
        invert. Returns a dict with the fusion-side keys (without the gain
        comparison; the gain is computed once at the outer level).

        Convention: fusion-sigma keys carry the ``_nm`` suffix in line with
        the rest of the fisher package.  ``fusion_sigma_xy_nm`` is the
        L2 sum sqrt(sigma_x^2 + sigma_y^2) — the *total* 2-D bound, the same
        definition used in compute_localization_crlb above (NOT the rms).
        ``fusion_sigma_xyz_nm`` is the analogous L2 sum in 3-D."""
        F_sum = np.zeros((dim, dim), dtype=float)
        for s in subset:
            F_sum = F_sum + registration_adjusted_per_candidate_fisher[s]
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
            "candidates_used": list(subset),
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
        chosen_subset = tuple(candidate_keys)
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
        for subset in itertools.combinations(candidate_keys, subset_size):
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
            chosen_subset = next(iter(itertools.combinations(candidate_keys, subset_size)))
            fused = _fuse_subset(chosen_subset)
        else:
            fused = best
        xy_optimized_fused = best_xy

    # Step 4: assemble result + fusion-gain comparisons.
    candidate_records = candidate_metadata_records(candidate_list)
    record_by_key = {record["candidate_key"]: record for record in candidate_records}
    for candidate in candidate_list:
        record = record_by_key[candidate.key]
        record["fisher_derivative_basis"] = derivative_basis[candidate.key]
        record["noise_variance_units"] = (
            per_candidate_crlb[candidate.key].get("noise_variance_units")
            or candidate.noise_variance_units
        )

    result: dict[str, Any] = {
        "per_candidate_fisher": per_candidate_fisher,
        "registration_adjusted_per_candidate_fisher": (
            registration_adjusted_per_candidate_fisher
        ),
        "per_candidate_crlb": per_candidate_crlb,
        "registration_adjusted_per_candidate_crlb": (
            registration_adjusted_per_candidate_crlb
        ),
        "fusion_fisher": fused["fusion_fisher"],
        "fusion_sigma_x_nm": fused["fusion_sigma_x_nm"],
        "fusion_sigma_y_nm": fused["fusion_sigma_y_nm"],
        "fusion_sigma_xy_nm": fused["fusion_sigma_xy_nm"],
        "fusion_singular": fused["fusion_singular"],
        "fusion_rank": fused["fusion_rank"],
        "fusion_axes_singular": fused["fusion_axes_singular"],
        "fusion_xy_singular": fused["fusion_xy_singular"],
        "best_single_candidate": best_single_candidate,
        "best_single_xy_candidate": best_single_xy_candidate,
        "best_single_sigma_xy_nm": best_single_sigma_xy,
        "candidates_used": fused["candidates_used"],
        "xy_optimized_candidates_used": (
            None if xy_optimized_fused is None else xy_optimized_fused["candidates_used"]
        ),
        "fusion_complementarity": _fusion_complementarity_metrics(
            registration_adjusted_per_candidate_fisher,
            fused["candidates_used"],
        ),
        "subset_size": subset_size,
        "subset_search_count": subset_search_count,
        "registration_covariance": (
            None
            if registration_covariance is None
            else np.asarray(registration_covariance, dtype=float)
        ),
        "registration_covariance_units": "nm^2",
        "candidate_records": candidate_records,
        "candidate_keys": candidate_keys,
    }
    result["fusion_physical_metadata"] = _fusion_subset_metadata_for_precomputed_matrices(
        fused["candidates_used"],
        candidate_profile_cards,
        candidate_parent_metadata,
    )
    selected_parent_metadata = (
        {
            m: candidate_parent_metadata[m]
            for m in fused["candidates_used"]
            if m in candidate_parent_metadata
        }
        if candidate_parent_metadata else None
    ) or None
    result["parent_status_metadata"] = combine_parent_statuses(selected_parent_metadata)
    result["parent_convergence_statuses"] = _parent_convergence_statuses(selected_parent_metadata)
    result["parent_validation_statuses"] = _parent_validation_statuses(selected_parent_metadata)
    result["validation_status"] = result["parent_status_metadata"]["validation_status"]
    result["fusion_validation_status"] = result["parent_status_metadata"]["validation_status"]
    result["production_grid_diagnostic"] = result["parent_status_metadata"]["production_grid_diagnostic"]
    physical_fusion_allowed = bool(
        result["fusion_physical_metadata"]["physically_feasible_fusion_allowed"]
    )
    result["safe_for_fusion"] = (
        bool(result["parent_status_metadata"]["safe_for_fusion"])
        and physical_fusion_allowed
    )
    result["fusion_interpretation"] = result["fusion_physical_metadata"]["fusion_mode"]
    result["physical_compatibility_status"] = result["fusion_physical_metadata"][
        "physical_compatibility_status"
    ]
    if not physical_fusion_allowed:
        result["fusion_validation_status"] = ValidationStatus.DIAGNOSTIC_ONLY.value
        result["production_grid_diagnostic"] = True
    if dim == 3:
        result["fusion_sigma_z_nm"] = fused["fusion_sigma_z_nm"]
        result["fusion_sigma_xyz_nm"] = fused["fusion_sigma_xyz_nm"]
        result["best_single_sigma_xyz_nm"] = (
            best_single_sigma if best_single_candidate is not None else None
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
            best_single_candidate is not None
            and not fused["fusion_singular"]
            and fused["fusion_sigma_xyz_nm"] > 0.0
        ):
            result["fusion_gain_xyz"] = float(
                best_single_sigma / fused["fusion_sigma_xyz_nm"]
            )
        else:
            result["fusion_gain_xyz"] = None

    return result


def _algebraic_only_fusion_metadata(
    candidates: list[str],
    *,
    reason: str,
) -> dict[str, Any]:
    labels = [str(candidate) for candidate in candidates]
    return {
        "candidate_keys": labels,
        "modalities": [],
        "modality_physical_profiles": {},
        "pairwise_compatibility": [],
        "compatible": False,
        "reason": reason,
        "required_review": True,
        "incompatible_hard_stop": False,
        "compatible_only_as_sequential": False,
        "compatible_only_as_algebraic_diagnostic": True,
        "physical_feasibility_status": "unregistered_candidate_labels",
        "physical_compatibility_status": "unregistered_candidate_labels",
        "fusion_mode": "algebraic_diagnostic_only",
        "algebraic_fusion_allowed": True,
        "physically_feasible_fusion_allowed": False,
        "independent_noise_assumption": True,
        "same_sample_state_required": True,
        "double_count_risk": False,
        "same_quanta_reconstruction_risk": False,
        "destructive_measurement_conflict": False,
        "live_sample_conflict": False,
        "preparation_conflict": False,
        "fusion_interpretation": "algebraic_diagnostic_only",
        "acquisition_cost_models": {},
        "requires_physical_design_review": True,
    }


def _fusion_subset_metadata_for_precomputed_matrices(
    candidates: list[str],
    candidate_profile_cards: dict[str, dict[str, Any]] | None,
    candidate_parent_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from modality_compatibility import fusion_subset_metadata

    cards = {} if candidate_profile_cards is None else dict(candidate_profile_cards)
    parent_metadata = (
        {}
        if candidate_parent_metadata is None
        else dict(candidate_parent_metadata)
    )

    def _physical_modality_for_candidate(candidate: str) -> str:
        """Return backend modality metadata for a fusion candidate key."""

        metadata = parent_metadata.get(candidate, {})
        if isinstance(metadata, dict):
            for key in ("modality", "canonical_modality_name", "imaging_model"):
                value = metadata.get(key)
                if value:
                    return str(value)
        card = cards.get(candidate, {})
        if isinstance(card, dict):
            for key in ("canonical_modality_name", "modality", "imaging_model"):
                value = card.get(key)
                if value:
                    return str(value)
        return str(candidate)

    compatibility_modalities = [
        _physical_modality_for_candidate(candidate) for candidate in candidates
    ]
    compatibility_cards: dict[str, dict[str, Any]] = {}
    for candidate, physical_modality in zip(candidates, compatibility_modalities):
        card = cards.get(candidate, {})
        # Fusion algebra is keyed by microscope/candidate identity, but physical
        # compatibility belongs to the backend modality installed for that
        # microscope. Map candidate-keyed cards onto modality labels here so an
        # explicit microscope name such as ``widefield_high_na`` is not treated
        # as an unregistered physical modality.
        if isinstance(card, dict) and physical_modality not in compatibility_cards:
            compatibility_cards[physical_modality] = dict(card)

    try:
        metadata = fusion_subset_metadata(compatibility_modalities, compatibility_cards)
        metadata["candidate_keys"] = list(candidates)
        return metadata
    except ValueError as exc:
        return _algebraic_only_fusion_metadata(
            candidates,
            reason=(
                "Precomputed Fisher matrices were fused algebraically, but one or "
                f"more candidate labels do not declare registered physical modalities: {exc}"
            ),
        )


def compute_candidate_fusion_crlb_from_fisher_matrices(
    candidates: Sequence[FisherMatrixCandidate],
    *,
    subset_size: int | None = None,
    registration_covariance: np.ndarray | None = None,
    candidate_profile_cards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fusion CRLB directly from precomputed Fisher matrices.

    This is the core-owned route for workflows that already selected
    rerendered finite-difference Fisher matrices and now carry them as
    ``FisherMatrixCandidate`` objects. It uses the same lateral L2 sigma
    convention, registration-covariance model, physical-compatibility metadata,
    and parent-status gating as image-domain fusion.
    """
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

    import itertools

    candidate_list = _matrix_candidate_list(
        candidates,
        context="compute_candidate_fusion_crlb_from_fisher_matrices",
    )
    candidates = [candidate.key for candidate in candidate_list]
    matrices = {candidate.key: np.asarray(candidate.fisher_matrix, dtype=float) for candidate in candidate_list}
    candidate_parent_metadata = _parent_metadata_from_matrix_candidates(candidate_list)
    shape = matrices[candidates[0]].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        raise ValueError(f"Fisher matrices must be square with at least two axes; got {shape}.")
    for candidate_key, F in matrices.items():
        if F.shape != shape:
            raise ValueError(
                f"All Fisher matrices must have shape {shape}; {candidate_key!r} has {F.shape}."
            )
        if not np.all(np.isfinite(F)):
            raise ValueError(f"Fisher matrix for {candidate_key!r} contains non-finite values.")
    if subset_size is not None and (
        not isinstance(subset_size, int) or subset_size < 1 or subset_size > len(candidates)
    ):
        raise ValueError(
            f"subset_size must be an integer in [1, {len(candidates)}]; got {subset_size!r}."
        )

    dim = int(shape[0])
    adjusted = {
        candidate_key: _registration_adjusted_fisher(F, registration_covariance)
        for candidate_key, F in matrices.items()
    }
    adjusted_crlb: dict[str, dict[str, Any]] = {}
    best_single_candidate: str | None = None
    best_single_sigma_xy: float | None = None
    for candidate_key, F in adjusted.items():
        sigma_xy, xy_singular = _sigma_xy_from_fisher(F)
        axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F)
        adjusted_crlb[candidate_key] = {
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
            best_single_candidate = candidate_key

    def _fuse_subset(subset: tuple[str, ...]) -> dict[str, Any]:
        F_sum = np.zeros(shape, dtype=float)
        for candidate_key in subset:
            F_sum = F_sum + adjusted[candidate_key]
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
            "candidates_used": list(subset),
        }

    if subset_size is None:
        fused = _fuse_subset(tuple(candidates))
        subset_search_count = 1
    else:
        fused = None
        subset_search_count = 0
        for subset in itertools.combinations(candidates, subset_size):
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
            fused = _fuse_subset(next(iter(itertools.combinations(candidates, subset_size))))

    selected_parent_metadata = (
        {
            m: candidate_parent_metadata[m]
            for m in fused["candidates_used"]
            if m in candidate_parent_metadata
        }
        if candidate_parent_metadata else None
    ) or None
    parent_status_metadata = combine_parent_statuses(selected_parent_metadata)
    physical_metadata = _fusion_subset_metadata_for_precomputed_matrices(
        fused["candidates_used"],
        candidate_profile_cards,
        candidate_parent_metadata,
    )
    physical_fusion_allowed = bool(physical_metadata["physically_feasible_fusion_allowed"])
    fusion_validation_status = parent_status_metadata["validation_status"]
    production_grid_diagnostic = parent_status_metadata["production_grid_diagnostic"]
    if not physical_fusion_allowed:
        fusion_validation_status = ValidationStatus.DIAGNOSTIC_ONLY.value
        production_grid_diagnostic = True

    result: dict[str, Any] = {
        "per_candidate_fisher": matrices,
        "registration_adjusted_per_candidate_fisher": adjusted,
        "registration_adjusted_per_candidate_crlb": adjusted_crlb,
        "fusion_fisher": fused["fusion_fisher"],
        "fusion_sigma_x_nm": fused["fusion_sigma_x_nm"],
        "fusion_sigma_y_nm": fused["fusion_sigma_y_nm"],
        "fusion_sigma_xy_nm": fused["fusion_sigma_xy_nm"],
        "fusion_singular": fused["fusion_singular"],
        "fusion_xy_singular": fused["fusion_xy_singular"],
        "fusion_rank": fused["fusion_rank"],
        "fusion_axes_singular": fused["fusion_axes_singular"],
        "best_single_candidate": best_single_candidate,
        "best_single_xy_candidate": best_single_candidate,
        "best_single_sigma_xy_nm": best_single_sigma_xy,
        "candidates_used": fused["candidates_used"],
        "subset_size": subset_size,
        "subset_search_count": subset_search_count,
        "registration_covariance": (
            None
            if registration_covariance is None
            else np.asarray(registration_covariance, dtype=float)
        ),
        "registration_covariance_units": "nm^2",
        "candidate_records": matrix_candidate_metadata_records(candidate_list),
        "candidate_keys": candidates,
        "fusion_complementarity": _fusion_complementarity_metrics(
            adjusted,
            fused["candidates_used"],
        ),
        "fusion_physical_metadata": physical_metadata,
        "parent_status_metadata": parent_status_metadata,
        "parent_convergence_statuses": _parent_convergence_statuses(selected_parent_metadata),
        "parent_validation_statuses": _parent_validation_statuses(selected_parent_metadata),
        "validation_status": parent_status_metadata["validation_status"],
        "fusion_validation_status": fusion_validation_status,
        "production_grid_diagnostic": production_grid_diagnostic,
        "safe_for_fusion": bool(parent_status_metadata["safe_for_fusion"]) and physical_fusion_allowed,
        "fusion_interpretation": physical_metadata["fusion_mode"],
        "physical_compatibility_status": physical_metadata["physical_compatibility_status"],
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

__all__ = ['sigma_xy_from_fisher', 'compute_candidate_registration_degradation_curve', 'compute_fisher_candidate_fusion_crlb', 'compute_candidate_fusion_crlb_from_fisher_matrices']
