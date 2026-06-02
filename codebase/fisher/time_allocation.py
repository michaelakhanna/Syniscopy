"""Time-allocation and Loewner-dominance Fisher diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment_contracts import combine_parent_statuses, normalize_convergence_status

from ._constants import (
    _LINE_SEARCH_ARMIJO_C,
    _LINE_SEARCH_DESCENT_TOL,
    _LINE_SEARCH_MAX_STEPS,
    _LINE_SEARCH_SHRINK,
    _RELATIVE_DET_SINGULAR_TOL,
)

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
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
    acquisition_cost_by_modality: dict[str, dict[str, Any]] | None = None,
    modality_constraints: dict[str, Any] | None = None,
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
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

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
    status_metadata = combine_parent_statuses(parent_result_metadata_by_modality)
    selected_status = None
    if best_single_m is not None and parent_result_metadata_by_modality:
        selected_status = parent_result_metadata_by_modality.get(best_single_m, {}).get("convergence_status")
    result["parent_status_metadata"] = status_metadata
    result["parent_convergence_statuses"] = _parent_convergence_statuses(parent_result_metadata_by_modality)
    result["parent_validation_statuses"] = _parent_validation_statuses(parent_result_metadata_by_modality)
    result["validation_status"] = status_metadata["validation_status"]
    result["time_allocation_validation_status"] = status_metadata["validation_status"]
    result["production_grid_diagnostic"] = status_metadata["production_grid_diagnostic"]
    result["safe_for_time_allocation"] = status_metadata["safe_for_time_allocation"]
    result["selected_modality_convergence_status"] = normalize_convergence_status(selected_status)
    result["acquisition_cost_by_modality"] = acquisition_cost_by_modality or {}
    result["modality_constraints"] = modality_constraints or {}
    result["boundary_solution"] = any(v <= 1e-12 or abs(v - total_time_seconds) <= 1e-12 for v in optimal_time_seconds.values())
    result["physical_recommendation_status"] = (
        "constrained_physical_recommendation"
        if acquisition_cost_by_modality or modality_constraints
        else "equal-cost-serial-diagnostic"
    )

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

__all__ = ['compute_loewner_dominance', 'compute_optimal_time_allocation_crlb']
