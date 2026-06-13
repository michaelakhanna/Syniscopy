"""Unified Fisher information-object algebra for microscope comparison.

Motivation
----------
Ranking, fusion, and serial time-allocation are, in the current codebase, three
separate subsystems (``fisher/fusion.py``, ``fisher/time_allocation.py``, and the
ranking logic in ``lab_fisher_report/tables.py``), each with its own keying and
status bookkeeping. They are, however, three operations on a single underlying
object: a Fisher information matrix on a shared physical parameter frame.

* ranking is ``argmin`` of a scalarized Cramer--Rao functional over a set,
* fusion is ``+`` of independent-channel Fisher matrices on the same frame,
* serial allocation is a convex combination of information-*rate* matrices
  (``F / dt``) on the same frame.

A fused subset is therefore itself a derived information object on that frame,
and a scheduled allocation is a weighted information object on that frame.
Loewner ordering gives the partial order that underlies both dominance pruning
for allocation and the (more honest) partial-order view of a comparison.

This module is pure-NumPy and depends only on ``fisher/_constants`` (the shared
numerical-tolerance source of truth), so it stays safe to add while the rest of
the engine is mid-refactor, and the microscope-keyed report layer can adopt it
incrementally by wrapping the Fisher matrices it already computes.

Conventions
-----------
* ``F`` is a symmetric positive-semidefinite Fisher matrix in detector-domain
  information units on a named parameter frame (``axes``).
* Covariance is the Cramer--Rao bound ``cov = F^{-1}`` (pseudo-inverse for
  singular axes; variance along a Fisher-null axis is ``+inf``).
* All scalarizations follow the "smaller is better" convention (they are
  functionals of the covariance, not of the information).
* ``delta_t`` is the per-frame acquisition cost used to form information-rate
  matrices ``A = F / delta_t`` for serial scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

# Single source of truth for the scale-relative rank/singularity tolerance:
# the same gate fusion uses, imported from fisher/_constants rather than
# re-declared as a magic number here.
from ._constants import _FISHER_RANK_RELATIVE_TOL as _SINGULAR_REL_TOL


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Fisher matrix must be square 2D; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Fisher matrix must contain only finite values.")
    return 0.5 * (arr + arr.T)


def _eigh_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues (clipped at 0) and eigenvectors of a symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(matrix))
    eigvals = np.where(eigvals < 0.0, 0.0, eigvals)
    return eigvals, eigvecs


@dataclass(frozen=True)
class InformationObject:
    """A Fisher information matrix on a named parameter frame, plus provenance.

    The object is the single type that ranking, fusion, and allocation all act
    on. Construct one per microscope (or per fused subset, or per allocation),
    then compose with :meth:`fuse`, :meth:`rate`, and the module-level
    ``rank``/``allocate``/``loewner_partial_order`` helpers.
    """

    fisher: np.ndarray
    axes: tuple[str, ...]
    label: str = ""
    modality: str = ""
    delta_t: float = 1.0
    kind: str = "single"  # single | fused | rate | allocation
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        f = _symmetrize(self.fisher)
        if len(self.axes) != f.shape[0]:
            raise ValueError(
                f"axes length {len(self.axes)} must match Fisher dimension {f.shape[0]}."
            )
        if not (np.isfinite(self.delta_t) and self.delta_t > 0.0):
            raise ValueError(f"delta_t must be a positive finite frame time; got {self.delta_t!r}.")
        object.__setattr__(self, "fisher", f)
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))

    # -- spectral helpers -------------------------------------------------
    @property
    def dim(self) -> int:
        return self.fisher.shape[0]

    def _eig(self) -> tuple[np.ndarray, np.ndarray]:
        return _eigh_psd(self.fisher)

    def singular_axis_mask(self) -> np.ndarray:
        """Boolean mask of parameter axes with no finite Cramer--Rao bound."""
        eigvals, eigvecs = self._eig()
        if eigvals.size == 0:
            return np.zeros(0, dtype=bool)
        tol = _SINGULAR_REL_TOL * float(eigvals.max()) if eigvals.max() > 0 else 0.0
        null = eigvals <= tol
        # An axis is singular if it has weight in any null eigenvector.
        return np.any(np.abs(eigvecs[:, null]) > 1e-12, axis=1)

    @property
    def rank(self) -> int:
        eigvals, _ = self._eig()
        if eigvals.size == 0:
            return 0
        tol = _SINGULAR_REL_TOL * float(eigvals.max()) if eigvals.max() > 0 else 0.0
        return int(np.sum(eigvals > tol))

    @property
    def is_singular(self) -> bool:
        return self.rank < self.dim

    # -- Cramer--Rao bound ------------------------------------------------
    def covariance(self) -> np.ndarray:
        """Cramer--Rao covariance ``F^{-1}`` (``+inf`` on Fisher-null axes)."""
        eigvals, eigvecs = self._eig()
        if eigvals.size == 0:
            return np.zeros((0, 0))
        tol = _SINGULAR_REL_TOL * float(eigvals.max()) if eigvals.max() > 0 else 0.0
        inv = np.where(eigvals > tol, 1.0 / np.where(eigvals > tol, eigvals, 1.0), np.inf)
        if np.any(np.isinf(inv)):
            # Build covariance with explicit infinities only on null directions.
            cov = np.full((self.dim, self.dim), np.nan)
            finite = eigvals > tol
            cov_finite = (eigvecs[:, finite] * inv[finite]) @ eigvecs[:, finite].T
            null_dir = np.any(np.abs(eigvecs[:, ~finite]) > 1e-12, axis=1)
            cov = cov_finite
            cov[null_dir, :] = np.inf
            cov[:, null_dir] = np.inf
            return cov
        return (eigvecs * inv) @ eigvecs.T

    def sigma_per_axis(self) -> dict[str, float]:
        cov = self.covariance()
        if cov.size == 0:
            return {}
        diag = np.diag(cov)
        return {axis: float(np.sqrt(v)) if np.isfinite(v) else float("inf")
                for axis, v in zip(self.axes, diag)}

    def _axis_sigmas(self, axes: Sequence[str] | None = None) -> list[float]:
        per = self.sigma_per_axis()
        keys = list(self.axes) if axes is None else [str(a) for a in axes]
        return [per[k] for k in keys if k in per]

    def sigma_l2(self, axes: Sequence[str] | None = None) -> float:
        """Total joint Cramer--Rao sigma over ``axes``.

        For lateral ``("x", "y")`` this is the paper-facing convention
        ``sqrt(sigma_x^2 + sigma_y^2)``, not an RMS one-axis precision.
        """

        vals = self._axis_sigmas(axes)
        if not vals or any(not np.isfinite(v) for v in vals):
            return float("inf")
        return float(np.sqrt(np.sum(np.square(vals))))

    def sigma_rms(self, axes: Sequence[str] | None = None) -> float:
        """RMS Cramer--Rao sigma over ``axes`` under an explicit RMS name."""

        vals = self._axis_sigmas(axes)
        if not vals or any(not np.isfinite(v) for v in vals):
            return float("inf")
        return float(np.sqrt(np.mean(np.square(vals))))

    # -- scalarizations (smaller is better) -------------------------------
    def scalarize(self, criterion: str = "A", axes: Sequence[str] | None = None) -> float:
        """A/D/E-optimal Cramer--Rao scalar on an axis sub-block.

        ``A`` = trace(cov) (sum of variances), ``D`` = det(cov) (volume), and
        ``E`` = max eigenvalue of cov (worst axis). Returns ``+inf`` when the
        requested sub-block is Fisher-singular.
        """
        idx = self._axis_indices(axes)
        sub = self.fisher[np.ix_(idx, idx)]
        eigvals, _ = _eigh_psd(sub)
        if eigvals.size == 0:
            return float("inf")
        tol = _SINGULAR_REL_TOL * float(eigvals.max()) if eigvals.max() > 0 else 0.0
        if np.any(eigvals <= tol):
            return float("inf")
        inv = 1.0 / eigvals
        crit = criterion.strip().upper()
        if crit == "A":
            return float(np.sum(inv))
        if crit == "D":
            return float(np.prod(inv))
        if crit == "E":
            return float(np.max(inv))
        raise ValueError(f"Unknown scalarization criterion {criterion!r}; use 'A', 'D', or 'E'.")

    def _axis_indices(self, axes: Sequence[str] | None) -> list[int]:
        if axes is None:
            return list(range(self.dim))
        lookup = {a: i for i, a in enumerate(self.axes)}
        try:
            return [lookup[str(a)] for a in axes]
        except KeyError as exc:
            raise ValueError(f"axis {exc} not in frame {self.axes}.") from None

    # -- operations -------------------------------------------------------
    def rate(self) -> "InformationObject":
        """Information-rate object ``A = F / delta_t`` for serial scheduling."""
        return replace(
            self,
            fisher=self.fisher / self.delta_t,
            delta_t=1.0,
            kind="rate",
            metadata={**dict(self.metadata), "rate_from_delta_t": self.delta_t},
        )

    def with_registration(self, registration_cov: np.ndarray) -> "InformationObject":
        """Degrade by an additive lateral registration covariance (nm^2 space).

        Applies ``F' = (F^{-1} + Sigma)^{-1}`` in covariance space, the standard
        independent-Gaussian alignment-error model used before fusion. ``Sigma``
        must be a PSD matrix on this object's frame (typically nonzero only on
        the lateral block).
        """
        sigma = _symmetrize(registration_cov)
        if sigma.shape[0] != self.dim:
            raise ValueError("registration covariance must match the object frame dimension.")
        cov = self.covariance()
        if np.any(np.isinf(cov)):
            raise ValueError("registration adjustment requires a finite (nonsingular) covariance.")
        adjusted = np.linalg.pinv(cov + sigma)
        return replace(self, fisher=_symmetrize(adjusted), kind=self.kind,
                       metadata={**dict(self.metadata), "registration_adjusted": True})

    def fuse(self, *others: "InformationObject", label: str | None = None) -> "InformationObject":
        """Independent-channel Fisher sum on a shared frame (the ``+`` operation)."""
        return fuse((self, *others), label=label)


def _require_shared_frame(objects: Sequence[InformationObject]) -> tuple[str, ...]:
    if not objects:
        raise ValueError("at least one information object is required.")
    axes = objects[0].axes
    for obj in objects[1:]:
        if obj.axes != axes:
            raise ValueError(
                f"information objects must share one parameter frame; "
                f"{obj.axes} != {axes}."
            )
    return axes


def fuse(objects: Sequence[InformationObject], label: str | None = None) -> InformationObject:
    """Sum independent-channel Fisher matrices into one derived object.

    A fused subset is itself an :class:`InformationObject` on the same frame, so
    fusion outputs feed straight back into ranking and allocation.
    """
    axes = _require_shared_frame(objects)
    total = np.zeros((len(axes), len(axes)))
    for obj in objects:
        total = total + obj.fisher
    members = [obj.label or obj.modality for obj in objects]
    return InformationObject(
        fisher=total,
        axes=axes,
        label=label or "+".join(m for m in members if m),
        kind="fused",
        metadata={"fused_members": members},
    )


def loewner_relation(a: InformationObject, b: InformationObject, *, tol_rel: float = 1e-9) -> str:
    """Loewner comparison of two information objects on a shared frame.

    Returns ``"a>=b"`` if ``a`` dominates ``b`` (``a - b`` PSD: ``a`` carries at
    least as much information on every direction), ``"b>=a"`` if ``b`` dominates,
    ``"equal"`` if both, or ``"incomparable"``.
    """
    _require_shared_frame((a, b))
    diff = a.fisher - b.fisher
    scale = max(np.linalg.norm(a.fisher), np.linalg.norm(b.fisher), 1.0)
    eig = np.linalg.eigvalsh(_symmetrize(diff))
    tol = tol_rel * scale
    a_dom = bool(np.all(eig >= -tol))
    b_dom = bool(np.all(eig <= tol))
    if a_dom and b_dom:
        return "equal"
    if a_dom:
        return "a>=b"
    if b_dom:
        return "b>=a"
    return "incomparable"


def loewner_partial_order(objects: Sequence[InformationObject]) -> dict[tuple[int, int], str]:
    """Pairwise Loewner relations for a set (the partial order behind dominance)."""
    rels: dict[tuple[int, int], str] = {}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            rels[(i, j)] = loewner_relation(objects[i], objects[j])
    return rels


def dominated_indices(objects: Sequence[InformationObject]) -> set[int]:
    """Indices strictly Loewner-dominated by another object on the rate basis.

    Used to prune candidates before serial time-allocation: a strictly
    dominated information-rate matrix can be dropped without changing a
    total-information-monotone scheduling optimum.
    """
    rates = [obj.rate() for obj in objects]
    dominated: set[int] = set()
    for i in range(len(rates)):
        for j in range(len(rates)):
            if i == j:
                continue
            rel = loewner_relation(rates[j], rates[i])
            if rel == "a>=b":  # j dominates i
                # strict only if not equal
                if loewner_relation(rates[i], rates[j]) != "a>=b":
                    dominated.add(i)
    return dominated


def rank(
    objects: Sequence[InformationObject],
    *,
    criterion: str = "A",
    axes: Sequence[str] | None = None,
    tie_rel_tol: float = 1e-9,
) -> list[dict[str, Any]]:
    """Order information objects by a scalarized Cramer--Rao functional.

    This is the unified ``argmin`` that replaces the bespoke ranking pass. Ties
    (within ``tie_rel_tol``) share a rank, and Fisher-singular objects sort last
    with rank ``None`` (not rankable for ordering).
    """
    scored = [
        (obj, obj.scalarize(criterion, axes=axes))
        for obj in objects
    ]
    finite = sorted((s for s in scored if np.isfinite(s[1])), key=lambda s: s[1])
    singular = [s for s in scored if not np.isfinite(s[1])]

    rows: list[dict[str, Any]] = []
    current_rank = 0
    last_score: float | None = None
    for obj, score in finite:
        if last_score is None or abs(score - last_score) > tie_rel_tol * max(abs(score), 1.0):
            current_rank += 1
            last_score = score
        rows.append({
            "label": obj.label,
            "modality": obj.modality,
            "rank": current_rank,
            "criterion": criterion.upper(),
            "score": score,
            "sigma_l2_nm": obj.sigma_l2(axes),
            "sigma_rms_nm": obj.sigma_rms(axes),
            "rankable": True,
        })
    for obj, _ in singular:
        rows.append({
            "label": obj.label,
            "modality": obj.modality,
            "rank": None,
            "criterion": criterion.upper(),
            "score": float("inf"),
            "sigma_l2_nm": float("inf"),
            "sigma_rms_nm": float("inf"),
            "rankable": False,
        })
    return rows


def pareto_front(
    objects: Sequence[InformationObject],
    metric_fns: Sequence[Any],
) -> list[int]:
    """Indices on the Pareto front of several "lower-is-better" metrics.

    A partial-order view of a comparison (the honest codomain when a single
    scalar ranking would be misleading): an object is dominated only if another
    is at least as good on every metric and strictly better on one.
    """
    metrics = [tuple(float(fn(obj)) for fn in metric_fns) for obj in objects]
    front: list[int] = []
    for i, mi in enumerate(metrics):
        dominated = False
        for j, mj in enumerate(metrics):
            if i == j:
                continue
            if all(a <= b for a, b in zip(mj, mi)) and any(a < b for a, b in zip(mj, mi)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def combine_rate(objects: Sequence[InformationObject], weights: Sequence[float]) -> InformationObject:
    """Weighted sum of information-*rate* matrices: ``sum_i w_i (F_i / dt_i)``.

    This is the object a serial time-allocation produces; with ``sum w_i = T``
    it is the total-time Fisher information of a schedule.
    """
    axes = _require_shared_frame(objects)
    if len(weights) != len(objects):
        raise ValueError("weights length must match number of objects.")
    total = np.zeros((len(axes), len(axes)))
    for obj, w in zip(objects, weights):
        if w < 0:
            raise ValueError("allocation weights must be nonnegative.")
        total = total + float(w) * (obj.fisher / obj.delta_t)
    return InformationObject(
        fisher=total, axes=axes, label="allocation", kind="allocation",
        metadata={"weights": [float(w) for w in weights]},
    )


def allocate(
    objects: Sequence[InformationObject],
    *,
    total_time: float = 1.0,
    criterion: str = "A",
    axes: Sequence[str] | None = None,
    prune_dominated: bool = True,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Reference serial time-allocation over the information-rate algebra.

    Minimizes the A/D/E-optimal scalarization of ``combine_rate`` on the
    time simplex via projected gradient (Frank--Wolfe style). This is a compact
    reference that demonstrates allocation as an operation on the same algebra;
    the production solver in ``fisher/time_allocation.py`` remains authoritative
    for the paper tables.
    """
    objs = list(objects)
    keep = list(range(len(objs)))
    if prune_dominated:
        dominated = dominated_indices(objs)
        keep = [i for i in keep if i not in dominated] or keep
    active = [objs[i] for i in keep]

    n = len(active)
    w = np.full(n, total_time / n)

    def objective(weights: np.ndarray) -> float:
        return combine_rate(active, weights).scalarize(criterion, axes=axes)

    best = objective(w)
    for _ in range(max_iter):
        # numerical gradient on the simplex
        grad = np.zeros(n)
        eps = total_time * 1e-4
        for k in range(n):
            wp = w.copy(); wp[k] += eps
            grad[k] = (objective(wp) - best) / eps
        # Frank-Wolfe vertex: all time to the steepest-descent coordinate
        vertex = np.zeros(n)
        vertex[int(np.argmin(grad))] = total_time
        # line search
        improved = False
        for step in (1.0, 0.5, 0.25, 0.1, 0.05):
            cand = (1 - step) * w + step * vertex
            val = objective(cand)
            if val < best - 1e-12 * max(abs(best), 1.0):
                w, best, improved = cand, val, True
                break
        if not improved:
            break

    weights = {active[i].label or active[i].modality or str(i): float(w[i]) for i in range(n)}
    result = combine_rate(active, w)
    return {
        "criterion": criterion.upper(),
        "total_time": float(total_time),
        "weights": weights,
        "objective": float(best),
        "sigma_l2_nm": result.sigma_l2(axes),
        "sigma_rms_nm": result.sigma_rms(axes),
        "pruned_dominated": prune_dominated,
        "active_labels": [active[i].label or active[i].modality for i in range(n)],
        "information_object": result,
    }


__all__ = [
    "InformationObject",
    "fuse",
    "loewner_relation",
    "loewner_partial_order",
    "dominated_indices",
    "rank",
    "pareto_front",
    "combine_rate",
    "allocate",
]
