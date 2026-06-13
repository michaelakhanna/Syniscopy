"""Partial-order / Pareto view of a microscope comparison.

Why this exists
---------------
A microscope comparison answers "which instrument is better?", and the report
has historically answered with a single scalar rank over ``sigma_xy``. The moment
the project needed four separate contracts (LP, LZ, Q, NR) to stop readers from
treating one number as a universal verdict, the data was signalling that the
honest codomain is a *partial order*, not a total order on the reals.

Two microscopes are often genuinely incomparable: one localizes better but
costs more dose, is destructive, or works only in a different measurement
domain. Forcing them onto one axis hides that. This module produces the partial
order directly:

* **Loewner information order** -- microscope A information-dominates B when its
  Fisher matrix is at least as large on every parameter direction (``A - B`` is
  PSD). This is contract-independent: it compares the actual information, not a
  scalarization.
* **Multi-criterion Pareto front** -- over precision plus any "lower-is-better"
  cost axes the user supplies (dose, acquisition time, destructiveness, ...), a
  microscope is dominated only if another is at least as good on *every* axis
  and strictly better on one. The front is the set of defensible choices.

The single scalar rank is retained for reference, but as one projection of the
partial order rather than the verdict.

This module is additive and depends only on ``fisher.information_object`` (pure
NumPy); it takes generic candidate records, so it does not couple to the
in-flux report row schema and can be emitted alongside the scalar ranking once
the microscope-keyed report lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from fisher.information_object import InformationObject, loewner_relation


@dataclass(frozen=True)
class MicroscopeCandidate:
    """One comparison candidate: an information object plus lower-is-better costs.

    ``info`` carries the Fisher matrix (the precision axis). ``costs`` are
    additional acquisition-cost / penalty axes where smaller is better, e.g.
    ``{"dose": ..., "acquisition_time_s": ..., "destructive": 0 or 1}``. Axes a
    given microscope does not define are simply absent and are skipped for that
    candidate's comparisons.
    """

    name: str
    info: InformationObject
    costs: Mapping[str, float] = field(default_factory=dict)
    modality: str = ""

    def sigma_xy_nm(self, axes: Sequence[str] | None = ("x", "y")) -> float:
        keys = [a for a in (axes or ()) if a in self.info.axes] or None
        return self.info.sigma_l2(keys)


def loewner_information_order(
    candidates: Sequence[MicroscopeCandidate],
) -> dict[str, Any]:
    """Information dominance edges and the information-maximal set.

    Returns dominance edges ``(winner, loser)`` where ``winner`` strictly
    Loewner-dominates ``loser`` on the Fisher matrix, plus the set of candidates
    not dominated by any other (the information-maximal microscopes).
    """
    edges: list[tuple[str, str]] = []
    dominated: set[str] = set()
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i == j:
                continue
            if a.info.axes != b.info.axes:
                continue  # not on a shared frame; not comparable here
            rel = loewner_relation(a.info, b.info)
            rel_back = loewner_relation(b.info, a.info)
            if rel == "a>=b" and rel_back != "a>=b":  # strict domination
                edges.append((a.name, b.name))
                dominated.add(b.name)
    maximal = [c.name for c in candidates if c.name not in dominated]
    return {"edges": edges, "information_maximal": maximal, "dominated": sorted(dominated)}


def _criterion_axes(
    candidates: Sequence[MicroscopeCandidate],
    extra_cost_axes: Sequence[str] | None,
) -> list[str]:
    if extra_cost_axes is not None:
        return list(extra_cost_axes)
    seen: list[str] = []
    for c in candidates:
        for k in c.costs:
            if k not in seen:
                seen.append(k)
    return seen


def pareto_front(
    candidates: Sequence[MicroscopeCandidate],
    *,
    cost_axes: Sequence[str] | None = None,
    include_precision: bool = True,
) -> dict[str, Any]:
    """Multi-criterion Pareto front over precision + lower-is-better cost axes.

    ``cost_axes`` defaults to the union of all candidates' cost keys. A candidate
    is dominated only if another is no worse on every axis and strictly better on
    at least one. Candidates missing a cost axis are treated as ``+inf`` on it
    (an undeclared cost is not silently assumed best).
    """
    axes = _criterion_axes(candidates, cost_axes)

    def vector(c: MicroscopeCandidate) -> tuple[float, ...]:
        vals: list[float] = []
        if include_precision:
            vals.append(c.sigma_xy_nm())
        for ax in axes:
            vals.append(float(c.costs.get(ax, np.inf)))
        return tuple(vals)

    vectors = [vector(c) for c in candidates]

    def dominates(p: tuple[float, ...], q: tuple[float, ...]) -> bool:
        return all(a <= b for a, b in zip(p, q)) and any(a < b for a, b in zip(p, q))

    front: list[str] = []
    dominated_by: dict[str, list[str]] = {}
    for i, ci in enumerate(candidates):
        doms = [candidates[j].name for j in range(len(candidates))
                if j != i and dominates(vectors[j], vectors[i])]
        if doms:
            dominated_by[ci.name] = doms
        else:
            front.append(ci.name)
    return {
        "criteria": (["sigma_xy_nm"] if include_precision else []) + list(axes),
        "front": front,
        "dominated_by": dominated_by,
    }


def scalar_ranking(
    candidates: Sequence[MicroscopeCandidate],
) -> list[dict[str, Any]]:
    """The single-axis projection (sigma_xy), kept for reference only."""
    rows = []
    for c in candidates:
        rows.append({"name": c.name, "modality": c.modality, "sigma_xy_nm": c.sigma_xy_nm()})
    finite = [r for r in rows if np.isfinite(r["sigma_xy_nm"])]
    singular = [r for r in rows if not np.isfinite(r["sigma_xy_nm"])]
    finite.sort(key=lambda r: r["sigma_xy_nm"])
    rank = 0
    last = None
    for r in finite:
        if last is None or abs(r["sigma_xy_nm"] - last) > 1e-9 * max(r["sigma_xy_nm"], 1.0):
            rank += 1
            last = r["sigma_xy_nm"]
        r["scalar_rank"] = rank
    for r in singular:
        r["scalar_rank"] = None
    return finite + singular


def partial_order_report(
    candidates: Sequence[MicroscopeCandidate],
    *,
    cost_axes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full partial-order comparison plus CSV-ready rows.

    Combines the Loewner information order, the multi-criterion Pareto front, and
    the reference scalar projection into one report payload. Each output row
    states whether a microscope is on the Pareto front, whether it is
    information-maximal, who (if anyone) dominates it, and its scalar rank.
    """
    loewner = loewner_information_order(candidates)
    pareto = pareto_front(candidates, cost_axes=cost_axes)
    scalar = {r["name"]: r for r in scalar_ranking(candidates)}

    front = set(pareto["front"])
    info_max = set(loewner["information_maximal"])
    rows: list[dict[str, Any]] = []
    axes = _criterion_axes(candidates, cost_axes)
    for c in candidates:
        row = {
            "name": c.name,
            "modality": c.modality,
            "sigma_xy_nm": c.sigma_xy_nm(),
            "on_pareto_front": c.name in front,
            "loewner_information_maximal": c.name in info_max,
            "dominated_by": pareto["dominated_by"].get(c.name, []),
            "scalar_rank": scalar.get(c.name, {}).get("scalar_rank"),
        }
        for ax in axes:
            row[ax] = c.costs.get(ax)
        rows.append(row)

    n_front = len(front)
    summary = (
        f"{len(candidates)} microscopes; {n_front} on the Pareto front over "
        f"{pareto['criteria']}. "
        + ("A single best microscope is defensible." if n_front == 1
           else "No single best microscope: the choices on the front are incomparable "
                "without a stated preference among the criteria.")
    )
    return {
        "summary": summary,
        "rows": rows,
        "pareto_criteria": pareto["criteria"],
        "pareto_front": pareto["front"],
        "information_maximal": loewner["information_maximal"],
        "loewner_edges": loewner["edges"],
        "scalar_ranking": list(scalar.values()),
    }


__all__ = [
    "MicroscopeCandidate",
    "loewner_information_order",
    "pareto_front",
    "scalar_ranking",
    "partial_order_report",
]
