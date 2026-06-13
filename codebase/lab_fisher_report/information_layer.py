"""One-call information layer for microscope comparison.

This is the integration seam for the new conceptual modules. Given a set of
per-microscope observations (the contrast image, noise map, pixel size, and any
acquisition costs the report already has on hand), it produces the full
comparison payload:

* per-microscope Cramer--Rao bounds via the continuous band-limited derivative
  (``fisher.spectral_fisher``) -- no derivative-step convergence gate;
* a unified information object per microscope (``fisher.information_object``);
* a scalar ranking, a best-k fusion search, and a serial time allocation, all as
  operations on that one algebra;
* a partial-order / Pareto view (``lab_fisher_report.partial_order``) -- the
  honest codomain when one scalar would mislead.

It also implements the paper's distinction between **algebraic** fusion (the
Fisher sum, always computable on the shared position frame) and **physically
valid** fusion (independent channels, mutually compatible, with no two channels
that are alternate reconstructions of the same detected quanta). A fused subset
is reported with both, never collapsing them.

The refactored ``coordinator`` adopts this by building one
:class:`MicroscopeObservation` per rendered microscope and calling
:func:`build_information_layer`; nothing here depends on the in-flux report row
schema. Pure NumPy plus the additive modules above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from fisher.information_object import InformationObject, fuse, allocate
from fisher.spectral_fisher import lateral_fisher_continuous
from lab_fisher_report.partial_order import MicroscopeCandidate, partial_order_report

LATERAL_AXES = ("x", "y")


@dataclass(frozen=True)
class MicroscopeObservation:
    """Inputs for one microscope's information bound.

    Either supply ``contrast`` + ``noise_variance`` + ``pixel_size_nm`` (the
    Fisher matrix is computed with the continuous derivative), or pass a
    precomputed 2x2 ``fisher_override``. ``costs`` are lower-is-better
    acquisition-cost axes (dose, time, destructiveness). ``independent_channel``
    and ``compatibility_class`` drive physically-valid fusion: two channels in
    the same nonempty class are treated as alternate reconstructions of the same
    detected quanta and must not be fused together.
    """

    name: str
    modality: str = ""
    contrast: np.ndarray | None = None
    noise_variance: np.ndarray | float | None = None
    pixel_size_nm: float = 1.0
    delta_t: float = 1.0
    costs: Mapping[str, float] = field(default_factory=dict)
    independent_channel: bool = True
    compatibility_class: str = ""


def information_object_for(obs: MicroscopeObservation) -> tuple[InformationObject, dict[str, Any]]:
    """Build the InformationObject for one observation, plus its diagnostics."""
    if obs.contrast is not None and obs.noise_variance is not None:
        diag = lateral_fisher_continuous(obs.contrast, obs.noise_variance, obs.pixel_size_nm)
        fisher = np.asarray(diag["fisher_matrix"], dtype=float)
    else:
        raise ValueError(
            f"microscope {obs.name!r} needs contrast + noise_variance "
            "(no precomputed Fisher path is configured here)."
        )
    info = InformationObject(
        fisher=fisher, axes=LATERAL_AXES, label=obs.name, modality=obs.modality,
        delta_t=obs.delta_t,
        metadata={
            "independent_channel": bool(obs.independent_channel),
            "compatibility_class": str(obs.compatibility_class),
            "derivative_basis": diag.get("derivative_basis"),
            "boundary_energy_fraction": diag.get("boundary_energy_fraction"),
            "nyquist_band_fraction": diag.get("nyquist_band_fraction"),
        },
    )
    return info, diag


def physical_fusion_status(infos: Sequence[InformationObject]) -> dict[str, Any]:
    """Whether a subset is a physically valid fusion, with the reason if not.

    Valid only when every channel is independent and no two share a nonempty
    compatibility class (which would double-count the same detected quanta).
    """
    reasons: list[str] = []
    if any(not bool(i.metadata.get("independent_channel", True)) for i in infos):
        reasons.append("contains a non-independent channel")
    classes = [str(i.metadata.get("compatibility_class", "")) for i in infos]
    nonempty = [c for c in classes if c]
    if len(nonempty) != len(set(nonempty)):
        reasons.append("two channels share a compatibility class (same detected quanta)")
    return {"physically_valid": not reasons, "reasons": reasons}


def _best_k_fusion(
    infos: Sequence[InformationObject], *, max_k: int, axes: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(infos)
    for k in range(1, min(max_k, n) + 1):
        best = None
        for subset in combinations(range(n), k):
            members = [infos[i] for i in subset]
            fused = fuse(members)
            sigma = fused.sigma_l2(axes)
            if best is None or sigma < best["fusion_sigma_xy_nm"]:
                status = physical_fusion_status(members)
                best = {
                    "subset_size": k,
                    "microscopes": [m.label for m in members],
                    "fusion_sigma_xy_nm": sigma,
                    "algebraic": True,  # the Fisher sum is always computable
                    "physically_valid": status["physically_valid"],
                    "physical_reasons": status["reasons"],
                }
        if best is not None:
            rows.append(best)
    return rows


def build_information_layer(
    observations: Sequence[MicroscopeObservation],
    *,
    criterion: str = "A",
    axes: Sequence[str] = LATERAL_AXES,
    max_fusion_k: int = 4,
    total_time: float = 1.0,
) -> dict[str, Any]:
    """Compose the full microscope comparison from per-microscope observations.

    Returns a payload with per-microscope bounds + diagnostics, the scalar
    ranking, best-k fusion (algebraic and physically-valid flags), the serial
    time allocation, and the partial-order / Pareto report. Each section is an
    operation on the same :class:`InformationObject` set, so they stay mutually
    consistent.
    """
    if not observations:
        raise ValueError("build_information_layer requires at least one observation.")

    infos: list[InformationObject] = []
    per_microscope: list[dict[str, Any]] = []
    candidates: list[MicroscopeCandidate] = []
    for obs in observations:
        info, diag = information_object_for(obs)
        infos.append(info)
        per_microscope.append({
            "name": obs.name,
            "modality": obs.modality,
            "sigma_xy_nm": info.sigma_l2(axes),
            "singular": info.is_singular,
            "derivative_basis": diag.get("derivative_basis"),
            "step_size_free": diag.get("step_size_free"),
            "nyquist_band_fraction": diag.get("nyquist_band_fraction"),
            "boundary_energy_fraction": diag.get("boundary_energy_fraction"),
        })
        candidates.append(MicroscopeCandidate(
            name=obs.name, info=info, costs=dict(obs.costs), modality=obs.modality))

    from fisher.information_object import rank as rank_objects

    ranking = rank_objects(infos, criterion=criterion, axes=axes)
    fusion = _best_k_fusion(infos, max_k=max_fusion_k, axes=axes)
    allocation = allocate(infos, total_time=total_time, criterion=criterion, axes=axes)
    partial_order = partial_order_report(candidates)

    return {
        "criterion": criterion.upper(),
        "derivative_basis": "spectral_band_limited",
        "per_microscope": per_microscope,
        "scalar_ranking": ranking,
        "fusion_best_k": fusion,
        "time_allocation": {
            "criterion": allocation["criterion"],
            "weights": allocation["weights"],
            "sigma_xy_nm": allocation["sigma_l2_nm"],
            "active": allocation["active_labels"],
        },
        "partial_order": partial_order,
    }


__all__ = [
    "MicroscopeObservation",
    "information_object_for",
    "physical_fusion_status",
    "build_information_layer",
    "LATERAL_AXES",
]
