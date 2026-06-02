"""Fisher convergence gates and result-status metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment_contracts import (
    ConvergenceStatus,
    ValidationStatus,
    combine_parent_statuses,
    normalize_convergence_status,
    stable_hash,
    wrap_legacy_crlb_result,
)

from ._constants import (
    _FISHER_EIGENVALUE_UNDERFLOW_FLOOR,
    _FISHER_RANK_RELATIVE_TOL,
    _FISHER_VARIANCE_FLOOR,
)

def _matrix_rank_condition(F: np.ndarray, rank_tolerance: float = _FISHER_RANK_RELATIVE_TOL) -> tuple[int, float | None, list[str]]:
    from .fusion import _axis_names_for_dim, _axis_sigmas_from_fisher

    F = np.asarray(F, dtype=float)
    eig = np.linalg.eigvalsh((F + F.T) / 2.0)
    if not np.all(np.isfinite(eig)):
        return 0, None, ["nonfinite"]
    max_abs = float(np.max(np.abs(eig))) if eig.size else 0.0
    tol = max(float(rank_tolerance) * max(max_abs, 1.0), _FISHER_EIGENVALUE_UNDERFLOW_FLOOR)
    positive = eig[eig > tol]
    rank = int(positive.size)
    cond = float(positive[-1] / positive[0]) if positive.size == eig.size and positive.size else None
    try:
        _axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F)
        axes = [
            axis
            for axis, singular in zip(_axis_names_for_dim(F.shape[0]), axis_singular)
            if singular
        ]
    except ValueError:
        axes = _axis_names_for_dim(F.shape[0]) if rank < F.shape[0] else []
    return rank, cond, axes

def _relative_span(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    if finite.size < 2:
        return float("inf")
    return float((np.max(finite) - np.min(finite)) / max(float(np.median(finite)), _FISHER_VARIANCE_FLOOR))

def _select_convergence_status(per_step: list[dict[str, Any]], scalar_key: str, tolerance: float, min_stable_steps: int) -> tuple[str, float | None, str]:
    if not per_step:
        return ConvergenceStatus.NOT_APPLICABLE.value, None, "no candidate steps supplied"
    if any(not np.all(np.isfinite(np.asarray(item.get("fisher_matrix"), dtype=float))) for item in per_step):
        return ConvergenceStatus.NONFINITE.value, None, "nonfinite Fisher matrix in candidate sweep"
    ranks = [int(item.get("fisher_rank", item.get("rank", 0))) for item in per_step]
    singular = [bool(item.get("singular", False)) or bool(item.get("fisher_singular", False)) for item in per_step]
    if all(singular) and len(set(ranks)) == 1:
        return ConvergenceStatus.STABLE_SINGULAR.value, float(per_step[-1]["derivative_step"]), "all tested steps are singular with stable rank"
    finite = [item for item in per_step if np.isfinite(float(item.get(scalar_key, float("nan"))))]
    if len(finite) >= min_stable_steps:
        tail = finite[-min_stable_steps:]
        span = _relative_span([float(item[scalar_key]) for item in tail])
        tail_ranks = {int(item.get("fisher_rank", item.get("rank", 0))) for item in tail}
        if span <= tolerance and len(tail_ranks) == 1:
            return ConvergenceStatus.FINITE_CONVERGED.value, float(tail[-1]["derivative_step"]), f"last {min_stable_steps} finite steps have relative {scalar_key} span {span:.3g}"
        if len(tail_ranks) > 1:
            return ConvergenceStatus.ILL_CONDITIONED.value, float(tail[-1]["derivative_step"]), "rank changes across accepted tail steps"
    return ConvergenceStatus.FAILED_CONVERGENCE.value, None, "no stable finite tail satisfied convergence tolerance"

def annotate_fisher_result_status(result: dict[str, Any], *, convergence_status: str, source_contract: str, modality: str, result_id: str | None = None) -> dict[str, Any]:
    out = dict(result)
    rid = result_id or "fisher:" + stable_hash({"modality": modality, "contract": source_contract, "result": result})[:16]
    convergence_status = normalize_convergence_status(convergence_status)
    wrapped = wrap_legacy_crlb_result(out, result_id=rid, source_contract=source_contract, modality=modality, convergence_status=convergence_status)
    out["fisher_result"] = wrapped.to_dict()
    out["result_id"] = rid
    out["convergence_status"] = convergence_status
    out["validation_status"] = wrapped.validation_status
    out["production_grid_diagnostic"] = wrapped.production_grid_diagnostic
    out["safe_for_ordering"] = wrapped.safe_for_ordering
    out["safe_for_fusion"] = wrapped.safe_for_fusion
    out["safe_for_time_allocation"] = wrapped.safe_for_time_allocation
    out["safe_for_registration"] = wrapped.safe_for_registration
    out["safe_for_detected_quanta_ranking"] = wrapped.safe_for_detected_quanta_ranking
    return out

@dataclass
class FisherConvergenceStatus:
    """Structured convergence envelope for rerendered Fisher/CRLB sweeps.

    ``status`` is always normalized through
    :func:`experiment_contracts.normalize_convergence_status` and therefore
    always equals one of the ``ConvergenceStatus`` string values. The remaining
    fields are diagnostics that explain why that scalar status was assigned.
    """

    status: str
    derivative_mode: str
    steps_tested: tuple[float, ...]
    selected_step: float | None
    max_adjacent_relative_change: float | None
    rank_range: tuple[int, int]
    singular_axes: tuple[str, ...]
    reason: str
    validation_status: str
    source_contract: str
    modality: str
    scalar_key: str
    per_step_scalars: tuple[float | None, ...]

    def __post_init__(self) -> None:
        self.status = normalize_convergence_status(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "convergence_status": self.status,
            "derivative_mode": self.derivative_mode,
            "steps_tested": list(self.steps_tested),
            "selected_step": self.selected_step,
            "selected_step_nm": self.selected_step,
            "max_adjacent_relative_change": self.max_adjacent_relative_change,
            "rank_range": list(self.rank_range),
            "singular_axes": list(self.singular_axes),
            "reason": self.reason,
            "convergence_reason": self.reason,
            "validation_status": self.validation_status,
            "source_contract": self.source_contract,
            "modality": self.modality,
            "scalar_key": self.scalar_key,
            "per_step_scalars": list(self.per_step_scalars),
        }

def _finite_positive_scalar(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(val) and val > 0.0:
        return val
    return None

def _max_adjacent_relative_change(values: list[Any]) -> float | None:
    finite = [_finite_positive_scalar(v) for v in values]
    finite = [v for v in finite if v is not None]
    if len(finite) < 2:
        return None
    changes: list[float] = []
    for prev, curr in zip(finite[:-1], finite[1:]):
        denom = max(abs(prev), abs(curr), _FISHER_VARIANCE_FLOOR)
        changes.append(float(abs(curr - prev) / denom))
    return float(max(changes)) if changes else None

def _parent_convergence_statuses(parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None) -> dict[str, str]:
    if not parent_result_metadata_by_modality:
        return {}
    out: dict[str, str] = {}
    for modality, meta in parent_result_metadata_by_modality.items():
        if hasattr(meta, "to_dict"):
            meta = meta.to_dict()
        if not isinstance(meta, dict):
            out[str(modality)] = ConvergenceStatus.UNCHECKED.value
            continue
        status = meta.get("convergence_status")
        if status is None and isinstance(meta.get("fisher_result"), dict):
            status = meta["fisher_result"].get("convergence_status")
        out[str(modality)] = normalize_convergence_status(status)
    return out

def _parent_validation_statuses(parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None) -> dict[str, str]:
    if not parent_result_metadata_by_modality:
        return {}
    out: dict[str, str] = {}
    for modality, meta in parent_result_metadata_by_modality.items():
        if hasattr(meta, "to_dict"):
            meta = meta.to_dict()
        if isinstance(meta, dict):
            status = meta.get("validation_status")
            if status is None and isinstance(meta.get("fisher_result"), dict):
                status = meta["fisher_result"].get("validation_status")
            out[str(modality)] = str(status or ValidationStatus.UNCHECKED.value)
        else:
            out[str(modality)] = ValidationStatus.UNCHECKED.value
    return out

def _structured_status_from_adaptive_result(
    adaptive_result: dict[str, Any],
    *,
    derivative_mode: str,
    source_contract: str,
    modality: str,
    scalar_key: str,
) -> FisherConvergenceStatus:
    per_step = list(adaptive_result.get("per_step_results", []))
    steps = tuple(float(v) for v in adaptive_result.get("candidate_steps_nm", []))
    scalars_raw = [item.get(scalar_key) for item in per_step]
    scalars: tuple[float | None, ...] = tuple(_finite_positive_scalar(v) for v in scalars_raw)
    rank_range_raw = adaptive_result.get("rank_range", [0, 0])
    try:
        rank_range = (int(rank_range_raw[0]), int(rank_range_raw[1]))
    except (TypeError, ValueError, IndexError):
        ranks = [int(item.get("fisher_rank", item.get("rank", 0))) for item in per_step]
        rank_range = (min(ranks, default=0), max(ranks, default=0))
    singular_axes: list[str] = []
    for item in per_step:
        axes = item.get("singular_axes") or item.get("axes_singular") or []
        if isinstance(axes, str):
            axes = [axes]
        for axis in axes:
            if str(axis) not in singular_axes:
                singular_axes.append(str(axis))
    status = normalize_convergence_status(adaptive_result.get("convergence_status"))
    validation = combine_parent_statuses({modality: {"convergence_status": status}})["validation_status"]
    return FisherConvergenceStatus(
        status=status,
        derivative_mode=derivative_mode,
        steps_tested=steps,
        selected_step=adaptive_result.get("selected_step_nm"),
        max_adjacent_relative_change=_max_adjacent_relative_change(scalars_raw),
        rank_range=rank_range,
        singular_axes=tuple(singular_axes),
        reason=str(adaptive_result.get("reason", "")),
        validation_status=str(validation),
        source_contract=source_contract,
        modality=str(modality),
        scalar_key=str(scalar_key),
        per_step_scalars=scalars,
    )

def convergence_status_metadata(status: FisherConvergenceStatus | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable convergence metadata record."""
    if hasattr(status, "to_dict"):
        return status.to_dict()
    if isinstance(status, dict):
        out = dict(status)
        out["convergence_status"] = normalize_convergence_status(out.get("convergence_status", out.get("status")))
        out.setdefault("status", out["convergence_status"])
        out.setdefault("validation_status", combine_parent_statuses({"_": {"convergence_status": out["convergence_status"]}})["validation_status"])
        return out
    raise TypeError(f"Unsupported convergence status payload {type(status).__name__}")

def compute_converged_lateral_crlb(
    render_at_xy: Any,
    base_x_nm: float,
    base_y_nm: float,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    *,
    step_pixels: tuple[float, ...] | list[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    max_relative_change: float = 0.05,
    min_stable_steps: int = 3,
    source_contract: str = "Contract-LP",
    modality: str = "unknown",
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """Compute convergence-gated lateral CRLB using rerendered finite differences.

    ``render_at_xy(x_nm, y_nm)`` must return a finite 2D signal image at the
    requested particle centre.  The function never falls back to stationary
    image-gradient derivatives; it constructs explicit x/y rerender pairs for
    each tested step and returns a structured ``FisherConvergenceStatus``.
    """
    from .lateral import adaptive_lateral_crlb_from_rerender_pairs

    if not callable(render_at_xy):
        raise TypeError("render_at_xy must be callable")
    px = float(pixel_size_nm)
    if not np.isfinite(px) or px <= 0.0:
        raise ValueError("pixel_size_nm must be positive and finite")
    pairs: dict[float, dict[str, np.ndarray]] = {}
    for step_px in step_pixels:
        h_nm = float(step_px) * px
        if not np.isfinite(h_nm) or h_nm <= 0.0:
            raise ValueError(f"step_pixels must be positive finite values; got {step_px!r}")
        pairs[h_nm] = {
            "x_minus": np.asarray(render_at_xy(float(base_x_nm) - h_nm, float(base_y_nm)), dtype=float),
            "x_plus": np.asarray(render_at_xy(float(base_x_nm) + h_nm, float(base_y_nm)), dtype=float),
            "y_minus": np.asarray(render_at_xy(float(base_x_nm), float(base_y_nm) - h_nm), dtype=float),
            "y_plus": np.asarray(render_at_xy(float(base_x_nm), float(base_y_nm) + h_nm), dtype=float),
        }
    adaptive = adaptive_lateral_crlb_from_rerender_pairs(
        pairs,
        noise_variance_map,
        px,
        convergence_tolerance=float(max_relative_change),
        min_stable_steps=int(min_stable_steps),
        source_contract=source_contract,
        modality=modality,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    structured = _structured_status_from_adaptive_result(
        adaptive,
        derivative_mode="rerendered_central_difference_xy",
        source_contract=source_contract,
        modality=modality,
        scalar_key="sigma_xy_nm",
    )
    final = dict(adaptive.get("final_result", {}))
    final["convergence_status_record"] = structured.to_dict()
    final["convergence_status"] = structured.status
    final["validation_status"] = structured.validation_status
    final["selected_step_nm"] = structured.selected_step
    final["steps_tested_nm"] = list(structured.steps_tested)
    final["max_adjacent_relative_change"] = structured.max_adjacent_relative_change
    final["parent_convergence_statuses"] = {modality: structured.status}
    adaptive["convergence_status_record"] = structured.to_dict()
    adaptive["final_result"] = final
    adaptive["validation_status"] = structured.validation_status
    adaptive["max_adjacent_relative_change"] = structured.max_adjacent_relative_change
    return adaptive

def compute_converged_axial_crlb(
    render_at_z: Any,
    base_z_nm: float,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    *,
    z_steps_nm: tuple[float, ...] | list[float] = (400.0, 200.0, 100.0, 50.0),
    max_relative_span: float = 0.10,
    min_stable_steps: int = 2,
    source_contract: str = "Contract-LZ",
    modality: str = "unknown",
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """Compute convergence-gated axial CRLB from explicit z-rerender stacks."""
    from .axial import adaptive_axial_crlb_from_stacks

    if not callable(render_at_z):
        raise TypeError("render_at_z must be callable")
    px = float(pixel_size_nm)
    if not np.isfinite(px) or px <= 0.0:
        raise ValueError("pixel_size_nm must be positive and finite")
    stacks: dict[float, np.ndarray] = {}
    for step_nm in z_steps_nm:
        h_nm = float(step_nm)
        if not np.isfinite(h_nm) or h_nm <= 0.0:
            raise ValueError(f"z_steps_nm must be positive finite values; got {step_nm!r}")
        z0 = float(base_z_nm)
        stacks[h_nm] = np.stack([
            np.asarray(render_at_z(z0 - h_nm), dtype=float),
            np.asarray(render_at_z(z0), dtype=float),
            np.asarray(render_at_z(z0 + h_nm), dtype=float),
        ], axis=0)
    adaptive = adaptive_axial_crlb_from_stacks(
        stacks,
        noise_variance_map,
        px,
        convergence_tolerance=float(max_relative_span),
        min_stable_steps=int(min_stable_steps),
        source_contract=source_contract,
        modality=modality,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    structured = _structured_status_from_adaptive_result(
        adaptive,
        derivative_mode="rerendered_central_difference_z_stack",
        source_contract=source_contract,
        modality=modality,
        scalar_key="sigma_xyz_nm",
    )
    final = dict(adaptive.get("final_result", {}))
    final["convergence_status_record"] = structured.to_dict()
    final["convergence_status"] = structured.status
    final["validation_status"] = structured.validation_status
    final["selected_step_nm"] = structured.selected_step
    final["steps_tested_nm"] = list(structured.steps_tested)
    final["max_adjacent_relative_change"] = structured.max_adjacent_relative_change
    final["parent_convergence_statuses"] = {modality: structured.status}
    adaptive["convergence_status_record"] = structured.to_dict()
    adaptive["final_result"] = final
    adaptive["validation_status"] = structured.validation_status
    adaptive["max_adjacent_relative_change"] = structured.max_adjacent_relative_change
    return adaptive

__all__ = [
    "FisherConvergenceStatus",
    "annotate_fisher_result_status",
    "compute_converged_lateral_crlb",
    "compute_converged_axial_crlb",
    "convergence_status_metadata",
]
