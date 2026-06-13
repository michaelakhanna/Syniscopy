"""Detected-quanta normalization, budget, and convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any
import json

import numpy as np

from experiment_contracts import (
    ConvergenceStatus,
    ValidationStatus,
    combine_parent_statuses,
    detected_quanta_contract_metadata,
    normalize_convergence_status,
)
from modality_registry import modality_uses_relative_reference_contrast

from ._constants import _FISHER_VARIANCE_FLOOR
from .candidates import FisherCandidate


@dataclass(frozen=True)
class DetectedQuantaCandidate:
    """One candidate's complete Contract-Q normalization payload."""

    key: str
    contrast: Any
    modality: str
    pixel_size_nm: float
    measurement_model: str | None = None
    detected_count_image: Any | None = None
    reference_count_image: Any | None = None
    phase_visibility: float | None = None
    phase_readout_variance: float | None = None
    parent_convergence_status: str | None = None
    derivative_context: Any | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("DetectedQuantaCandidate.key must be non-empty.")
        modality = str(self.modality).strip()
        if not modality:
            raise ValueError(
                f"DetectedQuantaCandidate[{key!r}] must declare physical modality metadata."
            )
        pixel_size = float(self.pixel_size_nm)
        if not np.isfinite(pixel_size) or pixel_size <= 0.0:
            raise ValueError(
                f"DetectedQuantaCandidate[{key!r}].pixel_size_nm must be positive and finite; "
                f"got {self.pixel_size_nm!r}."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "pixel_size_nm", pixel_size)
        if self.measurement_model is not None:
            object.__setattr__(
                self,
                "measurement_model",
                _normalise_measurement_model(str(self.measurement_model)),
            )
        if self.parent_convergence_status in {"", None}:
            object.__setattr__(self, "parent_convergence_status", None)
        elif self.parent_convergence_status is not None:
            object.__setattr__(
                self,
                "parent_convergence_status",
                normalize_convergence_status(self.parent_convergence_status),
            )


@dataclass(frozen=True)
class _NormalizedDetectedQuantaCandidate:
    """One Contract-Q candidate after budget normalization and variance resolution."""

    fisher_candidate: FisherCandidate
    measurement_model: str
    quanta_scale: float | None
    readout_variance_fraction: float
    budget_scaling_note: str
    is_count_domain: bool = False
    is_proxy_count: bool = False
    count_mean_source: str | None = None
    reference_count_mean_source: str | None = None
    derivative_input_basis: str | None = None
    pre_normalization_signal_count_sum: float | None = None
    pre_normalization_reference_count_sum: float | None = None
    budgeted_count_sum: float | None = None
    reference_budget_included: bool | None = None
    budget_normalization_basis: str | None = None
    quanta_per_pixel: float | None = None
    phase_variance: float | None = None
    phase_visibility: float | None = None
    phase_readout_limited: bool | None = None
    count_readout_limited: bool | None = None
    parent_convergence_status: str | None = None

    @property
    def key(self) -> str:
        return self.fisher_candidate.key

    @property
    def modality(self) -> str:
        return self.fisher_candidate.modality

    @property
    def measurement_domain(self) -> str:
        return self.fisher_candidate.measurement_domain

    @property
    def signal_units(self) -> str:
        return self.fisher_candidate.signal_units

    @property
    def noise_variance_units(self) -> str | None:
        return self.fisher_candidate.noise_variance_units


def _detected_quanta_candidate_list(
    candidates: Sequence[DetectedQuantaCandidate],
) -> list[DetectedQuantaCandidate]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError(
            "compare_detected_quanta_normalized_fisher_candidates requires a "
            "non-empty sequence of DetectedQuantaCandidate objects."
        )
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError(
            "compare_detected_quanta_normalized_fisher_candidates requires at least one "
            "DetectedQuantaCandidate."
        )
    invalid = [
        type(candidate).__name__
        for candidate in candidate_list
        if not isinstance(candidate, DetectedQuantaCandidate)
    ]
    if invalid:
        raise ValueError(
            "compare_detected_quanta_normalized_fisher_candidates accepts only "
            f"DetectedQuantaCandidate objects; got {invalid!r}."
        )
    keys = [candidate.key for candidate in candidate_list]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"DetectedQuantaCandidate keys must be unique; duplicates: {duplicates!r}."
        )
    return candidate_list


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

def _uses_relative_reference_contrast_for_contract_q(modality: str) -> bool:
    try:
        return bool(modality_uses_relative_reference_contrast(modality))
    except Exception:
        return False

def _validated_reference_count_image(
    reference_count_image: Any | None,
    candidate: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if reference_count_image is None:
        return None
    reference_image = np.asarray(reference_count_image, dtype=float)
    if reference_image.shape != expected_shape:
        raise ValueError(
            "DetectedQuantaCandidate[%r].reference_count_image has shape %s; expected %s."
            % (candidate, reference_image.shape, expected_shape)
        )
    if not np.all(np.isfinite(reference_image)) or np.any(reference_image < 0.0):
        raise ValueError(
            "DetectedQuantaCandidate[%r].reference_count_image must contain finite "
            "non-negative detector counts." % candidate
        )
    return reference_image


def _central_plane_or_image(values: np.ndarray, *, z_step_nm: float | None, key: str, kind: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if z_step_nm is None:
        if arr.ndim != 2:
            raise ValueError(
                f"2D mode expects (H, W) {kind} for candidate {key!r}; got shape {arr.shape}."
            )
        return arr
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            f"3D mode expects (3, H, W) {kind} for candidate {key!r}; got shape {arr.shape}."
        )
    return arr[1]


def _candidate_map(
    candidates: Sequence[_NormalizedDetectedQuantaCandidate],
    attr: str,
    *,
    omit_none: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for candidate in candidates:
        value = getattr(candidate, attr)
        if omit_none and value is None:
            continue
        out[candidate.key] = value
    return out


def _normalize_detected_quanta_candidate(
    candidate_spec: DetectedQuantaCandidate,
    *,
    quanta_budget: float,
    readout_variance: float,
    z_step_nm: float | None,
    derivative_target: str,
) -> _NormalizedDetectedQuantaCandidate:
    candidate = candidate_spec.key
    modality = candidate_spec.modality
    c = np.asarray(candidate_spec.contrast, dtype=float)
    if not np.all(np.isfinite(c)):
        raise ValueError(f"contrast image for candidate {candidate!r} must contain only finite values.")
    central = _central_plane_or_image(c, z_step_nm=z_step_nm, key=candidate, kind="contrast")

    model = (
        candidate_spec.measurement_model
        if candidate_spec.measurement_model is not None
        else _default_measurement_model_for_modality(modality)
    )
    model = _normalise_measurement_model(model)

    parent_convergence_status = (
        None
        if candidate_spec.parent_convergence_status is None
        else normalize_convergence_status(candidate_spec.parent_convergence_status)
    )
    parent_result_metadata = (
        {}
        if parent_convergence_status is None
        else {"convergence_status": parent_convergence_status}
    )

    if model == "count":
        modality_key = str(modality).lower()
        if modality_key.startswith(("tem", "sem")) or "electron" in modality_key:
            measurement_domain = "electron_count"
            signal_units = "electron_count"
            noise_variance_units = "electron_count_squared"
        else:
            measurement_domain = "detected_quanta"
            signal_units = "detected_quanta"
            noise_variance_units = "detected_quanta_squared"

        if candidate_spec.detected_count_image is None:
            raise ValueError(
                "DetectedQuantaCandidate.detected_count_image must include the real detector-count "
                f"mean image for count-domain candidate {candidate!r}. Signed contrast "
                "alone is not a valid Poisson mean for detected-quanta normalization."
            )
        count_image = np.asarray(candidate_spec.detected_count_image, dtype=float)
        if count_image.shape != c.shape:
            raise ValueError(
                "DetectedQuantaCandidate[%r].detected_count_image has shape %s; expected %s."
                % (candidate, count_image.shape, c.shape)
            )
        if not np.all(np.isfinite(count_image)):
            raise ValueError(
                "DetectedQuantaCandidate[%r].detected_count_image must contain only finite values."
                % candidate
            )

        central_count = _central_plane_or_image(
            count_image,
            z_step_nm=z_step_nm,
            key=candidate,
            kind="detected_count_image",
        )
        central_count = np.where(np.isfinite(central_count), central_count, 0.0)
        central_count = np.maximum(central_count, 0.0)

        total_signal = float(np.sum(central_count))
        scale = 0.0 if total_signal <= 0.0 else float(quanta_budget) / total_signal
        reference_budget_included = False
        budget_normalization_basis = "signal_count_image"
        pre_normalization_reference_count_sum = None
        reference_count_mean_source = None
        mean_quanta = scale * central_count
        budgeted_count_sum = float(np.sum(mean_quanta))

        if derivative_target == "count_mean_derivative":
            rescaled_c = scale * count_image
            var = mean_quanta + float(readout_variance)
            derivative_input_basis = "count_mean"
            mean_signal = float(np.mean(mean_quanta)) if mean_quanta.size else 0.0
            denom = mean_signal + float(readout_variance)
            readout_fraction = float(readout_variance) / denom if denom > 0.0 else 0.0
        elif _uses_relative_reference_contrast_for_contract_q(modality):
            reference_image = _validated_reference_count_image(
                candidate_spec.reference_count_image,
                candidate,
                c.shape,
            )
            if reference_image is None:
                raise ValueError(
                    "relative-reference detected-quanta normalization for "
                    f"candidate {candidate!r} requires DetectedQuantaCandidate.reference_count_image "
                    "because the analysis contrast is (signal-reference)/reference."
                )
            central_reference = _central_plane_or_image(
                reference_image,
                z_step_nm=z_step_nm,
                key=candidate,
                kind="reference_count_image",
            )
            total_reference = float(np.sum(central_reference))
            total_observation_counts = total_signal + total_reference
            scale = (
                0.0
                if total_observation_counts <= 0.0
                else float(quanta_budget) / total_observation_counts
            )
            reference_budget_included = True
            budget_normalization_basis = "signal_plus_reference_count_images"
            mean_quanta = scale * central_count
            scaled_reference = scale * central_reference
            budgeted_count_sum = float(np.sum(mean_quanta) + np.sum(scaled_reference))
            var_signal = mean_quanta + float(readout_variance)
            var_reference = scaled_reference + float(readout_variance)
            ref_safe = np.maximum(np.abs(scaled_reference), 1e-12)
            var = var_signal / (ref_safe ** 2) + (
                (mean_quanta ** 2) * var_reference / (ref_safe ** 4)
            )
            if readout_variance > 0.0:
                readout_component = float(readout_variance) / (ref_safe ** 2) + (
                    (mean_quanta ** 2) * float(readout_variance) / (ref_safe ** 4)
                )
                total_var_mean = float(np.mean(var)) if var.size else 0.0
                readout_fraction = (
                    float(np.mean(readout_component)) / total_var_mean
                    if total_var_mean > 0.0
                    else 0.0
                )
            else:
                readout_fraction = 0.0
            rescaled_c = c
            measurement_domain = "contrast"
            signal_units = "relative_reference"
            noise_variance_units = "relative_reference_squared"
            reference_count_mean_source = "reference_count_image"
            derivative_input_basis = "relative_reference_contrast"
            pre_normalization_reference_count_sum = total_reference
        else:
            reference_image = _validated_reference_count_image(
                candidate_spec.reference_count_image,
                candidate,
                c.shape,
            )
            if reference_image is not None:
                central_reference = _central_plane_or_image(
                    reference_image,
                    z_step_nm=z_step_nm,
                    key=candidate,
                    kind="reference_count_image",
                )
                total_reference = float(np.sum(central_reference))
                total_observation_counts = total_signal + total_reference
                scale = (
                    0.0
                    if total_observation_counts <= 0.0
                    else float(quanta_budget) / total_observation_counts
                )
                reference_budget_included = True
                budget_normalization_basis = "signal_plus_reference_count_images"
                mean_quanta = scale * central_count
                scaled_reference = scale * central_reference
                budgeted_count_sum = float(np.sum(mean_quanta) + np.sum(scaled_reference))
                rescaled_c = scale * c
                var = mean_quanta + scaled_reference + 2.0 * float(readout_variance)
                derivative_input_basis = "additive_reference_count_contrast"
                reference_count_mean_source = "reference_count_image"
                mean_signal = float(np.mean(mean_quanta)) if mean_quanta.size else 0.0
                mean_reference = float(np.mean(scaled_reference)) if scaled_reference.size else 0.0
                denom = mean_signal + mean_reference + 2.0 * float(readout_variance)
                readout_fraction = (
                    (2.0 * float(readout_variance)) / denom
                    if denom > 0.0
                    else 0.0
                )
                pre_normalization_reference_count_sum = total_reference
            else:
                rescaled_c = scale * c
                var = mean_quanta + float(readout_variance)
                derivative_input_basis = "additive_count_contrast"
                mean_signal = float(np.mean(mean_quanta)) if mean_quanta.size else 0.0
                denom = mean_signal + float(readout_variance)
                readout_fraction = float(readout_variance) / denom if denom > 0.0 else 0.0

        budget_scaling_note = (
            "relative-reference contrast keeps dimensionless signal units; "
            "signal/reference count means are budget-scaled for variance propagation"
            if _uses_relative_reference_contrast_for_contract_q(modality)
            and derivative_target == "signed_contrast_scaled"
            else "count-domain ideal F is proportional to N scaling is exact only when additive readout variance is negligible"
            if readout_variance > 0.0
            else "count-domain ideal F is proportional to N scaling"
        )

        fisher_candidate = FisherCandidate(
            key=candidate_spec.key,
            signal=rescaled_c,
            noise_variance=np.maximum(var, _FISHER_VARIANCE_FLOOR),
            modality=candidate_spec.modality,
            pixel_size_nm=candidate_spec.pixel_size_nm,
            measurement_domain=measurement_domain,
            signal_units=signal_units,
            noise_variance_units=noise_variance_units,
            parent_result_metadata=parent_result_metadata,
            noise_covariance_kind="independent_pixels",
            derivative_context=candidate_spec.derivative_context,
        )
        return _NormalizedDetectedQuantaCandidate(
            fisher_candidate=fisher_candidate,
            measurement_model=model,
            quanta_scale=scale,
            readout_variance_fraction=readout_fraction,
            budget_scaling_note=budget_scaling_note,
            is_count_domain=True,
            count_mean_source="detected_count_image",
            reference_count_mean_source=reference_count_mean_source,
            derivative_input_basis=derivative_input_basis,
            pre_normalization_signal_count_sum=total_signal,
            pre_normalization_reference_count_sum=pre_normalization_reference_count_sum,
            budgeted_count_sum=budgeted_count_sum,
            reference_budget_included=reference_budget_included,
            budget_normalization_basis=budget_normalization_basis,
            count_readout_limited=bool(readout_variance > 0.0),
            parent_convergence_status=parent_convergence_status,
        )

    if model != "phase":
        raise AssertionError(f"Unhandled measurement model {model!r}.")

    visibility = float(1.0 if candidate_spec.phase_visibility is None else candidate_spec.phase_visibility)
    phase_readout_variance = float(
        0.0
        if candidate_spec.phase_readout_variance is None
        else candidate_spec.phase_readout_variance
    )
    if not np.isfinite(visibility) or visibility <= 0.0:
        raise ValueError(
            f"phase visibility for candidate {candidate!r} must be positive and finite; got {visibility!r}."
        )
    if not np.isfinite(phase_readout_variance) or phase_readout_variance < 0.0:
        raise ValueError(
            f"phase readout variance for candidate {candidate!r} must be non-negative and finite; "
            f"got {phase_readout_variance!r}."
        )

    quanta_per_pixel = float(quanta_budget) / float(central.size)
    phase_variance = 1.0 / (visibility * visibility * quanta_per_pixel) + phase_readout_variance
    readout_fraction = (
        phase_readout_variance / phase_variance
        if phase_variance > 0.0
        else 0.0
    )
    var = np.full(central.shape, phase_variance, dtype=float)
    budget_scaling_note = (
        "phase-domain exact quanta scaling is broken by additive phase readout variance"
        if phase_readout_variance > 0.0
        else "phase-domain shot-noise scaling with var(phi)=1/(V squared n_Q)"
    )

    fisher_candidate = FisherCandidate(
        key=candidate_spec.key,
        signal=c,
        noise_variance=np.maximum(var, _FISHER_VARIANCE_FLOOR),
        modality=candidate_spec.modality,
        pixel_size_nm=candidate_spec.pixel_size_nm,
        measurement_domain="phase",
        signal_units="radian",
        noise_variance_units="radian_squared",
        parent_result_metadata=parent_result_metadata,
        noise_covariance_kind="independent_pixels",
        derivative_context=candidate_spec.derivative_context,
    )
    return _NormalizedDetectedQuantaCandidate(
        fisher_candidate=fisher_candidate,
        measurement_model=model,
        quanta_scale=None,
        readout_variance_fraction=float(readout_fraction),
        budget_scaling_note=budget_scaling_note,
        quanta_per_pixel=quanta_per_pixel,
        phase_variance=float(phase_variance),
        phase_visibility=float(visibility),
        phase_readout_limited=bool(phase_readout_variance > 0.0),
        parent_convergence_status=parent_convergence_status,
    )


def compute_quanta_scaling_law(
    fisher_at_budget: np.ndarray,
    budget: float,
    target_budgets: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Scale a Fisher matrix under the ideal detected-quanta law F ∝ N_Q."""
    from .fusion import _sigma_xy_from_fisher

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

def check_budget_ordering_invariance(
    results_by_budget: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Check whether candidate ordering is invariant across budget-normalized runs."""
    if not isinstance(results_by_budget, dict) or len(results_by_budget) < 2:
        raise ValueError("results_by_budget must contain at least two budget results.")

    ordering_by_budget: dict[float, list[str]] = {}
    readout_limited: set[str] = set()
    for budget in sorted(float(b) for b in results_by_budget):
        result = results_by_budget[budget]
        ordering = result.get("ordering_xy")
        if ordering is None:
            raise ValueError("Each result must contain ordering_xy.")
        ordering_by_budget[budget] = [str(item[0]) for item in ordering]
        for key in ("phase_readout_limited", "count_readout_limited"):
            for candidate, limited in (result.get(key, {}) or {}).items():
                if limited:
                    readout_limited.add(str(candidate))

    orders = list(ordering_by_budget.values())
    invariant = all(order == orders[0] for order in orders[1:])
    notes: list[str] = []
    if invariant:
        notes.append("candidate ordering is invariant across supplied budgets")
    else:
        notes.append("candidate ordering changes across supplied budgets")
    if readout_limited:
        notes.append("readout-limited candidates can break ideal quanta scaling")

    return {
        "ordering_invariant": bool(invariant),
        "ordering_by_budget": ordering_by_budget,
        "readout_limited_candidates": sorted(readout_limited),
        "invariance_notes": notes,
    }

def _infer_contract_q_derivative_status(
    parent_convergence_statuses: Any | None,
) -> str:
    parent_map = {} if parent_convergence_statuses is None else dict(parent_convergence_statuses)
    if not parent_map:
        return ConvergenceStatus.UNCHECKED.value

    statuses = [
        normalize_convergence_status(status)
        for status in parent_map.values()
        if str(status).strip()
    ]
    if not statuses:
        return ConvergenceStatus.UNCHECKED.value

    if any(
        status in {
            ConvergenceStatus.FAILED_CONVERGENCE.value,
            ConvergenceStatus.ILL_CONDITIONED.value,
            ConvergenceStatus.NONFINITE.value,
        }
        for status in statuses
    ):
        return ConvergenceStatus.FAILED_CONVERGENCE.value
    if all(status == ConvergenceStatus.FINITE_CONVERGED.value for status in statuses):
        return ConvergenceStatus.FINITE_CONVERGED.value
    if all(
        status
        in {
            ConvergenceStatus.FINITE_CONVERGED.value,
            ConvergenceStatus.STABLE_SINGULAR.value,
        }
        for status in statuses
    ):
        return ConvergenceStatus.STABLE_SINGULAR.value
    return ConvergenceStatus.UNCHECKED.value

def compare_detected_quanta_normalized_fisher_candidates(
    candidates: Sequence[DetectedQuantaCandidate],
    quanta_budget: float,
    *,
    readout_variance: float = 0.0,
    z_step_nm: float | None = None,
    detected_quanta_derivative_target: str = "signed_contrast_scaled",
) -> dict[str, Any]:
    r"""
    Detected-quanta-budget-normalized cross-candidate CRLB comparator.

    Normalizes candidate signals into a common detected-quanta budget, then
    delegates candidate ranking to ``compare_fisher_candidates``. Count-domain
    candidates should be
    supplied as detector-domain count images after the imaging model's
    model-output detector-frame conversion on each ``DetectedQuantaCandidate``.
    Those count images are rescaled to the same total detected-quanta budget and
    used only as the Poisson mean/noise floor; the Fisher derivative target
    remains the signed per-particle contrast image. Phase-domain candidates (for
    example QPI) keep the phase signal in radians and use a shot-noise
    phase-variance model set by the same detected-quanta budget.

    Parameters
    ----------
    candidates : sequence of DetectedQuantaCandidate
        Candidate objects carrying the per-particle contrast image, modality,
        pixel geometry, measurement model, detector-count mean images, optional
        reference-count images, parent convergence status, and derivative
        provenance needed by Contract-Q.
    quanta_budget : float
        Total detected quanta per frame (per candidate, per particle). For
        optical count modes this means photons; for electron modalities this
        means detected electrons or dose quanta. The same value is used for
        every candidate so the comparison is budget-fair by construction.
    DetectedQuantaCandidate.detected_count_image
        Required count-domain mean image, already in detector-count units before
        budget rescaling. The supplied count image sets the Poisson mean and
        exposure scale only; it is not substituted for the derivative image.
        Count-domain detected-quanta comparisons cannot be computed from signed
        contrast alone because that would fabricate a Poisson mean and return
        proxy-dependent Fisher values.
    DetectedQuantaCandidate.reference_count_image
        Required for relative-reference count-domain candidates when
        ``detected_quanta_derivative_target='signed_contrast_scaled'``. The
        analysis contrast is dimensionless ``(signal-reference)/reference``;
        this reference-count image is needed to propagate count shot/read noise
        into relative-reference variance under the common detected-quanta
        budget. Because both signal and reference frames are stochastic in
        this branch, their pre-normalization count sums are jointly charged to
        the declared total budget.
    readout_variance : float, default 0.0
        Additive Gaussian readout variance, in count-quanta squared, added
        to count-domain candidates after shot noise.
    DetectedQuantaCandidate.phase_visibility
        Interferometric / demodulation visibility factor for phase-domain
        candidates. The default is ``1.0``. The phase-noise convention is
        ``var(phi) = 1 / (visibility^2 * quanta_per_pixel)`` plus optional
        phase readout variance.
    DetectedQuantaCandidate.phase_readout_variance
        Additive phase-readout variance in radians squared for phase-domain
        candidates. The default is ``0.0``.
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
        Same keys as the generic Fisher candidate comparator, with
        additional metadata recording the normalization parameters,
        measurement models, count-domain scale factors, and phase-domain
        variance terms.

    Notes
    -----
    For count-domain candidates, the detector-domain count image is the
    Poisson mean and sets the exposure/dose scale. The signed per-particle
    contrast image remains the derivative target after that same exposure
    scaling, so background slope in a supplied count image cannot by itself
    create localization information for a zero-contrast particle. For
    phase-domain candidates, the phase image is already the detector-domain
    signal in radians, so the signal amplitude is not normalized away; the
    budget controls phase readout variance instead. This is the phase-domain
    analogue of photon normalization and avoids treating radians as counts.
    """
    from .comparison import (
        COMPARISON_TARGET_LATERAL_XY,
        COMPARISON_TARGET_LOCALIZATION_XYZ,
        compare_fisher_candidates,
    )

    candidate_specs = _detected_quanta_candidate_list(candidates)
    if not np.isfinite(quanta_budget) or quanta_budget <= 0.0:
        raise ValueError(
            f"quanta_budget must be a positive finite scalar; got {quanta_budget!r}."
        )
    if not np.isfinite(readout_variance) or readout_variance < 0.0:
        raise ValueError(
            f"readout_variance must be a non-negative finite scalar; got {readout_variance!r}."
        )
    derivative_target = str(detected_quanta_derivative_target).strip().lower()
    if derivative_target not in {"signed_contrast_scaled", "count_mean_derivative"}:
        raise ValueError(
            "detected_quanta_derivative_target must be 'signed_contrast_scaled' "
            f"or 'count_mean_derivative'; got {detected_quanta_derivative_target!r}."
        )

    normalized_candidates = [
        _normalize_detected_quanta_candidate(
            candidate_spec,
            quanta_budget=float(quanta_budget),
            readout_variance=float(readout_variance),
            z_step_nm=z_step_nm,
            derivative_target=derivative_target,
        )
        for candidate_spec in candidate_specs
    ]
    fisher_candidates = [candidate.fisher_candidate for candidate in normalized_candidates]
    parent_status_map = _candidate_map(
        normalized_candidates,
        "parent_convergence_status",
    )
    result = compare_fisher_candidates(
        fisher_candidates,
        target=(
            COMPARISON_TARGET_LOCALIZATION_XYZ
            if z_step_nm is not None
            else COMPARISON_TARGET_LATERAL_XY
        ),
        z_step_nm=z_step_nm,
    )

    result["quanta_budget"] = float(quanta_budget)
    result["readout_variance"] = float(readout_variance)
    result["normalization"] = "detected_quanta_domain_aware"
    result["detected_quanta_derivative_target"] = derivative_target
    readout_variance_fraction_map = _candidate_map(
        normalized_candidates,
        "readout_variance_fraction",
        omit_none=False,
    )
    record_by_key = {
        record["candidate_key"]: record
        for record in result.get("candidate_records", [])
        if "candidate_key" in record
    }
    for candidate in normalized_candidates:
        record = record_by_key.setdefault(candidate.key, {"candidate_key": candidate.key})
        record.update(
            {
                "modality": candidate.modality,
                "measurement_model": candidate.measurement_model,
                "measurement_domain": candidate.measurement_domain,
                "signal_units": candidate.signal_units,
                "noise_variance_units": candidate.noise_variance_units,
                "quanta_scale": candidate.quanta_scale,
                "quanta_per_pixel": candidate.quanta_per_pixel,
                "phase_variance": candidate.phase_variance,
                "phase_visibility": candidate.phase_visibility,
                "phase_readout_limited": candidate.phase_readout_limited,
                "count_readout_limited": candidate.count_readout_limited,
                "readout_variance_fraction": candidate.readout_variance_fraction,
                "budget_scaling_note": candidate.budget_scaling_note,
                "count_mean_source": candidate.count_mean_source,
                "reference_count_mean_source": candidate.reference_count_mean_source,
                "derivative_input_basis": candidate.derivative_input_basis,
                "pre_normalization_signal_count_sum": (
                    candidate.pre_normalization_signal_count_sum
                ),
                "pre_normalization_reference_count_sum": (
                    candidate.pre_normalization_reference_count_sum
                ),
                "budgeted_count_sum": candidate.budgeted_count_sum,
                "reference_budget_included": candidate.reference_budget_included,
                "budget_normalization_basis": candidate.budget_normalization_basis,
            }
        )
    result["candidate_records"] = [record_by_key[candidate.key] for candidate in normalized_candidates]
    count_domain_candidates = [
        candidate.key for candidate in normalized_candidates if candidate.is_count_domain
    ]
    proxy_count_candidates = [
        candidate.key for candidate in normalized_candidates if candidate.is_proxy_count
    ]
    all_count_domain_candidates_have_detected_count_images = all(
        candidate.count_mean_source == "detected_count_image"
        for candidate in normalized_candidates
        if candidate.is_count_domain
    )
    strict_detected_count_basis = bool(all_count_domain_candidates_have_detected_count_images)
    if not count_domain_candidates:
        distribution_rule = "phase_domain_quanta_variance"
    elif strict_detected_count_basis:
        distribution_rule = "profile_specific_detected_count_image"
    else:
        distribution_rule = "mixed_detected_count_and_contrast_proxy_diagnostic"
    result["detected_quanta_contract"] = detected_quanta_contract_metadata(
        total_detected_quanta_budget=float(quanta_budget),
        distribution_rule=distribution_rule,
        normalization_domain="central_plane" if z_step_nm is not None else "2d_image",
        support_mask_used=False,
        candidate_readout_variance_fraction=readout_variance_fraction_map,
    )
    result["detected_quanta_contract"]["derivative_target"] = derivative_target
    result["detected_quanta_contract"]["safe_for_detected_quanta_ranking"] = bool(strict_detected_count_basis)
    result["count_domain_candidates"] = count_domain_candidates
    result["proxy_count_candidates"] = proxy_count_candidates
    result["all_count_domain_candidates_have_detected_count_images"] = bool(
        all_count_domain_candidates_have_detected_count_images
    )
    result["contract_q_proxy_diagnostic"] = bool(proxy_count_candidates)
    result["source_contract"] = "Contract-Q"
    result["contract_q_derivative_convergence_status"] = _infer_contract_q_derivative_status(
        parent_status_map
    )
    result["parent_convergence_statuses"] = parent_status_map
    if parent_status_map:
        contract_q_parent_meta = {
            candidate: {"convergence_status": status}
            for candidate, status in parent_status_map.items()
        }
        parent_meta = combine_parent_statuses(contract_q_parent_meta)
        result["parent_status_metadata"] = parent_meta
        result["validation_status"] = parent_meta["validation_status"]
        result["safe_for_detected_quanta_ranking"] = (
            bool(parent_meta["safe_for_detected_quanta_ranking"])
            and bool(strict_detected_count_basis)
        )
    else:
        result["validation_status"] = ValidationStatus.UNCHECKED.value
        result["safe_for_detected_quanta_ranking"] = False
    if not strict_detected_count_basis:
        result["validation_status"] = ValidationStatus.DIAGNOSTIC_ONLY.value
        result["safe_for_detected_quanta_ranking"] = False
        result["safe_for_ordering"] = False
    result["contract_q_derivative_convergence_record"] = {
        "source_contract": "Contract-Q",
        "derivative_target": derivative_target,
        "convergence_status": result["contract_q_derivative_convergence_status"],
        "parent_convergence_statuses": parent_status_map,
        "validation_status": result["validation_status"],
    }
    note = (
        "Contract-Q derivative convergence status was inherited from parent lateral statuses."
        if result["contract_q_derivative_convergence_status"] != ConvergenceStatus.UNCHECKED.value
        else "Contract-Q derivative convergence was not inferable; parent lateral statuses were not supplied."
    )
    result["contract_q_status_note"] = f"{note} Contract-Q normalization metadata is emitted here."
    return result

def build_detected_quanta_derivative_convergence_rows(
    contract_q_result: dict[str, Any],
    candidate_parent_convergence_results: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build CSV-ready Contract-Q derivative convergence rows.

    Rows are intentionally source-contract level: the actual writing location is
    decided by notebooks/scripts, but the schema is owned by core.
    """
    from .convergence import FisherConvergenceStatus

    candidate_records = [
        record for record in contract_q_result.get("candidate_records", [])
        if isinstance(record, dict) and record.get("candidate_key")
    ]
    record_by_key = {str(record["candidate_key"]): record for record in candidate_records}
    candidates = sorted(record_by_key)
    if not candidates and candidate_parent_convergence_results:
        candidates = sorted(candidate_parent_convergence_results.keys())
    parent_statuses = dict(contract_q_result.get("parent_convergence_statuses", {}))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_record = record_by_key.get(candidate, {})
        parent = (candidate_parent_convergence_results or {}).get(candidate, {})
        record = parent.get("convergence_status_record", parent)
        if isinstance(record, FisherConvergenceStatus):
            record = record.to_dict()
        if not isinstance(record, dict):
            record = {}
        status = normalize_convergence_status(
            record.get("convergence_status", record.get("status", parent_statuses.get(candidate, contract_q_result.get("contract_q_derivative_convergence_status"))))
        )
        rows.append({
            "candidate": candidate,
            "modality": candidate_record.get("modality", ""),
            "source_contract": "Contract-Q",
            "derivative_target": contract_q_result.get("detected_quanta_derivative_target", "unknown"),
            "measurement_model": candidate_record.get("measurement_model", "unknown"),
            "count_mean_source": candidate_record.get("count_mean_source", ""),
            "proxy_count_diagnostic": bool(candidate in contract_q_result.get("proxy_count_candidates", [])),
            "convergence_status": status,
            "validation_status": combine_parent_statuses({candidate: {"convergence_status": status}})["validation_status"],
            "selected_step": record.get("selected_step", record.get("selected_step_nm")),
            "steps_tested": json.dumps(record.get("steps_tested", record.get("candidate_steps_nm", []))),
            "max_adjacent_relative_change": record.get("max_adjacent_relative_change"),
            "rank_range": json.dumps(record.get("rank_range", [])),
            "reason": record.get("reason", record.get("convergence_reason", contract_q_result.get("contract_q_status_note", ""))),
        })
    return rows

def write_detected_quanta_derivative_convergence_csv(
    path: str | Any,
    contract_q_result: dict[str, Any],
    candidate_parent_convergence_results: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Write Contract-Q derivative convergence rows to CSV and return them."""
    rows = build_detected_quanta_derivative_convergence_rows(
        contract_q_result,
        candidate_parent_convergence_results,
    )
    fieldnames = [
        "candidate", "modality", "source_contract", "derivative_target", "measurement_model",
        "count_mean_source", "proxy_count_diagnostic",
        "convergence_status", "validation_status", "selected_step", "steps_tested",
        "max_adjacent_relative_change", "rank_range", "reason",
    ]
    import csv as _csv
    from pathlib import Path as _Path
    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return rows

__all__ = [
    "DetectedQuantaCandidate",
    "compute_quanta_scaling_law",
    "check_budget_ordering_invariance",
    "compare_detected_quanta_normalized_fisher_candidates",
    "build_detected_quanta_derivative_convergence_rows",
    "write_detected_quanta_derivative_convergence_csv",
]
