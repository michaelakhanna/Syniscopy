"""Detected-quanta normalization, budget, and convergence diagnostics."""

from __future__ import annotations

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

from ._constants import _FISHER_VARIANCE_FLOOR

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

def _mapping_or_scalar_value(
    value: float | dict[str, float] | None,
    modality: str,
    default: float,
) -> float:
    if value is None:
        return float(default)
    if isinstance(value, dict):
        return float(value.get(modality, default))
    return float(value)

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

def check_budget_ranking_invariance(
    results_by_budget: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Check whether modality ordering is invariant across budget-normalized runs."""
    if not isinstance(results_by_budget, dict) or len(results_by_budget) < 2:
        raise ValueError("results_by_budget must contain at least two budget results.")

    ordering_by_budget: dict[float, list[str]] = {}
    readout_limited: set[str] = set()
    for budget in sorted(float(b) for b in results_by_budget):
        result = results_by_budget[budget]
        ordering = result.get("ordering_xy", result.get("ranking_xy"))
        if ordering is None:
            raise ValueError("Each result must contain ordering_xy or ranking_xy.")
        ordering_by_budget[budget] = [str(item[0]) for item in ordering]
        for key in ("phase_readout_limited", "count_readout_limited"):
            for modality, limited in (result.get(key, {}) or {}).items():
                if limited:
                    readout_limited.add(str(modality))

    orders = list(ordering_by_budget.values())
    invariant = all(order == orders[0] for order in orders[1:])
    notes: list[str] = []
    if invariant:
        notes.append("modality ordering is invariant across supplied budgets")
    else:
        notes.append("modality ordering changes across supplied budgets")
    if readout_limited:
        notes.append("readout-limited modalities can break ideal quanta scaling")

    return {
        "ordering_invariant": bool(invariant),
        "ordering_by_budget": ordering_by_budget,
        "ranking_invariant": bool(invariant),
        "ranking_by_budget": ordering_by_budget,
        "readout_limited_modalities": sorted(readout_limited),
        "invariance_notes": notes,
    }

def check_budget_ordering_invariance(
    results_by_budget: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Preferred-name wrapper for :func:`check_budget_ranking_invariance`."""
    return check_budget_ranking_invariance(results_by_budget)

def _infer_contract_q_derivative_status(
    parent_convergence_status_by_modality: Any | None,
) -> str:
    parent_map = {} if parent_convergence_status_by_modality is None else dict(parent_convergence_status_by_modality)
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

def compare_modality_information_content_detected_quanta_normalized(
    contrast_by_modality: dict[str, np.ndarray],
    quanta_budget: float,
    pixel_size_nm: float | dict[str, float],
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    measurement_model_by_modality: dict[str, str] | None = None,
    detected_count_image_by_modality: dict[str, np.ndarray] | None = None,
    require_detected_count_images: bool = False,
    readout_variance: float = 0.0,
    phase_visibility_by_modality: float | dict[str, float] | None = None,
    phase_readout_variance_by_modality: float | dict[str, float] | None = None,
    z_step_nm: float | None = None,
    parent_convergence_status_by_modality: dict[str, str] | None = None,
    detected_quanta_derivative_target: str = "signed_contrast_scaled",
) -> dict[str, Any]:
    r"""
    Detected-quanta-budget-normalized cross-modality CRLB comparator.

    Wraps :func:`compare_modality_information_content` with a per-modality
    measurement-domain normalization. Count-domain modalities should be
    supplied as detector-domain count images after the imaging model's
    ``scale_intensity_to_counts`` step, or through
    ``detected_count_image_by_modality`` when ``contrast_by_modality`` carries
    a separate signed derivative image. Those count images are rescaled to the
    same total detected-quanta budget and used only as the Poisson mean/noise
    floor; the Fisher derivative target remains the signed per-particle
    contrast image. Phase-domain modalities (currently QPI) keep the phase
    signal in radians and use a shot-noise phase-variance model set by the same
    detected-quanta budget.

    Parameters
    ----------
    contrast_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> per-particle contrast image``. For the 2D
        comparison each value is a ``(H, W)`` array; for the 3D comparison
        (``z_step_nm`` supplied) each value is a ``(3, H, W)`` three-plane
        stack in the same convention as :func:`compute_localization_crlb_3d`.
    quanta_budget : float
        Total detected quanta per frame (per modality, per particle). For
        optical count modes this means photons; for electron modalities this
        means detected electrons or dose quanta. The same value is used for
        every modality so the comparison is budget-fair by construction.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies one pitch per modality.
    measurement_model_by_modality : dict[str, str] or None
        Optional mapping from modality name to measurement model. Accepted
        models are ``"count"`` and ``"phase"``. If omitted, QPI-like names
        (``"quantitative_phase"``, ``"qpi"``) default to ``"phase"`` and all
        others default to ``"count"``.
    detected_count_image_by_modality : dict[str, ndarray] or None
        Optional count-domain mean image for each modality, already in
        detector-count units before budget rescaling. Calibrated count-domain
        runs should pass true detector-domain count images. The supplied count
        image sets the Poisson mean and exposure scale only; it is not
        substituted for the derivative image. If this mapping is omitted, the
        function builds a non-negative diagnostic proxy from
        ``contrast_by_modality[modality]`` for exploratory comparisons.
    require_detected_count_images : bool, default False
        If True, every count-domain modality must be present in
        ``detected_count_image_by_modality``. This disables the exploratory
        contrast-proxy fallback and is intended for paper/release generation.
    readout_variance : float, default 0.0
        Additive Gaussian readout variance, in count-quanta squared, added
        to count-domain modalities after shot noise.
    phase_visibility_by_modality : float or dict, optional
        Interferometric / demodulation visibility factor for phase-domain
        modalities. The default is ``1.0``. The phase-noise convention is
        ``var(phi) = 1 / (visibility^2 * quanta_per_pixel)`` plus optional
        phase readout variance.
    phase_readout_variance_by_modality : float or dict, optional
        Additive phase-readout variance in radians squared for phase-domain
        modalities. The default is ``0.0``.
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
        Same keys as :func:`compare_modality_information_content`, with
        additional metadata recording the normalization parameters,
        measurement models, count-domain scale factors, and phase-domain
        variance terms.

    Notes
    -----
    For count-domain modalities, the detector-domain count image is the
    Poisson mean and sets the exposure/dose scale. The signed per-particle
    contrast image remains the derivative target after that same exposure
    scaling, so background slope in a supplied count image cannot by itself
    create localization information for a zero-contrast particle. For
    phase-domain modalities, the phase image is already the detector-domain
    signal in radians, so the signal amplitude is not normalized away; the
    budget controls phase readout variance instead. This is the phase-domain
    analogue of photon normalization and avoids treating radians as counts.
    """
    from .lateral import compare_modality_information_content

    if not isinstance(contrast_by_modality, dict) or not contrast_by_modality:
        raise ValueError(
            "contrast_by_modality must be a non-empty dict keyed by modality name."
        )
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

    rescaled_contrast: dict[str, np.ndarray] = {}
    rescaled_noise: dict[str, np.ndarray] = {}
    quanta_scale_by_modality: dict[str, float | None] = {}
    measurement_model_record: dict[str, str] = {}
    measurement_domain_record_units: dict[str, str] = {}
    signal_units_record: dict[str, str] = {}
    noise_variance_units_record: dict[str, str] = {}
    quanta_per_pixel_by_modality: dict[str, float] = {}
    phase_variance_by_modality: dict[str, float] = {}
    phase_visibility_record: dict[str, float] = {}
    phase_readout_limited: dict[str, bool] = {}
    count_readout_limited: dict[str, bool] = {}
    readout_variance_fraction_by_modality: dict[str, float] = {}
    budget_scaling_notes: dict[str, str] = {}
    count_domain_modalities: list[str] = []
    proxy_count_modalities: list[str] = []
    count_mean_source_by_modality: dict[str, str] = {}

    for modality, contrast in contrast_by_modality.items():
        c = np.asarray(contrast, dtype=float)
        if not np.all(np.isfinite(c)):
            raise ValueError(f"contrast image for modality {modality!r} must contain only finite values.")
        if z_step_nm is None:
            if c.ndim != 2:
                raise ValueError(
                    f"2D mode expects (H, W) contrast for modality {modality!r}; "
                    f"got shape {c.shape}."
                )
            central = c
        else:
            if c.ndim != 3 or c.shape[0] != 3:
                raise ValueError(
                    f"3D mode expects (3, H, W) z-stack for modality {modality!r}; "
                    f"got shape {c.shape}."
            )
            central = c[1]

        if measurement_model_by_modality is None:
            model = _default_measurement_model_for_modality(modality)
        else:
            model = measurement_model_by_modality.get(
                modality, _default_measurement_model_for_modality(modality)
            )
        model = _normalise_measurement_model(model)
        measurement_model_record[modality] = model

        if model == "count":
            count_domain_modalities.append(str(modality))
            modality_key = str(modality).lower()
            if modality_key.startswith(("tem", "sem")) or "electron" in modality_key:
                measurement_domain_record_units[modality] = "electron_count"
                signal_units_record[modality] = "electron_count"
                noise_variance_units_record[modality] = "electron_count_squared"
            else:
                measurement_domain_record_units[modality] = "detected_quanta"
                signal_units_record[modality] = "detected_quanta"
                noise_variance_units_record[modality] = "detected_quanta_squared"
            if detected_count_image_by_modality is not None and modality in detected_count_image_by_modality:
                count_image = np.asarray(
                    detected_count_image_by_modality[modality], dtype=float
                )
                if count_image.shape != c.shape:
                    raise ValueError(
                        "detected_count_image_by_modality[%r] has shape %s; "
                        "expected %s." % (modality, count_image.shape, c.shape)
                    )
                if not np.all(np.isfinite(count_image)):
                    raise ValueError(
                        "detected_count_image_by_modality[%r] must contain only finite values."
                        % modality
                    )
                count_mean_source_by_modality[str(modality)] = "detected_count_image"
            else:
                if require_detected_count_images:
                    raise ValueError(
                        "detected_count_image_by_modality must include count-domain "
                        f"modality {modality!r} when require_detected_count_images=True."
                    )
                # Contrast-only diagnostics may not have detector-count images.
                # Calibrated count-domain runs should pass actual count images
                # after scale_intensity_to_counts.
                if float(np.nanmin(central)) < 0.0:
                    shift = -float(np.nanmin(central))
                    count_image = c + shift
                    count_source = "contrast_proxy_shifted_nonnegative"
                else:
                    count_image = c.copy()
                    count_source = "contrast_proxy_nonnegative"
                count_mean_source_by_modality[str(modality)] = count_source
                proxy_count_modalities.append(str(modality))

            if z_step_nm is None:
                central_count = np.asarray(count_image, dtype=float)
            else:
                central_count = np.asarray(count_image, dtype=float)[1]
            central_count = np.where(np.isfinite(central_count), central_count, 0.0)
            central_count = np.maximum(central_count, 0.0)

            total_signal = float(np.sum(central_count))
            if total_signal <= 0.0:
                scale = 0.0
            else:
                scale = float(quanta_budget) / total_signal
            quanta_scale_by_modality[modality] = scale

            # The count image determines the Poisson mean/noise and the common
            # detected-quanta exposure scale. The default Fisher derivative
            # target is the signed particle contrast image; the count-mean
            # derivative mode is retained as an explicit diagnostic alternative.
            rescaled_c = scale * (np.asarray(count_image, dtype=float) if derivative_target == "count_mean_derivative" else c)
            if z_step_nm is None:
                mean_quanta = scale * central_count
            else:
                mean_quanta = scale * central_count
            var = mean_quanta + float(readout_variance)
            mean_signal = float(np.mean(mean_quanta)) if mean_quanta.size else 0.0
            denom = mean_signal + float(readout_variance)
            readout_fraction = (
                float(readout_variance) / denom if denom > 0.0 else 0.0
            )
            readout_variance_fraction_by_modality[modality] = readout_fraction
            count_readout_limited[modality] = bool(readout_variance > 0.0)
            budget_scaling_notes[modality] = (
                "count-domain ideal F is proportional to N scaling is exact only when additive readout variance is negligible"
                if readout_variance > 0.0
                else "count-domain ideal F is proportional to N scaling"
            )

        elif model == "phase":
            measurement_domain_record_units[modality] = "phase"
            signal_units_record[modality] = "radian"
            noise_variance_units_record[modality] = "radian_squared"
            visibility = _mapping_or_scalar_value(
                phase_visibility_by_modality, modality, 1.0
            )
            phase_readout_variance = _mapping_or_scalar_value(
                phase_readout_variance_by_modality, modality, 0.0
            )
            if not np.isfinite(visibility) or visibility <= 0.0:
                raise ValueError(
                    f"phase visibility for modality {modality!r} must be "
                    f"positive and finite; got {visibility!r}."
                )
            if (
                not np.isfinite(phase_readout_variance)
                or phase_readout_variance < 0.0
            ):
                raise ValueError(
                    f"phase readout variance for modality {modality!r} must be "
                    "non-negative and finite; got "
                    f"{phase_readout_variance!r}."
                )

            quanta_per_pixel = float(quanta_budget) / float(central.size)
            phase_variance = (
                1.0 / (visibility * visibility * quanta_per_pixel)
                + phase_readout_variance
            )
            quanta_per_pixel_by_modality[modality] = quanta_per_pixel
            phase_variance_by_modality[modality] = float(phase_variance)
            phase_visibility_record[modality] = float(visibility)
            quanta_scale_by_modality[modality] = None
            readout_variance_fraction_by_modality[modality] = float(
                phase_readout_variance / phase_variance
                if phase_variance > 0.0 else 0.0
            )
            phase_readout_limited[modality] = bool(phase_readout_variance > 0.0)
            budget_scaling_notes[modality] = (
                "phase-domain exact quanta scaling is broken by additive phase readout variance"
                if phase_readout_variance > 0.0
                else "phase-domain shot-noise scaling with var(phi)=1/(V squared n_Q)"
            )

            rescaled_c = c
            var = np.full(central.shape, phase_variance, dtype=float)

        else:
            raise AssertionError(f"Unhandled measurement model {model!r}.")

        # Floor the variance to a tiny positive constant to avoid
        # divide-by-zero in the Fisher gradient sum at zero-signal pixels in
        # the shot-noise-only regime.
        var = np.maximum(var, _FISHER_VARIANCE_FLOOR)

        rescaled_contrast[modality] = rescaled_c
        rescaled_noise[modality] = var

    result = compare_modality_information_content(
        rescaled_contrast,
        rescaled_noise,
        pixel_size_nm,
        z_step_nm=z_step_nm,
        pixel_size_nm_by_modality=pixel_size_nm_by_modality,
        measurement_domain_by_modality=measurement_domain_record_units,
        signal_units_by_modality=signal_units_record,
        noise_variance_units_by_modality=noise_variance_units_record,
    )

    result["quanta_budget"] = float(quanta_budget)
    result["readout_variance"] = float(readout_variance)
    result["normalization"] = "detected_quanta_domain_aware"
    result["detected_quanta_derivative_target"] = derivative_target
    result["measurement_model_by_modality"] = measurement_model_record
    result["measurement_domain_by_modality"] = measurement_domain_record_units
    result["signal_units_by_modality"] = signal_units_record
    result["noise_variance_units_by_modality"] = noise_variance_units_record
    result["quanta_scale_by_modality"] = quanta_scale_by_modality
    result["quanta_per_pixel_by_modality"] = quanta_per_pixel_by_modality
    result["phase_variance_by_modality"] = phase_variance_by_modality
    result["phase_visibility_by_modality"] = phase_visibility_record
    result["phase_readout_limited"] = phase_readout_limited
    result["count_readout_limited"] = count_readout_limited
    result["readout_variance_fraction_by_modality"] = readout_variance_fraction_by_modality
    result["budget_scaling_notes"] = budget_scaling_notes
    all_count_domain_modalities_have_detected_count_images = all(
        count_mean_source_by_modality.get(modality) == "detected_count_image"
        for modality in count_domain_modalities
    )
    strict_detected_count_basis = bool(all_count_domain_modalities_have_detected_count_images)
    if not count_domain_modalities:
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
        readout_variance_fraction_by_modality=readout_variance_fraction_by_modality,
    )
    result["detected_quanta_contract"]["derivative_target"] = derivative_target
    result["detected_quanta_contract"]["safe_for_detected_quanta_ranking"] = bool(strict_detected_count_basis)
    result["count_domain_modalities"] = count_domain_modalities
    result["proxy_count_modalities"] = proxy_count_modalities
    result["count_mean_source_by_modality"] = count_mean_source_by_modality
    result["all_count_domain_modalities_have_detected_count_images"] = bool(
        all_count_domain_modalities_have_detected_count_images
    )
    result["contract_q_proxy_diagnostic"] = bool(proxy_count_modalities)
    result["source_contract"] = "Contract-Q"
    result["contract_q_derivative_convergence_status"] = _infer_contract_q_derivative_status(
        parent_convergence_status_by_modality
    )
    parent_status_map = {
        str(modality): normalize_convergence_status(status)
        for modality, status in (parent_convergence_status_by_modality or {}).items()
    }
    result["parent_convergence_statuses"] = parent_status_map
    if parent_status_map:
        contract_q_parent_meta = {
            modality: {"convergence_status": status}
            for modality, status in parent_status_map.items()
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
    parent_convergence_results_by_modality: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build CSV-ready Contract-Q derivative convergence rows.

    Rows are intentionally source-contract level: the actual writing location is
    decided by notebooks/scripts, but the schema is owned by core.
    """
    from .convergence import FisherConvergenceStatus

    modalities = sorted(contract_q_result.get("measurement_model_by_modality", {}).keys())
    if not modalities and parent_convergence_results_by_modality:
        modalities = sorted(parent_convergence_results_by_modality.keys())
    parent_statuses = dict(contract_q_result.get("parent_convergence_statuses", {}))
    rows: list[dict[str, Any]] = []
    for modality in modalities:
        parent = (parent_convergence_results_by_modality or {}).get(modality, {})
        record = parent.get("convergence_status_record", parent)
        if isinstance(record, FisherConvergenceStatus):
            record = record.to_dict()
        if not isinstance(record, dict):
            record = {}
        status = normalize_convergence_status(
            record.get("convergence_status", record.get("status", parent_statuses.get(modality, contract_q_result.get("contract_q_derivative_convergence_status"))))
        )
        rows.append({
            "modality": modality,
            "source_contract": "Contract-Q",
            "derivative_target": contract_q_result.get("detected_quanta_derivative_target", "unknown"),
            "measurement_model": contract_q_result.get("measurement_model_by_modality", {}).get(modality, "unknown"),
            "count_mean_source": contract_q_result.get("count_mean_source_by_modality", {}).get(modality, ""),
            "proxy_count_diagnostic": bool(modality in contract_q_result.get("proxy_count_modalities", [])),
            "convergence_status": status,
            "validation_status": combine_parent_statuses({modality: {"convergence_status": status}})["validation_status"],
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
    parent_convergence_results_by_modality: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Write Contract-Q derivative convergence rows to CSV and return them."""
    rows = build_detected_quanta_derivative_convergence_rows(contract_q_result, parent_convergence_results_by_modality)
    fieldnames = [
        "modality", "source_contract", "derivative_target", "measurement_model",
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
    "compute_quanta_scaling_law",
    "check_budget_ranking_invariance",
    "check_budget_ordering_invariance",
    "compare_modality_information_content_detected_quanta_normalized",
    "build_detected_quanta_derivative_convergence_rows",
    "write_detected_quanta_derivative_convergence_csv",
]
