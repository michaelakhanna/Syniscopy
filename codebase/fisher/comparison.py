"""Single Fisher candidate-comparison owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from experiment_contracts import (
    ConvergenceStatus,
    ValidationStatus,
    combine_parent_statuses,
    normalize_convergence_status,
)
from noise_contracts import (
    fisher_noise_input_to_analysis_model,
    independent_pixel_noise_model,
)

from ._metadata_helpers import _variance_units
from .candidates import FisherCandidate, candidate_metadata_records


COMPARISON_TARGET_LATERAL_XY = "lateral_xy"
COMPARISON_TARGET_LOCALIZATION_XYZ = "localization_xyz"
COMPARISON_TARGET_AXIAL_Z = "axial_z"
COMPARISON_TARGET_ORIENTATION = "orientation"


@dataclass(frozen=True)
class CandidateRankingSpec:
    """One ranking view over already-computed per-candidate CRLB rows."""

    sigma_key: str
    ordering_key: str
    diagnostic_ordering_key: str
    best_key: str
    relative_key: str
    frames_to_match_key: str
    singular_keys: tuple[str, ...]
    rankable_key: str = "rankable_for_ordering"
    exclusion_reason_key: str = "ordering_exclusion_reason"


def _sort_key_finite_then_value(pair: tuple[str, float]) -> tuple[int, float]:
    value = pair[1]
    if not np.isfinite(value) or value <= 0.0:
        return (1, 0.0)
    return (0, value)


def _positive_finite_or_inf(value: Any) -> float:
    value_float = float(value)
    return value_float if np.isfinite(value_float) and value_float > 0.0 else float("inf")


def _comparison_rank_metadata(
    per_candidate: Mapping[str, Mapping[str, Any]],
    parent_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    supplied = parent_metadata or {}
    for candidate, row in per_candidate.items():
        row_payload = dict(row or {})
        payload = dict(supplied.get(candidate, {})) if parent_metadata is not None else {}
        for key in (
            "convergence_status",
            "validation_status",
            "production_grid_diagnostic",
            "safe_for_ordering",
            "safe_for_fusion",
            "safe_for_time_allocation",
            "safe_for_registration",
            "safe_for_detected_quanta_ranking",
            "status_reason",
        ):
            if key not in payload and key in row_payload:
                payload[key] = row_payload[key]
        payload["convergence_status"] = normalize_convergence_status(
            payload.get("convergence_status", ConvergenceStatus.UNCHECKED.value)
        )
        payload.setdefault("validation_status", ValidationStatus.UNCHECKED.value)
        metadata[str(candidate)] = payload
    return metadata


def _metadata_allows_recommendation_ordering(metadata: Mapping[str, Any]) -> bool:
    status = normalize_convergence_status(
        metadata.get("convergence_status", ConvergenceStatus.UNCHECKED.value)
    )
    if status != ConvergenceStatus.FINITE_CONVERGED.value:
        return False
    if bool(metadata.get("production_grid_diagnostic", False)):
        return False
    if "safe_for_ordering" in metadata and not bool(metadata.get("safe_for_ordering")):
        return False
    return True


def _recommendation_exclusion_reason(
    *,
    sigma: float,
    metadata: Mapping[str, Any],
    rankable: bool,
) -> str:
    if rankable:
        return ""
    if not (np.isfinite(sigma) and sigma > 0.0):
        return "nonfinite_or_singular_sigma"
    status = normalize_convergence_status(
        metadata.get("convergence_status", ConvergenceStatus.UNCHECKED.value)
    )
    if status != ConvergenceStatus.FINITE_CONVERGED.value:
        return f"convergence_status={status}"
    if bool(metadata.get("production_grid_diagnostic", False)):
        return "production_grid_diagnostic=True"
    if "safe_for_ordering" in metadata and not bool(metadata.get("safe_for_ordering")):
        return "safe_for_ordering=False"
    return "not_rankable_for_ordering"


def _build_recommendation_ranking(
    per_candidate: dict[str, dict[str, Any]],
    *,
    spec: CandidateRankingSpec,
    candidates_in_order: Sequence[str],
    rank_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostic_values: list[tuple[str, float]] = []
    recommendation_values: list[tuple[str, float]] = []
    rankability: dict[str, bool] = {}
    exclusion_reasons: dict[str, str] = {}
    for candidate in candidates_in_order:
        row = per_candidate[candidate]
        sigma = _positive_finite_or_inf(row.get(spec.sigma_key, float("inf")))
        if any(bool(row.get(key, False)) for key in spec.singular_keys):
            sigma = float("inf")
        metadata = rank_metadata.get(str(candidate), {})
        rankable = bool(
            _metadata_allows_recommendation_ordering(metadata)
            and np.isfinite(sigma)
            and sigma > 0.0
        )
        recommendation_sigma = sigma if rankable else float("inf")
        diagnostic_values.append((candidate, sigma))
        recommendation_values.append((candidate, recommendation_sigma))
        reason = _recommendation_exclusion_reason(
            sigma=sigma,
            metadata=metadata,
            rankable=rankable,
        )
        rankability[candidate] = rankable
        exclusion_reasons[candidate] = reason
        row[spec.rankable_key] = rankable
        row[spec.exclusion_reason_key] = reason
        row.setdefault(spec.diagnostic_ordering_key.replace("ordering", "sigma"), sigma)

    ordering = sorted(recommendation_values, key=_sort_key_finite_then_value)
    diagnostic_ordering = sorted(diagnostic_values, key=_sort_key_finite_then_value)
    finite_ordering = [
        (candidate, sigma)
        for candidate, sigma in ordering
        if np.isfinite(sigma) and sigma > 0.0
    ]
    best_candidate = finite_ordering[0][0] if finite_ordering else None
    best_sigma = finite_ordering[0][1] if finite_ordering else float("inf")
    recommendation_sigma = dict(recommendation_values)
    relative = {
        candidate: (
            float(sigma / best_sigma)
            if np.isfinite(sigma)
            and sigma > 0.0
            and np.isfinite(best_sigma)
            and best_sigma > 0.0
            else float("inf")
        )
        for candidate, sigma in recommendation_sigma.items()
    }
    frames_to_match = {
        candidate: (
            float(relative[candidate] ** 2)
            if np.isfinite(relative[candidate])
            else float("inf")
        )
        for candidate in recommendation_sigma
    }
    return {
        "ordering": ordering,
        "diagnostic_ordering": diagnostic_ordering,
        "best_candidate": best_candidate,
        "best_sigma": best_sigma,
        "relative": relative,
        "frames_to_match": frames_to_match,
        "rankability": rankability,
        "exclusion_reasons": exclusion_reasons,
    }


def _parent_metadata_from_candidates(candidates: Sequence[FisherCandidate]) -> dict[str, dict[str, Any]] | None:
    metadata = {
        candidate.key: dict(candidate.parent_result_metadata)
        for candidate in candidates
        if candidate.parent_result_metadata
    }
    return metadata or None


def _resolve_fisher_candidate_noise_input(
    candidate: FisherCandidate,
    *,
    context: str,
) -> Any:
    candidate_context = f"{context}[{candidate.key!r}]"
    if candidate.analysis_noise_model is not None:
        return fisher_noise_input_to_analysis_model(
            candidate.analysis_noise_model,
            context=candidate_context,
        )
    covariance_kind = str(candidate.noise_covariance_kind or "").strip()
    if covariance_kind != "independent_pixels":
        raise ValueError(
            f"{candidate_context}: candidates with diagonal noise_variance must declare "
            "noise_covariance_kind='independent_pixels' or carry analysis_noise_model."
        )
    return independent_pixel_noise_model(
        candidate.noise_variance,
        measurement_domain=candidate.measurement_domain,
        signal_units=candidate.signal_units,
        noise_variance_units=candidate.noise_variance_units,
        context=f"{candidate_context} independent-pixel noise",
    )


def resolve_fisher_candidate_noise_inputs(
    candidates: Sequence[FisherCandidate],
    *,
    context: str,
) -> dict[str, Any]:
    return {
        candidate.key: _resolve_fisher_candidate_noise_input(candidate, context=context)
        for candidate in candidates
    }


def fisher_derivative_basis_for_candidate(
    candidate: FisherCandidate,
    *,
    target: str,
    z_step_nm: float | None,
) -> dict[str, Any]:
    if target == COMPARISON_TARGET_ORIENTATION:
        return {"fisher_derivative_basis": "se3_explicit_pose_rerenders"}
    from .lateral_derivative_contracts import (
        array_only_derivative_context_metadata,
        normalize_array_only_fisher_derivative_context,
        require_array_only_3d_fisher_derivative_basis_safe,
        require_array_only_spectral_lateral_derivative_ready,
        spectral_lateral_derivative_plan,
    )

    context = normalize_array_only_fisher_derivative_context(
        candidate.derivative_context,
        context=f"compare_fisher_candidates[{candidate.key!r}]",
    )
    if target == COMPARISON_TARGET_LATERAL_XY:
        from .dhm_demodulated import (
            OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
            is_off_axis_demodulated_fisher_payload,
            is_off_axis_holography_modality,
        )

        candidate_noise = (
            candidate.analysis_noise_model
            if candidate.analysis_noise_model is not None
            else candidate.noise_variance
        )
        try:
            candidate_noise = fisher_noise_input_to_analysis_model(
                candidate_noise,
                context=f"compare_fisher_candidates[{candidate.key!r}] derivative-basis noise",
            )
        except (TypeError, ValueError):
            pass
        if is_off_axis_holography_modality(candidate.modality) and is_off_axis_demodulated_fisher_payload(
            candidate.signal,
            candidate_noise,
        ):
            payload = array_only_derivative_context_metadata(context, spectral_lateral_derivative_plan())
            payload.update(
                {
                    "fisher_lateral_derivative_basis": "off_axis_demodulated_complex_spectral_band_limited",
                    "fisher_lateral_derivative_basis_resolution": "caller_supplied_demodulated_dhm_sideband_fft_spectral_gradient",
                    "fisher_noise_covariance_model": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
                    "demodulated_field_has_detector_fixed_carrier": False,
                }
            )
            return payload
        lateral_plan = require_array_only_spectral_lateral_derivative_ready(
            modality=candidate.modality,
            params=context.params,
            model=context.model,
            response_function=context.response_function,
            num_particles=context.num_particles,
            structured_environment_active=context.structured_environment_active,
            context=f"compare_fisher_candidates[{candidate.key!r}]",
        )
        return array_only_derivative_context_metadata(context, lateral_plan)
    if target in {COMPARISON_TARGET_LOCALIZATION_XYZ, COMPARISON_TARGET_AXIAL_Z}:
        if z_step_nm is None:
            raise ValueError(f"{target} comparison requires z_step_nm.")
        basis = require_array_only_3d_fisher_derivative_basis_safe(
            modality=candidate.modality,
            z_step_nm=float(z_step_nm),
            params=context.params,
            model=context.model,
            response_function=context.response_function,
            num_particles=context.num_particles,
            structured_environment_active=context.structured_environment_active,
            context=f"compare_fisher_candidates[{candidate.key!r}]",
        )
        basis.update(context.to_metadata())
        return basis
    raise ValueError(f"unknown Fisher comparison target {target!r}.")


def _evaluate_candidate(
    candidate: FisherCandidate,
    noise_input: Any,
    *,
    target: str,
    z_step_nm: float | None,
    rotation_step_rad: float | None,
) -> dict[str, Any]:
    if target == COMPARISON_TARGET_LATERAL_XY:
        from .dhm_demodulated import (
            compute_off_axis_demodulated_localization_crlb_from_field,
            is_off_axis_demodulated_fisher_payload,
            is_off_axis_holography_modality,
        )

        if is_off_axis_holography_modality(candidate.modality) and is_off_axis_demodulated_fisher_payload(
            candidate.signal,
            noise_input,
        ):
            result = compute_off_axis_demodulated_localization_crlb_from_field(
                candidate.signal,
                noise_input,
                candidate.pixel_size_nm,
            )
        else:
            from .lateral import compute_localization_crlb

            result = compute_localization_crlb(
                candidate.signal,
                noise_input,
                candidate.pixel_size_nm,
                signal_units=candidate.signal_units,
                measurement_domain=candidate.measurement_domain,
                noise_variance_units=candidate.noise_variance_units,
            )
    elif target in {COMPARISON_TARGET_LOCALIZATION_XYZ, COMPARISON_TARGET_AXIAL_Z}:
        if z_step_nm is None or not np.isfinite(z_step_nm) or float(z_step_nm) <= 0.0:
            raise ValueError(f"{target} comparison requires positive finite z_step_nm.")
        from .axial import compute_localization_crlb_3d

        result = compute_localization_crlb_3d(
            candidate.signal,
            noise_input,
            candidate.pixel_size_nm,
            float(z_step_nm),
            signal_units=candidate.signal_units,
            measurement_domain=candidate.measurement_domain,
            noise_variance_units=candidate.noise_variance_units,
        )
    elif target == COMPARISON_TARGET_ORIENTATION:
        if z_step_nm is None or not np.isfinite(z_step_nm) or float(z_step_nm) <= 0.0:
            raise ValueError("orientation comparison requires positive finite z_step_nm.")
        if (
            rotation_step_rad is None
            or not np.isfinite(rotation_step_rad)
            or float(rotation_step_rad) <= 0.0
        ):
            raise ValueError("orientation comparison requires positive finite rotation_step_rad.")
        from .se3 import compute_localization_orientation_crlb

        result = compute_localization_orientation_crlb(
            renders=candidate.signal,
            noise_variance_map=noise_input,
            pixel_size_nm=candidate.pixel_size_nm,
            z_step_nm=float(z_step_nm),
            rotation_step_rad=float(rotation_step_rad),
            signal_units=candidate.signal_units,
            measurement_domain=candidate.measurement_domain,
            noise_variance_units=candidate.noise_variance_units,
        )
    else:
        raise ValueError(f"unknown Fisher comparison target {target!r}.")
    result["modality"] = candidate.modality
    if candidate.parent_result_metadata:
        result["parent_convergence_status"] = normalize_convergence_status(
            candidate.parent_result_metadata.get("convergence_status", "unchecked")
        )
    return result


def _ranking_specs_for_target(target: str) -> tuple[CandidateRankingSpec, ...]:
    if target == COMPARISON_TARGET_LATERAL_XY:
        return (
            CandidateRankingSpec(
                sigma_key="sigma_xy_nm",
                ordering_key="ordering_xy",
                diagnostic_ordering_key="diagnostic_ordering_xy",
                best_key="best_candidate_xy",
                relative_key="relative_sigma_xy",
                frames_to_match_key="frames_to_match_best_xy",
                singular_keys=("singular", "fisher_singular"),
            ),
        )
    if target == COMPARISON_TARGET_LOCALIZATION_XYZ:
        return (
            CandidateRankingSpec(
                sigma_key="sigma_xy_nm",
                ordering_key="ordering_xy",
                diagnostic_ordering_key="diagnostic_ordering_xy",
                best_key="best_candidate_xy",
                relative_key="relative_sigma_xy",
                frames_to_match_key="frames_to_match_best_xy",
                singular_keys=("xy_singular",),
            ),
            CandidateRankingSpec(
                sigma_key="sigma_xyz_nm",
                ordering_key="ordering_xyz",
                diagnostic_ordering_key="diagnostic_ordering_xyz",
                best_key="best_candidate_xyz",
                relative_key="relative_sigma_xyz",
                frames_to_match_key="frames_to_match_best_xyz",
                singular_keys=("singular", "fisher_singular", "axially_singular"),
                rankable_key="rankable_for_xyz_ordering",
                exclusion_reason_key="xyz_ordering_exclusion_reason",
            ),
        )
    if target == COMPARISON_TARGET_AXIAL_Z:
        return (
            CandidateRankingSpec(
                sigma_key="sigma_z_nm",
                ordering_key="ordering_z",
                diagnostic_ordering_key="diagnostic_ordering_z",
                best_key="best_candidate_z",
                relative_key="relative_sigma_z",
                frames_to_match_key="frames_to_match_best_z",
                singular_keys=("axially_singular",),
                rankable_key="rankable_for_z_ordering",
                exclusion_reason_key="z_ordering_exclusion_reason",
            ),
        )
    if target == COMPARISON_TARGET_ORIENTATION:
        return (
            CandidateRankingSpec(
                sigma_key="sigma_omega_total_rad",
                ordering_key="ordering",
                diagnostic_ordering_key="diagnostic_ordering",
                best_key="best_candidate",
                relative_key="relative_sigma_omega",
                frames_to_match_key="frames_to_match_best",
                singular_keys=("singular",),
                rankable_key="rankable_for_orientation_ordering",
                exclusion_reason_key="orientation_ordering_exclusion_reason",
            ),
        )
    raise ValueError(f"unknown Fisher comparison target {target!r}.")


def _add_target_specific_outputs(out: dict[str, Any], *, target: str) -> None:
    per_candidate = out["per_candidate"]
    if target == COMPARISON_TARGET_AXIAL_Z:
        out["axially_singular_per_candidate"] = {
            key: bool(row.get("axially_singular", True))
            for key, row in per_candidate.items()
        }
    elif target == COMPARISON_TARGET_ORIENTATION:
        out["best_candidate_full_rank"] = None
        for candidate, sigma in out["ordering"]:
            row = per_candidate[candidate]
            if (
                np.isfinite(sigma)
                and sigma > 0.0
                and row.get("rank", 0) == 6
                and not row.get("axes_singular", [])
            ):
                out["best_candidate_full_rank"] = candidate
                break
        out["axes_singular_per_candidate"] = {
            key: list(row.get("axes_singular", []))
            for key, row in per_candidate.items()
        }


def compare_fisher_candidates(
    candidates: Sequence[FisherCandidate],
    *,
    target: str = COMPARISON_TARGET_LATERAL_XY,
    z_step_nm: float | None = None,
    rotation_step_rad: float | None = None,
) -> dict[str, Any]:
    """Compare Fisher candidates with one ranking/comparison implementation."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("compare_fisher_candidates requires at least one candidate.")
    keys = [candidate.key for candidate in candidate_list]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"Fisher candidate keys must be unique; duplicates: {duplicates!r}.")
    target = str(target).strip().lower()
    if target not in {
        COMPARISON_TARGET_LATERAL_XY,
        COMPARISON_TARGET_LOCALIZATION_XYZ,
        COMPARISON_TARGET_AXIAL_Z,
        COMPARISON_TARGET_ORIENTATION,
    }:
        raise ValueError(f"unknown Fisher comparison target {target!r}.")

    noise_inputs = resolve_fisher_candidate_noise_inputs(
        candidate_list,
        context=f"compare_fisher_candidates:{target}",
    )
    per_candidate: dict[str, dict[str, Any]] = {}
    derivative_basis: dict[str, dict[str, Any]] = {}
    for candidate in candidate_list:
        derivative_basis[candidate.key] = fisher_derivative_basis_for_candidate(
            candidate,
            target=target,
            z_step_nm=z_step_nm,
        )
        per_candidate[candidate.key] = _evaluate_candidate(
            candidate,
            noise_inputs[candidate.key],
            target=target,
            z_step_nm=z_step_nm,
            rotation_step_rad=rotation_step_rad,
        )

    parent_metadata = _parent_metadata_from_candidates(candidate_list)
    rank_metadata = _comparison_rank_metadata(
        per_candidate,
        parent_metadata,
    )
    candidate_records = candidate_metadata_records(candidate_list)
    record_by_key = {record["candidate_key"]: record for record in candidate_records}
    for candidate in candidate_list:
        record = record_by_key[candidate.key]
        record["fisher_derivative_basis"] = derivative_basis[candidate.key]
        record["noise_variance_units"] = (
            per_candidate[candidate.key].get("noise_variance_units")
            or candidate.noise_variance_units
            or _variance_units(candidate.signal_units)
        )
    out: dict[str, Any] = {
        "per_candidate": per_candidate,
        "candidate_records": candidate_records,
        "comparison_target": target,
        "candidate_keys": keys,
    }
    for spec in _ranking_specs_for_target(target):
        ranking = _build_recommendation_ranking(
            per_candidate,
            spec=spec,
            candidates_in_order=keys,
            rank_metadata=rank_metadata,
        )
        out[spec.ordering_key] = ranking["ordering"]
        out[spec.diagnostic_ordering_key] = ranking["diagnostic_ordering"]
        out[spec.best_key] = ranking["best_candidate"]
        out[spec.relative_key] = ranking["relative"]
        out[spec.frames_to_match_key] = ranking["frames_to_match"]
        for candidate_key in keys:
            record_by_key[candidate_key][spec.rankable_key] = ranking["rankability"][candidate_key]
            record_by_key[candidate_key][spec.exclusion_reason_key] = ranking["exclusion_reasons"][candidate_key]

    _add_target_specific_outputs(out, target=target)
    parent_metadata = combine_parent_statuses(rank_metadata)
    out["parent_status_metadata"] = parent_metadata
    out["candidate_parent_convergence_statuses"] = parent_metadata["parent_convergence_statuses"]
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


__all__ = [
    "COMPARISON_TARGET_AXIAL_Z",
    "COMPARISON_TARGET_LATERAL_XY",
    "COMPARISON_TARGET_LOCALIZATION_XYZ",
    "COMPARISON_TARGET_ORIENTATION",
    "CandidateRankingSpec",
    "compare_fisher_candidates",
    "fisher_derivative_basis_for_candidate",
    "resolve_fisher_candidate_noise_inputs",
]
