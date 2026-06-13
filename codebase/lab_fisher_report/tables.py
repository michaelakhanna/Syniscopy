"""Numeric table builders for lab Fisher reports."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from common_utils import init_infinite_dict
from config import AcquisitionProfile, MotionDynamicsSettings
from experiment_contracts import (
    ConvergenceStatus,
    ValidationStatus,
    combine_parent_statuses,
)
from fisher import (
    FisherMatrixCandidate,
    build_brownian_process_covariance,
    compute_candidate_fusion_crlb_from_fisher_matrices,
    compute_dynamic_bayesian_crlb_from_fisher_sequence,
    sequence_sum_fisher_to_crlb,
)
from fisher.fusion import _fusion_subset_metadata_for_precomputed_matrices
from modality_registry import modality_display_name, require_modality_name
from noise_contracts import resolve_fisher_likelihood_eligibility
from trajectory import stokes_einstein_diffusion_coefficient, resolve_translational_diameters_nm

__all__ = [
    "_build_sequence_summary_rows",
    "_compute_dynamic_sequence_summary",
    "_crlb_from_fisher_matrix",
    "_dynamic_sequence_summary_to_ranking_summary",
    "_dynamic_fusion_rows_from_fisher_sequences",
    "_fusion_rows_from_fisher_matrices",
    "_fusion_subset_sizes_for_report",
    "_ranking_rows",
    "_select_fusion_fisher_inputs",
    "_sequence_information_content",
]


STATIC_FRAME_EQUIVALENCE_MODEL = "static_independent_same_state_inverse_sqrt_n"
DYNAMIC_FRAME_EQUIVALENCE_MODEL = "dynamic_bayesian_process_model"
NO_FRAME_EQUIVALENCE_MODEL = "not_applicable"
FRAME_EQUIVALENCE_COMPUTED_STATUS = "computed_static_same_state_inverse_sqrt_n"
FRAME_EQUIVALENCE_DYNAMIC_STATUS = "not_applicable_dynamic_bayesian_process_model"
FRAME_EQUIVALENCE_NO_CONTRACT_STATUS = "not_applicable_no_static_frame_scaling_contract"
FRAME_EQUIVALENCE_UNSAFE_STATUS = "not_applicable_nonfinite_or_unsafe"
FISHER_DIAGNOSTIC_FIELDS = (
    "derivative_basis",
    "nyquist_band_fraction",
    "boundary_energy_fraction",
    "convergence_status",
)


def _candidate_identity_fields(candidate_key: str, summary: dict[str, Any]) -> tuple[str, str, str]:
    """Return microscope identity, canonical modality metadata, and display name.

    The ranking engine is being rekeyed from modality strings to microscope
    names. When the comparison key is not itself a modality, the summary must
    carry explicit modality metadata so same-modality microscopes remain
    distinct candidates without losing the backend identity used for rendering.
    """

    raw_modality = summary.get("modality", candidate_key)
    try:
        modality = require_modality_name(raw_modality)
    except ValueError as exc:
        raise ValueError(
            f"Candidate {candidate_key!r} must provide valid 'modality' metadata "
            "when the ranking key is not a supported modality name."
        ) from exc
    microscope = str(
        summary.get("microscope")
        or summary.get("microscope_name")
        or candidate_key
    ).strip()
    if not microscope:
        raise ValueError(f"Candidate {candidate_key!r} resolved to an empty microscope identity.")
    display_name = str(
        summary.get("display_name")
        or summary.get("microscope_display_name")
        or (microscope if microscope != modality else modality_display_name(modality))
    )
    return microscope, modality, display_name


def _frame_equivalence_model_for_summary(summary: dict[str, Any]) -> str:
    """Return the declared temporal scaling law for report frame-equivalence."""
    explicit = str(summary.get("frame_equivalence_model", "")).strip()
    if explicit:
        return explicit
    sequence_model = str(summary.get("sequence_crlb_model", "")).strip()
    same_state = bool(summary.get("same_state_assumption", False))
    if same_state and sequence_model in {
        "single_frame_static",
        "static_same_state_cumulative",
        "static_same_state_cumulative_diagnostic",
    }:
        return STATIC_FRAME_EQUIVALENCE_MODEL
    if sequence_model == "dynamic_bayesian_estimator" or not same_state:
        return DYNAMIC_FRAME_EQUIVALENCE_MODEL
    return NO_FRAME_EQUIVALENCE_MODEL


def _can_use_static_frame_equivalence(summary: dict[str, Any]) -> bool:
    return (
        str(summary.get("frame_equivalence_model", ""))
        == STATIC_FRAME_EQUIVALENCE_MODEL
    )


def _frame_equivalence_status(summary: dict[str, Any], *, finite_sigma: bool, best_static_contract: bool) -> str:
    if not finite_sigma:
        return FRAME_EQUIVALENCE_UNSAFE_STATUS
    if _can_use_static_frame_equivalence(summary) and best_static_contract:
        return FRAME_EQUIVALENCE_COMPUTED_STATUS
    if str(summary.get("frame_equivalence_model", "")) == DYNAMIC_FRAME_EQUIVALENCE_MODEL:
        return FRAME_EQUIVALENCE_DYNAMIC_STATUS
    if not best_static_contract:
        return FRAME_EQUIVALENCE_NO_CONTRACT_STATUS
    return FRAME_EQUIVALENCE_NO_CONTRACT_STATUS


def _crlb_from_fisher_matrix(fisher_matrix: np.ndarray) -> dict[str, Any]:
    from fisher._metadata_helpers import _fisher_rank_metadata
    from fisher.fusion import _axis_sigmas_from_fisher

    fisher = np.asarray(fisher_matrix, dtype=float)
    if fisher.shape != (2, 2) or not np.all(np.isfinite(fisher)):
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "fisher_xx": float("inf"),
            "fisher_xy": float("nan"),
            "fisher_yy": float("nan"),
            "fisher_det": float("nan"),
            "fisher_matrix": np.full((2, 2), float("nan")),
            "fisher_rank": 0,
            "singular": True,
        }

    symmetric = 0.5 * (fisher + fisher.T)
    det = float(np.linalg.det(symmetric))
    rank_metadata = _fisher_rank_metadata(symmetric)
    axis_sigmas, axis_singular = _axis_sigmas_from_fisher(symmetric)
    sigma_x = float(axis_sigmas[0])
    sigma_y = float(axis_sigmas[1])
    singular = bool(axis_singular[0] or axis_singular[1])
    sigma_xy = (
        float(math.hypot(sigma_x, sigma_y))
        if not singular and np.isfinite(sigma_x) and np.isfinite(sigma_y)
        else float("inf")
    )

    return {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_xy_nm": sigma_xy,
        "fisher_xx": float(symmetric[0, 0]),
        "fisher_xy": float(symmetric[0, 1]),
        "fisher_yy": float(symmetric[1, 1]),
        "fisher_det": det,
        "fisher_matrix": symmetric,
        "fisher_rank": int(rank_metadata["numerical_fisher_rank"]),
        "singular": singular,
        "axes_singular": [
            axis
            for axis, is_singular in zip(("x", "y"), axis_singular)
            if is_singular
        ],
    }



def _sequence_information_content(
    microscope_order: list[str],
    final_fisher_by_microscope: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not final_fisher_by_microscope:
        raise ValueError("No microscope summaries were provided for comparison.")

    if not microscope_order:
        microscope_order = list(final_fisher_by_microscope)

    # The comparison-key map is microscope-owned. Modality remains metadata inside
    # each row so same-modality microscopes are not accidentally collapsed by a
    # downstream table/ranking/fusion consumer.
    per_microscope: dict[str, dict[str, Any]] = {}
    for microscope_key in microscope_order:
        if microscope_key not in final_fisher_by_microscope:
            continue
        summary = dict(final_fisher_by_microscope[microscope_key])
        microscope, modality, display_name = _candidate_identity_fields(microscope_key, summary)
        safe_for_ordering = bool(summary.get("safe_for_ordering", False))
        sigma_x_nm = float(summary.get("sigma_x_nm", float("inf")))
        sigma_y_nm = float(summary.get("sigma_y_nm", float("inf")))
        sigma_xy_nm = float(summary.get("sigma_xy_nm", float("inf")))
        if not safe_for_ordering:
            sigma_x_nm = float("inf")
            sigma_y_nm = float("inf")
            sigma_xy_nm = float("inf")
        per_microscope[microscope_key] = {
            "microscope": microscope,
            "modality": modality,
            "display_name": display_name,
            "sigma_x_nm": sigma_x_nm,
            "sigma_y_nm": sigma_y_nm,
            "sigma_xy_nm": sigma_xy_nm,
            "latent_scene_id": str(summary.get("latent_scene_id", "")),
            "latent_schedule_id": str(summary.get("latent_schedule_id", "")),
            "state_time_policy": str(summary.get("state_time_policy", "")),
            "fusion_time_alignment": str(summary.get("fusion_time_alignment", "")),
            "shared_coordinate_frame": str(summary.get("shared_coordinate_frame", "")),
            "same_latent_scene": bool(summary.get("same_latent_scene", False)),
            "measurement_domain": str(summary.get("measurement_domain", "")),
            "signal_units": str(summary.get("signal_units", "")),
            "noise_variance_units": str(summary.get("noise_variance_units", "")),
            "detector_noise_input_domain": str(summary.get("detector_noise_input_domain", "")),
            "nonlinear_detector_effects_active": bool(summary.get("nonlinear_detector_effects_active", False)),
            "deterministic_detector_transfer_active": bool(summary.get("deterministic_detector_transfer_active", False)),
            "safe_for_linear_fisher_variance": bool(summary.get("safe_for_linear_fisher_variance", False)),
            "safe_for_covariance_fisher_variance": bool(summary.get("safe_for_covariance_fisher_variance", False)),
            "detector_safe_for_report_fisher": bool(summary.get("detector_safe_for_report_fisher", False)),
            "fisher_likelihood_uses_covariance": bool(summary.get("fisher_likelihood_uses_covariance", False)),
            "fisher_likelihood_eligibility_contract_id": str(summary.get("fisher_likelihood_eligibility_contract_id", "")),
            "fisher_variance_model_scope": str(summary.get("fisher_variance_model_scope", "")),
            "covariance_fisher_variance_model_scope": str(summary.get("covariance_fisher_variance_model_scope", "")),
            "detector_likelihood_status": str(summary.get("detector_likelihood_status", "")),
            "fisher_xx": float(summary.get("fisher_xx", float("nan"))),
            "fisher_xy": float(summary.get("fisher_xy", float("nan"))),
            "fisher_yy": float(summary.get("fisher_yy", float("nan"))),
            "fisher_matrix": np.asarray(summary.get("fisher_matrix", np.full((2, 2), float("nan")))),
            "singular": bool(summary.get("singular", True)),
            "fisher_rank": int(summary.get("fisher_rank", 0)),
            "derivative_basis": str(summary.get("derivative_basis", "")),
            "nyquist_band_fraction": summary.get("nyquist_band_fraction", ""),
            "boundary_energy_fraction": summary.get("boundary_energy_fraction", ""),
            "convergence_status": str(summary.get("convergence_status", "")),
            "num_frames": int(summary.get("num_frames", 1)),
            "sequence_crlb_model": str(summary.get("sequence_crlb_model", "")),
            "same_state_assumption": bool(summary.get("same_state_assumption", False)),
            "frame_equivalence_model": _frame_equivalence_model_for_summary(summary),
            "safe_for_ordering": safe_for_ordering,
            "safe_for_fusion": bool(summary.get("safe_for_fusion", False)),
            "status_reason": str(summary.get("status_reason", "")),
        }

    def _positive_sigma_order(item: tuple[str, dict[str, Any]]) -> tuple[int, float, int]:
        sigma = float(item[1]["sigma_xy_nm"])
        try:
            order_index = microscope_order.index(item[0])
        except ValueError:
            order_index = len(microscope_order)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return (1, 0.0, order_index)
        return (0, sigma, order_index)

    def _positive_sigma_value(value: Any) -> float:
        sigma = float(value)
        return sigma if np.isfinite(sigma) and sigma > 0.0 else float("inf")

    ordered = sorted(per_microscope.items(), key=_positive_sigma_order)
    ordering_xy = [
        (microscope_key, _positive_sigma_value(summary["sigma_xy_nm"]))
        for microscope_key, summary in ordered
    ]

    best_sigma_xy = ordered[0][1]["sigma_xy_nm"] if ordered else float("inf")
    if np.isfinite(best_sigma_xy) and best_sigma_xy > 0.0:
        best_microscope_xy = ordered[0][0]
        # A tied minimum sigma_xy_nm is a tied scientific recommendation. Keep
        # the singular field as the first representative, but expose the
        # complete tied-best microscope set so manifests cannot imply that one
        # equal Fisher/CRLB result is worse than another.
        best_microscopes_xy = [
            microscope_key
            for microscope_key, summary in ordered
            if bool(summary.get("safe_for_ordering", False))
            and np.isfinite(float(summary["sigma_xy_nm"]))
            and float(summary["sigma_xy_nm"]) > 0.0
            and math.isclose(
                float(summary["sigma_xy_nm"]),
                float(best_sigma_xy),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ]
        relative_sigma_xy = {
            microscope_key: (
                float(summary["sigma_xy_nm"]) / float(best_sigma_xy)
                if np.isfinite(float(summary["sigma_xy_nm"])) and float(summary["sigma_xy_nm"]) > 0.0
                else float("inf")
            )
            for microscope_key, summary in per_microscope.items()
        }
        best_static_contract = _can_use_static_frame_equivalence(ordered[0][1])
        frames_to_match_best_xy: dict[str, float | None] = {}
        frames_to_match_best_xy_status: dict[str, str] = {}
        for microscope_key, summary in per_microscope.items():
            sigma = float(summary["sigma_xy_nm"])
            finite_sigma = bool(np.isfinite(sigma) and sigma > 0.0)
            status = _frame_equivalence_status(
                summary,
                finite_sigma=finite_sigma,
                best_static_contract=best_static_contract,
            )
            frames_to_match_best_xy_status[microscope_key] = status
            if status == FRAME_EQUIVALENCE_COMPUTED_STATUS:
                frames_to_match_best_xy[microscope_key] = (sigma / float(best_sigma_xy)) ** 2
            elif finite_sigma:
                frames_to_match_best_xy[microscope_key] = None
            else:
                frames_to_match_best_xy[microscope_key] = float("inf")
    else:
        best_microscope_xy = None
        best_microscopes_xy = []
        relative_sigma_xy = init_infinite_dict(per_microscope)
        frames_to_match_best_xy = init_infinite_dict(per_microscope)
        frames_to_match_best_xy_status = {
            microscope_key: FRAME_EQUIVALENCE_UNSAFE_STATUS
            for microscope_key in per_microscope
        }

    return {
        "per_microscope": per_microscope,
        "ordering_xy": ordering_xy,
        "best_microscope_xy": best_microscope_xy,
        "best_microscopes_xy": best_microscopes_xy,
        "relative_sigma_xy": relative_sigma_xy,
        "frames_to_match_best_xy": frames_to_match_best_xy,
        "frames_to_match_best_xy_status": frames_to_match_best_xy_status,
    }

def _report_optional_float(value: Any) -> float | str:
    """Serialize optional scientific report numbers without fabricating values.

    ``frames_to_match_best_xy`` is a numeric acquisition-time claim only for
    static independent observations of the same particle state.  Dynamic
    Bayesian Brownian rows deliberately carry ``None`` because their
    current-state covariance includes process noise between frames; coercing
    that sentinel through ``float`` either crashes a valid comparison or tempts
    a future patch to emit a false static frame count.  Keep the CSV/report cell
    blank and let the adjacent status/model columns state the temporal contract.
    """
    if value is None:
        return ""
    return float(value)



def _ranking_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_rank = 1
    last_rankable_sigma_xy: float | None = None
    last_rank_value = 0
    best_microscopes_xy = set(result.get("best_microscopes_xy", []) or [])
    if not best_microscopes_xy and result.get("best_microscope_xy"):
        best_microscopes_xy.add(str(result.get("best_microscope_xy")))
    frame_equivalence_values = dict(result.get("frames_to_match_best_xy", {}) or {})
    frame_equivalence_statuses = dict(result.get("frames_to_match_best_xy_status", {}) or {})
    for microscope_key, sigma_xy in result["ordering_xy"]:
        rec = result["per_microscope"][microscope_key]
        microscope = str(rec.get("microscope", microscope_key))
        modality = str(rec.get("modality", microscope_key))
        display_name = str(rec.get("display_name") or (
            microscope if microscope != modality else modality_display_name(modality)
        ))
        sigma_xy_float = float(sigma_xy)
        safe_for_ordering = bool(rec.get("safe_for_ordering", False))
        finite_positive_sigma = bool(np.isfinite(sigma_xy_float) and sigma_xy_float > 0.0)
        rankable_for_ordering = bool(safe_for_ordering and finite_positive_sigma)
        if rankable_for_ordering:
            # Rank is a user-facing scientific ordinal over sigma_xy_nm, not a
            # diagnostic row number. Equal finite scores are tied so equally
            # performing microscopes cannot be reported as better/worse.
            if (
                last_rankable_sigma_xy is not None
                and math.isclose(
                    sigma_xy_float,
                    last_rankable_sigma_xy,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
            ):
                rank_value = last_rank_value
            else:
                rank_value = next_rank
                last_rank_value = int(rank_value)
                last_rankable_sigma_xy = sigma_xy_float
                next_rank += 1
            ranking_status = "ranked"
        else:
            # A positive rank is a scientific recommendation ordinal, not a
            # diagnostic display index. Keep unsafe/non-finite rows visible in
            # microscope_ranking.csv, but do not let them masquerade as the best
            # microscope for a configured sample.
            rank_value = ""
            ranking_status = str(
                rec.get("status_reason")
                or "not rankable: unsafe, singular, or non-finite CRLB"
            )
        rows.append(
            {
                "rank": rank_value,
                "rankable_for_ordering": rankable_for_ordering,
                "rankable": rankable_for_ordering,
                "is_best_xy": bool(rankable_for_ordering and microscope_key in best_microscopes_xy),
                "ranking_status": ranking_status,
                "microscope": microscope,
                "modality": modality,
                "display_name": display_name,
                "sigma_xy_nm": sigma_xy_float,
                "sigma_x_nm": float(rec.get("sigma_x_nm", float("nan"))),
                "sigma_y_nm": float(rec.get("sigma_y_nm", float("nan"))),
                "latent_scene_id": rec.get("latent_scene_id", ""),
                "latent_schedule_id": rec.get("latent_schedule_id", ""),
                "state_time_policy": rec.get("state_time_policy", ""),
                "fusion_time_alignment": rec.get("fusion_time_alignment", ""),
                "shared_coordinate_frame": rec.get("shared_coordinate_frame", ""),
                "same_latent_scene": bool(rec.get("same_latent_scene", False)),
                "measurement_domain": rec.get("measurement_domain", ""),
                "signal_units": rec.get("signal_units", ""),
                "noise_variance_units": rec.get("noise_variance_units", ""),
                "detector_noise_input_domain": rec.get("detector_noise_input_domain", ""),
                "nonlinear_detector_effects_active": bool(rec.get("nonlinear_detector_effects_active", False)),
                "deterministic_detector_transfer_active": bool(rec.get("deterministic_detector_transfer_active", False)),
                "safe_for_linear_fisher_variance": bool(rec.get("safe_for_linear_fisher_variance", False)),
                "safe_for_covariance_fisher_variance": bool(rec.get("safe_for_covariance_fisher_variance", False)),
                "detector_safe_for_report_fisher": bool(rec.get("detector_safe_for_report_fisher", False)),
                "fisher_likelihood_uses_covariance": bool(rec.get("fisher_likelihood_uses_covariance", False)),
                "fisher_likelihood_eligibility_contract_id": rec.get("fisher_likelihood_eligibility_contract_id", ""),
                "fisher_variance_model_scope": rec.get("fisher_variance_model_scope", ""),
                "covariance_fisher_variance_model_scope": rec.get("covariance_fisher_variance_model_scope", ""),
                "detector_likelihood_status": rec.get("detector_likelihood_status", ""),
                "safe_for_ordering": safe_for_ordering,
                "safe_for_fusion": bool(rec.get("safe_for_fusion", False)),
                "status_reason": rec.get("status_reason", ""),
                "sequence_crlb_model": rec.get("sequence_crlb_model", ""),
                "same_state_assumption": bool(rec.get("same_state_assumption", False)),
                "frame_equivalence_model": rec.get("frame_equivalence_model", ""),
                "relative_sigma_xy": float(result["relative_sigma_xy"][microscope_key]),
                "frames_to_match_best_xy": _report_optional_float(
                    frame_equivalence_values.get(microscope_key, float("inf"))
                ),
                "frames_to_match_best_xy_status": frame_equivalence_statuses.get(microscope_key, ""),
                "num_frames": int(rec.get("num_frames", 1)),
                "fisher_xx": float(rec["fisher_matrix"][0, 0]),
                "fisher_xy": float(rec["fisher_matrix"][0, 1]),
                "fisher_yy": float(rec["fisher_matrix"][1, 1]),
                "singular": bool(rec.get("singular", False)),
                "derivative_basis": rec.get("derivative_basis", ""),
                "nyquist_band_fraction": _report_optional_float(
                    rec.get("nyquist_band_fraction")
                    if rec.get("nyquist_band_fraction") != ""
                    else None
                ),
                "boundary_energy_fraction": _report_optional_float(
                    rec.get("boundary_energy_fraction")
                    if rec.get("boundary_energy_fraction") != ""
                    else None
                ),
                "convergence_status": rec.get("convergence_status", ""),
            }
        )
    return rows

def _fusion_subset_sizes_for_report(
    eligible_microscope_count: int,
    *,
    max_k: int,
    include_full: bool,
) -> list[int]:
    """Return physically valid fusion subset sizes for report-table emission.

    The lower-level Fisher fusion API is intentionally strict: a fusion CRLB is
    defined only for at least one eligible microscope Fisher matrix.  The lab
    report has a broader public contract because valid configurations can leave
    every rendered microscope diagnostic-only, for example a multi-frame static
    sequence without the dynamic Bayesian estimator.  In that report state the
    correct user-facing output is an empty fusion table plus explicit exclusion
    reasons, not a forced subset_size=1 call with an empty Fisher dictionary.
    """

    n = int(eligible_microscope_count)
    if n < 0:
        raise ValueError(f"eligible_microscope_count must be nonnegative; got {eligible_microscope_count!r}.")
    if n == 0:
        return []
    capped = max(1, min(int(max_k), n))
    sizes = list(range(1, capped + 1))
    if include_full and n > 1 and capped < n:
        sizes.append(n)
    return sizes



def _fusion_rows_from_fisher_matrices(
    fisher_by_microscope: dict[str, np.ndarray],
    max_k: int,
    include_full: bool,
    microscope_profile_cards: dict[str, dict[str, Any]] | None = None,
    parent_result_metadata_by_microscope: dict[str, dict[str, Any]] | None = None,
    *,
    fusion_input_basis: str = "precomputed_sequence_fisher_matrix",
    fusion_frame_count: int | None = None,
    fisher_lateral_derivative_basis: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(fisher_by_microscope)
    parent_metadata = (
        {}
        if parent_result_metadata_by_microscope is None
        else dict(parent_result_metadata_by_microscope)
    )
    fusion_candidates = [
        FisherMatrixCandidate(
            key=microscope_name,
            fisher_matrix=fisher_matrix,
            parent_result_metadata=parent_metadata.get(microscope_name, {}),
        )
        for microscope_name, fisher_matrix in fisher_by_microscope.items()
    ]

    def _modality_for_microscope(microscope_name: str) -> str:
        metadata = parent_metadata.get(microscope_name, {})
        if isinstance(metadata, dict):
            value = metadata.get("modality") or metadata.get("canonical_modality_name")
            if value:
                return str(value)
        card = (microscope_profile_cards or {}).get(microscope_name, {})
        if isinstance(card, dict):
            value = card.get("canonical_modality_name") or card.get("modality")
            if value:
                return str(value)
        return str(microscope_name)

    def _row_for_subset_size(k: int) -> dict[str, Any]:
        result = compute_candidate_fusion_crlb_from_fisher_matrices(
            fusion_candidates,
            subset_size=k,
            candidate_profile_cards=microscope_profile_cards,
        )
        complementarity = result.get("fusion_complementarity", {}) or {}
        used_microscopes = [str(candidate) for candidate in result["candidates_used"]]
        # Fusion algebra is candidate-keyed. This report adds physical modality
        # labels only as metadata for readers; they are not the comparison keys.
        used_modalities = [_modality_for_microscope(name) for name in used_microscopes]
        return {
            "subset_size": k,
            "microscopes_used": ";".join(used_microscopes),
            "modalities_used": ";".join(used_modalities),
            "fusion_sigma_xy_nm": float(result["fusion_sigma_xy_nm"]),
            "fusion_gain_xy": result.get("fusion_gain_xy", ""),
            "mean_principal_angle_deg": complementarity.get("mean_principal_angle_deg", ""),
            "determinant_gain_vs_best_single": complementarity.get("determinant_gain_vs_best_single", ""),
            "fusion_singular": bool(result.get("fusion_singular", False)),
            "fusion_interpretation": result.get("fusion_interpretation", ""),
            "physical_compatibility_status": result.get("physical_compatibility_status", ""),
            "fusion_validation_status": result.get("fusion_validation_status", ""),
            "safe_for_fusion": bool(result.get("safe_for_fusion", False)),
            "production_grid_diagnostic": bool(result.get("production_grid_diagnostic", False)),
            "fusion_input_basis": fusion_input_basis,
            "fusion_frame_count": "" if fusion_frame_count is None else int(fusion_frame_count),
            "fisher_lateral_derivative_basis": fisher_lateral_derivative_basis,
        }

    for k in _fusion_subset_sizes_for_report(n, max_k=max_k, include_full=include_full):
        rows.append(_row_for_subset_size(k))
    return rows


def _select_fusion_fisher_inputs(
    microscope_result: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    """Return fusion-eligible microscope Fisher matrices and exclusions."""
    selected: dict[str, np.ndarray] = {}
    excluded: dict[str, dict[str, str]] = {}
    per_microscope = microscope_result.get("per_microscope", {})
    for microscope_name, rec in per_microscope.items():
        if not bool(rec.get("safe_for_fusion", False)):
            excluded[microscope_name] = {
                "representative": "",
                "reason": str(
                    rec.get(
                        "status_reason",
                        "microscope Fisher summary is not safe for fusion",
                    )
                ),
            }
            continue
        fisher = np.asarray(rec.get("fisher_matrix"), dtype=float)
        if fisher.shape != (2, 2) or not np.all(np.isfinite(fisher)):
            excluded[microscope_name] = {
                "representative": "",
                "reason": "microscope Fisher matrix is not a finite 2x2 lateral matrix",
            }
            continue
        selected[microscope_name] = 0.5 * (fisher + fisher.T)

    return selected, excluded


def _dynamic_sequence_metadata_compatible(
    microscope_names: tuple[str, ...],
    dynamic_summary_by_microscope: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Return whether dynamic summaries describe the same Brownian estimator."""

    if not microscope_names:
        return False, "empty dynamic fusion subset"
    reference = dynamic_summary_by_microscope[microscope_names[0]]
    scalar_keys = ("fps", "state_transition_fps", "process_noise_fps", "process_model")
    matrix_keys = ("state_transition_matrix", "process_noise_covariance", "initial_covariance")
    axes = tuple(str(axis) for axis in reference.get("state_axes", []))
    if not axes:
        return False, f"{microscope_names[0]} dynamic summary does not declare state_axes"
    for microscope_name in microscope_names[1:]:
        candidate = dynamic_summary_by_microscope[microscope_name]
        candidate_axes = tuple(str(axis) for axis in candidate.get("state_axes", []))
        if candidate_axes != axes:
            return False, "dynamic fusion requires identical state axes across microscopes"
        for key in scalar_keys:
            left = reference.get(key)
            right = candidate.get(key)
            if left is None or right is None:
                if left != right:
                    return False, f"dynamic fusion requires matching {key}"
                continue
            if key == "process_model":
                if str(left) != str(right):
                    return False, "dynamic fusion requires matching process_model"
            elif not np.isclose(float(left), float(right), rtol=1e-12, atol=0.0):
                return False, f"dynamic fusion requires matching {key}"
        for key in matrix_keys:
            left_arr = np.asarray(reference.get(key), dtype=float)
            right_arr = np.asarray(candidate.get(key), dtype=float)
            if left_arr.shape != right_arr.shape or not np.allclose(left_arr, right_arr, rtol=1e-12, atol=1e-15):
                return False, f"dynamic fusion requires matching {key}"
    return True, ""


def _dynamic_sequence_subset_compatible(
    microscope_names: tuple[str, ...],
    dynamic_summary_by_microscope: dict[str, dict[str, Any]],
    eligible_sequences: dict[str, list[np.ndarray]],
) -> tuple[bool, str]:
    """Return whether a selected dynamic fusion subset can share one filter."""

    compatible, reason = _dynamic_sequence_metadata_compatible(
        microscope_names,
        dynamic_summary_by_microscope,
    )
    if not compatible:
        return False, reason
    if not microscope_names:
        return False, "empty dynamic fusion subset"
    reference_frame_count = len(eligible_sequences[microscope_names[0]])
    for microscope_name in microscope_names[1:]:
        frame_count = len(eligible_sequences[microscope_name])
        if frame_count != reference_frame_count:
            return (
                False,
                "dynamic fusion requires the same frame count across selected microscopes",
            )
    return True, ""


def _dynamic_fusion_rows_from_fisher_sequences(
    per_frame_fisher_by_microscope: dict[str, list[np.ndarray]],
    dynamic_summary_by_microscope: dict[str, dict[str, Any]],
    ranking_summary_by_microscope: dict[str, dict[str, Any]],
    max_k: int,
    include_full: bool,
    microscope_profile_cards: dict[str, dict[str, Any]] | None = None,
    *,
    fisher_lateral_derivative_basis: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, list[np.ndarray]]]:
    """Fuse dynamic sequences by summing per-frame measurement Fisher first.

    A per-microscope dynamic posterior already includes the shared Brownian
    transition and initial prior. Those posteriors are valid ranking summaries,
    but they are not independent Fisher channels. Dynamic fusion must combine
    the measurement Fisher matrices at each frame and then run one estimator.
    """

    eligible_sequences: dict[str, list[np.ndarray]] = {}
    excluded: dict[str, dict[str, str]] = {}
    for microscope_name, ranking_summary in ranking_summary_by_microscope.items():
        if str(ranking_summary.get("sequence_crlb_model", "")) != "dynamic_bayesian_estimator":
            continue
        if not bool(ranking_summary.get("safe_for_ordering", False)):
            excluded[microscope_name] = {
                "representative": "",
                "reason": str(
                    ranking_summary.get(
                        "status_reason",
                        "dynamic Bayesian sequence is not safe for ranking or fusion",
                    )
                ),
            }
            continue
        dynamic_summary = dynamic_summary_by_microscope.get(microscope_name, {})
        if not bool(dynamic_summary.get("dynamic_enabled", False)):
            excluded[microscope_name] = {
                "representative": "",
                "reason": "dynamic Bayesian estimator was not enabled for this microscope",
            }
            continue
        matrices = [
            0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
            for matrix in per_frame_fisher_by_microscope.get(microscope_name, [])
        ]
        if not matrices:
            excluded[microscope_name] = {
                "representative": "",
                "reason": "dynamic fusion requires per-frame measurement Fisher matrices",
            }
            continue
        if any(matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)) for matrix in matrices):
            excluded[microscope_name] = {
                "representative": "",
                "reason": "dynamic fusion requires finite 2x2 per-frame lateral Fisher matrices",
            }
            continue
        eligible_sequences[microscope_name] = matrices

    eligible_names = list(eligible_sequences)
    if not eligible_names:
        return [], excluded, eligible_sequences

    parent_metadata = {
        microscope_name: dict(ranking_summary_by_microscope.get(microscope_name, {}))
        for microscope_name in eligible_names
    }

    def _modality_for_microscope(microscope_name: str) -> str:
        metadata = parent_metadata.get(microscope_name, {})
        value = metadata.get("modality") or metadata.get("canonical_modality_name")
        if value:
            return str(value)
        card = (microscope_profile_cards or {}).get(microscope_name, {})
        if isinstance(card, dict):
            value = card.get("canonical_modality_name") or card.get("modality")
            if value:
                return str(value)
        return str(microscope_name)

    def _row_candidate_for_subset(subset: tuple[str, ...]) -> dict[str, Any] | None:
        compatible, reason = _dynamic_sequence_subset_compatible(
            subset,
            dynamic_summary_by_microscope,
            eligible_sequences,
        )
        if not compatible:
            return None
        reference = dynamic_summary_by_microscope[subset[0]]
        summed_sequence = [
            sum(eligible_sequences[microscope_name][frame_index] for microscope_name in subset)
            for frame_index in range(len(eligible_sequences[subset[0]]))
        ]
        dynamic = compute_dynamic_bayesian_crlb_from_fisher_sequence(
            summed_sequence,
            np.asarray(reference["process_noise_covariance"], dtype=float),
            state_axes=tuple(str(axis) for axis in reference.get("state_axes", ["x", "y"])),
            state_transition_matrix=np.asarray(reference["state_transition_matrix"], dtype=float),
            state_transition_fps=reference.get("state_transition_fps"),
            process_noise_fps=reference.get("process_noise_fps"),
            fps=reference.get("fps"),
            initial_covariance=np.asarray(reference["initial_covariance"], dtype=float),
            include_smoothing=False,
            measurement_domain=reference.get("measurement_domain"),
            signal_units=reference.get("signal_units"),
            noise_variance_units=reference.get("noise_variance_units"),
            state_axis_units=reference.get("state_axis_units"),
            process_model=reference.get("process_model", "brownian_translation_lateral_xy"),
            dynamic_validation_status="joint_dynamic_bayesian_fusion_lateral_xy",
            include_fisher_matrices=True,
        )
        covariances = dynamic.get("dynamic_covariance_matrices") or []
        final_covariance = np.asarray(covariances[-1], dtype=float) if covariances else np.full((2, 2), float("nan"))
        fusion_fisher = _safe_inverse_covariance_for_ranking(final_covariance)
        crlb = _crlb_from_fisher_matrix(fusion_fisher)
        physical_metadata = _fusion_subset_metadata_for_precomputed_matrices(
            list(subset),
            microscope_profile_cards,
            parent_metadata,
        )
        subset_parent_status = combine_parent_statuses(
            {
                microscope_name: parent_metadata[microscope_name]
                for microscope_name in subset
                if microscope_name in parent_metadata
            }
        )
        physical_fusion_allowed = bool(physical_metadata["physically_feasible_fusion_allowed"])
        singular = bool(crlb.get("singular", True))
        safe_for_fusion = (
            bool(subset_parent_status["safe_for_ordering"])
            and physical_fusion_allowed
            and not singular
        )
        validation_status = subset_parent_status["validation_status"]
        production_grid_diagnostic = bool(subset_parent_status["production_grid_diagnostic"])
        if not safe_for_fusion:
            validation_status = ValidationStatus.DIAGNOSTIC_ONLY.value
            production_grid_diagnostic = True
        return {
            "subset_size": len(subset),
            "microscopes_used": ";".join(str(name) for name in subset),
            "modalities_used": ";".join(_modality_for_microscope(name) for name in subset),
            "fusion_sigma_xy_nm": float(crlb["sigma_xy_nm"]),
            "fusion_gain_xy": "",
            "mean_principal_angle_deg": "",
            "determinant_gain_vs_best_single": "",
            "fusion_singular": singular,
            "fusion_interpretation": (
                f"{physical_metadata.get('fusion_mode', '')};joint_dynamic_bayesian_filter"
            ),
            "physical_compatibility_status": physical_metadata.get("physical_compatibility_status", ""),
            "fusion_validation_status": validation_status,
            "safe_for_fusion": safe_for_fusion,
            "production_grid_diagnostic": production_grid_diagnostic,
            "fusion_input_basis": "joint_dynamic_bayesian_per_frame_measurement_fisher_sequence",
            "fusion_frame_count": int(len(summed_sequence)),
            "fisher_lateral_derivative_basis": fisher_lateral_derivative_basis,
        }

    rows_by_size: list[dict[str, Any]] = []
    best_single_sigma = float("inf")
    for k in _fusion_subset_sizes_for_report(len(eligible_names), max_k=max_k, include_full=include_full):
        best_row: dict[str, Any] | None = None
        incompatible_reasons: list[str] = []
        for subset in combinations(eligible_names, k):
            compatible, reason = _dynamic_sequence_subset_compatible(
                subset,
                dynamic_summary_by_microscope,
                eligible_sequences,
            )
            if not compatible:
                incompatible_reasons.append(reason)
                continue
            row = _row_candidate_for_subset(subset)
            if row is None:
                continue
            sigma = float(row["fusion_sigma_xy_nm"])
            if best_row is None:
                best_row = row
            elif np.isfinite(sigma) and sigma > 0.0 and (
                not np.isfinite(float(best_row["fusion_sigma_xy_nm"]))
                or sigma < float(best_row["fusion_sigma_xy_nm"])
            ):
                best_row = row
        if best_row is None:
            reason = sorted(set(incompatible_reasons))[0] if incompatible_reasons else "no compatible dynamic fusion subset"
            rows_by_size.append(
                {
                    "subset_size": k,
                    "microscopes_used": "",
                    "modalities_used": "",
                    "fusion_sigma_xy_nm": float("inf"),
                    "fusion_gain_xy": "",
                    "mean_principal_angle_deg": "",
                    "determinant_gain_vs_best_single": "",
                    "fusion_singular": True,
                    "fusion_interpretation": reason,
                    "physical_compatibility_status": "dynamic_fusion_incompatible",
                    "fusion_validation_status": ValidationStatus.DIAGNOSTIC_ONLY.value,
                    "safe_for_fusion": False,
                    "production_grid_diagnostic": True,
                    "fusion_input_basis": "joint_dynamic_bayesian_per_frame_measurement_fisher_sequence",
                    "fusion_frame_count": "",
                    "fisher_lateral_derivative_basis": fisher_lateral_derivative_basis,
                }
            )
            continue
        if k == 1:
            sigma = float(best_row["fusion_sigma_xy_nm"])
            if np.isfinite(sigma) and sigma > 0.0:
                best_single_sigma = min(best_single_sigma, sigma)
        rows_by_size.append(best_row)

    for row in rows_by_size:
        sigma = float(row["fusion_sigma_xy_nm"])
        if np.isfinite(best_single_sigma) and best_single_sigma > 0.0 and np.isfinite(sigma) and sigma > 0.0:
            row["fusion_gain_xy"] = float(best_single_sigma / sigma)
    return rows_by_size, excluded, eligible_sequences


def _safe_inverse_covariance_for_ranking(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return np.full((2, 2), float("nan"))
    symmetric = 0.5 * (cov + cov.T)
    try:
        return np.linalg.pinv(symmetric)
    except np.linalg.LinAlgError:
        return np.full((2, 2), float("nan"))


def _build_sequence_summary_rows(
    modality: str,
    per_frame_records: list[dict[str, Any]],
    *,
    microscope: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_frame_fishers = [record["fisher_matrix"] for record in per_frame_records]
    if not per_frame_fishers:
        raise ValueError(f"modality {modality!r} returned no frame Fisher matrices.")
    microscope_name = str(microscope or modality)

    cumulative_fishers, _cumulative_covariances, cumulative_ranks = sequence_sum_fisher_to_crlb(
        per_frame_fishers
    )
    frame_rows: list[dict[str, Any]] = []
    final_crlb = _crlb_from_fisher_matrix(cumulative_fishers[-1])
    final_crlb["microscope"] = microscope_name
    final_crlb["modality"] = modality
    final_crlb["num_frames"] = len(per_frame_fishers)
    final_crlb["cumulative_final_rank"] = int(cumulative_ranks[-1])
    sequence_is_single_frame = len(per_frame_fishers) == 1
    final_crlb["sequence_crlb_model"] = (
        "single_frame_static"
        if sequence_is_single_frame
        else "static_same_state_cumulative_diagnostic"
    )
    alignment_values = [
        str(record.get("fusion_time_alignment", "")).strip()
        for record in per_frame_records
        if str(record.get("fusion_time_alignment", "")).strip()
    ]
    fusion_time_alignment = alignment_values[-1] if alignment_values else "coincident"
    same_state_assumption = fusion_time_alignment == "coincident"
    final_crlb["same_state_assumption"] = bool(same_state_assumption)
    final_crlb["fusion_time_alignment"] = fusion_time_alignment
    # Frame-equivalence is a report-facing acquisition-time claim, not a
    # generic Fisher metadata field.  It is finite only for static same-state
    # inverse-sqrt-N scaling; dynamic Brownian estimators set a different model
    # below so the ranking table cannot silently reuse this static law.
    final_crlb["frame_equivalence_model"] = STATIC_FRAME_EQUIVALENCE_MODEL
    final_crlb["safe_for_dynamic_sequence_claim"] = sequence_is_single_frame
    final_crlb["safe_for_ordering"] = (
        sequence_is_single_frame
        and same_state_assumption
        and bool(per_frame_records[-1].get("safe_for_ordering", False))
    )
    final_crlb["safe_for_fusion"] = (
        sequence_is_single_frame
        and same_state_assumption
        and bool(per_frame_records[-1].get("safe_for_fusion", False))
    )
    if not same_state_assumption:
        final_crlb["status_reason"] = (
            "static ranking/fusion requires coincident latent state times; "
            f"this report schedule is {fusion_time_alignment}."
        )
    elif not sequence_is_single_frame:
        final_crlb["status_reason"] = (
            "multi-frame static same-state cumulative Fisher is diagnostic only; "
            "Brownian/dynamic sequences require the dynamic Bayesian estimator "
            "for ranking or fusion."
        )
    derivative_basis_values = [
        str(record.get("derivative_basis", "")).strip()
        for record in per_frame_records
        if str(record.get("derivative_basis", "")).strip()
    ]
    final_crlb["derivative_basis"] = derivative_basis_values[-1] if derivative_basis_values else ""
    for diagnostic_key in ("nyquist_band_fraction", "boundary_energy_fraction"):
        values: list[float] = []
        for record in per_frame_records:
            if diagnostic_key not in record or record[diagnostic_key] in {"", None}:
                continue
            try:
                value = float(record[diagnostic_key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        final_crlb[diagnostic_key] = max(values) if values else ""
    convergence_values = [
        str(record.get("convergence_status", "")).strip()
        for record in per_frame_records
        if str(record.get("convergence_status", "")).strip()
    ]
    final_crlb["convergence_status"] = convergence_values[-1] if convergence_values else ""
    for key in (
        "latent_scene_id",
        "latent_schedule_id",
        "state_time_policy",
        "fusion_time_alignment",
        "shared_coordinate_frame",
        "same_latent_scene",
        "measurement_domain",
        "signal_units",
        "noise_variance_units",
        "detector_noise_input_domain",
        "nonlinear_detector_effects_active",
        "deterministic_detector_transfer_active",
        "safe_for_linear_fisher_variance",
        "safe_for_covariance_fisher_variance",
        "detector_safe_for_report_fisher",
        "fisher_likelihood_uses_covariance",
        "fisher_likelihood_eligibility_contract_id",
        "fisher_variance_model_scope",
        "covariance_fisher_variance_model_scope",
        "detector_likelihood_status",
        "fisher_noise_covariance_model",
    ):
        if key in per_frame_records[-1]:
            final_crlb[key] = per_frame_records[-1][key]

    for frame_idx, (frame_fisher, cumulative_fisher, cumulative_rank) in enumerate(
        zip(per_frame_fishers, cumulative_fishers, cumulative_ranks)
    ):
        frame_crlb = _crlb_from_fisher_matrix(frame_fisher)
        cumulative_crlb = _crlb_from_fisher_matrix(cumulative_fisher)
        frame_rows.append(
            {
                "microscope": microscope_name,
                "modality": modality,
                "frame_index": frame_idx,
                "num_frames": len(per_frame_fishers),
                "latent_scene_id": per_frame_records[frame_idx].get("latent_scene_id", ""),
                "latent_schedule_id": per_frame_records[frame_idx].get("latent_schedule_id", ""),
                "observation_time_s": per_frame_records[frame_idx].get("observation_time_s", ""),
                "state_time_policy": per_frame_records[frame_idx].get("state_time_policy", ""),
                "fusion_time_alignment": per_frame_records[frame_idx].get("fusion_time_alignment", ""),
                "shared_coordinate_frame": per_frame_records[frame_idx].get("shared_coordinate_frame", ""),
                "same_latent_scene": bool(per_frame_records[frame_idx].get("same_latent_scene", False)),
                "sequence_crlb_model": final_crlb["sequence_crlb_model"],
                "same_state_assumption": bool(same_state_assumption),
                "safe_for_dynamic_sequence_claim": sequence_is_single_frame,
                "safe_for_ordering": bool(final_crlb["safe_for_ordering"]),
                "safe_for_fusion": bool(final_crlb["safe_for_fusion"]),
                "frame_fisher_xx": frame_crlb["fisher_xx"],
                "frame_fisher_xy": frame_crlb["fisher_xy"],
                "frame_fisher_yy": frame_crlb["fisher_yy"],
                "frame_fisher_det": frame_crlb["fisher_det"],
                "frame_fisher_singular": bool(frame_crlb.get("singular", False)),
                "frame_fisher_rank": int(frame_crlb.get("fisher_rank", 0)),
                "frame_sigma_x_nm": frame_crlb["sigma_x_nm"],
                "frame_sigma_y_nm": frame_crlb["sigma_y_nm"],
                "frame_sigma_xy_nm": frame_crlb["sigma_xy_nm"],
                "measurement_domain": per_frame_records[frame_idx].get("measurement_domain", ""),
                "signal_units": per_frame_records[frame_idx].get("signal_units", ""),
                "noise_variance_units": per_frame_records[frame_idx].get("noise_variance_units", ""),
                "detector_noise_input_domain": per_frame_records[frame_idx].get("detector_noise_input_domain", ""),
                "nonlinear_detector_effects_active": bool(per_frame_records[frame_idx].get("nonlinear_detector_effects_active", False)),
                "deterministic_detector_transfer_active": bool(per_frame_records[frame_idx].get("deterministic_detector_transfer_active", False)),
                "safe_for_linear_fisher_variance": bool(per_frame_records[frame_idx].get("safe_for_linear_fisher_variance", False)),
                "safe_for_covariance_fisher_variance": bool(per_frame_records[frame_idx].get("safe_for_covariance_fisher_variance", False)),
                "detector_safe_for_report_fisher": bool(per_frame_records[frame_idx].get("detector_safe_for_report_fisher", False)),
                "fisher_likelihood_uses_covariance": bool(per_frame_records[frame_idx].get("fisher_likelihood_uses_covariance", False)),
                "fisher_likelihood_eligibility_contract_id": per_frame_records[frame_idx].get("fisher_likelihood_eligibility_contract_id", ""),
                "fisher_variance_model_scope": per_frame_records[frame_idx].get("fisher_variance_model_scope", ""),
                "covariance_fisher_variance_model_scope": per_frame_records[frame_idx].get("covariance_fisher_variance_model_scope", ""),
                "detector_likelihood_status": per_frame_records[frame_idx].get("detector_likelihood_status", ""),
                "derivative_basis": per_frame_records[frame_idx].get("derivative_basis", ""),
                "nyquist_band_fraction": per_frame_records[frame_idx].get("nyquist_band_fraction", ""),
                "boundary_energy_fraction": per_frame_records[frame_idx].get("boundary_energy_fraction", ""),
                "convergence_status": per_frame_records[frame_idx].get("convergence_status", ""),
                "cumulative_fisher_xx": cumulative_crlb["fisher_xx"],
                "cumulative_fisher_xy": cumulative_crlb["fisher_xy"],
                "cumulative_fisher_yy": cumulative_crlb["fisher_yy"],
                "cumulative_fisher_det": cumulative_crlb["fisher_det"],
                "cumulative_fisher_rank": int(cumulative_rank),
                "cumulative_sigma_x_nm": cumulative_crlb["sigma_x_nm"],
                "cumulative_sigma_y_nm": cumulative_crlb["sigma_y_nm"],
                "cumulative_sigma_xy_nm": cumulative_crlb["sigma_xy_nm"],
            }
        )
    return frame_rows, final_crlb


def _dynamic_sequence_summary_to_ranking_summary(
    dynamic_summary: dict[str, Any],
    static_summary: dict[str, Any],
) -> dict[str, Any]:
    """Convert final dynamic Bayesian covariance into the report ranking contract."""
    if not bool(dynamic_summary.get("dynamic_enabled", False)):
        return dict(static_summary)
    covariances = dynamic_summary.get("dynamic_covariance") or dynamic_summary.get(
        "dynamic_covariance_matrices"
    )
    if not covariances:
        out = dict(static_summary)
        out["safe_for_ordering"] = False
        out["safe_for_fusion"] = False
        out["status_reason"] = str(
            dynamic_summary.get(
                "dynamic_error",
                "dynamic Bayesian summary did not include final covariance",
            )
        )
        return out

    covariance = np.asarray(covariances[-1], dtype=float)
    fisher = _safe_inverse_covariance_for_ranking(covariance)
    out = _crlb_from_fisher_matrix(fisher)
    out.update(
        {
            "microscope": static_summary.get("microscope", dynamic_summary.get("microscope", "")),
            "modality": static_summary.get("modality", dynamic_summary.get("modality", "")),
            "display_name": static_summary.get("display_name", ""),
            "num_frames": int(static_summary.get("num_frames", len(covariances))),
            "sequence_crlb_model": "dynamic_bayesian_estimator",
            "same_state_assumption": False,
            # The dynamic estimator ranks the current Brownian state after a
            # Kalman-style prediction/update sequence.  Its sigma_xy_nm is valid
            # for ordering, but it has no static inverse-sqrt-N frame-equivalence
            # unless a separate dynamic frame-count solver is implemented.
            "frame_equivalence_model": DYNAMIC_FRAME_EQUIVALENCE_MODEL,
            "safe_for_dynamic_sequence_claim": True,
            "safe_for_ordering": not bool(out.get("singular", True)),
            "safe_for_fusion": False,
            "safe_for_dynamic_joint_fusion_source": not bool(out.get("singular", True)),
            "status_reason": (
                "dynamic Bayesian sequence covariance used for ranking; fusion "
                "must run one joint dynamic estimator over summed per-frame "
                "measurement Fisher matrices"
            ),
        }
    )
    for key in (
        "measurement_domain",
        "signal_units",
        "noise_variance_units",
        "detector_noise_input_domain",
        "nonlinear_detector_effects_active",
        "deterministic_detector_transfer_active",
        "safe_for_linear_fisher_variance",
        "safe_for_covariance_fisher_variance",
        "detector_safe_for_report_fisher",
        "fisher_likelihood_uses_covariance",
        "fisher_likelihood_eligibility_contract_id",
        "fisher_variance_model_scope",
        "covariance_fisher_variance_model_scope",
        "detector_likelihood_status",
        "fisher_noise_covariance_model",
        "latent_scene_id",
        "latent_schedule_id",
        "state_time_policy",
        "fusion_time_alignment",
        "shared_coordinate_frame",
        "same_latent_scene",
        *FISHER_DIAGNOSTIC_FIELDS,
    ):
        if key in static_summary:
            out[key] = static_summary[key]
    # The dynamic estimator changes the temporal inference model, not the
    # detector-transfer likelihood.  Reuse the shared eligibility contract so
    # row-covariance exceptions and diagnostic-only transfer states cannot
    # diverge from static report and matched-packet paths.
    eligibility = resolve_fisher_likelihood_eligibility(
        {
            "safe_for_ordering": not bool(out.get("singular", True)),
            "safe_for_fusion": False,
            "status_reason": out.get("status_reason", ""),
        },
        out,
        out,
        context="dynamic Bayesian report summary",
    )
    out["detector_safe_for_report_fisher"] = bool(eligibility.detector_safe_for_report_fisher)
    out["fisher_likelihood_uses_covariance"] = bool(eligibility.used_covariance_fisher)
    out["fisher_likelihood_eligibility_contract_id"] = eligibility.contract_id
    if not eligibility.safe_for_ordering:
        out["safe_for_ordering"] = False
        out["safe_for_fusion"] = False
        out["safe_for_dynamic_joint_fusion_source"] = False
        out["status_reason"] = eligibility.status_reason
    fusion_time_alignment = str(out.get("fusion_time_alignment", "")).strip()
    if fusion_time_alignment and fusion_time_alignment != "coincident":
        out["safe_for_ordering"] = False
        out["safe_for_fusion"] = False
        out["safe_for_dynamic_joint_fusion_source"] = False
        out["status_reason"] = (
            "dynamic Bayesian ranking/fusion requires an event-time asynchronous "
            f"estimator for non-coincident schedules; this schedule is {fusion_time_alignment}."
        )
    out["convergence_status"] = (
        ConvergenceStatus.STABLE_SINGULAR.value
        if bool(out.get("singular", True))
        else (
            ConvergenceStatus.FINITE_CONVERGED.value
            if bool(out.get("safe_for_ordering", False))
            else ConvergenceStatus.PRODUCTION_GRID_ONLY.value
        )
    )
    out["validation_status"] = (
        ValidationStatus.VALIDATED.value
        if bool(out.get("safe_for_ordering", False))
        else ValidationStatus.DIAGNOSTIC_ONLY.value
    )
    return out


def _compute_dynamic_sequence_summary(
    modality: str,
    per_frame_fisher: list[np.ndarray],
    params: dict[str, Any],
    per_frame_records: list[dict[str, Any]] | None = None,
    *,
    microscope: str | None = None,
) -> dict[str, Any]:
    if not per_frame_fisher:
        raise ValueError(f"no Fisher matrices for modality {modality!r}")
    if len(per_frame_fisher) < 2:
        return {
            "microscope": str(microscope or modality),
            "modality": modality,
            "dynamic_enabled": False,
            "dynamic_reason": "sequence length < 2; dynamic path is vacuous.",
            "dynamic_crlb_final": [],
            "dynamic_improvement_final": [],
            "dynamic_ranks": [],
            "static_ranks": [],
        }

    diameters = resolve_translational_diameters_nm(params)
    if len(diameters) != 1:
        raise ValueError(
            "dynamic Bayesian CRLB currently requires exactly one hydrodynamic diameter for this report workflow."
        )
    diameter_nm = float(diameters[0])
    dynamics = MotionDynamicsSettings.from_params(params)
    D = stokes_einstein_diffusion_coefficient(
        diameter_nm,
        dynamics.temperature_K,
        dynamics.viscosity_Pa_s,
    )
    if not np.isfinite(D) or D < 0.0:
        raise ValueError(f"invalid Brownian diffusion coefficient from particle size={diameter_nm} nm.")

    process_scale = dynamics.dynamic_process_noise_scale
    acquisition = AcquisitionProfile.from_params(params)
    process_covariance = build_brownian_process_covariance(
        ("x", "y"),
        fps=acquisition.fps,
        translational_diffusion_coeff_m2_s=float(D) * process_scale,
    )
    initial_covariance = np.eye(2, dtype=float) * dynamics.dynamic_initial_variance_nm2
    first_record = (
        per_frame_records[0]
        if per_frame_records and isinstance(per_frame_records[0], dict)
        else {}
    )
    measurement_domain = str(first_record.get("measurement_domain", "contrast"))
    signal_units = str(
        first_record.get("signal_units", "relative_reference_or_modality_analysis_units")
    )
    noise_variance_units = str(
        first_record.get("noise_variance_units", f"{signal_units}^2")
    )
    summary = compute_dynamic_bayesian_crlb_from_fisher_sequence(
        per_frame_fisher,
        process_covariance,
        state_transition_fps=acquisition.fps,
        process_noise_fps=acquisition.fps,
        fps=acquisition.fps,
        initial_covariance=initial_covariance,
        initial_variance_fallback=dynamics.dynamic_initial_variance_nm2,
        include_smoothing=dynamics.dynamic_include_smoothing,
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units,
        state_axis_units={"x": "nm", "y": "nm"},
        process_model="brownian_translation_lateral_xy",
        dynamic_validation_status="implemented_estimator_layer_lateral_xy",
        include_fisher_matrices=True,
    )
    summary["microscope"] = str(microscope or modality)
    summary["modality"] = modality
    summary["dynamic_enabled"] = True
    summary["dynamic_scope"] = "lateral_xy"
    summary["translational_diffusion_coeff_m2_s"] = float(D)
    summary["dynamic_process_noise_scale"] = float(process_scale)
    return summary
