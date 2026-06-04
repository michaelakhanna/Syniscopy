"""Numeric table builders for lab Fisher reports."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from common_utils import init_infinite_dict
from config import param_value
from fisher import (
    build_brownian_process_covariance,
    compute_dynamic_bayesian_crlb_from_fisher_sequence,
    compute_modality_fusion_crlb,
    sequence_sum_fisher_to_crlb,
)
from modality_registry import modality_display_name
from trajectory import stokes_einstein_diffusion_coefficient, resolve_translational_diameters_nm

__all__ = [
    "_build_sequence_summary_rows",
    "_compute_dynamic_sequence_summary",
    "_crlb_from_fisher_matrix",
    "_fusion_rows",
    "_ranking_rows",
    "_select_fusion_inputs",
    "_sequence_information_content",
]


def _crlb_from_fisher_matrix(fisher_matrix: np.ndarray) -> dict[str, Any]:
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
    if not np.all(np.isfinite(symmetric)):
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "fisher_xx": float("inf"),
            "fisher_xy": float("nan"),
            "fisher_yy": float("nan"),
            "fisher_det": float("nan"),
            "fisher_matrix": symmetric,
            "fisher_rank": 0,
            "singular": True,
        }

    det = float(np.linalg.det(symmetric))
    if det <= 0.0 or det <= 1.0e-30:
        return {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "fisher_xx": float(symmetric[0, 0]),
            "fisher_xy": float(symmetric[0, 1]),
            "fisher_yy": float(symmetric[1, 1]),
            "fisher_det": det,
            "fisher_matrix": symmetric,
            "fisher_rank": int(np.linalg.matrix_rank(symmetric)),
            "singular": True,
        }

    try:
        cov = np.linalg.inv(symmetric)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(symmetric)

    sigma_x2 = float(cov[0, 0])
    sigma_y2 = float(cov[1, 1])
    if not np.isfinite(sigma_x2) or not np.isfinite(sigma_y2):
        sigma_x2 = float("inf")
        sigma_y2 = float("inf")

    sigma_x = math.sqrt(max(sigma_x2, 0.0)) if np.isfinite(sigma_x2) else float("inf")
    sigma_y = math.sqrt(max(sigma_y2, 0.0)) if np.isfinite(sigma_y2) else float("inf")
    sigma_xy = float(math.inf if sigma_x == float("inf") or sigma_y == float("inf") else math.hypot(sigma_x, sigma_y))

    return {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_xy_nm": sigma_xy,
        "fisher_xx": float(symmetric[0, 0]),
        "fisher_xy": float(symmetric[0, 1]),
        "fisher_yy": float(symmetric[1, 1]),
        "fisher_det": det,
        "fisher_matrix": symmetric,
        "fisher_rank": int(np.linalg.matrix_rank(symmetric)),
        "singular": False,
    }


def _sequence_information_content(
    modality_order: list[str],
    final_fisher_by_modality: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not final_fisher_by_modality:
        raise ValueError("No modality summaries were provided for comparison.")

    per_modality: dict[str, dict[str, Any]] = {}
    for modality in modality_order:
        if modality not in final_fisher_by_modality:
            continue
        summary = dict(final_fisher_by_modality[modality])
        per_modality[modality] = {
            "sigma_x_nm": float(summary.get("sigma_x_nm", float("inf"))),
            "sigma_y_nm": float(summary.get("sigma_y_nm", float("inf"))),
            "sigma_xy_nm": float(summary.get("sigma_xy_nm", float("inf"))),
            "measurement_domain": str(summary.get("measurement_domain", "")),
            "signal_units": str(summary.get("signal_units", "")),
            "noise_variance_units": str(summary.get("noise_variance_units", "")),
            "detector_noise_input_domain": str(summary.get("detector_noise_input_domain", "")),
            "nonlinear_detector_effects_active": bool(summary.get("nonlinear_detector_effects_active", False)),
            "deterministic_detector_transfer_active": bool(summary.get("deterministic_detector_transfer_active", False)),
            "safe_for_linear_fisher_variance": bool(summary.get("safe_for_linear_fisher_variance", True)),
            "fisher_variance_model_scope": str(summary.get("fisher_variance_model_scope", "")),
            "detector_likelihood_status": str(summary.get("detector_likelihood_status", "")),
            "fisher_xx": float(summary.get("fisher_xx", float("nan"))),
            "fisher_xy": float(summary.get("fisher_xy", float("nan"))),
            "fisher_yy": float(summary.get("fisher_yy", float("nan"))),
            "fisher_matrix": np.asarray(summary.get("fisher_matrix", np.full((2, 2), float("nan")))),
            "singular": bool(summary.get("singular", True)),
            "fisher_rank": int(summary.get("fisher_rank", 0)),
            "num_frames": int(summary.get("num_frames", 1)),
        }

    def _positive_sigma_order(item: tuple[str, dict[str, Any]]) -> tuple[int, float, int]:
        sigma = float(item[1]["sigma_xy_nm"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            return (1, 0.0, modality_order.index(item[0]))
        return (0, sigma, modality_order.index(item[0]))

    def _positive_sigma_value(value: Any) -> float:
        sigma = float(value)
        return sigma if np.isfinite(sigma) and sigma > 0.0 else float("inf")

    ordered = sorted(per_modality.items(), key=_positive_sigma_order)
    ranking_xy = [
        (m, _positive_sigma_value(v["sigma_xy_nm"]))
        for m, v in ordered
    ]

    best_sigma_xy = ordered[0][1]["sigma_xy_nm"] if ordered else float("inf")
    if np.isfinite(best_sigma_xy) and best_sigma_xy > 0.0:
        best_modality_xy = ordered[0][0]
        relative_sigma_xy = {
            modality: (
                float(v["sigma_xy_nm"]) / float(best_sigma_xy)
                if np.isfinite(float(v["sigma_xy_nm"])) and float(v["sigma_xy_nm"]) > 0.0
                else float("inf")
            )
            for modality, v in per_modality.items()
        }
        frames_to_match_best_xy = {
            modality: (
                float("inf")
                if not np.isfinite(float(v["sigma_xy_nm"])) or float(v["sigma_xy_nm"]) <= 0.0
                else (float(v["sigma_xy_nm"]) / float(best_sigma_xy)) ** 2
            )
            for modality, v in per_modality.items()
        }
    else:
        best_modality_xy = None
        relative_sigma_xy = init_infinite_dict(per_modality)
        frames_to_match_best_xy = init_infinite_dict(per_modality)

    return {
        "per_modality": per_modality,
        "ordering_xy": ranking_xy,
        "ranking_xy": ranking_xy,
        "best_modality_xy": best_modality_xy,
        "relative_sigma_xy": relative_sigma_xy,
        "frames_to_match_best_xy": frames_to_match_best_xy,
    }


def _ranking_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (modality, sigma_xy) in enumerate(result["ranking_xy"], start=1):
        rec = result["per_modality"][modality]
        rows.append(
            {
                "rank": rank,
                "modality": modality,
                "display_name": modality_display_name(modality),
                "sigma_xy_nm": float(sigma_xy),
                "sigma_x_nm": float(rec.get("sigma_x_nm", float("nan"))),
                "sigma_y_nm": float(rec.get("sigma_y_nm", float("nan"))),
                "measurement_domain": rec.get("measurement_domain", ""),
                "signal_units": rec.get("signal_units", ""),
                "noise_variance_units": rec.get("noise_variance_units", ""),
                "detector_noise_input_domain": rec.get("detector_noise_input_domain", ""),
                "nonlinear_detector_effects_active": bool(rec.get("nonlinear_detector_effects_active", False)),
                "deterministic_detector_transfer_active": bool(rec.get("deterministic_detector_transfer_active", False)),
                "safe_for_linear_fisher_variance": bool(rec.get("safe_for_linear_fisher_variance", True)),
                "fisher_variance_model_scope": rec.get("fisher_variance_model_scope", ""),
                "detector_likelihood_status": rec.get("detector_likelihood_status", ""),
                "relative_sigma_xy": float(result["relative_sigma_xy"][modality]),
                "frames_to_match_best_xy": float(result["frames_to_match_best_xy"][modality]),
                "num_frames": int(rec.get("num_frames", 1)),
                "fisher_xx": float(rec["fisher_matrix"][0, 0]),
                "fisher_xy": float(rec["fisher_matrix"][0, 1]),
                "fisher_yy": float(rec["fisher_matrix"][1, 1]),
                "singular": bool(rec.get("singular", False)),
            }
        )
    return rows


def _fusion_rows(
    contrasts: dict[str, np.ndarray],
    noise: dict[str, np.ndarray],
    pixel_size_nm: float,
    max_k: int,
    include_full: bool,
    modality_profile_cards: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(contrasts)
    for k in range(1, max(1, min(max_k, n)) + 1):
        result = compute_modality_fusion_crlb(
            contrasts,
            noise,
            pixel_size_nm,
            subset_size=k,
            modality_profile_cards=modality_profile_cards,
        )
        complementarity = result.get("fusion_complementarity", {}) or {}
        rows.append(
            {
                "subset_size": k,
                "modalities_used": ";".join(result["modalities_used"]),
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
            }
        )
    if include_full and n > 1 and max_k < n:
        result = compute_modality_fusion_crlb(
            contrasts,
            noise,
            pixel_size_nm,
            subset_size=n,
            modality_profile_cards=modality_profile_cards,
        )
        complementarity = result.get("fusion_complementarity", {}) or {}
        rows.append(
            {
                "subset_size": n,
                "modalities_used": ";".join(result["modalities_used"]),
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
            }
        )
    return rows


def _select_fusion_inputs(
    contrasts: dict[str, np.ndarray],
    noise: dict[str, np.ndarray],
    modality_result: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, str]]]:
    """Keep one representative for identical Fisher profiles before fusion."""
    selected: list[tuple[str, np.ndarray, bool]] = []
    duplicate_of: dict[str, dict[str, str]] = {}
    per_modality = modality_result.get("per_modality", {})
    for modality in contrasts:
        rec = per_modality.get(modality, {})
        fisher = np.asarray(rec.get("fisher_matrix"), dtype=float)
        singular = bool(rec.get("singular", False))
        if fisher.shape != (2, 2) or not np.all(np.isfinite(fisher)):
            selected.append((modality, fisher, singular))
            continue
        representative = None
        for existing_modality, existing_fisher, existing_singular in selected:
            if existing_fisher.shape != fisher.shape:
                continue
            if singular == existing_singular and np.allclose(
                fisher,
                existing_fisher,
                rtol=1.0e-8,
                atol=1.0e-12,
            ):
                representative = existing_modality
                break
        if representative is not None:
            duplicate_of[modality] = {
                "representative": representative,
                "reason": "numerically identical lateral Fisher matrix",
            }
            continue
        selected.append((modality, fisher, singular))

    selected_names = [modality for modality, _fisher, _singular in selected]
    return (
        {modality: contrasts[modality] for modality in selected_names},
        {modality: noise[modality] for modality in selected_names},
        duplicate_of,
    )


def _build_sequence_summary_rows(
    modality: str,
    per_frame_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_frame_fishers = [record["fisher_matrix"] for record in per_frame_records]
    if not per_frame_fishers:
        raise ValueError(f"modality {modality!r} returned no frame Fisher matrices.")

    cumulative_fishers, _cumulative_covariances, cumulative_ranks = sequence_sum_fisher_to_crlb(
        per_frame_fishers
    )
    frame_rows: list[dict[str, Any]] = []
    final_crlb = _crlb_from_fisher_matrix(cumulative_fishers[-1])
    final_crlb["num_frames"] = len(per_frame_fishers)
    final_crlb["cumulative_final_rank"] = int(cumulative_ranks[-1])
    final_crlb["sequence_crlb_model"] = "static_same_state_cumulative"
    final_crlb["same_state_assumption"] = True
    final_crlb["safe_for_dynamic_sequence_claim"] = False
    for key in (
        "measurement_domain",
        "signal_units",
        "noise_variance_units",
        "detector_noise_input_domain",
        "nonlinear_detector_effects_active",
        "deterministic_detector_transfer_active",
        "safe_for_linear_fisher_variance",
        "fisher_variance_model_scope",
        "detector_likelihood_status",
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
                "modality": modality,
                "frame_index": frame_idx,
                "num_frames": len(per_frame_fishers),
                "sequence_crlb_model": "static_same_state_cumulative",
                "same_state_assumption": True,
                "safe_for_dynamic_sequence_claim": False,
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
                "safe_for_linear_fisher_variance": bool(per_frame_records[frame_idx].get("safe_for_linear_fisher_variance", True)),
                "fisher_variance_model_scope": per_frame_records[frame_idx].get("fisher_variance_model_scope", ""),
                "detector_likelihood_status": per_frame_records[frame_idx].get("detector_likelihood_status", ""),
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


def _compute_dynamic_sequence_summary(
    modality: str,
    per_frame_fisher: list[np.ndarray],
    params: dict[str, Any],
    per_frame_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not per_frame_fisher:
        raise ValueError(f"no Fisher matrices for modality {modality!r}")
    if len(per_frame_fisher) < 2:
        return {
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
    D = stokes_einstein_diffusion_coefficient(
        diameter_nm,
        float(param_value(params, 'temperature_K')),
        float(param_value(params, 'viscosity_Pa_s')),
    )
    if not np.isfinite(D) or D < 0.0:
        raise ValueError(f"invalid Brownian diffusion coefficient from particle size={diameter_nm} nm.")

    process_scale = float(param_value(params, 'dynamic_process_noise_scale'))
    process_covariance = build_brownian_process_covariance(
        ("x", "y"),
        fps=float(params["fps"]),
        translational_diffusion_coeff_m2_s=float(D) * process_scale,
    )
    initial_covariance = np.eye(2, dtype=float) * float(
        param_value(params, 'dynamic_initial_variance_nm2')
    )
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
        state_transition_fps=float(params["fps"]),
        fps=float(params["fps"]),
        initial_covariance=initial_covariance,
        initial_variance_fallback=float(param_value(params, 'dynamic_initial_variance_nm2')),
        include_smoothing=bool(param_value(params, 'dynamic_include_smoothing')),
        include_fisher_matrices=False,
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units,
        state_axis_units={"x": "nm", "y": "nm"},
        process_model="brownian_translation_lateral_xy",
        dynamic_validation_status="implemented_estimator_layer_lateral_xy",
    )
    summary["modality"] = modality
    summary["dynamic_enabled"] = True
    summary["dynamic_scope"] = "lateral_xy"
    summary["translational_diffusion_coeff_m2_s"] = float(D)
    summary["dynamic_process_noise_scale"] = float(process_scale)
    return summary
