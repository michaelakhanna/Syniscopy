"""
Native-regime reference-check profiles for Syniscopy modalities.

These profiles are separate from the shared configured-microscope comparison
profile.
Each case configures one modality in a literature-adjacent native regime,
renders a single centered particle, propagates counts-domain detector noise
into the contrast frame, and computes a lateral localization CRLB. The
`classification` field records how the cited source is used: a direct
localization precision, a formula/theory-derived localization scale, a
literature-scale comparison, a modality-principle citation, or a dimensional
metrology scale that must not be read as a center-localization bound.
"""

from __future__ import annotations
from configured_parameters import configured_assign

from copy import deepcopy
from pathlib import Path
import csv
import json
from typing import Any

import numpy as np

from camera_noise import (
    analysis_contrast_noise_model,
    analysis_noise_params_for_frame,
    detector_contrast_frames_for_analysis,
)
from config import SamplingGeometry, SemSettings, TemSettings, default_params
from config.runtime import FluorescenceSettings
from fisher import (
    compute_fisher_information,
    compute_localization_crlb,
    compute_off_axis_demodulated_localization_crlb,
    is_off_axis_holography_modality,
    lateral_derivative_plan_metadata,
    require_array_only_spectral_lateral_derivative_ready,
)
from imaging_models import (
    get_imaging_model,
    modality_uses_relative_reference_contrast,
)
from json_utils import json_safe
from noise_contracts import summarize_analysis_noise_model
from simulation import generate_single_frame_views
from modality_registry import SUPPORTED_MODALITIES, canonical_modality_name, modality_display_name
from postprocessing import compute_single_frame_contrast


DEFAULT_IMAGE_SIZE = 128
DEFAULT_PUPIL_SAMPLES = 128
DEFAULT_PIXEL_SIZE_NM = 20.0
DEFAULT_WAVELENGTH_NM = 532.0

LOCALIZATION_SCALE_CLASSIFICATIONS = {
    "DIRECT_QUOTED_LOCALIZATION_PRECISION",
    "LITERATURE_LOCALIZATION_SCALE",
    "FORMULA_DERIVED_LOCALIZATION_SCALE",
    "THEORY_DERIVED_LOCALIZATION_SCALE",
}

NONLOCALIZATION_NUMERIC_SCALE_CLASSIFICATIONS = {
    "DIMENSIONAL_METROLOGY_SCALE_NOT_LOCALIZATION",
}

PRINCIPLE_CITATION_CLASSIFICATIONS = {
    "MODALITY_PRINCIPLE_CITATION_ONLY",
}

STRICT_VALIDATION = "strict_validation"
REFERENCE_SCALE_CHECK = "reference_scale_check"
PROXY_CALIBRATION_CHECK = "proxy_calibration_check"
NONLOCALIZATION_REFERENCE = "nonlocalization_reference"
PRINCIPLE_CITATION = "principle_citation"

VALIDATION_TIERS = {
    STRICT_VALIDATION,
    REFERENCE_SCALE_CHECK,
    PROXY_CALIBRATION_CHECK,
    NONLOCALIZATION_REFERENCE,
    PRINCIPLE_CITATION,
}

VALIDATION_TIER_LABELS = {
    STRICT_VALIDATION: "strict validation",
    REFERENCE_SCALE_CHECK: "reference scale check",
    PROXY_CALIBRATION_CHECK: "proxy calibration check",
    NONLOCALIZATION_REFERENCE: "nonlocalization reference",
    PRINCIPLE_CITATION: "principle citation",
}

VALIDATION_TIER_OUTPUT_FILES = {
    STRICT_VALIDATION: "strict_validation_table.csv",
    REFERENCE_SCALE_CHECK: "reference_scale_checks.csv",
    PROXY_CALIBRATION_CHECK: "proxy_calibration_checks.csv",
    NONLOCALIZATION_REFERENCE: "nonlocalization_reference_checks.csv",
    PRINCIPLE_CITATION: "principle_citations.csv",
}

STRICT_BACKEND_FIDELITY_LEVELS = {"high_fidelity", "reference_validated"}
STRICT_BACKEND_VALIDATION_STATUSES = {"validated", "reference_validated"}


def _normalized_text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _backend_fidelity_record(response: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(response or {})
    nested = payload.get("backend_fidelity_metadata")
    metadata = dict(nested) if isinstance(nested, dict) else {}
    reference = metadata.get("reference_backend_metadata")
    if reference is None:
        reference = payload.get("reference_backend_metadata")
    return {
        "present": bool(metadata) or any(
            key in payload
            for key in (
                "backend_fidelity_level",
                "validation_status",
                "implemented_approximation_level",
                "reference_backend_metadata",
            )
        ),
        "backend_name": (
            metadata.get("backend_name")
            or payload.get("backend_name")
            or payload.get("kind")
            or payload.get("tem_backend")
            or payload.get("sem_backend")
            or payload.get("fluorescence_backend")
            or ""
        ),
        "backend_fidelity_level": _normalized_text(
            metadata.get("backend_fidelity_level", payload.get("backend_fidelity_level"))
        ),
        "backend_validation_status": _normalized_text(
            metadata.get("validation_status", payload.get("validation_status"))
        ),
        "implemented_approximation_level": _normalized_text(
            metadata.get(
                "implemented_approximation_level",
                payload.get("implemented_approximation_level", payload.get("fidelity_label")),
            )
        ),
        "reference_backend_metadata": reference if isinstance(reference, dict) else {},
    }


def _source_claim_flags(classification: str) -> tuple[bool, bool, bool]:
    normalized = str(classification).upper()
    return (
        normalized in LOCALIZATION_SCALE_CLASSIFICATIONS,
        normalized in NONLOCALIZATION_NUMERIC_SCALE_CLASSIFICATIONS,
        normalized in PRINCIPLE_CITATION_CLASSIFICATIONS,
    )


def _strict_validation_blockers(
    *,
    modality: str,
    case: dict[str, Any],
    params: dict[str, Any],
    response: dict[str, Any] | None,
    classification: str,
    parameter_match_status: str,
    source_has_localization_scale: bool,
    source_has_nonlocalization_scale: bool,
    is_principle_citation: bool,
    proxy_tuning_target_sigma_xy_nm: float | None,
) -> list[str]:
    blockers: list[str] = []
    if not source_has_localization_scale:
        if source_has_nonlocalization_scale:
            blockers.append("nonlocalization_reference_not_center_localization")
        elif is_principle_citation:
            blockers.append("principle_citation_no_numeric_localization_target")
        else:
            blockers.append("no_source_localization_bound")
    if parameter_match_status != "yes":
        blockers.append(f"parameter_match_status={parameter_match_status}")
    if proxy_tuning_target_sigma_xy_nm is not None:
        blockers.append("proxy_tuning_target_present")
    if str(case.get("comparison_kind", "")).strip().lower() == "scale_context_only":
        blockers.append("comparison_kind=scale_context_only")

    text_for_tuning_check = " ".join(
        str(case.get(key, ""))
        for key in ("classification_reason", "parameter_match_note", "profile_summary", "notes")
    ).lower()
    if "proxy tuning" in text_for_tuning_check:
        blockers.append("proxy_tuning_note_present")
    elif "scale tuning" in text_for_tuning_check:
        blockers.append("scale_tuning_note_present")

    backend = _backend_fidelity_record(response)
    validates_lower_level_formula = bool(case.get("validates_lower_level_analytic_formula", False))
    if backend["present"] and not validates_lower_level_formula:
        fidelity_level = str(backend["backend_fidelity_level"])
        validation_status = str(backend["backend_validation_status"])
        approximation_level = str(backend["implemented_approximation_level"])
        if fidelity_level and fidelity_level not in STRICT_BACKEND_FIDELITY_LEVELS:
            blockers.append(f"backend_fidelity_level={fidelity_level}")
        if validation_status and validation_status not in STRICT_BACKEND_VALIDATION_STATUSES:
            blockers.append(f"backend_validation_status={validation_status}")
        if "proxy" in approximation_level:
            blockers.append(f"implemented_approximation_level={approximation_level}")

    modality_name = str(modality)
    if "fluorescence" in modality_name or "tirf" in modality_name:
        fluorescence_settings = FluorescenceSettings.from_params(params)
        if fluorescence_settings.absorbed_excitation_photons_per_fluorophore_per_frame <= 0.0:
            blockers.append("fluorescence_absorbed_excitation_photon_budget_missing")

    if modality_name == "tem_phase_contrast":
        tem_settings = TemSettings.from_params(params)
        if tem_settings.reference_status != "reference_validated":
            blockers.append("tem_reference_status_not_reference_validated")
        if not tem_settings.reference_validation_hash:
            blockers.append("tem_reference_validation_hash_missing")

    if modality_name == "sem_secondary_electron":
        reference_meta = backend.get("reference_backend_metadata", {})
        reference_status = _normalized_text(reference_meta.get("reference_status"))
        sem_settings = SemSettings.from_params(params)
        reference_hash = (
            reference_meta.get("reference_validation_hash")
            or reference_meta.get("sem_reference_kernel_sha256")
            or sem_settings.reference_kernel_sha256
        )
        if reference_status != "reference_validated":
            blockers.append("sem_reference_status_not_reference_validated")
        if not reference_hash:
            blockers.append("sem_reference_validation_hash_missing")

    return sorted(set(blockers))


def _validation_tier_for_claim(
    *,
    source_has_localization_scale: bool,
    source_has_nonlocalization_scale: bool,
    is_principle_citation: bool,
    strict_validation_blockers: list[str],
) -> str:
    if source_has_localization_scale:
        return STRICT_VALIDATION if not strict_validation_blockers else REFERENCE_SCALE_CHECK
    if source_has_nonlocalization_scale:
        return NONLOCALIZATION_REFERENCE
    if is_principle_citation:
        return PRINCIPLE_CITATION
    return PROXY_CALIBRATION_CHECK


def _profile_claim_metadata(
    modality: str,
    case: dict[str, Any],
    params: dict[str, Any],
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    classification = str(case.get("classification", "LITERATURE_LOCALIZATION_SCALE")).upper()
    source_has_localization_scale, source_has_nonlocalization_scale, is_principle_citation = (
        _source_claim_flags(classification)
    )
    parameter_match_status = str(case.get("parameter_match_status", "partial")).lower()
    if parameter_match_status not in {"yes", "partial", "no", "not_applicable"}:
        parameter_match_status = "partial"
    target = float(case["target_sigma_xy_nm"])
    proxy_tuning_target_sigma_xy_nm = target if not source_has_localization_scale else None
    blockers = _strict_validation_blockers(
        modality=modality,
        case=case,
        params=params,
        response=response,
        classification=classification,
        parameter_match_status=parameter_match_status,
        source_has_localization_scale=source_has_localization_scale,
        source_has_nonlocalization_scale=source_has_nonlocalization_scale,
        is_principle_citation=is_principle_citation,
        proxy_tuning_target_sigma_xy_nm=proxy_tuning_target_sigma_xy_nm,
    )
    validation_tier = _validation_tier_for_claim(
        source_has_localization_scale=source_has_localization_scale,
        source_has_nonlocalization_scale=source_has_nonlocalization_scale,
        is_principle_citation=is_principle_citation,
        strict_validation_blockers=blockers,
    )
    return {
        "classification": classification,
        "parameter_match_status": parameter_match_status,
        "source_has_localization_scale": bool(source_has_localization_scale),
        "source_has_nonlocalization_scale": bool(source_has_nonlocalization_scale),
        "is_principle_citation": bool(is_principle_citation),
        "proxy_tuning_target_sigma_xy_nm": proxy_tuning_target_sigma_xy_nm,
        "validation_tier": validation_tier,
        "validation_tier_label": VALIDATION_TIER_LABELS[validation_tier],
        "strict_validation_blockers": blockers,
    }


def _particle(diameter_nm: float, material: str = "polystyrene") -> dict[str, Any]:
    material_properties = None
    if material == "fluorescent_polystyrene":
        material_properties = {
            "fluorophore_density": 0.08,
            "excitation_peak_nm": 488.0,
            "emission_peak_nm": 520.0,
        }
    return {
        "name": f"{material}_{diameter_nm:g}nm",
        "motion": {
            "hydrodynamic_diameter_nm": float(diameter_nm),
            "initial_position_nm": None,
        },
        "signal_multiplier": 1.0,
        "source_multiplier": 1.0,
        "components": [
            {
                "shape": "sphere",
                "offset_nm": [0.0, 0.0, 0.0],
                "diameter_nm": float(diameter_nm),
                "material": material,
                "refractive_index": None,
                "signal_multiplier": 1.0,
                "source_multiplier": 1.0,
                "material_properties": material_properties,
            }
        ],
    }


def native_params(case: dict[str, Any]) -> dict[str, Any]:
    modality = str(case["modality"])
    diameter_nm = float(case.get("diameter_nm", 100.0))
    z_nm = float(case.get("z_nm", 0.0))
    image_size = int(case.get("image_size_pixels", DEFAULT_IMAGE_SIZE))
    pixel_size_nm = float(case.get("pixel_size_nm", DEFAULT_PIXEL_SIZE_NM))
    material = str(case.get(
        "particle_material",
        "fluorescent_polystyrene" if "fluorescence" in modality else "polystyrene",
    ))
    pupil_samples = int(case.get("pupil_samples", DEFAULT_PUPIL_SAMPLES))
    vectorial_pupil_samples = int(case.get("vectorial_pupil_samples", pupil_samples))

    params = default_params()
    params.update(
        {
            "imaging_model": modality,
            "image_size_pixels": image_size,
            "pixel_size_nm": pixel_size_nm,
            "pupil_samples": pupil_samples,
            "vectorial_pupil_samples": vectorial_pupil_samples,
            "psf_oversampling_factor": int(case.get("psf_oversampling_factor", 2)),
            "fps": 24.0,
            "num_frames": 1,
            "duration_seconds": 1.0 / 24.0,
            "wavelength_nm": float(case.get("wavelength_nm", DEFAULT_WAVELENGTH_NM)),
            "numerical_aperture": float(case.get("numerical_aperture", 1.0)),
            "refractive_index_medium": float(case.get("refractive_index_medium", 1.33)),
            "refractive_index_immersion": float(case.get("refractive_index_immersion", 1.518)),
            "background_intensity": float(case.get("background_intensity", 1.0e4)),
            "read_noise_counts": float(case.get("read_noise_counts", 1.0)),
            "camera_gain_e_per_count": float(case.get("camera_gain_e_per_count", 1.0)),
            "shot_noise_enabled": True,
            "gaussian_noise_enabled": True,
            "fixed_pattern_gain_std": 0.0,
            "fixed_pattern_offset_counts": 0.0,
            "hot_pixel_fraction": 0.0,
            "scan_line_noise_counts": 0.0,
            "return_ideal_float_frames": True,
            "save_frame_sequence": False,
            "save_raw_camera_video": False,
            "save_raw_camera_frame_sequence": False,
            "save_raw_frame_views": False,
            "mask_generation_enabled": False,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "sample_environment_pattern": "none",
            "sample_environment_pattern_preset": "empty_background",
            "background_subtraction_method": "reference_frame",
            "z_motion_constraint_model": "unconstrained",
            "z_stack_step_nm": 100.0,
            "initial_z_span_nm": max(4000.0, abs(z_nm) * 2.0 + 1000.0),
            "channels": None,
        }
    )
    params.update(deepcopy(case.get("overrides", {})))
    center_nm = 0.5 * (image_size - 1) * pixel_size_nm
    p = _particle(diameter_nm, material=material)
    p["motion"]["initial_position_nm"] = [center_nm, center_nm, z_nm]
    configured_assign(params, 'particles', [p])
    return params


def _calibration_analysis_payload(params: dict[str, Any], canonical_modality: str) -> dict[str, Any]:
    views = generate_single_frame_views(params)
    analysis_params = dict(views.get("params_resolved", params) or params)
    render_metadata = dict(views.get("render_metadata", {}) or {})
    # Calibration uses the same analysis likelihood sidecar as reports and
    # matched packets.  For QPI, the renderer owns per-frame detected quanta
    # after reference patterns/roughness, so a plain params+exposure merge is
    # not a scientifically valid phase-noise basis.
    noise_params = dict(
        views.get("analysis_noise_params")
        or analysis_noise_params_for_frame(analysis_params, render_metadata, frame_index=0)
    )
    contrast = views.get("contrast_frame")
    signal_counts = views.get("detector_input_signal_frame")
    if signal_counts is None:
        signal_counts = views.get("ideal_signal_frame")
    if signal_counts is None:
        signal_counts = views.get("raw_signal_frame")
    reference_counts = views.get("detector_input_reference_frame")
    if reference_counts is None:
        reference_counts = views.get("ideal_reference_frame")
    if reference_counts is None:
        reference_counts = views.get("raw_reference_frame")
    object_field = views.get("detector_object_field_frame")
    if contrast is None or signal_counts is None:
        raise RuntimeError(f"{canonical_modality} did not produce calibration contrast/count frames.")
    signal_arr = np.asarray(signal_counts, dtype=float)
    reference_arr = None if reference_counts is None else np.asarray(reference_counts, dtype=float)
    model = get_imaging_model(analysis_params)
    if getattr(model, "output_type", "intensity") == "phase":
        contrast_arr = np.asarray(contrast, dtype=float)
    else:
        contrast_signal, contrast_reference = detector_contrast_frames_for_analysis(
            signal_arr,
            reference_arr,
            noise_params,
            relative_reference=modality_uses_relative_reference_contrast(canonical_modality),
        )
        contrast_arr = np.asarray(
            compute_single_frame_contrast(contrast_signal, contrast_reference, analysis_params),
            dtype=float,
        )
    noise_model = analysis_contrast_noise_model(
        signal_arr,
        reference_arr,
        noise_params,
        relative_reference=modality_uses_relative_reference_contrast(canonical_modality),
    )
    noise_summary = summarize_analysis_noise_model(
        noise_model,
        expected_shape=contrast_arr.shape,
        context=f"calibration contrast frame shape for modality {canonical_modality!r}",
    )
    return {
        "views": views,
        "analysis_params": analysis_params,
        "render_metadata": render_metadata,
        "signal_arr": signal_arr,
        "reference_arr": reference_arr,
        "object_field": object_field,
        "contrast_arr": contrast_arr,
        "noise_model": noise_model,
        "noise_summary": noise_summary,
        "model": model,
    }


def run_calibration_profile(modality: str) -> dict[str, Any]:
    requested_modality = str(modality)
    canonical_modality = canonical_modality_name(requested_modality)
    case = CALIBRATION_PROFILES[canonical_modality]
    params = native_params(case)
    payload = _calibration_analysis_payload(params, canonical_modality)
    analysis_params = payload["analysis_params"]
    signal_arr = payload["signal_arr"]
    reference_arr = payload["reference_arr"]
    object_field = payload.get("object_field")
    contrast_arr = payload["contrast_arr"]
    noise_model = payload["noise_model"]
    noise_summary = payload["noise_summary"]
    model = payload["model"]
    sampling = SamplingGeometry.from_params(analysis_params)
    response = model.compute_response_function(signal_arr.shape, analysis_params)
    if is_off_axis_holography_modality(canonical_modality):
        crlb, _observation = compute_off_axis_demodulated_localization_crlb(
            signal_arr,
            reference_arr,
            analysis_params,
            sampling.detector_pixel_size_nm,
            response_function=response,
            object_field_detector=object_field,
        )
    else:
        require_array_only_spectral_lateral_derivative_ready(
            modality=canonical_modality,
            params=analysis_params,
            model=model,
            response_function=response,
            num_particles=1,
            structured_environment_active=False,
            context=f"run_calibration_profile[{canonical_modality!r}]",
        )
        crlb = compute_localization_crlb(
            contrast_arr,
            noise_model,
            sampling.detector_pixel_size_nm,
        )
    crlb.update(lateral_derivative_plan_metadata())
    fisher = (
        np.asarray(crlb["fisher_matrix"], dtype=float)
        if is_off_axis_holography_modality(canonical_modality)
        else compute_fisher_information(
            contrast_arr,
            noise_model,
            sampling.detector_pixel_size_nm,
        )
    )
    computed = float(crlb["sigma_xy_nm"])
    target = float(case["target_sigma_xy_nm"])
    claim = _profile_claim_metadata(canonical_modality, case, analysis_params, response)
    classification = str(claim["classification"])
    source_has_localization_scale = bool(claim["source_has_localization_scale"])
    source_has_nonlocalization_scale = bool(claim["source_has_nonlocalization_scale"])
    parameter_match_status = str(claim["parameter_match_status"])
    validation_tier = str(claim["validation_tier"])
    is_strict_validation = validation_tier == STRICT_VALIDATION
    backend_fidelity = _backend_fidelity_record(response)
    source_localization_sigma_xy_nm = target if source_has_localization_scale else None
    source_nonlocalization_scale_nm = target if source_has_nonlocalization_scale else None
    proxy_tuning_target_sigma_xy_nm = claim["proxy_tuning_target_sigma_xy_nm"]
    comparison_target_sigma_xy_nm = (
        source_localization_sigma_xy_nm
        if is_strict_validation
        else None
    )
    strict_ratio = (
        computed / comparison_target_sigma_xy_nm
        if is_strict_validation
        and comparison_target_sigma_xy_nm is not None
        and np.isfinite(computed)
        and comparison_target_sigma_xy_nm > 0
        else None
    )
    scale_ratio_not_validation = (
        computed / target
        if not is_strict_validation
        and np.isfinite(computed)
        and np.isfinite(target)
        and target > 0.0
        else None
    )
    if source_has_localization_scale:
        target_kind = "source_localization_scale"
    elif source_has_nonlocalization_scale:
        target_kind = "source_nonlocalization_scale"
    else:
        target_kind = "proxy_comparison_target"
    agreement_ratio = strict_ratio
    within_order = (
        bool(0.1 <= agreement_ratio <= 10.0)
        if agreement_ratio is not None and np.isfinite(agreement_ratio)
        else None
    )
    comparison_row_role = {
        STRICT_VALIDATION: "computed_localization_comparison",
        REFERENCE_SCALE_CHECK: "reference_scale_check",
        PROXY_CALIBRATION_CHECK: "proxy_calibration_check",
        NONLOCALIZATION_REFERENCE: "nonlocalization_reference",
        PRINCIPLE_CITATION: "principle_citation",
    }[validation_tier]
    phase_domain_output = str(getattr(model, "output_type", "intensity")) == "phase"
    signal_sum = float(np.nansum(signal_arr))
    total_detected_quanta = None if phase_domain_output else signal_sum
    phase_display_counts_sum = signal_sum if phase_domain_output else None
    detector_count_sum_semantics = (
        "phase_display_counts_not_detected_quanta"
        if phase_domain_output
        else "detected_quanta"
    )
    return {
        "modality": canonical_modality,
        "requested_modality": requested_modality,
        "display_name": modality_display_name(canonical_modality),
        "profile_id": case["profile_id"],
        "profile_summary": case["profile_summary"],
        "classification": classification,
        "classification_reason": case.get("classification_reason", ""),
        "parameter_match_status": parameter_match_status,
        "parameter_match_note": case.get("parameter_match_note", ""),
        "comparison_kind": str(case.get("comparison_kind", "")),
        "validation_tier": validation_tier,
        "validation_tier_label": claim["validation_tier_label"],
        "strict_validation_blockers": list(claim["strict_validation_blockers"]),
        "strict_validation_blocker_count": len(claim["strict_validation_blockers"]),
        "row_role": comparison_row_role,
        "source_has_localization_scale": bool(source_has_localization_scale),
        "source_has_nonlocalization_scale": bool(source_has_nonlocalization_scale),
        "is_parameter_matched_localization_comparison": bool(is_strict_validation),
        "is_validation_comparison": bool(is_strict_validation),
        "particle_material": case.get("particle_material", ""),
        "diameter_nm": float(case.get("diameter_nm", 100.0)),
        "pixel_size_nm": SamplingGeometry.from_params(params).detector_pixel_size_nm,
        "image_size_pixels": SamplingGeometry.from_params(params).image_size_pixels,
        "computed_sigma_xy_nm": computed,
        "comparison_target_sigma_xy_nm": comparison_target_sigma_xy_nm,
        "comparison_target_kind": target_kind,
        "computed_localization_comparison_eligible": bool(is_strict_validation),
        "source_reported_quantity": case.get("source_reported_quantity", ""),
        "source_scale_applies_to_localization": bool(source_has_localization_scale),
        "source_localization_sigma_xy_nm": source_localization_sigma_xy_nm,
        "source_nonlocalization_scale_nm": source_nonlocalization_scale_nm,
        "proxy_tuning_target_sigma_xy_nm": proxy_tuning_target_sigma_xy_nm,
        "reference_target_sigma_xy_nm": source_localization_sigma_xy_nm,
        "published_reference_sigma_xy_nm": source_localization_sigma_xy_nm,
        "reference_target_kind": target_kind,
        "agreement_ratio": agreement_ratio,
        "computed_to_comparison_ratio": agreement_ratio,
        "scale_ratio_not_validation": scale_ratio_not_validation,
        "within_order_of_magnitude": within_order,
        "backend_name": backend_fidelity["backend_name"],
        "backend_fidelity_level": backend_fidelity["backend_fidelity_level"],
        "backend_validation_status": backend_fidelity["backend_validation_status"],
        "backend_implemented_approximation_level": backend_fidelity["implemented_approximation_level"],
        "citation": case["citation"],
        "citation_url": case["citation_url"],
        "total_detected_quanta": total_detected_quanta,
        "phase_display_counts_sum": phase_display_counts_sum,
        "detector_count_sum_semantics": detector_count_sum_semantics,
        "mean_noise_variance": noise_summary.mean_diagonal_variance,
        "mean_noise_variance_units": noise_summary.noise_variance_units,
        "noise_measurement_domain": noise_summary.measurement_domain,
        "noise_signal_units": noise_summary.signal_units,
        "analysis_noise_covariance_kind": noise_summary.covariance_kind,
        "analysis_noise_status_reason": noise_summary.status_reason,
        "fisher_trace": float(np.trace(fisher)),
        "fisher_det": float(np.linalg.det(fisher)),
        "finite_crlb": bool(np.isfinite(computed)),
        "fisher_lateral_derivative_basis": "spectral_band_limited",
        "fisher_lateral_derivative_basis_resolution": "single_center_render_fft_spectral_gradient",
        "notes": case.get("notes", ""),
    }


def assert_calibration_within_order_of_magnitude(modality: str) -> dict[str, Any]:
    row = run_calibration_profile(modality)
    if row.get("validation_tier") != STRICT_VALIDATION:
        blockers = row.get("strict_validation_blockers", [])
        raise AssertionError(
            f"{modality} calibration profile is not eligible for strict validation "
            f"(validation_tier={row.get('validation_tier')!r}, blockers={blockers!r})."
        )
    assert row["finite_crlb"], f"{modality} calibration returned non-finite CRLB: {row}"
    assert row["agreement_ratio"] is not None, (
        f"{modality} strict calibration has no source-localization scale to compare: {row}"
    )
    assert row["within_order_of_magnitude"], (
        f"{modality} calibration ratio {row['agreement_ratio']:.3g} outside "
        f"one order of magnitude for profile {row['profile_id']}: {row}"
    )
    return row


def run_all_calibration_profiles(modalities: list[str] | None = None) -> list[dict[str, Any]]:
    selected = list(modalities or CALIBRATION_PROFILES.keys())
    return [run_calibration_profile(modality) for modality in selected]


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _write_rows_csv_with_fields(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_docs(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for modality, case in CALIBRATION_PROFILES.items():
        params_preview = native_params(case)
        model = get_imaging_model(params_preview)
        response = model.compute_response_function(
            (int(params_preview["image_size_pixels"]), int(params_preview["image_size_pixels"])),
            params_preview,
        )
        claim = _profile_claim_metadata(modality, case, params_preview, response)
        probe_wavelength = response.get("probe_wavelength_nm", params_preview.get("probe_wavelength_nm"))
        path = output_dir / f"{modality}.md"
        classification = str(case.get("classification", "")).upper()
        if classification in LOCALIZATION_SCALE_CLASSIFICATIONS:
            scale_line = (
                f"- Source/localization scale: {case['target_sigma_xy_nm']} nm lateral localization"
            )
        elif classification in NONLOCALIZATION_NUMERIC_SCALE_CLASSIFICATIONS:
            scale_line = (
                f"- Source non-localization scale: {case['target_sigma_xy_nm']} nm"
            )
        else:
            scale_line = (
                f"- Proxy comparison target: {case['target_sigma_xy_nm']} nm "
                "lateral localization (not source-quoted)"
            )
        lines = [
            f"# {modality_display_name(modality)} calibration profile",
            "",
            f"- Claim tier: {claim['validation_tier_label']}",
            f"- Direct validation comparison: {'yes' if claim['validation_tier'] == STRICT_VALIDATION else 'no'}",
            *(
                []
                if claim["validation_tier"] == STRICT_VALIDATION
                else ["- This profile is not a direct validation comparison."]
            ),
            f"- Strict-validation blockers: {', '.join(claim['strict_validation_blockers']) or 'none'}",
            f"- Profile id: `{case['profile_id']}`",
            f"- Registry modality: `{modality}`",
            scale_line,
            f"- Classification: {case.get('classification', 'REFERENCE_MATCHED_CHECK')}",
            f"- Parameter match: {case.get('parameter_match_status', 'partial')}",
            f"- Parameter-match note: {case.get('parameter_match_note', '')}",
            f"- Source-reported quantity: {case.get('source_reported_quantity', '')}",
            f"- Citation: {case['citation']} ({case['citation_url']})",
            f"- Summary: {case['profile_summary']}",
            "",
            "## Parameters",
            "",
            f"- Configured optical wavelength: {params_preview.get('wavelength_nm')} nm",
            f"- Probe wavelength: {probe_wavelength} nm",
            f"- Response kind: {response.get('kind', '')}",
            f"- Measurement domain: {response.get('measurement_domain', '')}",
            f"- Fidelity label: {response.get('fidelity_label', '')}",
            f"- NA: {params_preview.get('numerical_aperture')}",
            f"- Particle material: {case.get('particle_material', '')}",
            f"- Particle diameter: {case.get('diameter_nm')} nm",
            f"- Pixel pitch: {params_preview.get('pixel_size_nm')} nm",
            f"- Background intensity: {params_preview.get('background_intensity')} counts",
            f"- Read noise: {params_preview.get('read_noise_counts')} counts",
            "",
            "## Overrides",
            "",
            "```json",
            json.dumps(
                json_safe(case.get("overrides", {})),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            "```",
            "",
            f"Classification reason: {case.get('classification_reason', '')}",
            "",
            f"Notes: {case.get('notes', '')}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def write_calibration_outputs(output_dir: str | Path) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    rows = run_all_calibration_profiles()
    rows_csv = output_dir / "calibration_reference_check_table.csv"
    write_rows_csv(rows_csv, rows)
    fieldnames = list(rows[0].keys()) if rows else []
    validation_tier_counts = {tier: 0 for tier in VALIDATION_TIER_OUTPUT_FILES}
    grouped_rows_csv = {}
    for tier, filename in VALIDATION_TIER_OUTPUT_FILES.items():
        tier_rows = [row for row in rows if row.get("validation_tier") == tier]
        validation_tier_counts[tier] = len(tier_rows)
        grouped_rows_csv[tier] = filename
        _write_rows_csv_with_fields(output_dir / filename, tier_rows, fieldnames)
    write_profile_docs(output_dir / "calibration_profiles")
    (output_dir / "calibration_reference_check_manifest.json").write_text(
        json.dumps(
            json_safe({
                "schema_version": "syniscopy-calibration-reference-check-v1",
                "modalities": list(CALIBRATION_PROFILES),
                "rows_csv": rows_csv.name,
                "grouped_rows_csv": grouped_rows_csv,
                "validation_tier_counts": validation_tier_counts,
                "validation_tiers": sorted(VALIDATION_TIERS),
            }),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return rows


CALIBRATION_PROFILES: dict[str, dict[str, Any]] = {
    "bright_field": {
        "profile_id": "pc_brightfield_covari_2019_native",
        "modality": "bright_field",
        "profile_summary": "Partially coherent bright-field at native optical sampling with a 100 nm polystyrene sphere.",
        "classification": "LITERATURE_LOCALIZATION_SCALE",
        "classification_reason": "Kovari et al. support nanometer-scale bright-field tracking, but the 15 nm value is used as a literature-scale comparison rather than a quote-matched 100 nm-particle/detector-budget bound.",
        "parameter_match_status": "partial",
        "parameter_match_note": "literature-scale comparison; detector budget and sample profile are not quote-matched",
        "source_reported_quantity": "nanometer-scale bright-field tracking precision; 15 nm is a representative literature-scale comparison used for the native-regime audit",
        "citation": "Kovari et al., Optics Express 2019",
        "citation_url": "https://doi.org/10.1364/OE.27.029875",
        "target_sigma_xy_nm": 15.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 3.0e4,
        "overrides": {"kohler_source_samples": 11, "kohler_coherence_factor": 0.7},
    },
    "partially_coherent_bright_field": {
        "profile_id": "pc_brightfield_explicit_covari_2019_native",
        "modality": "partially_coherent_bright_field",
        "profile_summary": "Explicit registry alias for partially coherent Köhler bright-field.",
        "classification": "LITERATURE_LOCALIZATION_SCALE",
        "classification_reason": "Kovari et al. support nanometer-scale bright-field tracking, but partial-coherence parameters and the current count budget are not independently quoted from the source.",
        "parameter_match_status": "partial",
        "parameter_match_note": "literature-scale comparison; partial-coherence and count parameters are not quote-matched",
        "source_reported_quantity": "nanometer-scale bright-field tracking precision; 15 nm is a representative literature-scale comparison used for the native-regime audit",
        "citation": "Kovari et al., Optics Express 2019",
        "citation_url": "https://doi.org/10.1364/OE.27.029875",
        "target_sigma_xy_nm": 15.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 2.0e4,
        "overrides": {"kohler_source_samples": 11, "kohler_coherence_factor": 0.7},
    },
    "coherent_bright_field": {
        "profile_id": "coherent_bright_field_huang_2017_native",
        "modality": "coherent_bright_field",
        "profile_summary": "Coherent bright-field contrast for gold nanoparticle tracking.",
        "classification": "LITERATURE_LOCALIZATION_SCALE",
        "classification_reason": "Huang et al. report sub-3 nm interferometric tracking of virus-scale particles, but this 60 nm gold/count profile is not a full quote-matched reconstruction.",
        "parameter_match_status": "partial",
        "parameter_match_note": "literature-scale comparison; target class is closer than the detector-count profile",
        "source_reported_quantity": "sub-3 nm interferometric bright-field tracking scale",
        "citation": "Huang et al., ACS Nano 2017",
        "citation_url": "https://doi.org/10.1021/acsnano.6b05601",
        "target_sigma_xy_nm": 3.0,
        "particle_material": "gold",
        "diameter_nm": 60.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 5.0e4,
    },
    "dark_field": {
        "profile_id": "annular_darkfield_ando_2018_native",
        "modality": "dark_field",
        "profile_summary": "Annular dark-field gold-particle tracking at native optical sampling.",
        "classification": "DIRECT_QUOTED_LOCALIZATION_PRECISION",
        "classification_reason": "Ando/Kurihara et al. report 1.3 Angstrom localization precision for 40 nm AuNP dark-field tracking; this is 0.13 nm. The simulator field gain and count budget remain a scale audit, not a quote-matched photon-count reconstruction.",
        "parameter_match_status": "partial",
        "parameter_match_note": "direct quoted precision; particle size/material match, detector-count and optical-gain parameters not quote-matched",
        "source_reported_quantity": "1.3 Angstrom lateral localization precision for 40 nm AuNPs, converted to 0.13 nm",
        "citation": "Ando et al., Biophysical Journal 2018",
        "citation_url": "https://doi.org/10.1016/j.bpj.2018.11.016",
        "target_sigma_xy_nm": 0.13,
        "particle_material": "gold",
        "diameter_nm": 40.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 1.0e5,
        "overrides": {
            "dark_field_illumination_count": 1.0e5,
            "dark_field_background_count": 10.0,
            "dark_field_field_gain": 1.0,
            "annular_dark_field_inner_sigma": 1.02,
            "annular_dark_field_outer_sigma": 1.08,
        },
        "comparison_kind": "scale_context_only",
    },
    "coherent_dark_field": {
        "profile_id": "coherent_darkfield_dong_2021_native",
        "modality": "coherent_dark_field",
        "profile_summary": "Coherent dark-field native-regime tracking check using a gold nanoparticle.",
        "classification": "THEORY_DERIVED_LOCALIZATION_SCALE",
        "classification_reason": "Dong et al. provide per-collected-scattered-photon CRBs, not the configured detector-count budget.",
        "parameter_match_status": "partial",
        "parameter_match_note": "theory-derived scale; source photon normalization differs from this detector-count profile",
        "source_reported_quantity": "per-collected-scattered-photon CRB scale; detector-count budget here is not source-quoted",
        "citation": "Dong et al., Journal of Physics D 2021",
        "citation_url": "https://doi.org/10.1088/1361-6463/ac0f22",
        "target_sigma_xy_nm": 3.0,
        "particle_material": "gold",
        "diameter_nm": 40.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 1.0e5,
        "overrides": {"dark_field_field_gain": 1.0},
        "comparison_kind": "scale_context_only",
    },
    "zernike_phase_contrast": {
        "profile_id": "zernike_kurata_2024_native",
        "modality": "zernike_phase_contrast",
        "profile_summary": "Zernike phase-contrast proxy; the cited ZPM paper reports phase-retrieval optics but not a localization photon budget.",
        "classification": "NO_QUOTED_LOCALIZATION_BOUND",
        "classification_reason": "Kurata et al. do not report a particle-localization CRLB/precision target, so the row is not a validation comparison.",
        "parameter_match_status": "not_applicable",
        "parameter_match_note": "no source localization precision is quoted",
        "source_reported_quantity": "phase-retrieval optics/residuals; no quoted particle-localization bound",
        "citation": "Kurata et al., Optics Express 2024",
        "citation_url": "https://doi.org/10.1364/OE.509877",
        "target_sigma_xy_nm": 10.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 1.0e6,
        "overrides": {"zernike_phase_ring_gain": 0.35},
        "comparison_kind": "scale_context_only",
        "notes": "Proxy tuning: Kurata et al. report ZPM optics and phase-retrieval residuals, but not a particle-localization photon budget or 10 nm lateral CRLB. The 1.0e6 background count is therefore a proxy budget chosen to match the reference scale, not an independently derived detector budget.",
    },
    "differential_phase_contrast": {
        "profile_id": "dpc_tian_waller_2015_native",
        "modality": "differential_phase_contrast",
        "profile_summary": "Illumination-side asymmetric DPC with propagated detector shot noise.",
        "classification": "NO_QUOTED_LOCALIZATION_BOUND",
        "classification_reason": "Tian and Waller validate quantitative DPC phase reconstruction, but do not report a particle-localization precision target matching this row.",
        "parameter_match_status": "not_applicable",
        "parameter_match_note": "no source localization precision is quoted",
        "source_reported_quantity": "quantitative DPC phase-reconstruction method; no quoted particle-localization bound",
        "citation": "Tian and Waller, Optics Express 2015",
        "citation_url": "https://doi.org/10.1364/OE.23.011394",
        "target_sigma_xy_nm": 10.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 65.0,
        "background_intensity": 1000.0,
        "read_noise_counts": 2.0,
        "overrides": {
            "dpc_channel_model": "vectorial_debye_asymmetric_illumination",
            "dpc_transfer_model": "asymmetric_illumination",
            "dpc_output_channel": "x",
            "dpc_illumination_sigma": 0.7,
            "dpc_source_samples": 19,
            "vectorial_detection_mode": "full_vector",
            "dpc_intensity_gain": 1.0,
        },
    },
    "quantitative_phase": {
        "profile_id": "qpi_bon_2015_native",
        "modality": "quantitative_phase",
        "profile_summary": "Quantitative phase imaging native-regime localization check.",
        "classification": "DIRECT_QUOTED_LOCALIZATION_PRECISION",
        "classification_reason": "Bon et al. report nanometre localization for gold nanoparticles; the row now uses a gold-particle material basis while retaining the simulator's simplified count/noise profile.",
        "parameter_match_status": "partial",
        "parameter_match_note": "direct quoted precision; gold-particle basis retained but detector/noise/profile parameters are not quote-matched",
        "source_reported_quantity": "about 1.5 nm lateral localization precision for gold nanoparticles",
        "citation": "Bon et al., Nature Communications 2015",
        "citation_url": "https://doi.org/10.1038/ncomms8764",
        "target_sigma_xy_nm": 1.5,
        "particle_material": "gold",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 2.0e4,
        "overrides": {"qpi_phase_to_count_scale": 100.0, "qpi_phase_noise_std_rad": 0.06},
    },
    "off_axis_holography": {
        "profile_id": "dhm_verpillat_2011_native",
        "modality": "off_axis_holography",
        "profile_summary": "Off-axis DHM native-regime localization check.",
        "classification": "DIRECT_QUOTED_LOCALIZATION_PRECISION",
        "classification_reason": "Verpillat et al. report approximately 3 nm lateral resolution for 100 nm gold particles in dark-field digital holographic microscopy; the simulator count profile is still a scale audit.",
        "parameter_match_status": "partial",
        "parameter_match_note": "direct quoted precision; particle size/material match, detector-count and holographic reconstruction details not quote-matched",
        "source_reported_quantity": "approximately 3 nm lateral localization/tracking resolution for 100 nm gold particles",
        "citation": "Verpillat et al., Optics Express 2011",
        "citation_url": "https://doi.org/10.1364/OE.19.026044",
        "target_sigma_xy_nm": 3.0,
        "particle_material": "gold",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 5.0e4,
        "overrides": {"off_axis_fringe_period_px": 8.0},
    },
    "ricm": {
        "profile_id": "ricm_clack_groves_2005_native",
        "modality": "ricm",
        "profile_summary": "RICM Fresnel-interface literature-scale check; the cited precision is reported for micron silica spheres without a detector-count basis.",
        "classification": "DIRECT_QUOTED_LOCALIZATION_PRECISION",
        "classification_reason": "Clack and Groves report 16 nm lateral RICM precision, but not the detector-count budget; the current 100 nm polystyrene profile also differs from their micron silica spheres.",
        "parameter_match_status": "partial",
        "parameter_match_note": "direct quoted precision; source particle/sample and detector-count profile differ",
        "source_reported_quantity": "16 nm lateral precision reported for micron silica spheres",
        "citation": "Clack and Groves, Langmuir 2005",
        "citation_url": "https://doi.org/10.1021/la050372r",
        "target_sigma_xy_nm": 16.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 2.5e6,
        "overrides": {
            "reference_field_amplitude": 1.0,
            "ricm_interface_reflection_model": "fresnel",
            "ricm_particle_reflection_model": "fresnel",
            "ricm_interface_medium_material": "water",
            "ricm_interface_substrate_material": "glass",
            "ricm_particle_medium_material": "water",
            "ricm_particle_material": "polystyrene",
        },
        "comparison_kind": "scale_context_only",
        "notes": "Scale tuning: Clack and Groves report 16 nm lateral precision for 6.8 um silica microspheres near borosilicate, but the accessible paper metadata/abstract do not provide the photon/count budget needed to derive 2.5e6 background counts. The current 100 nm polystyrene profile is a scale-matched proxy, not an independently parameter-matched reconstruction.",
    },
    "interferometric": {
        "profile_id": "iscat_dong_2021_native",
        "modality": "interferometric",
        "profile_summary": "iSCAT Fresnel-reference/high-NA check with a literature-scale photon budget.",
        "classification": "THEORY_DERIVED_LOCALIZATION_SCALE",
        "classification_reason": "Dong et al. normalize CRBs per collected scattered photon and do not specify the detector background/reference-count budget used here.",
        "parameter_match_status": "partial",
        "parameter_match_note": "theory-derived scale; photon-normalized source bound differs from detector-count audit profile",
        "source_reported_quantity": "per-collected-scattered-photon iSCAT CRB scale; detector-count budget here is not source-quoted",
        "citation": "Dong et al., Journal of Physics D 2021",
        "citation_url": "https://doi.org/10.1088/1361-6463/ac0f22",
        "target_sigma_xy_nm": 2.0,
        "particle_material": "gold",
        "diameter_nm": 40.0,
        "pixel_size_nm": 20.0,
        "background_intensity": 7.5e3,
        "numerical_aperture": 1.3,
        "overrides": {
            "reference_field_amplitude": 1.0,
            "iscat_reference_model": "fresnel",
            "iscat_reference_medium_material": "water",
            "iscat_reference_substrate_material": "glass",
            "iscat_collection_model": "scalar",
            "read_noise_counts": 0.5,
        },
        "comparison_kind": "scale_context_only",
        "notes": "Scale tuning: Dong et al. give CRBs normalized per collected scattered photon and list wavelength/NA/material parameters, but do not provide a detector background_intensity or reference-count budget. The 7.5e3 background count was selected by an illumination sweep to match the 2 nm CRLB scale and should not be described as an independently derived experimental photon budget.",
    },
    "fluorescence_widefield": {
        "profile_id": "widefield_thompson_2002_native",
        "modality": "fluorescence_widefield",
        "profile_summary": "Widefield fluorescence single-emitter localization native-regime check.",
        "classification": "FORMULA_DERIVED_LOCALIZATION_SCALE",
        "classification_reason": "Thompson et al. provide the photon/background localization model, but the configured 5 nm target and 40 nm bead profile are not fully quoted source parameters.",
        "parameter_match_status": "partial",
        "parameter_match_note": "formula-derived scale; source formula is real, configured numeric target is not a quote-matched row",
        "source_reported_quantity": "formula-derived localization scale from the Thompson-Larson-Webb pixelated fluorescence model",
        "citation": "Thompson, Larson, and Webb, Biophysical Journal 2002",
        "citation_url": "https://doi.org/10.1016/S0006-3495(02)75618-X",
        "target_sigma_xy_nm": 5.0,
        "particle_material": "fluorescent_polystyrene",
        "diameter_nm": 40.0,
        "pixel_size_nm": 100.0,
        "background_intensity": 1000.0,
        "overrides": {
            "fluorescence_emission_psf_sigma_nm": 130.0,
            "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": 4.0e4,
            "fluorescence_background": 0.001,
            "fluorescence_excitation_wavelength_nm": 488.0,
            "fluorescence_emission_wavelength_nm": 520.0,
        },
    },
    "tirf_fluorescence": {
        "profile_id": "tirf_axelrod_oheim_native",
        "modality": "tirf_fluorescence",
        "profile_summary": "TIRF fluorescence with angle-derived evanescent penetration and near-surface particle.",
        "classification": "MODALITY_PRINCIPLE_CITATION_ONLY",
        "classification_reason": "Axelrod's cited paper is a TIRF principle/application source and does not quote a 5 nm particle-localization bound for this profile.",
        "parameter_match_status": "not_applicable",
        "parameter_match_note": "modality-principle citation only; no source localization precision is quoted",
        "source_reported_quantity": "TIRF near-surface excitation principle; no quoted lateral localization bound",
        "citation": "Axelrod, Journal of Cell Biology 1981",
        "citation_url": "https://doi.org/10.1083/jcb.89.1.141",
        "target_sigma_xy_nm": 5.0,
        "particle_material": "fluorescent_polystyrene",
        "diameter_nm": 40.0,
        "z_nm": 10.0,
        "pixel_size_nm": 100.0,
        "background_intensity": 1000.0,
        "overrides": {
            "tirf_fluorescence_backend": "parametric_psf",
            "tirf_source_representation": "projected_2d",
            "fluorescence_emission_psf_sigma_nm": 130.0,
            "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": 4.0e4,
            "fluorescence_background": 0.0002,
            "fluorescence_excitation_wavelength_nm": 488.0,
            "fluorescence_emission_wavelength_nm": 520.0,
            "tirf_use_angle_derived_penetration_depth": True,
            "tirf_incident_angle_deg": 66.0,
            "tirf_prism_refractive_index": 1.518,
            "tirf_sample_refractive_index": 1.333,
            "tirf_effective_numerical_aperture": 1.3,
        },
    },
    "tem_phase_contrast": {
        "profile_id": "tem_multislice_bonevich_nist_native",
        "modality": "tem_phase_contrast",
        "profile_summary": "Native-pitch physical multislice TEM check for nanoparticle dimensional metrology scale.",
        "classification": "DIMENSIONAL_METROLOGY_SCALE_NOT_LOCALIZATION",
        "classification_reason": "Bonevich et al. address TEM nanoparticle size/dimensional metrology, not a lateral particle-center localization bound.",
        "parameter_match_status": "not_applicable",
        "parameter_match_note": "dimensional metrology source, not a localization precision source",
        "source_reported_quantity": "TEM nanoparticle dimensional-metrology scale, not center-localization sigma",
        "citation": "Bonevich et al., Metrologia 2013",
        "citation_url": "https://doi.org/10.1088/0026-1394/50/6/663",
        "target_sigma_xy_nm": 2.0,
        "particle_material": "polystyrene",
        "diameter_nm": 100.0,
        "pixel_size_nm": 2.0,
        "background_intensity": 1.0e8,
        "overrides": {
            "tem_model": "multislice_physical",
            "tem_backend": "multislice_physical",
            "tem_dose_per_pixel": 1.0e8,
        },
    },
    "sem_secondary_electron": {
        "profile_id": "sem_crouzier_2019_native",
        "modality": "sem_secondary_electron",
        "profile_summary": "Native-pitch SEM secondary-electron nanoparticle dimensional metrology proxy.",
        "classification": "DIMENSIONAL_METROLOGY_SCALE_NOT_LOCALIZATION",
        "classification_reason": "Crouzier et al. report SEM nanoparticle diameter/dimensional measurement uncertainty, not a lateral particle-center localization bound.",
        "parameter_match_status": "not_applicable",
        "parameter_match_note": "dimensional metrology source, not a localization precision source",
        "source_reported_quantity": "SEM nanoparticle dimensional-metrology scale, not center-localization sigma",
        "citation": "Crouzier et al., Ultramicroscopy 2019",
        "citation_url": "https://doi.org/10.1016/j.ultramic.2019.112847",
        "target_sigma_xy_nm": 1.3,
        "particle_material": "gold",
        "diameter_nm": 100.0,
        "pixel_size_nm": 5.0,
        "image_size_pixels": 160,
        "background_intensity": 100.0,
        "overrides": {
            "sem_model": "physical_electron_transport",
            "sem_backend": "monte_carlo_physical",
            "sem_probe_sigma_nm": 5.0,
            "sem_electrons_per_pixel": 100.0,
        },
    },
}


missing = set(SUPPORTED_MODALITIES) - set(CALIBRATION_PROFILES)
if missing:
    raise RuntimeError(f"Missing calibration profiles for registry modalities: {sorted(missing)}")
