"""
Native-regime reference-check profiles for Syniscopy modalities.

These profiles are separate from the shared cross-modality ranking profile.
Each case configures one modality in a literature-adjacent native regime,
renders a single centered particle, propagates counts-domain detector noise
into the contrast frame, and computes a lateral localization CRLB. The
`classification` field records how the cited source is used: a direct
localization precision, a formula/theory-derived localization scale, a
literature-scale comparison, a modality-principle citation, or a dimensional
metrology scale that must not be read as a center-localization bound.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import json
from typing import Any

import numpy as np

from camera_noise import analysis_contrast_noise_variance
from config import PARAMS
from fisher_diagnostic import compute_fisher_information, compute_localization_crlb
from imaging_model import (
    SUPPORTED_MODALITIES,
    canonical_modality_name,
    modality_display_name,
    modality_uses_relative_reference_contrast,
)
from main import generate_single_frame_views


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


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
        "components": [
            {
                "shape": "sphere",
                "offset_nm": [0.0, 0.0, 0.0],
                "diameter_nm": float(diameter_nm),
                "material": material,
                "refractive_index": None,
                "signal_multiplier": 1.0,
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

    params = deepcopy(PARAMS)
    params.update(
        {
            "imaging_model": modality,
            "image_size_pixels": image_size,
            "pixel_size_nm": pixel_size_nm,
            "pupil_samples": int(case.get("pupil_samples", DEFAULT_PUPIL_SAMPLES)),
            "psf_oversampling_factor": int(case.get("psf_oversampling_factor", 2)),
            "fps": 24.0,
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
    side_nm = image_size * pixel_size_nm
    p = _particle(diameter_nm, material=material)
    p["motion"]["initial_position_nm"] = [0.5 * side_nm, 0.5 * side_nm, z_nm]
    params["particles"] = [p]
    return params


def _relative_reference_for_noise(modality: str) -> bool:
    return modality_uses_relative_reference_contrast(modality)


def run_calibration_profile(modality: str) -> dict[str, Any]:
    requested_modality = str(modality)
    canonical_modality = canonical_modality_name(requested_modality)
    case = CALIBRATION_PROFILES[canonical_modality]
    params = native_params(case)
    views = generate_single_frame_views(params)
    contrast = views.get("contrast_frame")
    signal_counts = views.get("ideal_signal_frame")
    if signal_counts is None:
        signal_counts = views.get("raw_signal_frame")
    reference_counts = views.get("ideal_reference_frame")
    if reference_counts is None:
        reference_counts = views.get("raw_reference_frame")
    if contrast is None or signal_counts is None:
        raise RuntimeError(f"{modality} did not produce calibration contrast/count frames.")
    contrast_arr = np.asarray(contrast, dtype=float)
    signal_arr = np.asarray(signal_counts, dtype=float)
    reference_arr = None if reference_counts is None else np.asarray(reference_counts, dtype=float)
    noise_var = analysis_contrast_noise_variance(
        signal_arr,
        reference_arr,
        params,
        relative_reference=_relative_reference_for_noise(canonical_modality),
    )
    crlb = compute_localization_crlb(contrast_arr, noise_var, float(params["pixel_size_nm"]))
    fisher = compute_fisher_information(contrast_arr, noise_var, float(params["pixel_size_nm"]))
    computed = float(crlb["sigma_xy_nm"])
    target = float(case["target_sigma_xy_nm"])
    ratio = computed / target if np.isfinite(computed) and target > 0 else float("inf")
    classification = str(case.get("classification", "LITERATURE_LOCALIZATION_SCALE")).upper()
    source_scale_applies = classification in LOCALIZATION_SCALE_CLASSIFICATIONS
    numeric_nonlocalization_scale = classification in NONLOCALIZATION_NUMERIC_SCALE_CLASSIFICATIONS
    if source_scale_applies:
        target_kind = "source_localization_scale"
    elif numeric_nonlocalization_scale:
        target_kind = "source_nonlocalization_scale"
    else:
        target_kind = "proxy_comparison_target"
    agreement_ratio = ratio if source_scale_applies else None
    within_order = (
        bool(0.1 <= agreement_ratio <= 10.0)
        if agreement_ratio is not None and np.isfinite(agreement_ratio)
        else None
    )
    parameter_match_status = str(case.get("parameter_match_status", "partial")).lower()
    if parameter_match_status not in {"yes", "partial", "no", "not_applicable"}:
        parameter_match_status = "partial"
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
        "particle_material": case.get("particle_material", ""),
        "diameter_nm": float(case.get("diameter_nm", 100.0)),
        "pixel_size_nm": float(params["pixel_size_nm"]),
        "image_size_pixels": int(params["image_size_pixels"]),
        "computed_sigma_xy_nm": computed,
        "comparison_target_sigma_xy_nm": target,
        "comparison_target_kind": target_kind,
        "source_reported_quantity": case.get("source_reported_quantity", ""),
        "source_scale_applies_to_localization": bool(source_scale_applies),
        "reference_target_sigma_xy_nm": (
            target if source_scale_applies else None
        ),
        "published_reference_sigma_xy_nm": (
            target if source_scale_applies else None
        ),
        "reference_target_kind": target_kind,
        "agreement_ratio": agreement_ratio,
        "computed_to_comparison_ratio": ratio,
        "within_order_of_magnitude": within_order,
        "citation": case["citation"],
        "citation_url": case["citation_url"],
        "total_detected_quanta": float(np.nansum(signal_arr)),
        "mean_noise_variance": float(np.nanmean(noise_var)),
        "fisher_trace": float(np.trace(fisher)),
        "fisher_det": float(np.linalg.det(fisher)),
        "finite_crlb": bool(np.isfinite(computed)),
        "notes": case.get("notes", ""),
    }


def assert_calibration_within_order_of_magnitude(modality: str) -> dict[str, Any]:
    row = run_calibration_profile(modality)
    assert row["finite_crlb"], f"{modality} calibration returned non-finite CRLB: {row}"
    assert row["agreement_ratio"] is not None, (
        f"{modality} calibration has no source-localization scale to compare: {row}"
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_docs(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for modality, case in CALIBRATION_PROFILES.items():
        params_preview = native_params(case)
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
            f"- Wavelength: {params_preview.get('wavelength_nm')} nm",
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
                _json_safe(case.get("overrides", {})),
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
    write_profile_docs(output_dir / "calibration_profiles")
    (output_dir / "calibration_reference_check_manifest.json").write_text(
        json.dumps(
            _json_safe({
                "schema_version": "syniscopy-calibration-reference-check-v1",
                "modalities": list(CALIBRATION_PROFILES),
                "rows_csv": rows_csv.name,
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
            "dark_field_field_gain": 30.0,
            "annular_dark_field_inner_sigma": 1.02,
            "annular_dark_field_outer_sigma": 1.45,
        },
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
        "overrides": {"dark_field_field_gain": 110.0},
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
        "notes": "Proxy tuning: Kurata et al. report ZPM optics and phase-retrieval residuals, but not a particle-localization photon budget or 10 nm lateral CRLB. The 1.0e6 background count is therefore a proxy budget chosen to match the reference scale, not an independently derived detector budget.",
    },
    "differential_phase_contrast": {
        "profile_id": "dpc_tian_waller_2015_native",
        "modality": "differential_phase_contrast",
        "profile_summary": "DPC with propagated detector shot noise and conservative phase-gradient gain.",
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
        "overrides": {"dpc_phase_gradient_gain": 500.0},
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
            "iscat_collection_model": "dipole_high_na",
            "read_noise_counts": 0.5,
        },
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
            "fluorescence_emission_psf_sigma_px": 1.3,
            "fluorescence_photon_count_scale": 2.0e4,
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
            "fluorescence_emission_psf_sigma_px": 1.3,
            "fluorescence_photon_count_scale": 2.0e4,
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
        "profile_id": "tem_ctf_bonevich_nist_native",
        "modality": "tem_phase_contrast",
        "profile_summary": "Native-pitch TEM weak-phase CTF check for nanoparticle sizing/localization.",
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
        "overrides": {"tem_dose_per_pixel": 1.0e8},
    },
    "sem_secondary_electron": {
        "profile_id": "sem_crouzier_2019_native",
        "modality": "sem_secondary_electron",
        "profile_summary": "Native-pitch SEM secondary-electron nanoparticle localization/sizing proxy.",
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
        "overrides": {"sem_probe_sigma_pixels": 1.0, "sem_electrons_per_pixel": 100.0},
    },
}


missing = set(SUPPORTED_MODALITIES) - set(CALIBRATION_PROFILES)
if missing:
    raise RuntimeError(f"Missing calibration profiles for registry modalities: {sorted(missing)}")
