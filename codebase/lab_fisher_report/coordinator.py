"""Coordinator for lab Fisher report generation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bootstrap import REPO_ROOT, ensure_codebase_on_path

ensure_codebase_on_path()

from fisher import compute_information_density_maps
from config import AcquisitionProfile, SamplingGeometry
from imaging_models import get_imaging_model
from json_utils import json_safe
from modality_profiles import profile_card_for_model
from modality_registry import SUPPORTED_MODALITIES, modality_display_name
from presets import get_instrument_preset_names

from .cli import _parse_args, _write_microscope_set_template
from .microscopes import (
    MicroscopeSet,
    MicroscopeSpec,
    load_microscope_set,
    microscopes_from_modality_sweep,
    resolve_microscope_params,
)
from .params_assembly import (
    _make_microscope_base_and_shared_params,
    _make_params,
    _resolve_modalities,
)
from .render import _density_uint8, _display_uint8, _render_microscope
from .report_writer import _clear_managed_outputs, _write_csv, _write_report
from .scene_view import scene_provenance_from_params
from .shared_latent_scene import build_shared_latent_scene
from .tables import (
    _build_sequence_summary_rows,
    _compute_dynamic_sequence_summary,
    _dynamic_fusion_rows_from_fisher_sequences,
    _dynamic_sequence_summary_to_ranking_summary,
    _fusion_rows_from_fisher_matrices,
    _ranking_rows,
    _select_fusion_fisher_inputs,
    _sequence_information_content,
)

__all__ = ["main", "run_report"]


def _artifact_stem(candidate_name: str, used: set[str]) -> str:
    """Return a filesystem-safe stem owned by the microscope identity."""

    stem = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(candidate_name).strip()
    ).strip("._")
    if not stem:
        stem = "microscope"
    candidate = stem
    index = 2
    while candidate in used:
        candidate = f"{stem}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _microscope_candidates_from_args(
    args: argparse.Namespace,
) -> tuple[list[MicroscopeSpec], dict[str, Any], str, list[str]]:
    """Resolve the report comparison axis as microscopes, not raw modalities."""

    microscopes_path = getattr(args, "microscopes", None)
    if microscopes_path:
        microscope_set = load_microscope_set(microscopes_path)
        scopes = list(microscope_set.microscopes)
        return (
            scopes,
            dict(microscope_set.shared_params),
            "explicit_microscope_set",
            [scope.name for scope in scopes],
        )

    sweep_spec = getattr(args, "modality_sweep", "lab-default")
    modalities = _resolve_modalities(sweep_spec)
    scopes = list(microscopes_from_modality_sweep(modalities))
    return scopes, {}, "modality_sweep", [scope.name for scope in scopes]


def _validated_microscope_set_for_cli(
    args: argparse.Namespace,
    path: str | Path,
) -> tuple[MicroscopeSet, dict[str, dict[str, Any]]]:
    """Load a microscope JSON and validate the complete resolved report params."""

    microscope_set = load_microscope_set(path)
    base_params, shared_params = _make_microscope_base_and_shared_params(
        args,
        microscope_shared_params=microscope_set.shared_params,
    )
    resolved: dict[str, dict[str, Any]] = {}
    for scope in microscope_set.microscopes:
        # Schema-only validation is not sufficient for the public microscope
        # contract: the final candidate is parameters + report defaults + instrument
        # + shared_params + microscope.params + modality + microscope runtime
        # owner validation.
        # This path exercises the same resolved validation owner as rendering
        # without running the expensive renderer/Fisher pipeline.
        resolved[scope.name] = resolve_microscope_params(
            scope,
            base_params,
            shared_params=shared_params,
        )
    return microscope_set, resolved


def run_report(args: argparse.Namespace) -> Path:
    microscopes, microscope_shared_params, microscope_input_mode, requested_microscopes = (
        _microscope_candidates_from_args(args)
    )
    modalities = [scope.modality for scope in microscopes]
    modality_by_microscope = {scope.name: scope.modality for scope in microscopes}
    # The comparison unit is now a microscope. For explicit microscope JSON,
    # keep the render base below run-level JSON/CLI/shared-scene overlays so
    # resolve_microscope_params can apply each microscope instrument preset
    # before the shared overlay, matching the public layering contract.
    if microscope_input_mode == "explicit_microscope_set":
        render_base_params, microscope_shared_params = _make_microscope_base_and_shared_params(
            args,
            microscope_shared_params=microscope_shared_params,
        )
        params = deepcopy(render_base_params)
        params.update(deepcopy(microscope_shared_params))
    else:
        params = _make_params(args, resolved_modalities=modalities)
        render_base_params = params

    # Scene provenance is resolved before any microscope-local overlay and is
    # emitted into the manifest/report so candidate-specific sampling or detector
    # choices cannot masquerade as a different particle/scene authority.
    report_scene_provenance = scene_provenance_from_params(params)

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed_outputs(output_dir)
    preview_dir = output_dir / "previews"
    density_dir = output_dir / "fisher_density"
    if not args.no_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
        density_dir.mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}
    resolved_candidate_params: dict[str, dict[str, Any]] = {}
    for scope in microscopes:
        try:
            resolved_candidate_params[scope.name] = resolve_microscope_params(
                scope,
                render_base_params,
                shared_params=microscope_shared_params,
            )
        except Exception as exc:
            errors[scope.name] = repr(exc)
    if not resolved_candidate_params:
        raise RuntimeError(f"No microscopes resolved successfully. Errors: {errors}")
    shared_latent_scene = build_shared_latent_scene(params, resolved_candidate_params)

    contrasts: dict[str, np.ndarray] = {}
    noise: dict[str, np.ndarray] = {}
    sequence_rows: list[dict[str, Any]] = []
    sequence_summaries: dict[str, dict[str, Any]] = {}
    dynamic_summaries: dict[str, dict[str, Any]] = {}
    fisher_matrices_by_microscope: dict[str, list[np.ndarray]] = {}
    resolved_microscopes: list[str] = []
    resolved_example: dict[str, Any] | None = None
    resolved_by_microscope: dict[str, dict[str, Any]] = {}
    profile_cards_by_microscope: dict[str, dict[str, Any]] = {}
    used_artifact_stems: set[str] = set()
    base_num_frames = AcquisitionProfile.from_params(params).num_frames

    for scope in microscopes:
        microscope_name = scope.name
        modality = scope.modality
        artifact_stem = _artifact_stem(microscope_name, used_artifact_stems)
        print(
            f"Rendering microscope {microscope_name} "
            f"({modality}, {base_num_frames} frame(s))..."
        )
        if microscope_name not in resolved_candidate_params:
            continue
        try:
            rendered, render_meta = _render_microscope(
                render_base_params,
                scope,
                shared_params=microscope_shared_params,
                resolved_params=resolved_candidate_params[microscope_name],
                latent_scene_view=shared_latent_scene.view_for_microscope(microscope_name),
                shared_latent_metadata=shared_latent_scene.metadata_for_microscope(
                    microscope_name
                ),
            )
        except Exception as exc:
            errors[microscope_name] = repr(exc)
            continue

        resolved = render_meta["resolved_params"]
        num_frames = int(render_meta.get("num_frames", 1))
        per_frame = rendered.get("per_frame", [])
        fisher_matrices = rendered.get("fisher_matrices", [])
        if not per_frame:
            errors[microscope_name] = (
                f"no per-frame records for microscope {microscope_name!r} "
                f"(modality {modality!r})"
            )
            continue
        if len(per_frame) != num_frames:
            errors[microscope_name] = (
                f"reported {len(per_frame)} frames, expected {num_frames} "
                f"for microscope {microscope_name!r} (modality {modality!r})"
            )
            continue

        frame_rows, summary = _build_sequence_summary_rows(
            modality,
            per_frame,
            microscope=microscope_name,
        )
        sequence_rows.extend(frame_rows)
        sequence_summaries[microscope_name] = summary
        fisher_matrices_by_microscope[microscope_name] = [
            np.asarray(matrix, dtype=float) for matrix in fisher_matrices
        ]
        resolved_microscopes.append(microscope_name)

        first = per_frame[0]
        contrast = np.asarray(first["contrast"], dtype=float)
        noise_var = np.asarray(first["noise_variance"], dtype=float)
        contrasts[microscope_name] = contrast
        noise[microscope_name] = noise_var
        if resolved_example is None:
            resolved_example = resolved
        resolved_by_microscope[microscope_name] = resolved
        try:
            profile_cards_by_microscope[microscope_name] = profile_card_for_model(
                resolved,
                get_imaging_model(resolved),
                modality_name=modality,
            )
        except Exception:
            profile_cards_by_microscope.pop(microscope_name, None)

        if bool(args.dynamic_bayesian):
            try:
                dynamic_summary = _compute_dynamic_sequence_summary(
                    modality,
                    fisher_matrices,
                    resolved,
                    per_frame,
                    microscope=microscope_name,
                )
            except Exception as exc:
                dynamic_summary = {
                    "microscope": microscope_name,
                    "modality": modality,
                    "dynamic_enabled": bool(args.dynamic_bayesian),
                    "dynamic_summary_available": False,
                    "dynamic_error": repr(exc),
                }
            dynamic_summaries[microscope_name] = dynamic_summary

        if not args.no_previews:
            preview = _display_uint8(contrast)
            Image.fromarray(preview).save(preview_dir / f"{artifact_stem}.png")
            noise_for_density = first.get("analysis_noise_model", noise_var)
            density_params = dict(params)
            density_params.update(dict(resolved or {}))
            maps = compute_information_density_maps(
                contrast,
                noise_for_density,
                SamplingGeometry.from_params(density_params).detector_pixel_size_nm,
            )
            lateral_density = maps["Ix_info_map"] + maps["Iy_info_map"]
            np.save(density_dir / f"{artifact_stem}_lateral_density.npy", lateral_density)
            Image.fromarray(_density_uint8(lateral_density)).save(
                density_dir / f"{artifact_stem}_lateral_density.png"
            )

    if not contrasts:
        raise RuntimeError(f"No microscopes rendered successfully. Errors: {errors}")

    ranking_summaries = dict(sequence_summaries)
    if dynamic_summaries:
        for microscope_name, dynamic_summary in dynamic_summaries.items():
            if microscope_name in sequence_summaries:
                ranking_summaries[microscope_name] = _dynamic_sequence_summary_to_ranking_summary(
                    dynamic_summary,
                    sequence_summaries[microscope_name],
                )

    result = _sequence_information_content(resolved_microscopes, ranking_summaries)
    ranking = _ranking_rows(result)

    def _fusion_profile_payload(microscope_name: str) -> dict[str, Any]:
        """Return the modality/config payload required by fusion compatibility."""

        payload: dict[str, Any] = {}
        if microscope_name in resolved_by_microscope:
            payload.update(dict(resolved_by_microscope[microscope_name]))
        if microscope_name in profile_cards_by_microscope:
            payload.update(dict(profile_cards_by_microscope[microscope_name]))
        return payload

    fusion_input_basis = "precomputed_sequence_fisher_matrix"
    fusion_microscopes_used: list[str]
    fusion_duplicates: dict[str, dict[str, str]]
    fusion_frame_count: int | None
    if dynamic_summaries:
        fusion, fusion_duplicates, fusion_sequences = _dynamic_fusion_rows_from_fisher_sequences(
            fisher_matrices_by_microscope,
            dynamic_summaries,
            ranking_summaries,
            max_k=max(1, int(args.max_fusion_k)),
            include_full=bool(args.include_full_fusion),
            microscope_profile_cards={
                microscope_name: _fusion_profile_payload(microscope_name)
                for microscope_name in ranking_summaries
            },
            fisher_lateral_derivative_basis="spectral_band_limited",
        )
        fusion_microscopes_used = list(fusion_sequences)
        fusion_frame_counts = {
            len(sequence) for sequence in fusion_sequences.values()
        }
        fusion_frame_count = next(iter(fusion_frame_counts)) if len(fusion_frame_counts) == 1 else None
        fusion_input_basis = "joint_dynamic_bayesian_per_frame_measurement_fisher_sequence"
    else:
        fusion_fisher, fusion_duplicates = _select_fusion_fisher_inputs(result)
        fusion_microscopes_used = list(fusion_fisher)
        fusion_frame_counts = {
            int(result["per_microscope"][microscope_name].get("num_frames", base_num_frames))
            for microscope_name in fusion_fisher
            if microscope_name in result.get("per_microscope", {})
        }
        fusion_frame_count = next(iter(fusion_frame_counts)) if len(fusion_frame_counts) == 1 else None
        fusion = _fusion_rows_from_fisher_matrices(
            fusion_fisher,
            max_k=max(1, int(args.max_fusion_k)),
            include_full=bool(args.include_full_fusion),
            microscope_profile_cards={
                microscope_name: _fusion_profile_payload(microscope_name)
                for microscope_name in fusion_fisher
            },
            parent_result_metadata_by_microscope={
                microscope_name: result["per_microscope"][microscope_name]
                for microscope_name in fusion_fisher
                if microscope_name in result.get("per_microscope", {})
            },
            fusion_frame_count=fusion_frame_count,
            fisher_lateral_derivative_basis="spectral_band_limited",
        )

    _write_csv(
        output_dir / "microscope_ranking.csv",
        ranking,
        [
            "rank", "rankable_for_ordering", "ranking_status", "rankable", "is_best_xy",
            "microscope", "modality", "display_name", "sigma_xy_nm", "sigma_x_nm", "sigma_y_nm",
            "latent_scene_id", "latent_schedule_id", "state_time_policy",
            "fusion_time_alignment", "shared_coordinate_frame", "same_latent_scene",
            "measurement_domain", "signal_units", "noise_variance_units",
            "detector_noise_input_domain", "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active", "safe_for_linear_fisher_variance",
            "safe_for_covariance_fisher_variance", "detector_safe_for_report_fisher",
            "fisher_likelihood_uses_covariance", "fisher_likelihood_eligibility_contract_id",
            "fisher_variance_model_scope", "covariance_fisher_variance_model_scope",
            "detector_likelihood_status", "safe_for_ordering", "safe_for_fusion",
            "status_reason", "sequence_crlb_model", "same_state_assumption",
            "frame_equivalence_model", "relative_sigma_xy", "frames_to_match_best_xy",
            "frames_to_match_best_xy_status", "num_frames", "fisher_xx", "fisher_xy", "fisher_yy", "singular",
            "derivative_basis", "nyquist_band_fraction", "boundary_energy_fraction", "convergence_status",
        ],
    )
    _write_csv(
        output_dir / "sequence_fisher_summary.csv",
        sequence_rows,
        [
            "microscope", "modality", "frame_index", "num_frames", "sequence_crlb_model",
            "latent_scene_id", "latent_schedule_id", "observation_time_s",
            "state_time_policy", "fusion_time_alignment", "shared_coordinate_frame", "same_latent_scene",
            "same_state_assumption", "safe_for_dynamic_sequence_claim", "safe_for_ordering",
            "safe_for_fusion", "frame_fisher_xx", "frame_fisher_xy", "frame_fisher_yy",
            "frame_fisher_det", "frame_fisher_singular", "frame_fisher_rank", "frame_sigma_x_nm",
            "frame_sigma_y_nm", "frame_sigma_xy_nm", "measurement_domain", "signal_units",
            "noise_variance_units", "detector_noise_input_domain", "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active", "safe_for_linear_fisher_variance",
            "safe_for_covariance_fisher_variance", "detector_safe_for_report_fisher",
            "fisher_likelihood_uses_covariance", "fisher_likelihood_eligibility_contract_id",
            "fisher_variance_model_scope", "covariance_fisher_variance_model_scope",
            "detector_likelihood_status", "cumulative_fisher_xx", "cumulative_fisher_xy",
            "cumulative_fisher_yy", "cumulative_fisher_det", "cumulative_fisher_rank",
            "cumulative_sigma_x_nm", "cumulative_sigma_y_nm", "cumulative_sigma_xy_nm",
            "derivative_basis", "nyquist_band_fraction", "boundary_energy_fraction", "convergence_status",
        ],
    )
    _write_csv(
        output_dir / "fusion_crlb.csv",
        fusion,
        [
            "subset_size", "microscopes_used", "modalities_used", "fusion_sigma_xy_nm", "fusion_gain_xy",
            "mean_principal_angle_deg", "determinant_gain_vs_best_single", "fusion_singular",
            "fusion_interpretation", "physical_compatibility_status", "fusion_validation_status",
            "safe_for_fusion", "production_grid_diagnostic", "fusion_input_basis", "fusion_frame_count",
            "fisher_lateral_derivative_basis",
        ],
    )
    if dynamic_summaries:
        (output_dir / "dynamic_microscope_summary.json").write_text(
            json.dumps(json_safe(dynamic_summaries, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    resolved_dir = output_dir / "params_resolved_by_microscope"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    used_resolved_stems: set[str] = set()
    for microscope_name, resolved in resolved_by_microscope.items():
        artifact_stem = _artifact_stem(microscope_name, used_resolved_stems)
        (resolved_dir / f"{artifact_stem}.json").write_text(
            json.dumps(json_safe(resolved, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "params_base.json").write_text(
        json.dumps(json_safe(params, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "params_resolved_example.json").write_text(
        json.dumps(json_safe(resolved_example or params, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reported_modalities = [
        str(result.get("per_microscope", {}).get(name, {}).get("modality", modality_by_microscope.get(name, "")))
        for name in resolved_microscopes
    ]
    manifest = {
        "schema_version": "syniscopy-lab-fisher-report-v3-microscope",
        "comparison_unit": "microscope",
        "microscope_input_mode": microscope_input_mode,
        "microscopes_requested": requested_microscopes,
        "microscopes_reported": resolved_microscopes,
        "microscopes_failed": sorted(errors),
        "modality_by_microscope": modality_by_microscope,
        "modalities_requested": modalities,
        "modalities_reported": reported_modalities,
        "fusion_microscopes_used": fusion_microscopes_used,
        "fusion_modalities_used": [
            str(result.get("per_microscope", {}).get(name, {}).get("modality", modality_by_microscope.get(name, "")))
            for name in fusion_microscopes_used
        ],
        "fusion_profiles_excluded": fusion_duplicates,
        "fusion_input_basis": fusion_input_basis,
        "fusion_frame_count": fusion_frame_count,
        "fisher_lateral_derivative_basis": "spectral_band_limited",
        # Manifest frame counts are microscope-owned for explicit configuration
        # comparisons. A single sequence_frames scalar is retained as the base
        # request, but consumers that audit ranking provenance must use the
        # per-microscope map derived from the same summaries used for CRLB rows.
        "sequence_frames": base_num_frames,
        "sequence_frames_by_microscope": {
            name: int(result.get("per_microscope", {}).get(name, {}).get("num_frames", base_num_frames))
            for name in resolved_microscopes
        },
        "configuration_summary_basis": "params_resolved_by_microscope",
        "latent_scene_id": shared_latent_scene.provenance_id,
        "latent_schedule_id": shared_latent_scene.schedule.schedule_id,
        "latent_scene_time_policy": shared_latent_scene.schedule.policy,
        "fusion_time_alignment": shared_latent_scene.schedule.fusion_time_alignment,
        "latent_schedule_times_s": [
            float(value) for value in shared_latent_scene.schedule.times_s
        ],
        "same_latent_scene": True,
        "scene_coordinate_frame": report_scene_provenance.get("coordinate_frame", ""),
        "scene_fingerprint": report_scene_provenance.get("scene_fingerprint", ""),
        "scene_provenance": report_scene_provenance,
        "dynamic_bayesian_enabled": bool(args.dynamic_bayesian),
        "allow_partial": bool(args.allow_partial),
        "best_microscope_xy": result.get("best_microscope_xy"),
        "best_microscopes_xy": list(result.get("best_microscopes_xy", []) or []),
        "rankable_microscope_count": sum(1 for row in ranking if bool(row.get("rankable_for_ordering", False))),
        "ranking_status": "ranked" if any(bool(row.get("rankable_for_ordering", False)) for row in ranking) else "no_rankable_microscope_for_ordering",
        "output_files": {
            "report": "report.md",
            "sequence_fisher_summary": "sequence_fisher_summary.csv",
            "microscope_ranking": "microscope_ranking.csv",
            "fusion_crlb": "fusion_crlb.csv",
            "params_resolved_example": "params_resolved_example.json",
            "params_base": "params_base.json",
            "params_resolved_by_microscope": "params_resolved_by_microscope/",
            "dynamic_microscope_summary": "dynamic_microscope_summary.json" if dynamic_summaries else None,
            "render_errors": "render_errors.json" if errors else None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if errors:
        (output_dir / "render_errors.json").write_text(
            json.dumps(errors, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    _write_report(
        output_dir / "report.md",
        params=params,
        microscopes=resolved_microscopes,
        ranking_rows=ranking,
        fusion_rows=fusion,
        fusion_duplicates=fusion_duplicates,
        num_frames=base_num_frames,
        dynamic_requested=bool(args.dynamic_bayesian),
        sequence_summary_rows=len(sequence_rows),
        dynamic_summary_exists=bool(dynamic_summaries),
        errors=errors,
        wrote_previews=not args.no_previews,
        resolved_params_by_microscope=resolved_by_microscope,
    )
    if errors and not args.allow_partial:
        failed = ", ".join(sorted(errors))
        raise RuntimeError(
            f"Partial lab Fisher report written to {output_dir}, but requested "
            f"microscopes failed: {failed}. Re-run with --allow-partial to accept "
            "a partial report."
        )
    return output_dir

def main() -> int:
    args = _parse_args()
    try:
        if args.list_modalities:
            for modality in SUPPORTED_MODALITIES:
                print(modality)
            return 0
        if args.list_instruments:
            for name in sorted(get_instrument_preset_names()):
                print(name)
            return 0
        if args.write_template:
            path = _write_microscope_set_template(
                args.write_template,
                modality=args.template_modality,
            )
            print(f"Wrote lab Fisher microscope template: {path}")
            return 0
        if args.validate_microscopes:
            microscope_set, resolved = _validated_microscope_set_for_cli(
                args,
                args.validate_microscopes,
            )
            print(
                "Validated resolved microscope set: "
                f"{len(microscope_set.microscopes)} microscope(s), "
                f"{len(microscope_set.shared_params)} shared param(s), "
                f"{len(resolved)} resolved parameters record(s)."
            )
            return 0
        if args.list_microscopes:
            microscope_set, _validated = _validated_microscope_set_for_cli(
                args,
                args.list_microscopes,
            )
            for scope in microscope_set.microscopes:
                instrument = scope.instrument or "-"
                print(f"{scope.name}	{scope.modality}	{instrument}")
            return 0
        out = run_report(args)
        print(f"Lab Fisher report ready: {out}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
