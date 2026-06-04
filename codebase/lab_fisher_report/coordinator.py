"""Coordinator for lab Fisher report generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bootstrap import REPO_ROOT, ensure_codebase_on_path

ensure_codebase_on_path()

from fisher import compute_information_density_maps
from imaging_models import get_imaging_model
from json_utils import json_safe
from modality_profiles import profile_card_for_model
from modality_registry import SUPPORTED_MODALITIES, modality_display_name
from presets import get_instrument_preset_names

from .cli import _parse_args, _write_template
from .params_assembly import _make_params, _resolve_modalities
from .render import _density_uint8, _display_uint8, _render_modality
from .report_writer import _clear_managed_outputs, _write_csv, _write_report
from .tables import (
    _build_sequence_summary_rows,
    _compute_dynamic_sequence_summary,
    _fusion_rows,
    _ranking_rows,
    _select_fusion_inputs,
    _sequence_information_content,
)

__all__ = ["main", "run_report"]


def run_report(args: argparse.Namespace) -> Path:
    params = _make_params(args)
    modalities = _resolve_modalities(args.modalities)

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

    contrasts: dict[str, np.ndarray] = {}
    noise: dict[str, np.ndarray] = {}
    sequence_rows: list[dict[str, Any]] = []
    sequence_summaries: dict[str, dict[str, Any]] = {}
    dynamic_summaries: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    resolved_modalities: list[str] = []
    resolved_example: dict[str, Any] | None = None
    resolved_by_modality: dict[str, dict[str, Any]] = {}
    profile_cards_by_modality: dict[str, dict[str, Any]] = {}

    for modality in modalities:
        print(f"Rendering {modality} ({int(params['num_frames'])} frame(s))...")
        try:
            rendered, render_meta = _render_modality(params, modality)
        except Exception as exc:  # keep other modalities useful
            errors[modality] = repr(exc)
            continue

        resolved = render_meta["resolved_params"]
        resolved_modalities.append(modality)
        num_frames = int(render_meta.get("num_frames", 1))
        per_frame = rendered.get("per_frame", [])
        fisher_matrices = rendered.get("fisher_matrices", [])
        if not per_frame:
            errors[modality] = f"no per-frame records for modality {modality!r}"
            continue
        if len(per_frame) != num_frames:
            errors[modality] = (
                f"reported {len(per_frame)} frames, expected {num_frames} for modality {modality!r}"
            )
            continue

        frame_rows, summary = _build_sequence_summary_rows(modality, per_frame)
        sequence_rows.extend(frame_rows)
        sequence_summaries[modality] = summary

        first = per_frame[0]
        contrast = np.asarray(first["contrast"], dtype=float)
        noise_var = np.asarray(first["noise_variance"], dtype=float)
        contrasts[modality] = contrast
        noise[modality] = noise_var
        if resolved_example is None:
            resolved_example = resolved
        resolved_by_modality[modality] = resolved
        try:
            profile_cards_by_modality[modality] = profile_card_for_model(
                resolved,
                get_imaging_model(resolved),
                modality_name=modality,
            )
        except Exception:
            profile_cards_by_modality.pop(modality, None)

        if bool(args.dynamic_bayesian):
            try:
                dynamic_summary = _compute_dynamic_sequence_summary(
                    modality,
                    fisher_matrices,
                    resolved,
                    per_frame,
                )
            except Exception as exc:
                dynamic_summary = {
                    "modality": modality,
                    "dynamic_enabled": bool(args.dynamic_bayesian),
                    "dynamic_summary_available": False,
                    "dynamic_error": repr(exc),
                }
            dynamic_summaries[modality] = dynamic_summary

        if not args.no_previews:
            preview = _display_uint8(contrast)
            Image.fromarray(preview).save(preview_dir / f"{modality}.png")
            maps = compute_information_density_maps(
                contrast,
                noise_var,
                float(params["pixel_size_nm"]),
            )
            lateral_density = maps["Ix_info_map"] + maps["Iy_info_map"]
            np.save(density_dir / f"{modality}_lateral_density.npy", lateral_density)
            Image.fromarray(_density_uint8(lateral_density)).save(
                density_dir / f"{modality}_lateral_density.png"
            )

    if not contrasts:
        raise RuntimeError(f"No modalities rendered successfully. Errors: {errors}")

    result = _sequence_information_content(resolved_modalities, sequence_summaries)
    ranking = _ranking_rows(result)
    fusion_contrasts, fusion_noise, fusion_duplicates = _select_fusion_inputs(
        contrasts,
        noise,
        result,
    )
    fusion = _fusion_rows(
        fusion_contrasts,
        fusion_noise,
        float(params["pixel_size_nm"]),
        max_k=max(1, int(args.max_fusion_k)),
        include_full=bool(args.include_full_fusion),
        modality_profile_cards={
            modality: profile_cards_by_modality[modality]
            for modality in fusion_contrasts
            if modality in profile_cards_by_modality
        },
    )

    _write_csv(
        output_dir / "modality_ranking.csv",
        ranking,
        [
            "rank",
            "modality",
            "display_name",
            "sigma_xy_nm",
            "sigma_x_nm",
            "sigma_y_nm",
            "measurement_domain",
            "signal_units",
            "noise_variance_units",
            "detector_noise_input_domain",
            "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active",
            "safe_for_linear_fisher_variance",
            "fisher_variance_model_scope",
            "detector_likelihood_status",
            "relative_sigma_xy",
            "frames_to_match_best_xy",
            "num_frames",
            "fisher_xx",
            "fisher_xy",
            "fisher_yy",
            "singular",
        ],
    )
    _write_csv(
        output_dir / "sequence_fisher_summary.csv",
        sequence_rows,
        [
            "modality",
            "frame_index",
            "num_frames",
            "frame_fisher_xx",
            "frame_fisher_xy",
            "frame_fisher_yy",
            "frame_fisher_det",
            "frame_fisher_singular",
            "frame_fisher_rank",
            "frame_sigma_x_nm",
            "frame_sigma_y_nm",
            "frame_sigma_xy_nm",
            "measurement_domain",
            "signal_units",
            "noise_variance_units",
            "detector_noise_input_domain",
            "nonlinear_detector_effects_active",
            "deterministic_detector_transfer_active",
            "safe_for_linear_fisher_variance",
            "fisher_variance_model_scope",
            "detector_likelihood_status",
            "cumulative_fisher_xx",
            "cumulative_fisher_xy",
            "cumulative_fisher_yy",
            "cumulative_fisher_det",
            "cumulative_fisher_rank",
            "cumulative_sigma_x_nm",
            "cumulative_sigma_y_nm",
            "cumulative_sigma_xy_nm",
        ],
    )
    _write_csv(
        output_dir / "fusion_crlb.csv",
        fusion,
        [
            "subset_size",
            "modalities_used",
            "fusion_sigma_xy_nm",
            "fusion_gain_xy",
            "mean_principal_angle_deg",
            "determinant_gain_vs_best_single",
            "fusion_singular",
            "fusion_interpretation",
            "physical_compatibility_status",
            "fusion_validation_status",
            "safe_for_fusion",
            "production_grid_diagnostic",
        ],
    )
    if dynamic_summaries:
        (output_dir / "dynamic_modality_summary.json").write_text(
            json.dumps(json_safe(dynamic_summaries, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    resolved_dir = output_dir / "params_resolved_by_modality"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    for modality, resolved in resolved_by_modality.items():
        (resolved_dir / f"{modality}.json").write_text(
            json.dumps(json_safe(resolved, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "params_base.json").write_text(
        json.dumps(json_safe(params, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Backward-compatible convenience copy: this is the first successful
    # modality's resolved configuration. Use params_resolved_by_modality/ for
    # modality-specific audit.
    (output_dir / "params_resolved.json").write_text(
        json.dumps(json_safe(resolved_example or params, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "syniscopy-lab-fisher-report-v2-sequence",
        "modalities_requested": modalities,
        "modalities_reported": resolved_modalities,
        "modalities_failed": sorted(errors),
        "fusion_modalities_used": list(fusion_contrasts),
        "fusion_duplicate_profiles_excluded": fusion_duplicates,
        "sequence_frames": int(params["num_frames"]),
        "dynamic_bayesian_enabled": bool(args.dynamic_bayesian),
        "allow_partial": bool(args.allow_partial),
        "output_files": {
            "report": "report.md",
            "sequence_fisher_summary": "sequence_fisher_summary.csv",
            "modality_ranking": "modality_ranking.csv",
            "fusion_crlb": "fusion_crlb.csv",
            "params_resolved_example": "params_resolved.json",
            "params_base": "params_base.json",
            "params_resolved_by_modality": "params_resolved_by_modality/",
            "dynamic_modality_summary": "dynamic_modality_summary.json" if dynamic_summaries else None,
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
        modalities=modalities,
        ranking_rows=ranking,
        fusion_rows=fusion,
        fusion_duplicates=fusion_duplicates,
        num_frames=int(params["num_frames"]),
        dynamic_requested=bool(args.dynamic_bayesian),
        sequence_summary_rows=len(sequence_rows),
        dynamic_summary_exists=bool(dynamic_summaries),
        errors=errors,
        wrote_previews=not args.no_previews,
    )
    if errors and not args.allow_partial:
        failed = ", ".join(sorted(errors))
        raise RuntimeError(
            f"Partial lab Fisher report written to {output_dir}, but requested "
            f"modalities failed: {failed}. Re-run with --allow-partial to accept "
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
            path = _write_template(args.write_template)
            print(f"Wrote lab Fisher template: {path}")
            return 0
        out = run_report(args)
        print(f"Lab Fisher report ready: {out}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
