"""CSV, Markdown, and managed-output writing for lab Fisher reports."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from config import (
    AcquisitionProfile,
    BackgroundSubtractionSettings,
    CountBudgetSettings,
    ModalitySettings,
    OpticalInstrumentSettings,
    SamplingGeometry,
)
from .scene_view import scene_provenance_from_params

__all__ = ["_clear_managed_outputs", "_format_float", "_write_csv", "_write_report"]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        # Public report CSV headers are part of the user-facing recommendation
        # contract.  Duplicate names make fields such as ranking_status ambiguous
        # for spreadsheets and automated validators even when row values match.
        raise ValueError(
            f"CSV fieldnames for {path.name!r} must be unique; duplicate field(s): {duplicates}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _noise_config_for_report(params: Mapping[str, Any]):
    from camera_noise import CameraNoiseConfig

    return CameraNoiseConfig.from_params(dict(params))


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(val):
        return "nan"
    if math.isinf(val):
        return "inf"
    return f"{val:.{digits}g}"



def _write_scene_provenance_section(lines: list[str], params: Mapping[str, Any]) -> None:
    """Append the shared scene-view contract that makes ranking comparable."""

    provenance = scene_provenance_from_params(params)
    if not provenance:
        return
    target = provenance.get("target_initial_position_nm", [])
    target_text = ", ".join(_format_float(v) for v in target) if isinstance(target, list) else ""
    lines.extend(
        [
            f"- Shared scene coordinate frame: `{provenance.get('coordinate_frame', '')}`",
            f"- Shared scene fingerprint: `{provenance.get('scene_fingerprint', '')}`",
            f"- Shared target initial position: `[{target_text}]` nm",
            f"- Shared target position defaulted from template: `{bool(provenance.get('target_position_defaulted', False))}`",
        ]
    )


def _write_microscope_configuration_table(
    lines: list[str],
    *,
    microscopes: list[str],
    resolved_params_by_microscope: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Append the per-microscope configuration contract exposed in report.md.

    The lab report ranks microscope configurations, not just modalities. Rank-
    determining values such as pixel size, wavelength, NA, detector counts, and
    frame count may differ after per-microscope overlays/instrument presets. A
    single global parameters row is therefore not a safe report contract; this table
    is generated from the same resolved params dictionaries that produced the
    Fisher/CRLB rows so report.md cannot imply that different configurations
    shared one hidden acquisition basis.
    """

    if not resolved_params_by_microscope:
        lines.extend(
            [
                "",
                "Resolved per-microscope configuration was not supplied to the report writer; see `params_resolved_by_microscope/` when available.",
            ]
        )
        return

    reported = [name for name in microscopes if name in resolved_params_by_microscope]
    if not reported:
        lines.extend(
            [
                "",
                "No resolved per-microscope configuration rows were available for the reported microscopes.",
            ]
        )
        return

    lines.extend(
        [
            "",
            "Resolved per-microscope configuration summary. These values are the report-facing acquisition basis used for each candidate's Fisher/CRLB calculation.",
            "",
            "| Microscope | Modality | pixel nm | wavelength nm | NA | frames | fps | duration s | background counts | read noise counts | gain e-/count |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for microscope in reported:
        resolved = resolved_params_by_microscope[microscope]
        modality = ModalitySettings.from_params(resolved).modality
        sampling = SamplingGeometry.from_params(resolved)
        instrument = OpticalInstrumentSettings.from_params(resolved)
        acquisition = AcquisitionProfile.from_params(resolved)
        count_budget = CountBudgetSettings.from_params(resolved)
        noise_config = _noise_config_for_report(resolved)
        lines.append(
            "| `{microscope}` | `{modality}` | {pixel} | {wavelength} | {na} | {frames} | {fps} | {duration} | {background} | {read_noise} | {gain} |".format(
                microscope=microscope,
                modality=modality,
                pixel=_format_float(sampling.detector_pixel_size_nm),
                wavelength=_format_float(instrument.wavelength_nm),
                na=_format_float(instrument.numerical_aperture),
                frames=_format_float(acquisition.num_frames, digits=6),
                fps=_format_float(acquisition.fps),
                duration=_format_float(acquisition.duration_seconds),
                background=_format_float(count_budget.background_intensity),
                read_noise=_format_float(noise_config.read_noise_counts),
                gain=_format_float(noise_config.camera_gain_e_per_count),
            )
        )


def _write_report(
    path: Path,
    *,
    params: dict[str, Any],
    microscopes: list[str],
    ranking_rows: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
    fusion_duplicates: dict[str, dict[str, str]],
    num_frames: int,
    dynamic_requested: bool,
    sequence_summary_rows: int,
    dynamic_summary_exists: bool,
    errors: dict[str, str],
    wrote_previews: bool,
    resolved_params_by_microscope: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    rankable_rows = [
        row for row in ranking_rows if bool(row.get("rankable_for_ordering", False))
    ]
    diagnostic_rows = [
        row for row in ranking_rows if not bool(row.get("rankable_for_ordering", False))
    ]
    top = rankable_rows[:5]
    noise_config = _noise_config_for_report(params)
    instrument = OpticalInstrumentSettings.from_params(params)
    lines = [
        "# Syniscopy Lab Fisher Report",
        "",
        "This report ranks candidate microscopes for the same configured particle/scene; each microscope carries its own modality/backend metadata and resolved instrument parameters.",
        "It is a model-conditional Fisher/CRLB diagnostic, not a guarantee of native instrument performance.",
        "",
        "## Configuration",
        "",
        "The base/shared values below are provenance defaults. The resolved per-microscope table is the authoritative acquisition basis for ranking when candidates differ by instrument or overlay.",
        "",
        f"- Base/shared pixel size: `{_format_float(SamplingGeometry.from_params(params).detector_pixel_size_nm)}` nm",
        f"- Base/shared wavelength: `{_format_float(instrument.wavelength_nm)}` nm",
        f"- Base/shared numerical aperture: `{_format_float(instrument.numerical_aperture)}`",
        f"- Base/shared frame request: `{int(num_frames)}`",
        f"- Base/shared background intensity: `{_format_float(CountBudgetSettings.from_params(params).background_intensity)}` counts",
        f"- Base/shared read noise: `{_format_float(noise_config.read_noise_counts)}` counts RMS",
        f"- Base/shared camera gain: `{_format_float(noise_config.camera_gain_e_per_count)}` e-/count",
        f"- Background subtraction: `{BackgroundSubtractionSettings.from_params(params).method}`",
        f"- Microscopes reported: `{', '.join(microscopes)}`",
        f"- Dynamic Bayesian CRLB requested: `{bool(dynamic_requested)}`",
    ]
    _write_scene_provenance_section(lines, params)
    lines.extend([
        "",
        "## Lateral CRLB Ranking",
        "",
        "`sigma_xy_nm` is the 2D L2 bound, `sqrt(sigma_x^2 + sigma_y^2)`, not a one-axis precision.",
    ])
    _write_microscope_configuration_table(
        lines,
        microscopes=microscopes,
        resolved_params_by_microscope=resolved_params_by_microscope,
    )

    if top:
        lines.extend(
            [
                "",
                "| Rank | Microscope | Modality | Display name | sigma_xy_nm | Relative | Frames to match best |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in top:
            lines.append(
                "| {rank} | `{microscope}` | `{modality}` | {display_name} | {sigma} | {relative} | {frames} |".format(
                    rank=row["rank"],
                    microscope=row.get("microscope", ""),
                    modality=row["modality"],
                    display_name=row["display_name"],
                    sigma=_format_float(row["sigma_xy_nm"]),
                    relative=_format_float(row["relative_sigma_xy"]),
                    frames=_format_float(row["frames_to_match_best_xy"]),
                )
            )
    else:
        # A blank ranking table is intentional here: Rank is a user-facing
        # recommendation ordinal and must not be emitted for rows whose
        # Fisher/noise/sequence contract marked them diagnostic-only.
        lines.extend(
            [
                "",
                "No rankable microscope ordering is available for this configuration.",
                "Rendered microscopes are listed below as diagnostics only; see `microscope_ranking.csv` and `sequence_fisher_summary.csv` for status details.",
                "",
                "| Microscope | Modality | Display name | sigma_xy_nm | Ranking status |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for row in diagnostic_rows[:5]:
            lines.append(
                "| `{microscope}` | `{modality}` | {display_name} | {sigma} | {status} |".format(
                    microscope=row.get("microscope", ""),
                    modality=row["modality"],
                    display_name=row["display_name"],
                    sigma=_format_float(row["sigma_xy_nm"]),
                    status=str(row.get("ranking_status") or row.get("status_reason") or "diagnostic only"),
                )
            )
    lines.extend(
        [
            "",
            "Full ranking/diagnostic table: `microscope_ranking.csv`.",
            "`sequence_fisher_summary.csv` contains one row per frame and microscope. Multi-frame static same-state cumulative CRLB rows are diagnostic unless the dynamic Bayesian estimator is enabled and used as the ranking/fusion source.",
            "",
            "## Fusion Diagnostic",
            "",
            "Fusion rows assume independent measurements of the same particle state and zero cross-channel registration covariance. Do not interpret a fusion gain as experimentally available when microscopes reuse the same detected quanta, are alternate reconstructions of the same channel, or are physically incompatible for the sample.",
            "",
            "When no microscope is fusion-eligible, the correct report-facing result is an empty fusion table. This commonly occurs for multi-frame static diagnostic summaries that require the dynamic Bayesian estimator before ranking or fusion.",
            "",
            "Automatic fusion exclusions are based on sequence/noise safety and physical-compatibility metadata. Independent microscope candidates are not collapsed merely because their lateral Fisher matrices are numerically identical.",
            "",
            "For dynamic Bayesian sequences, fusion is computed by summing per-frame measurement Fisher matrices and running one joint dynamic estimator; per-microscope posterior precision matrices are not treated as independent fusion channels.",
            "",
        ]
    )
    if not fusion_rows:
        # Zero eligible fusion inputs is a valid comparison outcome, not a
        # rendering failure.  The ranking/report path must say this explicitly
        # so a user does not mistake an empty fusion section for missing output
        # or for a hidden best-candidate recommendation.
        lines.extend(
            [
                "No fusion-eligible microscope set is available for this configuration.",
                "See `microscope_ranking.csv` and the exclusion list below for the sequence/noise contract that kept each rendered microscope out of automatic fusion.",
                "",
            ]
        )
    lines.extend(
        [
            "| Subset size | Microscopes used | Modalities used | fusion_sigma_xy_nm | Gain | Mean principal angle | Safety |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fusion_rows:
        lines.append(
            "| {k} | {microscopes} | {mods} | {sigma} | {gain} | {angle} | {safety} |".format(
                k=row["subset_size"],
                microscopes=row.get("microscopes_used", ""),
                mods=row["modalities_used"],
                sigma=_format_float(row["fusion_sigma_xy_nm"]),
                gain=_format_float(row["fusion_gain_xy"]),
                angle=_format_float(row.get("mean_principal_angle_deg")),
                safety=(
                    "safe"
                    if str(row.get("safe_for_fusion", "")).strip().lower()
                    in {"1", "true", "yes", "pass", "passed"}
                    else "diagnostic only"
                ),
            )
        )
    if fusion_duplicates:
        lines.extend(["", "Fusion profiles excluded from automatic fusion:", ""])
        for microscope, info in sorted(fusion_duplicates.items()):
            representative = str(info.get("representative", "")).strip()
            reason = str(info.get("reason", "not fusion-eligible")).strip()
            if representative:
                lines.append(
                    f"- `{microscope}` represented by `{representative}` "
                    f"({reason})."
                )
            else:
                lines.append(f"- `{microscope}` not fused ({reason}).")
    if errors:
        lines.extend(["", "## Microscopes Not Reported", ""])
        for microscope, message in sorted(errors.items()):
            lines.append(f"- `{microscope}`: {message}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `sequence_fisher_summary.csv`: {int(sequence_summary_rows)} rows of sequence CRLB progression.",
            "- `microscope_ranking.csv`: per-microscope Fisher matrix and lateral CRLB summary.",
            "- `fusion_crlb.csv`: best-k fusion rows for the rendered profiles; the full-library row is included only when requested.",
            "- `params_base.json`: shared/base configuration used for report provenance; explicit microscope runs apply per-microscope instruments before shared overlays.",
            "- `params_resolved_by_microscope/`: resolved normalized configuration for each reported microscope.",
            "- `manifest.json`: requested/reported/failed microscope summary.",
        ]
    )
    if wrote_previews:
        lines.extend(
            [
                "- `previews/`: display-normalized first-frame contrast previews.",
                "- `fisher_density/`: per-pixel lateral Fisher-density images and arrays.",
            ]
        )
    else:
        lines.append("- Preview and Fisher-density image writes were skipped by `--no-previews`.")

    if dynamic_summary_exists:
        lines.extend(
            [
                "",
                "- `dynamic_microscope_summary.json`: optional dynamic Bayesian CRLB outputs by microscope.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_managed_outputs(output_dir: Path) -> None:
    for name in (
        "previews",
        "fisher_density",
        "params_resolved_by_microscope",
    ):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "microscope_ranking.csv",
        "fusion_crlb.csv",
        "sequence_fisher_summary.csv",
        "dynamic_microscope_summary.json",
        "params_base.json",
        "params_resolved.json",
        "params_resolved_example.json",
        "manifest.json",
        "render_errors.json",
        "report.md",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
