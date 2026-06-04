"""CSV, Markdown, and managed-output writing for lab Fisher reports."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any

from config import param_value

__all__ = ["_clear_managed_outputs", "_format_float", "_write_csv", "_write_report"]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def _write_report(
    path: Path,
    *,
    params: dict[str, Any],
    modalities: list[str],
    ranking_rows: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
    fusion_duplicates: dict[str, dict[str, str]],
    num_frames: int,
    dynamic_requested: bool,
    sequence_summary_rows: int,
    dynamic_summary_exists: bool,
    errors: dict[str, str],
    wrote_previews: bool,
) -> None:
    top = ranking_rows[:5]
    lines = [
        "# Syniscopy Lab Fisher Report",
        "",
        "This report ranks candidate modality profiles for the same configured particle, pixel pitch, and detector-noise model.",
        "It is a model-conditional Fisher/CRLB diagnostic, not a guarantee of native instrument performance.",
        "",
        "## Configuration",
        "",
        f"- Pixel size: `{_format_float(param_value(params, 'pixel_size_nm'))}` nm",
        f"- Wavelength: `{_format_float(param_value(params, 'wavelength_nm'))}` nm",
        f"- Numerical aperture: `{_format_float(param_value(params, 'numerical_aperture'))}`",
        f"- Frames per modality: `{int(num_frames)}`",
        f"- Background intensity: `{_format_float(param_value(params, 'background_intensity'))}` counts",
        f"- Read noise: `{_format_float(param_value(params, 'read_noise_counts'))}` counts RMS",
        f"- Camera gain: `{_format_float(param_value(params, 'camera_gain_e_per_count'))}` e-/count",
        f"- Background subtraction: `{param_value(params, 'background_subtraction_method')}`",
        f"- Modalities requested: `{', '.join(modalities)}`",
        f"- Dynamic Bayesian CRLB requested: `{bool(dynamic_requested)}`",
        "",
        "## Lateral CRLB Ranking",
        "",
        "`sigma_xy_nm` is the 2D L2 bound, `sqrt(sigma_x^2 + sigma_y^2)`, not a one-axis precision.",
        "",
        "| Rank | Modality key | Display name | sigma_xy_nm | Relative | Frames to match best |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in top:
        lines.append(
            "| {rank} | `{modality}` | {display_name} | {sigma} | {relative} | {frames} |".format(
                rank=row["rank"],
                modality=row["modality"],
                display_name=row["display_name"],
                sigma=_format_float(row["sigma_xy_nm"]),
                relative=_format_float(row["relative_sigma_xy"]),
                frames=_format_float(row["frames_to_match_best_xy"]),
            )
        )
    lines.extend(
        [
            "",
            "Full ranking: `modality_ranking.csv`.",
            "`sequence_fisher_summary.csv` contains one row per frame and modality with static same-state cumulative CRLB; Brownian/moving-particle tracking claims require the dynamic Bayesian summary.",
            "",
            "## Fusion Diagnostic",
            "",
            "Fusion rows assume independent measurements of the same particle state and zero cross-channel registration covariance. Do not interpret a fusion gain as experimentally available when modalities reuse the same detected quanta, are alternate reconstructions of the same channel, or are physically incompatible for the sample.",
            "",
            "When two configured modality profiles produce numerically identical lateral Fisher matrices, this report keeps the first profile as the fusion representative and excludes the duplicate from the automatic fusion input.",
            "",
            "| Subset size | Modalities used | fusion_sigma_xy_nm | Gain | Mean principal angle | Safety |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fusion_rows:
        lines.append(
            "| {k} | {mods} | {sigma} | {gain} | {angle} | {safety} |".format(
                k=row["subset_size"],
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
        lines.extend(["", "Fusion duplicate profiles excluded from automatic fusion:", ""])
        for modality, info in sorted(fusion_duplicates.items()):
            lines.append(
                f"- `{modality}` represented by `{info['representative']}` "
                f"({info['reason']})."
            )
    if errors:
        lines.extend(["", "## Modalities Not Reported", ""])
        for modality, message in sorted(errors.items()):
            lines.append(f"- `{modality}`: {message}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `sequence_fisher_summary.csv`: {int(sequence_summary_rows)} rows of sequence CRLB progression.",
            "- `modality_ranking.csv`: per-modality Fisher matrix and lateral CRLB summary.",
            "- `fusion_crlb.csv`: best-k fusion rows for the rendered profiles; the full-library row is included only when requested.",
            "- `params_base.json`: base configuration before per-modality imaging-model overrides.",
            "- `params_resolved_by_modality/`: resolved configuration for each reported modality.",
            "- `manifest.json`: requested/reported/failed modality summary.",
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
                "- `dynamic_modality_summary.json`: optional dynamic Bayesian CRLB outputs by modality.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_managed_outputs(output_dir: Path) -> None:
    for name in (
        "previews",
        "fisher_density",
        "params_resolved_by_modality",
    ):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (
        "crlb_ranking.csv",
        "modality_ranking.csv",
        "fusion_crlb.csv",
        "sequence_fisher_summary.csv",
        "dynamic_modality_summary.json",
        "params_base.json",
        "params_resolved.json",
        "manifest.json",
        "render_errors.json",
        "report.md",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
