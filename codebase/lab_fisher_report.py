#!/usr/bin/env python3
"""Generate a small lab-facing Fisher/CRLB modality report.

This command is intentionally separate from the paper notebooks. It gives a
lab a short path from a microscope/particle configuration to the core
Syniscopy outputs: shared-scene renders, lateral Fisher matrices, CRLB
rankings, Fisher-density maps, and a small fusion diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_noise import analysis_contrast_noise_variance
from config import PARAMS, validate_params
from fisher_diagnostic import (
    compare_modality_information_content,
    compute_information_density_maps,
    compute_modality_fusion_crlb,
)
from imaging_model import SUPPORTED_MODALITIES, canonical_modality_name, modality_display_name
from main import generate_single_frame_views
from particle_specs import normalize_particle_specs
from postprocessing import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)
from presets import apply_instrument_preset, get_instrument_preset_names


LAB_DEFAULT_MODALITIES = (
    "bright_field",
    "coherent_bright_field",
    "fluorescence_widefield",
    "tirf_fluorescence",
    "dark_field",
    "coherent_dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
)

ELECTRON_MODALITIES = ("tem_phase_contrast", "sem_secondary_electron")


TEMPLATE_OVERRIDES: dict[str, Any] = {
    "image_size_pixels": 192,
    "pixel_size_nm": 65.0,
    "pupil_samples": 192,
    "psf_oversampling_factor": 2,
    "wavelength_nm": 532.0,
    "numerical_aperture": 1.0,
    "refractive_index_medium": 1.33,
    "refractive_index_immersion": 1.33,
    "background_intensity": 1000.0,
    "shot_noise_enabled": True,
    "gaussian_noise_enabled": True,
    "background_subtraction_method": "reference_frame",
    "read_noise_counts": 1.0,
    "camera_gain_e_per_count": 1.0,
    "sample_environment_enabled": False,
    "sample_environment_pattern_enabled": False,
    "sample_environment_pattern": "none",
    "particles": [
        {
            "name": "target_particle",
            "motion": {
                "hydrodynamic_diameter_nm": 100.0,
                "initial_position_nm": None,
            },
            "signal_multiplier": 1.0,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "fluorescent_polystyrene",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        }
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the same particle through candidate modalities and write a "
            "lab-facing Fisher/CRLB report."
        )
    )
    parser.add_argument(
        "--output",
        default="lab_reports/fisher_report",
        help="Output directory for CSVs, previews, maps, and report.md.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Path to a JSON file containing PARAMS overrides for the lab scenario.",
    )
    parser.add_argument(
        "--write-template",
        default=None,
        help="Write a small editable lab parameter template and exit.",
    )
    parser.add_argument(
        "--list-modalities",
        action="store_true",
        help="List supported modality names and exit.",
    )
    parser.add_argument(
        "--list-instruments",
        action="store_true",
        help="List available instrument preset names and exit.",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Optional instrument preset name from codebase/presets.py.",
    )
    parser.add_argument(
        "--modalities",
        default="lab-default",
        help=(
            "Comma-separated modality names, or one of: lab-default, optical, "
            "all. Default: lab-default."
        ),
    )
    parser.add_argument(
        "--include-electron",
        action="store_true",
        help="Append simplified TEM/SEM modes to lab-default/optical modality lists.",
    )
    parser.add_argument("--diameter-nm", type=float, default=None, help="Target-particle component diameter in nm.")
    parser.add_argument("--material", default=None, help="Target-particle component material label.")
    parser.add_argument("--pixel-size-nm", type=float, default=None, help="Effective sample-plane pixel size in nm.")
    parser.add_argument("--wavelength-nm", type=float, default=None, help="Optical wavelength in nm.")
    parser.add_argument("--na", type=float, default=None, help="Numerical aperture.")
    parser.add_argument("--background-counts", type=float, default=None, help="Mean reference/background level in camera counts.")
    parser.add_argument("--read-noise-counts", type=float, default=None, help="Gaussian read-noise RMS in camera counts.")
    parser.add_argument("--camera-gain-e-per-count", type=float, default=None, help="Detected electrons per camera count/ADU.")
    parser.add_argument("--image-size-pixels", type=int, default=None, help="Square frame width/height in pixels.")
    parser.add_argument("--pupil-samples", type=int, default=None, help="Pupil samples used for optical PSF calculation.")
    parser.add_argument("--z-nm", type=float, default=0.0, help="Target-particle axial position in nm; this report still computes lateral CRLBs.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for run-scoped deterministic choices.")
    parser.add_argument(
        "--max-fusion-k",
        type=int,
        default=4,
        help="Largest subset size for best-k fusion search. Default: 4.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip preview PNGs and Fisher-density PNG/NPY writes.",
    )
    parser.add_argument(
        "--include-full-fusion",
        action="store_true",
        help="Also write the full-library fusion row when max-fusion-k is smaller than the modality count.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Return success even if one or more requested modalities fail. "
            "By default, partial reports write render_errors.json and exit nonzero."
        ),
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    return value


def _write_template(path: str | Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_json_ready(TEMPLATE_OVERRIDES), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError("--params-json must contain one JSON object.")
    return value


def _resolve_modalities(spec: str, include_electron: bool) -> list[str]:
    text = str(spec).strip().lower()
    if text in {"lab-default", "default", "optical"}:
        modalities = list(LAB_DEFAULT_MODALITIES)
    elif text == "all":
        modalities = list(SUPPORTED_MODALITIES)
    else:
        modalities = [
            canonical_modality_name(part.strip())
            for part in text.split(",")
            if part.strip()
        ]
    if include_electron:
        for modality in ELECTRON_MODALITIES:
            if modality not in modalities:
                modalities.append(modality)
    supported = set(SUPPORTED_MODALITIES)
    unsupported = sorted(set(modalities) - supported)
    if unsupported:
        raise ValueError(
            f"Unsupported modalities: {unsupported}. Supported: {sorted(supported)}"
        )
    return list(dict.fromkeys(modalities))


def _apply_cli_overrides(params: dict[str, Any], args: argparse.Namespace) -> None:
    direct = {
        "pixel_size_nm": args.pixel_size_nm,
        "wavelength_nm": args.wavelength_nm,
        "numerical_aperture": args.na,
        "background_intensity": args.background_counts,
        "read_noise_counts": args.read_noise_counts,
        "camera_gain_e_per_count": args.camera_gain_e_per_count,
        "image_size_pixels": args.image_size_pixels,
        "pupil_samples": args.pupil_samples,
        "random_seed": args.seed,
    }
    for key, value in direct.items():
        if value is not None:
            params[key] = value

    params["return_ideal_float_frames"] = True
    params["save_frame_sequence"] = False
    params["save_raw_frame_views"] = False
    params["mask_generation_enabled"] = False
    params["num_frames"] = 1
    params["duration_seconds"] = 1.0 / float(params.get("fps", 30.0))
    params["background_subtraction_method"] = params.get(
        "background_subtraction_method", "reference_frame"
    )

    particles = params.get("particles")
    if not isinstance(particles, list) or not particles:
        particles = deepcopy(TEMPLATE_OVERRIDES["particles"])
        params["particles"] = particles
    first = particles[0]
    first.setdefault("motion", {})
    first.setdefault("components", deepcopy(TEMPLATE_OVERRIDES["particles"][0]["components"]))
    components = first.get("components") or []
    if not components:
        components = deepcopy(TEMPLATE_OVERRIDES["particles"][0]["components"])
        first["components"] = components
    component = components[0]

    if args.diameter_nm is not None:
        component["diameter_nm"] = float(args.diameter_nm)
        first["motion"]["hydrodynamic_diameter_nm"] = float(args.diameter_nm)
    if args.material is not None:
        component["material"] = str(args.material)

    side_nm = float(params["image_size_pixels"]) * float(params["pixel_size_nm"])
    first["motion"]["initial_position_nm"] = [
        0.5 * side_nm,
        0.5 * side_nm,
        float(args.z_nm),
    ]
    params["initial_z_span_nm"] = max(
        float(params.get("initial_z_span_nm", 4000.0)),
        2.0 * abs(float(args.z_nm)) + 1000.0,
    )


def _make_params(args: argparse.Namespace) -> dict[str, Any]:
    params = deepcopy(PARAMS)
    params.update(deepcopy(TEMPLATE_OVERRIDES))
    if args.instrument:
        params = apply_instrument_preset(params, args.instrument)
    if args.params_json:
        params.update(_load_json(args.params_json))
    _apply_cli_overrides(params, args)
    validate_params(params)
    specs = normalize_particle_specs(params, mutate=False)
    if len(specs) != 1:
        raise ValueError(
            "lab_fisher_report currently expects exactly one logical particle. "
            "For multi-particle scenes, generate a dataset or run a targeted crop workflow."
        )
    method = str(params.get("background_subtraction_method", "reference_frame")).strip().lower()
    if method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "lab_fisher_report uses single-frame Fisher calculations and does not accept "
            "background_subtraction_method='video_median'. Use 'reference_frame'."
        )
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "lab_fisher_report requires background_subtraction_method='reference_frame' "
            "so its contrast image and noise variance share the same analysis units."
        )
    if method not in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "Unsupported background_subtraction_method for lab_fisher_report: "
            f"{method!r}. Use 'reference_frame'."
        )
    return params


def _display_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    center = float(np.median(finite))
    spread = float(np.percentile(np.abs(finite - center), 99.5))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 1.0
    out = 0.5 + 0.42 * (arr - center) / spread
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def _density_uint8(density: np.ndarray) -> np.ndarray:
    arr = np.asarray(density, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    if float(arr.max(initial=0.0)) <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    logged = np.log1p(arr)
    hi = float(np.percentile(logged[logged > 0.0], 99.0)) if np.any(logged > 0.0) else 1.0
    if not np.isfinite(hi) or hi <= 0.0:
        hi = float(logged.max())
    return np.clip(255.0 * logged / hi, 0.0, 255.0).astype(np.uint8)


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


def _render_modality(
    base_params: dict[str, Any],
    modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    params = deepcopy(base_params)
    params["imaging_model"] = modality
    views = generate_single_frame_views(params)
    contrast = views.get("contrast_frame")
    signal = views.get("ideal_signal_frame")
    if signal is None:
        signal = views.get("raw_signal_frame")
    reference = views.get("ideal_reference_frame")
    if reference is None:
        reference = views.get("raw_reference_frame")
    preview = views.get("final_frame_8bit")
    if contrast is None:
        raise RuntimeError("modality did not produce a contrast frame")
    if signal is None:
        raise RuntimeError("modality did not produce a signal-count frame")
    noise_var = analysis_contrast_noise_variance(
        np.asarray(signal, dtype=float),
        None if reference is None else np.asarray(reference, dtype=float),
        views.get("params_resolved", params),
    )
    if preview is None:
        preview = _display_uint8(contrast)
    return (
        np.asarray(contrast, dtype=float),
        np.asarray(noise_var, dtype=float),
        np.asarray(preview, dtype=np.uint8),
        np.asarray(signal, dtype=float),
        views.get("params_resolved", params),
    )


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
                "relative_sigma_xy": float(result["relative_sigma_xy"][modality]),
                "frames_to_match_best_xy": float(result["frames_to_match_best_xy"][modality]),
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(contrasts)
    for k in range(1, max(1, min(max_k, n)) + 1):
        result = compute_modality_fusion_crlb(contrasts, noise, pixel_size_nm, subset_size=k)
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
            }
        )
    if include_full and n > 1 and max_k < n:
        result = compute_modality_fusion_crlb(contrasts, noise, pixel_size_nm, subset_size=n)
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


def _write_report(
    path: Path,
    *,
    params: dict[str, Any],
    modalities: list[str],
    ranking_rows: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
    fusion_duplicates: dict[str, dict[str, str]],
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
        f"- Pixel size: `{_format_float(params.get('pixel_size_nm'))}` nm",
        f"- Wavelength: `{_format_float(params.get('wavelength_nm'))}` nm",
        f"- Numerical aperture: `{_format_float(params.get('numerical_aperture'))}`",
        f"- Background intensity: `{_format_float(params.get('background_intensity'))}` counts",
        f"- Read noise: `{_format_float(params.get('read_noise_counts'))}` counts RMS",
        f"- Camera gain: `{_format_float(params.get('camera_gain_e_per_count'))}` e-/count",
        f"- Background subtraction: `{params.get('background_subtraction_method')}`",
        f"- Modalities requested: `{', '.join(modalities)}`",
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
            "",
            "## Fusion Diagnostic",
            "",
            "Fusion rows assume independent measurements of the same particle state and zero cross-channel registration covariance. Do not interpret a fusion gain as experimentally available when modalities reuse the same detected quanta, are alternate reconstructions of the same channel, or are physically incompatible for the sample.",
            "",
            "When two configured modality profiles produce numerically identical lateral Fisher matrices, this report keeps the first profile as the fusion representative and excludes the duplicate from the automatic fusion input.",
            "",
            "| Subset size | Modalities used | fusion_sigma_xy_nm | Gain | Mean principal angle |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in fusion_rows:
        lines.append(
            "| {k} | {mods} | {sigma} | {gain} | {angle} |".format(
                k=row["subset_size"],
                mods=row["modalities_used"],
                sigma=_format_float(row["fusion_sigma_xy_nm"]),
                gain=_format_float(row["fusion_gain_xy"]),
                angle=_format_float(row.get("mean_principal_angle_deg")),
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
                "- `previews/`: display-normalized single-frame contrast previews.",
                "- `fisher_density/`: per-pixel lateral Fisher-density images and arrays.",
            ]
        )
    else:
        lines.append("- Preview and Fisher-density image writes were skipped by `--no-previews`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(args: argparse.Namespace) -> Path:
    params = _make_params(args)
    modalities = _resolve_modalities(args.modalities, args.include_electron)

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
    errors: dict[str, str] = {}
    resolved_example: dict[str, Any] | None = None
    resolved_by_modality: dict[str, dict[str, Any]] = {}

    for modality in modalities:
        print(f"Rendering {modality}...")
        try:
            contrast, noise_var, preview, _signal, resolved = _render_modality(params, modality)
        except Exception as exc:  # keep other modalities useful
            errors[modality] = repr(exc)
            continue
        contrasts[modality] = contrast
        noise[modality] = noise_var
        if resolved_example is None:
            resolved_example = resolved
        resolved_by_modality[modality] = resolved
        if not args.no_previews:
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

    result = compare_modality_information_content(
        contrasts,
        noise,
        float(params["pixel_size_nm"]),
    )
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
            "relative_sigma_xy",
            "frames_to_match_best_xy",
            "fisher_xx",
            "fisher_xy",
            "fisher_yy",
            "singular",
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
        ],
    )
    resolved_dir = output_dir / "params_resolved_by_modality"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    for modality, resolved in resolved_by_modality.items():
        (resolved_dir / f"{modality}.json").write_text(
            json.dumps(_json_ready(resolved), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "params_base.json").write_text(
        json.dumps(_json_ready(params), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Backward-compatible convenience copy: this is the first successful
    # modality's resolved configuration. Use params_resolved_by_modality/ for
    # modality-specific audit.
    (output_dir / "params_resolved.json").write_text(
        json.dumps(_json_ready(resolved_example or params), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "syniscopy-lab-fisher-report-v1",
        "modalities_requested": modalities,
        "modalities_reported": list(contrasts),
        "modalities_failed": sorted(errors),
        "fusion_modalities_used": list(fusion_contrasts),
        "fusion_duplicate_profiles_excluded": fusion_duplicates,
        "allow_partial": bool(args.allow_partial),
        "output_files": {
            "report": "report.md",
            "modality_ranking": "modality_ranking.csv",
            "fusion_crlb": "fusion_crlb.csv",
            "params_resolved_example": "params_resolved.json",
            "params_base": "params_base.json",
            "params_resolved_by_modality": "params_resolved_by_modality/",
            "render_errors": "render_errors.json" if errors else None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
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
        "params_base.json",
        "params_resolved.json",
        "manifest.json",
        "render_errors.json",
        "report.md",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


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


if __name__ == "__main__":
    raise SystemExit(main())
