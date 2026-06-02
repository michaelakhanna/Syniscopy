"""Normalize E07 condition manifests so named perturbations are explicit.

The E07 notebook can supply defaults when fields are blank, but the source
manifests should not rely on silent defaulting for named conditions such as
``lower_na`` or ``green``. This script rewrites the JSON/CSV manifests with
explicit resolved values and records which fields were resolved from the
condition label/default contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "supplemental" / "E07_training_conditions.json"
CSV_PATH = ROOT / "supplemental" / "E07_training_condition_index.csv"

DEFAULTS = {
    "na": 1.2,
    "wavelength_nm": 520.0,
    "exposure_ms": 25.0,
    "viscosity_pa_s": 0.0015,
    "empirical_std": 0.004,
    "empirical_gradient": 0.002,
}

TEXTURE_PROFILES = {
    "default": ([16.0, 64.0, 256.0], [0.4, 0.35, 0.25]),
    "low": ([16.0, 64.0, 256.0], [0.4, 0.35, 0.25]),
    "medium": ([16.0, 64.0, 256.0], [0.4, 0.35, 0.25]),
    "strong": ([16.0, 64.0, 256.0], [0.4, 0.35, 0.25]),
    "fine": ([8.0, 24.0, 64.0], [0.5, 0.35, 0.15]),
    "coarse": ([32.0, 96.0, 256.0], [0.25, 0.4, 0.35]),
}


def _present(value: Any) -> bool:
    return value not in (None, "")


def _texture_profile_for_name(name: str) -> str:
    if "coarse_texture" in name:
        return "coarse"
    if "fine_texture" in name or "fine_residual_texture" in name:
        return "fine"
    if "stronger_texture" in name:
        return "strong"
    if "medium_texture" in name or "moderate_residual_texture" in name:
        return "medium"
    if "low_texture" in name or "lower_residual_texture" in name:
        return "low"
    return "default"


def _resolved_values(condition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    name = str(condition.get("name") or "")
    resolved = dict(DEFAULTS)
    reasons = {key: "default_e07_caustic_training_contract" for key in DEFAULTS}

    if "lower_na" in name:
        resolved["na"] = 1.05
        reasons["na"] = "condition_name_lower_na"
    elif "higher_na" in name:
        resolved["na"] = 1.3
        reasons["na"] = "condition_name_higher_na"

    if "green" in name:
        resolved["wavelength_nm"] = 500.0
        reasons["wavelength_nm"] = "condition_name_green"
    elif "yellow" in name:
        resolved["wavelength_nm"] = 560.0
        reasons["wavelength_nm"] = "condition_name_yellow"

    if "short_exposure" in name:
        resolved["exposure_ms"] = 12.0
        reasons["exposure_ms"] = "condition_name_short_exposure"

    if "high_viscosity" in name or "slower_brownian" in name:
        resolved["viscosity_pa_s"] = 0.004
        reasons["viscosity_pa_s"] = "condition_name_high_viscosity_or_slow_brownian"

    if "no_residual_texture" in name:
        resolved["empirical_std"] = 0.0
        resolved["empirical_gradient"] = 0.0
        reasons["empirical_std"] = "condition_name_no_residual_texture"
        reasons["empirical_gradient"] = "condition_name_no_residual_texture"
    elif "lower_residual_texture" in name or "low_texture" in name:
        resolved["empirical_std"] = 0.002
        resolved["empirical_gradient"] = 0.001
        reasons["empirical_std"] = "condition_name_low_texture"
        reasons["empirical_gradient"] = "condition_name_low_texture"
    elif "moderate_residual_texture" in name or "medium_texture" in name:
        resolved["empirical_std"] = 0.008
        resolved["empirical_gradient"] = 0.004
        reasons["empirical_std"] = "condition_name_medium_texture"
        reasons["empirical_gradient"] = "condition_name_medium_texture"
    elif "fine_residual_texture" in name or "fine_texture" in name:
        resolved["empirical_std"] = 0.010
        resolved["empirical_gradient"] = 0.004
        reasons["empirical_std"] = "condition_name_fine_texture"
        reasons["empirical_gradient"] = "condition_name_fine_texture"
    elif "stronger_texture" in name:
        resolved["empirical_std"] = 0.012
        resolved["empirical_gradient"] = 0.006
        reasons["empirical_std"] = "condition_name_stronger_texture"
        reasons["empirical_gradient"] = "condition_name_stronger_texture"

    return resolved, reasons


def _source_from_family(family: str) -> str:
    if family == "lower":
        return "lower_bound_sweep"
    if family == "hard_fast":
        return "hard_fast_sweep"
    return "one_frame_matching"


def _normalize_condition(condition: dict[str, Any]) -> dict[str, Any]:
    out = dict(condition)
    name = str(out.get("name") or "")
    family = str(out.get("approved_family") or "")
    resolved, reasons = _resolved_values(out)
    resolution: dict[str, str] = {}

    for key, value in resolved.items():
        if not _present(out.get(key)):
            out[key] = value
            resolution[key] = reasons[key]

    if "small_vibration" in name or name.endswith("_vibration"):
        if not _present(out.get("vibration_nm")) or float(out.get("vibration_nm") or 0.0) == 0.0:
            out["vibration_nm"] = 12.0
            resolution["vibration_nm"] = "condition_name_vibration"

    if "stage_drift" in name:
        if not _present(out.get("drift_nm_s")) or out.get("drift_nm_s") == [0.0, 0.0, 0.0]:
            out["drift_nm_s"] = [45.0, -25.0, 0.0]
            resolution["drift_nm_s"] = "condition_name_stage_drift"

    profile_name = _texture_profile_for_name(name)
    scales, weights = TEXTURE_PROFILES[profile_name]
    if not _present(out.get("empirical_background_scales_px")):
        out["empirical_background_scales_px"] = scales
        resolution["empirical_background_scales_px"] = f"texture_profile_{profile_name}"
    if not _present(out.get("empirical_background_scale_weights")):
        out["empirical_background_scale_weights"] = weights
        resolution["empirical_background_scale_weights"] = f"texture_profile_{profile_name}"

    if not _present(out.get("condition_source")):
        out["condition_source"] = _source_from_family(family)
        resolution["condition_source"] = "approved_family_source_mapping"

    image_size = int(out.get("image_size_px") or 256)
    pixel_size = float(out.get("pixel_size_nm") or 116.0)
    z_nm = float(out.get("z_nm") or 0.0)
    if not _present(out.get("source_initial_position_nm")):
        half_side = 0.5 * image_size * pixel_size
        out["source_initial_position_nm"] = [half_side, half_side, z_nm]
        resolution["source_initial_position_nm"] = "computed_from_image_size_pixel_size_and_z"

    if not _present(out.get("source_metadata")):
        out["source_metadata"] = "condition_manifest_derived_without_source_video_metadata"
        resolution["source_metadata"] = "explicit_no_source_metadata_available"

    existing_resolution = dict(out.get("parameter_resolution") or {})
    existing_resolution.update(resolution)
    out["parameter_resolution"] = existing_resolution
    return out


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def main() -> None:
    conditions = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    normalized = [_normalize_condition(dict(condition)) for condition in conditions]
    JSON_PATH.write_text(json.dumps(normalized, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    row_by_id = {str(row.get("condition_id")): row for row in rows}
    if set(row_by_id) != {str(c.get("condition_id")) for c in normalized}:
        raise RuntimeError("E07 JSON/CSV condition_id sets differ; refusing to rewrite CSV.")

    extra_fields = [
        "empirical_background_scales_px",
        "empirical_background_scale_weights",
        "parameter_resolution",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            insert_at = fieldnames.index("background_method") if field == "parameter_resolution" else fieldnames.index("background_method")
            fieldnames.insert(insert_at, field)

    for condition in normalized:
        row = row_by_id[str(condition.get("condition_id"))]
        for key, value in condition.items():
            if key not in fieldnames:
                fieldnames.append(key)
            row[key] = _csv_value(value)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for condition in normalized:
            writer.writerow(row_by_id[str(condition.get("condition_id"))])


if __name__ == "__main__":
    main()
