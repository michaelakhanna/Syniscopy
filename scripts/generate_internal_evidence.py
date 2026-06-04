#!/usr/bin/env python3
"""Generate lightweight source-owned evidence artifacts without notebooks.

This script is intentionally small and CPU-bounded by default. It does not
replace E01-E09, but it gives maintainers a source-level command for producing
profile-card, strict detected-count, and fusion-compatibility evidence before
spending time on full supplemental notebook runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from camera_noise import analysis_contrast_noise_variance  # noqa: E402
from config import PARAMS  # noqa: E402
from fisher import (  # noqa: E402
    compare_modality_information_content_detected_quanta_normalized,
    compute_modality_fusion_crlb,
)
from imaging_models import get_imaging_model  # noqa: E402
from json_utils import json_safe_with_nonfinite_tags  # noqa: E402
from modality_compatibility import fusion_subset_metadata  # noqa: E402
from modality_profiles import profile_card_for_model  # noqa: E402
from simulation import generate_single_frame_views  # noqa: E402


DEFAULT_MODALITIES = [
    "bright_field",
    "fluorescence_widefield",
    "quantitative_phase",
    "ricm",
    "differential_phase_contrast",
    "zernike_phase_contrast",
    "tem_phase_contrast",
    "sem_secondary_electron",
]
DEFAULT_OUTPUT_DIR = "throw" + "out/evidence/internal"


def _json_safe(value: Any) -> Any:
    return json_safe_with_nonfinite_tags(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_params(modality: str, image_size: int, pupil_samples: int) -> dict[str, Any]:
    params = dict(PARAMS)
    side_nm = float(image_size) * float(params.get("pixel_size_nm", 65.0))
    params.update(
        {
            "imaging_model": modality,
            "image_size_pixels": int(image_size),
            "pupil_samples": int(pupil_samples),
            "num_frames": 1,
            "duration_seconds": 1.0 / float(params.get("fps", 24.0)),
            "psf_oversampling_factor": 1,
            "return_ideal_float_frames": True,
            "save_raw_frame_views": False,
            "mask_generation_enabled": False,
            "background_subtraction_method": "reference_frame",
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "particles": [
                {
                    "name": "evidence_particle",
                    "motion": {
                        "hydrodynamic_diameter_nm": 100.0,
                        "initial_position_nm": [0.5 * side_nm, 0.5 * side_nm, 0.0],
                    },
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "components": [
                        {
                            "shape": "sphere",
                            "offset_nm": [0.0, 0.0, 0.0],
                            "diameter_nm": 100.0,
                            "material": "fluorescent_polystyrene" if "fluorescence" in modality else "polystyrene",
                            "refractive_index": None,
                            "signal_multiplier": 1.0,
                            "source_multiplier": 1.0,
                            "material_properties": (
                                {"fluorophore_density": 0.08}
                                if "fluorescence" in modality
                                else None
                            ),
                        }
                    ],
                }
            ],
        }
    )
    return params


def _measurement_model_from_card(card: dict[str, Any]) -> str:
    return "phase" if card.get("measurement_domain") == "phase" else "count"


def render_evidence(modalities: list[str], image_size: int, pupil_samples: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, str], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contrasts: dict[str, np.ndarray] = {}
    noise: dict[str, np.ndarray] = {}
    pixel_sizes: dict[str, float] = {}
    measurement_models: dict[str, str] = {}
    count_images: dict[str, np.ndarray] = {}
    profile_cards: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        params = _base_params(modality, image_size, pupil_samples)
        model = get_imaging_model(params)
        card = profile_card_for_model(params, model, modality_name=modality)
        profile_cards[modality] = card
        try:
            views = generate_single_frame_views(params)
            contrast = np.asarray(views["contrast_frame"], dtype=float)
            signal = np.asarray(views.get("ideal_signal_frame", views.get("raw_signal_frame")), dtype=float)
            reference = views.get("ideal_reference_frame", views.get("raw_reference_frame"))
            reference_arr = None if reference is None else np.asarray(reference, dtype=float)
            detector_difference = views.get("detector_difference_frame")
            noise_var = analysis_contrast_noise_variance(signal, reference_arr, views.get("params_resolved", params))
            contrasts[modality] = contrast
            noise[modality] = np.asarray(noise_var, dtype=float)
            pixel_sizes[modality] = float(views.get("params_resolved", params).get("pixel_size_nm", params.get("pixel_size_nm", 1.0)))
            measurement_models[modality] = _measurement_model_from_card(card)
            if measurement_models[modality] == "count":
                count_images[modality] = signal
            status = "pass" if np.all(np.isfinite(contrast)) and np.all(np.isfinite(noise_var)) else "fail"
            reason = "finite contrast and noise" if status == "pass" else "nonfinite contrast or noise"
        except Exception as exc:
            status = "fail"
            reason = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "modality": modality,
                "measurement_domain": card.get("measurement_domain", ""),
                "signal_units": card.get("signal_units", ""),
                "paper_use_category": card.get("paper_use_category", ""),
                "validation_status": card.get("validation_status", ""),
                "fidelity_label": card.get("fidelity_label", ""),
                "render_smoke_status": status,
                "reason": reason,
                "modality_profile_card_json": json.dumps(_json_safe(card), sort_keys=True),
            }
        )
    return rows, contrasts, noise, pixel_sizes, measurement_models, count_images, profile_cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for evidence CSV/JSON files.")
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES), help="Comma-separated modality list.")
    parser.add_argument("--image-size-pixels", type=int, default=32)
    parser.add_argument("--pupil-samples", type=int, default=16)
    parser.add_argument("--quanta-budget", type=float, default=1.0e4)
    parser.add_argument("--strict-detected-counts", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]

    rows, contrasts, noise, pixel_sizes, measurement_models, count_images, profile_cards = render_evidence(
        modalities,
        args.image_size_pixels,
        args.pupil_samples,
    )
    _write_csv(out_dir / "modality_smoke_evidence.csv", rows)

    successful_modalities = [row["modality"] for row in rows if row["render_smoke_status"] == "pass"]
    if successful_modalities:
        subset_contrasts = {m: contrasts[m] for m in successful_modalities}
        subset_noise = {m: noise[m] for m in successful_modalities}
        subset_pixels = {m: pixel_sizes[m] for m in successful_modalities}
        fusion = compute_modality_fusion_crlb(
            subset_contrasts,
            subset_noise,
            subset_pixels,
            subset_size=min(len(successful_modalities), 5),
            modality_profile_cards={m: profile_cards[m] for m in successful_modalities},
        )
        fusion_meta = fusion.get("fusion_physical_metadata") or fusion_subset_metadata(
            fusion.get("modalities_used", successful_modalities),
            profile_cards,
        )
        (out_dir / "fusion_compatibility_evidence.json").write_text(
            json.dumps(_json_safe(fusion_meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.strict_detected_counts:
            dq = compare_modality_information_content_detected_quanta_normalized(
                {m: contrasts[m] for m in successful_modalities},
                quanta_budget=args.quanta_budget,
                pixel_size_nm={m: pixel_sizes[m] for m in successful_modalities},
                measurement_model_by_modality={m: measurement_models[m] for m in successful_modalities},
                detected_count_image_by_modality={m: count_images[m] for m in successful_modalities if m in count_images},
                require_detected_count_images=True,
            )
            (out_dir / "strict_detected_quanta_evidence.json").write_text(
                json.dumps(_json_safe(dq.get("detected_quanta_contract", {})), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    print(f"wrote internal evidence to {out_dir}")


if __name__ == "__main__":
    main()
