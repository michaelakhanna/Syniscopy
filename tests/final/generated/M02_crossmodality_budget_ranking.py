"""M02 [metamorphic guard] Shared photon-budget cross-candidate ranking.

For one fixed physical scene, every fixed-instrument microscope candidate is
rendered at a shared incident budget B and again at 4B. A shot-noise-limited
CRLB must scale as 1/sqrt(N), so sigma(4B) / sigma(B) should be near 0.5 for
every candidate using the same photon basis. The candidate ordering must also
be invariant under a common budget scale. A violation means the configured
comparison is not apples-to-apples.

Run:
    python M02_crosscandidate_budget_ranking.py
Writes:
    _runs/M02/M02_results.json
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import add_paths, banner, verdict

add_paths()

from calibration_profiles import CALIBRATION_PROFILES, native_params
from camera_noise import analysis_contrast_noise_variance
from fisher import compute_localization_crlb
from noise_contracts import independent_pixel_noise_model
from imaging_models import modality_uses_relative_reference_contrast
from modality_registry import canonical_modality_name
from simulation import generate_single_frame_views

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_runs", "M02")
os.makedirs(OUT, exist_ok=True)

MODALITIES = [
    "interferometric",
    "dark_field",
    "fluorescence_widefield",
    "partially_coherent_bright_field",
    "coherent_bright_field",
]

BUDGET = float(os.environ.get("M02_BUDGET", "20000"))


def _case(modality: str, budget: float) -> dict:
    canon = canonical_modality_name(modality)
    base = dict(CALIBRATION_PROFILES.get(canon, {"modality": canon, "target_sigma_xy_nm": 1.0}))
    base["modality"] = canon

    # One shared particle/optical scene. Fluorescent polystyrene is deliberately
    # used for all modalities so fluorescence has an emitter basis while the
    # label-free modes still see the same physical particle.
    base.update(
        {
            "particle_material": "fluorescent_polystyrene",
            "diameter_nm": 60.0,
            "image_size_pixels": 40,
            "pixel_size_nm": 20.0,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "psf_oversampling_factor": 2,
            "numerical_aperture": 1.0,
            "wavelength_nm": 532.0,
            "background_intensity": float(budget),
            "read_noise_counts": 0.0,
        }
    )

    overrides = dict(base.get("overrides", {}) or {})
    overrides.update(
        {
            "background_intensity": float(budget),
            "gaussian_noise_enabled": False,
            "read_noise_counts": 0.0,
            "random_seed": 123,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "background_subtraction_method": "reference_frame",
        }
    )
    if "dark" in canon:
        overrides["dark_field_illumination_count"] = float(budget)
    if "fluor" in canon:
        # Fluorescence's photon budget is its absorbed-excitation photons, NOT
        # the label-free background_intensity. Scale the modality's own budget so
        # the per-modality shot-noise law is actually exercised. Use zero
        # fluorescence background so this probes the pure shot-noise LIMIT (a
        # sub-photon fixed background would make the emitter edge pedestal-limited
        # and mix 1/N into 1/sqrt(N)).
        overrides["fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"] = float(budget)
        overrides["fluorescence_background"] = 0.0
    base["overrides"] = overrides
    return base


def crlb_sigma(case: dict) -> float:
    params = native_params(case)
    views = generate_single_frame_views(params)
    contrast = np.asarray(views["contrast_frame"], dtype=float)
    signal = np.asarray(views["ideal_signal_frame"], dtype=float)
    reference = views.get("ideal_reference_frame")
    reference = None if reference is None else np.asarray(reference, dtype=float)
    nv = analysis_contrast_noise_variance(
        signal,
        reference,
        params,
        relative_reference=modality_uses_relative_reference_contrast(case["modality"]),
    )
    crlb = compute_localization_crlb(
        contrast,
        independent_pixel_noise_model(nv),
        float(params["pixel_size_nm"]),
    )
    return float(crlb["sigma_xy_nm"])


def _sort_value(value: float) -> tuple[int, float]:
    value = float(value)
    if np.isfinite(value) and value > 0.0:
        return (0, value)
    return (1, float("inf"))


def _ordering(sigmas: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(sigmas.items(), key=lambda item: _sort_value(item[1]))]


banner("M02  shared-budget cross-modality CRLB ranking")

rows = []
sigmas_b: dict[str, float] = {}
sigmas_4b: dict[str, float] = {}
all_ratios_ok = True

for modality in MODALITIES:
    canon = canonical_modality_name(modality)
    try:
        sigma_b = crlb_sigma(_case(canon, BUDGET))
        sigma_4b = crlb_sigma(_case(canon, 4.0 * BUDGET))
        ratio = sigma_4b / sigma_b if sigma_b > 0.0 else float("inf")
        if "dark" in canon:
            # dark_field carries a FIXED stray-light/dark-current pedestal
            # (dark_field_background_count) that does not scale with illumination.
            # A weak scatterer is therefore pedestal/background-shot-limited
            # (sigma ~ 1/N, ratio ~0.25), crossing over to shot-limited
            # (1/sqrt(N), ~0.5) only once the scattered signal exceeds the
            # pedestal. Accept the pedestal-limited-through-shot-limited range;
            # this is correct physics, not a budget-basis violation.
            ratio_ok = 0.20 <= ratio <= 0.60
        else:
            ratio_ok = 0.42 <= ratio <= 0.60
        all_ratios_ok &= ratio_ok
        sigmas_b[canon] = sigma_b
        sigmas_4b[canon] = sigma_4b
        rows.append(
            {
                "modality": canon,
                "sigma_B_nm": sigma_b,
                "sigma_4B_nm": sigma_4b,
                "ratio_sigma_4B_over_B": ratio,
                "ratio_ok": bool(ratio_ok),
            }
        )
        print(
            f"  [{'OK' if ratio_ok else 'FAIL'}] {canon:34s} "
            f"sigma(B)={sigma_b:.6g} nm  sigma(4B)={sigma_4b:.6g} nm  "
            f"ratio={ratio:.3f}"
        )
    except Exception as exc:
        all_ratios_ok = False
        sigmas_b[canon] = float("inf")
        sigmas_4b[canon] = float("inf")
        rows.append(
            {
                "modality": canon,
                "error": f"{type(exc).__name__}: {exc}",
                "ratio_ok": False,
            }
        )
        print(f"  [ERR]  {canon:34s} {type(exc).__name__}: {exc}")

ordering_b = _ordering(sigmas_b)
ordering_4b = _ordering(sigmas_4b)
ranking_ok = ordering_b == ordering_4b
all_ok = bool(all_ratios_ok and ranking_ok)

print(f"\nOrdering at B : {ordering_b}")
print(f"Ordering at 4B: {ordering_4b}")
print(f"Ranking stable: {'YES' if ranking_ok else 'NO'}")

payload = {
    "budget_B": BUDGET,
    "budget_4B": 4.0 * BUDGET,
    "modalities": MODALITIES,
    "per_modality": rows,
    "ordering_B": ordering_b,
    "ordering_4B": ordering_4b,
    "all_ratios_ok": bool(all_ratios_ok),
    "ranking_ok": bool(ranking_ok),
}
out_path = os.path.join(OUT, "M02_results.json")
json.dump(payload, open(out_path, "w"), indent=2)
print(f"\nWROTE {out_path}")
print("Any FAIL means at least one candidate is not on the shared photon-budget basis")
print("or the cross-candidate ranking changes under a common budget scale.")
raise SystemExit(verdict(all_ok, "(shared-budget scaling and ranking stability)"))
