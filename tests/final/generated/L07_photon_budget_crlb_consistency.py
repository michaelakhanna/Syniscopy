"""L07 [diagnostic] Is the E04 CRLB blow-up a photon-budget normalization issue
or a CRLB-math bug?  (YOU run this; it writes results I can read.)

For each fixed-instrument candidate it renders the E04 calibration scene and reports:
  * N_eff  : effective detected SIGNAL photons in the particle region
             (sum of |contrast| counts over the bump) -- the shot-noise budget.
  * sigma_syn : Syniscopy's emitted lateral CRLB (nm).
  * s_nm   : measured PSF/bump width (nm), from a Gaussian fit of the contrast bump.
  * sigma_shot = s_nm / sqrt(N_eff) : the classic shot-noise (Thompson) bound.
  * consistency = sigma_syn / sigma_shot : ~O(1) means the CRLB is internally
        consistent with its own photon budget (=> the problem is the BUDGET /
        absolute normalization, NOT the CRLB math). >> or << 1 means a real
        CRLB/noise bug.
  * target_nm, ratio = sigma_syn/target, implied_photon_gap = ratio^2 : how many
        more detected photons the literature precision implies.

If consistency ~ O(1) for all candidates, the fix is purely the absolute
scattered-signal normalization (illumination/cross-section), and the CRLB engine
is correct.

Run on your machine:
    cd throwout/outputs/large_section_suite
    python L07_photon_budget_crlb_consistency.py
Results are printed AND written to:
    _runs/L07/L07_results.json   and   _runs/L07/L07_results.txt
Tell me when it's done; I'll read those files.
"""
from __future__ import annotations
import json, os
import numpy as np
from common import add_paths, banner
add_paths()

from calibration_profiles import native_params, CALIBRATION_PROFILES
from simulation import generate_single_frame_views
from camera_noise import analysis_contrast_noise_variance
from fisher import compute_localization_crlb
from modality_registry import canonical_modality_name
from imaging_models import modality_uses_relative_reference_contrast
from config import OpticalInstrumentSettings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_runs", "L07")
os.makedirs(OUT, exist_ok=True)

MODALITIES = ["dark_field", "coherent_dark_field", "interferometric",
              "coherent_bright_field", "partially_coherent_bright_field",
              "quantitative_phase"]


def gaussian_width_nm(contrast, pixel_size_nm):
    """Second-moment width (sigma) of the |contrast| bump, in nm."""
    a = np.abs(np.asarray(contrast, float))
    a = a - np.median(a)
    a = np.clip(a, 0, None)
    tot = a.sum()
    if tot <= 0:
        return float("nan")
    ny, nx = a.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    cx = (xx * a).sum() / tot
    cy = (yy * a).sum() / tot
    var = ((xx - cx) ** 2 + (yy - cy) ** 2) * a
    sigma_px = np.sqrt(0.5 * var.sum() / tot)   # average of x,y second moments
    return float(sigma_px * pixel_size_nm)


def run_one(modality):
    canon = canonical_modality_name(modality)
    case = dict(CALIBRATION_PROFILES[canon])
    p = native_params(case)
    v = generate_single_frame_views(p)
    C = np.asarray(v["contrast_frame"], float)
    S = np.asarray(v["ideal_signal_frame"], float)
    R = v.get("ideal_reference_frame")
    R = None if R is None else np.asarray(R, float)
    rel = modality_uses_relative_reference_contrast(canon)
    nv = analysis_contrast_noise_variance(S, R, p, relative_reference=rel)
    crlb = compute_localization_crlb(C, nv, float(p["pixel_size_nm"]))
    sigma_syn = float(crlb["sigma_xy_nm"])
    px = float(p["pixel_size_nm"])

    # Effective detected SIGNAL photons in the particle region.
    # For reference-free modes the contrast is in counts (S-R); the bump sum is N.
    # For relative modes contrast is dimensionless; use signal-pedestal counts instead.
    bump = np.abs(C - np.median(C))
    if rel:
        N_eff = float(np.clip(S - np.median(S), 0, None).sum())  # counts above pedestal
    else:
        N_eff = float(bump.sum())                                 # contrast counts
    s_nm = gaussian_width_nm(C, px)
    sigma_shot = s_nm / np.sqrt(N_eff) if (N_eff > 0 and np.isfinite(s_nm)) else float("nan")
    consistency = sigma_syn / sigma_shot if (sigma_shot and np.isfinite(sigma_shot)) else float("nan")
    target = float(case["target_sigma_xy_nm"])
    ratio = sigma_syn / target if target > 0 else float("inf")
    return {
        "modality": canon, "relative_reference": rel,
        "signal_min": float(S.min()), "signal_max": float(S.max()),
        "contrast_peak": float(bump.max()),
        "N_eff_detected": N_eff,
        "psf_sigma_nm": s_nm,
        "sigma_syn_nm": sigma_syn,
        "sigma_shot_bound_nm": sigma_shot,
        "internal_consistency_ratio": consistency,
        "target_nm": target,
        "ratio_to_target": ratio,
        "implied_photon_gap": ratio * ratio,
        "wavelength_nm": float(OpticalInstrumentSettings.from_params(p).probe_wavelength_nm),
        "numerical_aperture": float(p["numerical_aperture"]),
        "illumination_count_or_background": float(
            p.get("dark_field_illumination_count", p.get("background_intensity", float("nan")))),
    }


banner("L07  photon-budget vs CRLB internal-consistency diagnostic")
rows = []
for m in MODALITIES:
    try:
        r = run_one(m); rows.append(r)
        print(f"\n[{r['modality']}] relative={r['relative_reference']}")
        print(f"  signal {r['signal_min']:.4g}->{r['signal_max']:.4g}  contrast_peak={r['contrast_peak']:.4g}")
        print(f"  N_eff(detected)={r['N_eff_detected']:.4g}  psf_sigma={r['psf_sigma_nm']:.4g} nm")
        print(f"  sigma_syn={r['sigma_syn_nm']:.4g} nm | shot-bound s/sqrtN={r['sigma_shot_bound_nm']:.4g} nm "
              f"| consistency={r['internal_consistency_ratio']:.3g} (O(1) => budget issue, not CRLB bug)")
        print(f"  target={r['target_nm']:.4g} nm  ratio={r['ratio_to_target']:.4g}x  "
              f"implied_photon_gap={r['implied_photon_gap']:.4g}x")
    except Exception as ex:
        rows.append({"modality": m, "error": f"{type(ex).__name__}: {ex}"})
        print(f"\n[{m}] ERROR {type(ex).__name__}: {ex}")

json.dump(rows, open(os.path.join(OUT, "L07_results.json"), "w"), indent=2)
with open(os.path.join(OUT, "L07_results.txt"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nWROTE: {os.path.join(OUT, 'L07_results.json')}")
print("Tell Claude when done; it will read _runs/L07/L07_results.json")
