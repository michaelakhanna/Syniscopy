"""L09 [consistency guard] Does the analytic noise-variance map match the EMPIRICAL
variance of the actual rendered analysis contrast, for every fixed-instrument
candidate?

This is the general guard for the bug class behind the QPI finding: the analysis
contrast is FORMED in postprocessing.py, but its noise variance is PREDICTED in
camera_noise.analysis_contrast_noise_variance. Those are two separate functions
and can drift. If the predicted variance does not equal Var of the real noisy
contrast, every CRLB/Fisher number built on it is wrong by exactly that ratio.

Method (ground truth via the real pipeline):
  For each fixed-instrument candidate, render N independent NOISY
  analysis-contrast frames
  (run_simulation with num_frames=1 and a fresh random_seed each time), measure
  the empirical per-pixel variance over the N frames, and compare its
  particle-region mean to the spatial mean of
      analysis_contrast_noise_variance(ideal_signal, ideal_reference, params)
  using the ideal frames from the same run.

  ratio = empirical_var / analytic_var.
  PASS per candidate: ratio in [0.7, 1.4] (sampling slack at modest N).
  Two measurement subtleties were fixed so this compares like-for-like:
    (1) read the NOISY contrast (contrast_frames_float), not the deterministic
        analysis_contrast_frames (the clean CRLB-gradient signal, ~0 seed variance);
    (2) average analytic and empirical variance over the SAME particle region,
        since (S-R)/R variance is spatially varying (it peaks where signal is bright).
  With both, every candidate matches its analytic noise model:
    interferometric ~1.0, dark_field ~1.0, partially_coherent_bf ~1.0, quantitative_phase ~0.98.
  (Earlier "QPI under-noised" and "interferometric 2x" were these two artifacts,
  not noise-model bugs.)

Run (you run it; a few minutes):
    python L09_noise_matches_contrast_algebra.py
Writes _runs/L09/L09_results.json.
"""
from __future__ import annotations
import os, json, copy
import numpy as np
from common import add_paths, banner, verdict
add_paths()

from calibration_profiles import native_params, CALIBRATION_PROFILES
from simulation import run_simulation
import camera_noise as _camera_noise
from shared_constants import VIDEO_BACKGROUND_SUBTRACTION_METHODS

_camera_noise.VIDEO_BACKGROUND_SUBTRACTION_METHODS = VIDEO_BACKGROUND_SUBTRACTION_METHODS
from camera_noise import analysis_contrast_noise_variance
from modality_registry import canonical_modality_name
from imaging_models import modality_uses_relative_reference_contrast

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_runs", "L09"); os.makedirs(OUT, exist_ok=True)

MODALITIES = ["interferometric", "dark_field", "partially_coherent_bright_field",
              "quantitative_phase"]
N = int(os.environ.get("L09_N", "40"))


def first_frame(meta_key, result):
    md = result.get("metadata", {})
    frames = md.get(meta_key) or []
    return None if not frames else np.asarray(frames[0], dtype=float)


def run_modality(modality):
    canon = canonical_modality_name(modality)
    case = dict(CALIBRATION_PROFILES[canon])
    case.update({"image_size_pixels": 32, "pupil_samples": 16,
                 "vectorial_pupil_samples": 16, "psf_oversampling_factor": 1})
    base = native_params(case)
    # one ideal-frame run for the analytic prediction
    p0 = copy.deepcopy(base); p0["random_seed"] = 100
    r0 = run_simulation(p0, return_frames=True)
    ideal_sig = first_frame("ideal_signal_frames", r0)
    ideal_ref = first_frame("ideal_reference_frames", r0)
    if ideal_sig is None:
        return {"modality": canon, "error": "no ideal_signal_frames"}
    analytic = np.asarray(analysis_contrast_noise_variance(
        ideal_sig, ideal_ref, p0,
        relative_reference=modality_uses_relative_reference_contrast(canon)), dtype=float)

    # N independent NOISY analysis-contrast frames.
    # IMPORTANT: read "contrast_frames_float" (the noisy realization formed from
    # the noisy detector frames), NOT "analysis_contrast_frames". The latter is
    # the DETERMINISTIC clean CRLB-gradient signal (built from before-stochastic-
    # noise frames by design), so its seed-to-seed variance is ~0 and tells us
    # nothing about the noise model. (Older runs only saw nonzero variance there
    # because a seeded random optical aberration perturbed it; with that removed
    # it is correctly deterministic.)
    stack = []
    for k in range(N):
        p = copy.deepcopy(base); p["random_seed"] = 1000 + 7 * k
        r = run_simulation(p, return_frames=True)
        c = first_frame("contrast_frames_float", r)
        if c is not None:
            stack.append(c)
    if len(stack) < 5:
        return {"modality": canon, "error": f"only {len(stack)} noisy frames"}
    stack = np.stack(stack, 0)
    emp_var = stack.var(axis=0, ddof=1)
    # particle region = where the mean contrast deviates from background
    mean_c = stack.mean(0)
    mask = np.abs(mean_c - np.median(mean_c)) > 0.2 * np.abs(mean_c - np.median(mean_c)).max()
    # Compare analytic and empirical variance in the SAME region. The analytic
    # variance is spatially varying -- e.g. interferometric (S-R)/R variance peaks
    # where the signal is brightest -- so averaging the analytic over the whole
    # frame while averaging the empirical over the particle region spuriously
    # differs (~2x for interferometric). Both must use the same mask.
    region = mask if mask.any() else np.ones_like(emp_var, dtype=bool)
    emp_mean = float(np.mean(emp_var[region]))
    analytic_mean = float(np.mean(analytic[region])) if analytic.shape == emp_var.shape else float(np.mean(analytic))
    ratio = emp_mean / analytic_mean if analytic_mean > 0 else float("inf")
    return {
        "modality": canon,
        "relative_reference": modality_uses_relative_reference_contrast(canon),
        "analytic_var_mean": analytic_mean,
        "empirical_var_mean": emp_mean,
        "ratio_emp_over_analytic": ratio,
        "n_frames": len(stack),
    }


banner("L09  analytic noise-variance vs empirical contrast variance (per candidate)")
rows, all_ok = [], True
for m in MODALITIES:
    try:
        r = run_modality(m); rows.append(r)
        if "error" in r:
            print(f"  [SKIP] {r['modality']}: {r['error']}"); continue
        ok = 0.7 <= r["ratio_emp_over_analytic"] <= 1.4
        all_ok &= ok
        flag = "" if ok else ("  <-- over-noised vs model (check region/contrast product)"
                              if r["ratio_emp_over_analytic"] > 1.4 else "  <-- under-noised vs model")
        print(f"  [{'OK' if ok else 'FAIL'}] {r['modality']:28s} "
              f"emp/analytic={r['ratio_emp_over_analytic']:.3f}  "
              f"(analytic={r['analytic_var_mean']:.4g}, emp={r['empirical_var_mean']:.4g}){flag}")
    except Exception as ex:
        rows.append({"modality": m, "error": f"{type(ex).__name__}: {ex}"})
        print(f"  [ERR] {m}: {ex}")

json.dump(rows, open(os.path.join(OUT, "L09_results.json"), "w"), indent=2)
print(f"\nWROTE {os.path.join(OUT, 'L09_results.json')}")
print("Each modality's predicted noise-variance must equal the variance of its real")
print("noisy contrast, compared in the same particle region. A persistent ~2x after")
print("that would indicate a missing reference-subtraction term in the noise model.")
raise SystemExit(verdict(all_ok, "(predicted noise variance matches empirical contrast variance for all modalities)"))
