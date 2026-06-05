"""Level-B validation of camera_noise shot-noise model (Poisson statistics).

Two checks, no production-code changes:
  1. Analytic: shot_noise_std_counts(S) == sqrt(S*gain)/gain == sqrt(S/gain).
  2. Statistical: apply_camera_noise_counts on flat fields -> sampled variance
     obeys Poisson, var(counts) = S/gain (var/mean = 1/gain), recovering gain
     from the variance-vs-mean slope.
Run: python throwout/external/noise_validation.py
"""
import sys, numpy as np; sys.path.insert(0, "codebase")
import camera_noise as cn
from copy import deepcopy
from config import PARAMS

# ---- Check 1: analytic shot-noise std ----
print("=== Check 1: analytic shot-noise std vs sqrt(S/gain) ===")
for gain in [1.0, 2.0, 5.0]:
    p = deepcopy(PARAMS); p.update({"shot_noise_enabled": True, "camera_gain_e_per_count": gain})
    S = np.array([100.0, 400.0, 2500.0, 10000.0])
    std = np.asarray(cn.shot_noise_std_counts(S, p), dtype=float)
    pred = np.sqrt(S * gain) / gain
    print(f"  gain={gain}: max_rel_err={np.max(np.abs(std/pred-1)):.2e}")

# ---- Check 2: statistical Poisson behaviour ----
def shot_only(gain, seed):
    p = deepcopy(PARAMS)
    p.update({
        "shot_noise_enabled": True, "gaussian_noise_enabled": False,
        "camera_gain_e_per_count": gain, "emccd_enabled": False,
        "read_noise_counts": 0.0, "read_noise_e": None,
        "dark_current_e_per_pixel_per_s": 0.0,
        "fixed_pattern_gain_std": 0.0, "fixed_pattern_offset_counts": 0.0,
        "hot_pixel_fraction": 0.0, "scan_line_noise_counts": 0.0,
        "fixed_pattern_gain_map": None, "scmos_gain_map": None, "scmos_read_noise_map": None,
        "random_seed": seed,
    })
    return p

print("\n=== Check 2: sampled variance obeys Poisson (var = S/gain) ===")
levels = [100., 400., 1600., 6400.]
for gain in [1.0, 4.0]:
    p = shot_only(gain, seed=7)
    means, varis = [], []
    for S in levels:
        img = np.full((400, 400), S, dtype=float)
        out = np.asarray(cn.apply_camera_noise_counts(img, p), dtype=float)
        means.append(out.mean()); varis.append(out.var())
    means, varis = np.array(means), np.array(varis)
    slope = np.polyfit(means, varis, 1)[0]
    print(f"  gain={gain}: var/mean={np.round(varis/means,4)} (expect {1/gain:.3f}); "
          f"slope={slope:.4f} (expect {1/gain:.3f})")
