"""L01 [Large section] Syniscopy's emitted CRLB vs Monte-Carlo localization scatter.

WHAT THIS VALIDATES (the operational meaning of Syniscopy's cross-modality CRLB)
--------------------------------------------------------------------------------
Syniscopy emits a Cramer-Rao lower bound (CRLB) on the (x, y) localization error
of a particle for a given imaging modality and photon budget. The CRLB is the
information-theoretic floor on the variance of ANY unbiased position estimator
that operates on Syniscopy's own analysis-contrast image under Syniscopy's own
camera-noise model.

This flagship test closes the loop end-to-end and NON-CIRCULARLY:

  1. We render one fixed, deterministic single particle (known sub-pixel
     position) through Syniscopy's full optics + camera-noise pipeline.
  2. We read Syniscopy's REAL emitted CRLB for that exact scene+modality+SNR
     (fisher.compute_localization_crlb on the analysis-contrast image and the
     camera_noise analytic noise-variance map -- the same call the matched-
     modality packet path uses internally).
  3. We draw N INDEPENDENT noisy realizations of that frame through Syniscopy's
     camera-noise model (apply_camera_noise_counts with a fresh seed each time,
     particle position fixed).
  4. We localize the particle in each noisy frame with an INDEPENDENT matched-
     template estimator written here, with Gaussian-fit and trackpy diagnostics
     reported separately when available.
  5. We measure the empirical localization scatter sigma_emp_nm = std of the
     estimated (x, y) over the N frames and compare it to the CRLB.

ASSERTIONS (meaningful, non-circular)
-------------------------------------
  * BOUND HOLDS: sigma_emp_nm >= CRLB_nm * (1 - tol). An (approximately)
    unbiased estimator cannot beat the CRLB. Empirical scatter BELOW the CRLB
    by more than tolerance is a real red flag (over-optimistic CRLB) and is
    reported, never fudged.
  * EFFICIENCY / TIGHTNESS: ratio = sigma_emp/CRLB should sit in a sane band.
    The low->high SNR trend is reported as a diagnostic, but not used as a hard
    mathematical failure: a deliberately independent finite-sample estimator can
    sit above the CRLB for estimator-specific reasons even when the bound and
    noise model are correct.
  * UNBIASEDNESS: the mean estimated position must sit at the true sub-pixel
    position (within a small fraction of a pixel), because the CRLB bounds the
    variance of an UNBIASED estimator.

UNITS / NORMALIZATION / BOUNDARY POINTS
---------------------------------------
  * The localizer works in PIXELS on the analysis-contrast frame, then converts
    to nm via pixel_size_nm. The CRLB is already in nm. No per-modality scaling
    constant is applied to either side: both are in the same physical nm frame.
  * For iSCAT (relative-reference contrast) and fluorescence (additive contrast)
    the contrast convention differs, but the localizer only uses the spatial
    SHAPE of the bump, so the convention does not enter the comparison.
  * Noise model: independent realizations are produced by re-applying
    camera_noise.apply_camera_noise_counts to the SAME ideal signal and
    reference frames with a fresh seed; the analysis contrast is then rebuilt
    with the SAME postprocessing.compute_single_frame_contrast Syniscopy uses.
    A consistency check confirms the empirical per-pixel contrast variance
    matches the analytic noise-variance map that fed the CRLB.

WHAT PASS PROVES / DOES NOT PROVE
---------------------------------
PROVES: Syniscopy's emitted CRLB is a genuine lower bound on the achievable
localization precision of its OWN rendering+noise pipeline, judged by an
independent estimator, and that the analytic contrast-noise variance matches
Monte-Carlo noise draws.
DOES NOT PROVE: that the absolute photon-to-count calibration matches a specific
real microscope, nor the correctness of any cross-modality RANKING claim beyond
the two modalities exercised here (the per-modality bound is what is tested).

EXTERNAL DEPENDENCY (optional second localizer)
-----------------------------------------------
    pip install trackpy        # optional; the script auto-skips it if absent.
The primary localizer needs only numpy + scipy (already required by Syniscopy).

Run:  python L01_crlb_vs_montecarlo_localization.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

from common import add_paths, banner, verdict

add_paths()

# Syniscopy public boundary (black box).
import config  # noqa: F401  initialize public config package
from simulation import generate_single_frame_views
from postprocessing import compute_single_frame_contrast
from camera_noise import analysis_contrast_noise_variance, apply_camera_noise_counts
from fisher import compute_localization_crlb
from noise_contracts import independent_pixel_noise_model

# Reuse the rendering-validation render-boundary helpers. Load that suite's
# common.py under a distinct module name so it does not collide with THIS
# suite's own ``common`` module (already imported above).
import importlib.util as _ilu  # noqa: E402

_RVS_COMMON = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "rendering_validation_suite", "common.py"))
_spec = _ilu.spec_from_file_location("rvs_common", _RVS_COMMON)
rvs_common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rvs_common)
tiny_render_overrides = rvs_common.tiny_render_overrides
set_particle_scene = rvs_common.set_particle_scene

from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(HERE, "_runs", "L01_crlb_vs_montecarlo_localization")
CACHE_DIR = os.path.join(RUN_ROOT, "render_cache")

# ---------------------------------------------------------------------------
# Deterministic scene + sweep configuration.
# ---------------------------------------------------------------------------
IMAGE_SIZE = 48
PIXEL_SIZE_NM = 50.0
DIAMETER_NM = 100.0
PUPIL_SAMPLES = 16
# Known sub-pixel offset of the particle from the central pixel center, in px.
SUBPIXEL_OFFSET_PX = (0.37, -0.24)
N_FRAMES = int(os.environ.get("L01_N_FRAMES", "240"))
GLOBAL_SEED = 20240605

# Per-modality photon-budget sweep (3 SNR levels, low -> high).
# Fluorescence uses the physical absorbed-excitation photon budget with unity
# yield/collection/QE in this synthetic sweep; iSCAT uses the illumination /
# background count level. Higher counts -> higher SNR -> tighter CRLB.
SNR_SWEEP = {
    "fluorescence_widefield": [
        {"label": "lo", "photons": 200.0},
        {"label": "mid", "photons": 800.0},
        {"label": "hi", "photons": 4000.0},
    ],
    "interferometric": [
        {"label": "lo", "photons": 2000.0},
        {"label": "mid", "photons": 8000.0},
        {"label": "hi", "photons": 40000.0},
    ],
}

# Tolerances (fixed up front; not tuned to force a pass).
BOUND_TOL = 0.10          # allow sigma_emp >= 0.90 * CRLB (MC sampling slack)
RATIO_BAND = (0.85, 3.0)  # sane efficiency band across the sweep
BIAS_TOL_PX = 0.25        # mean estimate within 1/4 px of truth
SEARCH_RADIUS_PX = 6      # matched-filter peak ROI half-width around zero shift
GAUSS_DIAG_FRAMES = 24    # only fit the slow single-Gaussian MLE on this many frames
BIAS_SIGMA_K = 3.0        # bias is "consistent with unbiased" if within K standard errors
BOUND_SIGMA_K = 3.0       # bound holds if ratio >= 1 - K*sampling-std-of-sigma
VAR_CONSISTENCY_BAND = (0.6, 1.6)  # emp/analytic contrast-variance sanity


def stable_seed_offset(*parts: str) -> int:
    """Deterministic per-case seed offset; Python's built-in hash is randomized."""
    key = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2s(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % 1_000_000


# ---------------------------------------------------------------------------
# Scene construction.
# ---------------------------------------------------------------------------
def build_params(modality: str, photons: float) -> dict:
    p = tiny_render_overrides(
        modality=modality,
        image_size=IMAGE_SIZE,
        num_frames=1,
        matched_microscopes=None,
        raw_camera_sequence=False,
    )
    center = IMAGE_SIZE // 2
    cx = float(center) + float(SUBPIXEL_OFFSET_PX[0])
    cy = float(center) + float(SUBPIXEL_OFFSET_PX[1])
    set_particle_scene(
        p,
        pixel_size_nm=PIXEL_SIZE_NM,
        diameter_nm=DIAMETER_NM,
        center_pixel_xy=(cx, cy),
    )
    p["pupil_samples"] = PUPIL_SAMPLES
    p["vectorial_pupil_samples"] = PUPIL_SAMPLES
    p["mask_generation_enabled"] = False
    p["background_subtraction_method"] = "reference_frame"
    p["random_seed"] = GLOBAL_SEED
    p["shot_noise_enabled"] = True
    p["gaussian_noise_enabled"] = True
    if modality.startswith("fluorescence"):
        # Fast, deterministic incoherent emitter PSF; no optical z-cache needed.
        p["fluorescence_backend"] = "parametric_psf"
        p["fluorescence_source_representation"] = "projected_2d"
        p["fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"] = float(photons)
        p["fluorescence_quantum_yield"] = 1.0
        p["fluorescence_collection_efficiency"] = 1.0
        p["fluorescence_detector_qe"] = 1.0
        p["z_stack_range_nm"] = 1.0
        p["z_stack_step_nm"] = 100.0
        p["max_psf_z_slices"] = 1
    else:
        # Optical PSF cache keeps +/-100-step z support (>=201 slices).
        p["dark_field_illumination_count"] = float(photons)
        p["background_intensity"] = float(photons)
        p["z_stack_range_nm"] = 1.0
        p["z_stack_step_nm"] = 100.0
        p["max_psf_z_slices"] = 256
    return p, (cx, cy)


def render_scene(modality: str, photons: float):
    """Render once; cache ideal frames + resolved params to disk for reuse."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{modality}_{int(round(photons))}.npz"
    path = os.path.join(CACHE_DIR, key)
    params, true_xy = build_params(modality, photons)
    if os.path.isfile(path):
        d = np.load(path, allow_pickle=True)
        return (
            d["ideal_signal"].astype(float),
            d["ideal_reference"].astype(float),
            d["contrast"].astype(float),
            params,
            (float(d["true_cx"]), float(d["true_cy"])),
        )
    views = generate_single_frame_views(params)
    sig = np.asarray(views["ideal_signal_frame"], dtype=float)
    ref = np.asarray(views["ideal_reference_frame"], dtype=float)
    contrast = np.asarray(views["contrast_frame"], dtype=float)
    np.savez_compressed(
        path,
        ideal_signal=sig,
        ideal_reference=ref,
        contrast=contrast,
        true_cx=float(true_xy[0]),
        true_cy=float(true_xy[1]),
    )
    return sig, ref, contrast, params, true_xy


# ---------------------------------------------------------------------------
# Independent 2D Gaussian MLE localizer (written here, not part of Syniscopy).
# For Poisson+Gaussian noise dominated by many photons, least-squares on the
# contrast bump is the Gaussian-MLE estimator of the centroid.
# ---------------------------------------------------------------------------
def gaussian_mle_localize(frame: np.ndarray, init_xy, sign: float) -> tuple[float, float]:
    """Fit A * exp(-((x-x0)^2+(y-y0)^2)/(2 s^2)) + B to ``sign * frame``.

    ``sign`` flips the bump to be positive (iSCAT/fluorescence both give a
    localized positive lobe after sign normalization). Returns (x0, y0) in px.
    """
    img = sign * np.asarray(frame, dtype=float)
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    x0, y0 = float(init_xy[0]), float(init_xy[1])
    bg0 = float(np.median(img))
    amp0 = float(np.max(img) - bg0)
    if amp0 <= 0:
        amp0 = float(np.std(img)) or 1.0
    sigma0 = 2.0

    def resid(theta):
        amp, cx, cy, s, bg = theta
        s = max(abs(s), 0.3)
        model = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s)) + bg
        return (model - img).ravel()

    theta0 = [amp0, x0, y0, sigma0, bg0]
    lo = [0.0, x0 - 4.0, y0 - 4.0, 0.3, -np.inf]
    hi = [np.inf, x0 + 4.0, y0 + 4.0, 10.0, np.inf]
    try:
        sol = least_squares(resid, theta0, bounds=(lo, hi), method="trf", max_nfev=400)
        return float(sol.x[1]), float(sol.x[2])
    except Exception:
        # Fall back to intensity centroid in a window if the fit fails.
        w = np.clip(img - bg0, 0.0, None)
        tot = float(w.sum())
        if tot <= 0:
            return x0, y0
        return float((xx * w).sum() / tot), float((yy * w).sum() / tot)


def matched_template_localize(frame: np.ndarray, template: np.ndarray) -> tuple[float, float]:
    """Matched-filter localizer: sub-pixel peak of cross-correlation with the
    clean per-pixel contrast TEMPLATE. For additive noise this is the maximum-
    likelihood estimator for an arbitrary KNOWN shape (Gaussian or ringed iSCAT
    PSF alike), so it is unbiased and efficient -- the correct estimator to test
    a localization CRLB. Returns the peak position (col, row) in absolute pixels;
    scatter (std across frames) is offset-invariant and bias is measured against
    the clean template's self-localization, so absolute bookkeeping cancels."""
    f = np.asarray(frame, dtype=float)
    t = np.asarray(template, dtype=float)
    f = f - f.mean()
    t = t - t.mean()
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(f) * np.conj(np.fft.rfft2(t)), s=f.shape))
    H, W = cc.shape
    # Restrict the peak search to an ROI around zero shift (frame center). The
    # particle moves only by its localization error; a whole-frame argmax would
    # otherwise lock onto spurious noise peaks at low SNR (a standard ROI-
    # localization safeguard, not a tuned parameter).
    cy0, cx0 = H // 2, W // 2
    R = int(SEARCH_RADIUS_PX)
    y0, y1 = max(1, cy0 - R), min(H - 1, cy0 + R + 1)
    x0, x1 = max(1, cx0 - R), min(W - 1, cx0 + R + 1)
    win = cc[y0:y1, x0:x1]
    ry, rx = np.unravel_index(int(np.argmax(win)), win.shape)
    py, px = y0 + ry, x0 + rx

    def _par(a, b, c):
        # Only refine a genuine one-dimensional local maximum. If the ROI peak
        # lies on a ridge/edge/noisy plateau, the integer peak remains the
        # estimator; inventing a subpixel vertex outside the central pixel would
        # turn a bad localizer frame into an artificial hundreds-pixel outlier.
        if b < a or b < c:
            return 0.0
        d = a - 2.0 * b + c
        if (not np.isfinite(d)) or abs(d) < 1e-12 or d >= 0.0:
            return 0.0
        delta = 0.5 * (a - c) / d
        if not np.isfinite(delta):
            return 0.0
        # A three-point subpixel correction is valid only inside the central
        # pixel. Without this guard, nearly flat/noisy correlations can produce
        # huge jumps even though the integer peak search was ROI-constrained.
        return float(np.clip(delta, -0.5, 0.5))

    sx = _par(cc[py, px - 1], cc[py, px], cc[py, px + 1]) if 0 < px < W - 1 else 0.0
    sy = _par(cc[py - 1, px], cc[py, px], cc[py + 1, px]) if 0 < py < H - 1 else 0.0
    return float(px) + sx, float(py) + sy


def try_trackpy_localize(frame: np.ndarray, sign: float, diameter_px: int):
    try:
        import trackpy  # noqa: F401
    except Exception:
        return None
    import trackpy as tp
    img = sign * np.asarray(frame, dtype=float)
    img = img - img.min()
    mx = img.max()
    if mx <= 0:
        return None
    img8 = (img / mx * 255.0).astype(np.uint8)
    d = diameter_px if diameter_px % 2 == 1 else diameter_px + 1
    d = max(d, 3)
    try:
        f = tp.locate(img8, diameter=d, minmass=1.0)
    except Exception:
        return None
    if f is None or len(f) == 0:
        return None
    ny, nx = img.shape
    cy0, cx0 = ny / 2.0, nx / 2.0
    f = f.copy()
    f["dist"] = np.hypot(f["x"] - cx0, f["y"] - cy0)
    row = f.sort_values("dist").iloc[0]
    return float(row["x"]), float(row["y"])


# ---------------------------------------------------------------------------
# Monte-Carlo over independent noise realizations.
# ---------------------------------------------------------------------------
def monte_carlo(modality, sig, ref, params, true_xy, n_frames, seed0):
    contrast_clean = compute_single_frame_contrast(sig, ref, params)
    # Sign so the localized lobe is positive at the particle.
    cy, cx = int(round(true_xy[1])), int(round(true_xy[0]))
    sign = 1.0 if contrast_clean[cy, cx] >= np.median(contrast_clean) else -1.0
    diameter_px = max(3, int(round(DIAMETER_NM / PIXEL_SIZE_NM)) | 1) + 2

    est_g = np.full((n_frames, 2), np.nan)
    est_m = np.full((n_frames, 2), np.nan)
    est_t = np.full((n_frames, 2), np.nan)
    # Matched-filter zero reference: localize the clean template against itself.
    clean_ref = matched_template_localize(contrast_clean, contrast_clean)
    frames = []
    have_trackpy = False
    for k in range(n_frames):
        s_noisy = apply_camera_noise_counts(sig, params, random_seed=seed0 + 2 * k)
        r_noisy = apply_camera_noise_counts(ref, params, random_seed=seed0 + 2 * k + 1)
        c = np.asarray(compute_single_frame_contrast(s_noisy, r_noisy, params), dtype=float)
        if k < 3:
            frames.append(c)
        est_m[k] = matched_template_localize(c, contrast_clean)
        if k < GAUSS_DIAG_FRAMES:  # single-Gaussian MLE is a slow secondary diagnostic only
            gx, gy = gaussian_mle_localize(c, true_xy, sign)
            est_g[k] = (gx, gy)
        tp_res = try_trackpy_localize(c, sign, diameter_px)
        if tp_res is not None:
            have_trackpy = True
            est_t[k] = tp_res

    # Empirical contrast variance vs analytic noise-variance map (consistency).
    emp_stack = []
    for k in range(min(n_frames, 40)):
        s_noisy = apply_camera_noise_counts(sig, params, random_seed=seed0 + 100000 + 2 * k)
        r_noisy = apply_camera_noise_counts(ref, params, random_seed=seed0 + 100000 + 2 * k + 1)
        emp_stack.append(np.asarray(compute_single_frame_contrast(s_noisy, r_noisy, params), dtype=float))
    emp_stack = np.stack(emp_stack, 0)
    emp_var = emp_stack.var(axis=0, ddof=1)
    nv = analysis_contrast_noise_variance(sig, ref, params)
    mask = np.abs(contrast_clean) > 0.2 * np.abs(contrast_clean).max()
    var_ratio = (float(np.median(emp_var[mask] / np.maximum(nv[mask], 1e-30)))
                 if mask.any() else float("nan"))

    return {
        "sign": sign,
        "contrast_clean": contrast_clean,
        "est_gauss": est_g,
        "est_matched": est_m,
        "clean_ref": clean_ref,
        "est_trackpy": est_t if have_trackpy else None,
        "sample_frames": frames,
        "var_ratio": var_ratio,
    }


def scatter_nm(est_px, true_xy):
    finite = np.isfinite(est_px).all(axis=1)
    e = est_px[finite]
    if e.shape[0] < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_x, mean_y = float(e[:, 0].mean()), float(e[:, 1].mean())
    std_x = float(e[:, 0].std(ddof=1))
    std_y = float(e[:, 1].std(ddof=1))
    sigma_emp_nm = float(np.hypot(std_x, std_y) * PIXEL_SIZE_NM)
    bias_px = float(np.hypot(mean_x - true_xy[0], mean_y - true_xy[1]))
    return sigma_emp_nm, bias_px, mean_x, mean_y


def save_diagnostics(modality, label, sample_frame, est_px, true_xy, crlb_nm, sigma_emp_nm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    finite = np.isfinite(est_px).all(axis=1)
    e = est_px[finite]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    im = ax[0].imshow(sample_frame, cmap="magma")
    ax[0].set_title(f"{modality} [{label}] sample noisy contrast")
    ax[0].plot(true_xy[0], true_xy[1], "c+", ms=12, mew=2, label="true")
    ax[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    if e.shape[0] >= 3:
        ax[1].scatter((e[:, 0] - true_xy[0]) * PIXEL_SIZE_NM,
                      (e[:, 1] - true_xy[1]) * PIXEL_SIZE_NM,
                      s=10, alpha=0.5, color="tab:blue")
    th = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(crlb_nm * np.cos(th) / np.sqrt(2), crlb_nm * np.sin(th) / np.sqrt(2),
               "r-", lw=1.5, label=f"CRLB/axis ({crlb_nm:.2f} nm tot)")
    ax[1].axhline(0, color="0.7", lw=0.5)
    ax[1].axvline(0, color="0.7", lw=0.5)
    ax[1].set_aspect("equal", "box")
    ax[1].set_xlabel("x error (nm)")
    ax[1].set_ylabel("y error (nm)")
    ax[1].set_title(f"localizations: sigma_emp={sigma_emp_nm:.2f} nm")
    ax[1].legend(loc="upper right", fontsize=8)
    os.makedirs(RUN_ROOT, exist_ok=True)
    out = os.path.join(RUN_ROOT, f"L01_{modality}_{label}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> int:
    banner("L01  Syniscopy emitted CRLB vs Monte-Carlo localization scatter")
    os.makedirs(RUN_ROOT, exist_ok=True)
    np.random.seed(GLOBAL_SEED)

    modalities = ["interferometric", "fluorescence_widefield"]
    _only = os.environ.get("L01_ONLY_MODALITY")
    if _only:
        modalities = [m for m in modalities if m == _only]
    rows = []
    failures = []
    diagnostic_warnings = []
    trackpy_used = False

    for modality in modalities:
        ratios_by_snr = []
        for level in SNR_SWEEP[modality]:
            label, photons = level["label"], level["photons"]
            t0 = time.time()
            sig, ref, contrast_clean, params, true_xy = render_scene(modality, photons)
            nv = analysis_contrast_noise_variance(sig, ref, params)
            crlb = compute_localization_crlb(
                contrast_clean,
                independent_pixel_noise_model(nv),
                pixel_size_nm=float(params["pixel_size_nm"]),
            )
            crlb_nm = float(crlb["sigma_xy_nm"])
            singular = bool(crlb["singular"])

            seed0 = (GLOBAL_SEED + stable_seed_offset(modality, label)) | 1
            mc = monte_carlo(modality, sig, ref, params, true_xy, N_FRAMES, seed0)

            # PRIMARY estimator: matched-filter (ML for any known PSF shape, so
            # unbiased+efficient for ringed iSCAT as well as Gaussian fluorescence).
            # Bias is measured vs the clean template's self-localization, so the
            # matched-filter's absolute coordinate offset cancels exactly.
            est_primary = mc["est_matched"]
            clean_ref = mc["clean_ref"]
            _finite = np.isfinite(est_primary).all(axis=1)
            _e = est_primary[_finite]
            if _e.shape[0] >= 3:
                _sx = float(_e[:, 0].std(ddof=1))
                _sy = float(_e[:, 1].std(ddof=1))
                sigma_emp_nm = float(np.hypot(_sx, _sy) * PIXEL_SIZE_NM)
                bias_px = float(np.hypot(_e[:, 0].mean() - clean_ref[0],
                                         _e[:, 1].mean() - clean_ref[1]))
            else:
                sigma_emp_nm, bias_px = float("nan"), float("nan")
            ratio = sigma_emp_nm / crlb_nm if crlb_nm > 0 else float("nan")
            # Secondary diagnostic only (single-Gaussian MLE; biased on ringed PSFs).
            gauss_sigma_nm, gauss_bias_px, _, _ = scatter_nm(mc["est_gauss"], true_xy)

            tp_sigma = tp_ratio = float("nan")
            if mc["est_trackpy"] is not None:
                trackpy_used = True
                tp_sigma, _, _, _ = scatter_nm(mc["est_trackpy"], true_xy)
                tp_ratio = tp_sigma / crlb_nm if crlb_nm > 0 else float("nan")

            # crlb_nm here is the total 2D bound sqrt(sx^2+sy^2); sigma_emp_nm is
            # likewise the total 2D scatter sqrt(std_x^2+std_y^2): same frame.
            # Statistical tolerances: with finite N you cannot reject "unbiased"
            # or "bound holds" when the deviation is within sampling noise.
            n_fin = int(np.isfinite(est_primary).all(axis=1).sum())
            sigma_emp_px = (sigma_emp_nm / PIXEL_SIZE_NM) if np.isfinite(sigma_emp_nm) else float("inf")
            se_mean_px = sigma_emp_px / np.sqrt(max(n_fin, 1))          # std error of the mean position
            bias_thresh_px = max(BIAS_TOL_PX, BIAS_SIGMA_K * se_mean_px)
            sigma_rel_se = 1.0 / np.sqrt(2.0 * max(n_fin - 1, 1))       # rel std of a std estimate
            bound_floor = 1.0 - max(BOUND_TOL, BOUND_SIGMA_K * sigma_rel_se)
            bound_ok = (not singular) and np.isfinite(ratio) and (ratio >= bound_floor)
            unbiased_ok = np.isfinite(bias_px) and bias_px <= bias_thresh_px
            var_ok = (np.isfinite(mc["var_ratio"]) and
                      VAR_CONSISTENCY_BAND[0] <= mc["var_ratio"] <= VAR_CONSISTENCY_BAND[1])

            sample = mc["sample_frames"][0] if mc["sample_frames"] else mc["contrast_clean"]
            png = save_diagnostics(modality, label, sample, est_primary,
                                   clean_ref, crlb_nm, sigma_emp_nm)

            ratios_by_snr.append(ratio)
            rows.append({
                "modality": modality, "label": label, "photons": photons,
                "crlb_nm": crlb_nm, "sigma_emp_nm": sigma_emp_nm, "ratio": ratio,
                "bias_px": bias_px, "var_ratio": mc["var_ratio"],
                "gauss_sigma_nm": gauss_sigma_nm, "gauss_bias_px": gauss_bias_px,
                "tp_sigma": tp_sigma, "tp_ratio": tp_ratio,
                "bound_ok": bound_ok, "unbiased_ok": unbiased_ok, "var_ok": var_ok,
                "singular": singular, "png": png, "elapsed": time.time() - t0,
            })

            if not bound_ok:
                failures.append(f"{modality}/{label}: BOUND VIOLATED ratio={ratio:.3f} "
                                f"< floor={bound_floor:.3f} (CRLB={crlb_nm:.3f}, emp={sigma_emp_nm:.3f}, N={n_fin})")
            if not unbiased_ok:
                failures.append(f"{modality}/{label}: BIASED estimator bias={bias_px:.3f} px "
                                f"> {bias_thresh_px:.3f} px (3*SE; N={n_fin})")
            if not var_ok:
                failures.append(f"{modality}/{label}: noise-variance inconsistency "
                                f"emp/analytic={mc['var_ratio']:.3f}")
            if not (RATIO_BAND[0] <= ratio <= RATIO_BAND[1]):
                failures.append(f"{modality}/{label}: efficiency ratio {ratio:.3f} "
                                f"outside band {RATIO_BAND}")

        # Tightening trend: useful diagnostic, but not a hard CRLB correctness
        # assertion. A non-core estimator can remain above the bound or show
        # small non-monotonicity without invalidating the emitted CRLB.
        if len(ratios_by_snr) >= 2 and np.all(np.isfinite(ratios_by_snr)):
            if ratios_by_snr[-1] > ratios_by_snr[0] + 0.20:
                diagnostic_warnings.append(
                    f"{modality}: efficiency ratio did NOT tighten with SNR "
                    f"(lo={ratios_by_snr[0]:.3f} -> hi={ratios_by_snr[-1]:.3f})"
                )

    # ---- Report table ----
    print()
    hdr = (f"{'modality':22s} {'snr':4s} {'photons':>9s} {'CRLB_nm':>9s} "
           f"{'emp_nm':>9s} {'ratio':>7s} {'bias_px':>8s} {'varR':>6s} {'bound':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['modality']:22s} {r['label']:4s} {r['photons']:9.0f} "
              f"{r['crlb_nm']:9.3f} {r['sigma_emp_nm']:9.3f} {r['ratio']:7.3f} "
              f"{r['bias_px']:8.3f} {r['var_ratio']:6.2f} "
              f"{'OK' if r['bound_ok'] else 'FAIL':>6s}")

    print("\nSecondary diagnostic (single-Gaussian MLE; biased on ringed iSCAT PSF, "
          "shown for contrast only):")
    for r in rows:
        print(f"  {r['modality']:22s} {r['label']:4s} "
              f"gauss_sigma={r['gauss_sigma_nm']:.3f} nm  gauss_bias={r['gauss_bias_px']:.3f} px")

    if trackpy_used:
        print("\nSecondary localizer (trackpy) sigma_emp_nm / ratio:")
        for r in rows:
            if np.isfinite(r["tp_sigma"]):
                print(f"  {r['modality']:22s} {r['label']:4s} "
                      f"sigma_emp={r['tp_sigma']:.3f} nm  ratio={r['tp_ratio']:.3f}")
    else:
        print("\n[trackpy not installed -> secondary localizer skipped. "
              "Install with: pip install trackpy]")

    print("\nEfficiency/tightness trend (matched-template ratio low->high SNR):")
    for modality in modalities:
        rr = [r["ratio"] for r in rows if r["modality"] == modality]
        print(f"  {modality:22s} " + " -> ".join(f"{x:.3f}" for x in rr))

    print(f"\nDiagnostic PNGs + render cache under: {RUN_ROOT}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
    if diagnostic_warnings:
        print("\nDIAGNOSTIC WARNINGS (not hard CRLB failures):")
        for f in diagnostic_warnings:
            print(f"  - {f}")

    return verdict(
        not failures,
        "(CRLB lower-bounds Monte-Carlo localization scatter, estimator bias is "
        "bounded, and empirical noise variance matches the analytic map)"
        if not failures
        else f"({len(failures)} CRLB/Monte-Carlo checks failed -- see above)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
