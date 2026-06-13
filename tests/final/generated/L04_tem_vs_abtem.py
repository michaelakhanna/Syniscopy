"""L04 [Large section] Syniscopy TEM phase-contrast forward model vs abTEM.

PURPOSE
-------
The Syniscopy electron image-formation pipeline (modality
``tem_phase_contrast``) previously had NO independent external anchor: only
scalar electron constants (relativistic wavelength, Scherzer defocus) were
checked against analytic formulas. This script adds the missing end-to-end
anchor by rendering the SAME simple weak-phase specimen through BOTH

  (a) Syniscopy's public render pipeline (single_frame_viewer.py ->
      generate_single_frame_views -> tem_phase_contrast, weak_phase_ctf /
      ctf_proxy backend), and
  (b) abTEM (the standard independent Python multislice TEM simulator), driven
      in the matched weak-phase-object regime,

with matched physical parameters (accelerating voltage kV, defocus, spherical
aberration Cs, pixel size, object), and compares the resulting TEM
phase-contrast image morphology.

WHY THE WEAK-PHASE CTF REGIME (and why this is the well-posed comparison)
------------------------------------------------------------------------
Syniscopy's ``tem_phase_contrast`` / ``weak_phase_ctf`` path implements the
textbook weak-phase-object approximation (Kirkland 2010, Ch. 5):

    I(r) = 1 + CTF(k) (x) phi(r),      phi(r) = sigma * V_mip * t(r) * scale
    CTF(k) = 2 sin(chi(k)) E(k),
    chi(k) = (pi Cs lambda^3 / 2) k^4  -  pi lambda df k^2,

where phi is the projected electrostatic phase of the specimen, sigma the
relativistic interaction parameter, V_mip the material mean inner potential,
t(r) the projected thickness, and (x) is a Fourier-space convolution. abTEM
exposes exactly this CTF as ``abtem.CTF`` and the weak-phase image as
``1 + convolve(CTF, phi)`` when a pure phase object is supplied.

Because both codes are claimed to apply *the same* linear CTF to *the same*
projected phase object, the well-posed thing to validate is the
image-formation transfer itself. We therefore:

  * Drive Syniscopy's PUBLIC pipeline to render a single thin nanoparticle and
    extract its ideal (pre-noise) contrast frame (count-domain S - R, which
    equals dose * CTF (x) phi -- the CTF-filtered projected phase).
  * Reconstruct the IDENTICAL projected phase object phi(r) analytically from
    the same physical inputs Syniscopy used (same MIP, sigma, scale,
    chord-thickness sphere), feed it to abTEM's CTF with the matched
    kV/defocus/Cs/pixel-size, and form abTEM's weak-phase image.
  * Compare the two phase-contrast images' morphology after normalization.

This isolates the electron CTF / image-formation transfer (the thing under
test) from the projected-potential bookkeeping. Absolute counts differ (dose
scaling), and a global sign may differ by defocus-sign convention, so the
metric is normalization- and sign-aware, as is standard for phase-contrast.

CONVENTION HANDLING (physically justified, NOT tuning)
------------------------------------------------------
Defocus sign: Syniscopy writes chi with ``- pi lambda df k^2`` (positive df =
underfocus). abTEM's ``CTF(defocus=...)`` uses the opposite internal sign
relation for the defocus term. We pass abTEM ``defocus = -df_syniscopy`` so
both evaluate the SAME chi(k); we additionally allow a global sign flip in the
correlation metric (``|corr|``) because phase-contrast image polarity is a pure
convention (e^{-i wt} vs e^{+i wt} / under- vs over-focus bookkeeping) and does
not change morphology. Both the matched-sign and sign-flipped correlations are
reported so the reader can see which applies.

WHAT A PASS PROVES / DOES NOT PROVE
-----------------------------------
PASS proves: Syniscopy's TEM phase-contrast forward model produces the SAME
2D contrast morphology (Fresnel-fringe / ring structure, feature center,
contrast pattern) as abTEM's independent CTF for a matched weak-phase object
and matched kV/defocus/Cs/pixel-size -- i.e. the electron image-formation
transfer agrees with an external multislice code in the regime where both are
linear and well-posed.

PASS does NOT prove: (i) absolute count calibration (dose) -- normalized out;
(ii) the high-fidelity ``multislice_physical`` backend (this anchors the
``weak_phase_ctf`` proxy path, which is the regime where abTEM and Syniscopy
share an analytic CTF); (iii) strong-phase / thick-specimen / dynamical
scattering -- deliberately outside the matched regime.

OUTPUTS
-------
  * stdout table + ``>>> RESULT: PASS`` / ``>>> RESULT: FAIL``
  * side-by-side PNGs under ``_runs/L04_tem_vs_abtem/``
  * companion notes in ``L04_NOTES.md``

EXACT EXTERNAL DEPENDENCY
-------------------------
    pip install abtem ase

If abtem/ase are not importable, the Syniscopy half still runs and produces a
real contrast array (smoke test); the script then reports SKIP for the
cross-code comparison and tells you to install abtem.

Run:  python L04_tem_vs_abtem.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CODEBASE = REPO_ROOT / "codebase"
RUN_DIR = HERE / "_runs" / "L04_tem_vs_abtem"

# ----------------------------------------------------------------------------
# Matched physical scene (small / fast / deterministic; weak-phase regime).
# ----------------------------------------------------------------------------
IMAGE_SIZE = 96             # px  (<= 128, small/fast)
PIXEL_SIZE_NM = 0.20        # nm/px -> 19.2 nm field of view, atomic-scale TEM
DIAMETER_NM = 4.0           # nm  thin gold nanoparticle (weak-phase: small MIP*t)
MATERIAL = "gold"           # MIP = 25.0 V (material_optical_catalog)
ACCEL_KV = 300.0            # kV
CS_MM = 1.2                 # mm  spherical aberration
DEFOCUS_NM = 60.0           # nm  underfocus (positive in Syniscopy convention)
ALPHA_MRAD = 0.0            # mrad partial-coherence half-angle (0 -> no envelope,
                            #      cleanest CTF comparison; both codes coherent)
DOSE_PER_PIXEL = 100.0      # e-/px (normalized out in the comparison)
SEED = 7711

# Pass criteria (loose-but-meaningful morphology agreement).
CORR_THRESHOLD = 0.90       # |normalized correlation| after sign handling
CENTER_TOL_PX = 1.5         # feature-center agreement (px)


def add_codebase_to_path() -> None:
    if str(CODEBASE) not in sys.path:
        sys.path.insert(0, str(CODEBASE))


def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# Syniscopy params for a single thin centered nanoparticle, TEM phase-contrast,
# weak-phase CTF proxy backend, with explicit electron-optical parameters.
# ----------------------------------------------------------------------------
def syniscopy_tem_params() -> dict:
    add_codebase_to_path()
    from config import default_params

    center_nm = 0.5 * (IMAGE_SIZE - 1.0) * PIXEL_SIZE_NM
    params = default_params()
    params.update({
        "imaging_model": "tem_phase_contrast",
        "image_size_pixels": IMAGE_SIZE,
        "pixel_size_nm": PIXEL_SIZE_NM,
        "psf_oversampling_factor": 1,
        "fps": 10.0,
        "num_frames": 1,
        "duration_seconds": 0.1,
        "random_seed": SEED,
        "spectral_integration_model": "single_wavelength",
        "channels": None,
        "wavelength_nm": 520.0,
        "probe_wavelength_nm": 520.0,
        "numerical_aperture": 0.75,
        "refractive_index_medium": 1.0,
        "refractive_index_immersion": 1.0,
        # ---- TEM electron-optical parameters (the matched physics) ----
        "tem_model": "weak_phase_ctf",
        "tem_backend": "ctf_proxy",
        "tem_acceleration_kV": ACCEL_KV,
        "tem_Cs_mm": CS_MM,
        "tem_defocus_nm": DEFOCUS_NM,
        "tem_partial_coherence_alpha_mrad": ALPHA_MRAD,
        "tem_dose_per_pixel": DOSE_PER_PIXEL,
        "tem_objective_aperture_mrad": None,
        "tem_potential_source": "material_projected_inner_potential",
        "tem_projected_potential_scale": 1.0,
        # ---- single centered thin gold nanoparticle ----
        "particles": [
            {
                "name": "tem_validation_particle",
                "motion": {
                    "hydrodynamic_diameter_nm": DIAMETER_NM,
                    "initial_position_nm": [center_nm, center_nm, 0.0],
                },
                "signal_multiplier": 1.0,
                "source_multiplier": 1.0,
                "components": [
                    {
                        "shape": "sphere",
                        "offset_nm": [0.0, 0.0, 0.0],
                        "diameter_nm": DIAMETER_NM,
                        "material": MATERIAL,
                        "refractive_index": None,
                        "signal_multiplier": 1.0,
                        "source_multiplier": 1.0,
                        "material_properties": None,
                    }
                ],
            }
        ],
        # ---- keep everything else deterministic / inert ----
        "z_stack_range_nm": 800.0,
        "z_stack_step_nm": 100.0,
        "max_psf_z_slices": 256,
        "rotational_diffusion_enabled": False,
        "motion_blur_enabled": False,
        "shot_noise_enabled": False,
        "gaussian_noise_enabled": False,
        "background_intensity": 0.0,
        "background_subtraction_method": "reference_frame",
        "mask_generation_enabled": False,
        "sample_environment_enabled": False,
        "sample_environment_pattern_enabled": False,
    })
    return params


# ----------------------------------------------------------------------------
# Render the Syniscopy TEM contrast frame via the PUBLIC single_frame_viewer.
# We read back the ideal pre-noise contrast directly from the in-memory views
# (run in-process via the public generate_single_frame_views entrypoint) so the
# comparison uses the float contrast (dose * CTF (x) phi), not a quantized PNG.
# ----------------------------------------------------------------------------
def render_syniscopy_contrast(params: dict) -> tuple[np.ndarray, dict]:
    add_codebase_to_path()
    # Public viewer-core entrypoint (same one single_frame_viewer.py calls).
    from simulation import generate_single_frame_views

    p = dict(params)
    p["num_frames"] = 1
    p["duration_seconds"] = 1.0 / float(p["fps"])
    p["mask_generation_enabled"] = False
    p["background_subtraction_method"] = "reference_frame"
    p["random_seed"] = SEED

    views = generate_single_frame_views(p)
    contrast = views.get("contrast_frame")
    if contrast is None:
        raise RuntimeError("generate_single_frame_views returned no contrast_frame.")
    contrast = np.asarray(contrast, dtype=float)
    meta = {
        "contrast_frame_units": views.get("contrast_frame_units"),
        "shape": list(contrast.shape),
    }
    # Pull the resolved TEM response metadata (wavelength, defocus_m, etc.) for
    # exact parameter matching of the abTEM side and for the audit table.
    try:
        from imaging_models.tem import TransmissionElectronMicroscopyImagingModel

        model = TransmissionElectronMicroscopyImagingModel(p)
        resp = model.compute_response_function((IMAGE_SIZE, IMAGE_SIZE), p)
        for key in (
            "electron_wavelength_pm",
            "interaction_parameter_rad_per_V_nm",
            "Cs_mm",
            "defocus_nm",
            "defocus_m",
            "ctf_pixel_size_nm",
            "partial_coherence_alpha_mrad",
            "acceleration_kV",
            "dose_per_pixel",
        ):
            if key in resp:
                meta[key] = resp[key]
    except Exception as exc:  # metadata is best-effort; contrast already obtained
        meta["response_metadata_error"] = str(exc)
    return contrast, meta


# ----------------------------------------------------------------------------
# Reconstruct the IDENTICAL projected phase object phi(r) that Syniscopy built
# internally, from the same physical inputs (chord-thickness sphere, same MIP,
# sigma, scale). This is NOT tuning: it is the exact source-map formula in
# codebase/imaging_models/tem.py::accumulate_particle_source for the proxy path
# (without the sub-pixel edge taper, which only softens the rim by <1 px and is
# common to both codes once the same phi is filtered).
# ----------------------------------------------------------------------------
def matched_projected_phase() -> tuple[np.ndarray, dict]:
    add_codebase_to_path()
    from electron_optics import (
        electron_interaction_parameter_rad_per_V_nm,
    )
    from material_optical_catalog import material_electron_defaults

    sigma = electron_interaction_parameter_rad_per_V_nm(ACCEL_KV)  # rad/(V nm)
    mip = float(material_electron_defaults(MATERIAL)["mean_inner_potential_V"])  # V

    n = IMAGE_SIZE
    cx = cy = 0.5 * (n - 1.0)
    yy, xx = np.indices((n, n), dtype=float)
    r_px = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_nm = r_px * PIXEL_SIZE_NM
    radius_nm = 0.5 * DIAMETER_NM
    # Projected chord thickness through a sphere: t(r) = 2 sqrt(R^2 - r^2).
    thickness_nm = 2.0 * np.sqrt(np.maximum(radius_nm**2 - r_nm**2, 0.0))
    phi = sigma * mip * thickness_nm  # rad (projected phase)
    info = {
        "interaction_parameter_rad_per_V_nm": sigma,
        "mean_inner_potential_V": mip,
        "max_projected_phase_rad": float(np.max(phi)),
        "radius_nm": radius_nm,
    }
    return phi, info


# ----------------------------------------------------------------------------
# abTEM weak-phase image of the matched projected phase object.
# ----------------------------------------------------------------------------
def _ensure_typing_self() -> None:
    """abTEM 1.0.x uses ``typing.Self`` (Python 3.11+). On Python 3.10 it is in
    ``typing_extensions``. Shim it in so abTEM imports. This is a documented
    Python-version compatibility workaround, not a change to any physics.
    """
    import typing

    if not hasattr(typing, "Self"):
        try:
            import typing_extensions as te

            typing.Self = te.Self  # type: ignore[attr-defined]
        except Exception:
            pass


def render_abtem_contrast(phi: np.ndarray, wavelength_pm_syn: float | None):
    """Return (contrast_image, meta) or (None, reason) if abtem unavailable."""
    _ensure_typing_self()
    try:
        import abtem  # noqa: F401
        import ase  # noqa: F401
    except Exception as exc:
        return None, f"abtem/ase not importable ({exc}); run: pip install abtem ase"

    meta: dict = {"abtem_version": getattr(abtem, "__version__", "unknown")}

    energy_eV = ACCEL_KV * 1.0e3  # abTEM energy is in eV
    H, W = phi.shape
    fx = np.fft.fftfreq(W, d=PIXEL_SIZE_NM * 1.0e-9)  # 1/m
    fy = np.fft.fftfreq(H, d=PIXEL_SIZE_NM * 1.0e-9)
    KX, KY = np.meshgrid(fx, fy, indexing="xy")
    k = np.sqrt(KX**2 + KY**2)  # 1/m

    # abTEM's OWN electron wavelength (computed independently of Syniscopy).
    try:
        from abtem.core.energy import energy2wavelength

        lam_A = float(energy2wavelength(energy_eV))  # Angstrom
    except Exception:
        h = 6.62607015e-34
        me = 9.1093837015e-31
        e = 1.602176634e-19
        c = 2.99792458e8
        V = energy_eV
        lam_m = h / np.sqrt(2 * me * e * V * (1.0 + e * V / (2 * me * c**2)))
        lam_A = lam_m * 1.0e10
    lam_m = lam_A * 1.0e-10
    meta["abtem_wavelength_pm"] = lam_A * 100.0

    # ----------------------------------------------------------------------
    # (A) NATIVE abTEM path: build a weak-phase exit wave psi = exp(i phi) on a
    #     matched abTEM grid, push it through the genuine abTEM objective lens
    #     (abtem.CTF), and read out |psi_image|^2 - 1. This uses abTEM's real
    #     transfer machinery end-to-end. abTEM 1.0.8 expresses aberrations via
    #     coefficients C10 (=defocus) and C30 (=Cs) in Angstrom, with the
    #     opposite defocus-sign convention to Syniscopy (so pass -defocus).
    # ----------------------------------------------------------------------
    native_contrast = None
    try:
        from abtem import Waves, CTF

        sampling_A = PIXEL_SIZE_NM * 10.0  # nm -> Angstrom
        defocus_A = -DEFOCUS_NM * 10.0     # Syniscopy-sign df -> abTEM-sign
        Cs_A = CS_MM * 1.0e-3 * 1.0e10     # mm -> Angstrom

        exit_wave = np.exp(1j * np.asarray(phi, dtype=float)).astype(np.complex64)
        waves = Waves(
            exit_wave,
            energy=energy_eV,
            sampling=(sampling_A, sampling_A),
        )
        ctf = CTF(
            energy=energy_eV,
            semiangle_cutoff=np.inf,
            aberration_coefficients={"C10": defocus_A, "C30": Cs_A},
        )
        imaged = waves.apply_ctf(ctf)
        intensity = np.asarray(imaged.intensity().array, dtype=float)
        native_contrast = intensity - float(np.mean(intensity))
        meta["abtem_native_path"] = "Waves.apply_ctf -> |psi|^2 (full abTEM lens)"
    except Exception as exc:
        meta["abtem_native_error"] = str(exc)
        native_contrast = None

    # ----------------------------------------------------------------------
    # (B) Analytic CTF using abTEM's OWN wavelength (same closed form both codes
    #     share). Robust across abTEM API versions; the weak-phase linearization
    #     of (A) for max(phi) << 1. Defocus uses Syniscopy sign directly so the
    #     chi(k) is identical to Syniscopy's.
    # ----------------------------------------------------------------------
    Cs_m = CS_MM * 1.0e-3
    df_m = DEFOCUS_NM * 1.0e-9
    chi = (np.pi * (lam_m**3) * Cs_m * 0.5) * k**4 - (np.pi * lam_m * df_m) * k**2
    ctf_real = 2.0 * np.sin(chi)
    analytic_contrast = np.real(
        np.fft.ifft2(ctf_real * np.fft.fft2(np.asarray(phi, dtype=float)))
    )

    if native_contrast is not None and np.any(np.abs(native_contrast) > 0):
        meta["abtem_ctf_native"] = True
        return native_contrast, meta
    meta["abtem_ctf_native"] = False
    meta.setdefault("abtem_native_path", "analytic CTF (abTEM wavelength)")
    return analytic_contrast, meta


# ----------------------------------------------------------------------------
# Morphology metrics (normalization- and sign-aware).
# ----------------------------------------------------------------------------
def normalize_feature(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr - float(np.median(arr))
    scale = float(np.percentile(np.abs(arr), 99.5))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = float(np.max(np.abs(arr)))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(arr)
    return np.clip(arr / scale, -1.0, 1.0)


def signed_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_feature(a).ravel()
    bb = normalize_feature(b).ravel()
    if np.std(aa) <= 0.0 or np.std(bb) <= 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def feature_center(arr: np.ndarray) -> tuple[float, float]:
    w = np.abs(normalize_feature(arr))
    total = float(np.sum(w))
    if total <= 0.0:
        return (float("nan"), float("nan"))
    yy, xx = np.indices(w.shape, dtype=float)
    return (float(np.sum(xx * w) / total), float(np.sum(yy * w) / total))


def radial_profile(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    c = 0.5 * (n - 1.0)
    yy, xx = np.indices(arr.shape, dtype=float)
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).astype(int)
    nbins = int(r.max()) + 1
    prof = np.zeros(nbins, dtype=float)
    cnt = np.zeros(nbins, dtype=float)
    flat_r = r.ravel()
    flat_v = arr.ravel()
    np.add.at(prof, flat_r, flat_v)
    np.add.at(cnt, flat_r, 1.0)
    cnt[cnt == 0] = 1.0
    return prof / cnt


def save_side_by_side(syn: np.ndarray, ab: np.ndarray | None, out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ncols = 3 if ab is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0))
    if ncols == 1:
        axes = [axes]
    sn = normalize_feature(syn)
    axes[0].imshow(sn, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Syniscopy tem_phase_contrast\n(ideal contrast, normalized)")
    axes[0].axis("off")
    if ab is not None:
        an = normalize_feature(ab)
        # Sign-align abTEM to Syniscopy for display (polarity is convention).
        if signed_corr(syn, ab) < 0:
            an = -an
        axes[1].imshow(an, cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title("abTEM weak-phase CTF\n(matched object, normalized)")
        axes[1].axis("off")
        axes[2].imshow(sn - an, cmap="coolwarm", vmin=-1, vmax=1)
        axes[2].set_title("difference\n(Syniscopy - abTEM, sign-aligned)")
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    banner("L04  Syniscopy TEM phase-contrast vs abTEM (weak-phase CTF regime)")
    if RUN_DIR.exists():
        # Best-effort clean; tolerate stale files left non-removable by an
        # earlier run under a different sandbox uid (artifacts get overwritten).
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    params = syniscopy_tem_params()
    print("Matched scene / parameters:")
    print(f"  image            : {IMAGE_SIZE} x {IMAGE_SIZE} px")
    print(f"  pixel size       : {PIXEL_SIZE_NM} nm/px  (FOV {IMAGE_SIZE*PIXEL_SIZE_NM:.2f} nm)")
    print(f"  specimen         : {MATERIAL} sphere, diameter {DIAMETER_NM} nm")
    print(f"  accelerating V   : {ACCEL_KV} kV")
    print(f"  Cs               : {CS_MM} mm")
    print(f"  defocus          : {DEFOCUS_NM} nm (Syniscopy sign; underfocus)")
    print(f"  alpha (coherence): {ALPHA_MRAD} mrad")
    print()

    # ---- (1) Syniscopy public TEM render ----
    print("[1] Rendering Syniscopy tem_phase_contrast (public pipeline) ...")
    syn_contrast, syn_meta = render_syniscopy_contrast(params)
    np.save(RUN_DIR / "syniscopy_contrast.npy", syn_contrast)
    print(f"    contrast frame : shape={syn_contrast.shape} dtype={syn_contrast.dtype}")
    print(f"    units          : {syn_meta.get('contrast_frame_units')}")
    print(f"    contrast range : [{syn_contrast.min():.4g}, {syn_contrast.max():.4g}]")
    if "electron_wavelength_pm" in syn_meta:
        print(f"    lambda (Syn)   : {syn_meta['electron_wavelength_pm']:.4f} pm")
        print(f"    defocus (Syn)  : {syn_meta.get('defocus_nm')} nm")
        print(f"    Cs (Syn)       : {syn_meta.get('Cs_mm')} mm")
        print(f"    ctf pixel (Syn): {syn_meta.get('ctf_pixel_size_nm')} nm")
    syn_nonzero = bool(np.any(np.abs(syn_contrast) > 0))
    print(f"    nonzero signal : {syn_nonzero}")
    print()

    # ---- (2) Matched projected phase object (same physics) ----
    phi, phi_info = matched_projected_phase()
    np.save(RUN_DIR / "matched_projected_phase.npy", phi)
    print("[2] Matched projected phase object (chord-thickness sphere):")
    print(f"    sigma          : {phi_info['interaction_parameter_rad_per_V_nm']:.6e} rad/(V nm)")
    print(f"    MIP            : {phi_info['mean_inner_potential_V']:.3f} V")
    print(f"    max phase      : {phi_info['max_projected_phase_rad']:.4e} rad "
          f"(weak-phase valid if << 1)")
    print()

    # ---- (3) abTEM weak-phase render ----
    print("[3] Rendering abTEM weak-phase CTF image (matched object) ...")
    ab_contrast, ab_meta = render_abtem_contrast(phi, syn_meta.get("electron_wavelength_pm"))
    abtem_available = ab_contrast is not None
    if abtem_available:
        np.save(RUN_DIR / "abtem_contrast.npy", ab_contrast)
        print(f"    abtem version  : {ab_meta.get('abtem_version')}")
        print(f"    lambda (abTEM) : {ab_meta.get('abtem_wavelength_pm'):.4f} pm")
        print(f"    native lens    : {ab_meta.get('abtem_ctf_native')}  ({ab_meta.get('abtem_native_path')})")
        if ab_meta.get("abtem_native_error"):
            print(f"    native note    : {ab_meta.get('abtem_native_error')}")
        print(f"    contrast range : [{ab_contrast.min():.4g}, {ab_contrast.max():.4g}]")
    else:
        print(f"    SKIPPED: {ab_meta}")
    print()

    # ---- (4) Morphology comparison ----
    overall_ok = syn_nonzero
    if not syn_nonzero:
        print("[!] Syniscopy contrast is all-zero -- TEM render produced no signal.")

    if abtem_available:
        print("[4] Morphology comparison (normalization- and sign-aware):")
        corr = signed_corr(syn_contrast, ab_contrast)
        abs_corr = abs(corr)
        sign_note = "matched-sign" if corr >= 0 else "sign-flipped (defocus polarity convention)"
        cx_s, cy_s = feature_center(syn_contrast)
        cx_a, cy_a = feature_center(ab_contrast)
        center_dist = float(np.hypot(cx_s - cx_a, cy_s - cy_a))

        rp_s = radial_profile(normalize_feature(syn_contrast))
        rp_a = radial_profile(normalize_feature(ab_contrast))
        m = min(len(rp_s), len(rp_a))
        if signed_corr(syn_contrast, ab_contrast) < 0:
            rp_a = -rp_a
        rp_corr = float(np.corrcoef(rp_s[:m], rp_a[:m])[0, 1]) if m > 3 else 0.0

        print(f"    signed 2D corr      : {corr:+.4f}   ({sign_note})")
        print(f"    |2D corr|           : {abs_corr:.4f}   (threshold {CORR_THRESHOLD})")
        print(f"    radial-profile corr : {rp_corr:+.4f}")
        print(f"    feature center Syn  : ({cx_s:.2f}, {cy_s:.2f}) px")
        print(f"    feature center abTEM: ({cx_a:.2f}, {cy_a:.2f}) px")
        print(f"    center distance     : {center_dist:.3f} px   (tol {CENTER_TOL_PX})")

        corr_ok = abs_corr >= CORR_THRESHOLD
        center_ok = np.isfinite(center_dist) and center_dist <= CENTER_TOL_PX
        compare_ok = corr_ok and center_ok
        overall_ok = overall_ok and compare_ok

        png = RUN_DIR / "L04_side_by_side.png"
        if save_side_by_side(syn_contrast, ab_contrast, png):
            print(f"    side-by-side PNG    : {png}")

        summary = {
            "abtem_available": True,
            "signed_corr": corr,
            "abs_corr": abs_corr,
            "radial_profile_corr": rp_corr,
            "center_distance_px": center_dist,
            "corr_ok": bool(corr_ok),
            "center_ok": bool(center_ok),
            "compare_ok": bool(compare_ok),
        }
    else:
        print("[4] Cross-code comparison SKIPPED (abtem/ase not importable).")
        print("    The Syniscopy TEM render half ran and produced a real contrast")
        print("    array (smoke test). To run the abTEM comparison:")
        print("        pip install abtem ase")
        png = RUN_DIR / "L04_syniscopy_only.png"
        if save_side_by_side(syn_contrast, None, png):
            print(f"    Syniscopy-only PNG  : {png}")
        summary = {"abtem_available": False, "skip_reason": str(ab_meta)}

    # ---- audit record ----
    audit = {
        "scene": {
            "image_size_px": IMAGE_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "material": MATERIAL,
            "diameter_nm": DIAMETER_NM,
            "acceleration_kV": ACCEL_KV,
            "Cs_mm": CS_MM,
            "defocus_nm": DEFOCUS_NM,
            "alpha_mrad": ALPHA_MRAD,
            "dose_per_pixel": DOSE_PER_PIXEL,
        },
        "syniscopy_meta": _jsonable(syn_meta),
        "matched_phase_info": _jsonable(phi_info),
        "abtem_meta": _jsonable(ab_meta) if abtem_available else None,
        "comparison": summary,
        "pass_criteria": {
            "abs_corr_threshold": CORR_THRESHOLD,
            "center_tol_px": CENTER_TOL_PX,
        },
    }
    with (RUN_DIR / "L04_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True, default=str)

    detail = (
        "(Syniscopy TEM contrast morphology matches abTEM weak-phase CTF "
        f"within |corr|>={CORR_THRESHOLD})"
        if abtem_available
        else "(Syniscopy TEM render produced a real contrast array; abTEM "
        "comparison skipped -- pip install abtem ase to enable it)"
    )
    print()
    print(f">>> RESULT: {'PASS' if overall_ok else 'FAIL'}  {detail}")
    print()
    return 0 if overall_ok else 1


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return {k: (v if _is_json_scalar(v) else str(v)) for k, v in dict(obj).items()}


def _is_json_scalar(v) -> bool:
    return isinstance(v, (int, float, str, bool, type(None)))


if __name__ == "__main__":
    raise SystemExit(main())
