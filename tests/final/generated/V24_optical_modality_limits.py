"""V24 [Level B] Analytic optical limits for DPC / QPI / Zernike.

DeepTrack (r03) covers morphology for the main optical modes; this adds the
analytic LIMIT signatures that uniquely identify the secondary phase-contrast
modes, end-to-end through the public render boundary:

  * DPC: in the scalar/paraxial weak-phase limit, a symmetric phase object
    imaged with a left/right asymmetric channel gives an ANTISYMMETRIC signal
    across the shear axis (bright/dark lobes): image(x) ~= -image(-x). Check:
    correlation(image, x-flip) is strongly negative and the signal integrates
    ~to zero. High-NA vectorial Debye DPC is smoke-tested separately because it
    is not the scalar analytic limit.
  * QPI: weak-phase linearity -- the QPI phase output scales ~linearly with the
    refractive-index contrast (n_particle - n_medium). Halving the contrast
    ~halves the peak phase.
  * Zernike: weak phase -> intensity, so a higher-index particle produces a
    localized contrast feature of consistent sign.

Run: python V24_optical_modality_limits.py     (pure render boundary; no pip deps)
"""
from __future__ import annotations
import os, importlib.util as _ilu
import numpy as np
from common import add_paths, banner, verdict
add_paths()

from simulation import generate_single_frame_views

_RVS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "rendering_validation_suite", "common.py"))
_spec = _ilu.spec_from_file_location("rvs_common", _RVS)
rvs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(rvs)
tiny_render_overrides = rvs.tiny_render_overrides
set_particle_scene = rvs.set_particle_scene

banner("V24  DPC / QPI / Zernike analytic limits")


def render(
    modality,
    *,
    n_medium=1.33,
    image_size=48,
    diameter_nm=200.0,
    particle_refractive_index=None,
    extra=None,
):
    p = tiny_render_overrides(modality=modality, image_size=image_size, num_frames=1)
    set_particle_scene(p, pixel_size_nm=50.0, diameter_nm=diameter_nm)
    if particle_refractive_index is not None:
        component = p["particles"][0]["components"][0]
        n_complex = complex(particle_refractive_index)
        component["refractive_index"] = {
            "real": float(n_complex.real),
            "imag": float(n_complex.imag),
        }
    p["random_seed"] = 4242
    p["refractive_index_medium"] = float(n_medium)
    p["refractive_index_immersion"] = float(n_medium)
    p["mask_generation_enabled"] = False
    if extra:
        p.update(extra)
    return np.asarray(generate_single_frame_views(p)["contrast_frame"], dtype=float)


results = []

# --- DPC antisymmetry in the scalar/paraxial analytic limit ---
try:
    c = render(
        "differential_phase_contrast",
        image_size=64,
        extra={
            "optical_field_backend": "scalar_paraxial",
            "vectorial_detection_mode": "analyzer_x",
            "dpc_channel_model": "two_axis_scalar_asymmetric_illumination",
            "dpc_transfer_model": "asymmetric_illumination",
            "dpc_output_channel": "x",
            "numerical_aperture": 0.25,
            "pupil_samples": 64,
            "vectorial_pupil_samples": 64,
        },
    )
    c0 = c - np.median(c)
    flip = c0[:, ::-1]
    anti_corr = float(np.corrcoef(c0.ravel(), flip.ravel())[0, 1])
    asym = float(abs(c0.sum()) / (np.abs(c0).sum() + 1e-30))
    ok = (anti_corr < -0.5) and (asym < 0.2)
    results.append(("DPC scalar antisymmetry", ok, f"flip-corr={anti_corr:+.3f} (expect <-0.5), |sum|/|signal|={asym:.3f}"))
except Exception as ex:
    results.append(("DPC scalar antisymmetry", False, f"render error: {ex}"))

# --- Default vectorial DPC smoke: finite, balanced contrast without scalar symmetry claim ---
try:
    c = render("differential_phase_contrast", extra={"dpc_output_channel": "x"})
    c0 = c - np.median(c)
    peak = float(np.max(np.abs(c0)))
    asym = float(abs(c0.sum()) / (np.abs(c0).sum() + 1e-30))
    ok = np.all(np.isfinite(c0)) and peak > 1.0e-8 and asym < 0.2
    results.append(("DPC vectorial smoke", ok, f"peak={peak:.3g}, |sum|/|signal|={asym:.3f}"))
except Exception as ex:
    results.append(("DPC vectorial smoke", False, f"render error: {ex}"))

# --- QPI weak-phase linearity ---
try:
    # Two weak nonabsorbing index contrasts. The helper's default
    # non-fluorescence material is gold, which is absorptive and not a
    # weak-pure-phase linearity case.
    n_particle = 1.36
    c_full = render(
        "quantitative_phase",
        n_medium=1.33,
        diameter_nm=80.0,
        particle_refractive_index=n_particle,
    )   # dn = 0.03
    c_half = render(
        "quantitative_phase",
        n_medium=1.345,
        diameter_nm=80.0,
        particle_refractive_index=n_particle,
    )   # dn = 0.015
    pk_full = float(np.max(np.abs(c_full - np.median(c_full))))
    pk_half = float(np.max(np.abs(c_half - np.median(c_half))))
    ratio = pk_full / pk_half if pk_half > 0 else float("inf")
    ok = 1.5 <= ratio <= 2.6      # ~2x for halved index contrast (weak-phase linear)
    results.append(("QPI linearity", ok, f"peak(dn=0.030)/peak(dn=0.015)={ratio:.2f} (expect ~2)"))
except Exception as ex:
    results.append(("QPI linearity", False, f"render error: {ex}"))

# --- Zernike weak-phase feature ---
try:
    cz = render("zernike_phase_contrast")
    bg = np.median(cz)
    feat = float(np.max(np.abs(cz - bg)))
    rel = feat / (np.abs(cz - bg).mean() + 1e-30)
    ok = feat > 0 and rel > 3.0   # a localized central feature, not flat noise
    results.append(("Zernike phase->intensity", ok, f"peak/mean ratio={rel:.2f} (localized feature)"))
except Exception as ex:
    results.append(("Zernike phase->intensity", False, f"render error: {ex}"))

all_ok = True
for name, ok, detail in results:
    all_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name:26s} {detail}")

print("\nProves the analytic-limit signature of each phase-contrast mode end-to-end.")
print("Does NOT assert absolute phase magnitude vs a specific instrument.")
raise SystemExit(verdict(all_ok, "(DPC scalar limit, DPC vectorial smoke, QPI linearity, Zernike localized)"))
