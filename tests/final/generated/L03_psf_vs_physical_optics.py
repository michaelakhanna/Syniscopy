"""L03 [Large section] Pupil->focal PSF vs an independent physical-optics propagator.

WHAT THIS VALIDATES
-------------------
Syniscopy builds the coherent point-spread function with a pupil-propagation
integral in ``optics.compute_complex_psf_stack`` (scalar_paraxial backend): it
forms a circular aperture in k-space, applies an (optional) apodization, the Mie
S2 angular factor, aberration phase, and inverse-FFTs to the image plane. The
in-focus intensity |field(z=0)|^2 is the PSF.

We compare that PSF -- for a CLEAN circular pupil (apodization off, no
aberration), a small (Rayleigh) particle so S2 is essentially flat across the
NA cone, and a modest NA -- against an INDEPENDENT propagator:

  * Primary oracle: prysm (Brandon Dube), a separate physical-optics library
    with its own Fourier-optics propagation code. We build a clear circular
    pupil and propagate to the focal plane on the SAME physical pixel grid
    (focus_fixed_sampling) for the SAME NA / wavelength / pixel size.

  * Fallback oracle (if prysm cannot be imported): the analytic Airy pattern
    I(r) = [2 J1(v)/v]^2, v = (2 pi NA / lambda) r, first zero r0 = 0.61 lambda/NA.
    This is the closed-form diffraction-limited PSF and is likewise INDEPENDENT
    of Syniscopy's code.

Both oracles are different code lineages from Syniscopy AND from DeepTrack, so
agreement validates the pupil->focal-image propagation itself, not a shared
one-line formula.

COMPARISONS / PASS CRITERIA
---------------------------
  1. Radial intensity profile correlation (peak-normalized) > 0.97.
  2. First-zero (Airy) radius vs 0.61*lambda/NA within a few percent.
  3. FWHM of the central lobe agrees with the reference within a few percent.
  4. 2D normalized cross-correlation (peak of the centered, normalized images)
     > 0.97.
Normalization is by peak intensity, justified because both are ideal optical
intensities with no absolute-flux meaning; shape and zero-location are the
physics under test. No tolerance is loosened to force a pass.

UNITS / CONVENTIONS
-------------------
Lengths in nm internally. Syniscopy PSF pixel pitch = pixel_size_nm /
psf_oversampling_factor. prysm works in mm (pupil) and microns (focal dx) and
um wavelength; we convert explicitly. NA enters the focal-plane scale through
F/# = 1/(2 NA) (object-space, n=1) so the Airy zero is 0.61*lambda/NA.

WHAT PASS PROVES / DOES NOT
---------------------------
PROVES: for a diffraction-limited clear aperture, Syniscopy's pupil->image
propagation reproduces the focal field shape (main lobe width, first dark ring,
2D structure) of an independent propagator. DOES NOT: validate high-NA vectorial
effects (polarization broadening), aberrated pupils, or absolute photometric
scaling. High-NA/vectorial deviation, if the vectorial backend were selected,
would be physical and is documented rather than masked; here we deliberately
select the scalar backend and a modest NA.

INSTALL (whichever is available):
    pip install prysm     # primary, used if importable
    # analytic Airy fallback needs only numpy+scipy (already required)

Run:  python L03_psf_vs_physical_optics.py
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.special import j1

from common import add_paths, banner, verdict

add_paths()

from config import default_params, OpticalInstrumentSettings
from optics import compute_complex_psf_stack

banner("L03  Pupil->focal PSF: Syniscopy vs independent physical-optics propagator")

# ---------------------------------------------------------------------------
# 1. Build the Syniscopy scalar PSF for a clean aperture, in focus.
# ---------------------------------------------------------------------------
p = default_params()
overrides = {
    "optical_field_backend": "scalar_paraxial",  # Airy is the scalar limit
    "numerical_aperture": 0.6,                    # modest NA: scalar is a good model
    "refractive_index_medium": 1.0,              # object space, F/# = 1/(2 NA)
    "apodization_factor": 0.0,                    # clean (uniform) pupil
    "spherical_aberration_strength": 0.0,
    "random_aberration_strength": 0.0,
    "pupil_samples": 256,
    "psf_oversampling_factor": 1,
    "wavelength_nm": 520.0,
    "probe_wavelength_nm": 520.0,
    "pixel_size_nm": 30.0,
}
for k, v in overrides.items():
    if k in p:
        p[k] = v
# A few schemas gate coverslip aberration; force it off if present.
for key in ("coverslip_aberration_enabled", "coverslip_mismatch_enabled"):
    if key in p:
        p[key] = False

lam = float(OpticalInstrumentSettings.from_params(p).probe_wavelength_nm)
NA = float(p["numerical_aperture"])
px_nm = float(p["pixel_size_nm"]) / float(p.get("psf_oversampling_factor", 1) or 1)

interp = compute_complex_psf_stack(
    p,
    particle_diameter_nm=20.0,                 # Rayleigh: S2 ~ flat over the cone
    particle_refractive_index=1.59 + 0.0j,
    z_values_nm=np.array([0.0]),
)
field = np.asarray(interp(0.0))
meta = getattr(interp, "metadata", {}) or {}
backend = meta.get("backend", meta.get("scalar_compatibility_reduction", "unknown"))
if field.ndim == 3 and field.shape[0] == 3:
    syn = np.sum(np.abs(field) ** 2, axis=0)
else:
    syn = np.abs(field) ** 2
syn = syn / syn.max()
N = syn.shape[0]
print(f"Syniscopy PSF: backend={backend} shape={syn.shape} pixel={px_nm:.2f} nm "
      f"lambda={lam:.1f} nm NA={NA:.2f}")


# ---------------------------------------------------------------------------
# Radial-profile helper (shared by both sides).
# ---------------------------------------------------------------------------
def radial_profile(img: np.ndarray, px: float):
    cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * px
    order = np.argsort(r.ravel())
    rs = r.ravel()[order]
    vs = img.ravel()[order]
    bins = np.arange(0.0, rs.max(), px)
    prof = np.full(len(bins) - 1, np.nan)
    for k in range(len(bins) - 1):
        sel = (rs >= bins[k]) & (rs < bins[k] + px)
        if np.any(sel):
            prof[k] = vs[sel].mean()
    rc = bins[:-1] + px / 2.0
    return rc, prof


def first_zero_radius(rc, prof):
    valid = np.isfinite(prof)
    for i in range(2, len(prof) - 1):
        if valid[i] and prof[i] < prof[i - 1] and prof[i] <= prof[i + 1] and prof[i] < 0.2:
            return float(rc[i])
    return float("nan")


def fwhm_from_profile(rc, prof):
    valid = np.isfinite(prof)
    rcv, pv = rc[valid], prof[valid]
    if pv.size < 3 or pv[0] < 0.5:
        return float("nan")
    # first radius where the profile drops to 0.5; double it for full width
    below = np.where(pv <= 0.5)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float("nan")
    r1, r0 = rcv[i], rcv[i - 1]
    p1, p0 = pv[i], pv[i - 1]
    half_r = r0 + (0.5 - p0) * (r1 - r0) / (p1 - p0)
    return 2.0 * float(half_r)


# Syniscopy profile / metrics
rc_s, prof_s = radial_profile(syn, px_nm)
r0_s = first_zero_radius(rc_s, prof_s)
fwhm_s = fwhm_from_profile(rc_s, prof_s)
r0_airy = 0.61 * lam / NA


# ---------------------------------------------------------------------------
# 2. Build the independent reference PSF on the SAME grid.
# ---------------------------------------------------------------------------
def reference_prysm(N, px_nm, lam_nm, NA):
    """Diffraction-limited PSF from prysm via clear circular pupil + focus.

    Returns (psf_img, "prysm") or raises on any failure so we can fall back.
    """
    import prysm.propagation as prop
    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import circle

    # Model an f/# system in object space. Choose a pupil diameter D (mm) and
    # focal length efl (mm) with NA = D/(2 efl) -> F/# = 1/(2 NA). The absolute
    # scale is arbitrary; only NA and wavelength set the focal-plane physics.
    efl = 100.0                       # mm, arbitrary
    Dpupil = 2.0 * NA * efl           # mm, gives the requested NA
    npup = 512
    x, y = make_xy_grid(npup, diameter=Dpupil)  # mm
    dx_pup = x[0, 1] - x[0, 0]
    r, _ = cart_to_polar(x, y)
    amp = circle(Dpupil / 2.0, r).astype(float)  # clear circular aperture
    wf = prop.Wavefront.from_amp_and_phase(amp, None, lam_nm / 1000.0, dx_pup)  # um wavelength
    # focal-plane sample spacing in microns; match Syniscopy's nm grid.
    dx_focus_um = px_nm / 1000.0
    focused = wf.focus_fixed_sampling(efl, dx_focus_um, N)
    img = np.abs(focused.data) ** 2
    img = img / img.max()
    return img, "prysm"


def reference_airy(N, px_nm, lam_nm, NA):
    """Closed-form Airy intensity on an N x N grid, peak-normalized."""
    c = N // 2
    yy, xx = np.indices((N, N))
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) * px_nm
    v = (2.0 * np.pi * NA / lam_nm) * r
    img = np.ones_like(v)
    nz = v > 1e-9
    img[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return img / img.max(), "analytic_airy"


ref_img = None
ref_name = None
try:
    ref_img, ref_name = reference_prysm(N, px_nm, lam, NA)
    print(f"Reference: prysm physical-optics propagation (clear circular pupil).")
except Exception as exc:
    print(f"[info] prysm unavailable or propagation failed ({type(exc).__name__}: {exc}).")
    print("       Falling back to the analytic Airy reference (independent of Syniscopy).")
    print("       To use prysm:  pip install prysm")
    ref_img, ref_name = reference_airy(N, px_nm, lam, NA)

# Center the reference on its own peak so the 2D correlation is shift-free.
rc_r, prof_r = radial_profile(ref_img, px_nm)
r0_r = first_zero_radius(rc_r, prof_r)
fwhm_r = fwhm_from_profile(rc_r, prof_r)


# ---------------------------------------------------------------------------
# 3. Metrics.
# ---------------------------------------------------------------------------
def recenter_to_peak(img):
    cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
    c = img.shape[0] // 2
    return np.roll(np.roll(img, c - cy, axis=0), c - cx, axis=1)


syn_c = recenter_to_peak(syn)
ref_c = recenter_to_peak(ref_img)

# (a) radial profile correlation out to ~2.5 first-zero radii
rmax = 2.5 * r0_airy
mask = np.isfinite(prof_s) & np.isfinite(prof_r) & (rc_s < rmax)
if np.count_nonzero(mask) < 4:
    prof_corr = float("nan")
else:
    prof_corr = float(np.corrcoef(prof_s[mask], prof_r[mask])[0, 1])

# (b) 2D normalized cross-correlation (Pearson over all pixels)
a = syn_c.ravel() - syn_c.mean()
b = ref_c.ravel() - ref_c.mean()
denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
corr2d = float(np.sum(a * b) / denom) if denom > 0 else float("nan")

# (c) first-zero radius: compare Syniscopy and reference, both vs Airy formula
rel_r0_vs_ref = (abs(r0_s - r0_r) / r0_r) if np.isfinite(r0_s) and np.isfinite(r0_r) else np.inf
rel_r0_vs_airy = (abs(r0_s - r0_airy) / r0_airy) if np.isfinite(r0_s) else np.inf

# (d) FWHM
rel_fwhm = (abs(fwhm_s - fwhm_r) / fwhm_r) if np.isfinite(fwhm_s) and np.isfinite(fwhm_r) else np.inf


# ---------------------------------------------------------------------------
# 4. Report.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print(f"{'metric':<34}{'Syniscopy':>14}{'reference':>14}{'rel/score':>14}")
print("-" * 78)
print(f"{'first-zero radius (nm)':<34}{r0_s:>14.1f}{r0_r:>14.1f}{rel_r0_vs_ref:>14.3f}")
print(f"{'  vs analytic 0.61 lam/NA (nm)':<34}{r0_s:>14.1f}{r0_airy:>14.1f}{rel_r0_vs_airy:>14.3f}")
print(f"{'FWHM main lobe (nm)':<34}{fwhm_s:>14.1f}{fwhm_r:>14.1f}{rel_fwhm:>14.3f}")
print(f"{'radial-profile correlation':<34}{'':>14}{'':>14}{prof_corr:>14.4f}")
print(f"{'2D normalized cross-correlation':<34}{'':>14}{'':>14}{corr2d:>14.4f}")
print("-" * 78)
print(f"reference used: {ref_name}")

ok_profile = np.isfinite(prof_corr) and prof_corr > 0.97
ok_corr2d = np.isfinite(corr2d) and corr2d > 0.97
ok_r0 = np.isfinite(rel_r0_vs_ref) and rel_r0_vs_ref < 0.06 and rel_r0_vs_airy < 0.06
ok_fwhm = np.isfinite(rel_fwhm) and rel_fwhm < 0.06
all_ok = ok_profile and ok_corr2d and ok_r0 and ok_fwhm

print("\nChecks:")
print(f"  radial-profile corr > 0.97        : {ok_profile}  ({prof_corr:.4f})")
print(f"  2D cross-corr      > 0.97         : {ok_corr2d}  ({corr2d:.4f})")
print(f"  first-zero within 6% (ref & Airy) : {ok_r0}  (ref {rel_r0_vs_ref:.3f}, Airy {rel_r0_vs_airy:.3f})")
print(f"  FWHM within 6%                    : {ok_fwhm}  ({rel_fwhm:.3f})")
print("\nNote: a modest NA and the scalar backend are chosen on purpose so the")
print("scalar Airy limit is the correct physics. High-NA vectorial PSFs legitimately")
print("broaden (polarization), which is a real effect, not a bug -- not exercised here.")

raise SystemExit(verdict(
    all_ok,
    f"(PSF shape matches the {ref_name} oracle: profile & 2D corr > 0.97, "
    "first-zero & FWHM within 6%)",
))
