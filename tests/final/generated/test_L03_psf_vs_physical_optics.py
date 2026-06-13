from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from scipy.special import j1

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _radial_profile(intensity: np.ndarray, pixel_size_nm: float):
    cy, cx = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    yy, xx = np.indices(intensity.shape)
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * float(pixel_size_nm)

    order = np.argsort(radii.ravel())
    r_sorted = radii.ravel()[order]
    i_sorted = np.asarray(intensity, dtype=float).ravel()[order]

    rbins = np.arange(0.0, float(r_sorted.max()) + float(pixel_size_nm), float(pixel_size_nm))
    profile = np.full(len(rbins) - 1, np.nan, dtype=float)

    for idx in range(len(profile)):
        mask = (r_sorted >= rbins[idx]) & (r_sorted < rbins[idx] + float(pixel_size_nm))
        if np.any(mask):
            profile[idx] = float(np.mean(i_sorted[mask]))

    radius_center = rbins[:-1] + 0.5 * float(pixel_size_nm)
    return radius_center, profile


def _first_zero_radius(radii_nm: np.ndarray, profile: np.ndarray, *, expected_zero_nm: float) -> float:
    valid = np.isfinite(profile)
    if not np.any(valid):
        return float("nan")

    start = max(1, int(np.searchsorted(radii_nm[valid], 0.2 * expected_zero_nm)))
    r_valid = radii_nm[valid]
    p_valid = profile[valid]

    for i in range(start, len(p_valid) - 1):
        if not np.isfinite(p_valid[i]):
            continue
        if p_valid[i] < p_valid[i - 1] and p_valid[i] <= p_valid[min(i + 1, len(p_valid) - 1)] and p_valid[i] < 0.2:
            return float(r_valid[i])
    return float("nan")


def _fwhm_from_profile(radii_nm: np.ndarray, profile: np.ndarray) -> float:
    valid = np.isfinite(profile)
    r = radii_nm[valid]
    p = profile[valid]
    if r.size < 4:
        return float("nan")

    p0 = p[0]
    if not np.isfinite(p0) or p0 < 0.9:
        return float("nan")

    below = np.where(p <= 0.5)[0]
    if below.size == 0:
        return float("nan")

    i = int(below[0])
    if i == 0:
        return float("nan")

    r1, r0 = r[i], r[i - 1]
    p1, p0 = p[i], p[i - 1]
    if not (np.isfinite(r1) and np.isfinite(r0) and np.isfinite(p1) and np.isfinite(p0)):
        return float("nan")
    if p1 == p0:
        return float("nan")

    half_r = r0 + (0.5 - p0) * (r1 - r0) / (p1 - p0)
    return 2.0 * float(half_r)


def _recenter_to_peak(img: np.ndarray) -> np.ndarray:
    cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
    center = img.shape[0] // 2
    return np.roll(np.roll(img, center - cy, axis=0), center - cx, axis=1)


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(float).ravel()
    bf = b.astype(float).ravel()
    afm = af - float(np.mean(af))
    bfm = bf - float(np.mean(bf))
    denom = float(np.sqrt(np.sum(afm * afm) * np.sum(bfm * bfm)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sum(afm * bfm) / denom)


def _airy_profile(radius_nm: np.ndarray, wavelength_nm: float, numerical_aperture: float) -> np.ndarray:
    v = (2.0 * np.pi * numerical_aperture / float(wavelength_nm)) * np.asarray(radius_nm, dtype=float)
    profile = np.ones_like(v, dtype=float)
    nz = v > 1.0e-12
    profile[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return profile


def _build_reference_airy(N: int, pixel_size_nm: float, wavelength_nm: float, numerical_aperture: float) -> np.ndarray:
    c = N // 2
    yy, xx = np.indices((N, N))
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) * float(pixel_size_nm)
    v = (2.0 * np.pi * numerical_aperture / float(wavelength_nm)) * r
    ref = np.ones_like(v, dtype=float)
    nz = v > 1.0e-12
    ref[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return ref / float(np.max(ref))


def _build_reference_prysm(
    N: int,
    pixel_size_nm: float,
    wavelength_nm: float,
    numerical_aperture: float,
) -> tuple[np.ndarray, str]:
    try:
        from prysm.propagation import Wavefront
        from prysm.coordinates import make_xy_grid, cart_to_polar
        from prysm.geometry import circle

        efl = 100.0
        pupil_diameter = 2.0 * numerical_aperture * efl
        npup = 512
        x, y = make_xy_grid(npup, diameter=pupil_diameter)
        _, r = cart_to_polar(x, y)
        amp = circle(pupil_diameter / 2.0, r).astype(float)
        wf = Wavefront.from_amp_and_phase(amp, None, wavelength_nm / 1000.0, float(np.abs(x[0, 1] - x[0, 0])) )
        img = np.abs(wf.focus_fixed_sampling(efl, pixel_size_nm / 1000.0, N).data) ** 2
        return img / float(np.max(img)), "prysm"
    except Exception:
        return _build_reference_airy(N, pixel_size_nm, wavelength_nm, numerical_aperture), "analytic_airy"


def test_l03_psf_matches_independent_oracle() -> None:
    from config import default_params, OpticalInstrumentSettings
    from optics import compute_complex_psf_stack

    params = default_params()
    params.update(
        {
            "imaging_model": "bright_field",
            "numerical_aperture": 0.6,
            "refractive_index_medium": 1.0,
            "refractive_index_immersion": 1.0,
            "apodization_factor": 0.0,
            "spherical_aberration_strength": 0.0,
            "random_aberration_strength": 0.0,
            "wavelength_nm": 520.0,
            "probe_wavelength_nm": 520.0,
            "pupil_samples": 256,
            "psf_oversampling_factor": 1,
            "pixel_size_nm": 30.0,
            "image_size_pixels": 129,
            "optical_field_backend": "scalar_paraxial",
            "max_psf_z_slices": 1,
        }
    )
    instrument = OpticalInstrumentSettings.from_params(params)
    wavelength_nm = float(instrument.probe_wavelength_nm)
    numerical_aperture = float(instrument.numerical_aperture)
    pixel_nm = float(params["pixel_size_nm"]) / float(params.get("psf_oversampling_factor", 1) or 1)

    interp = compute_complex_psf_stack(
        params,
        particle_diameter_nm=20.0,
        particle_refractive_index=1.59 + 0.0j,
        z_values_nm=np.array([0.0]),
    )
    field = np.asarray(interp(0.0))

    if field.ndim == 3 and field.shape[0] == 3:
        syn = np.sum(np.abs(field) ** 2, axis=0)
    else:
        syn = np.abs(field) ** 2

    assert syn.ndim == 2
    assert syn.shape[0] == syn.shape[1]
    assert syn.size > 0

    syn = syn / float(np.max(syn))
    assert np.isfinite(syn).all()

    rc_s, prof_s = _radial_profile(syn, pixel_nm)
    expected_zero = 0.61 * wavelength_nm / numerical_aperture

    r0_s = _first_zero_radius(rc_s, prof_s, expected_zero_nm=expected_zero)
    assert np.isfinite(r0_s)

    ref, ref_name = _build_reference_prysm(syn.shape[0], pixel_nm, wavelength_nm, numerical_aperture)
    rc_r, prof_r = _radial_profile(ref, pixel_nm)
    r0_r = _first_zero_radius(rc_r, prof_r, expected_zero_nm=expected_zero)

    assert np.isfinite(np.max(ref))
    assert np.isfinite(r0_r)

    ref_norm = _recenter_to_peak(ref / float(np.max(ref)))
    syn_norm = _recenter_to_peak(syn)

    mask = np.isfinite(prof_s) & np.isfinite(prof_r) & (rc_s < 2.5 * expected_zero)
    assert np.count_nonzero(mask) >= 6

    radial_corr = float(np.corrcoef(prof_s[mask], prof_r[mask])[0, 1])
    assert radial_corr >= 0.95

    corr2d = _pearson_flat(syn_norm, ref_norm)
    assert np.isfinite(corr2d)
    assert corr2d >= 0.95

    fwhm_s = _fwhm_from_profile(rc_s, prof_s)
    fwhm_r = _fwhm_from_profile(rc_r, prof_r)
    assert np.isfinite(fwhm_s)
    assert np.isfinite(fwhm_r)

    rel_zero = abs(r0_s - r0_r) / max(r0_r, 1e-12)
    rel_zero_airy = abs(r0_s - expected_zero) / expected_zero
    rel_fwhm = abs(fwhm_s - fwhm_r) / max(fwhm_r, 1e-12)

    assert rel_zero < 0.06
    assert rel_zero_airy < 0.06
    assert rel_fwhm < 0.08

    assert str(ref_name) in {"prysm", "analytic_airy"}
