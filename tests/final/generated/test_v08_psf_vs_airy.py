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

    rbins = np.arange(0.0, r_sorted.max() + pixel_size_nm, float(pixel_size_nm))
    profile = np.full(len(rbins) - 1, np.nan, dtype=float)

    for k in range(len(profile)):
        mask = (r_sorted >= rbins[k]) & (r_sorted < rbins[k] + pixel_size_nm)
        if np.any(mask):
            profile[k] = np.mean(i_sorted[mask])

    radius_center = rbins[:-1] + float(pixel_size_nm) / 2.0
    return radius_center, profile


def _first_zero_radius(rc: np.ndarray, profile: np.ndarray, expected_zero: float) -> float:
    valid = np.isfinite(profile)
    start = max(1, int(np.searchsorted(rc, 0.5 * expected_zero)))

    for idx in range(start, len(profile) - 1):
        if not valid[idx]:
            continue
        if profile[idx] < profile[idx - 1] and profile[idx] <= profile[idx + 1] and profile[idx] < 0.2:
            return float(rc[idx])

    return float("nan")


def _airy_profile(radius_nm: np.ndarray, wavelength_nm: float, numerical_aperture: float) -> np.ndarray:
    k = 2.0 * np.pi / float(wavelength_nm)
    v = k * float(numerical_aperture) * np.asarray(radius_nm, dtype=float)
    airy = np.ones_like(v, dtype=float)
    nz = v > 1.0e-12
    airy[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return np.clip(airy, 0.0, 1.0)


def _params() -> dict:
    from config import default_params

    params = default_params()
    params.update(
        {
            "imaging_model": "bright_field",
            "numerical_aperture": 0.8,
            "refractive_index_medium": 1.0,
            "refractive_index_immersion": 1.0,
            "apodization_factor": 0.0,
            "spherical_aberration_strength": 0.0,
            "random_aberration_strength": 0.0,
            "wavelength_nm": 520.0,
            "probe_wavelength_nm": 520.0,
            "pupil_samples": 128,
            "psf_oversampling_factor": 1,
            "image_size_pixels": 129,
            "pixel_size_nm": 25.0,
            "optical_field_backend": "scalar_paraxial",
            "max_psf_z_slices": 1,
        }
    )
    return params


def test_scalar_psf_matches_airy_first_zero_and_shape() -> None:
    from config import OpticalInstrumentSettings
    from optics import compute_complex_psf_stack

    params = _params()
    instrument = OpticalInstrumentSettings.from_params(params)
    wavelength_nm = float(instrument.probe_wavelength_nm)
    numerical_aperture = float(instrument.numerical_aperture)
    px = float(params["pixel_size_nm"]) / float(params.get("psf_oversampling_factor", 1) or 1)

    interpolator = compute_complex_psf_stack(
        params,
        particle_diameter_nm=20.0,
        particle_refractive_index=1.59 + 0.0j,
        z_values_nm=np.array([0.0]),
    )
    field = np.asarray(interpolator(0.0), dtype=np.complex128)

    if field.ndim == 3 and field.shape[0] == 3:
        intensity = np.abs(field) ** 2
        intensity = intensity.sum(axis=0)
    else:
        intensity = np.abs(field) ** 2

    assert intensity.size > 0
    intensity = intensity / float(np.max(intensity))
    assert np.isfinite(intensity).all()
    assert np.isclose(float(intensity.max()), 1.0, rtol=0.0, atol=1.0e-12)

    rc, prof = _radial_profile(intensity, pixel_size_nm=px)
    assert np.isfinite(prof).any()

    expected_zero = 0.61 * wavelength_nm / numerical_aperture
    measured_zero = _first_zero_radius(rc, prof, expected_zero)
    assert np.isfinite(measured_zero)

    relative_zero_error = abs(measured_zero - expected_zero) / expected_zero
    assert relative_zero_error < 0.15

    airy = _airy_profile(rc, wavelength_nm=wavelength_nm, numerical_aperture=numerical_aperture)
    valid = np.isfinite(prof) & np.isfinite(airy)
    valid &= rc < 2.5 * expected_zero
    assert int(np.count_nonzero(valid)) >= 6

    corr = np.corrcoef(prof[valid], airy[valid])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95
