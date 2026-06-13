"""Airy-support guard radius helpers for rendering."""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from config import OpticalInstrumentSettings, OpticalPsfSupportSettings, SamplingGeometry

_AIRY_SUPPORT_RHO_MIN = 1e-4
_AIRY_SUPPORT_RHO_MAX = 200.0
_AIRY_SUPPORT_NUM_SAMPLES = 80000

def _airy_support_radius_pixels(
    params: dict,
    *,
    threshold_key: str | None,
    default_threshold: float,
    max_radius_fraction_of_fov: float,
) -> int:
    """
    Estimate a circular-pupil response support radius in oversampled pixels.

    This is a numerical guard-band calculation. It does not change the optical
    model; it determines how much scene context to include before cropping so
    finite FFTs do not wrap boundary content into the detector FOV.
    """
    sampling = SamplingGeometry.from_params(params)
    img_size = sampling.image_size_pixels
    pixel_size_nm = sampling.detector_pixel_size_nm
    os_factor = sampling.psf_oversampling_factor
    instrument = OpticalInstrumentSettings.from_params(params)
    NA = instrument.numerical_aperture
    n_medium = instrument.refractive_index_medium
    wavelength_nm = instrument.probe_wavelength_nm

    if img_size <= 0 or pixel_size_nm <= 0 or os_factor <= 0:
        raise ValueError(
            "parameters['image_size_pixels'], parameters['pixel_size_nm'], and "
            "parameters['psf_oversampling_factor'] must all be positive."
        )

    if NA <= 0.0 or wavelength_nm <= 0.0 or n_medium <= 0.0:
        return 0

    threshold = (
        float(default_threshold)
        if threshold_key is None
        else OpticalPsfSupportSettings.from_params(params).intensity_fraction_threshold
    )
    if not (0.0 < threshold < 1.0):
        if threshold_key is None:
            raise ValueError("Airy support threshold must be in the open interval (0, 1).")
        raise ValueError(f"parameters['{threshold_key}'] must be in the open interval (0, 1).")

    # Scan Airy intensity over a large normalized-radius interval; rho=200
    # covers far sidelobes for the supported threshold range, and 80k samples
    # keeps the guard-band estimate stable without affecting frame rendering.
    rho = np.linspace(
        _AIRY_SUPPORT_RHO_MIN,
        _AIRY_SUPPORT_RHO_MAX,
        _AIRY_SUPPORT_NUM_SAMPLES,
    )
    x = np.pi * rho

    I_rel = (2.0 * j1(x) / x) ** 2

    indices_above = np.where(I_rel >= threshold)[0]
    if indices_above.size == 0:
        rho_crit = 0.0
    else:
        rho_crit = float(rho[indices_above[-1]])

    radius_nm = rho_crit * wavelength_nm / NA

    psf_size_nm = img_size * pixel_size_nm
    max_radius_nm = float(max_radius_fraction_of_fov) * psf_size_nm
    if max_radius_nm > 0.0:
        radius_nm = min(radius_nm, max_radius_nm)

    radius_pixels_oversampled = radius_nm / pixel_size_nm * os_factor

    padding_pixels = int(np.ceil(radius_pixels_oversampled)) + 1
    return max(padding_pixels, 0)


def estimate_psf_padding_radius_pixels(params):
    """
    Estimate the extra padding radius (in oversampled pixels) required around
    the simulated field of view so that PSF contributions from particles
    located just outside the nominal FOV can be represented on the padded
    canvas without significant truncation in the central region that is
    ultimately written to the video.

    The estimate uses an Airy-pattern approximation for the PSF of a circular
    aperture. We compute the normalized intensity

        I_rel(r) = I(r) / I(0) ~= [2 J1(pi * rho) / (pi * rho)]^2,

    where rho = (NA * r) / lambda_vacuum is a dimensionless radial
    coordinate.  ``numerical_aperture`` is the physical NA = n sin(theta);
    dividing the wavelength by the immersion-medium index here would apply the
    medium factor twice and underestimate the guard band.

    We then find the largest radius r such that I_rel(r) is still above a
    user-controllable fraction:

        I_rel(r) >= psf_intensity_fraction_threshold,

    and treat everything beyond that radius as negligible. The corresponding
    physical radius is converted into oversampled pixels using the current
    imaging geometry.

    Args:
        params (dict): Global simulation parameter dictionary (parameters).

    Returns:
        int: Padding radius in oversampled pixels (>= 0).
    """
    return _airy_support_radius_pixels(
        params,
        threshold_key="psf_intensity_fraction_threshold",
        default_threshold=1e-4,
        max_radius_fraction_of_fov=0.5,
    )


def estimate_optical_filter_guard_radius_pixels(params):
    """
    Estimate the guard band for Fourier-domain optical filtering before crop.

    This uses a stricter Airy-tail threshold and a wider safety cap than the
    particle-placement padding because the coherent pupil filter is applied to
    the whole scene field, including empirical/background structure.
    """
    return _airy_support_radius_pixels(
        params,
        threshold_key=None,
        default_threshold=1e-5,
        max_radius_fraction_of_fov=1.0,
    )
