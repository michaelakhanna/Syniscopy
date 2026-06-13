"""Continuous-observable (band-limited) localization Fisher information.

Why this exists
---------------
The localization Fisher information is, physically, a property of the continuous
optical measurement. The old stationary-shift path computed it from
detector-grid central differences of the contrast image under
``I(r; x0, y0) = C(r - r0)``. That made the reported bound depend on the
finite-difference step size and the pixel pitch, so the quantity being gated was
partly an artifact of the discretization, not the physics.

The fix is to define the derivative on the continuous band-limited signal that
the samples represent, rather than as a finite difference between samples. By
the sampling theorem, a grid image sampled at pitch ``Delta`` represents a unique
band-limited continuous function, and the exact spatial derivative of that
function is obtained by the Fourier derivative theorem

    d/dx f  <->  i * 2*pi * xi_x * F(xi),

evaluated on the same grid. This derivative has **no step-size parameter**: it is
the analytic derivative of the represented signal. The only remaining assumption
is band-limiting at the grid Nyquist frequency -- which is a *physical* statement
about the optical transfer function (the renderer already imposes an NA-set band
limit), not a numerical convergence knob. Where that assumption is violated
(energy near Nyquist, or non-compact support breaking FFT periodicity), we report
explicit diagnostics instead of a silent finite-difference drift.

Under the stationary-shift convention, ``d I / d x0 = -(d C / d x)`` on the grid,
so the position Fisher matrix is built directly from the spatial gradient of the
contrast image (the sign drops out under squaring).

This module is pure NumPy and depends only on ``fisher/_constants`` for the
shared tolerances. It is the production lateral derivative owner for
independent-pixel Fisher likelihoods: callers supply a contrast image, a
noise-variance map, and the pixel size, and receive the lateral Fisher matrix
plus the sampling/support diagnostics that replace derivative-step convergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Reuse the shared Fisher tolerances rather than re-declaring magic numbers.
from ._constants import _FISHER_RANK_RELATIVE_TOL
from .precision import (
    apply_analysis_noise_precision,
    compute_fisher_from_gradients_with_noise,
)
from noise_contracts import independent_pixel_noise_model


def _as_image(arr: np.ndarray, *, name: str) -> np.ndarray:
    img = np.asarray(arr, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"{name} must be a 2D image; got shape {img.shape}.")
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{name} must contain only finite values.")
    return img


def spectral_gradient(image: np.ndarray, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact band-limited spatial gradient ``(dI/dx, dI/dy)`` in units of [I]/nm.

    Computes the derivative of the continuous band-limited interpolant of the
    sampled ``image`` via the Fourier derivative theorem. Independent of any
    finite-difference step. The returned arrays are the real-valued physical
    derivatives on the same grid (x is the last axis / columns, y is the first
    axis / rows), consistent with the renderer's image convention.
    """
    img = _as_image(image, name="image")
    if not (np.isfinite(pixel_size_nm) and pixel_size_nm > 0.0):
        raise ValueError(f"pixel_size_nm must be positive finite; got {pixel_size_nm!r}.")
    h, w = img.shape
    # fftfreq(n, d) returns cycles per nm; the derivative multiplier is i*2*pi*freq.
    fx = np.fft.fftfreq(w, d=pixel_size_nm)  # along columns (x)
    fy = np.fft.fftfreq(h, d=pixel_size_nm)  # along rows (y)
    spectrum = np.fft.fft2(img)
    jw_x = (1j * 2.0 * np.pi * fx)[None, :]
    jw_y = (1j * 2.0 * np.pi * fy)[:, None]
    gx = np.fft.ifft2(spectrum * jw_x).real
    gy = np.fft.ifft2(spectrum * jw_y).real
    return gx, gy


def boundary_energy_fraction(image: np.ndarray, border_px: int = 2) -> float:
    """Fraction of absolute image energy on the outer ``border_px`` ring.

    The spectral derivative assumes periodicity; a compactly supported,
    background-subtracted particle contrast decays to ~0 at the frame edge and
    satisfies this well. A large value flags that the particle is not contained
    in the frame, so the FFT-periodicity assumption (not a step size) is the
    thing to fix.
    """
    img = np.abs(_as_image(image, name="image"))
    total = float(img.sum())
    if total <= 0.0:
        return 0.0
    mask = np.zeros(img.shape, dtype=bool)
    b = max(1, int(border_px))
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = True
    return float(img[mask].sum() / total)


def nyquist_band_fraction(image: np.ndarray) -> float:
    """Fraction of spectral energy in the outer quarter of the frequency plane.

    A proxy for how close the contrast is to the grid Nyquist limit. Small means
    the band-limited-at-Nyquist assumption (the only remaining assumption of the
    continuous derivative) holds comfortably; large means the grid undersamples
    the optical band and the pitch should be finer -- a physical sampling
    statement, reported explicitly rather than hidden in a convergence gate.
    """
    img = _as_image(image, name="image")
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    total = float(spec.sum())
    if total <= 0.0:
        return 0.0
    h, w = img.shape
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    inner = float(spec[y0:y1, x0:x1].sum())
    return float(1.0 - inner / total)


def lateral_fisher_continuous(
    contrast: np.ndarray,
    noise_variance: np.ndarray | float,
    pixel_size_nm: float,
) -> dict[str, Any]:
    """Lateral (x, y) localization Fisher from the continuous band-limited derivative.

    Parameters mirror the existing report inputs: a signed ``contrast`` image,
    a pixelwise ``noise_variance`` map (or scalar), and the sample-plane pixel
    size. Returns the 2x2 Fisher matrix and Cramer--Rao summary together with
    band-limit/boundary diagnostics that *replace* the finite-difference
    convergence gate.

    Under the stationary-shift convention the position derivatives are the
    spatial gradient of the contrast image, so

        F_xx = sum_r (dC/dx)^2 / sigma^2(r),   etc.
    """
    c = _as_image(contrast, name="contrast")
    if np.isscalar(noise_variance):
        var = np.full(c.shape, float(noise_variance))
    else:
        var = _as_image(noise_variance, name="noise_variance")
        if var.shape != c.shape:
            raise ValueError("noise_variance map must match the contrast image shape.")
    gx, gy = spectral_gradient(c, pixel_size_nm)
    noise_model = independent_pixel_noise_model(
        var,
        measurement_domain="contrast",
        signal_units="contrast",
        noise_variance_units="contrast_squared",
        status_reason="spectral continuous lateral Fisher independent-pixel likelihood",
        context="lateral_fisher_continuous",
    )
    fisher = compute_fisher_from_gradients_with_noise(
        (gx, gy),
        noise_model,
        context="lateral_fisher_continuous",
    )
    f_xx = float(fisher[0, 0])
    f_xy = float(fisher[0, 1])
    f_yy = float(fisher[1, 1])

    det = f_xx * f_yy - f_xy * f_xy
    trace = f_xx + f_yy
    # scale-relative singularity tolerance from the shared Fisher constants
    singular = bool(det <= _FISHER_RANK_RELATIVE_TOL * max(trace * trace, 1.0))
    if singular or det <= 0.0:
        sigma_x = sigma_y = sigma_xy = float("inf")
    else:
        cov = np.linalg.inv(fisher)
        sigma_x = float(np.sqrt(cov[0, 0]))
        sigma_y = float(np.sqrt(cov[1, 1]))
        sigma_xy = float(np.hypot(sigma_x, sigma_y))

    return {
        "fisher_xx": f_xx,
        "fisher_xy": f_xy,
        "fisher_yy": f_yy,
        "fisher_matrix": fisher,
        "singular": singular,
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_xy_nm": sigma_xy,
        # Diagnostics that replace the derivative-step convergence gate:
        "derivative_basis": "spectral_band_limited",
        "step_size_free": True,
        "boundary_energy_fraction": boundary_energy_fraction(c),
        "nyquist_band_fraction": nyquist_band_fraction(c),
    }


def lateral_information_density_continuous(
    contrast: np.ndarray,
    noise_variance: np.ndarray | float,
    pixel_size_nm: float,
) -> dict[str, np.ndarray]:
    """Per-pixel Fisher density maps using the continuous derivative.

    Drop-in analogue of the report's information-density maps but with the
    spectral gradient, so the maps integrate to the continuous-derivative
    Fisher matrix above.
    """
    c = _as_image(contrast, name="contrast")
    if np.isscalar(noise_variance):
        var = np.full(c.shape, float(noise_variance))
    else:
        var = _as_image(noise_variance, name="noise_variance")
    gx, gy = spectral_gradient(c, pixel_size_nm)
    noise_model = independent_pixel_noise_model(
        var,
        measurement_domain="contrast",
        signal_units="contrast",
        noise_variance_units="contrast_squared",
        status_reason="spectral continuous lateral Fisher-density independent-pixel likelihood",
        context="lateral_information_density_continuous",
    )
    (weighted_gx, weighted_gy), _metadata = apply_analysis_noise_precision(
        (gx, gy),
        noise_model,
        context="lateral_information_density_continuous",
    )
    return {
        "Ix_info_map": gx * weighted_gx,
        "Iy_info_map": gy * weighted_gy,
        "Ixy_info_map": gx * weighted_gy,
    }


__all__ = [
    "spectral_gradient",
    "boundary_energy_fraction",
    "nyquist_band_fraction",
    "lateral_fisher_continuous",
    "lateral_information_density_continuous",
]
