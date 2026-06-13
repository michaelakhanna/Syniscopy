"""Mie scattering coefficient and amplitude helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import jv as jn, yv as yn

# Mie derivatives use half-integer Bessel orders. scipy.special.jv/yv accept
# arbitrary real orders, unlike the integer-order jn/yn aliases.


def _validate_mie_inputs(m, x) -> tuple[complex, float]:
    m_out = complex(m)
    x_out = float(x)
    if not np.isfinite(m_out) or abs(m_out) <= 0.0:
        raise ValueError(
            "Mie refractive-index ratio m must be finite and nonzero; "
            f"got {m_out!r}."
        )
    if not np.isfinite(x_out) or x_out <= 0.0:
        raise ValueError(f"Mie size parameter x must be finite and positive; got {x_out}.")
    return m_out, x_out


@lru_cache(maxsize=512)
def _mie_an_bn_cached(m_real: float, m_imag: float, x: float) -> tuple[np.ndarray, np.ndarray]:
    m = complex(m_real, m_imag)
    nmax = int(np.ceil(x + 4 * x ** (1 / 3) + 2))
    n = np.arange(1, nmax + 1)

    psi_n_x = np.sqrt(0.5 * np.pi * x) * jn(n + 0.5, x)
    psi_n_mx = np.sqrt(0.5 * np.pi * m * x) * jn(n + 0.5, m * x)
    chi_n_x = -np.sqrt(0.5 * np.pi * x) * yn(n + 0.5, x)

    psi_nm1_x = np.sqrt(0.5 * np.pi * x) * jn(n - 1 + 0.5, x)
    psi_nm1_mx = np.sqrt(0.5 * np.pi * m * x) * jn(n - 1 + 0.5, m * x)
    chi_nm1_x = -np.sqrt(0.5 * np.pi * x) * yn(n - 1 + 0.5, x)

    psi_prime_n_x = psi_nm1_x - n * psi_n_x / x
    psi_prime_n_mx = psi_nm1_mx - n * psi_n_mx / (m * x)

    xi_n_x = psi_n_x + 1j * chi_n_x
    chi_prime_n_x = chi_nm1_x - n * chi_n_x / x
    xi_prime_n_x = psi_prime_n_x + 1j * chi_prime_n_x

    denom_a = m * psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx
    denom_b = psi_n_mx * xi_prime_n_x - m * xi_n_x * psi_prime_n_mx
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a_n = (
            (m * psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx)
            / denom_a
        )
        b_n = (
            (psi_n_mx * psi_prime_n_x - m * psi_n_x * psi_prime_n_mx)
            / denom_b
        )
    if not (np.all(np.isfinite(a_n)) and np.all(np.isfinite(b_n))):
        raise FloatingPointError(
            "Mie coefficient calculation produced nonfinite coefficients; "
            "check m and x for a singular or unsupported regime."
        )
    a_n.setflags(write=False)
    b_n.setflags(write=False)
    return a_n, b_n


def mie_an_bn(m, x):
    """
    Calculate Mie scattering coefficients a_n and b_n.

    Args:
        m (complex): Complex refractive-index ratio, particle over medium.
        x (float): Size parameter, 2*pi*r/lambda.
    """
    m, x = _validate_mie_inputs(m, x)
    a_n, b_n = _mie_an_bn_cached(float(m.real), float(m.imag), float(x))
    return a_n.copy(), b_n.copy()


def mie_scattering_amplitudes_from_coefficients(a_n, b_n, mu, *, include_s1=True):
    """
    Calculate Mie angular scattering amplitudes from precomputed coefficients.

    Returns S2 only when ``include_s1`` is false, otherwise returns ``(S1, S2)``.
    """
    a_n = np.asarray(a_n, dtype=np.complex128)
    b_n = np.asarray(b_n, dtype=np.complex128)
    if a_n.ndim != 1 or b_n.ndim != 1 or a_n.shape != b_n.shape:
        raise ValueError(
            "Mie coefficient arrays a_n and b_n must be one-dimensional "
            "arrays with matching shape."
        )
    if a_n.size == 0:
        raise ValueError("Mie coefficient arrays must contain at least one term.")
    if not (np.all(np.isfinite(a_n)) and np.all(np.isfinite(b_n))):
        raise ValueError("Mie coefficient arrays must contain only finite values.")
    nmax = len(a_n)
    mu_arr = np.asarray(mu, dtype=float)
    scalar_input = mu_arr.ndim == 0
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("Mie angular cosine mu must contain only finite values.")
    if np.any((mu_arr < -1.0) | (mu_arr > 1.0)):
        raise ValueError("Mie angular cosine mu must be within [-1, 1].")

    out_shape = mu_arr.shape
    s1 = np.zeros(out_shape, dtype=np.complex128)
    s2 = np.zeros(out_shape, dtype=np.complex128)
    pi_n = np.zeros((nmax + 2,) + out_shape, dtype=float)
    tau_n = np.zeros((nmax + 2,) + out_shape, dtype=float)
    pi_n[1] = 1.0

    for n in range(1, nmax + 1):
        if n > 1:
            pi_n[n] = (
                ((2 * n - 1) / (n - 1)) * mu_arr * pi_n[n - 1]
                - (n / (n - 1)) * pi_n[n - 2]
            )

        tau_n[n] = n * mu_arr * pi_n[n] - (n + 1) * pi_n[n - 1]

        factor = (2 * n + 1) / (n * (n + 1))
        if include_s1:
            s1 += factor * (a_n[n - 1] * pi_n[n] + b_n[n - 1] * tau_n[n])
        s2 += factor * (a_n[n - 1] * tau_n[n] + b_n[n - 1] * pi_n[n])

    if not np.all(np.isfinite(s2)) or (
        include_s1 and not np.all(np.isfinite(s1))
    ):
        raise FloatingPointError(
            "Mie scattering amplitude calculation produced nonfinite values."
        )

    if scalar_input:
        s2 = s2.item()
        if include_s1:
            s1 = s1.item()
    if include_s1:
        return s1, s2
    return s2


def mie_S1_S2_from_coefficients(a_n, b_n, mu):
    """Return the standard pair of Mie scattering amplitudes, S1 and S2."""
    return mie_scattering_amplitudes_from_coefficients(
        a_n,
        b_n,
        mu,
        include_s1=True,
    )


def mie_S2_from_coefficients(a_n, b_n, mu):
    """Return S2 only for scalar coherent backends."""
    return mie_scattering_amplitudes_from_coefficients(
        a_n,
        b_n,
        mu,
        include_s1=False,
    )


def mie_scattering_cross_section_nm2(
    m,
    diameter_nm,
    wavelength_nm,
    n_medium,
    *,
    collection_half_angle_rad=None,
    n_theta=721,
):
    r"""Physical Mie scattering cross-section in nm^2 (optical-theorem definition).

    This is the absolute-normalization anchor for the rendered scattered field:
    the scattered power a particle actually puts into the (collected) solid angle.
    It is derived ONLY from the validated Mie amplitudes -- no tuned constant.

    With wavenumber in the medium ``k = 2*pi*n_medium/lambda`` and the unpolarized
    differential cross-section ``d_sigma/d_Omega = (|S1|^2 + |S2|^2) / (2 k^2)``,

        sigma = \int d_sigma/d_Omega * 2*pi*sin(theta) d_theta

    integrated from 0 to ``collection_half_angle_rad`` (forward cone). When the
    collection angle is ``None`` the integral runs 0..pi and returns the TOTAL
    scattering cross-section, which equals ``Qsca * pi * r^2`` and is directly
    checkable against PyMieScatt's ``MieQ`` (this is what L02 should assert).

    Args:
        m: complex refractive-index ratio (particle / medium), Syniscopy convention.
        diameter_nm: particle diameter (nm).
        wavelength_nm: vacuum wavelength (nm).
        n_medium: medium refractive index (real, > 0).
        collection_half_angle_rad: forward-cone half angle; e.g.
            ``arcsin(NA / n_medium)`` for an objective. ``None`` -> total (pi).
        n_theta: angular quadrature samples.

    Returns:
        float: scattering cross-section in nm^2 over the requested cone.
    """
    n_medium = float(n_medium)
    if not np.isfinite(n_medium) or n_medium <= 0.0:
        raise ValueError(f"n_medium must be finite and positive; got {n_medium!r}.")
    wavelength_nm = float(wavelength_nm)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(f"wavelength_nm must be finite and positive; got {wavelength_nm!r}.")
    radius_nm = 0.5 * float(diameter_nm)
    if not np.isfinite(radius_nm) or radius_nm <= 0.0:
        raise ValueError(f"diameter_nm must be finite and positive; got {diameter_nm!r}.")

    # Size parameter uses the in-medium wavelength: x = 2*pi*r*n_medium/lambda_vac.
    k = 2.0 * np.pi * n_medium / wavelength_nm  # 1/nm
    x = k * radius_nm
    a_n, b_n = mie_an_bn(m, x)

    theta_max = np.pi if collection_half_angle_rad is None else float(collection_half_angle_rad)
    theta_max = float(np.clip(theta_max, 0.0, np.pi))
    if theta_max <= 0.0:
        return 0.0
    theta = np.linspace(0.0, theta_max, int(n_theta))
    mu = np.cos(theta)
    S1, S2 = mie_S1_S2_from_coefficients(a_n, b_n, mu)
    dsigma_dOmega = (np.abs(S1) ** 2 + np.abs(S2) ** 2) / (2.0 * k * k)  # nm^2 / sr
    integrand = dsigma_dOmega * 2.0 * np.pi * np.sin(theta)
    sigma = float(np.trapz(integrand, theta))
    if not np.isfinite(sigma) or sigma < 0.0:
        raise FloatingPointError(f"Mie cross-section integral produced {sigma!r}.")
    return sigma
