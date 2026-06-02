"""Mie scattering coefficient and amplitude helpers."""

from __future__ import annotations

import numpy as np
from scipy.special import jv as jn, yv as yn

# Mie derivatives use half-integer Bessel orders. scipy.special.jv/yv accept
# arbitrary real orders, unlike the integer-order jn/yn aliases.


def mie_an_bn(m, x):
    """
    Calculate Mie scattering coefficients a_n and b_n.

    Args:
        m (complex): Complex refractive-index ratio, particle over medium.
        x (float): Size parameter, 2*pi*r/lambda.
    """
    m = complex(m)
    x = float(x)
    if not np.isfinite(m) or abs(m) <= 0.0:
        raise ValueError(
            "Mie refractive-index ratio m must be finite and nonzero; "
            f"got {m!r}."
        )
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"Mie size parameter x must be finite and positive; got {x}.")
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

    return a_n, b_n


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
