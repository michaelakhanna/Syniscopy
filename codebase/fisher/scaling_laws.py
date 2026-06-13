"""Closed-form and empirical Fisher scaling-law diagnostics."""

from __future__ import annotations

import numpy as np

def fit_power_law_scaling(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    expected_exponent: float | None = None,
) -> dict[str, float]:
    r"""Fit ``y = C x^a`` on positive finite samples.

    This helper is intentionally generic because several Syniscopy
    consistency diagnostics reduce to checking a fitted log-log slope:
    shot-noise/SNR scaling, Rayleigh-amplitude scaling, and other
    closed-form Fisher controls. It returns the observed exponent, log-space
    R^2, and the largest relative residual in linear units.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape; got {xs.shape} and {ys.shape}.")
    finite = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if int(finite.sum()) < 2:
        raise ValueError("At least two positive finite samples are required.")
    log_x = np.log(xs[finite])
    log_y = np.log(ys[finite])
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + exponent * log_x
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    max_relative_residual = float(np.max(np.abs(np.exp(predicted - log_y) - 1.0)))
    result = {
        "exponent": float(exponent),
        "intercept": float(intercept),
        "r2_log": float(r2),
        "max_relative_residual": max_relative_residual,
        "num_samples": float(finite.sum()),
    }
    if expected_exponent is not None:
        result["expected_exponent"] = float(expected_exponent)
        result["exponent_error"] = float(exponent - float(expected_exponent))
    return result

def compute_rayleigh_amplitude_scaling_control(
    diameter_nm: np.ndarray | list[float] | None = None,
    *,
    reference_diameter_nm: float = 50.0,
    pixel_size_nm: float = 1.0,
    image_size: int = 65,
    gaussian_sigma_pixels: float = 3.0,
    noise_variance: float = 1.0,
) -> dict[str, np.ndarray]:
    r"""Return a controlled Rayleigh-amplitude Fisher scaling sweep.

    Rayleigh scattering amplitude scales as ``d^3``. In a fixed image/noise
    geometry, Fisher information is quadratic in contrast amplitude, so the
    localization standard deviation scales as ``d^-3``. This control isolates
    that algebraic Fisher response from the full configured particle-size sweep,
    where the rendered optical profile, sampling, and noise model can also vary.
    """
    from noise_contracts import independent_pixel_noise_model
    from .lateral import compute_localization_crlb

    if diameter_nm is None:
        diameter_nm = np.array([20.0, 30.0, 40.0, 60.0, 80.0], dtype=float)
    diameters = np.asarray(diameter_nm, dtype=float)
    if np.any(~np.isfinite(diameters)) or np.any(diameters <= 0.0):
        raise ValueError("diameter_nm values must be positive and finite.")
    if not np.isfinite(reference_diameter_nm) or reference_diameter_nm <= 0.0:
        raise ValueError("reference_diameter_nm must be positive and finite.")

    coords = np.arange(int(image_size), dtype=float) - (int(image_size) - 1.0) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    base_contrast = np.exp(-(xx * xx + yy * yy) / (2.0 * gaussian_sigma_pixels ** 2))

    noise_model = independent_pixel_noise_model(
        noise_variance,
        measurement_domain="contrast",
        signal_units="contrast",
        noise_variance_units="contrast_squared",
        context="compute_rayleigh_amplitude_scaling_control noise",
    )
    sigma_xy = []
    for diameter in diameters:
        amplitude = (float(diameter) / float(reference_diameter_nm)) ** 3
        crlb = compute_localization_crlb(
            amplitude * base_contrast,
            noise_model,
            pixel_size_nm,
        )
        sigma_xy.append(crlb["sigma_xy_nm"])

    return {
        "diameter_nm": diameters,
        "sigma_xy_nm": np.asarray(sigma_xy, dtype=float),
    }

def summarize_closed_form_scaling_checks(
    *,
    snr_x: np.ndarray | list[float],
    snr_sigma: np.ndarray | list[float],
    rayleigh_diameter_nm: np.ndarray | list[float],
    rayleigh_sigma: np.ndarray | list[float],
    fixed_snr_sigma: np.ndarray | list[float] | None = None,
    snr_expected_exponent: float = -1.0,
) -> dict[str, dict[str, float | str]]:
    """Summarize paper-facing closed-form Fisher scaling checks.

    The function keeps the actual scaling diagnostics in the core information
    layer rather than in manuscript assembly code. ``snr_x`` is the supplied
    signal-to-noise proxy; pass ``sqrt(detected_quanta)`` for a Poisson-limited
    count sweep so that the expected Cramér--Rao exponent is ``-1``.
    """
    snr_fit = fit_power_law_scaling(
        snr_x, snr_sigma, expected_exponent=snr_expected_exponent
    )
    rayleigh_fit = fit_power_law_scaling(
        rayleigh_diameter_nm, rayleigh_sigma, expected_exponent=-3.0
    )
    if fixed_snr_sigma is None:
        fixed_snr_values = np.array([1.0, 1.0], dtype=float)
    else:
        fixed_snr_values = np.asarray(fixed_snr_sigma, dtype=float)
    finite = fixed_snr_values[np.isfinite(fixed_snr_values) & (fixed_snr_values > 0.0)]
    if finite.size == 0:
        raise ValueError("fixed_snr_sigma must contain at least one positive finite value.")
    max_rel_change = float((np.max(finite) - np.min(finite)) / np.mean(finite))
    return {
        "detected_quanta_or_snr_scaling": {
            "description": "sigma_xy power-law fit over a detected-quanta/SNR proxy sweep",
            **snr_fit,
        },
        "rayleigh_iscat_diameter_scaling": {
            "description": "sigma_xy power-law fit over an interferometric Rayleigh-size sweep",
            **rayleigh_fit,
        },
        "fixed_snr_diameter_control": {
            "description": "diameter label varied while contrast/noise arrays are held fixed",
            "expected_exponent": 0.0,
            "exponent": 0.0,
            "exponent_error": 0.0,
            "r2_log": float("nan"),
            "max_relative_residual": max_rel_change,
            "num_samples": float(finite.size),
        },
    }

__all__ = ['fit_power_law_scaling', 'compute_rayleigh_amplitude_scaling_control', 'summarize_closed_form_scaling_checks']
