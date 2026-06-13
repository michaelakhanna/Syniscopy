from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _gaussian_template(A: float, sigma_nm: float, pixel_size_nm: float) -> np.ndarray:
    half = int(np.ceil(6.0 * sigma_nm / pixel_size_nm))
    axis = (np.arange(2 * half + 1, dtype=float) - half) * pixel_size_nm
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    r2 = X * X + Y * Y
    return A * np.exp(-r2 / (2.0 * sigma_nm * sigma_nm))


def _gaussian_crlb_sigma(A: float, sigma_nm: float, pixel_size_nm: float) -> float:
    from fisher import compute_localization_crlb
    from noise_contracts import independent_pixel_noise_model

    contrast = _gaussian_template(A, sigma_nm, pixel_size_nm)
    noise = independent_pixel_noise_model(
        1.0,
        measurement_domain="contrast",
        signal_units="contrast",
        noise_variance_units="contrast_squared",
        context="test_v03_lateral_crlb",
    )
    crlb = compute_localization_crlb(contrast, noise, pixel_size_nm)
    return float(crlb["sigma_x_nm"])


def test_gaussian_crlb_matches_closed_form_pixel_scaling() -> None:
    pixel_sizes = np.array([150.0, 75.0, 37.5, 18.75], dtype=float)
    peak = 1.5
    sigma_nm = 150.0

    observed = np.array(
        [_gaussian_crlb_sigma(peak, sigma_nm, px) for px in pixel_sizes],
        dtype=float,
    )
    expected = pixel_sizes * np.sqrt(2.0 / np.pi) / peak
    rel_err = np.abs(observed - expected) / expected

    assert np.all(np.isfinite(observed))
    assert np.all(rel_err < 0.015)


def test_gaussian_crlb_scaling_invariant_wrt_width_and_amplitude() -> None:
    widths = np.array([100.0, 150.0, 200.0, 300.0], dtype=float)
    amps = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)
    px = 18.75

    width_sigmas = np.array(
        [_gaussian_crlb_sigma(1.0, s, px) for s in widths],
        dtype=float,
    )
    amp_sigmas = np.array(
        [_gaussian_crlb_sigma(a, 150.0, px) for a in amps],
        dtype=float,
    )

    slope_width = float(np.polyfit(np.log(widths), np.log(width_sigmas), 1)[0])
    slope_amp = float(np.polyfit(np.log(amps), np.log(amp_sigmas), 1)[0])

    assert slope_width == pytest.approx(0.0, abs=0.05)
    assert slope_amp == pytest.approx(-1.0, abs=0.05)
