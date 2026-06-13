from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _build_linear_z_stack() -> tuple[np.ndarray, np.ndarray]:
    A, s_nm, px, dz, alpha = 1.0, 150.0, 30.0, 40.0, 1.0e-3
    half = int(np.ceil(6 * s_nm / px))
    n = 2 * half + 1
    ax = (np.arange(n, dtype=float) - half) * px
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    G = np.exp(-(X**2 + Y**2) / (2 * s_nm**2))

    def plane(z: float) -> np.ndarray:
        return A * (1.0 + alpha * z) * G

    stack = np.stack([plane(-dz), plane(0.0), plane(+dz)], axis=0)
    dC_dz = alpha * A * G
    return stack, dC_dz


def test_axial_fisher_and_crlb_match_analytic_linear_z_model() -> None:
    from fisher import compute_fisher_information_3d, compute_localization_crlb_3d
    from noise_contracts import independent_pixel_noise_model

    stack, dC_dz = _build_linear_z_stack()
    px, dz = 30.0, 40.0
    var = 2.0
    noise_model = independent_pixel_noise_model(
        np.full_like(stack[1], var, dtype=float),
        measurement_domain="contrast",
        signal_units="contrast",
        noise_variance_units="contrast_squared",
    )

    fisher = compute_fisher_information_3d(
        stack,
        noise_model,
        pixel_size_nm=px,
        z_step_nm=dz,
    )
    crlb = compute_localization_crlb_3d(
        stack,
        noise_model,
        pixel_size_nm=px,
        z_step_nm=dz,
    )

    assert np.isfinite(fisher).all()
    assert np.allclose(fisher, fisher.T, rtol=0.0, atol=1.0e-12)
    assert np.abs(float(fisher[2, 2])) > 0.0

    fzz_analytic = float(np.sum((dC_dz**2) / var))
    fzz_syn = float(fisher[2, 2])
    assert fzz_syn == pytest.approx(fzz_analytic, rel=2.0e-5, abs=0.0)

    sigma_z_syn = float(crlb["sigma_z_nm"])
    sigma_z_analytic = 1.0 / np.sqrt(fzz_syn)
    assert sigma_z_syn == pytest.approx(sigma_z_analytic, rel=1.0e-6, abs=0.0)
    assert np.max(np.abs(fisher[:2, 2])) <= 1.0e-8 * max(1.0, abs(fzz_syn))
