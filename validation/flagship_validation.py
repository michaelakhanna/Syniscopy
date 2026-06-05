"""Level-B validation of flagship label-free contrast identities.

Checks:
- coherent bright-field (COBRI) contrast identity;
- DPC phase-gradient recovery;
- half-plane DPC sign behavior.
"""

from __future__ import annotations

from copy import deepcopy
import sys

import numpy as np

sys.path.insert(0, "codebase")

from config import PARAMS  # noqa: E402
from imaging_models.coherent_brightfield import CoherentBrightfieldImagingModel  # noqa: E402
from imaging_models.dpc import DifferentialPhaseContrastImagingModel as DPC  # noqa: E402


def _pass_fail(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def check_cobri_identity(tolerance: float) -> bool:
    rng = np.random.default_rng(1)
    h = w = 48
    e_sca = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))) * 0.05
    checks = []
    for amp in (1.0, 2.5):
        params = deepcopy(PARAMS)
        params.update(
            {
                "imaging_model": "coherent_bright_field",
                "reference_field_amplitude": amp,
            }
        )
        model = CoherentBrightfieldImagingModel(params)
        contrast = model.compute_per_particle_contrast(e_sca, None, params)
        incident = np.full((h, w), amp, dtype=np.complex128)
        expected = 2.0 * np.real(np.conj(incident) * e_sca) + np.abs(e_sca) ** 2
        error = float(np.max(np.abs(contrast - expected)))
        checks.append(
            _pass_fail(
                f"cobri_contrast_identity_amp_{amp:g}",
                error <= tolerance,
                f"max_abs_error={error:.3e}",
            )
        )
    return all(checks)


def check_dpc_phase_gradient(relative_tolerance: float) -> bool:
    pixel_size_nm = 100.0
    n = 128
    x = np.arange(n) * pixel_size_nm
    x_grid = np.broadcast_to(x, (n, n))
    checks = []
    for gradient in (2.0e-4, 5.0e-4, -3.0e-4):
        field = np.exp(1j * gradient * x_grid)
        dphi_dx, dphi_dy = DPC._dpc_components(field, pixel_size_nm)
        recovered = float(np.median(dphi_dx[8:-8, 8:-8]))
        y_residual = float(np.max(np.abs(dphi_dy[8:-8, 8:-8])))
        rel_error = abs(recovered / gradient - 1.0)
        checks.append(
            _pass_fail(
                f"dpc_phase_gradient_{gradient:+.1e}",
                rel_error <= relative_tolerance and y_residual <= 1.0e-12,
                f"recovered={recovered:+.3e} rel_error={rel_error:.3e} y_residual={y_residual:.3e}",
            )
        )
    return all(checks)


def check_dpc_half_plane_sign() -> bool:
    pixel_size_nm = 100.0
    n = 128
    params = deepcopy(PARAMS)
    params.update(
        {
            "imaging_model": "dpc",
            "numerical_aperture": 0.6,
            "refractive_index_medium": 1.0,
            "wavelength_nm": 550.0,
            "pixel_size_nm": pixel_size_nm,
            "psf_oversampling_factor": 1,
        }
    )
    flat = np.ones((n, n), dtype=np.complex128)
    dx_flat, _ = DPC._asymmetric_pupil_dpc_components(flat, pixel_size_nm, params)
    x_grid = np.broadcast_to(np.arange(n) * pixel_size_nm, (n, n))
    field_positive = np.exp(1j * 3.0e-4 * x_grid)
    field_negative = np.exp(-1j * 3.0e-4 * x_grid)
    dx_positive, _ = DPC._asymmetric_pupil_dpc_components(
        field_positive,
        pixel_size_nm,
        params,
    )
    dx_negative, _ = DPC._asymmetric_pupil_dpc_components(
        field_negative,
        pixel_size_nm,
        params,
    )
    flat_mean = float(np.mean(np.abs(dx_flat)))
    positive_median = float(np.median(dx_positive))
    negative_median = float(np.median(dx_negative))
    ok = (
        flat_mean <= 1.0e-12
        and positive_median != 0.0
        and negative_median != 0.0
        and np.sign(positive_median) == -np.sign(negative_median)
    )
    return _pass_fail(
        "dpc_half_plane_sign",
        ok,
        (
            f"flat_mean_abs={flat_mean:.3e} "
            f"positive_median={positive_median:+.3e} "
            f"negative_median={negative_median:+.3e}"
        ),
    )


def run_validation(*, tolerance: float, gradient_relative_tolerance: float) -> bool:
    return all(
        (
            check_cobri_identity(tolerance),
            check_dpc_phase_gradient(gradient_relative_tolerance),
            check_dpc_half_plane_sign(),
        )
    )


def main() -> int:
    ok = run_validation(tolerance=1.0e-12, gradient_relative_tolerance=1.0e-12)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
