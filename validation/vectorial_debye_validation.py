#!/usr/bin/env python3
"""Validate vectorial Debye/Richards-Wolf implementation consequences.

This is not a high-NA external package match. It checks analytic consequences
of the Richards-Wolf vectorial pupil construction: polarization rotation
covariance, zero longitudinal/cross-polarized field at the optical axis for an
x-polarized focus, and reduction to the scalar Airy first dark ring in the
low-NA limit.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from config import PARAMS  # noqa: E402
from vectorial_optics import compute_vectorial_debye_psf  # noqa: E402


def _base_params(**updates):
    params = dict(PARAMS)
    params.update(
        {
            "optical_field_backend": "vectorial_debye",
            "vectorial_pupil_samples": 256,
            "pupil_samples": 256,
            "psf_oversampling_factor": 2,
            "pixel_size_nm": 100.0,
            "wavelength_nm": 550.0,
            "probe_wavelength_nm": 550.0,
            "numerical_aperture": 0.2,
            "refractive_index_medium": 1.0,
            "polarization_model": "linear_x",
            "vectorial_polarization_rotation_deg": 0.0,
            "apodization_factor": 0.0,
            "spherical_aberration_strength": 0.0,
            "random_aberration_strength": 0.0,
            "coverslip_correction_enabled": False,
            "coverslip_aberration_model": "none",
            "vectorial_obliquity_apodization": False,
        }
    )
    params.update(updates)
    return params


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0e-30)
    return float(np.max(np.abs(a - b)) / denom)


def _first_dark_ring_radius_nm(intensity: np.ndarray, pitch_nm: float) -> tuple[float, int | None]:
    image = np.asarray(intensity, dtype=float)
    image = image / float(np.max(image))
    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    yy, xx = np.indices(image.shape, dtype=float)
    radius_px = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    bins = np.arange(0, int(radius_px.max()))
    profile = np.array(
        [
            image[(radius_px >= bin_index) & (radius_px < bin_index + 1)].mean()
            for bin_index in bins
        ]
    )
    for bin_index in range(2, len(profile) - 1):
        if (
            profile[bin_index] < profile[bin_index - 1]
            and profile[bin_index] <= profile[bin_index + 1]
            and profile[bin_index] < 0.2
        ):
            return float(bin_index * pitch_nm), int(bin_index)
    return float("nan"), None


def _print_result(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def check_rotation_covariance() -> bool:
    params_x = _base_params(numerical_aperture=0.7, psf_oversampling_factor=1)
    params_y = dict(params_x)
    params_y["polarization_model"] = "linear_y"
    x_components = compute_vectorial_debye_psf(params_x, [0.0])
    y_components = compute_vectorial_debye_psf(params_y, [0.0])
    errors = {
        "Ex_vs_Ey_transpose": _relative_error(x_components["Ex"][0], y_components["Ey"][0].T),
        "Ey_vs_Ex_transpose": _relative_error(x_components["Ey"][0], y_components["Ex"][0].T),
        "Ez_vs_Ez_transpose": _relative_error(x_components["Ez"][0], y_components["Ez"][0].T),
    }
    max_error = max(errors.values())
    return _print_result(
        "vectorial_rotation_covariance",
        max_error <= 1.0e-12,
        f"max_relative_error={max_error:.3e} component_errors={errors}",
    )


def check_on_axis_polarization() -> bool:
    params = _base_params(numerical_aperture=0.7, psf_oversampling_factor=1)
    components = compute_vectorial_debye_psf(params, [0.0])
    center = components["Ex"].shape[-1] // 2
    ex0 = abs(complex(components["Ex"][0, center, center]))
    ey0 = abs(complex(components["Ey"][0, center, center]))
    ez0 = abs(complex(components["Ez"][0, center, center]))
    ok = ex0 > 0.0 and ey0 <= 1.0e-14 and ez0 <= 1.0e-14
    return _print_result(
        "vectorial_on_axis_polarization",
        ok,
        f"|Ex(0)|={ex0:.3e} |Ey(0)|={ey0:.3e} |Ez(0)|={ez0:.3e}",
    )


def check_low_na_airy_limit() -> bool:
    numerical_aperture = 0.2
    wavelength_nm = 550.0
    pixel_size_nm = 100.0
    oversampling = 2
    params = _base_params(
        numerical_aperture=numerical_aperture,
        wavelength_nm=wavelength_nm,
        probe_wavelength_nm=wavelength_nm,
        pixel_size_nm=pixel_size_nm,
        psf_oversampling_factor=oversampling,
    )
    components = compute_vectorial_debye_psf(params, [0.0])
    intensity = sum(np.abs(components[name][0]) ** 2 for name in ("Ex", "Ey", "Ez"))
    measured_nm, first_bin = _first_dark_ring_radius_nm(intensity, pixel_size_nm / oversampling)
    expected_nm = 0.61 * wavelength_nm / numerical_aperture
    relative_error = abs(measured_nm / expected_nm - 1.0)
    return _print_result(
        "vectorial_low_na_airy_limit",
        first_bin is not None and relative_error <= 0.03,
        (
            f"measured_first_zero_nm={measured_nm:.3f} "
            f"expected_first_zero_nm={expected_nm:.3f} "
            f"relative_error={relative_error:.3e}"
        ),
    )


def run_validation() -> bool:
    checks = [
        check_rotation_covariance,
        check_on_axis_polarization,
        check_low_na_airy_limit,
    ]
    return all(check() for check in checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_validation() else 1)
