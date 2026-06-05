#!/usr/bin/env python3
"""Validate Syniscopy physical TEM multislice invariants.

The internal checks are fast and deterministic:
- free-space/zero-potential transmission preserves a unit plane wave;
- pure transmission phase equals the input slab phase;
- the weak-phase/single-slice limit matches the TEM CTF proxy convention.

The abTEM check is optional because it depends on a generated external
reference fixture. Run ``validation/run_abtem_reference.py`` once in an
environment with abTEM/ASE, then this validator compares Syniscopy against the
saved ``validation/abtem_reference.npz`` arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from config import PARAMS  # noqa: E402
from imaging_models.tem import TransmissionElectronMicroscopyImagingModel  # noqa: E402


ABTEM_REFERENCE_PATH = ROOT / "validation" / "abtem_reference.npz"


def _params(**updates: object) -> dict:
    params = dict(PARAMS)
    params.update(
        {
            "imaging_model": "tem_phase_contrast",
            "tem_model": "multislice_physical",
            "tem_backend": "multislice_physical",
            "tem_multislice_slices": 1,
            "tem_slice_thickness_nm": None,
            "tem_reference_status": "physics_based_unvalidated",
            "tem_reference_validation_hash": None,
        }
    )
    params.update(updates)
    return params


def _backend(**updates: object):
    return TransmissionElectronMicroscopyImagingModel(_params(**updates))._tem_high_fidelity_backend


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value)))) if np.asarray(value).size else 0.0


def _relative_l2(observed: np.ndarray, expected: np.ndarray) -> float:
    observed_arr = np.asarray(observed)
    expected_arr = np.asarray(expected)
    denominator = float(np.linalg.norm(expected_arr.ravel()))
    if denominator == 0.0:
        return float(np.linalg.norm(observed_arr.ravel()))
    return float(np.linalg.norm((observed_arr - expected_arr).ravel()) / denominator)


def _phase_aligned(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed_arr = np.asarray(observed, dtype=np.complex128)
    expected_arr = np.asarray(expected, dtype=np.complex128)
    alignment = np.vdot(observed_arr.ravel(), expected_arr.ravel())
    if abs(alignment) == 0.0:
        return observed_arr
    return observed_arr * (alignment / abs(alignment))


def _pearson_correlation(observed: np.ndarray, expected: np.ndarray) -> float:
    observed_arr = np.asarray(observed, dtype=float).ravel()
    expected_arr = np.asarray(expected, dtype=float).ravel()
    observed_arr = observed_arr - float(observed_arr.mean())
    expected_arr = expected_arr - float(expected_arr.mean())
    denominator = float(np.linalg.norm(observed_arr) * np.linalg.norm(expected_arr))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(observed_arr, expected_arr) / denominator)


def _pass_fail(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def check_free_space(tolerance: float) -> bool:
    backend = _backend()
    phase = np.zeros((64, 64), dtype=float)
    intensity = backend.intensity_from_projected_phase(phase)
    error = _max_abs(intensity - 1.0)
    return _pass_fail("free_space_unit_intensity", error <= tolerance, f"max_abs_error={error:.3e}")


def check_slab_phase(tolerance: float) -> bool:
    backend = _backend(tem_multislice_slices=4, tem_slice_thickness_nm=None)
    y, x = np.indices((48, 48), dtype=float)
    phase = 1.0e-3 * np.exp(-((x - 24.0) ** 2 + (y - 24.0) ** 2) / (2.0 * 7.0 ** 2))
    exit_wave = backend.exit_wave_after_transmission_only(phase)
    observed = np.angle(exit_wave)
    error = _max_abs(observed - phase)
    return _pass_fail("slab_phase_equals_sigma_v_t", error <= tolerance, f"max_abs_error_rad={error:.3e}")


def check_weak_phase_ctf(relative_tolerance: float, absolute_tolerance: float) -> bool:
    params = _params(tem_multislice_slices=1, tem_slice_thickness_nm=None)
    model = TransmissionElectronMicroscopyImagingModel(params)
    backend = model._tem_high_fidelity_backend
    y, x = np.indices((96, 96), dtype=float)
    phase = 1.0e-4 * np.exp(-((x - 48.0) ** 2 + (y - 48.0) ** 2) / (2.0 * 9.0 ** 2))
    physical = backend.contrast_from_projected_phase(phase)
    proxy = model._ctf_backend.apply_ctf(phase)
    error = _max_abs(physical - proxy)
    scale = max(_max_abs(proxy), absolute_tolerance)
    rel_error = error / scale
    ok = error <= absolute_tolerance or rel_error <= relative_tolerance
    return _pass_fail(
        "weak_phase_limit_matches_ctf_proxy",
        ok,
        f"max_abs_error={error:.3e} rel_error={rel_error:.3e}",
    )


def compare_abtem_reference_fixture(
    fixture_path: Path = ABTEM_REFERENCE_PATH,
    *,
    exit_relative_tolerance: float = 1.0e-2,
    image_relative_tolerance: float = 5.0e-2,
    image_correlation_min: float = 0.75,
    wavelength_relative_tolerance: float = 1.0e-6,
) -> tuple[bool, str, dict[str, object]]:
    fixture_path = Path(fixture_path)
    if not fixture_path.exists():
        return (
            False,
            f"missing abTEM fixture: {fixture_path}",
            {"fixture_path": str(fixture_path), "generate_command": "python validation/run_abtem_reference.py"},
        )

    with np.load(fixture_path) as fixture:
        pot_slices = np.asarray(fixture["pot_slices"], dtype=float)
        reference_exit = np.asarray(fixture["exit_wave"], dtype=np.complex128)
        reference_image = np.asarray(fixture["image"], dtype=float)
        sigma_rad_per_eV_A = float(fixture["sigma_rad_per_eV_A"])
        wavelength_A = float(fixture["wavelength_A"])
        energy_eV = float(fixture["energy_eV"])
        sampling_A = float(fixture["sampling_A"])
        slice_thickness_A = float(fixture["slice_thickness_A"])
        defocus_A = float(fixture["defocus_A"])
        Cs_A = float(fixture["Cs_A"])

    phase_slices = pot_slices * sigma_rad_per_eV_A
    params = _params(
        image_size_pixels=int(pot_slices.shape[-1]),
        pixel_size_nm=sampling_A * 0.1,
        psf_oversampling_factor=1,
        tem_multislice_slices=int(pot_slices.shape[0]),
        tem_slice_thickness_nm=slice_thickness_A * 0.1,
        tem_acceleration_kV=energy_eV / 1000.0,
        tem_Cs_mm=Cs_A * 1.0e-7,
        tem_defocus_nm=defocus_A * 0.1,
        tem_objective_aperture_mrad=None,
    )
    model = TransmissionElectronMicroscopyImagingModel(params)
    backend = model._tem_high_fidelity_backend
    if backend is None:
        raise RuntimeError("TEM multislice physical backend was not initialized.")

    observed_exit = backend.exit_wave_from_projected_phase(phase_slices)
    observed_image = backend.intensity_from_projected_phase(phase_slices)
    aligned_exit = _phase_aligned(observed_exit, reference_exit)

    wavelength_observed_A = float(backend.lambda_m * 1.0e10)
    wavelength_relative_error = abs(wavelength_observed_A / wavelength_A - 1.0)
    exit_relative_l2 = _relative_l2(aligned_exit, reference_exit)
    exit_amplitude_relative_l2 = _relative_l2(np.abs(observed_exit), np.abs(reference_exit))
    image_relative_l2 = _relative_l2(observed_image, reference_image)
    image_correlation = _pearson_correlation(observed_image, reference_image)
    image_mean_relative_error = abs(float(observed_image.mean()) / float(reference_image.mean()) - 1.0)

    metrics: dict[str, object] = {
        "fixture_path": str(fixture_path),
        "pot_slices_shape": list(map(int, pot_slices.shape)),
        "energy_eV": energy_eV,
        "sampling_A": sampling_A,
        "slice_thickness_A": slice_thickness_A,
        "defocus_A": defocus_A,
        "Cs_A": Cs_A,
        "wavelength_reference_A": wavelength_A,
        "wavelength_observed_A": wavelength_observed_A,
        "wavelength_relative_error": wavelength_relative_error,
        "exit_wave_relative_l2": exit_relative_l2,
        "exit_amplitude_relative_l2": exit_amplitude_relative_l2,
        "image_relative_l2": image_relative_l2,
        "image_pearson_correlation": image_correlation,
        "image_mean_relative_error": image_mean_relative_error,
        "exit_relative_tolerance": exit_relative_tolerance,
        "image_relative_tolerance": image_relative_tolerance,
        "image_correlation_min": image_correlation_min,
        "wavelength_relative_tolerance": wavelength_relative_tolerance,
    }
    ok = (
        wavelength_relative_error <= wavelength_relative_tolerance
        and exit_relative_l2 <= exit_relative_tolerance
        and image_relative_l2 <= image_relative_tolerance
        and image_correlation >= image_correlation_min
    )
    detail = (
        "abTEM fixture comparison "
        f"exit_rel={exit_relative_l2:.3e}, image_rel={image_relative_l2:.3e}, "
        f"image_corr={image_correlation:.3f}"
    )
    return ok, detail, metrics


def check_optional_abtem(
    require_abtem: bool,
    *,
    fixture_path: Path = ABTEM_REFERENCE_PATH,
    exit_relative_tolerance: float = 1.0e-2,
    image_relative_tolerance: float = 5.0e-2,
    image_correlation_min: float = 0.75,
) -> bool:
    fixture_path = Path(fixture_path)
    if not fixture_path.exists() and not require_abtem:
        return _pass_fail(
            "abtem_reference_fixture",
            True,
            f"not requested and fixture is absent: {fixture_path}",
        )
    ok, detail, metrics = compare_abtem_reference_fixture(
        fixture_path,
        exit_relative_tolerance=exit_relative_tolerance,
        image_relative_tolerance=image_relative_tolerance,
        image_correlation_min=image_correlation_min,
    )
    metric_text = (
        f"{detail}; wavelength_rel={float(metrics.get('wavelength_relative_error', float('nan'))):.3e}"
    )
    return _pass_fail("abtem_reference_fixture", ok, metric_text)


def run_validation(
    *,
    tolerance: float,
    weak_relative_tolerance: float,
    weak_absolute_tolerance: float,
    require_abtem: bool,
    abtem_reference_path: Path = ABTEM_REFERENCE_PATH,
    abtem_exit_relative_tolerance: float = 1.0e-2,
    abtem_image_relative_tolerance: float = 5.0e-2,
    abtem_image_correlation_min: float = 0.75,
) -> bool:
    checks = [
        check_free_space(tolerance),
        check_slab_phase(tolerance),
        check_weak_phase_ctf(weak_relative_tolerance, weak_absolute_tolerance),
        check_optional_abtem(
            require_abtem,
            fixture_path=abtem_reference_path,
            exit_relative_tolerance=abtem_exit_relative_tolerance,
            image_relative_tolerance=abtem_image_relative_tolerance,
            image_correlation_min=abtem_image_correlation_min,
        ),
    ]
    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--weak-relative-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--weak-absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--require-abtem", action="store_true", help="Require the generated abTEM fixture.")
    parser.add_argument("--abtem-reference", type=Path, default=ABTEM_REFERENCE_PATH)
    parser.add_argument("--abtem-exit-relative-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--abtem-image-relative-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--abtem-image-correlation-min", type=float, default=0.75)
    args = parser.parse_args()
    ok = run_validation(
        tolerance=args.tolerance,
        weak_relative_tolerance=args.weak_relative_tolerance,
        weak_absolute_tolerance=args.weak_absolute_tolerance,
        require_abtem=args.require_abtem,
        abtem_reference_path=args.abtem_reference,
        abtem_exit_relative_tolerance=args.abtem_exit_relative_tolerance,
        abtem_image_relative_tolerance=args.abtem_image_relative_tolerance,
        abtem_image_correlation_min=args.abtem_image_correlation_min,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
