from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import builtins
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for substrate-dependent bootstrap; missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _base_params(modality: str) -> dict:
    from config import default_params

    params = deepcopy(default_params())
    params.update(
        {
            "imaging_model": modality,
            "image_size_pixels": 16,
            "pixel_size_nm": 100.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 32,
            "vectorial_pupil_samples": 32,
            "z_stack_range_nm": 400.0,
            "z_stack_step_nm": 100.0,
            "max_psf_z_slices": 16,
            "random_seed": 44,
        }
    )

    if modality == "sem_secondary_electron":
        params.update(
            {
                "sem_model": "gaussian_probe_secondary_yield",
                "sem_backend": "gaussian_probe_proxy",
                "sem_source_representation": "projected",
            }
        )

    if modality == "tem_phase_contrast":
        params.update(
            {
                "tem_model": "weak_phase_ctf",
                "tem_backend": "ctf_proxy",
            }
        )

    return params


def _response(modality: str, overrides: dict | None = None) -> dict:
    _require_cv2_for_bootstrap()
    from imaging_models import get_imaging_model

    params = _base_params(modality)
    if overrides:
        params.update(overrides)

    model = get_imaging_model(params)
    return model.compute_response_function((16, 16), params)


def _assert_scalar_equal(left: object, right: object) -> None:
    if left is None and right is None:
        return
    if left is None or right is None:
        raise AssertionError(f"Missing value mismatch: {left!r} != {right!r}")

    is_scalar_numeric = isinstance(left, (int, float, np.integer, np.floating))
    is_scalar_numeric &= isinstance(right, (int, float, np.integer, np.floating))
    if is_scalar_numeric:
        assert np.isclose(float(left), float(right), rtol=1.0e-12, atol=1.0e-12)
        return

    assert left == right


def _assert_fields_match(reference: dict, candidate: dict, keys: list[str]) -> None:
    for key in keys:
        assert key in reference
        assert key in candidate
        _assert_scalar_equal(reference[key], candidate[key])


def test_electron_modalities_keep_electron_domain_metadata_under_optical_overrides() -> None:
    from modality_registry import is_electron_modality

    optical_change = {
        "wavelength_nm": 488.0,
        "probe_wavelength_nm": 532.0,
        "numerical_aperture": 0.65,
        "reference_field_amplitude": 3.0,
    }

    sem_a = _response("sem_secondary_electron")
    sem_b = _response("sem_secondary_electron", optical_change)

    sem_keys = [
        "measurement_domain",
        "signal_units",
        "final_measurement_domain",
        "final_signal_units",
        "count_scaling_mode",
        "acceleration_kV",
        "electrons_per_pixel",
    ]

    assert is_electron_modality("sem_secondary_electron")
    assert sem_a["measurement_domain"] == "electron_count"
    _assert_fields_match(sem_a, sem_b, sem_keys)

    tem_a = _response("tem_phase_contrast")
    tem_b = _response("tem_phase_contrast", optical_change)

    tem_keys = [
        "measurement_domain",
        "signal_units",
        "final_measurement_domain",
        "final_signal_units",
        "count_scaling_mode",
        "acceleration_kV",
        "dose_per_pixel",
        "electron_wavelength_pm",
    ]

    assert is_electron_modality("tem_phase_contrast")
    assert tem_a["measurement_domain"] == "electron_count"
    _assert_fields_match(tem_a, tem_b, tem_keys)


def test_optical_modality_metadata_is_stable_to_electron_only_overrides() -> None:
    from modality_registry import is_electron_modality

    electron_change = {
        "sem_acceleration_kV": 30.0,
        "sem_electrons_per_pixel": 1.0e6,
        "tem_acceleration_kV": 80.0,
        "tem_dose_per_pixel": 1.0e6,
    }

    optical_a = _response("bright_field")
    optical_b = _response("bright_field", electron_change)

    optical_keys = [
        "measurement_domain",
        "signal_units",
        "kind",
        "probe_wavelength_nm",
        "output_type",
    ]

    assert not is_electron_modality("bright_field")
    assert optical_a["measurement_domain"] != "electron_count"
    _assert_fields_match(optical_a, optical_b, optical_keys)
