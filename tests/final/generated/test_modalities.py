from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


_REP_MODALITIES = (
    "bright_field",
    "fluorescence_widefield",
    "dark_field",
)


def _base_params() -> dict:
    from config import default_params, default_param_value

    params = default_params()
    params.update(
        {
            "imaging_model": "bright_field",
            "image_size_pixels": 16,
            "pixel_size_nm": 50.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "max_psf_z_slices": 64,
            "num_frames": 1,
            "duration_seconds": 1.0 / 24.0,
            "random_seed": 101,
            "shot_noise_enabled": False,
            "gaussian_noise_enabled": False,
            "background_subtraction_method": "reference_frame",
            "mask_generation_enabled": False,
            "save_frame_sequence": False,
            "return_ideal_float_frames": True,
            "save_raw_camera_frame_sequence": False,
            "return_fps": False,
            "simulated_camera_gain_override": None,
            "fluorescence_backend": "parametric_psf",
        }
    )
    params["particles"] = [deepcopy(default_param_value("particles")[0])]
    return params


def _single_modality_params(modality: str) -> dict:
    params = _base_params()
    params.update(
        {
            "imaging_model": modality,
        }
    )
    return params


def test_supported_modalities_render_finite_and_contract_keys() -> None:
    from imaging_models import get_imaging_model
    from rendering import resolve_render_canvas_geometry

    from simulation import generate_single_frame_views

    for modality in _REP_MODALITIES:
        params = _single_modality_params(modality)
        model = get_imaging_model(params)

        geometry = resolve_render_canvas_geometry(params, imaging_model=model)
        output_size = int(geometry["detector_image_size_pixels"])
        response_size = int(geometry["model_canvas_size_pixels"])

        view = generate_single_frame_views(deepcopy(params))
        assert isinstance(view, dict), modality
        contrast = np.asarray(view["contrast_frame"], dtype=float)
        assert contrast.shape == (output_size, output_size), modality
        assert np.all(np.isfinite(contrast)), modality

        response = model.compute_response_function((response_size, response_size), params)
        assert isinstance(response, dict), modality
        assert {"kind", "output_type", "probe_wavelength_nm"}.issubset(response), modality
        assert response["output_type"] in {"intensity", "phase", "complex_field"}, modality

        response2 = model.compute_response_function((response_size, response_size), params)
        assert response["kind"] == response2["kind"], modality


def test_modalities_are_deterministic_given_seed() -> None:
    from simulation import generate_single_frame_views

    for modality in _REP_MODALITIES:
        first = generate_single_frame_views(_single_modality_params(modality))
        second = generate_single_frame_views(_single_modality_params(modality))

        c1 = np.asarray(first["contrast_frame"], dtype=float)
        c2 = np.asarray(second["contrast_frame"], dtype=float)
        assert c1.shape == c2.shape
        assert np.allclose(c1, c2), modality


def test_runtime_owner_settings_are_stable_on_repeated_construction() -> None:
    from config.runtime import CountBudgetSettings, DetectorSettings

    params = _base_params()
    one = deepcopy(params)
    two = deepcopy(one)

    budget_1 = CountBudgetSettings.from_params(one)
    budget_2 = CountBudgetSettings.from_params(two)
    detector_1 = DetectorSettings.from_params(one)
    detector_2 = DetectorSettings.from_params(two)
    assert budget_1 == budget_2
    assert detector_1.detector_qe == pytest.approx(detector_2.detector_qe)


def test_matched_microscope_packet_rejects_missing_coordinate_frame_and_accepts_valid_frame() -> None:
    from fisher import compute_fisher_information
    from matched_microscope_packets import build_matched_microscope_packet
    from noise_contracts import independent_pixel_noise_model
    from simulation import generate_single_frame_views

    images = {
        modality: np.asarray(
            generate_single_frame_views(_single_modality_params(modality))["contrast_frame"],
            dtype=float,
        )
        for modality in _REP_MODALITIES
    }
    fishers = {
        modality: compute_fisher_information(
            image,
            independent_pixel_noise_model(1.0),
            pixel_size_nm=float(_single_modality_params(modality)["pixel_size_nm"]),
        )
        for modality, image in images.items()
    }

    with pytest.raises(ValueError, match="shared_coordinate_frame"):
        build_matched_microscope_packet(
            latent_state={"frame_index": 0},
            images_by_microscope=images,
            modality_by_microscope={name: name for name in images},
            fisher_by_microscope=fishers,
            metadata={},
        )

    packet = build_matched_microscope_packet(
        latent_state={"frame_index": 0},
        images_by_microscope=images,
        modality_by_microscope={name: name for name in images},
        fisher_by_microscope=fishers,
        metadata={
            "shared_coordinate_frame": {
                "axes": ["x_nm", "y_nm"],
                "pixel_size_nm": 50.0,
                "fisher_frame": "shared_xy_detector_frame",
            }
        },
    )

    assert set(packet["images_by_microscope"]) == set(_REP_MODALITIES)
    assert packet["metadata"]["metadata"]["shared_coordinate_frame"]["axes"] == ["x_nm", "y_nm"]
    for fisher in packet["fisher_by_microscope"].values():
        arr = np.asarray(fisher, dtype=float)
        assert np.all(np.isfinite(arr))
        assert arr.shape == (2, 2)
