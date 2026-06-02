from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_source_map_base_scene_intensity_requires_override_even_without_maps() -> None:
    from imaging_models.base import ImagingModel

    class MissingSourceSceneModel(ImagingModel):
        uses_particle_material_sources = True
        requires_optical_scattered_field = False

        def compute_intensity(self, E_sca_total, background_field, params):
            return np.ones_like(background_field, dtype=float)

        def compute_per_particle_contrast(self, E_sca_particle, background_field, params):
            return np.zeros_like(background_field, dtype=float)

    model = MissingSourceSceneModel()

    try:
        model.compute_scene_intensity(
            [],
            [],
            np.zeros((2, 2)),
            np.ones((2, 2)),
            {},
            particle_source_maps=None,
        )
    except NotImplementedError as exc:
        assert "source-map-aware compute_scene_intensity" in str(exc)
    else:
        raise AssertionError("Expected source-map models to override compute_scene_intensity.")


def test_source_map_base_particle_contrast_requires_source_map_override() -> None:
    from imaging_models.base import ImagingModel

    class MissingSourceParticleModel(ImagingModel):
        uses_particle_material_sources = True
        requires_optical_scattered_field = True

        def compute_intensity(self, E_sca_total, background_field, params):
            return np.ones_like(background_field, dtype=float)

        def compute_per_particle_contrast(self, E_sca_particle, background_field, params):
            return np.zeros_like(background_field, dtype=float)

    model = MissingSourceParticleModel()

    try:
        model.compute_particle_contrast(np.zeros((2, 2)), np.ones((2, 2)), {})
    except RuntimeError as exc:
        assert "compute_particle_contrast_from_source_map" in str(exc)
    else:
        raise AssertionError(
            "Expected source-map models to override particle contrast handling."
        )


def test_native_calibration_params_carry_vectorial_pupil_samples() -> None:
    from calibration_profiles import native_params

    params = native_params(
        {
            "modality": "fluorescence_widefield",
            "pupil_samples": 64,
        }
    )
    assert params["pupil_samples"] == 64
    assert params["vectorial_pupil_samples"] == 64

    explicit = native_params(
        {
            "modality": "fluorescence_widefield",
            "pupil_samples": 64,
            "vectorial_pupil_samples": 96,
        }
    )
    assert explicit["pupil_samples"] == 64
    assert explicit["vectorial_pupil_samples"] == 96


def test_tem_response_kind_tracks_backend_model() -> None:
    from imaging_models.tem import TransmissionElectronMicroscopyImagingModel

    base_params = {
        "tem_slice_thickness_nm": 1.0,
        "tem_multislice_slices": 2,
        "tem_dose_per_pixel": 10.0,
        "pixel_size_nm": 2.0,
        "psf_oversampling_factor": 1,
        "tem_acceleration_kV": 300.0,
    }

    ctf = TransmissionElectronMicroscopyImagingModel(
        dict(base_params, tem_backend="ctf_proxy")
    ).compute_response_function((4, 4), base_params)
    lite = TransmissionElectronMicroscopyImagingModel(
        dict(base_params, tem_backend="multislice_lite")
    ).compute_response_function((4, 4), base_params)
    syniscopy = TransmissionElectronMicroscopyImagingModel(
        dict(base_params, tem_backend="syniscopy_multislice")
    ).compute_response_function((4, 4), base_params)

    assert ctf["kind"] == "tem_ctf"
    assert lite["kind"] == "tem_multislice_lite"
    assert syniscopy["kind"] == "tem_multislice"


def test_calibration_partial_rows_are_not_validation_comparisons() -> None:
    import calibration_profiles

    case = dict(calibration_profiles.CALIBRATION_PROFILES["dark_field"])
    assert case["classification"] == "DIRECT_QUOTED_LOCALIZATION_PRECISION"
    assert case["parameter_match_status"] == "partial"

    parameter_match_status = str(case.get("parameter_match_status", "partial")).lower()
    classification = str(case["classification"]).upper()
    source_has_localization_scale = (
        classification in calibration_profiles.LOCALIZATION_SCALE_CLASSIFICATIONS
    )
    is_parameter_matched_localization = bool(
        source_has_localization_scale and parameter_match_status == "yes"
    )

    assert source_has_localization_scale is True
    assert is_parameter_matched_localization is False
