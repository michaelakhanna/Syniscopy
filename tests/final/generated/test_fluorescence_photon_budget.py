from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from config import default_params
from imaging_models import FluorescenceWidefieldImagingModel
from imaging_models.fluorescence_tirf import TIRFFluorescenceImagingModel
from particle_specs import ParticleComponentSpec


def _params(**overrides):
    params = default_params()
    params.update(
        {
            "image_size_pixels": 16,
            "imaging_model": "fluorescence_widefield",
            "pixel_size_nm": 50.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "fluorescence_backend": "parametric_psf",
            "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": 10.0,
            "fluorescence_excitation_scale": 1.0,
            "fluorescence_quantum_yield": 1.0,
            "fluorescence_collection_efficiency": 1.0,
            "fluorescence_detector_qe": 1.0,
            "fluorescence_background": 0.0,
            "fluorescence_emission_psf_sigma_px": 0.0,
            "fluorescence_emission_psf_sigma_nm": None,
            "fluorescence_spectral_bandwidth_nm": 40.0,
            "fluorescence_bleaching_rate_per_frame": 0.0,
            "fluorescence_allow_psf_fallback": True,
        }
    )
    params.update(overrides)
    return params


def _single_source_map(scale: float = 1.0) -> np.ndarray:
    source = np.zeros((16, 16), dtype=float)
    source[8, 8] = float(scale)
    return source


def _particle_signal_sum(
    model_cls,
    params: dict,
    source: np.ndarray | None = None,
) -> float:
    source_map = _single_source_map() if source is None else np.asarray(source, dtype=float)
    model = model_cls(params)
    product = model.compute_particle_signal_product_from_source_map(
        source_map,
        np.zeros_like(source_map),
        params,
        frame_index=0,
    )
    return float(np.sum(product.values))


def _sphere_component(diameter_nm: float = 200.0) -> ParticleComponentSpec:
    return ParticleComponentSpec(
        shape="sphere",
        offset_nm=(0.0, 0.0, 0.0),
        diameter_nm=float(diameter_nm),
    )


def _physical_count_scale(params: dict) -> float:
    return float(
        params["fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"]
        * params["fluorescence_excitation_scale"]
        * params["fluorescence_quantum_yield"]
        * params["fluorescence_collection_efficiency"]
        * params["fluorescence_detector_qe"]
    )


def _tirf_source_map(params: dict, z_nm: float) -> np.ndarray:
    model = TIRFFluorescenceImagingModel(params)
    source = np.zeros((16, 16), dtype=float)
    material = SimpleNamespace(
        fluorophore_density=1.0,
        excitation_peak_nm=None,
        emission_peak_nm=None,
    )
    model.accumulate_particle_source(
        source,
        center_x_canvas=8,
        center_y_canvas=8,
        diameter_nm=200.0,
        pixel_size_nm=float(params["pixel_size_nm"]),
        os_factor=int(params["psf_oversampling_factor"]),
        material_properties=material,
        params=params,
        particle_z_nm=z_nm,
        component_geometry=_sphere_component(200.0),
    )
    return source


def test_quantum_yield_zero_removes_particle_signal() -> None:
    base = _particle_signal_sum(FluorescenceWidefieldImagingModel, _params(fluorescence_quantum_yield=1.0))
    zero = _particle_signal_sum(FluorescenceWidefieldImagingModel, _params(fluorescence_quantum_yield=0.0))

    assert base > 0.0
    np.testing.assert_allclose(zero, 0.0)


def test_excitation_and_absorbed_budget_scale_particle_signal() -> None:
    base_params = _params(fluorescence_quantum_yield=0.5, fluorescence_excitation_scale=1.0)
    base_signal = _particle_signal_sum(FluorescenceWidefieldImagingModel, base_params)

    excited_params = _params(
        fluorescence_quantum_yield=0.5,
        fluorescence_excitation_scale=2.0,
    )
    absorbed_params = _params(
        fluorescence_quantum_yield=0.5,
        fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame=20.0,
    )

    np.testing.assert_allclose(
        _particle_signal_sum(FluorescenceWidefieldImagingModel, excited_params),
        2.0 * base_signal,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _particle_signal_sum(FluorescenceWidefieldImagingModel, absorbed_params),
        2.0 * base_signal,
        rtol=1e-12,
    )


def test_collection_and_detector_qe_scale_particle_signal() -> None:
    base = _params(
        fluorescence_quantum_yield=0.5,
        fluorescence_collection_efficiency=1.0,
        fluorescence_detector_qe=1.0,
    )
    base_signal = _particle_signal_sum(FluorescenceWidefieldImagingModel, base)

    collection_half = _params(
        fluorescence_quantum_yield=0.5,
        fluorescence_collection_efficiency=0.5,
        fluorescence_detector_qe=1.0,
    )
    qe_half = _params(
        fluorescence_quantum_yield=0.5,
        fluorescence_collection_efficiency=1.0,
        fluorescence_detector_qe=0.5,
    )

    np.testing.assert_allclose(
        _particle_signal_sum(FluorescenceWidefieldImagingModel, collection_half),
        0.5 * base_signal,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _particle_signal_sum(FluorescenceWidefieldImagingModel, qe_half),
        0.5 * base_signal,
        rtol=1e-12,
    )


def test_source_sum_follows_area_scaled_count_formula() -> None:
    params = _params(
        fluorescence_quantum_yield=0.8,
        fluorescence_collection_efficiency=0.75,
        fluorescence_detector_qe=0.9,
    )
    source = _single_source_map(4.0)
    observed = _particle_signal_sum(FluorescenceWidefieldImagingModel, params, source)
    expected = float(np.sum(source)) * float(params["pixel_size_nm"]) ** 2 * _physical_count_scale(params)

    np.testing.assert_allclose(observed, expected, rtol=1e-9)


def test_vectorial_and_parametric_backends_share_budget_contract() -> None:
    base = _params(
        fluorescence_background=0.0,
        fluorescence_quantum_yield=0.5,
        fluorescence_collection_efficiency=0.8,
    )
    vectorial = _params(
        fluorescence_background=0.0,
        fluorescence_quantum_yield=0.5,
        fluorescence_collection_efficiency=0.8,
        fluorescence_backend="vectorial_photophysics",
    )
    source = _single_source_map()

    base_response = FluorescenceWidefieldImagingModel(base).compute_response_function((16, 16), base)
    vectorial_response = FluorescenceWidefieldImagingModel(vectorial).compute_response_function(
        (16, 16),
        vectorial,
    )

    assert (
        base_response["fluorescence_photon_budget_source"]
        == vectorial_response["fluorescence_photon_budget_source"]
        == "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"
    )
    assert (
        base_response["fluorescence_photon_budget_semantics"]
        == vectorial_response["fluorescence_photon_budget_semantics"]
        == "absorbed_excitation_photons_before_quantum_yield"
    )

    base_signal = _particle_signal_sum(FluorescenceWidefieldImagingModel, base, source)
    vectorial_signal = _particle_signal_sum(FluorescenceWidefieldImagingModel, vectorial, source)
    np.testing.assert_allclose(base_signal, vectorial_signal, rtol=1e-9)


def test_tirf_depth_penetration_weights_source_integral_and_counts() -> None:
    params = _params(
        imaging_model="tirf_fluorescence",
        tirf_source_representation="projected_2d",
        tirf_use_angle_derived_penetration_depth=False,
        tirf_penetration_depth_nm=50.0,
        tirf_height_offset_nm=0.0,
        fluorescence_background=0.0,
    )
    near = _tirf_source_map(params, z_nm=0.0)
    far = _tirf_source_map(params, z_nm=100.0)

    near_sum = float(np.sum(near))
    far_sum = float(np.sum(far))
    assert near_sum > far_sum

    near_signal = _particle_signal_sum(TIRFFluorescenceImagingModel, params, near)
    far_signal = _particle_signal_sum(TIRFFluorescenceImagingModel, params, far)

    assert near_signal > far_signal

    expected_near = near_sum * float(params["pixel_size_nm"]) ** 2 * _physical_count_scale(params)
    expected_far = far_sum * float(params["pixel_size_nm"]) ** 2 * _physical_count_scale(params)
    np.testing.assert_allclose(near_signal, expected_near, rtol=1e-9)
    np.testing.assert_allclose(far_signal, expected_far, rtol=1e-9)


def test_fluorescence_response_metadata_contracts_and_legacy_key_cleanup() -> None:
    response = FluorescenceWidefieldImagingModel(_params()).compute_response_function((16, 16), _params())

    assert response["fluorescence_photon_budget_source"] == (
        "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"
    )
    assert response["fluorescence_photon_budget_semantics"] == (
        "absorbed_excitation_photons_before_quantum_yield"
    )

    for removed in (
        "fluorescence_photon_count_scale",
        "fluorescence_require_physical_photon_budget",
        "legacy_proxy",
        "legacy_emitted",
        "emitted_photon_scale",
    ):
        assert removed not in response
