from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import builtins
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


_MODALITY_OVERRIDES = {
    "sem_secondary_electron": {
        "sem_model": "gaussian_probe_secondary_yield",
        "sem_backend": "gaussian_probe_proxy",
        "sem_source_representation": "projected",
    },
    "tem_phase_contrast": {
        "tem_model": "weak_phase_ctf",
        "tem_backend": "ctf_proxy",
    },
    "partially_coherent_bright_field": {
        "partially_coherent_bright_field_backend": "partially_coherent_bright_field_proxy",
    },
    "coherent_bright_field": {
        "coherent_bright_field_backend": "coherent_bright_field_proxy",
    },
    "coherent_dark_field": {
        "coherent_dark_field_backend": "coherent_dark_field_proxy",
    },
    "zernike_phase_contrast": {
        "zernike_phase_contrast_backend": "phase_contrast_proxy",
    },
    "differential_phase_contrast": {
        "differential_phase_contrast_backend": "simple_linear_dpc",
    },
    "quantitative_phase": {
        "quantitative_phase_background_method": "global_mean",
        "quantitative_phase_backend": "simple_phase_proxy",
    },
    "off_axis_holography": {
        "off_axis_holography_backend": "off_axis_holography_proxy",
    },
    "ricm": {
        "ricm_backend": "ricm_proxy",
    },
    "interferometric": {
        "interferometric_backend": "interferometric_psf_proxy",
    },
}


_SUPPORTED_MODALITIES = (
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "coherent_dark_field",
    "dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
    "fluorescence_widefield",
    "tirf_fluorescence",
    "tem_phase_contrast",
    "sem_secondary_electron",
)


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for substrate-dependent bootstrap; missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _base_controls() -> dict:
    return {
        "imaging_model": "bright_field",
        "image_size_pixels": 32,
        "pixel_size_nm": 100.0,
        "psf_oversampling_factor": 1,
        "pupil_samples": 32,
        "vectorial_pupil_samples": 32,
        "shot_noise_enabled": False,
        "gaussian_noise_enabled": False,
        "background_subtraction_method": "reference_frame",
        "fluorescence_backend": "parametric_psf",
        "mask_generation_enabled": False,
        "save_frame_sequence": False,
    }


def _params_for_modality(modality: str) -> dict:
    from param_utils import build_params_from_controls

    params = build_params_from_controls(_base_controls())
    params["imaging_model"] = modality
    params.update(deepcopy(_MODALITY_OVERRIDES.get(modality, {})))
    return params


def _assert_symmetric_psd(matrix: np.ndarray, *, atol: float = 1.0e-12) -> None:
    arr = np.asarray(matrix, dtype=float)
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1]
    assert np.all(np.isfinite(arr))
    assert np.allclose(arr, arr.T, rtol=0.0, atol=atol)
    eig = np.linalg.eigvalsh(0.5 * (arr + arr.T))
    assert float(np.min(eig)) >= -atol


def _spd_matrix(rng: np.random.Generator, dim: int, *, floor: float = 0.05) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    return a.T @ a + floor * np.eye(dim, dtype=float)


def test_all_supported_modalities_emit_json_safe_response_payloads() -> None:
    from json_utils import json_safe
    from imaging_models import get_imaging_model

    _require_cv2_for_bootstrap()
    for modality in _SUPPORTED_MODALITIES:
        params = _params_for_modality(modality)
        model = get_imaging_model(params)
        response = model.compute_response_function((params["image_size_pixels"], params["image_size_pixels"]), params)

        payload = json_safe(response)
        assert isinstance(payload, dict)
        assert {"kind", "output_type", "probe_wavelength_nm"}.issubset(payload)
        assert payload["output_type"] in {"intensity", "phase", "complex_field", "fringe"}


def test_parameter_controls_stable_under_reapplication() -> None:
    from config.runtime import CountBudgetSettings, DetectorSettings, SemSettings, TemSettings
    from param_utils import build_params_from_controls

    controls = {
        "background_intensity": float(10.0 ** np.random.default_rng(7).uniform(-1.0, 3.0)),
        "sem_electrons_per_pixel": float(10.0 ** np.random.default_rng(11).uniform(0.0, 6.0)),
        "tem_dose_per_pixel": float(10.0 ** np.random.default_rng(13).uniform(0.0, 4.0)),
        "detector_qe": float(np.random.default_rng(17).uniform(0.05, 1.0)),
        "fluorescence_detector_qe": None,
        **_base_controls(),
    }

    params = build_params_from_controls(controls)
    params_again = build_params_from_controls(controls)
    assert params == params_again

    count_budget = CountBudgetSettings.from_params(params)
    assert count_budget.background_intensity == pytest.approx(controls["background_intensity"])
    assert count_budget.sem_electrons_per_pixel == pytest.approx(controls["sem_electrons_per_pixel"])
    assert count_budget.tem_dose_per_pixel == pytest.approx(controls["tem_dose_per_pixel"])

    sem = SemSettings.from_params(params)
    tem = TemSettings.from_params(params)
    assert sem.electrons_per_pixel == pytest.approx(controls["sem_electrons_per_pixel"])
    assert tem.dose_per_pixel == pytest.approx(controls["tem_dose_per_pixel"])

    detector = DetectorSettings.from_params(params)
    fluorescence_detector = DetectorSettings.from_params(params, fluorescence=True)
    assert detector.detector_qe == pytest.approx(controls["detector_qe"])
    assert fluorescence_detector.detector_qe == pytest.approx(controls["detector_qe"])


def test_likelihood_fisher_is_invariant_to_derivative_scaling_and_noise_level() -> None:
    from fisher.lateral import compute_likelihood_fisher_information

    rng = np.random.default_rng(8675309)
    for _ in range(4):
        mean = rng.uniform(1.0, 100.0, size=(5, 5))
        derivatives = {
            "x": rng.normal(size=(5, 5)),
            "y": rng.normal(size=(5, 5)),
        }
        variance = rng.uniform(0.1, 10.0, size=(5, 5))

        base = compute_likelihood_fisher_information(
            mean,
            derivatives,
            variance_image=variance,
            fisher_mode="gaussian_fixed_variance",
        )["fisher_matrix"]
        _assert_symmetric_psd(base)

        scale = float(rng.uniform(0.2, 5.0))
        scaled = compute_likelihood_fisher_information(
            mean,
            {axis: scale * value for axis, value in derivatives.items()},
            variance_image=variance,
            fisher_mode="gaussian_fixed_variance",
        )["fisher_matrix"]
        assert np.allclose(scaled, scale * scale * base, rtol=1.0e-10, atol=1.0e-10)

        noise_factor = float(rng.uniform(1.01, 20.0))
        higher_noise = compute_likelihood_fisher_information(
            mean,
            derivatives,
            variance_image=noise_factor * variance,
            fisher_mode="gaussian_fixed_variance",
        )["fisher_matrix"]
        _assert_symmetric_psd(base - higher_noise)


def test_fisher_information_is_rotation_invariant_for_90deg_rotation() -> None:
    from fisher import compute_fisher_information
    from noise_contracts import independent_pixel_noise_model

    rng = np.random.default_rng(42)
    image = rng.normal(size=(17, 17))
    image += 2.0 * rng.normal()
    noise_variance = float(rng.uniform(0.2, 20.0))
    pixel_size_nm = float(rng.uniform(10.0, 250.0))
    noise = independent_pixel_noise_model(noise_variance, noise_variance_units="contrast_squared")

    fisher = compute_fisher_information(image, noise, pixel_size_nm=pixel_size_nm)
    rotated = compute_fisher_information(np.rot90(image), noise, pixel_size_nm=pixel_size_nm)

    _assert_symmetric_psd(fisher)
    _assert_symmetric_psd(rotated)
    assert np.isclose(np.trace(fisher), np.trace(rotated), rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(
        np.linalg.eigvalsh(fisher),
        np.linalg.eigvalsh(rotated),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_fisher_matrix_fusion_is_algebraic_sum_and_improves_xy_precision() -> None:
    from fisher import compute_candidate_fusion_crlb_from_fisher_matrices
    from fisher.candidates import FisherMatrixCandidate

    rng = np.random.default_rng(2026)
    fisher_a = _spd_matrix(rng, 2, floor=0.5)
    fisher_b = _spd_matrix(rng, 2, floor=0.5)

    result = compute_candidate_fusion_crlb_from_fisher_matrices(
        [
            FisherMatrixCandidate(key="A", fisher_matrix=fisher_a),
            FisherMatrixCandidate(key="B", fisher_matrix=fisher_b),
        ]
    )

    expected_joint = fisher_a + fisher_b
    expected_cov = np.linalg.inv(expected_joint)
    expected_sigma_xy = float(np.sqrt(expected_cov[0, 0] + expected_cov[1, 1]))

    assert np.allclose(result["fusion_fisher"], expected_joint, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(result["fusion_sigma_xy_nm"], expected_sigma_xy, rtol=1.0e-12)

    single = [
        float(np.sqrt(np.diag(np.linalg.inv(matrix)).sum())) for matrix in (fisher_a, fisher_b)
    ]
    assert result["fusion_sigma_xy_nm"] <= min(single) + 1.0e-12


def test_detected_quanta_budget_preserves_total_per_candidate() -> None:
    from fisher.detected_quanta import (
        DetectedQuantaCandidate,
        compare_detected_quanta_normalized_fisher_candidates,
    )

    rng = np.random.default_rng(12)
    yy, xx = np.indices((9, 9), dtype=float)
    count_a = rng.uniform(0.1, 10.0, size=(9, 9)) + np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 8.0)
    count_b = rng.uniform(0.1, 10.0, size=(9, 9)) + np.exp(-((xx - 3) ** 2 + (yy - 5) ** 2) / 10.0)

    result = compare_detected_quanta_normalized_fisher_candidates(
        [
            DetectedQuantaCandidate(
                key="A",
                contrast=count_a - float(np.mean(count_a)),
                modality="bright_field",
                pixel_size_nm=80.0,
                detected_count_image=count_a,
                reference_count_image=count_a,
                derivative_context={"stationary_template_provenance": "test_single_rigid_template"},
            ),
            DetectedQuantaCandidate(
                key="B",
                contrast=count_b - float(np.mean(count_b)),
                modality="dark_field",
                pixel_size_nm=80.0,
                detected_count_image=count_b,
                reference_count_image=count_b,
                derivative_context={"stationary_template_provenance": "test_single_rigid_template"},
            ),
        ],
        4.0e4,
    )

    assert result["all_count_domain_candidates_have_detected_count_images"] is True
    native_signal_sum_by_key = {
        "A": float(np.sum(count_a)),
        "B": float(np.sum(count_b)),
    }
    for rec in result["candidate_records"]:
        key = str(rec["candidate_key"])
        assert float(rec["budgeted_count_sum"]) == pytest.approx(
            4.0e4,
            rel=1.0e-12,
            abs=1.0e-9,
        )
        assert float(rec["pre_normalization_signal_count_sum"]) == pytest.approx(
            native_signal_sum_by_key[key],
            rel=1.0e-12,
            abs=1.0e-9,
        )
        assert float(rec["pre_normalization_reference_count_sum"]) == pytest.approx(
            native_signal_sum_by_key[key],
            rel=1.0e-12,
            abs=1.0e-9,
        )
        native_observation_total = (
            float(rec["pre_normalization_signal_count_sum"])
            + float(rec["pre_normalization_reference_count_sum"])
        )
        assert float(rec["quanta_scale"]) * native_observation_total == pytest.approx(
            4.0e4,
            rel=1.0e-12,
            abs=1.0e-9,
        )

    signal_only = compare_detected_quanta_normalized_fisher_candidates(
        [
            DetectedQuantaCandidate(
                key="signal_only_a",
                contrast=count_a - float(np.mean(count_a)),
                modality="dark_field",
                pixel_size_nm=80.0,
                detected_count_image=count_a,
                derivative_context={"stationary_template_provenance": "test_single_rigid_template"},
            ),
            DetectedQuantaCandidate(
                key="signal_only_b",
                contrast=count_b - float(np.mean(count_b)),
                modality="dark_field",
                pixel_size_nm=80.0,
                detected_count_image=count_b,
                derivative_context={"stationary_template_provenance": "test_single_rigid_template"},
            ),
        ],
        4.0e4,
    )
    for rec in signal_only["candidate_records"]:
        key = str(rec["candidate_key"])
        source_key = "A" if key.endswith("_a") else "B"
        assert rec["reference_budget_included"] is False
        assert rec["pre_normalization_reference_count_sum"] is None
        assert float(rec["budgeted_count_sum"]) == pytest.approx(
            4.0e4,
            rel=1.0e-12,
            abs=1.0e-9,
        )
        assert float(rec["quanta_scale"]) * native_signal_sum_by_key[source_key] == pytest.approx(
            4.0e4,
            rel=1.0e-12,
            abs=1.0e-9,
        )


def test_dynamic_bayesian_matches_static_and_memoryless_limits() -> None:
    from fisher.dynamic_bayesian import (
        compute_dynamic_bayesian_crlb_from_fisher_sequence,
        sequence_sum_fisher_to_crlb,
    )

    rng = np.random.default_rng(909)
    per_frame = [_spd_matrix(rng, 2, floor=0.4) for _ in range(3)]

    _, static_covariance, _ = sequence_sum_fisher_to_crlb(per_frame)
    static_diag = np.asarray([np.diag(cov) for cov in static_covariance], dtype=float)

    near_static = compute_dynamic_bayesian_crlb_from_fisher_sequence(
        per_frame,
        process_noise_covariance=1.0e-12 * np.eye(2, dtype=float),
        initial_covariance=1.0e12 * np.eye(2, dtype=float),
        include_fisher_matrices=True,
    )
    dynamic_diag = np.asarray(near_static["dynamic_crlb"], dtype=float)
    assert dynamic_diag.shape == static_diag.shape
    assert np.allclose(dynamic_diag, static_diag, rtol=1.0e-7, atol=1.0e-7)

    memoryless = compute_dynamic_bayesian_crlb_from_fisher_sequence(
        per_frame,
        process_noise_covariance=1.0e12 * np.eye(2, dtype=float),
        initial_covariance=1.0e12 * np.eye(2, dtype=float),
        include_fisher_matrices=True,
    )

    for frame_fisher, frame_crlb in zip(per_frame, memoryless["dynamic_crlb"]):
        expected = np.diag(np.linalg.inv(frame_fisher))
        assert np.allclose(frame_crlb, expected, rtol=1.0e-7, atol=1.0e-7)


def test_matched_microscope_packet_requires_and_retains_shared_coordinate_frame() -> None:
    from fisher import compute_fisher_information
    from matched_microscope_packets import build_matched_microscope_packet
    from noise_contracts import independent_pixel_noise_model

    noise_model = independent_pixel_noise_model(
        1.0,
        noise_variance_units="contrast_squared",
        measurement_domain="contrast",
        signal_units="contrast",
    )

    images_by_microscope = {
        "fluorescence_widefield": np.ones((8, 8), dtype=float),
        "sem_secondary_electron": np.full((8, 8), 2.0, dtype=float),
    }
    fishers_by_microscope = {
        key: compute_fisher_information(image, noise_model, pixel_size_nm=100.0)
        for key, image in images_by_microscope.items()
    }

    with pytest.raises(ValueError, match="shared_coordinate_frame"):
        build_matched_microscope_packet(
            latent_state={"frame_index": 0},
            images_by_microscope=images_by_microscope,
            modality_by_microscope={name: name for name in images_by_microscope},
            fisher_by_microscope=fishers_by_microscope,
            metadata={},
        )

    frame_spec = {
        "axes": ["x_nm", "y_nm"],
        "pixel_size_nm": 100.0,
        "fisher_frame": "shared_xy_detector_frame",
    }
    packet = build_matched_microscope_packet(
        latent_state={"frame_index": 0},
        images_by_microscope=images_by_microscope,
        modality_by_microscope={name: name for name in images_by_microscope},
        fisher_by_microscope=fishers_by_microscope,
        metadata={"shared_coordinate_frame": frame_spec},
    )

    retained = packet["metadata"]["metadata"]["shared_coordinate_frame"]
    assert retained == frame_spec
    assert set(packet["images_by_microscope"]) == set(images_by_microscope)
    assert set(packet["fisher_by_microscope"]) == set(images_by_microscope)
    for fisher in packet["fisher_by_microscope"].values():
        _assert_symmetric_psd(np.asarray(fisher))
