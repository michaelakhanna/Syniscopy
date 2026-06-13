from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


pytestmark = pytest.mark.quick


class _FluorMaterial:
    fluorophore_density = 1.0
    excitation_peak_nm = None
    emission_peak_nm = None


def _normalized_params(**updates):
    from config import PARAMS, normalize_params

    params = deepcopy(PARAMS)
    params.update(updates)
    return normalize_params(params)


def _fluorescence_probe_params(**updates):
    base = {
        "image_size_pixels": 33,
        "pixel_size_nm": 20.0,
        "psf_oversampling_factor": 1,
        "vectorial_pupil_samples": 64,
        "fluorescence_backend": "parametric_psf",
        "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": 1.0,
        "fluorescence_collection_efficiency": 1.0,
        "fluorescence_detector_qe": 1.0,
        "fluorescence_quantum_yield": 1.0,
        "fluorescence_excitation_scale": 1.0,
        "fluorescence_background": 0.0,
        "shot_noise_enabled": False,
        "gaussian_noise_enabled": False,
    }
    base.update(updates)
    return _normalized_params(**base)


def _single_particle_source_sum(params: dict) -> float:
    from imaging_models import get_imaging_model

    model = get_imaging_model(params)
    shape = (int(params["image_size_pixels"]), int(params["image_size_pixels"]))
    source = model.initialize_particle_source_canvas(shape, params)
    model.accumulate_particle_source(
        source,
        center_x_canvas=shape[1] // 2,
        center_y_canvas=shape[0] // 2,
        diameter_nm=100.0,
        pixel_size_nm=float(params["pixel_size_nm"]),
        os_factor=int(params["psf_oversampling_factor"]),
        material_properties=_FluorMaterial(),
        params=params,
        particle_z_nm=0.0,
    )
    return float(np.sum(source))


def test_widefield_volume_source_conserves_projected_source_signal() -> None:
    projected = _single_particle_source_sum(
        _fluorescence_probe_params(
            imaging_model="fluorescence_widefield",
            fluorescence_source_representation="projected_2d",
        )
    )
    volume = _single_particle_source_sum(
        _fluorescence_probe_params(
            imaging_model="fluorescence_widefield",
            fluorescence_source_representation="volume",
            fluorescence_volume_slices=21,
            fluorescence_volume_slice_thickness_nm=10.0,
            initial_z_span_nm=200.0,
        )
    )

    assert projected > 0.0
    assert np.isclose(volume, projected, rtol=1.0e-12, atol=1.0e-12)


def test_count_budget_resolvers_keep_optical_sem_and_tem_budgets_separate() -> None:
    from config.runtime import CountBudgetSettings, SemSettings, TemSettings, resolved_detector_qe

    params = _normalized_params(
        background_intensity=11.0,
        sem_electrons_per_pixel=222.0,
        tem_dose_per_pixel=333.0,
        detector_qe=0.25,
        fluorescence_detector_qe=None,
    )
    counts = CountBudgetSettings.from_params(params)

    assert counts.background_intensity == 11.0
    assert counts.sem_electrons_per_pixel == 222.0
    assert counts.tem_dose_per_pixel == 333.0
    assert SemSettings.from_params(params).electrons_per_pixel == 222.0
    assert TemSettings.from_params(params).dose_per_pixel == 333.0
    assert resolved_detector_qe(params) == 0.25
    assert resolved_detector_qe(params, fluorescence=True) == 0.25


def test_raw_fusion_sum_cancels_opposite_cross_terms_before_inversion() -> None:
    from fisher import (
        FisherMatrixCandidate,
        compute_candidate_fusion_crlb_from_fisher_matrices,
    )

    fisher_a = np.asarray([[1.0, 0.9], [0.9, 1.0]], dtype=float)
    fisher_b = np.asarray([[1.0, -0.9], [-0.9, 1.0]], dtype=float)
    result = compute_candidate_fusion_crlb_from_fisher_matrices(
        [
            FisherMatrixCandidate(
                key="positive_cross_term",
                fisher_matrix=fisher_a,
            ),
            FisherMatrixCandidate(
                key="negative_cross_term",
                fisher_matrix=fisher_b,
            ),
        ]
    )

    expected_joint = fisher_a + fisher_b
    expected_covariance = np.linalg.inv(expected_joint)
    expected_sigma_xy = float(
        np.sqrt(expected_covariance[0, 0] + expected_covariance[1, 1])
    )

    assert np.allclose(result["fusion_fisher"], expected_joint)
    assert np.isclose(result["fusion_sigma_xy_nm"], expected_sigma_xy)
    assert result["fusion_sigma_xy_nm"] < result["best_single_sigma_xy_nm"]


def test_dynamic_bayesian_filter_reduces_to_static_sum_when_process_noise_vanishes() -> None:
    from fisher.dynamic_bayesian import (
        compute_dynamic_bayesian_crlb,
        sequence_sum_fisher_to_crlb,
    )

    per_frame = [10.0 * np.eye(2, dtype=float) for _ in range(3)]
    _static_fisher, static_covariance, _static_rank = sequence_sum_fisher_to_crlb(per_frame)
    dynamic = compute_dynamic_bayesian_crlb(
        per_frame,
        process_noise_covariance=1.0e-12 * np.eye(2, dtype=float),
        initial_covariance=1.0e12 * np.eye(2, dtype=float),
    )

    static_diag = np.asarray([np.diag(cov) for cov in static_covariance], dtype=float)
    dynamic_diag = np.asarray(dynamic.dynamic_crlb, dtype=float)
    assert np.allclose(dynamic_diag, static_diag, rtol=1.0e-9, atol=1.0e-9)


def test_dynamic_bayesian_filter_reverts_to_per_frame_bound_with_huge_process_noise() -> None:
    from fisher.dynamic_bayesian import compute_dynamic_bayesian_crlb

    per_frame = [10.0 * np.eye(2, dtype=float) for _ in range(3)]
    dynamic = compute_dynamic_bayesian_crlb(
        per_frame,
        process_noise_covariance=1.0e12 * np.eye(2, dtype=float),
        initial_covariance=1.0e12 * np.eye(2, dtype=float),
    )

    per_frame_variance = np.asarray([0.1, 0.1], dtype=float)
    for diag in dynamic.dynamic_crlb:
        assert np.allclose(diag, per_frame_variance, rtol=1.0e-9, atol=1.0e-9)
