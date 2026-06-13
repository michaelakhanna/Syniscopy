from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.quick


def _gaussian(size: int, sigma_px: float, *, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=float)
    centre = 0.5 * (size - 1.0)
    dx = xx - centre - float(x0)
    dy = yy - centre - float(y0)
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_px * sigma_px))


def _template_context(*candidates: str) -> dict[str, dict[str, str]]:
    return {
        candidate: {"stationary_template_provenance": "test_single_rigid_template"}
        for candidate in candidates
    }


def test_zero_signal_erases_fisher_information() -> None:
    from fisher import compute_fisher_information, compute_localization_crlb

    zero = np.zeros((21, 21), dtype=float)
    F = compute_fisher_information(zero, 5.0, pixel_size_nm=100.0)
    crlb = compute_localization_crlb(zero, 5.0, pixel_size_nm=100.0)

    assert np.allclose(F, 0.0)
    assert crlb["singular"] is True
    assert crlb["sigma_x_nm"] == float("inf")
    assert crlb["sigma_y_nm"] == float("inf")


def test_fusion_adds_raw_fim_before_any_inverse() -> None:
    from fisher import (
        FisherMatrixCandidate,
        compute_candidate_fusion_crlb_from_fisher_matrices,
    )

    F_a = np.asarray([[10.0, 0.0], [0.0, 0.0]], dtype=float)
    F_b = np.asarray([[0.0, 0.0], [0.0, 5.0]], dtype=float)
    result = compute_candidate_fusion_crlb_from_fisher_matrices(
        [
            FisherMatrixCandidate(key="A", fisher_matrix=F_a),
            FisherMatrixCandidate(key="B", fisher_matrix=F_b),
        ]
    )

    expected_sum = F_a + F_b
    expected_cov = np.linalg.inv(expected_sum)
    expected_sigma_xy = float(np.sqrt(expected_cov[0, 0] + expected_cov[1, 1]))

    assert np.allclose(result["fusion_fisher"], expected_sum)
    assert result["fusion_singular"] is False
    assert np.isclose(result["fusion_sigma_xy_nm"], expected_sigma_xy)
    assert result["registration_adjusted_per_candidate_crlb"]["A"]["singular"] is True
    assert result["registration_adjusted_per_candidate_crlb"]["B"]["singular"] is True


def test_rendered_signal_fusion_consumes_candidate_objects() -> None:
    from fisher import (
        ArrayOnlyFisherDerivativeContext,
        FisherCandidate,
        compute_fisher_candidate_fusion_crlb,
    )

    yy, xx = np.indices((21, 21), dtype=float)
    template = np.exp(-((xx - 10.0) ** 2 + (yy - 10.0) ** 2) / (2.0 * 2.0 ** 2))
    derivative_context = ArrayOnlyFisherDerivativeContext.single_rigid_template(
        "test_single_rigid_template"
    )
    candidates = [
        FisherCandidate(
            key="A",
            signal=template,
            noise_variance=np.ones_like(template),
            modality="dark_field",
            pixel_size_nm=10.0,
            noise_covariance_kind="independent_pixels",
            derivative_context=derivative_context,
        ),
        FisherCandidate(
            key="B",
            signal=np.roll(template, 1, axis=0),
            noise_variance=np.ones_like(template),
            modality="dark_field",
            pixel_size_nm=10.0,
            noise_covariance_kind="independent_pixels",
            derivative_context=derivative_context,
        ),
    ]

    result = compute_fisher_candidate_fusion_crlb(candidates)

    assert result["candidate_keys"] == ["A", "B"]
    assert result["candidates_used"] == ["A", "B"]
    assert np.isfinite(result["fusion_sigma_xy_nm"])
    assert result["fusion_sigma_xy_nm"] <= result["best_single_sigma_xy_nm"]


def test_detected_quanta_budget_is_strictly_equalized() -> None:
    from fisher.detected_quanta import (
        DetectedQuantaCandidate,
        compare_detected_quanta_normalized_fisher_candidates,
    )

    count_a = 3.0 * _gaussian(31, 3.0) + 0.01
    count_b = 0.7 * _gaussian(31, 7.0) + 0.20
    contrast_a = count_a - float(count_a.mean())
    contrast_b = count_b - float(count_b.mean())
    budget = 5000.0

    derivative_context = _template_context("fluorescence_like", "coherent_like")
    result = compare_detected_quanta_normalized_fisher_candidates(
        [
            DetectedQuantaCandidate(
                key="fluorescence_like",
                contrast=contrast_a,
                modality="fluorescence_widefield",
                pixel_size_nm=100.0,
                detected_count_image=count_a,
                derivative_context=derivative_context["fluorescence_like"],
            ),
            DetectedQuantaCandidate(
                key="coherent_like",
                contrast=contrast_b,
                modality="bright_field",
                pixel_size_nm=100.0,
                detected_count_image=count_b,
                reference_count_image=count_b,
                derivative_context=derivative_context["coherent_like"],
            ),
        ],
        budget,
    )

    for candidate in ("fluorescence_like", "coherent_like"):
        scale = float(result["quanta_scale_by_candidate"][candidate])
        basis = float(result["budgeted_count_sum_by_candidate"][candidate])
        assert np.isclose(scale * basis, budget, rtol=1.0e-12)

    contract = result["detected_quanta_contract"]
    assert contract["total_detected_quanta_budget"] == budget
    assert result["all_count_domain_candidates_have_detected_count_images"] is True
    assert result["contract_q_proxy_diagnostic"] is False


def test_strict_detected_quanta_ranking_refuses_proxy_count_images() -> None:
    from fisher.detected_quanta import (
        DetectedQuantaCandidate,
        compare_detected_quanta_normalized_fisher_candidates,
    )

    contrast = _gaussian(21, 4.0)
    with pytest.raises(ValueError, match="detected_count_image"):
        compare_detected_quanta_normalized_fisher_candidates(
            [
                DetectedQuantaCandidate(
                    key="A",
                    contrast=contrast,
                    modality="bright_field",
                    pixel_size_nm=100.0,
                    derivative_context=_template_context("A", "B")["A"],
                ),
                DetectedQuantaCandidate(
                    key="B",
                    contrast=2.0 * contrast,
                    modality="dark_field",
                    pixel_size_nm=100.0,
                    derivative_context=_template_context("A", "B")["B"],
                ),
            ],
            1000.0,
        )


def test_readout_background_increases_detected_quanta_crlb() -> None:
    from fisher.detected_quanta import (
        DetectedQuantaCandidate,
        compare_detected_quanta_normalized_fisher_candidates,
    )

    yy, xx = np.indices((41, 41), dtype=float)
    xx = xx - 20.0
    yy = yy - 20.0
    rr = np.sqrt(xx * xx + yy * yy)
    broad = np.exp(-(xx * xx + yy * yy) / (2.0 * 7.0 * 7.0))
    ring = np.exp(-((rr - 8.0) ** 2) / (2.0 * 1.0 * 1.0))

    contrast_a = ring
    count_a = 1.0 + 10.0 * broad
    contrast_b = 0.3 * (xx / 7.0) * broad
    count_b = broad + 0.1

    derivative_context = _template_context("coherent_ring_limited", "fluorescence_broad")
    candidates = [
        DetectedQuantaCandidate(
            key="coherent_ring_limited",
            contrast=contrast_a,
            modality="bright_field",
            pixel_size_nm=100.0,
            detected_count_image=count_a,
            reference_count_image=count_a,
            derivative_context=derivative_context["coherent_ring_limited"],
        ),
        DetectedQuantaCandidate(
            key="fluorescence_broad",
            contrast=contrast_b,
            modality="fluorescence_widefield",
            pixel_size_nm=100.0,
            detected_count_image=count_b,
            derivative_context=derivative_context["fluorescence_broad"],
        ),
    ]
    low = compare_detected_quanta_normalized_fisher_candidates(
        candidates,
        10000.0,
        readout_variance=0.0,
    )
    high = compare_detected_quanta_normalized_fisher_candidates(
        candidates,
        10000.0,
        readout_variance=10000.0,
    )

    for candidate in ("coherent_ring_limited", "fluorescence_broad"):
        assert (
            high["per_candidate"][candidate]["sigma_xy_nm"]
            > low["per_candidate"][candidate]["sigma_xy_nm"]
        )
        assert (
            high["readout_variance_fraction_by_candidate"][candidate]
            > low["readout_variance_fraction_by_candidate"][candidate]
        )
    assert all(high["count_readout_limited"].values())


@pytest.mark.full
@pytest.mark.monte_carlo
def test_empirical_mle_variance_tracks_poisson_crlb() -> None:
    scipy_opt = pytest.importorskip("scipy.optimize")

    rng = np.random.default_rng(1234)
    samples = int(os.environ.get("SYNISCOPY_VERIFY_MONTE_CARLO_SAMPLES", "5000"))
    size = 17
    photons = 2500.0
    background = 2.0
    sigma_px = 1.65
    pixel_nm = 100.0
    true_theta = np.asarray([0.18, -0.22], dtype=float)

    def mean(theta: np.ndarray) -> np.ndarray:
        psf = _gaussian(size, sigma_px, x0=float(theta[0]), y0=float(theta[1]))
        psf = psf / float(np.sum(psf))
        return background + photons * psf

    mu0 = mean(true_theta)
    step = 1.0e-3
    derivs = []
    for axis in range(2):
        d = np.zeros(2, dtype=float)
        d[axis] = step
        derivs.append((mean(true_theta + d) - mean(true_theta - d)) / (2.0 * step * pixel_nm))
    F = np.asarray(
        [[float(np.sum(derivs[i] * derivs[j] / mu0)) for j in range(2)] for i in range(2)],
        dtype=float,
    )
    crb_nm2 = np.diag(np.linalg.inv(F))

    def nll(theta_px: np.ndarray, observed: np.ndarray) -> float:
        mu = np.maximum(mean(theta_px), 1.0e-12)
        return float(np.sum(mu - observed * np.log(mu)))

    estimates_nm = np.empty((samples, 2), dtype=float)
    bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    for idx in range(samples):
        obs = rng.poisson(mu0)
        result = scipy_opt.minimize(
            nll,
            x0=true_theta,
            args=(obs,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1.0e-10},
        )
        estimates_nm[idx] = (result.x - true_theta) * pixel_nm

    mse_nm2 = np.mean(estimates_nm * estimates_nm, axis=0)
    ratio = mse_nm2 / crb_nm2

    assert np.all(ratio > 0.70)
    assert np.all(ratio < 1.60)
