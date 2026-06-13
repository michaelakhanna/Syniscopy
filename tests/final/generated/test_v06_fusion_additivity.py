from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _make_candidates() -> tuple[list[np.ndarray], list["FisherMatrixCandidate"]]:
    from fisher.candidates import FisherMatrixCandidate

    fisher_mats = [
        np.array([[4.0, 0.15], [0.15, 6.0]], dtype=float),
        np.array([[2.2, -0.10], [-0.10, 3.5]], dtype=float),
        np.array([[1.5, 0.03], [0.03, 2.8]], dtype=float),
    ]

    return fisher_mats, [
        FisherMatrixCandidate(f"candidate_{idx}", matrix)
        for idx, matrix in enumerate(fisher_mats)
    ]


def test_fused_fisher_is_sum_of_inputs() -> None:
    from fisher import compute_candidate_fusion_crlb_from_fisher_matrices, sigma_xy_from_fisher

    fisher_mats, candidates = _make_candidates()
    result = compute_candidate_fusion_crlb_from_fisher_matrices(candidates)

    fused = np.asarray(result["fusion_fisher"], dtype=float)
    expected_sum = sum(fisher_mats)

    np.testing.assert_allclose(fused, expected_sum, rtol=1e-10, atol=1e-10)
    assert result["fusion_singular"] is False
    np.testing.assert_allclose(
        result["fusion_sigma_xy_nm"],
        sigma_xy_from_fisher(expected_sum),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result["fusion_rank"] == expected_sum.shape[0]


def test_fused_lateral_sigma_never_worse_than_best_single_candidate() -> None:
    from fisher import compute_candidate_fusion_crlb_from_fisher_matrices, sigma_xy_from_fisher

    fisher_mats, candidates = _make_candidates()
    fused = compute_candidate_fusion_crlb_from_fisher_matrices(candidates)

    single_sigmas = np.array([sigma_xy_from_fisher(F) for F in fisher_mats], dtype=float)
    best_single_sigma = float(np.min(single_sigmas[single_sigmas < float("inf")]))

    fused_sigma = float(fused["fusion_sigma_xy_nm"])
    assert np.isfinite(fused_sigma)
    assert fused_sigma <= best_single_sigma + 1e-12
