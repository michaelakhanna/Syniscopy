from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _candidate_set() -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray]]]:
    strong = np.array([[1.7, 0.1], [0.1, 1.5]], dtype=float)
    weak = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    other = np.array([[0.3, 0.0], [0.0, 4.0]], dtype=float)
    candidates = [
        ("strong", strong),
        ("weak", weak),
        ("other", other),
    ]
    return {"strong": strong, "weak": weak, "other": other}, candidates


def test_strong_psd_dominates_weak_and_updates_maximal_set() -> None:
    from fisher.candidates import FisherMatrixCandidate
    from fisher.time_allocation import compute_loewner_dominance

    _, candidate_pairs = _candidate_set()
    candidates = [FisherMatrixCandidate(name, matrix) for name, matrix in candidate_pairs]

    result = compute_loewner_dominance(candidates)

    assert "dominates" in result
    assert "dominated_by" in result
    assert "loewner_maximal_candidates" in result

    assert "weak" in result["dominates"].get("strong", [])
    assert "strong" in result["dominated_by"].get("weak", [])
    assert "weak" not in set(result["loewner_maximal_candidates"])
    assert "strong" in set(result["loewner_maximal_candidates"])

    eig_min = result["dominance_eigenvalue_min"]
    assert eig_min["strong"]["weak"] >= -1e-12


def test_incomparable_candidate_remains_maximal() -> None:
    from fisher.candidates import FisherMatrixCandidate
    from fisher.time_allocation import compute_loewner_dominance

    _, candidate_pairs = _candidate_set()
    candidates = [FisherMatrixCandidate(name, matrix) for name, matrix in candidate_pairs]
    result = compute_loewner_dominance(candidates)

    maximal = set(result["loewner_maximal_candidates"])
    assert "other" in maximal
    assert "strong" in maximal
    assert "weak" not in maximal

    strong_other = result["dominance_eigenvalue_min"].get("strong", {}).get("other")
    assert strong_other is not None and strong_other < 0.0

    other_strong = result["dominance_eigenvalue_min"].get("other", {}).get("strong")
    assert other_strong is not None and other_strong < 0.0


def test_dominance_diagnostics_stable_under_repeated_calls() -> None:
    from fisher.candidates import FisherMatrixCandidate
    from fisher.time_allocation import compute_loewner_dominance

    matrices, candidate_pairs = _candidate_set()
    candidates = [FisherMatrixCandidate(name, matrix) for name, matrix in candidate_pairs]

    first = compute_loewner_dominance(candidates)
    second = compute_loewner_dominance(candidates)

    assert first["dominates"] == second["dominates"]
    assert first["dominated_by"] == second["dominated_by"]
    assert first["loewner_maximal_candidates"] == second["loewner_maximal_candidates"]

    # candidate_information_rates are plain arrays; compare deterministically per key.
    assert set(first["candidate_information_rates"].keys()) == set(matrices.keys())
    for name in matrices:
        np.testing.assert_allclose(
            np.asarray(first["candidate_information_rates"][name], dtype=float),
            np.asarray(second["candidate_information_rates"][name], dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
