from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _ordering_by_budget() -> dict[float, dict]:
    return {
        1.0: {"ordering_xy": [("A", 1.0), ("B", 2.0), ("C", 3.0)]},
        4.0: {"ordering_xy": [("A", 0.5), ("B", 1.4), ("C", 2.5)]},
    }


def test_ordering_invariant_is_true_when_candidate_ranking_stable() -> None:
    from fisher.detected_quanta import check_budget_ordering_invariance

    result = check_budget_ordering_invariance(_ordering_by_budget())

    assert result["ordering_invariant"] is True
    assert dict(result["ordering_by_budget"]) == {
        1.0: ["A", "B", "C"],
        4.0: ["A", "B", "C"],
    }
    assert result["readout_limited_candidates"] == []


def test_ordering_invariant_becomes_false_and_tracks_readout_limited_flags() -> None:
    from fisher.detected_quanta import check_budget_ordering_invariance

    data = {
        1.0: {
            "ordering_xy": [("A", 1.0), ("B", 2.0), ("C", 3.0)],
            "count_readout_limited": {"B": False, "C": False, "A": False},
        },
        4.0: {
            "ordering_xy": [("B", 0.6), ("A", 0.9), ("C", 1.4)],
            "count_readout_limited": {"B": False, "C": False, "A": False},
        },
        9.0: {
            "ordering_xy": [("A", 0.8), ("C", 1.2), ("B", 1.9)],
            "count_readout_limited": {"A": False, "B": True, "C": False},
        },
    }

    result = check_budget_ordering_invariance(data)

    assert result["ordering_invariant"] is False
    assert dict(result["ordering_by_budget"]) == {
        1.0: ["A", "B", "C"],
        4.0: ["B", "A", "C"],
        9.0: ["A", "C", "B"],
    }
    assert "B" in set(result["readout_limited_candidates"])


def test_result_is_deterministic_under_identical_inputs() -> None:
    from fisher.detected_quanta import check_budget_ordering_invariance

    base = _ordering_by_budget()
    first = check_budget_ordering_invariance(base)
    second = check_budget_ordering_invariance(base)

    assert first["ordering_by_budget"] == second["ordering_by_budget"]
