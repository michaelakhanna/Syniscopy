from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def test_predicted_se3_rank_matches_continuous_stabilizer_dimension() -> None:
    from fisher.se3 import predict_se3_rank_from_contrast_stabilizer

    expected = {
        0: 6,
        1: 5,
        2: 4,
        3: 3,
    }
    for dim, exp in expected.items():
        result = predict_se3_rank_from_contrast_stabilizer(
            continuous_rotational_stabilizer_dim=dim,
            translation_rank=3,
            rotational_dimension=3,
        )
        assert result["predicted_rotational_rank"] == 3 - dim
        assert result["predicted_se3_rank"] == exp
        assert result["predicted_nullity"] == (3 - 3) + dim
        assert result["continuous_rotational_stabilizer_dim"] == dim


def test_fusion_rank_prediction_uses_intersection_dimension_when_available() -> None:
    from fisher.se3 import predict_fused_se3_rank_from_contrast_stabilizers

    result = predict_fused_se3_rank_from_contrast_stabilizers(
        continuous_rotational_stabilizer_intersection_dim=1,
        per_candidate_stabilizer_dims={"a": 2, "b": 1, "c": 0},
        translation_rank=3,
        rotational_dimension=3,
    )

    assert result["predicted_rotational_rank"] == 2
    assert result["predicted_se3_rank"] == 5
    assert result["continuous_rotational_stabilizer_intersection_dim"] == 1
    assert result["per_candidate_continuous_rotational_stabilizer_dim"] == {"a": 2, "b": 1, "c": 0}
    assert result["contrast_stabilizer_reduced_by_fusion"] is True


def test_rank_prediction_is_deterministic() -> None:
    from fisher.se3 import predict_se3_rank_from_contrast_stabilizer

    first = predict_se3_rank_from_contrast_stabilizer(1, translation_rank=3, rotational_dimension=3)
    second = predict_se3_rank_from_contrast_stabilizer(1, translation_rank=3, rotational_dimension=3)
    assert first == second
    assert all(np.isfinite([first["predicted_rotational_rank"], first["predicted_se3_rank"]]))
