from __future__ import annotations

import sys
from pathlib import Path


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_pairwise_compatibility_flags_shared_falsy_ids() -> None:
    from modality_compatibility import compatibility_status

    status = compatibility_status(
        "bright_field",
        "quantitative_phase",
        {"same_source_quanta_id": 0, "detector_channel_id": 0},
        {"same_source_quanta_id": 0, "detector_channel_id": 0},
    )

    assert status["double_count_risk"] is True
    assert status["same_quanta_reconstruction_risk"] is True
    assert status["independent_noise_assumption"] is False
    assert status["compatible_only_as_algebraic_diagnostic"] is True
    assert status["required_review"] is True
    assert "same quanta/detector channel" in status["reason"]


def test_fluorescence_independent_budget_accepts_zero_detector_id() -> None:
    from modality_compatibility import compatibility_status

    status = compatibility_status(
        "fluorescence_widefield",
        "tirf_fluorescence",
        {"independent_excitation_budget": True, "detector_channel_id": 0},
        {"independent_excitation_budget": True, "detector_channel_id": 1},
    )

    assert "fluorescence pair needs independent" not in status["reason"]
    assert status["double_count_risk"] is False


def test_fluorescence_budget_not_independent_when_detector_id_missing() -> None:
    from modality_compatibility import compatibility_status

    status = compatibility_status(
        "fluorescence_widefield",
        "tirf_fluorescence",
        {"independent_excitation_budget": True},
        {"independent_excitation_budget": True, "detector_channel_id": 0},
    )

    assert "fluorescence pair needs independent" in status["reason"]
    assert status["required_review"] is True
