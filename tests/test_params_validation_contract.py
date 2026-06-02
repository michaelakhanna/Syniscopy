from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_validate_params_does_not_mutate_dpc_normalization_inputs() -> None:
    from config import normalize_params, validate_params

    params = {
        "imaging_model": "differential_phase_contrast",
        "dpc_channel_model": "vectorial",
    }
    before = deepcopy(params)

    validate_params(params)

    assert params == before
    normalized = normalize_params(params)
    assert normalized is not params
    assert "optical_field_backend" not in params
    assert "vectorial_detection_mode" not in params
    assert normalized["optical_field_backend"] == "vectorial_debye"
    assert normalized["vectorial_detection_mode"] == "full_vector"


def test_validate_params_does_not_mutate_pattern_preset_inputs() -> None:
    from config import normalize_params, validate_params

    params = {
        "sample_environment_enabled": True,
        "sample_environment_pattern_enabled": True,
        "sample_environment_pattern": "nanopillars",
        "sample_environment_pattern_preset": "default",
    }
    before = deepcopy(params)

    validate_params(params)

    assert params == before
    normalized = normalize_params(params)
    assert normalized is not params
    assert params["sample_environment_pattern_preset"] == "default"
    assert normalized["sample_environment_pattern_preset"] == "default_nanopillars"
