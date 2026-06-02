from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_shared_constants_match_runtime_users() -> None:
    import camera_noise
    import config
    import counterfactual_packets
    import dataset_generator
    import particle_specs
    import postprocessing
    from shared_constants import (
        KNOWN_INTERNAL_PARAM_KEYS,
        MATCHED_INFORMATION_MASK_ROLES,
        NONNEGATIVE_MATERIAL_PROPERTY_FIELDS,
        NUM_FRAME_DURATION_SEARCH_STEPS,
        RAW_BACKGROUND_SUBTRACTION_METHODS,
    )

    assert set(config.KNOWN_INTERNAL_PARAM_KEYS) == set(KNOWN_INTERNAL_PARAM_KEYS)
    assert isinstance(NUM_FRAME_DURATION_SEARCH_STEPS, int)
    assert NUM_FRAME_DURATION_SEARCH_STEPS > 0
    assert particle_specs._NONNEGATIVE_MATERIAL_FIELDS is NONNEGATIVE_MATERIAL_PROPERTY_FIELDS
    assert counterfactual_packets._MATCHED_INFORMATION_MASK_ROLES == MATCHED_INFORMATION_MASK_ROLES
    assert "raw" in postprocessing.RAW_BACKGROUND_SUBTRACTION_METHODS
    assert "raw" in RAW_BACKGROUND_SUBTRACTION_METHODS
    assert camera_noise.RAW_BACKGROUND_SUBTRACTION_METHODS is RAW_BACKGROUND_SUBTRACTION_METHODS


def test_json_safe_contract_serializes_shared_cases() -> None:
    from json_utils import json_safe, json_safe_with_nonfinite_tags

    class Status(Enum):
        OK = "ok"

    @dataclass(frozen=True)
    class Payload:
        status: Status
        z: complex
        values: object

    payload = Payload(
        status=Status.OK,
        z=1.0 + 2.0j,
        values=np.array([1.0, np.nan, np.inf, -np.inf]),
    )
    strict = json_safe(payload)
    tagged = json_safe_with_nonfinite_tags(payload)

    assert strict == {
        "status": "ok",
        "z": {"real": 1.0, "imag": 2.0},
        "values": [1.0, None, None, None],
    }
    assert tagged["values"][1:] == [
        {"nonfinite": "nan"},
        {"nonfinite": "posinf"},
        {"nonfinite": "neginf"},
    ]
    json.dumps(strict, allow_nan=False)
    json.dumps(tagged, allow_nan=False)


def test_manifest_modules_do_not_redefine_json_safe_helpers() -> None:
    codebase = CODEBASE
    forbidden = tuple(
        "def " + name
        for name in (
            "_json_safe",
            "_json_ready",
            "_strict_json_safe",
            "_packet_jsonable",
        )
    )
    offenders = []
    for path in sorted(codebase.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                offenders.append(f"{path.name}:{marker}")

    assert offenders == []


def test_acquisition_cost_public_api_separates_contract_and_lookup() -> None:
    import acquisition_costs
    import experiment_contracts

    assert acquisition_costs.AcquisitionCostModel is experiment_contracts.AcquisitionCostModel
    model = acquisition_costs.AcquisitionCostLookup.default()
    row = model.cost_for_modality("bright_field")
    assert row["schema_version"] == acquisition_costs.ACQUISITION_COST_SCHEMA_VERSION
    assert acquisition_costs.cost_for_modality("bright_field")["schema_version"] == row["schema_version"]


def test_model_card_prefers_backend_fidelity_convergence_status() -> None:
    from experiment_contracts import model_card_from_profile_card

    card = model_card_from_profile_card(
        {
            "canonical_modality_name": "bright_field",
            "display_name": "Bright field",
            "convergence_status": "failed_convergence",
            "backend_fidelity_metadata": {
                "convergence_status": "finite_converged",
                "validation_status": "validated",
            },
            "active_parameters": {},
        }
    )

    assert card["convergence_status"] == "finite_converged"
    assert card["validation_status"] == "validated"


def test_mie_helpers_are_import_light_and_reexported() -> None:
    import mie_scattering
    import optics

    a_n, b_n = mie_scattering.mie_an_bn(1.2 + 0.01j, 0.75)
    assert len(a_n) == len(b_n)
    assert len(a_n) > 0
    assert optics.mie_an_bn is mie_scattering.mie_an_bn

    s1, s2 = mie_scattering.mie_S1_S2_from_coefficients(a_n, b_n, np.array([0.0, 0.5]))
    assert np.asarray(s1).shape == (2,)
    assert np.asarray(s2).shape == (2,)


def test_dataset_append_resume_remaps_composition_assignments() -> None:
    from dataset.orchestrator import _remap_leaf_assignments_to_target_indices

    local_assignments = {
        0: {"video_index": 0, "leaf_spec": {"name": "leaf_a"}},
        1: {"video_index": 1, "leaf_spec": {"name": "leaf_b"}},
    }

    remapped, remapped_list = _remap_leaf_assignments_to_target_indices(
        local_assignments,
        list(local_assignments.values()),
        [10, 11],
    )

    assert sorted(remapped) == [10, 11]
    assert remapped[10]["leaf_spec"]["name"] == "leaf_a"
    assert remapped[11]["leaf_spec"]["name"] == "leaf_b"
    assert [item["video_index"] for item in remapped_list] == [10, 11]


def test_dataset_append_resume_recomputes_target_span_when_request_count_changes() -> None:
    from dataset.orchestrator import _remap_leaf_assignments_to_target_indices

    local_assignments = {
        0: {"video_index": 0, "leaf_spec": {"name": "leaf_a"}},
        1: {"video_index": 1, "leaf_spec": {"name": "leaf_b"}},
        2: {"video_index": 2, "leaf_spec": {"name": "leaf_c"}},
    }
    prior_target_indices = [10, 11]
    requested_indices = list(range(3))
    target_indices = [
        min(prior_target_indices) + request_index
        for request_index in requested_indices
    ]

    remapped, remapped_list = _remap_leaf_assignments_to_target_indices(
        local_assignments,
        list(local_assignments.values()),
        target_indices,
    )

    assert target_indices == [10, 11, 12]
    assert sorted(remapped) == [10, 11, 12]
    assert remapped[12]["leaf_spec"]["name"] == "leaf_c"
    assert [item["video_index"] for item in remapped_list] == [10, 11, 12]


def test_particle_schema_declares_nested_target_paths() -> None:
    import param_schema

    diameter = param_schema.PARAM_SCHEMA["particle_diameter_nm"]
    material = param_schema.PARAM_SCHEMA["particle_material"]

    schema_keys = [
        str(spec["key"])
        for spec in param_schema.PARAM_SCHEMA.values()
    ]
    assert len(schema_keys) == len(set(schema_keys))

    assert diameter["key"] == "particle_diameter_nm"
    assert diameter["container_key"] == "particles"
    assert diameter["target_path"] == "particles[0].components[0].diameter_nm"
    assert material["key"] == "particle_material"
    assert material["container_key"] == "particles"
    assert material["target_path"] == "particles[0].components[0].material"


def test_contract_truth_flags_reject_malformed_values() -> None:
    from param_utils import _coerce_contract_truthy_flag

    assert _coerce_contract_truthy_flag(True) is True
    assert _coerce_contract_truthy_flag("false") is False
    assert _coerce_contract_truthy_flag(1) is True
    assert _coerce_contract_truthy_flag(0.0) is False

    for value in ("maybe", "enabled", 2, 0.5, object()):
        try:
            _coerce_contract_truthy_flag(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for malformed flag {value!r}.")
