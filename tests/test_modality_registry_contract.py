from __future__ import annotations

import sys
from pathlib import Path


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_modality_registry_matches_backend_registry() -> None:
    import imaging_models
    import modality_registry

    assert imaging_models.SUPPORTED_MODALITIES == modality_registry.SUPPORTED_MODALITIES
    assert set(imaging_models._MODEL_REGISTRY) == set(modality_registry.SUPPORTED_MODALITIES)
    assert imaging_models.MODALITY_ALIASES is modality_registry.MODALITY_ALIASES
    assert (
        imaging_models.RELATIVE_REFERENCE_CONTRAST_MODALITIES
        == modality_registry.RELATIVE_REFERENCE_CONTRAST_MODALITIES
    )
    assert not hasattr(imaging_models, "canonical_modality_name")
    assert not hasattr(imaging_models, "modality_display_name")


def test_aliases_are_shared_by_validation_noise_and_models() -> None:
    from camera_noise import _resolved_detector_noise_input_domain
    from config import normalize_params
    from imaging_models import get_imaging_model_class
    from modality_registry import canonical_modality_name

    assert canonical_modality_name("brightfield") == "bright_field"
    assert canonical_modality_name("qpi") == "quantitative_phase"
    assert canonical_modality_name("tem") == "tem_phase_contrast"
    assert canonical_modality_name("sem") == "sem_secondary_electron"

    params = normalize_params({"imaging_model": "tem"})
    assert params["imaging_model"] == "tem"
    assert get_imaging_model_class(params["imaging_model"]).__name__ == (
        "TransmissionElectronMicroscopyImagingModel"
    )
    assert _resolved_detector_noise_input_domain({"imaging_model": "sem"}) == "electron_count"


def test_param_schema_declares_all_runtime_config_keys() -> None:
    import config
    import param_schema
    from modality_registry import SUPPORTED_MODALITIES

    declared_runtime_keys = {
        str(spec.get("container_key", spec.get("key", schema_key)))
        for schema_key, spec in param_schema.PARAM_SCHEMA.items()
    }

    assert set(config.PARAMS) <= declared_runtime_keys
    assert param_schema.PARAM_SCHEMA["imaging_model"]["choices"] == list(SUPPORTED_MODALITIES)
