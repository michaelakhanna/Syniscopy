"""Parameter schema public API assembled from category fragments."""

from __future__ import annotations

from ._spec import ParamSpec, ParamType
from .optics import OPTICS_SCHEMA
from .electron import ELECTRON_SCHEMA
from .particle import PARTICLE_SCHEMA
from .sample_environment import SAMPLE_ENVIRONMENT_SCHEMA
from .noise import NOISE_SCHEMA
from .supervision import SUPERVISION_SCHEMA
from .dataset import DATASET_SCHEMA
from .fisher_contracts import FISHER_CONTRACT_SCHEMA


def _merge_schema() -> dict[str, ParamSpec]:
    items = [
    *OPTICS_SCHEMA.items(),
    *ELECTRON_SCHEMA.items(),
    *PARTICLE_SCHEMA.items(),
    *SAMPLE_ENVIRONMENT_SCHEMA.items(),
    *NOISE_SCHEMA.items(),
    *SUPERVISION_SCHEMA.items(),
    *DATASET_SCHEMA.items(),
    *FISHER_CONTRACT_SCHEMA.items(),
    ]
    schema: dict[str, ParamSpec] = {}
    for key, spec in items:
        if key in schema:
            raise RuntimeError(f"Duplicate parameter schema key: {key}")
        schema[key] = spec
    return schema


PARAM_SCHEMA: dict[str, ParamSpec] = _merge_schema()

__all__ = [
    "PARAM_SCHEMA",
    "ParamSpec",
    "ParamType",
]
