"""Parameter schema public API assembled from category fragments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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
PUBLIC_PARAM_KEYS = frozenset(
    key
    for key, spec in PARAM_SCHEMA.items()
    if "container_key" not in spec
)


def default_param_value(key: str) -> Any:
    """Return a fresh copy of the concept-owned default for one public key."""
    try:
        spec = PARAM_SCHEMA[str(key)]
    except KeyError as exc:
        raise KeyError(f"Unknown public parameter key: {key!r}.") from exc
    if "container_key" in spec:
        raise KeyError(
            f"{key!r} is a projected control, not a top-level public parameter."
        )
    if "default" not in spec:
        raise KeyError(f"Public parameter {key!r} has no concept-owned default.")
    return deepcopy(spec["default"])


def default_params() -> dict[str, Any]:
    """Assemble a fresh complete public parameter mapping from concept defaults."""
    return {
        key: default_param_value(key)
        for key, spec in PARAM_SCHEMA.items()
        if "container_key" not in spec
    }

__all__ = [
    "PARAM_SCHEMA",
    "PUBLIC_PARAM_KEYS",
    "ParamSpec",
    "ParamType",
    "default_param_value",
    "default_params",
]
