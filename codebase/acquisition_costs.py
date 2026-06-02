"""Public acquisition-cost model API for Syniscopy time-allocation diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiment_contracts import AcquisitionCostModel
from experiment_contracts import acquisition_cost_from_profile
from modality_registry import SUPPORTED_MODALITIES, canonical_modality_name

ACQUISITION_COST_SCHEMA_VERSION = "syniscopy-acquisition-cost-v1"


def _validated_cost_modality(modality: str) -> str:
    canonical = canonical_modality_name(modality)
    if canonical not in SUPPORTED_MODALITIES:
        supported = ", ".join(SUPPORTED_MODALITIES)
        raise ValueError(
            f"Unknown acquisition-cost modality {modality!r}. "
            f"Supported modalities are: {supported}."
        )
    return canonical


def _with_schema(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("schema_version", ACQUISITION_COST_SCHEMA_VERSION)
    return out


@dataclass(frozen=True)
class AcquisitionCostLookup:
    """Small public wrapper around per-modality acquisition-cost metadata."""

    default_params: Mapping[str, Any] | None = None

    @classmethod
    def default(cls) -> "AcquisitionCostLookup":
        return cls(default_params={})

    def cost_for_modality(self, modality: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.default_params or {})
        if params:
            merged.update(dict(params))
        return _with_schema(acquisition_cost_from_profile(_validated_cost_modality(modality), merged).to_dict())


def cost_for_modality(modality: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the declared acquisition-cost model for one modality."""
    return _with_schema(acquisition_cost_from_profile(_validated_cost_modality(modality), params or {}).to_dict())


def contract_cost_model(**kwargs: Any) -> AcquisitionCostModel:
    """Construct the lower-level contract dataclass when direct control is needed."""
    return AcquisitionCostModel(**kwargs)


__all__ = [
    "ACQUISITION_COST_SCHEMA_VERSION",
    "AcquisitionCostLookup",
    "AcquisitionCostModel",
    "cost_for_modality",
    "contract_cost_model",
]
