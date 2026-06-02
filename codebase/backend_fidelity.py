"""Central backend-fidelity metadata helpers.

This module defines canonical fidelity levels and a single helper that enforces
the machine-readable backend metadata contract for each imaging-model result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from experiment_contracts import (
    ConvergenceStatus,
    ValidationStatus,
    normalize_convergence_status,
    normalize_validation_status,
    stable_hash,
)


class BackendFidelityLevel(str, Enum):
    """Canonical fidelity levels used by response metadata."""

    UNKNOWN = "unknown"
    PROXY = "proxy"
    PHYSICS_BASED = "physics_based"
    HIGH_FIDELITY = "high_fidelity"
    REFERENCE_VALIDATED = "reference_validated"


def normalize_backend_fidelity_level(value: Any) -> str:
    """Normalize heterogeneous legacy values to :class:`BackendFidelityLevel`."""

    if isinstance(value, BackendFidelityLevel):
        raw = value.value
    else:
        raw = "proxy" if value is None else str(value).strip().lower().replace(" ", "_")
    if raw in {"physics_based_unvalidated", "physics_based", "physics-based", "physical"}:
        return BackendFidelityLevel.PHYSICS_BASED.value
    if raw in {
        "high_fidelity",
        "high-fidelity",
        "highfidelity",
        "multislice",
        "multislice_lite",
    }:
        return BackendFidelityLevel.HIGH_FIDELITY.value
    if raw in {"reference_validated", "validated", "reference-valid"}:
        return BackendFidelityLevel.REFERENCE_VALIDATED.value
    if raw == "proxy":
        return BackendFidelityLevel.PROXY.value
    return BackendFidelityLevel.UNKNOWN.value


@dataclass(frozen=True)
class BackendResultMetadata:
    backend_fidelity_level: str
    backend_name: str
    equations_or_model_family: str
    implemented_approximation_level: str
    native_operating_assumptions: str
    reference_backend_metadata: Mapping[str, Any] | None = None
    validation_status: str = ValidationStatus.UNCHECKED.value
    convergence_status: str = ConvergenceStatus.UNCHECKED.value
    comparison_contract_id: str = "Contract-NR"
    artifact_provenance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_nonempty(*parts: Any) -> str | None:
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            return text
    return None


def extract_backend_fidelity_metadata(
    response: Mapping[str, Any] | None, *, backend_contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Extract a canonical backend fidelity row from any response-like payload."""

    payload = dict(response or {})
    contract = dict(backend_contract or {})
    fidelity_row = BackendResultMetadata(
        backend_fidelity_level=normalize_backend_fidelity_level(
            payload.get(
                "backend_fidelity_level",
                contract.get("backend_fidelity_level", "proxy"),
            )
        ),
        backend_name=_first_nonempty(
            payload.get("backend_name"),
            payload.get("backend_family"),
            payload.get("backend_family_id"),
            contract.get("backend_family"),
            payload.get("kind"),
            "imaging-model",
        )
        or "imaging-model",
        equations_or_model_family=_first_nonempty(
            payload.get("equations_or_model_family"),
            payload.get("forward_observable"),
            payload.get("kind"),
            payload.get("kind", "imaging-model"),
            "imaging-model",
        )
        or "imaging-model",
        implemented_approximation_level=_first_nonempty(
            payload.get("implemented_approximation_level"),
            payload.get("fidelity_label"),
            payload.get("fidelity_class"),
            contract.get("fidelity_class"),
            payload.get("fidelity_tag"),
            "proxy_model",
        )
        or "proxy_model",
        native_operating_assumptions=_first_nonempty(
            payload.get("native_operating_assumptions"),
            payload.get("native_units"),
            contract.get("native_units"),
            "not_declared",
        )
        or "not_declared",
        reference_backend_metadata=payload.get("reference_backend_metadata"),
        validation_status=normalize_validation_status(
            payload.get("validation_status")
            if payload.get("validation_status") is not None
            else contract.get("validation_status")
        ),
        convergence_status=normalize_convergence_status(
            payload.get("convergence_status")
            if payload.get("convergence_status") is not None
            else contract.get("convergence_status")
        ),
        comparison_contract_id=_first_nonempty(
            payload.get("comparison_contract_id"),
            contract.get("comparison_contract_id"),
            "Contract-NR",
        )
        or "Contract-NR",
        artifact_provenance_id=payload.get("artifact_provenance_id"),
    ).to_dict()

    fidelity_row["known_omissions"] = payload.get(
        "known_omissions",
        contract.get("known_omissions", ()),
    )

    if fidelity_row["artifact_provenance_id"] is None:
        fidelity_row["artifact_provenance_id"] = stable_hash(
            {
                "backend_name": fidelity_row["backend_name"],
                "backend_fidelity_level": fidelity_row["backend_fidelity_level"],
                "equations_or_model_family": fidelity_row["equations_or_model_family"],
                "implemented_approximation_level": fidelity_row["implemented_approximation_level"],
                "comparison_contract_id": fidelity_row["comparison_contract_id"],
                "native_operating_assumptions": fidelity_row["native_operating_assumptions"],
            }
        )

    return fidelity_row


def attach_backend_fidelity_metadata(
    response: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None = None,
    backend_name: str | None = None,
    equations_or_model_family: str | None = None,
    implemented_approximation_level: str | None = None,
    native_operating_assumptions: str | None = None,
    comparison_contract_id: str | None = None,
    artifact_provenance_id: str | None = None,
) -> dict[str, Any]:
    """Return a response metadata payload with enforceable fidelity keys."""

    payload: dict[str, Any] = dict(response or {})

    raw_level = payload.get("backend_fidelity_level")
    if params is not None and "backend_fidelity_level" in params:
        raw_level = params["backend_fidelity_level"]
    fidelity = normalize_backend_fidelity_level(raw_level)

    metadata = BackendResultMetadata(
        backend_fidelity_level=fidelity,
        backend_name=_first_nonempty(
            backend_name,
            payload.get("backend_name"),
            payload.get("tem_backend"),
            payload.get("sem_backend"),
            payload.get("fluorescence_backend"),
            payload.get("optical_field_backend"),
            payload.get("kind"),
        )
        or "unknown-backend",
        equations_or_model_family=_first_nonempty(
            equations_or_model_family,
            payload.get("equations_or_model_family"),
            payload.get("forward_observable"),
            payload.get("kind"),
            "imaging-model",
        )
        or "imaging-model",
        implemented_approximation_level=_first_nonempty(
            implemented_approximation_level,
            payload.get("implemented_approximation_level"),
            payload.get("fidelity_label"),
            payload.get("fidelity_class"),
            "proxy_model",
        )
        or "proxy_model",
        native_operating_assumptions=_first_nonempty(
            native_operating_assumptions,
            payload.get("native_operating_assumptions"),
            payload.get("scalar_vectorial_backend"),
            payload.get("optical_field_backend"),
            payload.get("native_units"),
            "not_declared",
        )
        or "not_declared",
        reference_backend_metadata=payload.get("reference_backend_metadata"),
        validation_status=normalize_validation_status(
            payload.get("validation_status")
            if payload.get("validation_status") is not None
            else params.get("validation_status")
            if params is not None
            else None
        ),
        convergence_status=normalize_convergence_status(
            payload.get("convergence_status")
            if payload.get("convergence_status") is not None
            else params.get("convergence_status")
            if params is not None
            else None
        ),
        comparison_contract_id=_first_nonempty(
            comparison_contract_id,
            payload.get("comparison_contract_id"),
            params.get("comparison_contract_id") if params is not None else None,
            "Contract-NR",
        )
        or "Contract-NR",
        artifact_provenance_id=artifact_provenance_id,
    )

    enriched = metadata.to_dict()

    if enriched["artifact_provenance_id"] is None:
        enriched["artifact_provenance_id"] = stable_hash(
            {
                "backend_name": enriched["backend_name"],
                "backend_fidelity_level": enriched["backend_fidelity_level"],
                "equations_or_model_family": enriched["equations_or_model_family"],
                "implemented_approximation_level": enriched[
                    "implemented_approximation_level"
                ],
                "comparison_contract_id": enriched["comparison_contract_id"],
                "native_operating_assumptions": enriched["native_operating_assumptions"],
            }
        )

    overwritten: dict[str, Any] = {}
    for key, value in enriched.items():
        if key in payload and payload[key] != value:
            overwritten[key] = payload[key]
        payload[key] = value
    if overwritten:
        existing = payload.get("raw_backend_fidelity_metadata")
        raw_metadata = dict(existing) if isinstance(existing, Mapping) else {}
        raw_metadata.update(overwritten)
        payload["raw_backend_fidelity_metadata"] = raw_metadata

    return payload


__all__ = [
    "BackendFidelityLevel",
    "BackendResultMetadata",
    "extract_backend_fidelity_metadata",
    "attach_backend_fidelity_metadata",
    "normalize_backend_fidelity_level",
]
