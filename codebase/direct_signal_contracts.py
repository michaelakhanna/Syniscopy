"""Typed direct particle/source-map signal contracts for Fisher-facing APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from array_representation import (
    ArrayRepresentation,
    COORD_DETECTOR_XY,
    COORD_PROJECTED_XY,
    DOMAIN_CAMERA_COUNT,
    DOMAIN_ELECTRON_COUNT,
    DOMAIN_FLUORESCENCE_EMISSION_DENSITY,
    DOMAIN_PHASE,
    DOMAIN_RELATIVE_REFERENCE,
    DOMAIN_SEM_SECONDARY_ELECTRON_YIELD,
    DOMAIN_TEM_PROJECTED_PHASE_CONTRAST,
    STAGE_ANALYSIS_CONTRAST,
    STAGE_DIRECT_SIGNAL,
    STAGE_SOURCE_MAP,
    UNKNOWN_ARRAY_REPRESENTATION,
    VALUE_DELTA,
    VALUE_SOURCE_DENSITY,
)
from modality_registry import require_modality_name


DIRECT_PARTICLE_SIGNAL_CONTRACT_ID = "syniscopy-direct-particle-signal-v1"


def _representation_label(representation: ArrayRepresentation | None) -> str | None:
    if representation is None or representation is UNKNOWN_ARRAY_REPRESENTATION:
        return None
    return representation.semantic_label or representation.value_form


def _direct_representation_safe_for_fisher(representation: ArrayRepresentation) -> bool:
    """Return whether the direct signal descriptor is an analysis/Fisher basis."""

    return bool(
        representation.domain
        in {
            DOMAIN_RELATIVE_REFERENCE,
            DOMAIN_CAMERA_COUNT,
            DOMAIN_ELECTRON_COUNT,
            DOMAIN_PHASE,
        }
        and representation.pipeline_stage in {STAGE_ANALYSIS_CONTRAST, STAGE_DIRECT_SIGNAL}
    )


@dataclass(frozen=True)
class DirectSignalIdentity:
    """Canonical identity/provenance for a direct signal product."""

    modality: str
    producer: str
    model_class: str
    method_name: str
    source_input_kind: str | None = None
    source_z_basis: str | None = None
    source_projection_policy: str | None = None
    extra_provenance: Mapping[str, Any] = field(default_factory=dict)

    def provenance_payload(self, *, frame_index: int) -> dict[str, Any]:
        payload = dict(self.extra_provenance)
        payload.update(
            {
                "frame_index": int(frame_index),
                "direct_signal_identity_modality": self.modality,
                "direct_signal_identity_model_class": self.model_class,
                "direct_signal_identity_method": self.method_name,
            }
        )
        if self.source_input_kind is not None:
            payload["source_input_kind"] = self.source_input_kind
        if self.source_z_basis is not None:
            payload["source_z_basis"] = self.source_z_basis
        if self.source_projection_policy is not None:
            payload["source_projection_policy"] = self.source_projection_policy
        return payload


def direct_signal_identity_from_model(
    model: Any,
    params: Mapping[str, Any] | None,
    *,
    method_name: str,
    default_modality: str,
    expected_model_classes_by_modality: Mapping[str, tuple[str, ...] | list[str] | set[str]],
    source_input_kind: str | None = None,
    source_z_basis: str | None = None,
    source_projection_policy: str | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> DirectSignalIdentity:
    """Resolve direct-signal identity from config plus concrete model class."""

    payload = dict(params or {})
    raw_modality = payload.get("imaging_model", default_modality)
    modality = require_modality_name(raw_modality, item_label="direct signal product modality")
    expected_raw = expected_model_classes_by_modality.get(modality)
    if expected_raw is None:
        allowed = sorted(expected_model_classes_by_modality)
        raise ValueError(
            f"Direct signal modality {modality!r} is not supported by "
            f"{type(model).__name__}; expected one of {allowed!r}."
        )
    expected = tuple(str(name) for name in expected_raw)
    model_class = type(model).__name__
    if model_class not in expected:
        raise ValueError(
            "Direct signal model/config mismatch: "
            f"modality {modality!r} requires model class in {expected!r}, "
            f"got {model_class!r}."
        )
    method = str(method_name).strip()
    if not method:
        raise ValueError("direct signal method_name must be non-empty.")
    producer = f"{model_class}.{method}"
    return DirectSignalIdentity(
        modality=modality,
        producer=producer,
        model_class=model_class,
        method_name=method,
        source_input_kind=source_input_kind,
        source_z_basis=source_z_basis,
        source_projection_policy=source_projection_policy,
        extra_provenance=dict(extra_provenance or {}),
    )


@dataclass(frozen=True)
class DirectParticleSignalProduct:
    """A typed direct particle/source-map response for analysis/Fisher code.

    ``values`` are never implicitly array-coerced.  The product's composed
    ``ArrayRepresentation`` owns the signal domain, units, value form, and
    stage.  Source-map provenance may carry its own representation so emitter
    density, secondary-yield, and projected-phase sources cannot be mistaken for
    detector-count Fisher derivatives.
    """

    values: np.ndarray
    representation: ArrayRepresentation
    modality: str
    producer: str
    safe_for_fisher: bool
    detector_scale_applied: bool
    background_included: bool
    source_representation: ArrayRepresentation | None = None
    detector_scale_factor: float | None = None
    conversion_note: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract_id: str = DIRECT_PARTICLE_SIGNAL_CONTRACT_ID

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                f"DirectParticleSignalProduct.values must be a 2D analysis image; got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("DirectParticleSignalProduct.values must contain only finite values.")
        representation = (
            self.representation
            if isinstance(self.representation, ArrayRepresentation)
            else UNKNOWN_ARRAY_REPRESENTATION
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "safe_for_fisher", bool(self.safe_for_fisher))
        object.__setattr__(self, "detector_scale_applied", bool(self.detector_scale_applied))
        object.__setattr__(self, "background_included", bool(self.background_included))
        if self.source_representation is not None and not isinstance(
            self.source_representation,
            ArrayRepresentation,
        ):
            object.__setattr__(self, "source_representation", UNKNOWN_ARRAY_REPRESENTATION)
        if self.safe_for_fisher and not _direct_representation_safe_for_fisher(representation):
            raise ValueError(
                "DirectParticleSignalProduct marked safe_for_fisher=True but its "
                "ArrayRepresentation is not a Fisher analysis basis. Convert the "
                "source response to detector/analysis units first."
            )

    @property
    def signal_basis(self) -> str | None:
        return _representation_label(self.representation)

    @property
    def units(self) -> str | None:
        if self.representation is UNKNOWN_ARRAY_REPRESENTATION:
            return None
        return self.representation.units

    @property
    def source_basis(self) -> str | None:
        return _representation_label(self.source_representation)

    @property
    def source_units(self) -> str | None:
        if self.source_representation is None or self.source_representation is UNKNOWN_ARRAY_REPRESENTATION:
            return None
        return self.source_representation.units

    def __array__(self, dtype: Any = None) -> np.ndarray:  # pragma: no cover - exercised by callers.
        del dtype
        raise TypeError(
            "DirectParticleSignalProduct refuses implicit ndarray conversion. "
            "Use .fisher_signal_array() only after checking the representation, "
            "units, and detector_scale_applied metadata."
        )

    def require_safe_for_fisher(self, *, context: str) -> None:
        """Raise unless this product is explicitly in a Fisher-compatible basis."""

        if not self.safe_for_fisher:
            raise ValueError(
                f"{context} received direct signal product from {self.producer!r} "
                f"in representation {self.representation!r}, which is not Fisher-safe. "
                "Use the modality's detector/analysis conversion product instead of "
                "feeding source-density/yield/phase responses directly to CRLB code."
            )

    def fisher_signal_array(self, *, context: str = "fisher") -> np.ndarray:
        """Return the array only after the basis contract says Fisher may consume it."""

        self.require_safe_for_fisher(context=context)
        return np.asarray(self.values, dtype=float)

    def metadata(self, *, prefix: str = "direct_particle_signal") -> dict[str, Any]:
        """Serialize the basis contract for reports/provenance sidecars."""

        payload = {
            f"{prefix}_contract_id": self.contract_id,
            f"{prefix}_modality": self.modality,
            f"{prefix}_producer": self.producer,
            f"{prefix}_signal_basis": self.signal_basis,
            f"{prefix}_units": self.units,
            f"{prefix}_safe_for_fisher": bool(self.safe_for_fisher),
            f"{prefix}_detector_scale_applied": bool(self.detector_scale_applied),
            f"{prefix}_background_included": bool(self.background_included),
            f"{prefix}_source_basis": self.source_basis,
            f"{prefix}_source_units": self.source_units,
            f"{prefix}_detector_scale_factor": self.detector_scale_factor,
            f"{prefix}_conversion_note": self.conversion_note,
            f"{prefix}_provenance": dict(self.provenance),
        }
        payload.update(self.representation.metadata(prefix=f"{prefix}_array"))
        if self.source_representation is not None:
            payload.update(
                self.source_representation.metadata(prefix=f"{prefix}_source_array")
            )
        return payload


def reject_direct_particle_signal_product(value: Any, *, context: str) -> None:
    """Guard Fisher functions from receiving a typed product as a bare array."""

    if isinstance(value, DirectParticleSignalProduct):
        raise TypeError(
            f"{context} received a DirectParticleSignalProduct instead of a plain "
            "analysis array. This object deliberately blocks implicit conversion so "
            "source-density/yield/phase products cannot cross into Fisher unseen. "
            "Call product.fisher_signal_array(context=...) after verifying the "
            "metadata, and pass a matching typed AnalysisNoiseModel."
        )


def detector_count_delta_representation(*, units: str = "counts") -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_CAMERA_COUNT,
        value_form=VALUE_DELTA,
        units=units,
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=STAGE_DIRECT_SIGNAL,
        semantic_label="detector_counts_delta",
    )


def electron_count_delta_representation(*, units: str = "electrons") -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_ELECTRON_COUNT,
        value_form=VALUE_DELTA,
        units=units,
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=STAGE_DIRECT_SIGNAL,
        semantic_label="detector_electrons_delta",
    )


def analysis_contrast_representation(*, units: str = "contrast") -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_RELATIVE_REFERENCE,
        value_form=VALUE_DELTA,
        units=units,
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=STAGE_ANALYSIS_CONTRAST,
        semantic_label="analysis_contrast",
    )


def fluorescence_emission_source_representation(
    *,
    units: str = "emission_density_per_detector_pixel_area",
) -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_FLUORESCENCE_EMISSION_DENSITY,
        value_form=VALUE_SOURCE_DENSITY,
        units=units,
        coordinate_frame=COORD_PROJECTED_XY,
        pipeline_stage=STAGE_SOURCE_MAP,
        semantic_label="fluorescence_emission_density",
    )


def sem_secondary_electron_source_representation(
    *,
    units: str = "secondary_electron_yield_delta",
) -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_SEM_SECONDARY_ELECTRON_YIELD,
        value_form=VALUE_SOURCE_DENSITY,
        units=units,
        coordinate_frame=COORD_PROJECTED_XY,
        pipeline_stage=STAGE_SOURCE_MAP,
        semantic_label="sem_secondary_electron_yield",
    )


def tem_projected_phase_source_representation(
    *,
    units: str = "relative_projected_phase_contrast",
) -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=DOMAIN_TEM_PROJECTED_PHASE_CONTRAST,
        value_form=VALUE_SOURCE_DENSITY,
        units=units,
        coordinate_frame=COORD_PROJECTED_XY,
        pipeline_stage=STAGE_SOURCE_MAP,
        semantic_label="tem_projected_phase_contrast",
    )


__all__ = [
    "DIRECT_PARTICLE_SIGNAL_CONTRACT_ID",
    "DirectParticleSignalProduct",
    "DirectSignalIdentity",
    "analysis_contrast_representation",
    "detector_count_delta_representation",
    "direct_signal_identity_from_model",
    "electron_count_delta_representation",
    "fluorescence_emission_source_representation",
    "reject_direct_particle_signal_product",
    "sem_secondary_electron_source_representation",
    "tem_projected_phase_source_representation",
]
