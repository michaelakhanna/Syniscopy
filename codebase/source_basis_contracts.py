"""Source-map representation contracts for roughness coupling.

Roughness/source-map transfer is governed by the source map's composed
``ArrayRepresentation`` domain, not by a separate source-basis enum.  This keeps
fluorescence density, SEM yield, and TEM projected phase in the same
representation system used by direct-signal and volume products.
"""

from __future__ import annotations

from typing import Any

from array_representation import (
    ArrayRepresentation,
    COORD_PROJECTED_XY,
    DOMAIN_FLUORESCENCE_EMISSION_DENSITY,
    DOMAIN_SEM_SECONDARY_ELECTRON_YIELD,
    DOMAIN_TEM_PROJECTED_PHASE_CONTRAST,
    STAGE_SOURCE_MAP,
    UNKNOWN_ARRAY_REPRESENTATION,
    VALUE_SOURCE_DENSITY,
)


class RoughnessSourceBasis(str):
    """Physical meaning of the public roughness source before coupling."""

    OPTICAL_INTERFACE_FIELD = "optical_interface_field"
    FLUORESCENCE_EXCITATION_INTENSITY_GAIN = "fluorescence_excitation_intensity_gain"


ROUGHNESS_SOURCE_BASIS_CHOICES = (
    RoughnessSourceBasis.OPTICAL_INTERFACE_FIELD,
    RoughnessSourceBasis.FLUORESCENCE_EXCITATION_INTENSITY_GAIN,
)
ROUGHNESS_SOURCE_COUPLING_CHOICES = (
    "independent",
    "coherent_amplitude",
    "field_weighted",
    "scene_weighted",
    "channel_weighted",
)

_INTENSITY_GAIN_SOURCE_COUPLINGS = {
    "independent",
    "field_weighted",
    "scene_weighted",
    "channel_weighted",
}


def normalize_roughness_source_basis(value: Any) -> str:
    """Return a valid roughness-source basis string."""

    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if raw not in ROUGHNESS_SOURCE_BASIS_CHOICES:
        raise ValueError(
            "parameters['sample_environment_pattern_roughness_source_basis'] must "
            f"be one of {ROUGHNESS_SOURCE_BASIS_CHOICES!r}; got {value!r}."
        )
    return raw


def normalize_roughness_source_coupling(value: Any) -> str:
    """Return a valid roughness-source coupling string."""

    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if raw not in ROUGHNESS_SOURCE_COUPLING_CHOICES:
        raise ValueError(
            "parameters['sample_environment_pattern_roughness_source_coupling'] must "
            f"be one of {ROUGHNESS_SOURCE_COUPLING_CHOICES!r}; got {value!r}."
        )
    return raw


def source_map_representation_label(source_representation: ArrayRepresentation | None) -> str:
    """Return a compact metadata label for a source-map representation."""

    if source_representation is None:
        return "no_source_map"
    if source_representation is UNKNOWN_ARRAY_REPRESENTATION:
        return "unknown_source_map_representation"
    return source_representation.semantic_label or source_representation.domain


def _source_representation(
    *,
    domain: str,
    units: str,
    semantic_label: str,
) -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=domain,
        value_form=VALUE_SOURCE_DENSITY,
        units=units,
        coordinate_frame=COORD_PROJECTED_XY,
        pipeline_stage=STAGE_SOURCE_MAP,
        semantic_label=semantic_label,
    )


def infer_source_map_representation(
    *,
    source_input_kind: Any = None,
    modality_name: Any = None,
    source_map: Any = None,
) -> ArrayRepresentation | None:
    """Infer a renderer source-map representation from public metadata."""

    kind = "" if source_input_kind is None else str(source_input_kind).strip().lower()
    modality = "" if modality_name is None else str(modality_name).strip().lower()
    source_class = "" if source_map is None else type(source_map).__name__.lower()

    if "fluorophore" in kind or "emitter_density" in kind or "tirf" in kind:
        return _source_representation(
            domain=DOMAIN_FLUORESCENCE_EMISSION_DENSITY,
            units="emission_density_per_detector_pixel_area",
            semantic_label="fluorescence_emission_density",
        )
    if "sem" in kind or modality.startswith("sem") or source_class == "semmaterialsourcecanvas":
        return _source_representation(
            domain=DOMAIN_SEM_SECONDARY_ELECTRON_YIELD,
            units="secondary_electron_yield_delta",
            semantic_label="sem_secondary_electron_yield",
        )
    if "tem" in kind or modality.startswith("tem"):
        return _source_representation(
            domain=DOMAIN_TEM_PROJECTED_PHASE_CONTRAST,
            units="relative_projected_phase_contrast",
            semantic_label="tem_projected_phase_contrast",
        )
    if not kind and not modality and source_map is None:
        return None
    return UNKNOWN_ARRAY_REPRESENTATION


def describe_roughness_source_transfer(
    *,
    source_representation: ArrayRepresentation | None,
    roughness_source_basis: Any,
    coupling_mode: Any,
) -> dict[str, Any]:
    """Return machine-readable roughness/source-map transfer policy metadata."""

    representation = (
        source_representation
        if isinstance(source_representation, ArrayRepresentation) or source_representation is None
        else UNKNOWN_ARRAY_REPRESENTATION
    )
    label = source_map_representation_label(representation)
    domain = "none" if representation is None else representation.domain
    roughness_basis = normalize_roughness_source_basis(roughness_source_basis)
    coupling = normalize_roughness_source_coupling(coupling_mode)
    allowed = False
    reason = "unclassified_source_representation"
    gain_semantics = None

    if representation is None:
        allowed = True
        reason = "no_particle_source_map_present"
        gain_semantics = "none"
    elif coupling == "coherent_amplitude":
        allowed = False
        reason = "coherent_amplitude_is_field_only_not_source_map_representation"
    elif domain == DOMAIN_FLUORESCENCE_EMISSION_DENSITY:
        allowed = (
            roughness_basis
            == RoughnessSourceBasis.FLUORESCENCE_EXCITATION_INTENSITY_GAIN
            and coupling in _INTENSITY_GAIN_SOURCE_COUPLINGS
        )
        reason = (
            "fluorescence_density_accepts_declared_excitation_intensity_gain"
            if allowed
            else "fluorescence_density_requires_explicit_excitation_intensity_basis"
        )
        gain_semantics = "excitation_intensity_gain" if allowed else None
    elif domain == DOMAIN_SEM_SECONDARY_ELECTRON_YIELD:
        allowed = False
        reason = "sem_yield_source_rejects_optical_roughness_gain_without_electron_policy"
    elif domain == DOMAIN_TEM_PROJECTED_PHASE_CONTRAST:
        allowed = False
        reason = "tem_projected_phase_rejects_optical_roughness_gain_without_potential_policy"
    else:
        allowed = False
        reason = "unknown_source_map_representation_requires_explicit_contract"

    payload = {
        "allowed": bool(allowed),
        "reason": reason,
        "source_map_representation_label": label,
        "source_map_representation_domain": domain,
        "roughness_source_basis": roughness_basis,
        "coupling_mode": coupling,
        "gain_semantics": gain_semantics,
    }
    if isinstance(representation, ArrayRepresentation):
        payload.update(representation.metadata(prefix="source_map_array"))
    return payload


def require_roughness_source_transfer_allowed(
    *,
    source_representation: ArrayRepresentation | None,
    roughness_source_basis: Any,
    coupling_mode: Any,
    source_input_kind: Any = None,
    modality_name: Any = None,
) -> dict[str, Any]:
    """Raise if roughness would cross an unsupported source-map representation."""

    policy = describe_roughness_source_transfer(
        source_representation=source_representation,
        roughness_source_basis=roughness_source_basis,
        coupling_mode=coupling_mode,
    )
    if not policy["allowed"]:
        raise ValueError(
            "Roughness/source-map coupling is not physically declared for this "
            "source representation: "
            f"source_map_representation_label={policy['source_map_representation_label']!r}, "
            f"source_map_representation_domain={policy['source_map_representation_domain']!r}, "
            f"source_input_kind={source_input_kind!r}, "
            f"modality={modality_name!r}, "
            f"roughness_source_basis={policy['roughness_source_basis']!r}, "
            f"coupling_mode={policy['coupling_mode']!r}, "
            f"reason={policy['reason']}. "
            "Use sample_environment_pattern_roughness_source_basis="
            "'fluorescence_excitation_intensity_gain' only for fluorescence "
            "excitation-density coupling, or leave electron source maps "
            "uncoupled until an explicit electron-material roughness policy is "
            "implemented."
        )
    return policy


__all__ = [
    "ROUGHNESS_SOURCE_BASIS_CHOICES",
    "ROUGHNESS_SOURCE_COUPLING_CHOICES",
    "RoughnessSourceBasis",
    "describe_roughness_source_transfer",
    "infer_source_map_representation",
    "normalize_roughness_source_basis",
    "normalize_roughness_source_coupling",
    "require_roughness_source_transfer_allowed",
    "source_map_representation_label",
]
