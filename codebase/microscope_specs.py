"""Canonical microscope candidate specifications.

The comparison object is a microscope: a unique candidate name, a contrast
modality used as backend metadata, and a sparse microscope-local parameter
overlay. Lab reports and matched packets both consume this module so microscope
identity, overlay validation, and fixed-instrument modality-sweep construction
cannot drift across workflows.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from json_utils import load_typed_json
from microscope_axes import assert_microscope_overlay
from modality_parameter_surface import REPORT_SHARED_PARAM_KEYS
from modality_registry import (
    modality_report_parameter_surface,
    relevant_param_keys,
    require_modality_name,
)
from param_schema import PUBLIC_PARAM_KEYS


MICROSCOPE_SET_SCHEMA_VERSION = "syniscopy-microscope-set-v1"

OVERLAY_SURFACE_CANDIDATE = "candidate"
OVERLAY_SURFACE_COMMON_GRID_PACKET = "common_grid_packet"


@dataclass(frozen=True)
class MicroscopeSpec:
    """One configured microscope candidate.

    ``name`` is the comparison key. ``modality`` selects the backend/contrast
    mechanism. ``params_overlay`` is microscope-local; shared sample/scene
    authority is rejected before a candidate is resolved.
    """

    name: str
    modality: str
    params_overlay: Mapping[str, Any] = field(default_factory=dict)
    instrument: str | None = None


@dataclass(frozen=True)
class MicroscopeSet:
    """Parsed microscope JSON with shared scene/run parameters."""

    microscopes: tuple[MicroscopeSpec, ...]
    shared_params: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MICROSCOPE_SET_SCHEMA_VERSION


def require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Return a plain dict or raise a field-specific schema error."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a JSON object/dict; got {type(value).__name__}."
        )
    return dict(value)


def string_keyed_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Return a mapping with string keys, rejecting post-canonical duplicates."""

    raw = require_mapping(value, field_name=field_name)
    out: dict[str, Any] = {}
    for key, payload in raw.items():
        text_key = str(key)
        if text_key in out:
            raise ValueError(
                f"{field_name} contains duplicate parameter key after string "
                f"canonicalization: {text_key!r}."
            )
        out[text_key] = payload
    return out


def _microscope_name(value: Any, *, index: int, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}[{index}].name must be a non-empty string.")
    return value.strip()


def microscope_overlay_allowed_keys(
    modality: str,
    *,
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
) -> frozenset[str]:
    """Return canonical public keys allowed in a microscope-local overlay."""

    modality_key = require_modality_name(modality)
    if overlay_surface == OVERLAY_SURFACE_CANDIDATE:
        return relevant_param_keys(modality_key)
    if overlay_surface == OVERLAY_SURFACE_COMMON_GRID_PACKET:
        # Matched information packets currently require one common stored image
        # grid. Until a reprojection owner exists, packet-local microscope
        # overlays may vary instrument/backend/detector knobs but not the shared
        # run/grid keys excluded by template_keys().
        return modality_report_parameter_surface(modality_key).template_keys(
            REPORT_SHARED_PARAM_KEYS
        )
    raise ValueError(
        f"Unknown microscope overlay surface {overlay_surface!r}; expected "
        f"{OVERLAY_SURFACE_CANDIDATE!r} or {OVERLAY_SURFACE_COMMON_GRID_PACKET!r}."
    )


def assert_public_param_keys(overlay: Mapping[str, Any], *, field_name: str) -> None:
    """Reject keys outside the canonical public parameter schema."""

    unknown = sorted(str(key) for key in overlay if str(key) not in PUBLIC_PARAM_KEYS)
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown parameter key(s) {unknown}. "
            "Use canonical public parameter-schema keys only."
        )


def validate_microscope_overlay_surface(
    spec: MicroscopeSpec,
    *,
    field_name: str | None = None,
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
    forbidden_param_keys: frozenset[str] = frozenset(),
    forbidden_param_message: str | None = None,
) -> None:
    """Validate one microscope-local sparse overlay.

    This is source authority for microscope overlay validation. Workflow-local
    constraints may pass additional forbidden keys, but they should not recreate
    modality surface or shared-scene checks elsewhere.
    """

    overlay = dict(spec.params_overlay or {})
    if not overlay:
        return
    label = field_name or f"{spec.name}.params"
    if "imaging_model" in overlay:
        raise ValueError(
            f"Microscope {spec.name!r} must use the microscope modality field, "
            "not params.imaging_model."
        )
    assert_public_param_keys(overlay, field_name=label)
    assert_microscope_overlay(overlay, field_name=label)
    blocked = sorted(str(key) for key in overlay if str(key) in forbidden_param_keys)
    if blocked:
        message = forbidden_param_message or (
            "These keys are owned by the surrounding workflow, not by an "
            "individual microscope overlay."
        )
        raise ValueError(f"{label} sets workflow-owned key(s) {blocked!r}. {message}")
    allowed = microscope_overlay_allowed_keys(
        spec.modality,
        overlay_surface=overlay_surface,
    )
    irrelevant = sorted(str(key) for key in overlay if str(key) not in allowed)
    if irrelevant:
        raise ValueError(
            f"Microscope {spec.name!r} ({spec.modality}) sets overlay key(s) "
            f"outside the {overlay_surface!r} surface: {irrelevant}. "
            f"Allowed local keys are {sorted(allowed)}."
        )


def microscope_spec_from_mapping(
    raw: Mapping[str, Any],
    *,
    index: int,
    context: str = "microscopes",
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
    forbidden_param_keys: frozenset[str] = frozenset(),
    forbidden_param_message: str | None = None,
) -> MicroscopeSpec:
    """Parse and validate one explicit microscope-spec mapping."""

    name = _microscope_name(raw.get("name"), index=index, context=context)
    modality = require_modality_name(
        raw.get("modality", ""),
        item_label=f"{context}[{index}].modality",
    )
    if "params" in raw and "params_overlay" in raw:
        raise ValueError(
            f"{context}[{index}] must use only 'params' for the microscope "
            "overlay; do not also provide 'params_overlay'."
        )
    overlay = string_keyed_mapping(
        raw.get("params", raw.get("params_overlay", {})),
        field_name=f"{context}[{index}].params",
    )
    instrument_raw = raw.get("instrument")
    if instrument_raw is not None and (
        not isinstance(instrument_raw, str) or not instrument_raw.strip()
    ):
        raise ValueError(
            f"{context}[{index}].instrument must be a non-empty string when provided."
        )
    spec = MicroscopeSpec(
        name=name,
        modality=modality,
        params_overlay=overlay,
        instrument=instrument_raw.strip() if isinstance(instrument_raw, str) else None,
    )
    validate_microscope_overlay_surface(
        spec,
        field_name=f"{context}[{index}].params",
        overlay_surface=overlay_surface,
        forbidden_param_keys=forbidden_param_keys,
        forbidden_param_message=forbidden_param_message,
    )
    return spec


def normalize_microscope_specs(
    microscopes: Any,
    *,
    context: str = "microscopes",
    allow_modality_strings: bool = False,
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
    forbidden_param_keys: frozenset[str] = frozenset(),
    forbidden_param_message: str | None = None,
) -> tuple[MicroscopeSpec, ...]:
    """Normalize an explicit microscope-spec list."""

    if isinstance(microscopes, (str, bytes)) or not isinstance(microscopes, (list, tuple)):
        raise ValueError(f"{context} must be a list/tuple of at least two microscope specs.")
    if len(microscopes) < 2:
        raise ValueError(f"{context} must contain at least two microscope specs.")

    specs: list[MicroscopeSpec] = []
    seen_names: set[str] = set()
    for index, item in enumerate(microscopes):
        if isinstance(item, Mapping):
            spec = microscope_spec_from_mapping(
                item,
                index=index,
                context=context,
                overlay_surface=overlay_surface,
                forbidden_param_keys=forbidden_param_keys,
                forbidden_param_message=forbidden_param_message,
            )
        elif allow_modality_strings:
            modality = require_modality_name(
                str(item).strip(),
                item_label=f"{context}[{index}] modality shorthand",
            )
            spec = MicroscopeSpec(name=modality, modality=modality)
        else:
            raise ValueError(
                f"{context}[{index}] must be a microscope object with name, "
                "modality, and optional params; string modality shorthand is not "
                "a packet schema."
            )
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate microscope name {spec.name!r}; names are comparison keys."
            )
        seen_names.add(spec.name)
        specs.append(spec)
    return tuple(specs)


def microscope_specs_from_modalities(
    modalities: Iterable[str],
    shared_overlay: Mapping[str, Any] | None = None,
    *,
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
    forbidden_param_keys: frozenset[str] = frozenset(),
    forbidden_param_message: str | None = None,
) -> tuple[MicroscopeSpec, ...]:
    """Build the explicit fixed-instrument modality-sweep special case."""

    overlay = string_keyed_mapping(shared_overlay or {}, field_name="shared_overlay")
    specs = tuple(
        MicroscopeSpec(
            name=require_modality_name(modality),
            modality=require_modality_name(modality),
            params_overlay=deepcopy(overlay),
        )
        for modality in modalities
    )
    seen = [spec.name for spec in specs]
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate microscope names in modality sweep: {duplicates!r}.")
    for index, spec in enumerate(specs):
        validate_microscope_overlay_surface(
            spec,
            field_name=f"fixed_instrument_sweep[{index}].params",
            overlay_surface=overlay_surface,
            forbidden_param_keys=forbidden_param_keys,
            forbidden_param_message=forbidden_param_message,
        )
    return specs


def load_microscope_set(
    path: str | Path,
    *,
    overlay_surface: str = OVERLAY_SURFACE_CANDIDATE,
    forbidden_param_keys: frozenset[str] = frozenset(),
    forbidden_param_message: str | None = None,
) -> MicroscopeSet:
    """Load and validate a microscope-set JSON file."""

    payload = load_typed_json(path, expected=dict, context="--microscopes")
    schema = payload.get(
        "schema",
        payload.get("schema_version", MICROSCOPE_SET_SCHEMA_VERSION),
    )
    if schema != MICROSCOPE_SET_SCHEMA_VERSION:
        raise ValueError(
            f"--microscopes schema must be {MICROSCOPE_SET_SCHEMA_VERSION!r}; "
            f"got {schema!r}."
        )
    raw_microscopes = payload.get("microscopes")
    specs = normalize_microscope_specs(
        raw_microscopes,
        context="microscopes",
        allow_modality_strings=False,
        overlay_surface=overlay_surface,
        forbidden_param_keys=forbidden_param_keys,
        forbidden_param_message=forbidden_param_message,
    )
    return MicroscopeSet(
        microscopes=specs,
        shared_params=require_mapping(
            payload.get("shared_params", {}),
            field_name="shared_params",
        ),
        schema_version=MICROSCOPE_SET_SCHEMA_VERSION,
    )


__all__ = [
    "MICROSCOPE_SET_SCHEMA_VERSION",
    "OVERLAY_SURFACE_CANDIDATE",
    "OVERLAY_SURFACE_COMMON_GRID_PACKET",
    "MicroscopeSet",
    "MicroscopeSpec",
    "assert_public_param_keys",
    "load_microscope_set",
    "microscope_overlay_allowed_keys",
    "microscope_spec_from_mapping",
    "microscope_specs_from_modalities",
    "normalize_microscope_specs",
    "require_mapping",
    "string_keyed_mapping",
    "validate_microscope_overlay_surface",
]
