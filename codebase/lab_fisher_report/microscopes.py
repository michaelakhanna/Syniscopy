"""Microscope assembly primitives for lab Fisher reports.

A microscope owns the report comparison identity. Its modality remains backend
metadata installed into the resolved parameter mapping immediately before
per-candidate runtime-owner validation.
"""

from __future__ import annotations

from configured_parameters import configured_assign

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from config import BackgroundSubtractionSettings, MicroscopeRuntimeSettings
from microscope_specs import (
    MICROSCOPE_SET_SCHEMA_VERSION,
    MicroscopeSet,
    MicroscopeSpec,
    assert_public_param_keys,
    load_microscope_set as _load_microscope_set,
    microscope_specs_from_modalities,
    require_mapping,
    validate_microscope_overlay_surface,
)
from param_schema import PUBLIC_PARAM_KEYS
from particle_specs import normalize_particle_specs
from postprocessing import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)
from presets import apply_instrument_preset

from .report_contracts import MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS


_LAB_REPORT_FORBIDDEN_PARAM_MESSAGE = (
    "The lab Fisher report owns dynamic-CRLB toggles, deterministic output "
    "flags, random seed, and shared particle geometry at the run/shared_params "
    "layer. Keep microscope.params as sparse instrument/backend overrides, plus "
    "explicitly reportable acquisition timing keys, so microscope_ranking.csv, "
    "sequence_fisher_summary.csv, manifest.json, and report.md describe one "
    "coherent comparison basis."
)


def _validate_single_report_particle(params: Mapping[str, Any], *, microscope_name: str) -> None:
    specs = normalize_particle_specs(dict(params), mutate=False)
    if len(specs) != 1:
        raise ValueError(
            f"lab_fisher_report microscope {microscope_name!r} resolved {len(specs)} "
            "logical particles; microscope ranking requires exactly one shared "
            "particle/scene target."
        )


def _validate_lab_microscope_overlay_surface(spec: MicroscopeSpec) -> None:
    validate_microscope_overlay_surface(
        spec,
        field_name=f"{spec.name}.params_overlay",
        forbidden_param_keys=MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS,
        forbidden_param_message=_LAB_REPORT_FORBIDDEN_PARAM_MESSAGE,
    )


def _validate_lab_report_background_subtraction_method(
    params: Mapping[str, Any],
    *,
    microscope_name: str,
) -> None:
    """Enforce the report contrast/noise-frame contract after all overlays."""

    method = BackgroundSubtractionSettings.from_params(params).method
    # Explicit microscope configs can install background_subtraction_method in
    # shared_params or a microscope overlay after the base report defaults. The
    # resolved microscope params therefore own this validation; otherwise one
    # candidate could silently use video/raw subtraction while the ranking table
    # still claims one common contrast/noise basis.
    if method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            f"lab_fisher_report microscope {microscope_name!r} renders frame sequences; "
            "background_subtraction_method='video_median' is unsupported because "
            "it mixes temporal content into the contrast frames. Use 'reference_frame'."
        )
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            f"lab_fisher_report microscope {microscope_name!r} requires "
            "background_subtraction_method='reference_frame' so its contrast image "
            "and noise variance share the same analysis units."
        )
    if method not in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            f"Unsupported background_subtraction_method for lab_fisher_report "
            f"microscope {microscope_name!r}: {method!r}. Use 'reference_frame'."
        )


def load_microscope_set(path: str | Path) -> MicroscopeSet:
    """Load and validate a lab Fisher microscope-set JSON file."""

    return _load_microscope_set(
        path,
        forbidden_param_keys=MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS,
        forbidden_param_message=_LAB_REPORT_FORBIDDEN_PARAM_MESSAGE,
    )


def microscopes_from_modality_sweep(
    modalities: str | Iterable[str],
    shared_overlay: Mapping[str, Any] | None = None,
) -> tuple[MicroscopeSpec, ...]:
    """Build the fixed-instrument modality-sweep special case."""

    from .params_assembly import _resolve_modalities

    modality_spec = (
        modalities
        if isinstance(modalities, str)
        else ",".join(str(modality) for modality in modalities)
    )
    return microscope_specs_from_modalities(
        _resolve_modalities(modality_spec),
        shared_overlay=shared_overlay,
        forbidden_param_keys=MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS,
        forbidden_param_message=_LAB_REPORT_FORBIDDEN_PARAM_MESSAGE,
    )


def resolve_microscope_params(
    spec: MicroscopeSpec,
    base_params: Mapping[str, Any],
    *,
    shared_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return effective parameters for one microscope candidate.

    Layering is deterministic: caller-provided base params, optional named
    instrument preset, shared params, microscope overlay, then the
    microscope's canonical modality installed as ``imaging_model`` immediately
    before runtime-owner validation.
    """

    if not isinstance(spec, MicroscopeSpec):
        raise TypeError(f"spec must be a MicroscopeSpec; got {type(spec).__name__}.")
    params = deepcopy(require_mapping(base_params, field_name="base_params"))
    if spec.instrument:
        params = apply_instrument_preset(params, spec.instrument)
    missing_base_keys = sorted(key for key in PUBLIC_PARAM_KEYS if key not in params)
    if missing_base_keys:
        raise ValueError(
            "base_params must be a complete public parameters mapping before resolving "
            f"microscope {spec.name!r}; missing {missing_base_keys[:12]}"
            + ("..." if len(missing_base_keys) > 12 else "")
        )
    if shared_params:
        shared_mapping = require_mapping(shared_params, field_name="shared_params")
        assert_public_param_keys(shared_mapping, field_name="shared_params")
        params.update(deepcopy(shared_mapping))
    _validate_lab_microscope_overlay_surface(spec)
    overlay = require_mapping(
        spec.params_overlay,
        field_name=f"{spec.name}.params_overlay",
    )
    params.update(deepcopy(overlay))
    configured_assign(params, "imaging_model", spec.modality)
    MicroscopeRuntimeSettings.from_params(params)
    _validate_lab_report_background_subtraction_method(
        params,
        microscope_name=spec.name,
    )
    # Validate the final resolved microscope scene, not only the shared base.
    # shared_params may legitimately own particles, but every microscope must
    # still rank the same single logical target after all overlays are applied.
    _validate_single_report_particle(params, microscope_name=spec.name)
    return params


__all__ = [
    "MICROSCOPE_SET_SCHEMA_VERSION",
    "MicroscopeSet",
    "MicroscopeSpec",
    "load_microscope_set",
    "microscopes_from_modality_sweep",
    "resolve_microscope_params",
]
