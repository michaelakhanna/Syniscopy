from __future__ import annotations

from typing import Any

from modality_registry import (
    ELECTRON_MODALITIES,
    canonical_modality_name,
    modality_uses_relative_reference_contrast,
)

_MEASUREMENT_DOMAIN_ALIASES = {
    'counts': 'count',
    'count': 'count',
    'detector_counts': 'count',
    'detector_count': 'count',
    'phase': 'phase',
    'radian': 'phase',
    'radians': 'phase',
    'electron': 'electron_count',
    'electrons': 'electron_count',
    'electron_counts': 'electron_count',
    'electron_count': 'electron_count',
    'fringe': 'fringe_count',
    'fringe_count': 'fringe_count',
    'contrast': 'contrast',
    'relative_reference': 'contrast',
}

_SIGNAL_UNIT_ALIASES = {
    'radians': 'radian',
    'radian': 'radian',
    'counts': 'detector_count',
    'count': 'detector_count',
    'detector_counts': 'detector_count',
    'detector_count': 'detector_count',
    'electron_counts': 'electron_count',
    'electrons': 'electron_count',
    'electron_count': 'electron_count',
    'relative_reference': 'relative_reference',
    'model_contrast': 'model_contrast',
}

def _normalize_measurement_domain(value: Any) -> str:
    key = str(value or "").strip().lower()
    return _MEASUREMENT_DOMAIN_ALIASES.get(key, key)


def _normalize_signal_units(value: Any) -> str:
    key = str(value or "").strip().lower()
    return _SIGNAL_UNIT_ALIASES.get(key, key)


def _canonical_measurement_domain_and_signal_units(
    params: dict,
    model,
    modality: str,
    response_function: dict | None = None,
) -> tuple[str, str]:
    """Return canonical analysis-contrast domain labels for Fisher metadata."""
    modality_key = canonical_modality_name(modality or params.get("imaging_model", "bright_field"))
    if modality_key in ELECTRON_MODALITIES:
        return "electron_count", "electron_count"

    output_type = str(getattr(model, "output_type", "intensity")).strip().lower()
    if output_type == "phase":
        return "phase", "radian"

    if modality_uses_relative_reference_contrast(modality_key):
        return "contrast", "relative_reference"

    response = response_function or {}
    response_domain = response.get("measurement_domain")
    response_units = response.get("signal_units")
    if response_domain not in (None, "") and response_units not in (None, ""):
        normalized_domain = _normalize_measurement_domain(response_domain)
        normalized_units = _normalize_signal_units(response_units)
        if normalized_domain == "fringe_count":
            return "fringe_count", normalized_units or "detector_count"
        if (normalized_domain, normalized_units) != ("count", "detector_count"):
            return normalized_domain, normalized_units

    return "count", "detector_count"


def _canonical_contrast_frame_units(
    params: dict,
    model,
    modality: str,
    response_function: dict | None = None,
) -> str:
    measurement_domain, signal_units = _canonical_measurement_domain_and_signal_units(
        params,
        model,
        modality,
        response_function=response_function,
    )
    modality_key = canonical_modality_name(modality or params.get("imaging_model", "bright_field"))
    if measurement_domain == "phase":
        return "radian"
    if measurement_domain == "electron_count" or signal_units == "electron_count" or modality_key in ELECTRON_MODALITIES:
        return "electron_count_difference"
    if measurement_domain == "contrast" or signal_units == "relative_reference":
        return "relative_reference"
    return "detector_count_difference"


__all__ = [
    "_MEASUREMENT_DOMAIN_ALIASES",
    "_SIGNAL_UNIT_ALIASES",
    "_canonical_contrast_frame_units",
    "_canonical_measurement_domain_and_signal_units",
    "_normalize_measurement_domain",
    "_normalize_signal_units",
]
