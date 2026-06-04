"""Canonical measurement-domain and signal-unit conventions."""

from __future__ import annotations

from typing import Any

from config.runtime import resolved_modality
from modality_registry import (
    ELECTRON_MODALITIES,
    canonical_modality_name,
    modality_uses_relative_reference_contrast,
)

MEASUREMENT_DOMAINS = (
    "count",
    "phase",
    "electron_count",
    "fringe_count",
    "contrast",
    "detected_quanta",
    "model_signal",
)

SIGNAL_UNITS = (
    "detector_count",
    "radian",
    "electron_count",
    "relative_reference",
    "detected_quanta",
    "model_contrast",
    "model_signal",
)

DETECTOR_NOISE_INPUT_DOMAINS = ("camera_counts", "electron_count")
DETECTOR_NOISE_INPUT_DOMAIN_ALIASES = {
    "count": "camera_counts",
    "counts": "camera_counts",
    "detector_count": "camera_counts",
    "detector_counts": "camera_counts",
    "camera_count": "camera_counts",
    "camera_counts": "camera_counts",
    "adu": "camera_counts",
    "electron": "electron_count",
    "electrons": "electron_count",
    "electron_count": "electron_count",
    "electron_counts": "electron_count",
}


def normalize_measurement_domain(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key not in MEASUREMENT_DOMAINS:
        raise ValueError(
            "measurement_domain must be one of "
            f"{list(MEASUREMENT_DOMAINS)}; got {value!r}."
        )
    return key


def normalize_signal_units(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key not in SIGNAL_UNITS:
        raise ValueError(
            "signal_units must be one of "
            f"{list(SIGNAL_UNITS)}; got {value!r}."
        )
    return key


def normalize_detector_noise_input_domain(value: Any) -> str:
    """Return the canonical detector-noise input domain."""
    key = str(value or "").strip().lower()
    try:
        return DETECTOR_NOISE_INPUT_DOMAIN_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "detector_noise_input_domain must be 'camera_counts' or "
            f"'electron_count'; got {value!r}."
        ) from exc


def canonical_measurement_domain_and_signal_units_for_output(
    modality: str,
    output_type: str,
    response_function: dict | None = None,
) -> tuple[str, str]:
    """Return canonical domain/unit labels from modality, output type, and response metadata."""
    modality_key = canonical_modality_name(modality)
    if modality_key in ELECTRON_MODALITIES:
        return "electron_count", "electron_count"

    output_key = str(output_type or "intensity").strip().lower()
    if output_key == "phase":
        return "phase", "radian"

    if modality_uses_relative_reference_contrast(modality_key):
        return "contrast", "relative_reference"

    response = response_function or {}
    response_domain = response.get("measurement_domain")
    response_units = response.get("signal_units")
    if response_domain not in (None, "") and response_units not in (None, ""):
        normalized_domain = normalize_measurement_domain(response_domain)
        normalized_units = normalize_signal_units(response_units)
        if normalized_domain == "fringe_count":
            return "fringe_count", normalized_units or "detector_count"
        if (normalized_domain, normalized_units) != ("count", "detector_count"):
            return normalized_domain, normalized_units

    if output_key == "fringe":
        return "fringe_count", "detector_count"
    return "count", "detector_count"


def canonical_measurement_domain_and_signal_units(
    params: dict,
    model,
    modality: str,
    response_function: dict | None = None,
) -> tuple[str, str]:
    """Return canonical analysis-domain labels for a model/modality pair."""
    modality_key = canonical_modality_name(modality) if modality else resolved_modality(params)
    output_type = str(getattr(model, "output_type", "intensity")).strip().lower()
    return canonical_measurement_domain_and_signal_units_for_output(
        modality_key,
        output_type,
        response_function=response_function,
    )


def canonical_contrast_frame_units(
    params: dict,
    model,
    modality: str,
    response_function: dict | None = None,
) -> str:
    measurement_domain, signal_units = canonical_measurement_domain_and_signal_units(
        params,
        model,
        modality,
        response_function=response_function,
    )
    modality_key = canonical_modality_name(modality) if modality else resolved_modality(params)
    if measurement_domain == "phase":
        return "radian"
    if measurement_domain == "electron_count" or signal_units == "electron_count" or modality_key in ELECTRON_MODALITIES:
        return "electron_count_difference"
    if measurement_domain == "contrast" or signal_units == "relative_reference":
        return "relative_reference"
    return "detector_count_difference"


__all__ = [
    "MEASUREMENT_DOMAINS",
    "SIGNAL_UNITS",
    "DETECTOR_NOISE_INPUT_DOMAINS",
    "DETECTOR_NOISE_INPUT_DOMAIN_ALIASES",
    "canonical_contrast_frame_units",
    "canonical_measurement_domain_and_signal_units",
    "canonical_measurement_domain_and_signal_units_for_output",
    "normalize_detector_noise_input_domain",
    "normalize_measurement_domain",
    "normalize_signal_units",
]
