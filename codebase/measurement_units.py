"""Canonical measurement-domain and signal-unit conventions."""

from __future__ import annotations

from typing import Any

from modality_registry import (
    ELECTRON_MODALITIES,
    modality_uses_relative_reference_contrast,
    require_modality_name,
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
POISSON_MEAN_BASES = (
    "camera_count",
    "detected_quanta",
    "electron_count",
)


def _modality_from_params(params: dict) -> str:
    """Resolve modality lazily so measurement_units stays import-order pure."""
    from config.runtime import ModalitySettings

    return ModalitySettings.from_params(params).modality


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
    if key not in DETECTOR_NOISE_INPUT_DOMAINS:
        raise ValueError(
            "detector_noise_input_domain must be 'camera_counts' or "
            f"'electron_count'; got {value!r}."
        )
    return key


def normalize_poisson_mean_basis(value: Any) -> str:
    """Return the canonical basis for a Poisson mean image."""
    key = str(value or "").strip().lower()
    if key not in POISSON_MEAN_BASES:
        raise ValueError(
            "poisson_mean_units must declare one of "
            f"{list(POISSON_MEAN_BASES)}; got {value!r}."
        )
    return key


def canonical_measurement_domain_and_signal_units_for_output(
    modality: str,
    output_type: str,
    response_function: dict | None = None,
) -> tuple[str, str]:
    """Return canonical domain/unit labels from modality, output type, and response metadata."""
    modality_key = require_modality_name(modality, item_label="measurement-units modality")
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
        if normalized_domain == "fringe_count" and normalized_units == "detector_count":
            return "count", "detector_count"
        if (normalized_domain, normalized_units) != ("count", "detector_count"):
            return normalized_domain, normalized_units

    if output_key == "fringe":
        return "count", "detector_count"
    return "count", "detector_count"


def canonical_measurement_domain_and_signal_units(
    params: dict,
    model,
    modality: str,
    response_function: dict | None = None,
) -> tuple[str, str]:
    """Return canonical analysis-domain labels for a model/modality pair."""
    modality_key = (
        require_modality_name(modality, item_label="measurement-units modality")
        if modality
        else _modality_from_params(params)
    )
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
    modality_key = (
        require_modality_name(modality, item_label="measurement-units modality")
        if modality
        else _modality_from_params(params)
    )
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
    "POISSON_MEAN_BASES",
    "canonical_contrast_frame_units",
    "canonical_measurement_domain_and_signal_units",
    "canonical_measurement_domain_and_signal_units_for_output",
    "normalize_detector_noise_input_domain",
    "normalize_measurement_domain",
    "normalize_poisson_mean_basis",
    "normalize_signal_units",
]
