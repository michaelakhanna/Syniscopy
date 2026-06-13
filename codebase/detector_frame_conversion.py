"""Canonical model-output to detector-frame conversion policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from array_representation import ArrayRepresentation, representation_from_detector_frame_conversion
from unit_contracts import assert_compatible


MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY = "relative_intensity"
MODEL_OUTPUT_DOMAIN_SCATTERED_INTENSITY = "scattered_intensity"
MODEL_OUTPUT_DOMAIN_PHASE_RADIANS = "phase_radians"
MODEL_OUTPUT_DOMAIN_EMISSION_DENSITY = "emission_density"
MODEL_OUTPUT_DOMAIN_ELECTRON_YIELD = "electron_yield"

DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS = "camera_counts"
DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT = "electron_count"
DETECTOR_OUTPUT_DOMAIN_PHASE_DISPLAY_COUNTS = "phase_display_counts"

VALUE_FORM_ABSOLUTE = "absolute"
VALUE_FORM_DISPLAY = "display"

REFERENCE_BASIS_NONE = "none"
REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY = "relative_reference_intensity"


@dataclass(frozen=True)
class DetectorFrameConversion:
    """Declared conversion from model-native output to detector-frame values.

    The conversion procedure is owned here rather than by every imaging model.
    Model classes supply only the physically model-specific scale, offset, and
    reference normalization parameters.
    """

    model_output_domain: str
    detector_output_domain: str
    value_form: str
    reference_basis: str
    scale: float = 1.0
    offset: Any = 0.0
    reference_scale: float = 1.0
    require_nonnegative: bool = True
    measurement_domain: str | None = "count"
    signal_units: str | None = "detector_count"
    representation: ArrayRepresentation | None = None

    def __post_init__(self) -> None:
        if self.representation is None:
            object.__setattr__(
                self,
                "representation",
                representation_from_detector_frame_conversion(
                    model_output_domain=self.model_output_domain,
                    detector_output_domain=self.detector_output_domain,
                    value_form=self.value_form,
                    signal_units=self.signal_units,
                ),
            )


def _finite_array(value: Any, *, label: str, context: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise FloatingPointError(f"{context}: {label} contains non-finite values.")
    return arr


def _assert_output_units(
    conversion: DetectorFrameConversion,
    *,
    context: str,
    params: Mapping[str, Any] | None,
) -> None:
    if conversion.measurement_domain is None and conversion.signal_units is None:
        return
    assert_compatible(
        context=context,
        measurement_domain=conversion.measurement_domain,
        signal_units=conversion.signal_units,
        params=params,
    )


def convert_model_output_to_detector_frame(
    *,
    model_output: Any,
    background_frame: Any,
    reference_intensity_frame: Any,
    conversion: DetectorFrameConversion,
    params: Mapping[str, Any] | None = None,
    context: str = "model-output to detector-frame conversion",
) -> np.ndarray:
    """Convert a model-native array to the renderer's detector-frame basis."""

    output = _finite_array(model_output, label="model_output", context=context)
    scale = float(conversion.scale)
    reference_scale = float(conversion.reference_scale)
    if not np.isfinite(scale):
        raise ValueError(f"{context}: conversion scale must be finite; got {conversion.scale!r}.")
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        raise ValueError(
            f"{context}: reference_scale must be finite and positive; "
            f"got {conversion.reference_scale!r}."
        )

    if conversion.reference_basis == REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY:
        background = _finite_array(background_frame, label="background_frame", context=context)
        reference = _finite_array(
            reference_intensity_frame,
            label="reference_intensity_frame",
            context=context,
        )
        denominator = np.maximum(reference * reference_scale, 1e-12)
        detector_frame = background * (output / denominator)
    elif conversion.reference_basis == REFERENCE_BASIS_NONE:
        offset = _finite_array(conversion.offset, label="conversion.offset", context=context)
        detector_frame = scale * output + offset
    else:
        raise ValueError(
            f"{context}: unknown detector-frame reference basis "
            f"{conversion.reference_basis!r}."
        )

    if np.any(~np.isfinite(detector_frame)):
        raise FloatingPointError(f"{context}: detector-frame conversion produced non-finite values.")
    if conversion.require_nonnegative and np.any(detector_frame < 0.0):
        raise ValueError(
            f"{context}: detector-frame conversion produced negative values; "
            "signed model outputs must declare a display or analysis basis explicitly."
        )
    _assert_output_units(conversion, context=context, params=params)
    return np.asarray(detector_frame, dtype=float)


__all__ = [
    "DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS",
    "DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT",
    "DETECTOR_OUTPUT_DOMAIN_PHASE_DISPLAY_COUNTS",
    "MODEL_OUTPUT_DOMAIN_ELECTRON_YIELD",
    "MODEL_OUTPUT_DOMAIN_EMISSION_DENSITY",
    "MODEL_OUTPUT_DOMAIN_PHASE_RADIANS",
    "MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY",
    "MODEL_OUTPUT_DOMAIN_SCATTERED_INTENSITY",
    "REFERENCE_BASIS_NONE",
    "REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY",
    "VALUE_FORM_ABSOLUTE",
    "VALUE_FORM_DISPLAY",
    "DetectorFrameConversion",
    "convert_model_output_to_detector_frame",
]
