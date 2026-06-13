"""Composed representation descriptors for Syniscopy arrays.

An array's meaning is a product of orthogonal axes: value domain, value form,
units, coordinate frame, and pipeline stage.  Stage-specific wrappers may keep
their local labels for metadata, but those labels project onto this descriptor
rather than each owning a separate flat basis taxonomy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DOMAIN_CAMERA_COUNT = "camera_count"
DOMAIN_ELECTRON_COUNT = "electron_count"
DOMAIN_INCIDENT_QUANTA = "incident_quanta"
DOMAIN_PHASE = "phase"
DOMAIN_RELATIVE_REFERENCE = "relative_reference"
DOMAIN_DISPLAY_RGB = "display_rgb"
DOMAIN_DISPLAY_COUNT = "display_count"
DOMAIN_FLUORESCENCE_EMISSION_DENSITY = "fluorescence_emission_density"
DOMAIN_SEM_SECONDARY_ELECTRON_YIELD = "sem_secondary_electron_yield"
DOMAIN_TEM_PROJECTED_PHASE_CONTRAST = "tem_projected_phase_contrast"
DOMAIN_REFRACTIVE_INDEX_CONTRAST = "refractive_index_contrast"
DOMAIN_ELECTRON_POTENTIAL_DENSITY = "electron_potential_density"
DOMAIN_UNKNOWN = "unknown"

VALUE_ABSOLUTE = "absolute"
VALUE_DELTA = "delta"
VALUE_RELATIVE = "relative"
VALUE_DENSITY_PER_NM = "density_per_nm"
VALUE_LINE_INTEGRAL = "line_integral"
VALUE_DISPLAY = "display"
VALUE_NOISY_SAMPLE = "noisy_sample"
VALUE_SOURCE_DENSITY = "source_density"
VALUE_MODEL_OUTPUT = "model_output"
VALUE_UNKNOWN = "unknown"

COORD_DETECTOR_XY = "detector_xy"
COORD_PHYSICAL_SAMPLE_ZYX = "physical_sample_zyx"
COORD_PHYSICAL_SAMPLE_Z = "physical_sample_z"
COORD_FOCUS_RELATIVE_Z = "focus_relative_z"
COORD_ENTRY_SURFACE_DEPTH_Z = "entry_surface_depth_z"
COORD_PROJECTED_XY = "projected_xy"
COORD_NONE = "none"
COORD_UNKNOWN = "unknown"

STAGE_DETECTOR_INPUT = "detector_input"
STAGE_DETECTOR_MEAN = "detector_mean"
STAGE_PHOTO_RESPONSE = "photo_response"
STAGE_RAW_CAMERA_NOISY = "raw_camera_noisy"
STAGE_PHASE_DISPLAY = "phase_display"
STAGE_ANALYSIS_CONTRAST = "analysis_contrast"
STAGE_DIRECT_SIGNAL = "direct_signal"
STAGE_SOURCE_MAP = "source_map"
STAGE_VOLUME_FIELD = "volume_field"
STAGE_VOLUME_PROJECTION = "volume_projection"
STAGE_DISPLAY = "display"
STAGE_MODEL_OUTPUT = "model_output"
STAGE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArrayRepresentation:
    """Orthogonal representation axes attached to an array-like product."""

    domain: str
    value_form: str
    units: str
    coordinate_frame: str
    pipeline_stage: str
    semantic_label: str = ""

    def metadata(self, *, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_representation": asdict(self),
            f"{prefix}_representation_domain": self.domain,
            f"{prefix}_representation_value_form": self.value_form,
            f"{prefix}_representation_units": self.units,
            f"{prefix}_representation_coordinate_frame": self.coordinate_frame,
            f"{prefix}_representation_pipeline_stage": self.pipeline_stage,
            f"{prefix}_representation_semantic_label": self.semantic_label,
        }


UNKNOWN_ARRAY_REPRESENTATION = ArrayRepresentation(
    domain=DOMAIN_UNKNOWN,
    value_form=VALUE_UNKNOWN,
    units="unknown",
    coordinate_frame=COORD_UNKNOWN,
    pipeline_stage=STAGE_UNKNOWN,
    semantic_label="unknown",
)


def _text(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip().lower()
    return text if text else fallback


def representation_from_detector_frame_conversion(
    *,
    model_output_domain: str,
    detector_output_domain: str,
    value_form: str,
    signal_units: str | None,
) -> ArrayRepresentation:
    """Project a model-output conversion declaration onto array axes."""

    source_domain = _text(model_output_domain, fallback="unknown")
    detector_domain = _text(detector_output_domain, fallback="unknown")
    form = _text(value_form, fallback=VALUE_UNKNOWN)
    units = _text(signal_units, fallback="unknown")

    if detector_domain == "camera_counts":
        domain = DOMAIN_CAMERA_COUNT
        default_units = "detector_count"
        stage = STAGE_DETECTOR_INPUT
    elif detector_domain == "electron_count":
        domain = DOMAIN_ELECTRON_COUNT
        default_units = "electron_count"
        stage = STAGE_DETECTOR_INPUT
    elif detector_domain == "phase_display_counts":
        domain = DOMAIN_DISPLAY_COUNT
        default_units = "display_count"
        stage = STAGE_PHASE_DISPLAY
        form = VALUE_DISPLAY
    else:
        domain = DOMAIN_UNKNOWN
        default_units = "unknown"
        stage = STAGE_MODEL_OUTPUT

    if source_domain == "phase_radians" and detector_domain != "phase_display_counts":
        domain = DOMAIN_PHASE
        default_units = "radian"
    if units == "unknown":
        units = default_units
    value_axis = VALUE_DISPLAY if form == "display" else VALUE_ABSOLUTE
    return ArrayRepresentation(
        domain=domain,
        value_form=value_axis,
        units=units,
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=stage,
        semantic_label=f"{source_domain}_to_{detector_domain}",
    )


def representation_from_volume_basis(
    *,
    volume_basis: str,
    coordinate_role: str,
    value_units: str,
    output_basis: str,
    output_units: str,
    physical_line_integral_performed: bool,
) -> ArrayRepresentation:
    basis = _text(volume_basis, fallback="unknown")
    coordinate = _text(coordinate_role, fallback="unknown")
    units = _text(value_units, fallback="unknown")
    if coordinate == "physical_sample_z_nm":
        coord = COORD_PHYSICAL_SAMPLE_Z
    elif coordinate == "focus_relative_z_nm":
        coord = COORD_FOCUS_RELATIVE_Z
    elif coordinate == "focus_plane_z_nm":
        coord = COORD_DETECTOR_XY
    else:
        coord = COORD_UNKNOWN

    domain_by_basis = {
        "phase_density_rad_per_nm": DOMAIN_PHASE,
        "refractive_index_contrast": DOMAIN_REFRACTIVE_INDEX_CONTRAST,
        "emitter_density_per_nm": DOMAIN_FLUORESCENCE_EMISSION_DENSITY,
        "electron_potential_density_per_nm": DOMAIN_ELECTRON_POTENTIAL_DENSITY,
        "detector_count_density_per_nm": DOMAIN_CAMERA_COUNT,
        "focus_stack_contrast": DOMAIN_RELATIVE_REFERENCE,
    }
    form = VALUE_LINE_INTEGRAL if physical_line_integral_performed else (
        VALUE_DENSITY_PER_NM if units.endswith("_per_nm") or units.endswith("/nm") else VALUE_ABSOLUTE
    )
    stage = STAGE_VOLUME_PROJECTION if physical_line_integral_performed else STAGE_VOLUME_FIELD
    return ArrayRepresentation(
        domain=domain_by_basis.get(basis, DOMAIN_UNKNOWN),
        value_form=form,
        units=_text(output_units if physical_line_integral_performed else value_units, fallback="unknown"),
        coordinate_frame=coord,
        pipeline_stage=stage,
        semantic_label=_text(output_basis if physical_line_integral_performed else basis, fallback=basis),
    )


__all__ = [
    "ArrayRepresentation",
    "UNKNOWN_ARRAY_REPRESENTATION",
    "representation_from_detector_frame_conversion",
    "representation_from_volume_basis",
]
