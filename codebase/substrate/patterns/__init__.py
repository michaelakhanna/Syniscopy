"""Substrate-pattern geometry, layout, roughness, and map generation."""

from __future__ import annotations

from .geometry import (
    is_position_in_substrate_solid,
    project_position_to_fluid_region,
    reflect_position_across_substrate_boundary,
)
from .layout import clear_sample_environment_pattern_layout_cache
from .reference_maps import (
    compute_contrast_scale_for_frame,
    generate_reference_and_background_maps,
)
from .registry import (
    canonical_sample_environment_pattern_and_preset,
    generate_sample_environment_pattern_maps,
)
from ._shared import sample_environment_pattern_is_active
from .roughness import (
    generate_empirical_background_field,
    generate_sample_environment_roughness_field,
    resize_empirical_background_field,
)

__all__ = [
    "canonical_sample_environment_pattern_and_preset",
    "clear_sample_environment_pattern_layout_cache",
    "compute_contrast_scale_for_frame",
    "generate_empirical_background_field",
    "generate_reference_and_background_maps",
    "generate_sample_environment_pattern_maps",
    "generate_sample_environment_roughness_field",
    "is_position_in_substrate_solid",
    "project_position_to_fluid_region",
    "reflect_position_across_substrate_boundary",
    "resize_empirical_background_field",
    "sample_environment_pattern_is_active",
]
