"""Import-light shared constants for cross-module contracts."""

from __future__ import annotations

from modality_registry import (
    VECTORIAL_FULL_FIELD_MODALITIES,
    modality_name_set,
)
from param_schema.sample_environment import PATTERN_DEFAULT_PRESETS

KNOWN_INTERNAL_PARAM_KEYS = frozenset(
    {
        "_camera_noise_static_seed",
        "_particle_specs",
        "_particle_specs_fingerprint",
        "_resolved_particles",
        "_resolved_primary_component_refractive_indices",
        "_resolved_particle_material_properties",
        "_resolved_particle_material_properties_metadata",
        "_return_mask_arrays",
        "_write_mask_files",
        "_substrate_pattern_layout_cache_token",
        "_substrate_pattern_layout_extent_nm",
        "_substrate_pattern_layout_rng",
        "_generated_spectral_channels",
        "_spectral_channel_count",
    }
)

NUM_FRAME_DURATION_SEARCH_STEPS = 32

COHERENT_REFERENCE_MODALITIES = modality_name_set(VECTORIAL_FULL_FIELD_MODALITIES)

NONNEGATIVE_MATERIAL_PROPERTY_FIELDS = frozenset(
    {
        "mean_inner_potential_V",
        "density_g_cm3",
        "se_yield_coefficient",
        "autofluorescence_per_nm",
        "fluorophore_density",
    }
)

SOURCE_MATERIAL_PROPERTY_FIELDS = NONNEGATIVE_MATERIAL_PROPERTY_FIELDS

SE3_STATE_AXES = ("x", "y", "z", "omega_x", "omega_y", "omega_z")

RAW_BACKGROUND_SUBTRACTION_METHODS = frozenset(
    {
        "none",
        "raw",
        "raw_signal",
        "off",
        "disabled",
        "no_subtraction",
    }
)
VIDEO_BACKGROUND_SUBTRACTION_METHODS = frozenset({"video_median"})
REFERENCE_BACKGROUND_SUBTRACTION_METHODS = frozenset({"reference_frame"})

MATCHED_INFORMATION_MASK_ROLES = (
    "mask_geometry",
    "mask_supported",
    "ignore_mask",
    "loss_weight",
)
