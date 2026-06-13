"""Sample Environment parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


PATTERN_PRESET_SPECS = {
    "gold_holes": {
        "preset": "default_gold_holes",
        "material": "gold",
        "thickness_dimension_key": "hole_depth_nm",
    },
    "holey_carbon": {
        "preset": "holey_carbon",
        "material": "carbon",
        "thickness_dimension_key": "carbon_film_thickness_nm",
    },
    "nanopillars": {
        "preset": "default_nanopillars",
        "material": "glass",
        "thickness_dimension_key": "pillar_height_nm",
    },
    "fiducial_dots": {
        "preset": "fiducial_dots",
        "material": "gold",
        "thickness_dimension_key": "dot_height_nm",
    },
    "grid_bars": {
        "preset": "grid_bars",
        "material": "gold",
        "thickness_dimension_key": "bar_height_nm",
    },
    "microfluidic_walls": {
        "preset": "microfluidic_walls",
        "material": "glass",
        "thickness_dimension_key": "wall_height_nm",
    },
    "patterned_coverslip": {
        "preset": "patterned_coverslip",
        "material": "glass",
        "thickness_dimension_key": "coverslip_patch_height_nm",
    },
}

PATTERN_DEFAULT_PRESETS = {
    pattern: spec["preset"] for pattern, spec in PATTERN_PRESET_SPECS.items()
}

BAR_ORIENTATION_CHOICES = ("vertical", "horizontal", "both")


SAMPLE_ENVIRONMENT_SCHEMA: dict[str, ParamSpec] = {
"empirical_background_enabled": ParamSpec(
    key="empirical_background_enabled",
    default=False,
    type="bool",
    ui_label="Empirical shading field",
    group="Imaging",
    description="Enable a smooth empirical illumination/background nuisance field.",
),
"empirical_background_model": ParamSpec(
    key="empirical_background_model",
    default='multiscale_gaussian_field',
    type="enum",
    choices=["multiscale_gaussian_field", "none"],
    ui_label="Empirical background model",
    group="Imaging",
    description="Low-frequency background shaping model applied before rendering.",
),
"empirical_background_relative_std": ParamSpec(
    key="empirical_background_relative_std",
    default=0.03,
    type="float",
    min=0.0,
    max=0.25,
    ui_label="Shading relative std",
    group="Imaging",
    description="Relative standard deviation of the empirical shading field.",
),
"empirical_background_scales_px": ParamSpec(
    key="empirical_background_scales_px",
    default=[16.0, 64.0, 256.0],
    type="json",
    ui_label="Empirical background scales (px)",
    group="Imaging",
    description="Multiscale Gaussian kernel radii (pixels) for empirical background synthesis.",
),
"empirical_background_scale_weights": ParamSpec(
    key="empirical_background_scale_weights",
    default=[0.4, 0.35, 0.25],
    type="json",
    ui_label="Empirical background scale weights",
    group="Imaging",
    description="Relative weights for multiscale empirical background radii.",
),
"empirical_background_gradient_relative_strength": ParamSpec(
    key="empirical_background_gradient_relative_strength",
    default=0.02,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Background gradient strength",
    group="Imaging",
    description="Gradient modulation strength for empirical shading.",
),
"sample_environment_pattern_roughness_model": ParamSpec(
    key="sample_environment_pattern_roughness_model",
    default='none',
    type="enum",
    choices=["none", "static", "flicker", "source_matched"],
    ui_label="Sample-environment roughness model",
    group="Sample environment",
    description=(
        "Roughness/speckle model for substrate interface fields. "
        "'source_matched' uses a user-supplied reference field."
    ),
),
"sample_environment_pattern_roughness_source": ParamSpec(
    key="sample_environment_pattern_roughness_source",
    default=None,
    type="string",
    choices=[],
    ui_label="Sample-environment roughness source",
    group="Sample environment",
    description=(
        "Optional reference field (path or array) used when "
        "sample_environment_pattern_roughness_model='source_matched'."
    ),
),
"sample_environment_pattern_roughness_source_basis": ParamSpec(
    key="sample_environment_pattern_roughness_source_basis",
    default='optical_interface_field',
    type="enum",
    choices=[
        "optical_interface_field",
        "fluorescence_excitation_intensity_gain",
    ],
    ui_label="Roughness-source physical basis",
    group="Sample environment",
    description=(
        "Physical meaning of sample_environment_pattern_roughness_source before "
        "it is allowed to modulate material source maps. Optical interface "
        "fields may couple to coherent optical E-fields; fluorescence source "
        "maps require the explicit fluorescence_excitation_intensity_gain basis."
    ),
),
"sample_environment_pattern_roughness_source_coupling": ParamSpec(
    key="sample_environment_pattern_roughness_source_coupling",
    default='channel_weighted',
    type="enum",
    choices=[
        "independent",
        "coherent_amplitude",
        "field_weighted",
        "scene_weighted",
        "channel_weighted",
    ],
    ui_label="Roughness-source coupling",
    group="Sample environment",
    description=(
        "How roughness and source-channel terms are coupled after the "
        "source-map basis contract is validated. "
        "'independent' applies a real intensity gain where that gain is "
        "physically declared. 'coherent_amplitude' is field-only and is "
        "rejected for material source maps; 'field_weighted' scales by the "
        "normalized local particle scene envelope for each source map where "
        "available, with a scene fallback if needed. "
        "'scene_weighted' modulates all source maps by the same normalized "
        "scene envelope. "
        "'channel_weighted' uses channel-aware scene envelopes based on "
        "vectorial_detection_mode so source coupling follows the optical "
        "channel geometry."
    ),
),
"sample_environment_pattern_roughness_amplitude": ParamSpec(
    key="sample_environment_pattern_roughness_amplitude",
    default=0.0,
    type="float",
    min=0.0,
    max=2.0,
    ui_label="Sample roughness amplitude",
    group="Sample environment",
    description="Roughness strength before normalization; zero disables roughness perturbations.",
),
"sample_environment_pattern_roughness_correlation_pixels": ParamSpec(
    key="sample_environment_pattern_roughness_correlation_pixels",
    default=4.0,
    type="float",
    min=0.0,
    max=256.0,
    ui_label="Roughness correlation length (px)",
    group="Sample environment",
    description="Correlation scale in pixels for roughness/speckle interpolation.",
),
"sample_environment_pattern_roughness_phase_std": ParamSpec(
    key="sample_environment_pattern_roughness_phase_std",
    default=0.0,
    type="float",
    min=0.0,
    max=3.141592653589793,
    ui_label="Roughness phase std (rad)",
    group="Sample environment",
    description="Phase jitter strength (radians) for complex roughness field.",
),
"sample_environment_enabled": ParamSpec(
    key="sample_environment_enabled",
    default=True,
    type="bool",
    ui_label="Sample environment enabled",
    group="Sample environment",
    description="Enable structured sample-environment rendering and interactions.",
),
"sample_environment_pattern_enabled": ParamSpec(
    key="sample_environment_pattern_enabled",
    default=False,
    type="bool",
    ui_label="Sample-environment pattern enabled",
    group="Sample environment",
    description="Enable patterned substrate geometry in the sample environment.",
),
"medium_material": ParamSpec(
    key="medium_material",
    default='water',
    type="string",
    ui_label="Sample environment medium material",
    group="Sample environment",
    description="Material for sample-environment bulk medium.",
),
"mounting_interface_material": ParamSpec(
    key="mounting_interface_material",
    default='glass',
    type="string",
    ui_label="Mounting interface material",
    group="Sample environment",
    description="Material for the mounting interface between medium and substrate.",
),
"bulk_substrate_material": ParamSpec(
    key="bulk_substrate_material",
    default='glass',
    type="string",
    ui_label="Bulk substrate material",
    group="Sample environment",
    description="Material for the bulk substrate region.",
),
"mounting_interface_thickness_nm": ParamSpec(
    key="mounting_interface_thickness_nm",
    default=170000.0,
    type="float",
    min=0.0,
    max=1e9,
    ui_label="Mounting interface thickness (nm)",
    group="Sample environment",
    description="Thickness of the mounting/interface layer for optical stack estimates.",
),
"sample_environment_pattern_material": ParamSpec(
    key="sample_environment_pattern_material",
    default=None,
    type="string",
    ui_label="Pattern layer material override",
    group="Sample environment",
    description="Optional material override for the patterned sample-environment layer.",
),
"sample_environment_pattern": ParamSpec(
    key="sample_environment_pattern",
    default='none',
    type="enum",
    choices=[
        "none",
        "gold_holes",
        "nanopillars",
        "fiducial_dots",
        "grid_bars",
        "holey_carbon",
        "microfluidic_walls",
        "patterned_coverslip",
    ],
    ui_label="Sample-environment pattern",
    group="Sample environment",
    description="Pattern type for sample-environment layout.",
),
"sample_environment_pattern_preset": ParamSpec(
    key="sample_environment_pattern_preset",
    default='default_gold_holes',
    type="string",
    ui_label="Sample-environment pattern preset",
    group="Sample environment",
    description="Preset selection within the sample-environment pattern family.",
),
"sample_environment_pattern_dimensions": ParamSpec(
    key="sample_environment_pattern_dimensions",
    default={'hole_diameter_um': 15.0,
     'hole_edge_to_edge_spacing_um': 2.0,
     'hole_depth_nm': 20.0,
     'hole_intensity_factor': 0.7,
     'gold_intensity_factor': 1.0,
     'carbon_film_thickness_nm': 20.0,
     'pillar_diameter_um': 1.0,
     'pillar_edge_to_edge_spacing_um': 2.0,
     'pillar_height_nm': 20.0,
     'pillar_intensity_factor': 1.3,
     'background_intensity_factor': 1.0,
     'fiducial_dot_diameter_um': 0.5,
     'fiducial_dot_edge_to_edge_spacing_um': 2.0,
     'fiducial_dot_pitch_um': 5.0,
     'fiducial_dot_intensity_factor': 1.5,
     'fiducial_background_intensity_factor': 1.0,
     'grid_pitch_um': 5.0,
     'grid_bar_width_um': 0.5,
     'grid_bar_intensity_factor': 1.25,
     'grid_background_intensity_factor': 1.0,
     'microfluidic_channel_pitch_um': 10.0,
     'microfluidic_wall_width_um': 1.0,
     'microfluidic_wall_orientation': 'vertical',
     'microfluidic_wall_intensity_factor': 1.2,
     'microfluidic_channel_intensity_factor': 1.0,
     'coverslip_patch_diameter_um': 5.0,
     'coverslip_patch_edge_to_edge_spacing_um': 5.0,
     'coverslip_patch_pitch_um': 10.0,
     'coverslip_patch_intensity_factor': 1.08,
     'coverslip_background_intensity_factor': 1.0,
     'dot_height_nm': 20.0,
     'bar_height_nm': 20.0,
     'wall_height_nm': 20.0,
     'coverslip_patch_height_nm': 170000.0},
    type="json",
    ui_label="Sample-environment pattern dimensions",
    group="Sample environment",
    description="Pattern-geometry parameters used when a pattern preset is active.",
),
"sample_environment_exclusion_method": ParamSpec(
    key="sample_environment_exclusion_method",
    default='reflection',
    type="enum",
    choices=["reflection", "projection"],
    ui_label="Pattern exclusion method",
    group="Sample environment",
    description="How Brownian steps are handled when particles hit forbidden pattern regions.",
),
"sample_environment_pattern_contrast_amplitude": ParamSpec(
    key="sample_environment_pattern_contrast_amplitude",
    default=0.0,
    type="float",
    min=0.0,
    max=1e6,
    ui_label="Pattern contrast amplitude",
    group="Sample environment",
    description="Scalar contrast amplitude for patterned sample features.",
),
"sample_environment_pattern_contrast_model": ParamSpec(
    key="sample_environment_pattern_contrast_model",
    default='static',
    type="string",
    ui_label="Pattern contrast model",
    group="Sample environment",
    description="Pattern contrast evolution model.",
),
"sample_environment_pattern_position_jitter_std_nm": ParamSpec(
    key="sample_environment_pattern_position_jitter_std_nm",
    default=50.0,
    type="float",
    min=0.0,
    max=1e6,
    ui_label="Pattern position jitter (nm)",
    group="Sample environment",
    description="Standard deviation of pattern feature center jitter in nanometers.",
),
"sample_environment_pattern_shape_regularity": ParamSpec(
    key="sample_environment_pattern_shape_regularity",
    default=0.73,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Pattern shape regularity",
    group="Sample environment",
    description="Shape regularity in [0, 1], where 1 is perfectly regular.",
),
"sample_environment_pattern_edge_perturbation_max_rel_radius": ParamSpec(
    key="sample_environment_pattern_edge_perturbation_max_rel_radius",
    default=0.12,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Pattern edge perturbation max",
    group="Sample environment",
    description="Maximum edge perturbation amplitude for patterned feature boundaries.",
),
"sample_environment_pattern_edge_perturbation_mode_count": ParamSpec(
    key="sample_environment_pattern_edge_perturbation_mode_count",
    default=3,
    type="int",
    min=0,
    max=1000,
    ui_label="Pattern edge perturbation modes",
    group="Sample environment",
    description="Mode count for boundary perturbation basis of patterned features.",
),
"sample_environment_pattern_randomization_enabled": ParamSpec(
    key="sample_environment_pattern_randomization_enabled",
    default=True,
    type="bool",
    ui_label="Pattern randomization enabled",
    group="Sample environment",
    description="Enable random perturbations and jitter for patterned sample geometry.",
),
}

__all__ = [
    "PATTERN_DEFAULT_PRESETS",
    "PATTERN_PRESET_SPECS",
    "BAR_ORIENTATION_CHOICES",
    "SAMPLE_ENVIRONMENT_SCHEMA",
]
