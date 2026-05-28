from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ParamType = Literal["float", "int", "bool", "enum"]


class ParamSpec(TypedDict, total=False):
    """
    Container for user-facing parameter metadata.

    Keys:
      - key:         top-level PARAMS key; object edits are handled by callers
      - type:        "float" | "int" | "bool" | "enum"
      - default:     default value (used when PARAMS/base does not contain key)
      - min:         numeric lower bound (for float/int)
      - max:         numeric upper bound (for float/int)
      - choices:     list of allowed values (for enum)
      - ui_label:    human-readable name
      - group:       logical UI group ("Particle", "Optics", "Imaging", "Noise", ...)
      - description: short human-readable description for prompts/tooltips
    """
    key: str
    type: ParamType
    default: Any
    min: float
    max: float
    choices: List[Any]
    ui_label: str
    group: str
    description: str


PARAM_SCHEMA: Dict[str, ParamSpec] = {
    # Particle-related controls
    "particle_diameter_nm": ParamSpec(
        key="particles",
        type="float",
        default=100.0,
        min=5.0,
        max=500.0,
        ui_label="Particle diameter (nm)",
        group="Particle",
        description="Optical diameter of the particle in nanometers.",
    ),
    "particle_material": ParamSpec(
        key="particles",
        type="enum",
        default="Gold",
        choices=["Gold", "Silver", "Polystyrene"],
        ui_label="Particle material",
        group="Particle",
        description="Material label used for refractive index lookup.",
    ),
    # Optics
    "wavelength_nm": ParamSpec(
        key="wavelength_nm",
        type="float",
        default=635.0,
        min=400.0,
        max=800.0,
        ui_label="Wavelength (nm)",
        group="Optics",
        description="Illumination wavelength in vacuum (nm).",
    ),
    "numerical_aperture": ParamSpec(
        key="numerical_aperture",
        type="float",
        default=1.2,
        min=0.8,
        max=1.49,
        ui_label="Numerical aperture",
        group="Optics",
        description="Objective numerical aperture.",
    ),
    # Imaging / background
    "mask_outer_ring_count": ParamSpec(
        key="mask_outer_ring_count",
        type="int",
        default=0,
        min=0,
        max=6,
        ui_label="Mask outer rings",
        group="Mask",
        description="Number of PSF rings outside the central lobe to include in masks.",
    ),
    "background_intensity": ParamSpec(
        key="background_intensity",
        type="float",
        default=100.0,
        min=0.0,
        max=500.0,
        ui_label="Background intensity",
        group="Imaging",
        description="Average background intensity level (camera counts).",
    ),
    "empirical_background_enabled": ParamSpec(
        key="empirical_background_enabled",
        type="bool",
        default=False,
        ui_label="Empirical shading field",
        group="Imaging",
        description="Enable a smooth empirical illumination/background nuisance field.",
    ),
    "empirical_background_relative_std": ParamSpec(
        key="empirical_background_relative_std",
        type="float",
        default=0.03,
        min=0.0,
        max=0.25,
        ui_label="Shading relative std",
        group="Imaging",
        description="Relative standard deviation of the empirical shading field.",
    ),
    # Noise controls match canonical camera_noise.py parameter names.
    "shot_noise_enabled": ParamSpec(
        key="shot_noise_enabled",
        type="bool",
        default=True,
        ui_label="Shot noise",
        group="Noise",
        description="Enable Poisson photon/electron shot noise.",
    ),
    "gaussian_noise_enabled": ParamSpec(
        key="gaussian_noise_enabled",
        type="bool",
        default=True,
        ui_label="Read / Gaussian noise",
        group="Noise",
        description="Enable Gaussian read noise (controlled by read_noise_counts).",
    ),
    "camera_gain_e_per_count": ParamSpec(
        key="camera_gain_e_per_count",
        type="float",
        default=1.0,
        min=0.001,
        max=10000.0,
        ui_label="Camera gain (e⁻/count)",
        group="Noise",
        description=(
            "Detected photoelectrons per output camera count/ADU. "
            "Controls shot-noise magnitude: σ_shot = sqrt(counts / gain). "
            "Calibrate from the real video with "
            "camera_noise.calibrate_camera_gain_e_per_count_from_video()."
        ),
    ),
    "read_noise_counts": ParamSpec(
        key="read_noise_counts",
        type="float",
        default=1.0,
        min=0.0,
        max=10000.0,
        ui_label="Read noise (counts RMS)",
        group="Noise",
        description=(
            "RMS Gaussian read noise in camera counts. "
            "Convert from electrons: read_noise_counts = σ_e / camera_gain_e_per_count."
        ),
    ),

}
