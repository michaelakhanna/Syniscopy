"""Dataset parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


DATASET_SCHEMA: dict[str, ParamSpec] = {
"image_size_pixels": ParamSpec(
    key="image_size_pixels",
    default=1024,
    type="int",
    min=32,
    max=16384,
    ui_label="Image size (px)",
    group="Imaging",
    description="Square output frame size in pixels (x and y).",
),
"pixel_size_nm": ParamSpec(
    key="pixel_size_nm",
    default=244,
    type="float",
    min=1.0,
    max=1e5,
    ui_label="Pixel pitch (nm)",
    group="Imaging",
    description="Physical detector pixel size in nanometers.",
),
"bit_depth": ParamSpec(
    key="bit_depth",
    default=16,
    type="int",
    min=1,
    max=16,
    ui_label="Raw bit depth",
    group="Imaging",
    description="Output encoding bit depth for integer frame storage.",
),
"fps": ParamSpec(
    key="fps",
    default=24,
    type="int",
    min=1,
    max=10000,
    ui_label="Frames per second",
    group="Imaging",
    description="Frame rate for generated videos and derived frame duration.",
),
"duration_seconds": ParamSpec(
    key="duration_seconds",
    default=1,
    type="float",
    min=0.001,
    max=86400.0,
    ui_label="Duration (s)",
    group="Imaging",
    description="Video duration in seconds (derived from num_frames/fps if num_frames is set).",
),
"num_frames": ParamSpec(
    key="num_frames",
    default=None,
    type="int",
    min=1,
    max=10000000,
    ui_label="Number of frames",
    group="Imaging",
    description="Explicit frame count; resolves duration_seconds = num_frames / fps when set.",
),
"unit_contracts_enabled": ParamSpec(
    key="unit_contracts_enabled",
    default=True,
    type="bool",
    ui_label="Unit contracts",
    group="Workflow",
    description=(
        "Enable runtime assertions that declared measurement domains and "
        "signal/noise units match at cross-module seams. These checks do not "
        "coerce or change values."
    ),
),
"exposure_time_ms": ParamSpec(
    key="exposure_time_ms",
    default=None,
    type="float",
    min=0.0,
    max=1000000.0,
    ui_label="Exposure time (ms)",
    group="Imaging",
    description="Per-frame exposure in milliseconds; None uses full-frame exposure.",
),
"matched_microscopes": ParamSpec(
    key="matched_microscopes",
    default=None,
    type="json",
    ui_label="Matched microscopes",
    group="Imaging",
    description=(
        "Optional list of named microscope spec objects for matched packet "
        "rendering. Each entry must provide name, modality, and optional sparse "
        "params; modality strings are not a packet schema."
    ),
),
"channels": ParamSpec(
    key="channels",
    default=None,
    type="json",
    ui_label="Channels",
    group="Workflow",
    description=(
        "Optional channel definitions for spectral/multi-line rendering "
        "(e.g., list of per-channel wavelength dicts)."
    ),
),
"spectral_integration_model": ParamSpec(
    key="spectral_integration_model",
    default='single_wavelength',
    type="enum",
    choices=["single_wavelength", "configured_channels", "broadband_quadrature"],
    ui_label="Spectral integration",
    group="Workflow",
    description="Single wavelength, caller-configured channels, or automatic broadband quadrature.",
),
"illumination_spectrum_center_nm": ParamSpec(
    key="illumination_spectrum_center_nm",
    default=550.0,
    type="float",
    min=1.0,
    max=5000.0,
    ui_label="Spectrum center (nm)",
    group="Workflow",
    description="Center wavelength for automatic broadband quadrature.",
),
"illumination_spectrum_fwhm_nm": ParamSpec(
    key="illumination_spectrum_fwhm_nm",
    default=40.0,
    type="float",
    min=0.0,
    max=5000.0,
    ui_label="Spectrum FWHM (nm)",
    group="Workflow",
    description="Gaussian source-spectrum width for broadband quadrature.",
),
"illumination_spectrum_num_samples": ParamSpec(
    key="illumination_spectrum_num_samples",
    default=5,
    type="int",
    min=1,
    max=101,
    ui_label="Spectrum samples",
    group="Workflow",
    description="Number of spectral samples generated for broadband quadrature.",
),
"broadband_wavelengths_nm": ParamSpec(
    key="broadband_wavelengths_nm",
    default=None,
    type="json",
    ui_label="Broadband wavelengths",
    group="Workflow",
    description="Optional explicit broadband quadrature wavelengths in nm.",
),
"broadband_weights": ParamSpec(
    key="broadband_weights",
    default=None,
    type="json",
    ui_label="Broadband weights",
    group="Workflow",
    description="Optional explicit non-negative broadband quadrature weights.",
),
"allow_broadband_overwrite_channels": ParamSpec(
    key="allow_broadband_overwrite_channels",
    default=False,
    type="bool",
    ui_label="Overwrite channels",
    group="Workflow",
    description="Allow broadband quadrature to replace an existing channels list.",
),
"multichannel_output_mode": ParamSpec(
    key="multichannel_output_mode",
    default='rgb',
    type="enum",
    choices=["rgb", "channels", "both", "none"],
    ui_label="Multichannel output mode",
    group="Workflow",
    description=(
        "Controls how per-channel outputs are written: single RGB, sidecar channel "
        "videos, both, or none."
    ),
),
"multichannel_sidecar_directory": ParamSpec(
    key="multichannel_sidecar_directory",
    default=None,
    type="string",
    ui_label="Multichannel sidecar directory",
    group="Workflow",
    description="Optional directory for per-channel sidecar videos when multichannel output is used.",
),
"volumetric_imaging_mode": ParamSpec(
    key="volumetric_imaging_mode",
    default='single_plane',
    type="enum",
    choices=["single_plane", "z_stack", "confocal", "light_sheet", "holotomography_projection"],
    ui_label="Volumetric mode",
    group="Workflow",
    description="Optional z-stack, confocal, light-sheet, or holotomography-style particle-volume output.",
),
"scene_dimensionality": ParamSpec(
    key="scene_dimensionality",
    default='single_plane_particle_scene',
    type="enum",
    choices=["single_plane_particle_scene", "particle_volume_3d"],
    ui_label="Scene dimensionality",
    group="Workflow",
    description="Single focal-plane particle scene or configured 3D particle-volume helper.",
),
"volumetric_z_planes_nm": ParamSpec(
    key="volumetric_z_planes_nm",
    default=None,
    type="json",
    ui_label="Volume z planes (nm)",
    group="Workflow",
    description="Optional explicit z planes for volumetric helper outputs.",
),
"volumetric_z_range_nm": ParamSpec(
    key="volumetric_z_range_nm",
    default=1000.0,
    type="float",
    min=0.0,
    max=1e7,
    ui_label="Volume z range (nm)",
    group="Workflow",
    description="Total z span used when z planes are generated automatically.",
),
"volumetric_z_step_nm": ParamSpec(
    key="volumetric_z_step_nm",
    default=250.0,
    type="float",
    min=1e-9,
    max=1e7,
    ui_label="Volume z step (nm)",
    group="Workflow",
    description="Maximum z spacing used by automatic volumetric z-plane generation.",
),
"volumetric_z_count": ParamSpec(
    key="volumetric_z_count",
    default=5,
    type="int",
    min=1,
    max=1001,
    ui_label="Volume z count",
    group="Workflow",
    description="Minimum number of z planes generated when no explicit volume plane list is supplied.",
),
"volume_output_mode": ParamSpec(
    key="volume_output_mode",
    default='focus_weighted_average',
    type="enum",
    choices=["focus_weighted_average", "integrated_projection", "z_stack"],
    ui_label="Volume output",
    group="Workflow",
    description=(
        "Return a focus-weighted average, a physical integrated projection, or "
        "retain the full z stack. Physical integrated projection requires a "
        "declared per-z physical volume basis; rerendered focus-stack contrast "
        "uses focus_weighted_average to preserve contrast units."
    ),
),
"holotomography_projection_angles_deg": ParamSpec(
    key="holotomography_projection_angles_deg",
    default=[0.0, 45.0, 90.0, 135.0],
    type="json",
    ui_label="Holotomography angles",
    group="Workflow",
    description="Projection angles used by optional holotomography-style phase-stack helpers.",
),
"holotomography_output_mode": ParamSpec(
    key="holotomography_output_mode",
    default='phase_projection_stack',
    type="enum",
    choices=["phase_projection_stack", "reconstruction_volume"],
    ui_label="Holotomography output",
    group="Workflow",
    description="Return phase projections or request a reconstruction volume helper.",
),
"save_frame_sequence": ParamSpec(
    key="save_frame_sequence",
    default=True,
    type="bool",
    ui_label="Save frame sequence",
    group="Workflow",
    description="Write rendered PNG/AVI frame outputs for the final image sequence.",
),
"save_raw_camera_video": ParamSpec(
    key="save_raw_camera_video",
    default=True,
    type="bool",
    ui_label="Save raw-camera video",
    group="Workflow",
    description=(
        "Write a raw-camera signal AVI beside the contrast-analysis AVI. "
        "The raw-camera AVI is windowed from detector counts to 8-bit without "
        "background subtraction; quantitative counts remain in raw-frame arrays."
    ),
),
"save_raw_camera_frame_sequence": ParamSpec(
    key="save_raw_camera_frame_sequence",
    default=False,
    type="bool",
    ui_label="Save raw-camera frame sequence",
    group="Workflow",
    description=(
        "Write a uint16 PNG sequence of raw detector signal frames. This is "
        "larger than the 8-bit contrast frame sequence but preserves camera "
        "count values up to the configured bit depth."
    ),
),
"save_raw_frame_views": ParamSpec(
    key="save_raw_frame_views",
    default=False,
    type="bool",
    ui_label="Save raw/ideal arrays",
    group="Workflow",
    description="Persist raw/ideal per-frame arrays in auxiliary outputs.",
),
"return_ideal_float_frames": ParamSpec(
    key="return_ideal_float_frames",
    default=False,
    type="bool",
    ui_label="Return ideal float frames",
    group="Workflow",
    description="Request ideal float-domain output in returned payload.",
),
"output_filename": ParamSpec(
    key="output_filename",
    default='outputs/syniscopy_simulation.avi',
    type="string",
    ui_label="Output filename",
    group="Workflow",
    description="Base path for generated outputs (frames, metadata, sidecars).",
),
"random_seed": ParamSpec(
    key="random_seed",
    default=None,
    type="int",
    min=0,
    max=2**63 - 1,
    ui_label="Random seed",
    group="Workflow",
    description="Seed used by stochastic model components.",
),
}

__all__ = ["DATASET_SCHEMA"]
