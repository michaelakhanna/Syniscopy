"""Noise parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


NOISE_SCHEMA: dict[str, ParamSpec] = {
"detector_spectral_response_model": ParamSpec(
    key="detector_spectral_response_model",
    default='rgb_heuristic',
    type="enum",
    choices=["rgb_heuristic", "flat", "table"],
    ui_label="Detector spectral response",
    group="Workflow",
    description="Detector response model used for generated broadband channels.",
),
"background_subtraction_method": ParamSpec(
    key="background_subtraction_method",
    default='video_median',
    type="string",
    ui_label="Background subtraction method",
    group="Imaging",
    description="Background subtraction strategy for frame normalization.",
),
"shot_noise_enabled": ParamSpec(
    key="shot_noise_enabled",
    default=True,
    type="bool",
    ui_label="Shot noise",
    group="Noise",
    description="Enable Poisson photon/electron shot noise.",
),
"drift_velocity_nm_per_s": ParamSpec(
    key="drift_velocity_nm_per_s",
    default=[0.0, 0.0, 0.0],
    type="json",
    ui_label="Drift velocity (nm/s)",
    group="Dynamics",
    description="Rigid-frame translation rate for all particles, one value per axis in nm/s.",
),
"motion_blur_enabled": ParamSpec(
    key="motion_blur_enabled",
    default=True,
    type="bool",
    ui_label="Motion blur enabled",
    group="Dynamics",
    description="Enable per-frame motion blur during rendering.",
),
"motion_blur_subsamples": ParamSpec(
    key="motion_blur_subsamples",
    default=4,
    type="int",
    min=1,
    max=256,
    ui_label="Motion blur subsamples",
    group="Dynamics",
    description="Temporal supersampling count used when motion blur is enabled.",
),
"gaussian_noise_enabled": ParamSpec(
    key="gaussian_noise_enabled",
    default=True,
    type="bool",
    ui_label="Read / Gaussian noise",
    group="Noise",
    description="Enable Gaussian read noise (controlled by read_noise_counts).",
),
"adc_quantization": ParamSpec(
    key="adc_quantization",
    default=False,
    type="bool",
    ui_label="ADC quantization",
    group="Noise",
    description="Enable quantization after final count-space accumulation.",
),
"adc_quantization_counts": ParamSpec(
    key="adc_quantization_counts",
    default=1.0,
    type="float",
    min=0.1,
    max=1e6,
    ui_label="ADC quantization step",
    group="Noise",
    description="Quantization step size in count units when ADC quantization is enabled.",
),
"background_offset_counts": ParamSpec(
    key="background_offset_counts",
    default=0.0,
    type="float",
    min=0.0,
    max=1e12,
    ui_label="Background offset (counts)",
    group="Noise",
    description=(
        "Constant additive background offset in camera counts. "
        "Its shot-noise role is controlled by background_offset_stage."
    ),
),
"background_offset_stage": ParamSpec(
    key="background_offset_stage",
    default='pre_poisson',
    type="enum",
    choices=["pre_poisson", "post_poisson_pre_gain", "post_gain"],
    ui_label="Background offset stage",
    group="Noise",
    description=(
        "Detector stage for background_offset_counts. pre_poisson contributes "
        "to Poisson shot variance; post_poisson_pre_gain shifts the mean after "
        "shot sampling but before deterministic gain maps; post_gain is an "
        "electronic bias/pedestal term."
    ),
),
"nonlinearity_calibration": ParamSpec(
    key="nonlinearity_calibration",
    default=None,
    type="string",
    ui_label="ADC nonlinearity calibration",
    group="Noise",
    description="Optional nonlinearity calibration map or profile for count-domain response.",
),
"read_noise_e": ParamSpec(
    key="read_noise_e",
    default=None,
    type="float",
    min=0.0,
    max=1e12,
    ui_label="Read noise (e-)",
    group="Noise",
    description="Read noise specified in electrons; converted to counts when provided.",
),
"camera_gain_e_per_count": ParamSpec(
    key="camera_gain_e_per_count",
    default=1.0,
    type="float",
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
"detector_qe": ParamSpec(
    key="detector_qe",
    default=1.0,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Detector QE",
    group="Noise",
    description="Generic detector quantum efficiency; neutral at 1.0 and active when detector_input_is_incident_quanta is true.",
),
"detector_input_is_incident_quanta": ParamSpec(
    key="detector_input_is_incident_quanta",
    default=False,
    type="bool",
    ui_label="Input is incident quanta",
    group="Noise",
    description="Apply detector_qe before detection noise when rendered values are incident quanta.",
),
"emccd_enabled": ParamSpec(
    key="emccd_enabled",
    default=False,
    type="bool",
    ui_label="EMCCD enabled",
    group="Noise",
    description="Enable EMCCD excess-noise variance/sampling.",
),
"emccd_excess_noise_factor": ParamSpec(
    key="emccd_excess_noise_factor",
    default=1.0,
    type="float",
    min=1.0,
    max=4.0,
    ui_label="EMCCD excess-noise factor",
    group="Noise",
    description=(
        "Raw EMCCD excess-noise input. Its meaning is declared by "
        "emccd_excess_noise_factor_basis; the detector model resolves it to a "
        "canonical shot-variance multiplier before stochastic/Fisher use."
    ),
),
"emccd_excess_noise_factor_basis": ParamSpec(
    key="emccd_excess_noise_factor_basis",
    default='variance_multiplier',
    type="enum",
    choices=["variance_multiplier", "noise_factor_std"],
    ui_label="EMCCD excess-noise basis",
    group="Noise",
    description=(
        "Basis of emccd_excess_noise_factor. Use 'variance_multiplier' for "
        "shot-variance multipliers, or 'noise_factor_std' for vendor/literature "
        "EMCCD noise factors F_n, which enter variance as F_n^2."
    ),
),
"emccd_gain": ParamSpec(
    key="emccd_gain",
    default=1.0,
    type="float",
    min=0.1,
    max=1024.0,
    ui_label="EMCCD gain",
    group="Noise",
    description="EMCCD electronic gain applied as input-referred read-noise reduction when EMCCD mode is enabled.",
),
"dark_current_e_per_pixel_per_s": ParamSpec(
    key="dark_current_e_per_pixel_per_s",
    default=0.0,
    type="float",
    min=0.0,
    ui_label="Dark current (e-/pix/s)",
    group="Noise",
    description="Detector dark-current contribution in electrons per pixel per second.",
),
"exposure_time_s": ParamSpec(
    key="exposure_time_s",
    default=1.0,
    type="float",
    min=0.0,
    ui_label="Exposure time (s)",
    group="Noise",
    description="Exposure time used when converting dark current (e-) to count offsets.",
),
"read_noise_counts": ParamSpec(
    key="read_noise_counts",
    default=1.0,
    type="float",
    min=0.0,
    max=10000.0,
    ui_label="Read noise (counts RMS)",
    group="Noise",
    description=(
        "RMS Gaussian read noise in camera counts. "
        "Convert from electrons: read_noise_counts = σ_e / camera_gain_e_per_count."
    ),
),
"saturation_level": ParamSpec(
    key="saturation_level",
    default=None,
    type="float",
    ui_label="Saturation level (counts)",
    group="Noise",
    description="Output-level saturation clamp in camera counts; None disables count-domain clipping.",
),
"saturation_e": ParamSpec(
    key="saturation_e",
    default=None,
    type="float",
    ui_label="Saturation level (e-)",
    group="Noise",
    description="Absolute saturation clamp in electrons before conversion to counts; None disables.",
),
"dark_offset_counts": ParamSpec(
    key="dark_offset_counts",
    default=0.0,
    type="float",
    ui_label="Dark/frame offset (counts)",
    group="Noise",
    description=(
        "Constant additive offset in camera counts. Its shot-noise role is "
        "controlled by dark_offset_stage."
    ),
),
"dark_offset_stage": ParamSpec(
    key="dark_offset_stage",
    default='pre_poisson',
    type="enum",
    choices=["pre_poisson", "post_poisson_pre_gain", "post_gain"],
    ui_label="Dark offset stage",
    group="Noise",
    description=(
        "Detector stage for dark_offset_counts. pre_poisson models dark/background "
        "quanta that contribute shot noise; post_gain models electronic bias."
    ),
),
"fixed_pattern_gain_std": ParamSpec(
    key="fixed_pattern_gain_std",
    default=0.0,
    type="float",
    min=0.0,
    ui_label="Fixed-pattern gain std",
    group="Noise",
    description="Std dev of random fixed-pattern gain map, zero by default.",
),
"fixed_pattern_offset_counts": ParamSpec(
    key="fixed_pattern_offset_counts",
    default=0.0,
    type="float",
    ui_label="Fixed-pattern offset std (counts)",
    group="Noise",
    description="Std dev of random fixed-pattern additive offset map, zero by default.",
),
"scan_line_noise_counts": ParamSpec(
    key="scan_line_noise_counts",
    default=0.0,
    type="float",
    min=0.0,
    ui_label="Scan-line noise (counts RMS)",
    group="Noise",
    description="Additive row-correlated noise in counts, used for rolling-shutter / ADC coupling.",
),
"hot_pixel_fraction": ParamSpec(
    key="hot_pixel_fraction",
    default=0.0,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Hot-pixel fraction",
    group="Noise",
    description="Fraction of pixels flagged as hot pixels.",
),
"clip_output_to_nonnegative": ParamSpec(
    key="clip_output_to_nonnegative",
    default=True,
    type="bool",
    ui_label="Clip output to nonnegative",
    group="Noise",
    description="Clamp final noisy counts to a nonnegative range.",
),
"noise_parameterization": ParamSpec(
    key="noise_parameterization",
    default='camera_counts',
    type="enum",
    choices=["camera_counts"],
    ui_label="Noise parameterization",
    group="Noise",
    description="Top-level noise domain; only 'camera_counts' is currently implemented.",
),
"detector_noise_input_domain": ParamSpec(
    key="detector_noise_input_domain",
    default=None,
    type="enum",
    choices=["camera_counts", "electron_count"],
    ui_label="Detector-noise input domain",
    group="Noise",
    description=(
        "Optional detector-noise input-domain override. None derives from the "
        "canonical modality."
    ),
),
"read_noise_map_mode": ParamSpec(
    key="read_noise_map_mode",
    default='replace',
    type="enum",
    choices=["replace", "add"],
    ui_label="Read-noise map mode",
    group="Noise",
    description="How per-pixel read-noise maps combine with scalar read noise.",
),
"modality_noise": ParamSpec(
    key="modality_noise",
    default={},
    type="json",
    ui_label="Modalities noise overrides",
    group="Noise",
    description="Per-modality nested overrides for noise parameters.",
),
"noise_model": ParamSpec(
    key="noise_model",
    default={},
    type="json",
    ui_label="Noise model overrides",
    group="Noise",
    description="Global grouped overrides for noise-domain parameters.",
),
"fixed_pattern_gain_map": ParamSpec(
    key="fixed_pattern_gain_map",
    default=None,
    type="string",
    ui_label="Fixed-pattern gain map",
    group="Noise",
    description="Path or inline scalar for deterministic fixed-pattern gain map.",
),
"hot_pixel_value_counts": ParamSpec(
    key="hot_pixel_value_counts",
    default=None,
    type="float",
    min=0.0,
    ui_label="Hot-pixel value (counts)",
    group="Noise",
    description="Saturation value assigned to hot pixels. None uses frame maximum.",
),
"scmos_gain_map": ParamSpec(
    key="scmos_gain_map",
    default=None,
    type="string",
    ui_label="sCMOS gain map",
    group="Noise",
    description="Optional per-pixel multiplicative gain map for sCMOS-like nonuniform gain.",
),
"fixed_pattern_offset_map": ParamSpec(
    key="fixed_pattern_offset_map",
    default=None,
    type="string",
    ui_label="Fixed-pattern offset map",
    group="Noise",
    description="Path or inline scalar for deterministic DSNU/offset map.",
),
"hot_pixel_mask": ParamSpec(
    key="hot_pixel_mask",
    default=None,
    type="string",
    ui_label="Hot-pixel mask",
    group="Noise",
    description="Path or inline scalar/array mask for deterministic hot pixels.",
),
"scmos_variance_map": ParamSpec(
    key="scmos_variance_map",
    default=None,
    type="string",
    ui_label="sCMOS variance map",
    group="Noise",
    description="Path or inline scalar/array of per-pixel read-noise variance.",
),
"scmos_read_noise_map": ParamSpec(
    key="scmos_read_noise_map",
    default=None,
    type="string",
    ui_label="sCMOS read-noise map",
    group="Noise",
    description="Path or inline scalar/array of per-pixel read-noise standard deviation.",
),
"flat_field_map": ParamSpec(
    key="flat_field_map",
    default=None,
    type="string",
    ui_label="Flat-field map",
    group="Noise",
    description="Path or scalar multiplicative gain map applied to detector output.",
),
"dark_frame_map": ParamSpec(
    key="dark_frame_map",
    default=None,
    type="string",
    ui_label="Dark-frame map",
    group="Noise",
    description=(
        "Path or scalar dark-frame offset map added in detector counts. "
        "Its shot-noise role is controlled by dark_frame_map_stage."
    ),
),
"dark_frame_map_stage": ParamSpec(
    key="dark_frame_map_stage",
    default='pre_poisson',
    type="enum",
    choices=["pre_poisson", "post_poisson_pre_gain", "post_gain"],
    ui_label="Dark-frame map stage",
    group="Noise",
    description=(
        "Detector stage for dark_frame_map. pre_poisson contributes spatial "
        "Poisson variance; post_poisson_pre_gain is before deterministic gain maps; "
        "post_gain is a deterministic bias/pedestal map."
    ),
),
}

__all__ = ["NOISE_SCHEMA"]
