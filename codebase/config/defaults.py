"""Authoritative default simulation parameters."""

from __future__ import annotations

import math
import os

from shared_constants import KNOWN_INTERNAL_PARAM_KEYS


_PHASE_QUADRATURE_RAD = 0.5 * math.pi
_PHASE_REVERSAL_RAD = math.pi
_KNOWN_INTERNAL_PARAM_KEYS = set(KNOWN_INTERNAL_PARAM_KEYS)

RUNTIME_INTERNAL_DEFAULTS = {
    "_particle_specs": None,
    "_particle_specs_fingerprint": None,
    "_return_mask_arrays": False,
    "_write_mask_files": True,
    "_substrate_pattern_layout_cache_token": None,
    "_substrate_pattern_layout_extent_nm": None,
    "_substrate_pattern_layout_rng": None,
    "_generated_spectral_channels": False,
}

# --- SIMULATION PARAMETERS ---
# This dictionary centralizes all configurable parameters for the simulation.
PARAMS = {
    # --- IMAGE & VIDEO ---
    # Linear size (both width and height) of the square output frames in pixels.
    # Must be a positive integer (e.g., 512, 1024).
    "image_size_pixels": 1024,

    # Physical side length of a single camera pixel in nanometers.
    # Must be a positive float. Typical values are ~100–600 nm depending on the
    # objective and camera pixel pitch.
    "pixel_size_nm": 244,

    # Frame rate (frames per second). The frame count is resolved as
    # int(fps * duration_seconds).
    "fps": 24,

    # Exposure time for a single frame in milliseconds. This controls the
    # temporal window over which motion blur is simulated when
    # motion_blur_enabled is True. Must satisfy:
    #
    #     0 < exposure_time_ms <= 1000 / fps
    #
    # so that the exposure window lies entirely within a single frame
    # interval.
    #
    # None (the default) means full-frame exposure: exposure_time_ms is
    # resolved at render time as 1000 / fps. This is the physically natural
    # default — the shutter is open for the entire frame interval. An explicit
    # positive float selects partial exposure (rolling shutter / fast strobe),
    # e.g. 5.0 for a 5 ms strobe at 40 fps.
    "exposure_time_ms": None,

    # Total duration of the simulated video in seconds.
    # Positive float or int. Combined with fps determines num_frames.
    "duration_seconds": 1,

    # Optional exact frame-count request. When set, run_simulation resolves
    # duration_seconds from num_frames / fps before rendering.
    "num_frames": None,

    # Optional public seed for deterministic single-run simulation paths.
    # Dataset generation overwrites this per video after validating user
    # overrides, so the same dataset seed still produces distinct videos.
    "random_seed": None,

    # Bit depth of the raw simulated frames in camera counts.
    # Supported range: 1–16, matching uint16 frame storage.
    # Common values: 12, 14, 16.
    "bit_depth": 16,

    # Filesystem path (including filename) of the encoded contrast-analysis AVI.
    # A raw-camera signal AVI is written beside it when save_raw_camera_video is
    # true, using the same stem plus "_raw_signal". The raw-camera AVI is a
    # windowed 8-bit preview of detector counts; enable save_raw_frame_views or
    # save_raw_camera_frame_sequence for quantitative raw values. The PNG frame
    # sequence is a lossless encoding of the 8-bit display/training frames.
    # Absolute or relative paths are allowed; parent directories are created if
    # they do not exist. Relative paths are resolved by Python against the
    # caller's current working directory.
    "output_filename": os.path.join(
        "outputs", "syniscopy_simulation.avi"
    ),

    # --- MASK GENERATION ---
    # Master switch for segmentation mask generation.
    #   True  -> per-particle masks are generated and saved to disk under the
    #            canonical target-specific directories.
    #   False -> no masks are generated or saved (only the frames are
    #            rendered).
    "mask_generation_enabled": True,

    # Base directory where annotation masks are written under target-specific
    # subdirectories: mask_supported/, mask_geometry/, ignore_mask/,
    # loss_weight/. Relative paths are resolved by Python against the caller's
    # current working directory.
    "mask_output_directory": os.path.join(
        "outputs", "syniscopy_masks"
    ),

    # Number of PSF rings beyond the central lobe to include in each particle
    # mask. 0 means central lobe only. 1 means central lobe plus the first
    # surrounding opposite-sign ring, 2 includes the next ring as well, etc.
    # Ring boundaries are detected from radial sign changes in the particle's
    # contrast image, so the definition is invariant to bright/dark contrast
    # reversal.
    "mask_outer_ring_count": 0,

    # Reject a single-particle mask if lobe inference would cover more than
    # this fraction of the frame. This protects training targets from flat or
    # out-of-focus contrast images being misread as full-frame particles.
    "mask_max_area_fraction": 0.25,

    # Supervision-policy target written to the canonical mask path and consumed
    # by downstream training by default:
    #   mask_supported -> geometry filtered by configured support factors
    #   mask_geometry  -> projected object and contrast-support mask before support gating
    #
    # Every run emits:
    #   mask_geometry/, mask_supported/, ignore_mask/, loss_weight/,
    #   annotation_schema.json, supervision_audit.json.
    "supervision_target": "mask_supported",
    "supervision_support_factors": None,

    # Heuristic support-factor thresholds. These factors are not calibrated
    # probabilities; they are soft plausibility/support factors in [0, 1].
    "supervision_supported_threshold": 0.2,
    "supervision_temporal_support_enabled": True,
    "supervision_signal_support_enabled": True,
    "supervision_information_support_enabled": True,
    "supervision_ambiguity_support_enabled": True,
    "supervision_crlb_xy_max_nm": None,
    "supervision_stop_when_all_temporally_unsupported": False,
    "supervision_ambiguity_distance_scale_nm": None,
    "supervision_prior_log_odds": 0.0,
    "supervision_decision_rule": "log_odds",
    "supervision_log_odds_threshold": 0.0,
    "supervision_log_odds_clip_epsilon": 1e-12,
    "supervision_score_calibration_mode": "uncalibrated_support",
    "supervision_score_calibration_parameters": None,

    # Fisher/CRLB derivative controls. The stationary-shift mode uses detector-grid
    # spatial gradients; rerendered_xy uses explicit +/- scene renders and is the
    # required mode for structured sample-environment comparisons.
    "fisher_lateral_step_nm": 5.0,
    "fisher_lateral_derivative_mode": "stationary_shift",
    "fisher_particle_index": 0,
    "fisher_likelihood_model": "mean_fisher_diagnostic",
    "sequence_fisher_enabled": False,
    "detected_quanta_derivative_target": "signed_contrast_scaled",
    "profile_fidelity_label": "model_conditional_profile",
    "tem_backend": "multislice_physical",
    "sem_backend": "monte_carlo_physical",
    "sem_source_representation": "volume",
    "sem_volume_slices": 8,
    "sem_volume_slice_thickness_nm": None,
    "sem_source_z_origin": "entry_surface_depth",
    "sem_source_z_offset_nm": 0.0,
    "fluorescence_backend": "vectorial_photophysics",
    "fluorescence_require_physical_photon_budget": False,
    "fluorescence_allow_psf_fallback": False,
    "tem_reference_status": "physics_based_unvalidated",
    "tem_reference_validation_hash": None,
    "tem_potential_source": "material_projected_inner_potential",
    "tem_objective_aperture_mrad": None,
    "sem_reference_kernel_path": None,
    "sem_reference_kernel_sha256": None,
    "fluorescence_reference_status": "physics_based_unvalidated",
    "fluorescence_reference_validation_hash": None,
    "fluorescence_blinking_rate_per_frame": 0.0,
    "fluorescence_recovery_rate_per_frame": 0.0,
    "fluorescence_bleaching_rate_per_frame": 0.0,


    # --- OPTICAL SETUP ---
    # Illumination wavelength in vacuum, in nanometers.
    # Positive float (e.g., 445, 520, 635).
    "wavelength_nm": 635,

    # Optional detector/probe wavelength override. None falls back to
    # wavelength_nm unless a modality defines a more specific canonical
    # detector wavelength.
    "probe_wavelength_nm": None,
    "optical_field_backend": "vectorial_debye",
    "polarization_model": "linear_x",
    "vectorial_detection_mode": "full_vector",
    "vectorial_polarization_rotation_deg": 0.0,
    "vectorial_pupil_samples": None,
    "vectorial_obliquity_apodization": True,
    "coverslip_aberration_model": "none",
    "coverslip_thickness_um": 170.0,
    "coverslip_design_thickness_um": 170.0,
    "coverslip_refractive_index": 1.518,
    "coverslip_design_refractive_index": 1.518,
    "coverslip_aberration_subtract_piston": True,

    # Numerical aperture (NA) of the microscope objective.
    # Positive float; must satisfy 0 < NA <= refractive_index_medium.
    "numerical_aperture": 1.2,

    # Magnification of the objective (for reference/documentation only).
    # Positive float or int (e.g., 60, 100). Not directly used in the physics
    # calculations but useful for instrument metadata.
    "magnification": 60,

    # Optional objective description recorded as instrument metadata.
    "objective_model": None,

    # Objective focal length in millimeters.
    # For a 60x objective with a 180 mm tube lens, this is typically ~3.0 mm.
    "objective_focal_length_mm": 3.0,

    # Refractive index of the sample medium (e.g., water).
    # Positive float (e.g., 1.33 for water).
    "refractive_index_medium": 1.33,

    # Refractive index of the immersion medium used with the objective.
    # Positive float (e.g., 1.518 for standard immersion oil).
    "refractive_index_immersion": 1.518,

    # --- PARTICLE OBJECTS ---
    # Canonical particle description. Each object carries motion properties
    # plus one or more spherical renderable components.
    "particles": [
        {
            "name": "gold_100nm_0",
            "motion": {
                "hydrodynamic_diameter_nm": 100.0,
                "initial_position_nm": None,
            },
            "signal_multiplier": 0.5,
            "source_multiplier": 1.0,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "Gold",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        },
        {
            "name": "gold_100nm_1",
            "motion": {
                "hydrodynamic_diameter_nm": 100.0,
                "initial_position_nm": None,
            },
            "signal_multiplier": 0.5,
            "source_multiplier": 1.0,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "Gold",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        },
    ],

    # --- BROWNIAN MOTION ---
    "temperature_K": 298.15,
    "viscosity_Pa_s": 0.00089,
    # Default is pure unconstrained Brownian motion. Use
    # "reflecting_floor_z0" for motion constrained to z >= 0 or
    # "reflecting_ceiling_z0" for motion constrained to z <= 0.
    "z_motion_constraint_model": "unconstrained",
    # Span used only when sampling initial z positions. This is not a Brownian
    # motion boundary; unconstrained Brownian motion can move outside it.
    "initial_z_span_nm": 3000.0,
    "dynamic_bayesian_enabled": False,
    "dynamic_process_noise_scale": 1.0,
    "dynamic_initial_variance_nm2": 1.0e30,
    "dynamic_include_smoothing": False,
    "rotational_diffusion_enabled": True,
    "rotational_diffusion_mode": "empirical",
    # Empirical rotational diffusion uses this fixed per-frame angular std.
    # Use rotational_diffusion_mode="stokes_einstein" for a Stokes-Einstein-
    # Debye value from particle size, viscosity, temperature, and frame rate.
    "rotational_step_std_deg": 10.0,
    # Optional bench perturbations. Drift is a rigid scene translation over
    # time; vibration is per-exposure stochastic jitter. Defaults preserve
    # the Brownian-only trajectory model.
    "drift_velocity_nm_per_s": [0.0, 0.0, 0.0],
    "vibration_jitter_std_nm": 0.0,
    "vibration_include_axial": False,

    # --- IMAGING MODEL ---
    # Selects the imaging contrast model used by the renderer.
    # Supported values:
    #   "bright_field"                 — partially coherent Köhler bright-field
    #   "partially_coherent_bright_field" — explicit Köhler bright-field model
    #   "fluorescence_widefield"       — incoherent widefield fluorescence
    #   "tirf_fluorescence"            — evanescent-excitation fluorescence
    #   "dark_field"                   — annular Köhler dark-field
    #   "zernike_phase_contrast"       — scalar phase-ring approximation
    #   "differential_phase_contrast"  — scalar and optional vectorial DPC backends
    #   "quantitative_phase"           — recovered phase image
    #   "off_axis_holography"          — DHM fringe image
    #   "ricm"                         — reflection interference contrast
    #   "interferometric"              — standard iSCAT: I = |E_ref + E_sca|²
    #   "tem_phase_contrast"           — CTF-based TEM phase contrast
    #   "sem_secondary_electron"       — probe-blurred SEM secondary-electron yield
    #   "coherent_bright_field"       — transmitted-reference coherent bright-field
    #   "coherent_dark_field"          — coherent zero-order-blocked dark-field
    "imaging_model": "bright_field",

    # Optional independent spectral/channel rendering. Leave as None for the
    # ordinary single-channel path. If set, each entry can be a wavelength
    # number or a dict overriding channel-specific parameters, for example:
    # [{"name": "green", "wavelength_nm": 532}, {"name": "red", "wavelength_nm": 640}].
    "channels": None,
    "spectral_integration_model": "single_wavelength",
    "illumination_spectrum_center_nm": 550.0,
    "illumination_spectrum_fwhm_nm": 40.0,
    "illumination_spectrum_num_samples": 5,
    "broadband_wavelengths_nm": None,
    "broadband_weights": None,
    "detector_spectral_response_model": "rgb_heuristic",
    "allow_broadband_overwrite_channels": False,
    # Optional matched-modality packet generation for dataset runs. When set to
    # two or more imaging-model names, dataset generation renders the same
    # latent scene through each listed modality and stores a counterfactual
    # packet sidecar.
    "matched_modalities": None,
    # Multichannel/spectral video output mode.
    #
    # Single-wavelength simulations ignore this and remain grayscale.
    # Multichannel simulations can write:
    #   "rgb"      -> one RGB visualization video at output_filename
    #   "channels" -> per-channel grayscale sidecar videos only
    #   "both"     -> RGB video plus per-channel grayscale sidecars
    #   "none"     -> no video files, returned arrays only
    "multichannel_output_mode": "rgb",

    # Optional sidecar directory for multichannel_output_mode in {"channels", "both"}.
    # If None, sidecars are written beside output_filename using
    # "<output_stem>_channels/".
    "multichannel_sidecar_directory": None,

    # Optional particle-volume and optical-sectioning helpers. Non-default modes
    # are explicit configured outputs; ordinary simulations remain single-plane.
    "scene_dimensionality": "single_plane_particle_scene",
    "volumetric_imaging_mode": "single_plane",
    "volumetric_z_planes_nm": None,
    "volumetric_z_range_nm": 1000.0,
    "volumetric_z_step_nm": 250.0,
    "volumetric_z_count": 5,
    "volume_output_mode": "integrated_projection",
    "confocal_pinhole_sigma_nm": 350.0,
    "light_sheet_center_z_nm": 0.0,
    "light_sheet_sigma_nm": 500.0,
    "holotomography_projection_angles_deg": [0.0, 45.0, 90.0, 135.0],
    "holotomography_output_mode": "phase_projection_stack",

    # Fluorescence material rendering. Emitter density and material-specific
    # fluorescence/electron properties live on particle components under
    # particles[*].components[*].material_properties. The global keys below
    # describe microscope/detector settings, not emitter density.
    "fluorescence_quantum_yield": 0.5,
    "fluorescence_excitation_scale": 1.0,
    # Prefer physical PSF width when supplied; the legacy pixel value is kept
    # for existing configs and is converted through the active canvas pitch.
    "fluorescence_emission_psf_sigma_nm": None,
    "fluorescence_emission_psf_sigma_px": 1.0,
    "fluorescence_background": 0.0,
    "fluorescence_photon_count_scale": 500.0,
    "fluorescence_photons_per_fluorophore_per_frame": None,
    "fluorescence_collection_efficiency": 1.0,
    "fluorescence_detector_qe": 1.0,
    "fluorescence_spectral_bandwidth_nm": 40.0,
    "fluorescence_excitation_wavelength_nm": 488.0,
    "fluorescence_emission_wavelength_nm": 520.0,
    "fluorescence_photobleach_tau_frames": None,
    "fluorescence_sample_environment_excitation_modulation_gain": 0.25,
    "fluorescence_sample_environment_autofluorescence_gain": 1.0,
    # --- COMPLEX PSF & SCATTERING CALCULATION ---
    "psf_oversampling_factor": 2,
    "pupil_samples": 512,
    # Axial PSF/cache grid settings.
    "z_stack_range_nm": 30500,
    "z_stack_step_nm": 50,
    # Safety cap for automatically generated optical iPSF z stacks. Source-map
    # modalities that do not need optical PSFs skip this path entirely.
    "max_psf_z_slices": 4096,
    "shared_psf_z_grid_enabled": False,

    # --- PSF PLACEMENT & PADDING ---
    "psf_intensity_fraction_threshold": 1e-4,

    # --- ABERRATIONS & PUPIL FUNCTION ---
    "spherical_aberration_strength": 0.25,
    "apodization_factor": 1.8,
    "random_aberration_strength": 1.5,

    # --- INTERFERENCE, NOISE & BACKGROUND SUBTRACTION ---
    "reference_field_amplitude": 1,

    # Average background level in camera counts/ADU. This controls the
    # reference-arm brightness (interferometric/bright-field) or the stray-
    # light pedestal (dark-field via dark_field_background_count).
    "background_intensity": 100,

    # ---- Canonical counts-domain camera-noise model (camera_noise.py) ----
    #
    # All noise parameters below are in camera counts/ADU unless the name
    # explicitly says "_electrons". camera_noise.py is the canonical noise
    # implementation used by rendering, supervision, and metadata.
    #
    # Master toggles.
    "shot_noise_enabled": True,      # Poisson photon/electron shot noise
    "gaussian_noise_enabled": True,  # Gaussian read/thermal noise
    #
    # Camera conversion factor: detected photoelectrons per output ADU count.
    # This is the primary calibration parameter for a given camera / gain
    # setting. For real camera data it can be estimated with
    # camera_noise.calibrate_camera_gain_e_per_count_from_video().
    # Typical scientific CMOS: 0.5–5 e-/count; sCMOS at medium gain: ~1–3.
    # The default 1.0 preserves the cleanest statistical model (one Poisson
    # draw per count unit). Increase to simulate a low-gain / high-dynamic-
    # range regime where each count represents many electrons.
    "camera_gain_e_per_count": 1.0,
    "detector_qe": 1.0,
    "detector_input_is_incident_quanta": False,
    "emccd_enabled": False,
    "emccd_gain": 1.0,
    "emccd_excess_noise_factor": 1.0,
    "read_noise_e": None,
    "dark_current_e_per_pixel_per_s": 0.0,
    "exposure_time_s": 1.0,
    "saturation_level": None,
    "saturation_e": None,
    "adc_quantization": False,
    "adc_quantization_counts": 1.0,
    "background_offset_counts": 0.0,
    "fixed_pattern_gain_map": None,
    "fixed_pattern_offset_map": None,
    "scmos_gain_map": None,
    "hot_pixel_mask": None,
    "scmos_variance_map": None,
    "scmos_read_noise_map": None,
    "nonlinearity_calibration": None,
    "flat_field_map": None,
    "dark_frame_map": None,
    #
    # RMS Gaussian read noise in camera counts. Real cameras normally have a
    # nonzero readout floor; set this to 0 only for an idealized detector.
    # Convert from electrons: read_noise_counts = σ_e / gain.
    "read_noise_counts": 1.0,
    #
    # Constant offset added to all pixels before Poisson sampling (dark
    # current or bias pedestal in counts). Usually 0 for background-corrected
    # data.
    "dark_offset_counts": 0.0,
    #
    # Per-pixel multiplicative gain non-uniformity: σ of the zero-mean Gaussian
    # fractional deviation applied to each pixel. 0 disables.
    "fixed_pattern_gain_std": 0.0,
    #
    # Per-pixel additive offset non-uniformity: σ of the zero-mean Gaussian
    # offset (counts) applied to each pixel. 0 disables.
    "fixed_pattern_offset_counts": 0.0,
    #
    # Hot-pixel fraction: fraction of pixels that are permanently saturated.
    # 0 disables. hot_pixel_value_counts=None means use the frame maximum.
    "hot_pixel_fraction": 0.0,
    "hot_pixel_value_counts": None,
    #
    # Row-correlated scan-line noise: σ of per-row additive noise (counts).
    # Models rolling-shutter / ADC row-coupling artefacts. 0 disables.
    "scan_line_noise_counts": 0.0,
    #
    # Clip output to non-negative counts (true for physical cameras).
    "clip_output_to_nonnegative": True,
    #
    # Noise parameterization identifier consumed by camera_noise_metadata().
    "noise_parameterization": "camera_counts",
    # Optional detector-noise input-domain override. None derives from the
    # canonical modality: electron modalities use electron_count, all others
    # use camera_counts.
    "detector_noise_input_domain": None,
    # How per-pixel read-noise maps combine with scalar read noise.
    "read_noise_map_mode": "replace",
    #
    # Optional per-modality camera-noise overrides consumed by camera_noise.py, e.g.
    # {"sem_secondary_electron": {"scan_line_noise_counts": 2.0}}.
    "modality_noise": {},

    # Optional grouped overrides for the same canonical counts-domain keys above.
    # Values here override the flat defaults before modality_noise is applied.
    # Supported nested keys are the camera-noise controls in this section, for
    # example shot_noise_enabled, gaussian_noise_enabled,
    # camera_gain_e_per_count, read_noise_counts, dark_offset_counts,
    # fixed_pattern_gain_std, fixed_pattern_gain_map, fixed_pattern_offset_counts,
    # fixed_pattern_offset_map, scmos_gain_map, hot_pixel_fraction, hot_pixel_mask,
    # hot_pixel_value_counts, scmos_variance_map, scmos_read_noise_map,
    # flat_field_map, dark_frame_map,
    # scan_line_noise_counts,
    # clip_output_to_nonnegative, and noise_parameterization.
    "noise_model": {},

    "background_subtraction_method": "video_median",
    "save_raw_camera_video": True,
    "save_raw_camera_frame_sequence": False,
    # Dataset generation writes background-subtracted final frames as the
    # canonical PNG frame sequence. Set True to additionally save raw
    # signal/reference/final frame arrays as compressed NPZ audit artifacts.
    #
    # PNG frame sequences are lossless encodings of 8-bit display/training
    # frames. AVI is kept as a compact preview because temporal video codecs can
    # smooth noisy microscopy frames. Enable save_raw_frame_views to preserve
    # quantitative raw/ideal arrays.
    "save_frame_sequence": True,
    "save_raw_frame_views": False,
    "return_ideal_float_frames": False,

    # --- EMPIRICAL BACKGROUND / SHADING FIELD ---
    # Optional low-frequency nuisance field for spatially correlated background
    # structure. This models residual flat-field / dark-field variation,
    # illumination inhomogeneity, detector offset nonuniformity, and slow
    # substrate/background variation after ordinary correction. It is not an
    # out-of-focus-particle or fluorescence-fluctuation model.
    "empirical_background_enabled": False,
    "empirical_background_model": "multiscale_gaussian_field",
    "empirical_background_relative_std": 0.03,
    "empirical_background_scales_px": [16.0, 64.0, 256.0],
    "empirical_background_scale_weights": [0.4, 0.35, 0.25],
    "empirical_background_gradient_relative_strength": 0.02,

    # --- DARK-FIELD COUNT SCALING ---
    # Dark-field has no reference beam, so the base-class
    # ``background_final * |E_sca|^2 / |E_ref|^2`` formula does not apply.
    # ``dark_field_illumination_count`` sets the multiplicative scale
    # converting dimensionless |E_sca|^2 into detector counts (defaults to
    # ``background_intensity`` so dark-field peaks land at a comparable
    # fraction of the dynamic range to the other modalities' reference
    # brightness).  ``dark_field_background_count`` is a small pedestal
    # representing residual stray light + dark current; it gives read noise
    # a non-zero baseline to fluctuate around far from any particle, which
    # otherwise gets half-clipped to zero by the uint16 cast at the end of
    # the rendering loop.  Set to 0 to recover the canonical zero-baseline
    # behaviour explicitly.
    "dark_field_illumination_count": 100,
    "dark_field_background_count": 5,

    # --- MODALITY-SPECIFIC PHYSICS KNOBS ---
    "kohler_coherence_factor": 0.7,
    "kohler_source_samples": 19,
    "annular_dark_field_source_samples": 24,
    "annular_dark_field_inner_sigma": 1.02,
    "annular_dark_field_outer_sigma": 1.08,
    "dark_field_stop_radius_fraction": 0.35,
    "dark_field_field_gain": 1.0,
    "dark_field_sample_environment_edge_gain": 0.02,
    "dark_field_sample_environment_scatter_pedestal": 0.0,
    "bright_field_sample_environment_gain": 1.0,
    "bright_field_sample_environment_phase_gain": 0.05,
    "zernike_phase_ring_gain": 0.35,
    "zernike_phase_bias": 1.0,
    "zernike_model": "pupil_phase_ring",
    "zernike_phase_ring_inner_fraction": 0.0,
    "zernike_phase_ring_outer_fraction": 0.15,
    "zernike_phase_ring_shift_rad": _PHASE_QUADRATURE_RAD,
    "zernike_phase_ring_amplitude": 1.0,
    "dpc_channel_model": "vectorial_debye_asymmetric_illumination",
    "dpc_transfer_model": "pupil_half_plane_intensity",
    "dpc_output_channel": "x",
    "dpc_intensity_gain": 1.0,
    "dpc_intensity_gain_x": 1.0,
    "dpc_intensity_gain_y": 1.0,
    "dpc_phase_gradient_gain": 2500.0,
    "dpc_phase_gradient_gain_x": 2500.0,
    "dpc_phase_gradient_gain_y": 2500.0,
    "qpi_visibility": 1.0,
    "qpi_detected_quanta_per_pixel": None,
    "qpi_phase_to_count_scale": 100.0,
    # Optional QPI phase-domain calibration noise. None means infer phase shot
    # noise from qpi_visibility and the detected-quanta budget.
    "qpi_phase_noise_std_rad": None,
    "ricm_interface_reflection_coefficient": 0.20,
    "ricm_particle_reflection_coefficient": 0.04,
    "ricm_interface_phase_shift_rad": _PHASE_REVERSAL_RAD,
    "ricm_interface_reflection_model": "fresnel",
    "ricm_thinfilm_layers": [],
    "ricm_particle_reflection_model": "fresnel",
    "ricm_interface_medium_material": "water",
    "ricm_interface_substrate_material": "glass",
    "ricm_particle_medium_material": "water",
    # None uses the primary particle component material. Set a material label
    # here only when intentionally modeling a different particle-interface material.
    "ricm_particle_material": None,
    "ricm_wavelength_nm": 532.0,
    "ricm_gap_nm": 0.0,
    "ricm_use_particle_z_as_gap": True,
    "tirf_penetration_depth_nm": 120.0,
    "tirf_use_angle_derived_penetration_depth": False,
    "tirf_prism_refractive_index": 1.518,
    "tirf_sample_refractive_index": 1.333,
    "tirf_incident_angle_deg": 66.0,
    "tirf_particle_height_nm": 0.0,
    "tirf_height_offset_nm": 0.0,
    "tirf_effective_numerical_aperture": None,
    # iSCAT calibration knobs. Defaults preserve the renderer-provided scalar
    # reference and collected scattered field used by the shared diagnostic
    # profiles. Native iSCAT validation profiles can opt into Fresnel reference
    # scaling and high-NA dipole collection without special-casing the shared
    # cross-modality ranking table.
    "iscat_reference_model": "fresnel",
    "iscat_reference_medium_material": "water",
    "iscat_reference_substrate_material": "glass",
    "iscat_reference_amplitude_scale": 1.0,
    "iscat_reference_phase_rad": 0.0,
    "iscat_reference_coefficient": 1.0,
    "iscat_reference_normalize_fresnel_phase_only": False,
    "iscat_collection_model": "dipole_high_na",
    "iscat_collection_reference_fraction": 1.0,
    "off_axis_fringe_period_px": 10.0,
    "off_axis_fringe_angle_rad": 0.0,
    "off_axis_reference_amplitude_scale": 1.0,
    "tem_acceleration_kV": 300.0,
    "tem_model": "multislice_physical",
    "tem_multislice_slices": 8,
    "tem_slice_thickness_nm": 5.0,
    "tem_Cs_mm": 0.5,
    "tem_partial_coherence_alpha_mrad": 0.1,
    "tem_defocus_nm": None,
    "tem_pixel_size_pm": None,
    # None uses the acceleration-voltage-derived electron interaction constant.
    "tem_phase_shift_per_volt_nm": None,
    # Synthetic weak-phase source scale. Values below 1.0 keep thick/high-MIP
    # demo particles inside the linear weak-phase regime.
    "tem_projected_potential_scale": 1.0,
    "tem_filter_guard_pixels": 64,
    "tem_sample_environment_potential_scale": 1.0e-4,
    "tem_dose_per_pixel": 100.0,
    "sem_acceleration_kV": 5.0,
    "sem_model": "physical_electron_transport",
    "sem_interaction_volume_nm": 30.0,
    "sem_detector_direction_xy": [1.0, 0.0],
    "sem_topography_contrast_gain": 0.0,
    # Prefer physical probe width when supplied; the legacy pixel value is kept
    # for existing configs and is converted through the active canvas pitch.
    "sem_probe_sigma_nm": None,
    "sem_probe_sigma_pixels": 1.0,
    "sem_filter_guard_pixels": None,
    "sem_edge_contrast_gain": 10.0,
    "sem_bulk_contrast_gain": 1.0,
    "sem_baseline_yield": 0.05,
    "sem_sample_environment_edge_gain": 10.0,
    "sem_electrons_per_pixel": 1000.0,
    "sem_beam_current_nA": 0.0,
    "sem_dwell_time_us": 0.0,
    "sem_detector_takeoff_angle_deg": 45.0,
    "sem_detector_acceptance": 1.0,
    "sem_escape_depth_nm": 20.0,
    "sem_backscatter_fraction": 0.05,
    "sem_transport_material_scale": 1.0,
    "sem_transport_source_exponent": 1.0,
    "sem_transport_topography_exponent": 1.0,
    "sem_monte_carlo_trajectories": 4096,
    "sem_monte_carlo_steps": 64,
    "sem_monte_carlo_seed": None,
    "sem_monte_carlo_step_nm": None,
    "sem_monte_carlo_range_nm": None,
    "sem_monte_carlo_scatter_std_deg": 8.0,
    "sem_monte_carlo_kernel_size_px": None,
    "sem_physical_max_steps": 2048,
    "sem_physical_energy_cutoff_keV": 0.05,
    "sem_physical_elastic_model": "screened_rutherford",
    "sem_reference_material": "default",
    "sem_reference_geometry": "normal",
    "sem_reference_source_depth_nm": 0.0,
    "sem_reference_incident_angle_deg": 0.0,

    # --- MOTION BLUR ---
    "motion_blur_enabled": True,
    "motion_blur_subsamples": 4,

    # --- SAMPLE ENVIRONMENT / MOUNTING INTERFACE ---
    # A sample environment is everything in the rendered scene that is not the
    # particle: mounting interface, surrounding medium, and any pattern overlay.
    "sample_environment_enabled": True,
    "sample_environment_pattern_enabled": False,
    "medium_material": "water",
    "mounting_interface_material": "glass",
    "bulk_substrate_material": "glass",
    "mounting_interface_thickness_nm": 170000.0,
    "sample_environment_pattern_material": None,
    # Pattern values supported by both optical rendering and Brownian exclusion:
    # none, gold_holes, nanopillars, fiducial_dots, grid_bars, holey_carbon,
    # microfluidic_walls, patterned_coverslip.
    "sample_environment_pattern": "none",

    # Lateral-exclusion boundary condition for Brownian steps that would land
    # inside a solid pattern feature. ``"reflection"`` is the physical
    # hard-wall condition: the step is reflected across the boundary normal,
    # preserving Brownian step statistics and uniform equilibrium density in
    # the fluid region. ``"projection"`` clamps the proposed endpoint to the
    # nearest valid fluid point; it is faster but truncates steps near walls
    # and therefore underestimates apparent diffusion close to features.
    "sample_environment_exclusion_method": "reflection",

    # Contrast evolution model for the substrate pattern over the duration of
    # the video.
    "sample_environment_pattern_contrast_model": "static",
    "sample_environment_pattern_contrast_amplitude": 0.0,

    # Substrate/background preset used when substrate pattern rendering is enabled.
    "sample_environment_pattern_preset": "default_gold_holes",

    # Geometry and optical-intensity parameters for the substrate pattern.
    "sample_environment_pattern_dimensions": {
        # Gold-film-with-holes defaults
        "hole_diameter_um": 15.0,
        "hole_edge_to_edge_spacing_um": 2.0,
        "hole_depth_nm": 20.0,
        "hole_intensity_factor": 0.7,
        "gold_intensity_factor": 1.0,
        "carbon_film_thickness_nm": 20.0,

        # Nanopillar defaults
        "pillar_diameter_um": 1.0,
        "pillar_edge_to_edge_spacing_um": 2.0,
        "pillar_height_nm": 20.0,
        "pillar_intensity_factor": 1.3,
        "background_intensity_factor": 1.0,

        # Defaults for additional public sample-environment pattern families.
        "fiducial_dot_diameter_um": 0.5,
        "fiducial_dot_edge_to_edge_spacing_um": 2.0,
        "fiducial_dot_pitch_um": 5.0,
        "fiducial_dot_intensity_factor": 1.5,
        "fiducial_background_intensity_factor": 1.0,
        "grid_pitch_um": 5.0,
        "grid_bar_width_um": 0.5,
        "grid_bar_intensity_factor": 1.25,
        "grid_background_intensity_factor": 1.0,
        "microfluidic_channel_pitch_um": 10.0,
        "microfluidic_wall_width_um": 1.0,
        "microfluidic_wall_orientation": "vertical",
        "microfluidic_wall_intensity_factor": 1.2,
        "microfluidic_channel_intensity_factor": 1.0,
        "coverslip_patch_diameter_um": 5.0,
        "coverslip_patch_edge_to_edge_spacing_um": 5.0,
        "coverslip_patch_pitch_um": 10.0,
        "coverslip_patch_intensity_factor": 1.08,
        "coverslip_background_intensity_factor": 1.0,
        "dot_height_nm": 20.0,
        "bar_height_nm": 20.0,
        "wall_height_nm": 20.0,
        "coverslip_patch_height_nm": 170000.0,
    },

    # Randomization controls for substrate pattern imperfections.
    #
    # sample_environment_pattern_randomization_enabled:
    #   - False:
    #       The substrate pattern is perfectly periodic and features are perfect
    #       circles with no jitter or distortion.
    #   - True:
    #       Each feature is jittered and slightly distorted according to the
    #       two parameters below. The same randomized layout is used both for
    #       optical background generation and for Brownian exclusion geometry.
    "sample_environment_pattern_randomization_enabled": True,

    # Standard deviation of the positional jitter applied independently to
    # each feature center, in nanometers. This is converted internally to
    # micrometers and used to draw Gaussian offsets (dx, dy) ~ N(0, sigma^2).
    # Reasonable values are on the order of tens to a few hundred nanometers.
    "sample_environment_pattern_position_jitter_std_nm": 50.0,

    # Dimensionless shape regularity parameter in [0.0, 1.0]:
    #   1.0 -> perfectly regular circular features (no shape distortion).
    #   0.0 -> maximum allowed distortion (bounded internally so radii remain
    #          physically reasonable, e.g., not less than ~50% of nominal).
    #
    # Internally this is mapped to a fractional radius distortion:
    #   distortion_frac = max_distortion_frac * (1 - shape_regularity)
    # and per-feature semi-axes are drawn as:
    #   r_x = nominal_radius * (1 + delta_x)
    #   r_y = nominal_radius * (1 + delta_y)
    # with delta_x, delta_y ~ Uniform(-distortion_frac, distortion_frac).
    # The 0.73 default yields <= 6.75% semi-axis distortion under the 25%
    # distortion cap, a conservative heuristic for slight fabrication irregularity.
    "sample_environment_pattern_shape_regularity": 0.73,

    # --- EDGE PERTURBATION MODEL FOR SUBSTRATE FEATURES ---
    # Maximum relative radial deviation for per-feature edge perturbations.
    #
    # Semantics:
    #   - This parameter controls the strength of local boundary roughness for
    #     individual nanohole-array features in 'gold_holes'.
    #   - The perturbation is expressed as a fractional deviation δ(θ) of the
    #     baseline radius as a function of angle θ, so that:
    #
    #         r_boundary(θ) = r_baseline(θ) * (1 + δ(θ))
    #
    #   - The internal sampling strategy ensures that, in typical cases,
    #     |δ(θ)| <= sample_environment_pattern_edge_perturbation_max_rel_radius across
    #     all angles, so the perturbed radius remains within a modest band
    #     around the underlying circle/ellipse.
    #
    # Interaction with sample_environment_pattern_shape_regularity:
    #   - The effective amplitude used per layout is:
    #
    #         effective_amp = sample_environment_pattern_edge_perturbation_max_rel_radius
    #                         * (1 - sample_environment_pattern_shape_regularity)
    #
    #     so that:
    #       * sample_environment_pattern_shape_regularity = 1.0 -> perfectly smooth edges
    #         (no edge perturbation regardless of this max parameter).
    #       * sample_environment_pattern_shape_regularity = 0.0 -> full amplitude.
    #
    # Setting this parameter to 0.0 disables edge perturbations entirely and
    # yields smooth circular/elliptical boundaries.
    #
    # Recommended defaults:
    #   - Values in the range 0.05–0.12 (5–12%) produce visually apparent,
    #     heuristic edge irregularities for nanoholes. Calibrated fabrication
    #     realism requires explicit pattern provenance or fitted parameters.
    "sample_environment_pattern_edge_perturbation_max_rel_radius": 0.12,

    # Number of angular modes used in the edge perturbation series δ(θ).
    #
    # Semantics:
    #   - δ(θ) is represented as a short cosine series:
    #
    #         δ(θ) = Σ_{k=1..K} A_k * cos(k θ + φ_k)
    #
    #     where K = sample_environment_pattern_edge_perturbation_mode_count.
    #   - Each feature gets its own random set of coefficients {A_k, φ_k},
    #     sampled once per layout build using the same NumPy RNG as the rest
    #     of the geometry randomization.
    #
    # Performance:
    #   - K is kept small (default 3) so that classification and projection
    #     cost per point remains modest. For each candidate feature, a handful
    #     of cosine evaluations are added to the existing ellipse logic.
    #
    # If this is set to 0, the edge perturbation model is disabled even if
    # sample_environment_pattern_edge_perturbation_max_rel_radius is non-zero.
    "sample_environment_pattern_edge_perturbation_mode_count": 3,

    # --- ROUGHNESS / SPECKLE MODEL FOR SAMPLE INTERFACE ---
    # Roughness is a separate multiplicative interface perturbation used by the
    # coherent-background pathway. It is disabled by default and can be enabled
    # independently of periodic pattern geometry.
    "sample_environment_pattern_roughness_model": "none",
    # Supported models: "none", "static", "flicker", "source_matched".
    # "source_matched" expects a user-supplied complex or real field via
    # sample_environment_pattern_roughness_source.
    "sample_environment_pattern_roughness_source": None,
    # How roughness perturbations couple into source-matched channels.
    # "independent": legacy behavior, scale source maps by |roughness|².
    # "coherent_amplitude": apply roughness as a complex field amplitude to
    # per-particle source maps, enabling phase-aware channel coupling.
    # "field_weighted": couple source maps with both roughness intensity and
    # normalized local scene amplitude.
    # "scene_weighted": modulate all source maps by the same normalized scene
    # envelope derived from the channel field intensity.
    # "channel_weighted": use channel-aware envelopes based on vectorial detection
    # mode and per-particle scattered-field components where available.
    "sample_environment_pattern_roughness_source_coupling": "channel_weighted",
    "sample_environment_pattern_roughness_amplitude": 0.0,
    "sample_environment_pattern_roughness_correlation_pixels": 4.0,
    "sample_environment_pattern_roughness_phase_std": 0.0,
}


__all__ = [
    "KNOWN_INTERNAL_PARAM_KEYS",
    "PARAMS",
    "RUNTIME_INTERNAL_DEFAULTS",
    "_KNOWN_INTERNAL_PARAM_KEYS",
    "_PHASE_QUADRATURE_RAD",
    "_PHASE_REVERSAL_RAD",
]
