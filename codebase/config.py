import math
import numbers
import os

_PHASE_QUADRATURE_RAD = 0.5 * math.pi
_PHASE_REVERSAL_RAD = math.pi
_KNOWN_INTERNAL_PARAM_KEYS = {
    "_camera_noise_static_seed",
    "_particle_specs",
    "_resolved_particles",
    "_resolved_primary_component_refractive_indices",
    "_resolved_particle_material_properties",
    "_resolved_particle_material_properties_metadata",
    "_return_mask_arrays",
    "_write_mask_files",
    "_substrate_pattern_layout_cache_token",
    "_substrate_pattern_layout_extent_nm",
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

    # Filesystem path (including filename) of the final encoded AVI preview video.
    # The lossless PNG frame sequence is the canonical training/inference
    # artifact.
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


    # --- OPTICAL SETUP ---
    # Illumination wavelength in vacuum, in nanometers.
    # Positive float (e.g., 445, 520, 635).
    "wavelength_nm": 635,

    # Optional detector/probe wavelength override. None falls back to
    # wavelength_nm unless a modality defines a more specific canonical
    # detector wavelength.
    "probe_wavelength_nm": None,

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
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "Gold",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
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
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "Gold",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
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
    #   "differential_phase_contrast"  — scalar DPC phase-gradient approximation
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

    # Fluorescence material rendering. Emitter density and material-specific
    # fluorescence/electron properties live on particle components under
    # particles[*].components[*].material_properties. The global keys below
    # describe microscope/detector settings, not emitter density.
    "fluorescence_quantum_yield": 0.5,
    "fluorescence_excitation_scale": 1.0,
    "fluorescence_emission_psf_sigma_px": 1.0,
    "fluorescence_background": 0.0,
    "fluorescence_photon_count_scale": 500.0,
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
    #
    # Optional per-modality camera-noise overrides consumed by camera_noise.py, e.g.
    # {"sem_secondary_electron": {"scan_line_noise_counts": 2.0}}.
    "modality_noise": {},

    # Optional grouped overrides for the same canonical counts-domain keys above.
    # Values here override the flat defaults before modality_noise is applied.
    # Supported nested keys are the camera-noise controls in this section, for
    # example shot_noise_enabled, gaussian_noise_enabled,
    # camera_gain_e_per_count, read_noise_counts, dark_offset_counts,
    # fixed_pattern_gain_std, fixed_pattern_offset_counts, hot_pixel_fraction,
    # hot_pixel_value_counts, scan_line_noise_counts,
    # clip_output_to_nonnegative, and noise_parameterization.
    "noise_model": {},

    "background_subtraction_method": "video_median",
    # Dataset generation writes background-subtracted final frames as the
    # canonical PNG frame sequence. Set True to additionally save raw
    # signal/reference/final frame arrays as compressed NPZ audit artifacts.
    #
    # Lossless PNG frame sequences are the canonical training artifact. AVI is
    # kept as a compact preview because temporal video codecs can smooth noisy
    # microscopy frames.
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
    "annular_dark_field_inner_sigma": 1.05,
    "annular_dark_field_outer_sigma": 1.30,
    "dark_field_stop_radius_fraction": 0.35,
    "dark_field_field_gain": 1.0,
    "dark_field_sample_environment_edge_gain": 0.02,
    "dark_field_sample_environment_scatter_pedestal": 0.0,
    "bright_field_sample_environment_gain": 1.0,
    "bright_field_sample_environment_phase_gain": 0.05,
    "zernike_phase_ring_gain": 0.35,
    "zernike_phase_bias": 1.0,
    "zernike_phase_ring_shift_rad": _PHASE_QUADRATURE_RAD,
    "dpc_phase_gradient_gain": 2500.0,
    "qpi_phase_to_count_scale": 100.0,
    # Optional QPI phase-domain calibration noise. None means use the canonical
    # counts-domain camera-noise propagation.
    "qpi_phase_noise_std_rad": None,
    "ricm_interface_reflection_coefficient": 0.20,
    "ricm_particle_reflection_coefficient": 0.04,
    "ricm_interface_phase_shift_rad": _PHASE_REVERSAL_RAD,
    "ricm_interface_reflection_model": "param",
    "ricm_particle_reflection_model": "param",
    "ricm_interface_medium_material": "water",
    "ricm_interface_substrate_material": "glass",
    "ricm_particle_medium_material": "water",
    # None uses the primary particle component material. Set a material label
    # here only when intentionally modeling a different particle-interface material.
    "ricm_particle_material": None,
    "ricm_wavelength_nm": 532.0,
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
    "iscat_reference_model": "renderer",
    "iscat_reference_medium_material": "water",
    "iscat_reference_substrate_material": "glass",
    "iscat_reference_amplitude_scale": 1.0,
    "iscat_reference_phase_rad": 0.0,
    "iscat_reference_coefficient": 1.0,
    "iscat_reference_normalize_fresnel_phase_only": False,
    "iscat_collection_model": "scalar",
    "iscat_collection_reference_fraction": 1.0,
    "off_axis_fringe_period_px": 10.0,
    "off_axis_fringe_angle_rad": 0.0,
    "tem_acceleration_kV": 300.0,
    "tem_Cs_mm": 0.5,
    "tem_partial_coherence_alpha_mrad": 0.1,
    "tem_defocus_nm": None,
    "tem_pixel_size_pm": None,
    "tem_sample_environment_potential_scale": 1.0e-4,
    "tem_dose_per_pixel": 100.0,
    "sem_acceleration_kV": 5.0,
    "sem_probe_sigma_pixels": 1.0,
    "sem_edge_contrast_gain": 10.0,
    "sem_bulk_contrast_gain": 1.0,
    "sem_baseline_yield": 0.05,
    "sem_sample_environment_edge_gain": 10.0,
    "sem_electrons_per_pixel": 1000.0,

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
    # Pattern values supported by both optical rendering and Brownian exclusion:
    # none, gold_holes, nanopillars.
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
        "hole_intensity_factor": 0.7,
        "gold_intensity_factor": 1.0,

        # Nanopillar defaults
        "pillar_diameter_um": 1.0,
        "pillar_edge_to_edge_spacing_um": 2.0,
        "pillar_intensity_factor": 1.3,
        "background_intensity_factor": 1.0,
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
}


def validate_params(
    params: dict,
    *,
    allowed_extra_keys: set[str] | None = None,
    allowed_internal_keys: set[str] | None = None,
) -> None:
    """
    Validate the public PARAMS surface.

    Syniscopy v1 has one canonical public key per concept. Unknown keys raise
    immediately so aliases and typo-driven configuration drift do not enter
    generated datasets or manuscript artifacts.
    """
    allowed = set(PARAMS)
    if allowed_extra_keys:
        allowed.update(str(key) for key in allowed_extra_keys)
    if allowed_internal_keys:
        allowed.update(
            str(key)
            for key in allowed_internal_keys
            if str(key) in _KNOWN_INTERNAL_PARAM_KEYS
        )
    unknown = sorted(str(key) for key in params if str(key) not in allowed)
    if unknown:
        preview = ", ".join(repr(key) for key in unknown[:8])
        if len(unknown) > 8:
            preview += f", ... ({len(unknown)} total)"
        raise ValueError(
            "Unknown simulation parameter key(s): "
            f"{preview}. Use the canonical keys documented in config.PARAMS."
        )

    def _finite_float(key: str, *, positive: bool = False, nonnegative: bool = False) -> float:
        value = float(params.get(key, PARAMS.get(key)))
        if not math.isfinite(value):
            raise ValueError(f"PARAMS['{key}'] must be finite; got {value}.")
        if positive and value <= 0.0:
            raise ValueError(f"PARAMS['{key}'] must be positive; got {value}.")
        if nonnegative and value < 0.0:
            raise ValueError(f"PARAMS['{key}'] must be non-negative; got {value}.")
        return value

    def _positive_int(key: str) -> int:
        value = params.get(key, PARAMS.get(key))
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"PARAMS['{key}'] must be a positive integer; got {value!r}.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"PARAMS['{key}'] must be a positive integer; got {value!r}.")
        return value

    def _bool_value(key: str, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean true/false value; got {value!r}.")
        return bool(value)

    for key, default_value in PARAMS.items():
        if isinstance(default_value, bool):
            _bool_value(f"PARAMS['{key}']", params.get(key, default_value))

    noise_parameterization = str(
        params.get("noise_parameterization", PARAMS.get("noise_parameterization", "camera_counts"))
    ).strip().lower()
    if noise_parameterization != "camera_counts":
        raise ValueError(
            "PARAMS['noise_parameterization'] must be 'camera_counts'; "
            f"got {params.get('noise_parameterization')!r}."
        )

    _finite_float("temperature_K", positive=True)
    _finite_float("viscosity_Pa_s", positive=True)
    fps = _finite_float("fps", positive=True)
    _finite_float("duration_seconds", positive=True)
    _positive_int("image_size_pixels")
    _finite_float("pixel_size_nm", positive=True)
    _finite_float("wavelength_nm", positive=True)
    if params.get("probe_wavelength_nm", PARAMS.get("probe_wavelength_nm")) is not None:
        _finite_float("probe_wavelength_nm", positive=True)
    numerical_aperture = _finite_float("numerical_aperture", positive=True)
    refractive_index_medium = _finite_float("refractive_index_medium", positive=True)
    _finite_float("refractive_index_immersion", positive=True)
    _finite_float("magnification", positive=True)
    objective_model = params.get("objective_model", PARAMS.get("objective_model"))
    if objective_model is not None and not isinstance(objective_model, str):
        raise ValueError(
            "PARAMS['objective_model'] must be None or a string; "
            f"got {objective_model!r}."
        )
    _finite_float("objective_focal_length_mm", positive=True)
    _positive_int("psf_oversampling_factor")
    _positive_int("pupil_samples")
    _finite_float("background_intensity", nonnegative=True)
    _finite_float("qpi_phase_to_count_scale", positive=True)
    _finite_float("fluorescence_excitation_wavelength_nm", positive=True)
    _finite_float("fluorescence_emission_wavelength_nm", positive=True)
    _finite_float("tem_dose_per_pixel", nonnegative=True)
    _finite_float("supervision_prior_log_odds")
    _finite_float("supervision_log_odds_threshold")
    supervision_decision_rule = str(
        params.get(
            "supervision_decision_rule",
            PARAMS.get("supervision_decision_rule", "log_odds"),
        )
    ).strip().lower()
    if supervision_decision_rule not in {"log_odds", "product"}:
        raise ValueError(
            "PARAMS['supervision_decision_rule'] must be 'log_odds' or 'product'; "
            f"got {supervision_decision_rule!r}."
        )
    matched_modalities = params.get("matched_modalities", PARAMS.get("matched_modalities"))
    if matched_modalities is not None:
        if (
            isinstance(matched_modalities, (str, bytes))
            or not isinstance(matched_modalities, (list, tuple))
            or len(matched_modalities) < 2
        ):
            raise ValueError(
                "PARAMS['matched_modalities'] must be None or a list/tuple of "
                "at least two imaging model names."
            )
        for modality in matched_modalities:
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError(
                    "PARAMS['matched_modalities'] entries must be non-empty strings."
                )
    if numerical_aperture > refractive_index_medium:
        raise ValueError(
            "PARAMS['numerical_aperture'] must be <= PARAMS['refractive_index_medium']; "
            f"got {numerical_aperture} > {refractive_index_medium}."
        )
    exposure_time_ms = params.get("exposure_time_ms", PARAMS.get("exposure_time_ms"))
    if exposure_time_ms is not None:
        exposure_time_ms = _finite_float("exposure_time_ms", positive=True)
        frame_interval_ms = 1000.0 / fps
        if exposure_time_ms > frame_interval_ms:
            raise ValueError(
                "PARAMS['exposure_time_ms'] must be <= 1000 / PARAMS['fps']; "
                f"got {exposure_time_ms} ms with fps={fps}."
            )
    num_frames = params.get("num_frames", PARAMS.get("num_frames"))
    if num_frames is not None:
        if isinstance(num_frames, bool) or not isinstance(num_frames, numbers.Integral):
            raise ValueError(
                "PARAMS['num_frames'] must be None or a positive integer; "
                f"got {num_frames!r}."
            )
        if int(num_frames) <= 0:
            raise ValueError(
                "PARAMS['num_frames'] must be None or a positive integer; "
                f"got {num_frames!r}."
            )
    if params.get("qpi_phase_noise_std_rad", None) is not None:
        _finite_float("qpi_phase_noise_std_rad", nonnegative=True)
    bit_depth = params.get("bit_depth", PARAMS.get("bit_depth"))
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, numbers.Integral):
        raise ValueError("PARAMS['bit_depth'] must be an integer in the range [1, 16].")
    if int(bit_depth) < 1 or int(bit_depth) > 16:
        raise ValueError(
            "PARAMS['bit_depth'] must be an integer in the range [1, 16]; "
            f"got {bit_depth!r}."
        )
    hot_pixel_fraction = _finite_float("hot_pixel_fraction", nonnegative=True)
    if hot_pixel_fraction > 1.0:
        raise ValueError(
            "PARAMS['hot_pixel_fraction'] must be <= 1.0; "
            f"got {hot_pixel_fraction}."
        )

    pattern_model = str(params.get("sample_environment_pattern", "none")).strip().lower()
    supported_patterns = {"none", "gold_holes", "nanopillars"}
    if pattern_model not in supported_patterns:
        raise ValueError(
            "PARAMS['sample_environment_pattern'] must be one of "
            f"{sorted(supported_patterns)}; got {params.get('sample_environment_pattern')!r}."
        )
    pattern_enabled = bool(params.get("sample_environment_pattern_enabled", False))
    environment_enabled = bool(params.get("sample_environment_enabled", True))
    if environment_enabled and pattern_enabled:
        preset = str(
            params.get("sample_environment_pattern_preset", "empty_background")
        ).strip().lower()
        if pattern_model == "gold_holes" and preset not in {
            "empty_background",
            "default_gold_holes",
        }:
            raise ValueError(
                "PARAMS['sample_environment_pattern_preset'] must be "
                "'empty_background' or 'default_gold_holes' for gold_holes; "
                f"got {params.get('sample_environment_pattern_preset')!r}."
            )
        if pattern_model == "nanopillars" and preset not in {
            "empty_background",
            "nanopillars",
            "default_nanopillars",
        }:
            raise ValueError(
                "PARAMS['sample_environment_pattern_preset'] must be "
                "'empty_background', 'nanopillars', or 'default_nanopillars' "
                "for nanopillars; "
                f"got {params.get('sample_environment_pattern_preset')!r}."
            )

    noise_keys = {
        "shot_noise_enabled",
        "gaussian_noise_enabled",
        "camera_gain_e_per_count",
        "read_noise_counts",
        "dark_offset_counts",
        "fixed_pattern_gain_std",
        "fixed_pattern_offset_counts",
        "hot_pixel_fraction",
        "hot_pixel_value_counts",
        "scan_line_noise_counts",
        "clip_output_to_nonnegative",
        "noise_parameterization",
    }

    noise_model = params.get("noise_model", {})
    if noise_model is None:
        noise_model = {}
    if not isinstance(noise_model, dict):
        raise TypeError("PARAMS['noise_model'] must be a dictionary when provided.")
    unknown_noise = sorted(str(key) for key in noise_model if str(key) not in noise_keys)
    if unknown_noise:
        raise ValueError(
            "Unknown PARAMS['noise_model'] key(s): "
            + ", ".join(repr(key) for key in unknown_noise)
                + ". Use the counts-domain noise keys documented in config.PARAMS."
        )
    for key in ("shot_noise_enabled", "gaussian_noise_enabled", "clip_output_to_nonnegative"):
        if key in noise_model:
            _bool_value(f"PARAMS['noise_model']['{key}']", noise_model[key])
    if "noise_parameterization" in noise_model:
        nested_parameterization = str(noise_model["noise_parameterization"]).strip().lower()
        if nested_parameterization != "camera_counts":
            raise ValueError(
                "PARAMS['noise_model']['noise_parameterization'] must be 'camera_counts'; "
                f"got {noise_model['noise_parameterization']!r}."
            )

    modality_noise = params.get("modality_noise", {})
    if modality_noise is None:
        modality_noise = {}
    if not isinstance(modality_noise, dict):
        raise TypeError("PARAMS['modality_noise'] must be a dictionary when provided.")
    for modality_name, overrides in modality_noise.items():
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise TypeError(
                "Each PARAMS['modality_noise'] entry must be a dictionary; "
                f"got {type(overrides).__name__} for {modality_name!r}."
            )
        unknown_modality_noise = sorted(
            str(key) for key in overrides if str(key) not in noise_keys
        )
        if unknown_modality_noise:
            raise ValueError(
                f"Unknown PARAMS['modality_noise'][{modality_name!r}] key(s): "
                + ", ".join(repr(key) for key in unknown_modality_noise)
                + ". Use the counts-domain noise keys documented in config.PARAMS."
            )
        for key in ("shot_noise_enabled", "gaussian_noise_enabled", "clip_output_to_nonnegative"):
            if key in overrides:
                _bool_value(
                    f"PARAMS['modality_noise'][{modality_name!r}]['{key}']",
                    overrides[key],
                )
        if "noise_parameterization" in overrides:
            nested_parameterization = str(overrides["noise_parameterization"]).strip().lower()
            if nested_parameterization != "camera_counts":
                raise ValueError(
                    "PARAMS['modality_noise'][%r]['noise_parameterization'] must be 'camera_counts'; "
                    "got %r." % (modality_name, overrides["noise_parameterization"])
                )

# --- PHYSICAL CONSTANTS ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
