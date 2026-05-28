"""Editable public dataset recipe.

This file is the main user-facing configuration surface for local dataset
generation. It intentionally contains microscope/sample parameters, not paper
experiment settings and not internal cache details.

Run:

    python codebase/create_dataset.py --output datasets/default

All numeric units are included in key names where possible.
"""

DEFAULT = {
    # ------------------------------------------------------------------
    # Frame geometry and timing
    # ------------------------------------------------------------------
    # image_size_pixels: square frame width/height in pixels.
    "image_size_pixels": 256,

    # pixel_size_nm: effective sample-plane pixel size.
    "pixel_size_nm": 100.0,

    # fps: output frame rate in frames per second.
    "fps": 30.0,

    # num_frames: exact number of frames to generate.
    "num_frames": 60,

    # bit_depth options: integer 1..16. Stored frames use uint16 internally.
    "bit_depth": 16,

    # exposure_time_ms: None means full-frame exposure (1000/fps ms).
    # Set a positive value <= 1000/fps for strobe/partial exposure.
    "exposure_time_ms": None,

    # ------------------------------------------------------------------
    # Particle sample
    # ------------------------------------------------------------------
    # particles: list of physical particle objects.
    # Each particle has motion settings plus one or more spherical components.
    # A simple sphere is one component at offset [0, 0, 0]. A dimer/rod/stack is
    # represented by multiple components with fixed offsets in the particle body
    # frame. Material options include gold, silver, polystyrene, silica,
    # protein, lipid, fluorescent_polystyrene, pet, polyethylene,
    # polypropylene, water, air, glass, and carbon.
    "particles": [
        {
            "name": "particle_0",
            "motion": {
                # Hydrodynamic diameter used for translational and rotational
                # Brownian motion.
                "hydrodynamic_diameter_nm": 100.0,
                # None samples an initial [x, y, z] position from the configured
                # field of view and initial axial span. Or provide
                # [x_nm, y_nm, z_nm].
                "initial_position_nm": None,
            },
            # Per-logical-particle scattered-field amplitude multiplier.
            "signal_multiplier": 1.0,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "polystyrene",
                    # None uses material lookup. JSON can use
                    # {"real": 1.59, "imag": 0.0}; Python recipes can also
                    # use 1.59 + 0.0j.
                    "refractive_index": None,
                    # Per-component multiplier, applied inside the particle.
                    "signal_multiplier": 1.0,
                    # Optional component material-property overrides:
                    # fluorophore_density, emission_peak_nm,
                    # excitation_peak_nm, se_yield_coefficient,
                    # mean_inner_potential_V, density_g_cm3.
                    "material_properties": None,
                }
            ],
        }
    ],

    # ------------------------------------------------------------------
    # Microscope / imaging model
    # ------------------------------------------------------------------
    # imaging_model options:
    # bright_field, fluorescence_widefield, tirf_fluorescence, dark_field,
    # zernike_phase_contrast, differential_phase_contrast, quantitative_phase,
    # off_axis_holography, ricm, interferometric, tem_phase_contrast,
    # sem_secondary_electron, partially_coherent_bright_field,
    # coherent_bright_field, coherent_dark_field.
    # Accepted aliases: partially_coherent_brightfield, coherent_brightfield.
    "imaging_model": "bright_field",

    # wavelength_nm: optical wavelength. Electron modes derive probe wavelength
    # from their voltage-specific parameters when implemented by the model.
    "wavelength_nm": 550.0,

    # numerical_aperture: objective NA. Must be <= refractive_index_immersion.
    "numerical_aperture": 0.8,

    # magnification/objective_focal_length_mm are metadata and optical-context
    # knobs used by presets and reporting.
    "magnification": 40,
    "objective_focal_length_mm": 4.5,

    # refractive indices for sample and immersion media.
    "refractive_index_medium": 1.33,
    "refractive_index_immersion": 1.33,

    # medium_material options include:
    # water, buffer, air, vacuum, glass, sio2, silica, si, silicon, carbon.
    "medium_material": "water",

    # channels: None for one-channel rendering. Or a list such as:
    # [{"name": "green", "wavelength_nm": 532.0},
    #  {"name": "red", "wavelength_nm": 640.0}]
    "channels": None,

    # multichannel_output_mode options for dataset generation: rgb, both.
    # Low-level direct renders can also use channels or none.
    "multichannel_output_mode": "rgb",

    # Fluorescence controls used by fluorescence_widefield/tirf_fluorescence.
    "fluorescence_quantum_yield": 0.5,
    "fluorescence_excitation_scale": 1.0,
    "fluorescence_emission_psf_sigma_px": 1.0,
    "fluorescence_background": 0.0,
    "fluorescence_photon_count_scale": 500.0,
    "fluorescence_spectral_bandwidth_nm": 40.0,
    "fluorescence_excitation_wavelength_nm": 488.0,
    "fluorescence_emission_wavelength_nm": 520.0,
    "fluorescence_photobleach_tau_frames": None,

    # ------------------------------------------------------------------
    # Brownian motion and bench perturbations
    # ------------------------------------------------------------------
    "temperature_K": 298.15,
    "viscosity_Pa_s": 8.9e-4,

    # z_motion_constraint_model options:
    # unconstrained, reflecting_floor_z0, reflecting_ceiling_z0.
    # unconstrained is the default pure Brownian model. Use reflecting_ceiling_z0
    # only when you intentionally want a surface interaction that blocks motion
    # above the sample-interface plane z = 0.
    "z_motion_constraint_model": "unconstrained",

    # initial_z_span_nm: span used only when sampling initial z positions and
    # sizing the PSF cache. It is not a Brownian-motion boundary. Pure Brownian
    # motion can move outside this initial span; the renderer expands the PSF
    # cache to cover the realized trajectory.
    "initial_z_span_nm": 3000.0,

    # z_stack_step_nm: z spacing used for PSF cache slices.
    "z_stack_step_nm": 100.0,

    # motion_blur_enabled: averages sub-frame positions over exposure.
    "motion_blur_enabled": True,
    "motion_blur_subsamples": 4,

    # drift_velocity_nm_per_s: rigid scene drift [vx, vy, vz].
    "drift_velocity_nm_per_s": [0.0, 0.0, 0.0],

    # vibration_jitter_std_nm: per-exposure random jitter. Set 0 to disable.
    "vibration_jitter_std_nm": 0.0,
    "vibration_include_axial": False,

    # rotational_diffusion_enabled: relevant for composite/non-spherical objects.
    "rotational_diffusion_enabled": False,
    "rotational_diffusion_mode": "empirical",
    "rotational_step_std_deg": 10.0,

    # ------------------------------------------------------------------
    # Sample environment
    # ------------------------------------------------------------------
    "sample_environment_enabled": True,

    # sample_environment_pattern_enabled: whether to apply a mounting-interface
    # pattern overlay.
    "sample_environment_pattern_enabled": False,

    # sample_environment_pattern options:
    # none, gold_holes, nanopillars.
    "sample_environment_pattern": "none",

    # mounting_interface_material/bulk_substrate_material options include:
    # glass, water, air, vacuum, sio2, silica, si, silicon, carbon,
    # holey_carbon, gold.
    "mounting_interface_material": "glass",
    "bulk_substrate_material": "glass",
    "mounting_interface_thickness_nm": 170000.0,
    "sample_environment_exclusion_method": "reflection",

    # sample_environment_pattern_contrast_model options:
    # static, time_dependent.
    "sample_environment_pattern_contrast_model": "static",
    "sample_environment_pattern_contrast_amplitude": 0.0,

    # sample_environment_pattern_preset options depend on sample_environment_pattern.
    # Common examples: default_gold_holes, nanopillars.
    "sample_environment_pattern_preset": "default_gold_holes",

    # sample_environment_pattern_dimensions: edit only keys relevant to the chosen model.
    "sample_environment_pattern_dimensions": {
        "hole_diameter_um": 15.0,
        "hole_edge_to_edge_spacing_um": 2.0,
        "hole_depth_nm": 20.0,
        "hole_intensity_factor": 0.7,
        "gold_intensity_factor": 1.0,
        "pitch_nm": 2000.0,
        "bar_width_nm": 250.0,
        "dot_diameter_nm": 250.0,
        "pillar_diameter_um": 1.0,
        "pillar_edge_to_edge_spacing_um": 2.0,
        "pillar_height_nm": 20.0,
        "pillar_intensity_factor": 1.3,
        "background_intensity_factor": 1.0,
    },
    "sample_environment_pattern_randomization_enabled": False,
    "sample_environment_pattern_position_jitter_std_nm": 0.0,
    "sample_environment_pattern_shape_regularity": 1.0,
    "sample_environment_pattern_edge_perturbation_max_rel_radius": 0.0,
    "sample_environment_pattern_edge_perturbation_mode_count": 0,

    # ------------------------------------------------------------------
    # PSF / response calculation controls
    # ------------------------------------------------------------------
    "psf_oversampling_factor": 2,
    "pupil_samples": 256,
    # False is the recommended public default: the renderer builds each PSF
    # cache over the realized Brownian z range plus margin, so axial motion
    # remains covered by the PSF cache. Set True only when you deliberately
    # need a fixed global cache grid.
    "shared_psf_z_grid_enabled": False,
    "spherical_aberration_strength": 0.1,
    "apodization_factor": 1.0,
    "random_aberration_strength": 0.0,
    "psf_intensity_fraction_threshold": 1e-4,

    # ------------------------------------------------------------------
    # Detector noise and count scaling
    # ------------------------------------------------------------------
    # background_intensity: average reference/background level in camera counts.
    "background_intensity": 100.0,

    # shot_noise_enabled: enables Poisson photon/electron counting noise.
    "shot_noise_enabled": True,

    # gaussian_noise_enabled: enables Gaussian read noise in camera counts.
    "gaussian_noise_enabled": True,

    # noise_parameterization options: camera_counts.
    "noise_parameterization": "camera_counts",

    # camera_gain_e_per_count: detected electrons per camera count/ADU.
    "camera_gain_e_per_count": 1.0,

    # read_noise_counts: RMS Gaussian read noise in counts.
    "read_noise_counts": 1.0,

    # dark_offset_counts: constant camera offset before noise.
    "dark_offset_counts": 0.0,

    # Fixed-pattern and detector-artifact terms. Set 0 to disable.
    "fixed_pattern_gain_std": 0.0,
    "fixed_pattern_offset_counts": 0.0,
    "hot_pixel_fraction": 0.0,
    "hot_pixel_value_counts": None,
    "scan_line_noise_counts": 0.0,
    "clip_output_to_nonnegative": True,

    # modality_noise: per-imaging-model overrides, for example:
    # {"sem_secondary_electron": {"scan_line_noise_counts": 2.0}}
    "modality_noise": {},

    # Dark-field count scaling. Used only by dark_field/coherent_dark_field.
    "dark_field_illumination_count": 100.0,
    "dark_field_background_count": 5.0,

    # ------------------------------------------------------------------
    # Empirical background / shading field
    # ------------------------------------------------------------------
    "empirical_background_enabled": False,

    # empirical_background_model options: multiscale_gaussian_field.
    "empirical_background_model": "multiscale_gaussian_field",
    "empirical_background_relative_std": 0.03,
    "empirical_background_scales_px": [16.0, 64.0],
    "empirical_background_scale_weights": [0.6, 0.4],
    "empirical_background_gradient_relative_strength": 0.0,

    # ------------------------------------------------------------------
    # Training video post-processing and masks
    # ------------------------------------------------------------------
    # background_subtraction_method options:
    # none, reference_frame, video_median.
    "background_subtraction_method": "reference_frame",

    # save_frame_sequence: write lossless PNG training frames under
    # frames/video_XXXX/. Keep True for generated datasets; AVI is preview.
    "save_frame_sequence": True,

    # save_raw_frame_views: write raw signal/reference/final frame arrays for
    # audit. False keeps generated datasets smaller.
    "save_raw_frame_views": False,

    "mask_generation_enabled": True,
    "mask_outer_ring_count": 1,
    "mask_max_area_fraction": 0.25,

    # supervision_target options: mask_supported, mask_geometry.
    "supervision_target": "mask_supported",

    # supervision_support_factors options:
    # [], ["temporal"], ["signal"], ["information"], or any combination of
    # temporal, signal, information.
    "supervision_support_factors": [],
    "supervision_supported_threshold": 0.2,
    "supervision_temporal_support_enabled": True,
    "supervision_signal_support_enabled": True,
    "supervision_information_support_enabled": True,
    "supervision_crlb_xy_max_nm": None,
    "supervision_stop_when_all_temporally_unsupported": False,
}
