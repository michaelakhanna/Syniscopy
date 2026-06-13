"""Canonical report-facing microscope parameter surfaces.

The lab Fisher report compares microscope configurations, not merely modality
names. Public microscope templates and overlay diagnostics therefore need one
owner for the sparse instrument/backend parameter surface of each modality.
Historically that surface lived as hand-written key sets in ``modality_registry``;
those sets drifted from ``parameters``/``PARAM_SCHEMA`` and from renderer-consumed
canonical keys. This module is the durable authority that both template writers
and overlay relevance checks consume, so a future audit cannot revive stale alias
keys in one branch while another branch uses canonical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from microscope_axes import MICROSCOPE_LOCAL_AXES, classify


@dataclass(frozen=True)
class ModalityParameterSurface:
    """Public report-facing parameter surface for one modality.

    ``public_keys`` is intentionally a configuration contract, not a renderer
    dependency graph. Every key in the surface must be a canonical public
    ``parameters`` key. Historical aliases are not supported; only canonical keys
    are returned for template and overlay workflows.
    """

    modality: str
    public_keys: frozenset[str]

    def template_keys(self, shared_keys: frozenset[str]) -> frozenset[str]:
        """Return sparse microscope-local keys for generated JSON templates."""

        return frozenset(
            key
            for key in self.public_keys
            if key not in shared_keys
            and key not in {"imaging_model", "particles"}
            and classify(key) in MICROSCOPE_LOCAL_AXES
        )


REPORT_SHARED_PARAM_KEYS = frozenset(
    {
        "image_size_pixels",
        "pixel_size_nm",
        "num_frames",
        "duration_seconds",
        "fps",
        "particles",
        "initial_z_span_nm",
        "random_seed",
        "background_subtraction_method",
        "shot_noise_enabled",
        "gaussian_noise_enabled",
        "return_ideal_float_frames",
        "save_frame_sequence",
        "save_raw_camera_video",
        "save_raw_camera_frame_sequence",
        "save_raw_frame_views",
        "mask_generation_enabled",
        "fisher_likelihood_model",
        "dynamic_bayesian_enabled",
        "dynamic_process_noise_scale",
        "dynamic_initial_variance_nm2",
        "dynamic_include_smoothing",
    }
)

REPORT_DETECTOR_PARAM_KEYS = frozenset(
    {
        "background_intensity",
        "read_noise_counts",
        "camera_gain_e_per_count",
        "detector_noise_input_domain",
        "detector_qe",
        "dark_current_e_per_pixel_per_s",
        "exposure_time_ms",
        "exposure_time_s",
        "bit_depth",
        "adc_quantization",
        "adc_quantization_counts",
        "fixed_pattern_gain_std",
        "fixed_pattern_offset_counts",
        "hot_pixel_fraction",
        "scan_line_noise_counts",
    }
)

REPORT_COMMON_OPTICAL_PARAM_KEYS = frozenset(
    {
        "wavelength_nm",
        "probe_wavelength_nm",
        "numerical_aperture",
        "refractive_index_medium",
        "refractive_index_immersion",
        "pupil_samples",
        "vectorial_pupil_samples",
        "psf_oversampling_factor",
        "apodization_factor",
        "coverslip_aberration_model",
        "coverslip_thickness_um",
        "coverslip_refractive_index",
        "coverslip_design_thickness_um",
        "coverslip_design_refractive_index",
        "illumination_spectrum_center_nm",
        "illumination_spectrum_fwhm_nm",
        "illumination_spectrum_num_samples",
        "broadband_wavelengths_nm",
        "broadband_weights",
        "optical_field_backend",
        "optical_scattering_model",
        "optical_cluster_scattering_model",
        "optical_cluster_dda_voxel_size_nm",
        "optical_cluster_dda_max_dipoles",
        "vectorial_detection_mode",
        "vectorial_polarization_rotation_deg",
        "vectorial_obliquity_apodization",
        "polarization_model",
        "reference_field_amplitude",
    }
)

REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS = {
    "bright_field": frozenset(
        {
            "kohler_source_samples",
            "kohler_coherence_factor",
            "bright_field_sample_environment_gain",
            "bright_field_sample_environment_phase_gain",
        }
    ),
    "partially_coherent_bright_field": frozenset(
        {
            "kohler_source_samples",
            "kohler_coherence_factor",
            "bright_field_sample_environment_gain",
            "bright_field_sample_environment_phase_gain",
        }
    ),
    "coherent_bright_field": frozenset(
        {
            "bright_field_sample_environment_gain",
            "bright_field_sample_environment_phase_gain",
        }
    ),
    "dark_field": frozenset(
        {
            "dark_field_background_count",
            "dark_field_field_gain",
            "dark_field_illumination_count",
            "annular_dark_field_inner_sigma",
            "annular_dark_field_outer_sigma",
            "annular_dark_field_source_samples",
            "dark_field_sample_environment_edge_gain",
            "dark_field_sample_environment_scatter_pedestal",
        }
    ),
    "coherent_dark_field": frozenset(
        {
            "dark_field_background_count",
            "dark_field_field_gain",
            "dark_field_illumination_count",
            "dark_field_sample_environment_edge_gain",
            "dark_field_sample_environment_scatter_pedestal",
        }
    ),
    "zernike_phase_contrast": frozenset(
        {
            "zernike_model",
            "zernike_phase_ring_inner_fraction",
            "zernike_phase_ring_outer_fraction",
            "zernike_phase_ring_shift_rad",
            "zernike_phase_ring_amplitude",
            "zernike_phase_ring_gain",
            "zernike_phase_bias",
        }
    ),
    "differential_phase_contrast": frozenset(
        {
            "dpc_channel_model",
            "dpc_illumination_sigma",
            "dpc_intensity_gain",
            "dpc_intensity_gain_x",
            "dpc_intensity_gain_y",
            "dpc_output_channel",
            "dpc_phase_gradient_gain",
            "dpc_phase_gradient_gain_x",
            "dpc_phase_gradient_gain_y",
            "dpc_source_samples",
            "dpc_transfer_model",
        }
    ),
    "quantitative_phase": frozenset(
        {
            "qpi_visibility",
            "qpi_detected_quanta_per_pixel",
            "qpi_phase_to_count_scale",
            "qpi_phase_noise_std_rad",
        }
    ),
    "off_axis_holography": frozenset(
        {
            "off_axis_fringe_period_px",
            "off_axis_fringe_angle_rad",
            "off_axis_reference_amplitude_scale",
        }
    ),
    "interferometric": frozenset(
        {
            "iscat_reference_model",
            "iscat_reference_medium_material",
            "iscat_reference_substrate_material",
            "iscat_reference_amplitude_scale",
            "iscat_reference_phase_rad",
            "iscat_reference_coefficient",
            "iscat_reference_normalize_fresnel_phase_only",
            "iscat_collection_model",
            "iscat_collection_reference_fraction",
        }
    ),
    "ricm": frozenset(
        {
            "ricm_interface_reflection_model",
            "ricm_interface_reflection_coefficient",
            "ricm_interface_phase_shift_rad",
            "ricm_interface_medium_material",
            "ricm_interface_substrate_material",
            "ricm_particle_medium_material",
            "ricm_particle_reflection_coefficient",
            "ricm_particle_reflection_model",
            "ricm_particle_material",
            "ricm_thinfilm_layers",
            "ricm_wavelength_nm",
            "ricm_gap_nm",
            "ricm_use_particle_z_as_gap",
        }
    ),
}

REPORT_OPTICAL_PARAM_KEYS = frozenset(
    REPORT_COMMON_OPTICAL_PARAM_KEYS
    | frozenset(
        key
        for family_keys in REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS.values()
        for key in family_keys
    )
)

REPORT_FLUORESCENCE_PARAM_KEYS = frozenset(
    {
        "fluorescence_backend",
        "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame",
        "fluorescence_allow_psf_fallback",
        "fluorescence_background",
        "fluorescence_bleaching_rate_per_frame",
        "fluorescence_bright_to_dark_rate_per_frame",
        "fluorescence_collection_efficiency",
        "fluorescence_dark_to_bright_rate_per_frame",
        "fluorescence_detector_qe",
        "fluorescence_emission_psf_sigma_nm",
        "fluorescence_emission_wavelength_nm",
        "fluorescence_excitation_scale",
        "fluorescence_excitation_wavelength_nm",
        "fluorescence_quantum_yield",
        "fluorescence_reference_status",
        "fluorescence_reference_validation_hash",
        "fluorescence_sample_environment_autofluorescence_gain",
        "fluorescence_sample_environment_excitation_modulation_gain",
        "fluorescence_source_representation",
        "fluorescence_spectral_bandwidth_nm",
        "fluorescence_volume_slice_thickness_nm",
        "fluorescence_volume_slices",
    }
)

REPORT_TIRF_PARAM_KEYS = frozenset(
    {
        "tirf_fluorescence_backend",
        "tirf_source_representation",
        "tirf_penetration_depth_nm",
        "tirf_use_angle_derived_penetration_depth",
        "tirf_prism_refractive_index",
        "tirf_sample_refractive_index",
        "tirf_incident_angle_deg",
        "tirf_height_offset_nm",
        "tirf_effective_numerical_aperture",
    }
)

REPORT_TEM_PARAM_KEYS = frozenset(
    {
        "tem_backend",
        "tem_reference_status",
        "tem_reference_validation_hash",
        "tem_potential_source",
        "tem_objective_aperture_mrad",
        "tem_acceleration_kV",
        "tem_model",
        "tem_multislice_slices",
        "tem_slice_thickness_nm",
        "tem_Cs_mm",
        "tem_partial_coherence_alpha_mrad",
        "tem_defocus_nm",
        "tem_pixel_size_pm",
        "tem_phase_shift_per_volt_nm",
        "tem_projected_potential_scale",
        "tem_filter_guard_pixels",
        "tem_sample_environment_potential_scale",
        "tem_dose_per_pixel",
    }
)

REPORT_SEM_PARAM_KEYS = frozenset(
    {
        "sem_backend",
        "sem_source_representation",
        "sem_volume_slices",
        "sem_volume_slice_thickness_nm",
        "sem_source_z_origin",
        "sem_source_z_offset_nm",
        "sem_reference_kernel_path",
        "sem_reference_kernel_sha256",
        "sem_acceleration_kV",
        "sem_model",
        "sem_interaction_volume_nm",
        "sem_detector_direction_xy",
        "sem_topography_contrast_gain",
        "sem_probe_sigma_nm",
        "sem_filter_guard_pixels",
        "sem_edge_contrast_gain",
        "sem_bulk_contrast_gain",
        "sem_baseline_yield",
        "sem_sample_environment_edge_gain",
        "sem_electrons_per_pixel",
        "sem_beam_current_nA",
        "sem_dwell_time_us",
        "sem_detector_takeoff_angle_deg",
        "sem_detector_acceptance",
        "sem_escape_depth_nm",
        "sem_backscatter_fraction",
        "sem_transport_material_scale",
        "sem_transport_source_exponent",
        "sem_transport_topography_exponent",
        "sem_monte_carlo_trajectories",
        "sem_monte_carlo_steps",
        "sem_monte_carlo_seed",
        "sem_monte_carlo_step_nm",
        "sem_monte_carlo_range_nm",
        "sem_monte_carlo_scatter_std_deg",
        "sem_monte_carlo_kernel_size_px",
        "sem_physical_max_steps",
        "sem_physical_energy_cutoff_keV",
        "sem_physical_elastic_model",
        "sem_reference_material",
        "sem_reference_geometry",
        "sem_reference_source_depth_nm",
        "sem_reference_incident_angle_deg",
    }
)

REPORT_MODALITY_SPECIFIC_ELECTRON_PARAM_KEYS = {
    "tem_phase_contrast": REPORT_TEM_PARAM_KEYS,
    "sem_secondary_electron": REPORT_SEM_PARAM_KEYS,
}

REPORT_ELECTRON_PARAM_KEYS = frozenset(REPORT_TEM_PARAM_KEYS | REPORT_SEM_PARAM_KEYS)

REQUIRED_SURFACE_KEYS_BY_MODALITY = {
    "bright_field": frozenset({"kohler_source_samples", "kohler_coherence_factor"}),
    "partially_coherent_bright_field": frozenset({"kohler_source_samples", "kohler_coherence_factor"}),
    "zernike_phase_contrast": frozenset(
        {
            "zernike_phase_ring_shift_rad",
            "zernike_phase_ring_inner_fraction",
            "zernike_phase_ring_outer_fraction",
        }
    ),
    "quantitative_phase": frozenset({"qpi_phase_to_count_scale", "qpi_phase_noise_std_rad"}),
    "off_axis_holography": frozenset(
        {"off_axis_fringe_period_px", "off_axis_fringe_angle_rad", "off_axis_reference_amplitude_scale"}
    ),
    "interferometric": frozenset({"iscat_reference_amplitude_scale", "iscat_reference_model"}),
    "ricm": frozenset({"ricm_interface_reflection_coefficient", "ricm_particle_reflection_coefficient"}),
    "tirf_fluorescence": frozenset(
        {
            "tirf_incident_angle_deg",
            "tirf_use_angle_derived_penetration_depth",
            "tirf_effective_numerical_aperture",
        }
    ),
    "tem_phase_contrast": frozenset({"tem_acceleration_kV", "tem_Cs_mm", "tem_dose_per_pixel"}),
    "sem_secondary_electron": frozenset({"sem_acceleration_kV", "sem_probe_sigma_nm", "sem_electrons_per_pixel"}),
}


def modality_parameter_surface(*, modality: str, flags: Mapping[str, bool]) -> ModalityParameterSurface:
    """Build the canonical public parameter surface for one modality.

    The modality registry supplies only identity/capability flags; this function
    owns the parameter-surface policy. Keeping the two concepts separate avoids
    the previous architecture where stale alias names in the registry could
    control JSON templates and overlay warnings even though renderers consumed
    different canonical parameters keys.
    """

    keys = set(REPORT_SHARED_PARAM_KEYS | REPORT_DETECTOR_PARAM_KEYS)
    if (
        flags.get("label_free_optical")
        or flags.get("coherent_reference")
        or flags.get("lab_optical")
        or flags.get("fluorescence")
    ):
        keys.update(REPORT_COMMON_OPTICAL_PARAM_KEYS)
    if flags.get("label_free_optical") or flags.get("coherent_reference"):
        keys.update(REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS.get(modality, frozenset()))
    if flags.get("fluorescence"):
        keys.update(REPORT_FLUORESCENCE_PARAM_KEYS)
        if modality == "tirf_fluorescence":
            keys.update(REPORT_TIRF_PARAM_KEYS)
    if flags.get("electron"):
        keys.update(REPORT_MODALITY_SPECIFIC_ELECTRON_PARAM_KEYS.get(modality, frozenset()))
    return ModalityParameterSurface(
        modality=modality,
        public_keys=frozenset(keys),
    )


__all__ = [
    "ModalityParameterSurface",
    "REPORT_COMMON_OPTICAL_PARAM_KEYS",
    "REPORT_DETECTOR_PARAM_KEYS",
    "REPORT_ELECTRON_PARAM_KEYS",
    "REPORT_FLUORESCENCE_PARAM_KEYS",
    "REPORT_MODALITY_SPECIFIC_ELECTRON_PARAM_KEYS",
    "REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS",
    "REPORT_OPTICAL_PARAM_KEYS",
    "REPORT_SEM_PARAM_KEYS",
    "REPORT_SHARED_PARAM_KEYS",
    "REPORT_TEM_PARAM_KEYS",
    "REPORT_TIRF_PARAM_KEYS",
    "REQUIRED_SURFACE_KEYS_BY_MODALITY",
    "modality_parameter_surface",
]
