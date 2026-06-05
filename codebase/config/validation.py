"""Parameter normalization and validation rules."""
from __future__ import annotations
import math
import numbers
import os
from copy import deepcopy
import numpy as np
from measurement_units import normalize_detector_noise_input_domain
from modality_registry import (
    FLUORESCENCE_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name,
    is_electron_modality,
)
from param_schema import PARAM_SCHEMA
from param_schema.sample_environment import BAR_ORIENTATION_CHOICES, PATTERN_DEFAULT_PRESETS
from shared_constants import COHERENT_REFERENCE_MODALITIES
from .defaults import PARAMS, _KNOWN_INTERNAL_PARAM_KEYS

def _normalize_params_in_place(params: dict, *, allowed_extra_keys: set[str] | None=None, allowed_internal_keys: set[str] | None=None) -> None:
    """
    Validate the public PARAMS surface.

    Syniscopy v1 has one canonical public key per concept. Unknown keys raise
    immediately so aliases and typo-driven configuration drift do not enter
    generated datasets or manuscript artifacts.

    This private implementation normalizes a small number of
    dependent/canonical values in ``params`` in place. Public callers should use
    ``normalize_params`` when they need those values and ``validate_params``
    when they only need a non-mutating validation check.
    """
    allowed = set(PARAMS)
    if allowed_extra_keys:
        allowed.update((str(key) for key in allowed_extra_keys))
    if allowed_internal_keys:
        allowed.update((str(key) for key in allowed_internal_keys if str(key) in _KNOWN_INTERNAL_PARAM_KEYS))
    unknown = sorted((str(key) for key in params if str(key) not in allowed))
    if unknown:
        preview = ', '.join((repr(key) for key in unknown[:8]))
        if len(unknown) > 8:
            preview += f', ... ({len(unknown)} total)'
        raise ValueError(f'Unknown simulation parameter key(s): {preview}. Use the canonical keys documented in config.PARAMS.')
    provided_keys = set(params)
    for key, default_value in PARAMS.items():
        if key not in params:
            params[key] = deepcopy(default_value)

    def _choices(key: str) -> tuple:
        spec = PARAM_SCHEMA.get(key, {})
        return tuple(spec.get('choices') or ())

    def _require_choice(key: str, value) -> str:
        text = str(value).strip().lower()
        choices = {
            str(choice).strip().lower(): choice
            for choice in _choices(key)
        }
        if choices and text not in choices:
            raise ValueError(f"PARAMS['{key}'] must be one of {list(choices.values())}; got {value!r}.")
        return text

    def _validate_schema_enum_choices() -> None:
        """Enforce every direct runtime enum declared in PARAM_SCHEMA."""
        for schema_key, spec in PARAM_SCHEMA.items():
            if spec.get("type") != "enum":
                continue
            choices = tuple(spec.get("choices") or ())
            if not choices:
                continue
            key = str(spec.get("key") or schema_key)
            if key not in PARAMS or key not in params:
                continue
            value = params[key]
            if value is None and spec.get("default") is None:
                continue
            normalized_choices = {
                str(choice).strip().lower(): choice
                for choice in choices
            }
            normalized_value = str(value).strip().lower()
            if normalized_value not in normalized_choices:
                raise ValueError(
                    f"PARAMS['{key}'] must be one of {list(choices)}; got {value!r}."
                )

    def _finite_float(key: str, *, positive: bool=False, nonnegative: bool=False) -> float:
        value = float(params[key])
        if not math.isfinite(value):
            raise ValueError(f"PARAMS['{key}'] must be finite; got {value}.")
        if positive and value <= 0.0:
            raise ValueError(f"PARAMS['{key}'] must be positive; got {value}.")
        if nonnegative and value < 0.0:
            raise ValueError(f"PARAMS['{key}'] must be non-negative; got {value}.")
        return value

    def _positive_int(key: str) -> int:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"PARAMS['{key}'] must be a positive integer; got {value!r}.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"PARAMS['{key}'] must be a positive integer; got {value!r}.")
        return value

    def _nonnegative_int(key: str) -> int:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"PARAMS['{key}'] must be a non-negative integer; got {value!r}.")
        value = int(value)
        if value < 0:
            raise ValueError(f"PARAMS['{key}'] must be a non-negative integer; got {value!r}.")
        return value

    def _bool_value(key: str, value) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be a boolean true/false value; got {value!r}.')
        return bool(value)

    def _validate_noise_map_value(key: str, value) -> None:
        if value is None:
            return
        if isinstance(value, (bool, str, bytes, os.PathLike)):
            return
        if isinstance(value, (int, float, np.integer, np.floating, list, tuple, np.ndarray)):
            return
        raise ValueError(f'{key} must be a noise-map-like value; got {type(value).__name__}.')
    for (key, default_value) in PARAMS.items():
        if isinstance(default_value, bool):
            _bool_value(f"PARAMS['{key}']", params[key])
    _validate_schema_enum_choices()
    imaging_model_input = params['imaging_model']
    imaging_model = canonical_modality_name(str(imaging_model_input).strip().lower())
    if imaging_model not in set(SUPPORTED_MODALITIES):
        raise ValueError(f"PARAMS['imaging_model'] must be one of {sorted(SUPPORTED_MODALITIES)}; got {imaging_model_input!r}.")
    noise_parameterization = _require_choice('noise_parameterization', params['noise_parameterization'])
    _finite_float('temperature_K', positive=True)
    _finite_float('viscosity_Pa_s', positive=True)
    fps = _finite_float('fps', positive=True)
    _finite_float('duration_seconds', positive=True)
    _positive_int('image_size_pixels')
    _finite_float('pixel_size_nm', positive=True)
    _finite_float('wavelength_nm', positive=True)
    if params['probe_wavelength_nm'] is not None:
        _finite_float('probe_wavelength_nm', positive=True)
    numerical_aperture = _finite_float('numerical_aperture', positive=True)
    refractive_index_medium = _finite_float('refractive_index_medium', positive=True)
    _finite_float('refractive_index_immersion', positive=True)
    _finite_float('magnification', positive=True)
    objective_model = params['objective_model']
    optical_field_backend = _require_choice('optical_field_backend', params['optical_field_backend'])
    is_differential_phase_contrast = imaging_model == 'differential_phase_contrast'
    polarization_model = _require_choice('polarization_model', params['polarization_model'])
    rotation_deg = float(params['vectorial_polarization_rotation_deg'])
    if not np.isfinite(rotation_deg):
        raise ValueError(f"PARAMS['vectorial_polarization_rotation_deg'] must be finite; got {rotation_deg!r}.")
    vectorial_pupil_samples = params['vectorial_pupil_samples']
    if vectorial_pupil_samples is not None:
        if isinstance(vectorial_pupil_samples, bool) or not isinstance(vectorial_pupil_samples, numbers.Integral):
            raise ValueError(f"PARAMS['vectorial_pupil_samples'] must be None or a positive integer; got {vectorial_pupil_samples!r}.")
        if int(vectorial_pupil_samples) <= 0:
            raise ValueError(f"PARAMS['vectorial_pupil_samples'] must be None or a positive integer; got {vectorial_pupil_samples!r}.")
    coverslip_model = _require_choice('coverslip_aberration_model', params['coverslip_aberration_model'])
    _finite_float('coverslip_thickness_um', nonnegative=True)
    _finite_float('coverslip_design_thickness_um', nonnegative=True)
    _finite_float('coverslip_refractive_index', positive=True)
    _finite_float('coverslip_design_refractive_index', positive=True)
    spectral_integration_model = _require_choice('spectral_integration_model', params['spectral_integration_model'])
    _finite_float('illumination_spectrum_center_nm', positive=True)
    _finite_float('illumination_spectrum_fwhm_nm', nonnegative=True)
    _positive_int('illumination_spectrum_num_samples')
    detector_spectral_response_model = _require_choice('detector_spectral_response_model', params['detector_spectral_response_model'])
    broadband_wavelengths = params['broadband_wavelengths_nm']
    broadband_weights = params['broadband_weights']
    if broadband_wavelengths is not None:
        arr = np.asarray(broadband_wavelengths, dtype=float).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("PARAMS['broadband_wavelengths_nm'] must be a non-empty positive finite sequence.")
        if broadband_weights is not None:
            warr = np.asarray(broadband_weights, dtype=float).reshape(-1)
            if warr.shape != arr.shape or not np.all(np.isfinite(warr)) or np.any(warr < 0.0):
                raise ValueError("PARAMS['broadband_weights'] must match wavelengths and be finite non-negative.")
            if float(np.sum(warr)) <= 0.0:
                raise ValueError("PARAMS['broadband_weights'] must have positive sum.")
    elif broadband_weights is not None:
        raise ValueError("PARAMS['broadband_weights'] requires PARAMS['broadband_wavelengths_nm'].")
    scene_dimensionality = _require_choice('scene_dimensionality', params['scene_dimensionality'])
    volumetric_mode = _require_choice('volumetric_imaging_mode', params['volumetric_imaging_mode'])
    if params['volumetric_z_planes_nm'] is not None:
        z_planes = np.asarray(params['volumetric_z_planes_nm'], dtype=float).reshape(-1)
        if z_planes.size == 0 or not np.all(np.isfinite(z_planes)):
            raise ValueError("PARAMS['volumetric_z_planes_nm'] must be a non-empty finite sequence when set.")
    _finite_float('volumetric_z_range_nm', nonnegative=True)
    _finite_float('volumetric_z_step_nm', positive=True)
    _positive_int('volumetric_z_count')
    volume_output_mode = _require_choice('volume_output_mode', params['volume_output_mode'])
    _finite_float('confocal_pinhole_sigma_nm', positive=True)
    _finite_float('light_sheet_center_z_nm')
    _finite_float('light_sheet_sigma_nm', positive=True)
    h_angles = np.asarray(params['holotomography_projection_angles_deg'], dtype=float).reshape(-1)
    if h_angles.size == 0 or not np.all(np.isfinite(h_angles)):
        raise ValueError("PARAMS['holotomography_projection_angles_deg'] must be a non-empty finite sequence.")
    holo_output = _require_choice('holotomography_output_mode', params['holotomography_output_mode'])
    zernike_model = _require_choice('zernike_model', params['zernike_model'])
    zernike_inner = _finite_float('zernike_phase_ring_inner_fraction', nonnegative=True)
    zernike_outer = _finite_float('zernike_phase_ring_outer_fraction', positive=True)
    if not zernike_inner < zernike_outer <= 1.0:
        raise ValueError(f'Zernike phase ring fractions must satisfy 0 <= inner < outer <= 1; got inner={zernike_inner}, outer={zernike_outer}.')
    _finite_float('zernike_phase_ring_amplitude', nonnegative=True)
    dpc_channel_model = _require_choice('dpc_channel_model', params['dpc_channel_model'])
    is_vectorial_dpc_channel = dpc_channel_model == 'vectorial_debye_asymmetric_illumination'
    dpc_transfer_model = _require_choice('dpc_transfer_model', params['dpc_transfer_model'])
    _finite_float('dpc_intensity_gain', nonnegative=True)
    _finite_float('dpc_intensity_gain_x', nonnegative=True)
    _finite_float('dpc_intensity_gain_y', nonnegative=True)
    vectorial_detection_mode = _require_choice('vectorial_detection_mode', params['vectorial_detection_mode'])
    if optical_field_backend == 'vectorial_debye' and imaging_model in COHERENT_REFERENCE_MODALITIES and (vectorial_detection_mode in {'incoherent_sum', 'unpolarized'}):
        raise ValueError(f"Incoherent vectorial detection reductions cannot be used as coherent complex fields for imaging_model={imaging_model!r}. Use analyzer_x, analyzer_y, full_vector, or optical_field_backend='scalar_paraxial'.")
    if optical_field_backend == 'vectorial_debye' and imaging_model in COHERENT_REFERENCE_MODALITIES and (vectorial_detection_mode == 'full_vector') and (polarization_model == 'unpolarized'):
        raise ValueError(f"polarization_model='unpolarized' is an incoherent average and cannot define the coherent reference field required by vectorial_detection_mode='full_vector' for imaging_model={imaging_model!r}. Use polarization_model='linear_x' or 'linear_y', an analyzer mode, or optical_field_backend='scalar_paraxial'.")
    if is_differential_phase_contrast and is_vectorial_dpc_channel:
        if optical_field_backend != 'vectorial_debye':
            raise ValueError("PARAMS['dpc_channel_model']='%s' requires PARAMS['optical_field_backend']='vectorial_debye' for differential_phase_contrast; got optical_field_backend=%r." % (dpc_channel_model, optical_field_backend))
        if vectorial_detection_mode != 'full_vector':
            raise ValueError(
                "PARAMS['dpc_channel_model']='vectorial_debye_asymmetric_illumination' "
                "requires PARAMS['vectorial_detection_mode']='full_vector' for "
                f"differential_phase_contrast; got {vectorial_detection_mode!r}."
            )
        if polarization_model == 'unpolarized':
            raise ValueError(
                "PARAMS['polarization_model']='unpolarized' is an incoherent average "
                "and cannot define a full-vector DPC phase-gradient observable. Use "
                "linear_x, linear_y, an analyzer mode, or scalar_paraxial optics."
            )
    dpc_output_channel = _require_choice('dpc_output_channel', params['dpc_output_channel'])
    _finite_float('dpc_phase_gradient_gain_x', nonnegative=True)
    _finite_float('dpc_phase_gradient_gain_y', nonnegative=True)
    if objective_model is not None and (not isinstance(objective_model, str)):
        raise ValueError(f"PARAMS['objective_model'] must be None or a string; got {objective_model!r}.")
    _finite_float('objective_focal_length_mm', positive=True)
    _positive_int('psf_oversampling_factor')
    _positive_int('pupil_samples')
    max_psf_z_slices = params['max_psf_z_slices']
    if max_psf_z_slices is not None:
        if isinstance(max_psf_z_slices, bool) or not isinstance(max_psf_z_slices, numbers.Integral):
            raise ValueError(f"PARAMS['max_psf_z_slices'] must be None or a positive integer; got {max_psf_z_slices!r}.")
        if int(max_psf_z_slices) <= 0:
            raise ValueError(f"PARAMS['max_psf_z_slices'] must be None or a positive integer; got {max_psf_z_slices!r}.")
    _finite_float('background_intensity', nonnegative=True)
    ann_inner = _finite_float('annular_dark_field_inner_sigma', positive=True)
    ann_outer = _finite_float('annular_dark_field_outer_sigma', positive=True)
    if ann_inner <= 1.0:
        raise ValueError(
            "PARAMS['annular_dark_field_inner_sigma'] must exceed 1.0 for dark-field illumination; "
            f"got {ann_inner!r}."
        )
    if ann_inner >= ann_outer:
        raise ValueError(
            "Annular dark-field source sigmas must satisfy inner < outer; "
            f"got inner={ann_inner!r}, outer={ann_outer!r}."
        )
    ann_outer_na = ann_outer * numerical_aperture
    if ann_outer_na > refractive_index_medium + 1e-12:
        raise ValueError(
            "Annular dark-field source exceeds the immersion-medium NA: "
            "annular_dark_field_outer_sigma * numerical_aperture must be <= refractive_index_medium; "
            f"got {ann_outer!r} * {numerical_aperture!r} = {ann_outer_na!r} > {refractive_index_medium!r}."
        )
    qpi_visibility = _finite_float('qpi_visibility', positive=True)
    if qpi_visibility > 1.0:
        raise ValueError("PARAMS['qpi_visibility'] must be <= 1.0.")
    if params['qpi_detected_quanta_per_pixel'] is not None:
        _finite_float('qpi_detected_quanta_per_pixel', positive=True)
    _finite_float('qpi_phase_to_count_scale', positive=True)
    _finite_float('fluorescence_excitation_wavelength_nm', positive=True)
    _finite_float('fluorescence_emission_wavelength_nm', positive=True)
    _finite_float('fluorescence_background', nonnegative=True)
    _finite_float('fluorescence_photon_count_scale', nonnegative=True)
    if params['fluorescence_photons_per_fluorophore_per_frame'] is not None:
        _finite_float('fluorescence_photons_per_fluorophore_per_frame', nonnegative=True)
    collection_eff = _finite_float('fluorescence_collection_efficiency', nonnegative=True)
    if collection_eff > 1.0:
        raise ValueError("PARAMS['fluorescence_collection_efficiency'] must be <= 1.0.")
    if 'detector_qe' in params:
        if _finite_float('detector_qe', nonnegative=True) > 1.0:
            raise ValueError("PARAMS['detector_qe'] must be <= 1.0.")
    if 'fluorescence_detector_qe' in params:
        if _finite_float('fluorescence_detector_qe', nonnegative=True) > 1.0:
            raise ValueError("PARAMS['fluorescence_detector_qe'] must be <= 1.0.")
    if params['fluorescence_emission_psf_sigma_nm'] is not None:
        _finite_float('fluorescence_emission_psf_sigma_nm', nonnegative=True)
    _finite_float('fluorescence_emission_psf_sigma_px', nonnegative=True)
    fluorescence_backend = _require_choice('fluorescence_backend', params['fluorescence_backend'])
    fluorescence_reference_status = _require_choice('fluorescence_reference_status', params['fluorescence_reference_status'])
    if fluorescence_reference_status == 'reference_validated' and (not params['fluorescence_reference_validation_hash']):
        raise ValueError("reference_validated fluorescence requires PARAMS['fluorescence_reference_validation_hash'].")
    _finite_float('fluorescence_blinking_rate_per_frame', nonnegative=True)
    _finite_float('fluorescence_recovery_rate_per_frame', nonnegative=True)
    _finite_float('fluorescence_bleaching_rate_per_frame', nonnegative=True)
    ricm_interface_model = _require_choice('ricm_interface_reflection_model', params['ricm_interface_reflection_model'])
    _finite_float('ricm_gap_nm', nonnegative=True)
    ricm_layers = params['ricm_thinfilm_layers']
    if ricm_layers is None:
        ricm_layers = []
    if not isinstance(ricm_layers, (list, tuple)):
        raise ValueError("PARAMS['ricm_thinfilm_layers'] must be a list of layer dictionaries.")
    for layer in ricm_layers:
        if not isinstance(layer, dict):
            raise ValueError('Each RICM thin-film layer must be a dictionary.')
        thickness = layer.get('thickness_nm', 0.0)
        if not isinstance(thickness, numbers.Real) or not np.isfinite(float(thickness)) or float(thickness) < 0.0:
            raise ValueError(f'RICM layer thickness_nm must be finite and non-negative; got {thickness!r}.')
    tem_model = _require_choice('tem_model', params['tem_model'])
    tem_backend = _require_choice('tem_backend', params['tem_backend'])
    if tem_backend == 'multislice_lite' and tem_model != 'multislice_lite':
        raise ValueError("PARAMS['tem_backend']='multislice_lite' requires PARAMS['tem_model']='multislice_lite'.")
    if tem_model == 'syniscopy_multislice' and tem_backend != 'syniscopy_multislice':
        raise ValueError("PARAMS['tem_model']='syniscopy_multislice' requires PARAMS['tem_backend']='syniscopy_multislice'.")
    if tem_model == 'multislice_physical' and tem_backend != 'multislice_physical':
        raise ValueError("PARAMS['tem_model']='multislice_physical' requires PARAMS['tem_backend']='multislice_physical'.")
    if tem_model == 'weak_phase_ctf' and tem_backend != 'ctf_proxy':
        raise ValueError("PARAMS['tem_model']='weak_phase_ctf' requires PARAMS['tem_backend']='ctf_proxy'.")
    if tem_backend == 'ctf_proxy' and tem_model != 'weak_phase_ctf':
        raise ValueError("PARAMS['tem_backend']='ctf_proxy' requires PARAMS['tem_model']='weak_phase_ctf'.")
    if tem_backend == 'multislice_physical' and tem_model != 'multislice_physical':
        raise ValueError("PARAMS['tem_backend']='multislice_physical' requires PARAMS['tem_model']='multislice_physical'.")
    tem_potential_source = _require_choice('tem_potential_source', params['tem_potential_source'])
    tem_reference_status = _require_choice('tem_reference_status', params['tem_reference_status'])
    if tem_reference_status == 'reference_validated' and (not params['tem_reference_validation_hash']):
        raise ValueError("reference_validated TEM requires PARAMS['tem_reference_validation_hash'].")
    _positive_int('tem_multislice_slices')
    if params['tem_slice_thickness_nm'] is not None:
        _finite_float('tem_slice_thickness_nm', positive=True)
    _finite_float('tem_dose_per_pixel', nonnegative=True)
    _finite_float('tem_projected_potential_scale', nonnegative=True)
    if params['tem_phase_shift_per_volt_nm'] is not None:
        _finite_float('tem_phase_shift_per_volt_nm', nonnegative=True)
    if params['tem_filter_guard_pixels'] is not None:
        _finite_float('tem_filter_guard_pixels', nonnegative=True)
    if params['tem_objective_aperture_mrad'] is not None:
        _finite_float('tem_objective_aperture_mrad', positive=True)
    if params['sem_filter_guard_pixels'] is not None:
        _finite_float('sem_filter_guard_pixels', nonnegative=True)
    if params['sem_probe_sigma_nm'] is not None:
        _finite_float('sem_probe_sigma_nm', nonnegative=True)
    sem_model = _require_choice('sem_model', params['sem_model'])
    sem_backend = _require_choice('sem_backend', params['sem_backend'])
    if sem_backend == 'interaction_volume_proxy' and sem_model != 'interaction_volume_proxy':
        raise ValueError("PARAMS['sem_backend']='interaction_volume_proxy' requires PARAMS['sem_model']='interaction_volume_proxy'.")
    if sem_model == 'interaction_volume_proxy' and sem_backend != 'interaction_volume_proxy':
        raise ValueError("PARAMS['sem_model']='interaction_volume_proxy' requires PARAMS['sem_backend']='interaction_volume_proxy'.")
    if sem_backend == 'monte_carlo_physical' and sem_model != 'physical_electron_transport':
        raise ValueError("PARAMS['sem_backend']='monte_carlo_physical' requires PARAMS['sem_model']='physical_electron_transport'.")
    if sem_model == 'physical_electron_transport' and sem_backend != 'monte_carlo_physical':
        raise ValueError("PARAMS['sem_model']='physical_electron_transport' requires PARAMS['sem_backend']='monte_carlo_physical'.")
    if sem_backend == 'reference_kernel_table' and (not params['sem_reference_kernel_path']):
        raise ValueError("PARAMS['sem_backend']='reference_kernel_table' requires PARAMS['sem_reference_kernel_path'].")
    sem_source_representation = _require_choice('sem_source_representation', params['sem_source_representation'])
    _positive_int('sem_volume_slices')
    if params['sem_volume_slice_thickness_nm'] is not None:
        _finite_float('sem_volume_slice_thickness_nm', positive=True)
    sem_source_z_origin = _require_choice('sem_source_z_origin', params['sem_source_z_origin'])
    _finite_float('sem_source_z_offset_nm')
    _finite_float('sem_beam_current_nA', nonnegative=True)
    _finite_float('sem_dwell_time_us', nonnegative=True)
    sem_takeoff_angle = _finite_float('sem_detector_takeoff_angle_deg', nonnegative=True)
    if sem_takeoff_angle > 90.0:
        raise ValueError(
            "PARAMS['sem_detector_takeoff_angle_deg'] is measured above the specimen surface "
            f"and must be <= 90 degrees; got {sem_takeoff_angle!r}."
        )
    _finite_float('sem_detector_acceptance', nonnegative=True)
    _finite_float('sem_escape_depth_nm', nonnegative=True)
    _finite_float('sem_backscatter_fraction', nonnegative=True)
    if float(params['sem_detector_acceptance']) > 1.0:
        raise ValueError("PARAMS['sem_detector_acceptance'] must be <= 1.0.")
    if float(params['sem_backscatter_fraction']) > 1.0:
        raise ValueError("PARAMS['sem_backscatter_fraction'] must be <= 1.0.")
    _finite_float('sem_transport_material_scale', nonnegative=True)
    if _finite_float('sem_transport_source_exponent', positive=True) < 0.05:
        raise ValueError("PARAMS['sem_transport_source_exponent'] must be >= 0.05.")
    if _finite_float('sem_transport_topography_exponent', positive=True) < 0.05:
        raise ValueError("PARAMS['sem_transport_topography_exponent'] must be >= 0.05.")
    _positive_int('sem_monte_carlo_trajectories')
    _positive_int('sem_monte_carlo_steps')
    if params['sem_monte_carlo_step_nm'] is not None:
        _finite_float('sem_monte_carlo_step_nm', positive=True)
    if params['sem_monte_carlo_range_nm'] is not None:
        _finite_float('sem_monte_carlo_range_nm', positive=True)
    _finite_float('sem_monte_carlo_scatter_std_deg', nonnegative=True)
    if params['sem_monte_carlo_kernel_size_px'] is not None:
        _positive_int('sem_monte_carlo_kernel_size_px')
    _positive_int('sem_physical_max_steps')
    _finite_float('sem_physical_energy_cutoff_keV', positive=True)
    _require_choice('sem_physical_elastic_model', params['sem_physical_elastic_model'])
    _finite_float('sem_reference_source_depth_nm', nonnegative=True)
    _finite_float('sem_reference_incident_angle_deg', nonnegative=True)
    if float(params['sem_reference_incident_angle_deg']) > 180.0:
        raise ValueError("PARAMS['sem_reference_incident_angle_deg'] must be <= 180.0.")
    reference_material = str(params['sem_reference_material']).strip().lower()
    if reference_material == '':
        raise ValueError("PARAMS['sem_reference_material'] must be a non-empty string.")
    reference_geometry = str(params['sem_reference_geometry']).strip().lower()
    if reference_geometry == '':
        raise ValueError("PARAMS['sem_reference_geometry'] must be a non-empty string.")
    _finite_float('sem_interaction_volume_nm', nonnegative=True)
    _finite_float('sem_topography_contrast_gain', nonnegative=True)
    detector_direction = params['sem_detector_direction_xy']
    direction_arr = np.asarray(detector_direction, dtype=float)
    if direction_arr.shape != (2,) or not np.all(np.isfinite(direction_arr)):
        raise ValueError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
    if float(np.linalg.norm(direction_arr)) <= 0.0:
        raise ValueError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
    _finite_float('sem_probe_sigma_pixels', nonnegative=True)
    _finite_float('supervision_prior_log_odds')
    _finite_float('supervision_log_odds_threshold')
    log_odds_clip_epsilon = _finite_float('supervision_log_odds_clip_epsilon')
    if not 0.0 < log_odds_clip_epsilon < 0.5:
        raise ValueError(f"PARAMS['supervision_log_odds_clip_epsilon'] must lie in (0, 0.5); got {log_odds_clip_epsilon}.")
    _finite_float('fisher_lateral_step_nm', positive=True)
    _nonnegative_int('fisher_particle_index')
    fisher_likelihood_model = _require_choice('fisher_likelihood_model', params['fisher_likelihood_model'])
    detected_quanta_derivative_target = _require_choice('detected_quanta_derivative_target', params['detected_quanta_derivative_target'])
    fisher_lateral_derivative_mode = _require_choice('fisher_lateral_derivative_mode', params['fisher_lateral_derivative_mode'])
    supervision_decision_rule = _require_choice('supervision_decision_rule', params['supervision_decision_rule'])
    supervision_calibration_mode = _require_choice('supervision_score_calibration_mode', params['supervision_score_calibration_mode'])
    calibration_parameters = params['supervision_score_calibration_parameters']
    if calibration_parameters is not None and (not isinstance(calibration_parameters, dict)):
        raise ValueError("PARAMS['supervision_score_calibration_parameters'] must be None or a dict.")
    matched_modalities = params['matched_modalities']
    if matched_modalities is not None:
        if isinstance(matched_modalities, (str, bytes)) or not isinstance(matched_modalities, (list, tuple)) or len(matched_modalities) < 2:
            raise ValueError("PARAMS['matched_modalities'] must be None or a list/tuple of at least two imaging model names.")
        for modality in matched_modalities:
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("PARAMS['matched_modalities'] entries must be non-empty strings.")
    source_only_models = {'tem_phase_contrast', 'sem_secondary_electron', 'fluorescence_widefield', 'tirf_fluorescence'}
    imaging_model = str(params['imaging_model']).strip().lower()
    if imaging_model not in source_only_models and numerical_aperture > refractive_index_medium:
        raise ValueError(f"PARAMS['numerical_aperture'] must be <= PARAMS['refractive_index_medium']; got {numerical_aperture} > {refractive_index_medium}.")
    exposure_time_ms = params['exposure_time_ms']
    if exposure_time_ms is not None:
        exposure_time_ms = _finite_float('exposure_time_ms', positive=True)
        frame_interval_ms = 1000.0 / fps
        if exposure_time_ms > frame_interval_ms:
            raise ValueError(f"PARAMS['exposure_time_ms'] must be <= 1000 / PARAMS['fps']; got {exposure_time_ms} ms with fps={fps}.")
    num_frames = params['num_frames']
    if num_frames is not None:
        if isinstance(num_frames, bool) or not isinstance(num_frames, numbers.Integral):
            raise ValueError(f"PARAMS['num_frames'] must be None or a positive integer; got {num_frames!r}.")
        if int(num_frames) <= 0:
            raise ValueError(f"PARAMS['num_frames'] must be None or a positive integer; got {num_frames!r}.")
    if params['qpi_phase_noise_std_rad'] is not None:
        _finite_float('qpi_phase_noise_std_rad', nonnegative=True)
    _finite_float('dynamic_process_noise_scale', nonnegative=True)
    _finite_float('dynamic_initial_variance_nm2', positive=True)
    bit_depth = params['bit_depth']
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, numbers.Integral):
        raise ValueError("PARAMS['bit_depth'] must be an integer in the range [1, 16].")
    if int(bit_depth) < 1 or int(bit_depth) > 16:
        raise ValueError(f"PARAMS['bit_depth'] must be an integer in the range [1, 16]; got {bit_depth!r}.")
    hot_pixel_fraction = _finite_float('hot_pixel_fraction', nonnegative=True)
    if hot_pixel_fraction > 1.0:
        raise ValueError(f"PARAMS['hot_pixel_fraction'] must be <= 1.0; got {hot_pixel_fraction}.")
    pattern_model = _require_choice('sample_environment_pattern', params['sample_environment_pattern'])
    pattern_dimensions = params['sample_environment_pattern_dimensions']
    if not isinstance(pattern_dimensions, dict):
        raise ValueError("PARAMS['sample_environment_pattern_dimensions'] must be a dictionary.")
    if "microfluidic_wall_orientation" in pattern_dimensions:
        orientation = str(pattern_dimensions["microfluidic_wall_orientation"]).strip().lower()
        if orientation not in BAR_ORIENTATION_CHOICES:
            raise ValueError(
                "PARAMS['sample_environment_pattern_dimensions']['microfluidic_wall_orientation'] "
                f"must be one of {BAR_ORIENTATION_CHOICES!r}."
            )
    pattern_enabled = bool(params['sample_environment_pattern_enabled'])
    environment_enabled = bool(params['sample_environment_enabled'])
    if environment_enabled and pattern_enabled:
        preset = str(params['sample_environment_pattern_preset']).strip().lower()
        allowed_presets = {'empty_background'}
        if pattern_model in PATTERN_DEFAULT_PRESETS:
            allowed_presets.add(PATTERN_DEFAULT_PRESETS[pattern_model])
        if preset not in allowed_presets:
            raise ValueError(f"PARAMS['sample_environment_pattern_preset'] is not valid for {pattern_model!r}. Allowed values are {sorted(allowed_presets)}; got {params['sample_environment_pattern_preset']!r}.")
    roughness_model = _require_choice(
        'sample_environment_pattern_roughness_model',
        params['sample_environment_pattern_roughness_model'],
    )
    roughness_source = params['sample_environment_pattern_roughness_source']
    if roughness_model == 'source_matched' and roughness_source is None:
        raise ValueError("PARAMS['sample_environment_pattern_roughness_source'] must be provided when sample_environment_pattern_roughness_model='source_matched'.")
    if roughness_source is not None and (not isinstance(roughness_source, (str, bytes, os.PathLike, list, tuple, np.ndarray))):
        raise ValueError("PARAMS['sample_environment_pattern_roughness_source'] must be a file path or an array-like roughness field when set.")
    roughness_amplitude = _finite_float('sample_environment_pattern_roughness_amplitude', nonnegative=True)
    if roughness_model == 'none' and roughness_amplitude > 0.0:
        raise ValueError("PARAMS['sample_environment_pattern_roughness_amplitude'] can only be non-zero when sample_environment_pattern_roughness_model is 'static', 'flicker', or 'source_matched'.")
    roughness_correlation_pixels = _finite_float('sample_environment_pattern_roughness_correlation_pixels', nonnegative=True)
    if 0.0 < roughness_correlation_pixels < 1.0:
        raise ValueError("PARAMS['sample_environment_pattern_roughness_correlation_pixels'] must be at least 1.0 when positive.")
    roughness_phase_std = _finite_float('sample_environment_pattern_roughness_phase_std', nonnegative=True)
    if roughness_phase_std > np.pi:
        raise ValueError("PARAMS['sample_environment_pattern_roughness_phase_std'] must be <= pi.")
    roughness_source_coupling = _require_choice(
        'sample_environment_pattern_roughness_source_coupling',
        params['sample_environment_pattern_roughness_source_coupling'],
    )
    detector_qe_public = _finite_float('detector_qe', nonnegative=True)
    if detector_qe_public > 1.0:
        raise ValueError("PARAMS['detector_qe'] must be <= 1.0.")
    _finite_float('emccd_excess_noise_factor', positive=True)
    if float(params['emccd_excess_noise_factor']) < 1.0:
        raise ValueError("PARAMS['emccd_excess_noise_factor'] must be >= 1.0.")
    if params['read_noise_e'] is not None:
        _finite_float('read_noise_e', nonnegative=True)
    _finite_float('dark_current_e_per_pixel_per_s', nonnegative=True)
    _finite_float('exposure_time_s', positive=True)
    _finite_float('dark_offset_counts', nonnegative=True)
    if params['hot_pixel_value_counts'] is not None:
        _finite_float('hot_pixel_value_counts', nonnegative=True)
    if params['saturation_e'] is not None:
        _finite_float('saturation_e', positive=True)
    for key in ('fixed_pattern_gain_map', 'fixed_pattern_offset_map', 'scmos_gain_map', 'hot_pixel_mask', 'scmos_variance_map', 'scmos_read_noise_map', 'flat_field_map', 'dark_frame_map'):
        _validate_noise_map_value(f'PARAMS[{key!r}]', params[key])
    if params['detector_noise_input_domain'] is not None:
        normalize_detector_noise_input_domain(params['detector_noise_input_domain'])
    read_noise_map_mode = _require_choice('read_noise_map_mode', params['read_noise_map_mode'])
    noise_keys = {
        key
        for key, spec in PARAM_SCHEMA.items()
        if spec.get('group') == 'Noise' and key not in {'noise_model', 'modality_noise'}
    }
    noise_model = params['noise_model']
    if noise_model is None:
        noise_model = {}
    if not isinstance(noise_model, dict):
        raise TypeError("PARAMS['noise_model'] must be a dictionary when provided.")
    unknown_noise = sorted((str(key) for key in noise_model if str(key) not in noise_keys))
    if unknown_noise:
        raise ValueError("Unknown PARAMS['noise_model'] key(s): " + ', '.join((repr(key) for key in unknown_noise)) + '. Use the counts-domain noise keys documented in config.PARAMS.')
    for key in ('shot_noise_enabled', 'gaussian_noise_enabled', 'clip_output_to_nonnegative', 'detector_input_is_incident_quanta', 'emccd_enabled', 'adc_quantization'):
        if key in noise_model:
            _bool_value(f"PARAMS['noise_model']['{key}']", noise_model[key])
    for key in ('fixed_pattern_gain_map', 'fixed_pattern_offset_map', 'scmos_gain_map', 'hot_pixel_mask', 'scmos_variance_map', 'scmos_read_noise_map', 'flat_field_map', 'dark_frame_map'):
        if key in noise_model:
            _validate_noise_map_value(f"PARAMS['noise_model']['{key}']", noise_model[key])
    if 'detector_noise_input_domain' in noise_model:
        normalize_detector_noise_input_domain(noise_model['detector_noise_input_domain'])
    if 'read_noise_map_mode' in noise_model:
        mode = _require_choice('read_noise_map_mode', noise_model['read_noise_map_mode'])
    if 'noise_parameterization' in noise_model:
        nested_parameterization = str(noise_model['noise_parameterization']).strip().lower()
        if nested_parameterization != 'camera_counts':
            raise ValueError(f"PARAMS['noise_model']['noise_parameterization'] must be 'camera_counts'; got {noise_model['noise_parameterization']!r}.")
    modality_noise = params['modality_noise']
    if modality_noise is None:
        modality_noise = {}
    if not isinstance(modality_noise, dict):
        raise TypeError("PARAMS['modality_noise'] must be a dictionary when provided.")
    for (modality_name, overrides) in modality_noise.items():
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise TypeError(f"Each PARAMS['modality_noise'] entry must be a dictionary; got {type(overrides).__name__} for {modality_name!r}.")
        unknown_modality_noise = sorted((str(key) for key in overrides if str(key) not in noise_keys))
        if unknown_modality_noise:
            raise ValueError(f"Unknown PARAMS['modality_noise'][{modality_name!r}] key(s): " + ', '.join((repr(key) for key in unknown_modality_noise)) + '. Use the counts-domain noise keys documented in config.PARAMS.')
        for key in ('shot_noise_enabled', 'gaussian_noise_enabled', 'clip_output_to_nonnegative', 'detector_input_is_incident_quanta', 'emccd_enabled', 'adc_quantization'):
            if key in overrides:
                _bool_value(f"PARAMS['modality_noise'][{modality_name!r}]['{key}']", overrides[key])
        for key in ('fixed_pattern_gain_map', 'fixed_pattern_offset_map', 'scmos_gain_map', 'hot_pixel_mask', 'scmos_variance_map', 'scmos_read_noise_map', 'flat_field_map', 'dark_frame_map'):
            if key in overrides:
                _validate_noise_map_value(f"PARAMS['modality_noise'][{modality_name!r}]['{key}']", overrides[key])
        if 'detector_noise_input_domain' in overrides:
            normalize_detector_noise_input_domain(overrides['detector_noise_input_domain'])
        if 'read_noise_map_mode' in overrides:
            mode = _require_choice('read_noise_map_mode', overrides['read_noise_map_mode'])
        if 'noise_parameterization' in overrides:
            nested_parameterization = str(overrides['noise_parameterization']).strip().lower()
            if nested_parameterization != 'camera_counts':
                raise ValueError("PARAMS['modality_noise'][%r]['noise_parameterization'] must be 'camera_counts'; got %r." % (modality_name, overrides['noise_parameterization']))

def normalize_params(params: dict, *, allowed_extra_keys: set[str] | None=None, allowed_internal_keys: set[str] | None=None) -> dict:
    """
    Return a validated copy of ``params`` with canonical dependent values set.

    This is the entry point for runtime paths that need validation plus
    normalization without mutating the caller's dictionary.
    """
    normalized = deepcopy(params)
    _normalize_params_in_place(normalized, allowed_extra_keys=allowed_extra_keys, allowed_internal_keys=allowed_internal_keys)
    return normalized

def validate_params(params: dict, *, allowed_extra_keys: set[str] | None=None, allowed_internal_keys: set[str] | None=None) -> None:
    """
    Validate the public PARAMS surface without mutating ``params``.

    Syniscopy v1 has one canonical public key per concept. Unknown keys raise
    immediately so aliases and typo-driven configuration drift do not enter
    generated datasets or manuscript artifacts.
    """
    _normalize_params_in_place(dict(params), allowed_extra_keys=allowed_extra_keys, allowed_internal_keys=allowed_internal_keys)
__all__ = ['_normalize_params_in_place', 'normalize_params', 'validate_params']
