"""Parameter normalization and validation rules."""

from __future__ import annotations

import math
import numbers
import os
from copy import deepcopy

import numpy as np

from shared_constants import COHERENT_REFERENCE_MODALITIES, PATTERN_DEFAULT_PRESETS

from .defaults import PARAMS, _KNOWN_INTERNAL_PARAM_KEYS


def _normalize_params_in_place(
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

    This private implementation normalizes a small number of
    dependent/canonical values in ``params`` in place. Public callers should use
    ``normalize_params`` when they need those values and ``validate_params``
    when they only need a non-mutating validation check.
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

    def _validate_noise_map_value(key: str, value) -> None:
        if value is None:
            return
        if isinstance(value, (bool, str, bytes, os.PathLike)):
            return
        if isinstance(value, (int, float, np.integer, np.floating, list, tuple, np.ndarray)):
            return
        raise ValueError(f"{key} must be a noise-map-like value; got {type(value).__name__}.")

    for key, default_value in PARAMS.items():
        if isinstance(default_value, bool):
            _bool_value(f"PARAMS['{key}']", params.get(key, default_value))

    imaging_model_input = params.get("imaging_model", PARAMS.get("imaging_model", "bright_field"))
    imaging_model = str(imaging_model_input).strip().lower()
    if not imaging_model:
        imaging_model = str(PARAMS.get("imaging_model", "bright_field")).strip().lower()
    supported_imaging_models: set[str] | None = None
    try:
        from modality_registry import SUPPORTED_MODALITIES, canonical_modality_name

        imaging_model = canonical_modality_name(imaging_model)
        supported_imaging_models = set(SUPPORTED_MODALITIES)
    except ImportError:
        supported_imaging_models = None
    if supported_imaging_models is not None and imaging_model not in supported_imaging_models:
        raise ValueError(
            "PARAMS['imaging_model'] must be one of "
            f"{sorted(supported_imaging_models)}; got {imaging_model_input!r}."
        )

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
    optical_field_backend = str(
        params.get("optical_field_backend", PARAMS.get("optical_field_backend", "vectorial_debye"))
    ).strip().lower()
    is_differential_phase_contrast = imaging_model == "differential_phase_contrast"
    optical_backend_explicit = "optical_field_backend" in params
    vectorial_detection_mode_explicit = "vectorial_detection_mode" in params
    if optical_field_backend not in {"scalar_paraxial", "vectorial_debye"}:
        raise ValueError(
            "PARAMS['optical_field_backend'] must be 'scalar_paraxial' or "
            f"'vectorial_debye'; got {optical_field_backend!r}."
        )
    polarization_model = str(
        params.get("polarization_model", PARAMS.get("polarization_model", "scalar"))
    ).strip().lower()
    if polarization_model == "scalar":
        polarization_model = "linear_x"
    if polarization_model not in {"linear_x", "linear_y", "unpolarized"}:
        raise ValueError(
            "PARAMS['polarization_model'] must be 'linear_x', 'linear_y', "
            f"or 'unpolarized'; got {polarization_model!r}."
        )
    rotation_deg = float(params.get("vectorial_polarization_rotation_deg", PARAMS.get("vectorial_polarization_rotation_deg")))
    if not np.isfinite(rotation_deg):
        raise ValueError(
            "PARAMS['vectorial_polarization_rotation_deg'] must be finite; "
            f"got {rotation_deg!r}."
        )
    vectorial_pupil_samples = params.get(
        "vectorial_pupil_samples",
        PARAMS.get("vectorial_pupil_samples"),
    )
    if vectorial_pupil_samples is not None:
        if isinstance(vectorial_pupil_samples, bool) or not isinstance(vectorial_pupil_samples, numbers.Integral):
            raise ValueError(
                "PARAMS['vectorial_pupil_samples'] must be None or a positive integer; "
                f"got {vectorial_pupil_samples!r}."
            )
        if int(vectorial_pupil_samples) <= 0:
            raise ValueError(
                "PARAMS['vectorial_pupil_samples'] must be None or a positive integer; "
                f"got {vectorial_pupil_samples!r}."
            )
    coverslip_model = str(
        params.get("coverslip_aberration_model", PARAMS.get("coverslip_aberration_model", "none"))
    ).strip().lower()
    if coverslip_model not in {"none", "disabled", "off", "gibson_lanni", "coverslip_mismatch"}:
        raise ValueError(
            "PARAMS['coverslip_aberration_model'] must be 'none', 'gibson_lanni', "
            f"or 'coverslip_mismatch'; got {coverslip_model!r}."
        )
    _finite_float("coverslip_thickness_um", nonnegative=True)
    _finite_float("coverslip_design_thickness_um", nonnegative=True)
    _finite_float("coverslip_refractive_index", positive=True)
    _finite_float("coverslip_design_refractive_index", positive=True)

    spectral_integration_model = str(
        params.get("spectral_integration_model", PARAMS.get("spectral_integration_model", "single_wavelength"))
    ).strip().lower()
    if spectral_integration_model not in {"single_wavelength", "configured_channels", "broadband_quadrature"}:
        raise ValueError(
            "PARAMS['spectral_integration_model'] must be 'single_wavelength', "
            f"'configured_channels', or 'broadband_quadrature'; got {spectral_integration_model!r}."
        )
    _finite_float("illumination_spectrum_center_nm", positive=True)
    _finite_float("illumination_spectrum_fwhm_nm", nonnegative=True)
    _positive_int("illumination_spectrum_num_samples")
    detector_spectral_response_model = str(
        params.get("detector_spectral_response_model", PARAMS.get("detector_spectral_response_model", "rgb_heuristic"))
    ).strip().lower()
    if detector_spectral_response_model not in {"rgb_heuristic", "flat", "table"}:
        raise ValueError(
            "PARAMS['detector_spectral_response_model'] must be 'rgb_heuristic', "
            f"'flat', or 'table'; got {detector_spectral_response_model!r}."
        )
    broadband_wavelengths = params.get("broadband_wavelengths_nm", PARAMS.get("broadband_wavelengths_nm"))
    broadband_weights = params.get("broadband_weights", PARAMS.get("broadband_weights"))
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

    scene_dimensionality = str(
        params.get("scene_dimensionality", PARAMS.get("scene_dimensionality", "single_plane_particle_scene"))
    ).strip().lower()
    if scene_dimensionality not in {"single_plane_particle_scene", "particle_volume_3d"}:
        raise ValueError(
            "PARAMS['scene_dimensionality'] must be 'single_plane_particle_scene' "
            f"or 'particle_volume_3d'; got {scene_dimensionality!r}."
        )
    volumetric_mode = str(
        params.get("volumetric_imaging_mode", PARAMS.get("volumetric_imaging_mode", "single_plane"))
    ).strip().lower()
    if volumetric_mode not in {"single_plane", "z_stack", "confocal", "light_sheet", "holotomography_projection"}:
        raise ValueError(
            "PARAMS['volumetric_imaging_mode'] must be one of 'single_plane', "
            "'z_stack', 'confocal', 'light_sheet', or 'holotomography_projection'; "
            f"got {volumetric_mode!r}."
        )
    if params.get("volumetric_z_planes_nm", PARAMS.get("volumetric_z_planes_nm")) is not None:
        z_planes = np.asarray(params.get("volumetric_z_planes_nm"), dtype=float).reshape(-1)
        if z_planes.size == 0 or not np.all(np.isfinite(z_planes)):
            raise ValueError("PARAMS['volumetric_z_planes_nm'] must be a non-empty finite sequence when set.")
    _finite_float("volumetric_z_range_nm", nonnegative=True)
    _finite_float("volumetric_z_step_nm", positive=True)
    _positive_int("volumetric_z_count")
    volume_output_mode = str(params.get("volume_output_mode", PARAMS.get("volume_output_mode", "integrated_projection"))).strip().lower()
    if volume_output_mode not in {"integrated_projection", "z_stack"}:
        raise ValueError("PARAMS['volume_output_mode'] must be 'integrated_projection' or 'z_stack'.")
    _finite_float("confocal_pinhole_sigma_nm", positive=True)
    _finite_float("light_sheet_center_z_nm")
    _finite_float("light_sheet_sigma_nm", positive=True)
    h_angles = np.asarray(
        params.get("holotomography_projection_angles_deg", PARAMS.get("holotomography_projection_angles_deg", [0.0])),
        dtype=float,
    ).reshape(-1)
    if h_angles.size == 0 or not np.all(np.isfinite(h_angles)):
        raise ValueError("PARAMS['holotomography_projection_angles_deg'] must be a non-empty finite sequence.")
    holo_output = str(params.get("holotomography_output_mode", PARAMS.get("holotomography_output_mode", "phase_projection_stack"))).strip().lower()
    if holo_output not in {"phase_projection_stack", "reconstruction_volume"}:
        raise ValueError("PARAMS['holotomography_output_mode'] must be 'phase_projection_stack' or 'reconstruction_volume'.")
    zernike_model = str(
        params.get("zernike_model", PARAMS.get("zernike_model", "fourier_phase_ring_proxy"))
    ).strip().lower()
    if zernike_model not in {"pupil_phase_ring", "fourier_phase_ring_proxy", "legacy_scalar_phase_proxy"}:
        raise ValueError(
            "PARAMS['zernike_model'] must be 'pupil_phase_ring', "
            "'fourier_phase_ring_proxy', or "
            f"'legacy_scalar_phase_proxy'; got {zernike_model!r}."
        )
    zernike_inner = _finite_float("zernike_phase_ring_inner_fraction", nonnegative=True)
    zernike_outer = _finite_float("zernike_phase_ring_outer_fraction", positive=True)
    if not (zernike_inner < zernike_outer <= 1.0):
        raise ValueError(
            "Zernike phase ring fractions must satisfy 0 <= inner < outer <= 1; "
            f"got inner={zernike_inner}, outer={zernike_outer}."
        )
    _finite_float("zernike_phase_ring_amplitude", nonnegative=True)
    dpc_channel_model = str(
        params.get(
            "dpc_channel_model",
            PARAMS.get("dpc_channel_model", "vectorial_debye_asymmetric_illumination"),
        )
    ).strip().lower()
    if dpc_channel_model not in {
        "two_axis_scalar_asymmetric_illumination",
        "scalar_asymmetric_illumination",
        "scalar",
        "vectorial_debye_asymmetric_illumination",
        "two_axis_vectorial_debye_asymmetric_illumination",
        "vectorial",
    }:
        raise ValueError(
            "PARAMS['dpc_channel_model'] must be one of {'two_axis_scalar_asymmetric_illumination', "
            "'vectorial_debye_asymmetric_illumination'} or legacy aliases; "
            f"got {dpc_channel_model!r}."
        )
    is_vectorial_dpc_channel = dpc_channel_model in {
        "vectorial_debye_asymmetric_illumination",
        "two_axis_vectorial_debye_asymmetric_illumination",
        "vectorial",
    }
    if is_differential_phase_contrast and is_vectorial_dpc_channel:
        if not optical_backend_explicit:
            params["optical_field_backend"] = "vectorial_debye"
            optical_field_backend = "vectorial_debye"
        if not vectorial_detection_mode_explicit:
            params["vectorial_detection_mode"] = "full_vector"
    dpc_transfer_model = str(
        params.get(
            "dpc_transfer_model",
            PARAMS.get("dpc_transfer_model", "pupil_half_plane_intensity"),
        )
    ).strip().lower()
    if dpc_transfer_model not in {
        "pupil_half_plane_intensity",
        "half_pupil_intensity",
        "asymmetric_pupil_intensity",
        "phase_gradient_proxy",
        "phase_gradient",
        "legacy_phase_gradient",
    }:
        raise ValueError(
            "PARAMS['dpc_transfer_model'] must be 'pupil_half_plane_intensity' "
            f"or 'phase_gradient_proxy'; got {dpc_transfer_model!r}."
        )
    _finite_float("dpc_intensity_gain", nonnegative=True)
    _finite_float("dpc_intensity_gain_x", nonnegative=True)
    _finite_float("dpc_intensity_gain_y", nonnegative=True)
    vectorial_detection_mode = str(
        params.get(
            "vectorial_detection_mode",
            PARAMS.get("vectorial_detection_mode", "full_vector"),
        )
    ).strip().lower()
    if vectorial_detection_mode not in {
        "incoherent_sum",
        "analyzer_x",
        "analyzer_y",
        "unpolarized",
        "full_vector",
    }:
        raise ValueError(
            "PARAMS['vectorial_detection_mode'] must be 'incoherent_sum', "
            f"'analyzer_x', 'analyzer_y', 'unpolarized', or 'full_vector'; got "
            f"{vectorial_detection_mode!r}."
        )
    if (
        optical_field_backend == "vectorial_debye"
        and imaging_model in COHERENT_REFERENCE_MODALITIES
        and vectorial_detection_mode in {"incoherent_sum", "unpolarized"}
    ):
        raise ValueError(
            "Incoherent vectorial detection reductions cannot be used as "
            f"coherent complex fields for imaging_model={imaging_model!r}. "
            "Use analyzer_x, analyzer_y, full_vector, or "
            "optical_field_backend='scalar_paraxial'."
        )
    if (
        optical_field_backend == "vectorial_debye"
        and imaging_model in COHERENT_REFERENCE_MODALITIES
        and vectorial_detection_mode == "full_vector"
        and polarization_model == "unpolarized"
    ):
        raise ValueError(
            "polarization_model='unpolarized' is an incoherent average and "
            "cannot define the coherent reference field required by "
            f"vectorial_detection_mode='full_vector' for imaging_model={imaging_model!r}. "
            "Use polarization_model='linear_x' or 'linear_y', an analyzer mode, "
            "or optical_field_backend='scalar_paraxial'."
        )
    if is_differential_phase_contrast and is_vectorial_dpc_channel:
        if optical_field_backend != "vectorial_debye":
            raise ValueError(
                "PARAMS['dpc_channel_model']='%s' requires "
                "PARAMS['optical_field_backend']='vectorial_debye' for differential_phase_contrast; "
                "got optical_field_backend=%r."
                % (dpc_channel_model, optical_field_backend)
            )
    dpc_output_channel = str(
        params.get("dpc_output_channel", PARAMS.get("dpc_output_channel", "x"))
    ).strip().lower()
    if dpc_output_channel not in {"x", "y", "diagonal", "magnitude"}:
        raise ValueError(
            "PARAMS['dpc_output_channel'] must be x, y, diagonal, or magnitude; "
            f"got {dpc_output_channel!r}."
        )
    _finite_float("dpc_phase_gradient_gain_x", nonnegative=True)
    _finite_float("dpc_phase_gradient_gain_y", nonnegative=True)
    if objective_model is not None and not isinstance(objective_model, str):
        raise ValueError(
            "PARAMS['objective_model'] must be None or a string; "
            f"got {objective_model!r}."
        )
    _finite_float("objective_focal_length_mm", positive=True)
    _positive_int("psf_oversampling_factor")
    _positive_int("pupil_samples")
    max_psf_z_slices = params.get("max_psf_z_slices", PARAMS.get("max_psf_z_slices"))
    if max_psf_z_slices is not None:
        if isinstance(max_psf_z_slices, bool) or not isinstance(max_psf_z_slices, numbers.Integral):
            raise ValueError(
                "PARAMS['max_psf_z_slices'] must be None or a positive integer; "
                f"got {max_psf_z_slices!r}."
            )
        if int(max_psf_z_slices) <= 0:
            raise ValueError(
                "PARAMS['max_psf_z_slices'] must be None or a positive integer; "
                f"got {max_psf_z_slices!r}."
            )
    _finite_float("background_intensity", nonnegative=True)
    qpi_visibility = _finite_float("qpi_visibility", positive=True)
    if qpi_visibility > 1.0:
        raise ValueError("PARAMS['qpi_visibility'] must be <= 1.0.")
    if params.get("qpi_detected_quanta_per_pixel", None) is not None:
        _finite_float("qpi_detected_quanta_per_pixel", positive=True)
    _finite_float("qpi_phase_to_count_scale", positive=True)
    _finite_float("fluorescence_excitation_wavelength_nm", positive=True)
    _finite_float("fluorescence_emission_wavelength_nm", positive=True)
    _finite_float("fluorescence_background", nonnegative=True)
    _finite_float("fluorescence_photon_count_scale", nonnegative=True)
    if params.get("fluorescence_photons_per_fluorophore_per_frame", None) is not None:
        _finite_float("fluorescence_photons_per_fluorophore_per_frame", nonnegative=True)
    collection_eff = _finite_float("fluorescence_collection_efficiency", nonnegative=True)
    if collection_eff > 1.0:
        raise ValueError("PARAMS['fluorescence_collection_efficiency'] must be <= 1.0.")
    if "detector_qe" in params:
        if _finite_float("detector_qe", nonnegative=True) > 1.0:
            raise ValueError("PARAMS['detector_qe'] must be <= 1.0.")
    if "fluorescence_detector_qe" in params:
        if _finite_float("fluorescence_detector_qe", nonnegative=True) > 1.0:
            raise ValueError("PARAMS['fluorescence_detector_qe'] must be <= 1.0.")
    if params.get("fluorescence_emission_psf_sigma_nm", None) is not None:
        _finite_float("fluorescence_emission_psf_sigma_nm", nonnegative=True)
    _finite_float("fluorescence_emission_psf_sigma_px", nonnegative=True)
    fluorescence_backend = str(
        params.get(
            "fluorescence_backend", PARAMS.get("fluorescence_backend", "vectorial_photophysics")
        )
    ).strip().lower()
    if fluorescence_backend not in {"parametric_psf", "vectorial_photophysics"}:
        raise ValueError(
            "PARAMS['fluorescence_backend'] must be 'parametric_psf' or "
            f"'vectorial_photophysics'; got {fluorescence_backend!r}."
        )
    fluorescence_reference_status = str(params.get("fluorescence_reference_status", PARAMS.get("fluorescence_reference_status", "physics_based_unvalidated"))).strip().lower()
    if fluorescence_reference_status not in {"physics_based_unvalidated", "reference_validated"}:
        raise ValueError("PARAMS['fluorescence_reference_status'] must be 'physics_based_unvalidated' or 'reference_validated'.")
    if fluorescence_reference_status == "reference_validated" and not params.get("fluorescence_reference_validation_hash", PARAMS.get("fluorescence_reference_validation_hash")):
        raise ValueError("reference_validated fluorescence requires PARAMS['fluorescence_reference_validation_hash'].")
    _finite_float("fluorescence_blinking_rate_per_frame", nonnegative=True)
    _finite_float("fluorescence_recovery_rate_per_frame", nonnegative=True)
    _finite_float("fluorescence_bleaching_rate_per_frame", nonnegative=True)
    ricm_interface_model = str(
        params.get("ricm_interface_reflection_model", PARAMS.get("ricm_interface_reflection_model", "param"))
    ).strip().lower()
    if ricm_interface_model not in {"param", "fresnel", "thin_film_stack"}:
        raise ValueError(
            "PARAMS['ricm_interface_reflection_model'] must be 'param', 'fresnel', "
            f"or 'thin_film_stack'; got {ricm_interface_model!r}."
        )
    ricm_layers = params.get("ricm_thinfilm_layers", PARAMS.get("ricm_thinfilm_layers", []))
    if ricm_layers is None:
        ricm_layers = []
    if not isinstance(ricm_layers, (list, tuple)):
        raise ValueError("PARAMS['ricm_thinfilm_layers'] must be a list of layer dictionaries.")
    for layer in ricm_layers:
        if not isinstance(layer, dict):
            raise ValueError("Each RICM thin-film layer must be a dictionary.")
        thickness = layer.get("thickness_nm", 0.0)
        if not isinstance(thickness, numbers.Real) or not np.isfinite(float(thickness)) or float(thickness) < 0.0:
            raise ValueError(f"RICM layer thickness_nm must be finite and non-negative; got {thickness!r}.")
    tem_model_explicit = "tem_model" in params
    tem_backend_explicit = "tem_backend" in params
    tem_model = str(params.get("tem_model", PARAMS.get("tem_model", "syniscopy_multislice"))).strip().lower()
    if tem_model == "weak_phase":
        tem_model = "weak_phase_ctf"
    if tem_model == "ctf_proxy":
        tem_model = "weak_phase_ctf"
    if tem_model not in {"weak_phase_ctf", "multislice_lite", "syniscopy_multislice"}:
        raise ValueError(
            "PARAMS['tem_model'] must be 'weak_phase_ctf' (or legacy alias "
            "'ctf_proxy'), 'multislice_lite', or 'syniscopy_multislice'; "
            f"got {tem_model!r}."
        )
    tem_backend = str(params.get("tem_backend", PARAMS.get("tem_backend", "syniscopy_multislice"))).strip().lower()
    if tem_backend in {"weak_phase_ctf", "weak_phase"}:
        tem_backend = "ctf_proxy"
    if tem_backend not in {"ctf_proxy", "multislice_lite", "syniscopy_multislice"}:
        raise ValueError(
            "PARAMS['tem_backend'] must be 'ctf_proxy' (or legacy alias "
            "'weak_phase_ctf'), 'multislice_lite', or 'syniscopy_multislice'; "
            f"got {tem_backend!r}."
        )
    if tem_backend_explicit and not tem_model_explicit:
        if tem_backend == "ctf_proxy":
            tem_model = "weak_phase_ctf"
        elif tem_backend in {"multislice_lite", "syniscopy_multislice"}:
            tem_model = tem_backend
    elif tem_model_explicit and not tem_backend_explicit:
        if tem_model == "weak_phase_ctf":
            tem_backend = "ctf_proxy"
        else:
            tem_backend = tem_model
    if tem_backend == "multislice_lite" and tem_model != "multislice_lite":
        raise ValueError("PARAMS['tem_backend']='multislice_lite' requires PARAMS['tem_model']='multislice_lite'.")
    if tem_backend == "syniscopy_multislice" and params.get("tem_slice_thickness_nm", PARAMS["tem_slice_thickness_nm"]) is None:
        params["tem_slice_thickness_nm"] = PARAMS["tem_slice_thickness_nm"]
    if tem_model == "syniscopy_multislice" and tem_backend != "syniscopy_multislice":
        raise ValueError("PARAMS['tem_model']='syniscopy_multislice' requires PARAMS['tem_backend']='syniscopy_multislice'.")
    if tem_model == "weak_phase_ctf" and tem_backend != "ctf_proxy":
        raise ValueError("PARAMS['tem_model']='weak_phase_ctf' requires PARAMS['tem_backend']='ctf_proxy'.")
    tem_potential_source = str(
        params.get("tem_potential_source", PARAMS.get("tem_potential_source", "material_projected_inner_potential"))
    ).strip().lower()
    if tem_potential_source not in {
        "material_projected_inner_potential",
        "sample_environment_projected_potential",
        "material_plus_sample_environment",
    }:
        raise ValueError(
            "PARAMS['tem_potential_source'] must be one of "
            "'material_projected_inner_potential', "
            "'sample_environment_projected_potential', "
            "or 'material_plus_sample_environment'."
        )
    tem_reference_status = str(params.get("tem_reference_status", PARAMS.get("tem_reference_status", "physics_based_unvalidated"))).strip().lower()
    if tem_reference_status not in {"physics_based_unvalidated", "reference_validated"}:
        raise ValueError("PARAMS['tem_reference_status'] must be 'physics_based_unvalidated' or 'reference_validated'.")
    if tem_reference_status == "reference_validated" and not params.get("tem_reference_validation_hash", PARAMS.get("tem_reference_validation_hash")):
        raise ValueError("reference_validated TEM requires PARAMS['tem_reference_validation_hash'].")
    _positive_int("tem_multislice_slices")
    if params.get("tem_slice_thickness_nm", None) is not None:
        _finite_float("tem_slice_thickness_nm", positive=True)
    _finite_float("tem_dose_per_pixel", nonnegative=True)
    _finite_float("tem_projected_potential_scale", nonnegative=True)
    if params.get("tem_phase_shift_per_volt_nm", None) is not None:
        _finite_float("tem_phase_shift_per_volt_nm", nonnegative=True)
    if params.get("tem_filter_guard_pixels", None) is not None:
        _finite_float("tem_filter_guard_pixels", nonnegative=True)
    if params.get("tem_objective_aperture_mrad", None) is not None:
        _finite_float("tem_objective_aperture_mrad", positive=True)
    if params.get("sem_filter_guard_pixels", None) is not None:
        _finite_float("sem_filter_guard_pixels", nonnegative=True)
    if params.get("sem_probe_sigma_nm", None) is not None:
        _finite_float("sem_probe_sigma_nm", nonnegative=True)
    sem_model_explicit = "sem_model" in params
    sem_backend_explicit = "sem_backend" in params
    sem_model = str(params.get("sem_model", PARAMS.get("sem_model", "gaussian_probe_proxy"))).strip().lower()
    if sem_model == "gaussian_probe_proxy":
        sem_model = "gaussian_probe_secondary_yield"
    if sem_model not in {"gaussian_probe_secondary_yield", "interaction_volume_proxy"}:
        raise ValueError(
            "PARAMS['sem_model'] must be 'gaussian_probe_secondary_yield' "
            "(alias 'gaussian_probe_proxy') or "
            f"'interaction_volume_proxy'; got {sem_model!r}."
        )
    sem_backend = str(
        params.get("sem_backend", PARAMS.get("sem_backend", "monte_carlo_transport"))
    ).strip().lower()
    if sem_backend in {"syniscopy_monte_carlo", "monte_carlo_sem"}:
        sem_backend = "monte_carlo_transport"
    if sem_backend not in {
        "gaussian_probe_proxy",
        "interaction_volume_proxy",
        "monte_carlo_transport",
        "syniscopy_transport_lite",
        "reference_kernel_table",
    }:
        raise ValueError(
            "PARAMS['sem_backend'] must be 'gaussian_probe_proxy', "
            f"'interaction_volume_proxy', 'monte_carlo_transport', "
            "'syniscopy_transport_lite', or 'reference_kernel_table'; got "
            f"{sem_backend!r}."
        )
    if sem_backend_explicit and not sem_model_explicit and sem_backend == "interaction_volume_proxy":
        sem_model = "interaction_volume_proxy"
    elif sem_model_explicit and not sem_backend_explicit and sem_model == "interaction_volume_proxy":
        sem_backend = "interaction_volume_proxy"
    if sem_backend == "interaction_volume_proxy" and sem_model != "interaction_volume_proxy":
        raise ValueError("PARAMS['sem_backend']='interaction_volume_proxy' requires PARAMS['sem_model']='interaction_volume_proxy'.")
    if sem_backend == "reference_kernel_table" and not params.get("sem_reference_kernel_path", PARAMS.get("sem_reference_kernel_path")):
        raise ValueError("PARAMS['sem_backend']='reference_kernel_table' requires PARAMS['sem_reference_kernel_path'].")
    sem_source_representation = str(
        params.get("sem_source_representation", PARAMS.get("sem_source_representation", "volume"))
    ).strip().lower()
    if sem_source_representation in {"projected_2d", "projected-source", "projected_source"}:
        sem_source_representation = "projected"
    if sem_source_representation in {"sliced_volume", "voxel_volume", "volume_zyx"}:
        sem_source_representation = "volume"
    if sem_source_representation not in {"projected", "volume"}:
        raise ValueError("PARAMS['sem_source_representation'] must be 'projected' or 'volume'.")
    params["sem_source_representation"] = sem_source_representation
    _positive_int("sem_volume_slices")
    if params.get("sem_volume_slice_thickness_nm", None) is not None:
        _finite_float("sem_volume_slice_thickness_nm", positive=True)
    sem_source_z_origin = str(
        params.get("sem_source_z_origin", PARAMS.get("sem_source_z_origin", "entry_surface_depth"))
    ).strip().lower()
    if sem_source_z_origin not in {"entry_surface_depth", "focus_plane_relative"}:
        raise ValueError(
            "PARAMS['sem_source_z_origin'] must be 'entry_surface_depth' or "
            "'focus_plane_relative'."
        )
    params["sem_source_z_origin"] = sem_source_z_origin
    _finite_float("sem_source_z_offset_nm")
    _finite_float("sem_beam_current_nA", nonnegative=True)
    _finite_float("sem_dwell_time_us", nonnegative=True)
    _finite_float("sem_detector_takeoff_angle_deg", nonnegative=True)
    _finite_float("sem_detector_acceptance", nonnegative=True)
    _finite_float("sem_escape_depth_nm", nonnegative=True)
    _finite_float("sem_backscatter_fraction", nonnegative=True)
    if float(params.get("sem_detector_acceptance", PARAMS.get("sem_detector_acceptance"))) > 1.0:
        raise ValueError("PARAMS['sem_detector_acceptance'] must be <= 1.0.")
    if float(params.get("sem_backscatter_fraction", PARAMS.get("sem_backscatter_fraction"))) > 1.0:
        raise ValueError("PARAMS['sem_backscatter_fraction'] must be <= 1.0.")
    _finite_float("sem_transport_material_scale", nonnegative=True)
    _finite_float("sem_transport_source_exponent", positive=True)
    _finite_float("sem_transport_topography_exponent", positive=True)
    _positive_int("sem_monte_carlo_trajectories")
    _positive_int("sem_monte_carlo_steps")
    if params.get("sem_monte_carlo_step_nm", None) is not None:
        _finite_float("sem_monte_carlo_step_nm", positive=True)
    if params.get("sem_monte_carlo_range_nm", None) is not None:
        _finite_float("sem_monte_carlo_range_nm", positive=True)
    _finite_float("sem_monte_carlo_scatter_std_deg", nonnegative=True)
    if params.get("sem_monte_carlo_kernel_size_px", None) is not None:
        _positive_int("sem_monte_carlo_kernel_size_px")
    _finite_float("sem_reference_source_depth_nm", nonnegative=True)
    _finite_float("sem_reference_incident_angle_deg", nonnegative=True)
    if float(params.get("sem_reference_incident_angle_deg", PARAMS.get("sem_reference_incident_angle_deg"))) > 180.0:
        raise ValueError("PARAMS['sem_reference_incident_angle_deg'] must be <= 180.0.")
    reference_material = str(params.get("sem_reference_material", PARAMS.get("sem_reference_material", ""))).strip().lower()
    if reference_material == "":
        raise ValueError("PARAMS['sem_reference_material'] must be a non-empty string.")
    reference_geometry = str(params.get("sem_reference_geometry", PARAMS.get("sem_reference_geometry", ""))).strip().lower()
    if reference_geometry == "":
        raise ValueError("PARAMS['sem_reference_geometry'] must be a non-empty string.")
    _finite_float("sem_interaction_volume_nm", nonnegative=True)
    _finite_float("sem_topography_contrast_gain", nonnegative=True)
    detector_direction = params.get("sem_detector_direction_xy", PARAMS.get("sem_detector_direction_xy"))
    direction_arr = np.asarray(detector_direction, dtype=float)
    if direction_arr.shape != (2,) or not np.all(np.isfinite(direction_arr)):
        raise ValueError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
    if float(np.linalg.norm(direction_arr)) <= 0.0:
        raise ValueError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
    _finite_float("sem_probe_sigma_pixels", nonnegative=True)
    _finite_float("supervision_prior_log_odds")
    _finite_float("supervision_log_odds_threshold")
    log_odds_clip_epsilon = _finite_float("supervision_log_odds_clip_epsilon")
    if not (0.0 < log_odds_clip_epsilon < 0.5):
        raise ValueError(
            "PARAMS['supervision_log_odds_clip_epsilon'] must lie in (0, 0.5); "
            f"got {log_odds_clip_epsilon}."
        )
    _finite_float("fisher_lateral_step_nm", positive=True)
    fisher_likelihood_model = str(
        params.get(
            "fisher_likelihood_model",
            PARAMS.get("fisher_likelihood_model", "mean_fisher_diagnostic"),
        )
    ).strip().lower()
    if fisher_likelihood_model not in {
        "gaussian_fixed_variance",
        "poisson_exact",
        "gaussian_parameter_dependent_variance",
        "poisson_gaussian_approx",
        "poisson_gaussian_plugin",
        "mean_fisher_diagnostic",
    }:
        raise ValueError(
            "PARAMS['fisher_likelihood_model'] must be one of "
            "'gaussian_fixed_variance', 'poisson_exact', "
            "'gaussian_parameter_dependent_variance', 'poisson_gaussian_approx', "
            "or 'mean_fisher_diagnostic'; "
            f"got {fisher_likelihood_model!r}."
        )
    detected_quanta_derivative_target = str(
        params.get(
            "detected_quanta_derivative_target",
            PARAMS.get("detected_quanta_derivative_target", "signed_contrast_scaled"),
        )
    ).strip().lower()
    if detected_quanta_derivative_target not in {"signed_contrast_scaled", "count_mean_derivative"}:
        raise ValueError(
            "PARAMS['detected_quanta_derivative_target'] must be "
            "'signed_contrast_scaled' or 'count_mean_derivative'; "
            f"got {detected_quanta_derivative_target!r}."
        )
    fisher_lateral_derivative_mode = str(
        params.get(
            "fisher_lateral_derivative_mode",
            PARAMS.get("fisher_lateral_derivative_mode", "stationary_shift"),
        )
    ).strip().lower()
    if fisher_lateral_derivative_mode not in {"stationary_shift", "rerendered_xy"}:
        raise ValueError(
            "PARAMS['fisher_lateral_derivative_mode'] must be 'stationary_shift' "
            f"or 'rerendered_xy'; got {fisher_lateral_derivative_mode!r}."
        )
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
    supervision_calibration_mode = str(
        params.get(
            "supervision_score_calibration_mode",
            PARAMS.get("supervision_score_calibration_mode", "uncalibrated_support"),
        )
    ).strip().lower()
    if supervision_calibration_mode not in {"uncalibrated_support", "platt_logistic", "isotonic"}:
        raise ValueError(
            "PARAMS['supervision_score_calibration_mode'] must be "
            "'uncalibrated_support', 'platt_logistic', or 'isotonic'; "
            f"got {supervision_calibration_mode!r}."
        )
    calibration_parameters = params.get(
        "supervision_score_calibration_parameters",
        PARAMS.get("supervision_score_calibration_parameters"),
    )
    if calibration_parameters is not None and not isinstance(calibration_parameters, dict):
        raise ValueError("PARAMS['supervision_score_calibration_parameters'] must be None or a dict.")
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
    source_only_models = {
        "tem_phase_contrast",
        "sem_secondary_electron",
        "fluorescence_widefield",
        "tirf_fluorescence",
    }
    imaging_model = str(params.get("imaging_model", PARAMS.get("imaging_model", ""))).strip().lower()
    if imaging_model not in source_only_models and numerical_aperture > refractive_index_medium:
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
    _finite_float("dynamic_process_noise_scale", nonnegative=True)
    _finite_float("dynamic_initial_variance_nm2", positive=True)
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
    supported_patterns = {
        "none",
        "gold_holes",
        "nanopillars",
        "fiducial_dots",
        "grid_bars",
        "holey_carbon",
        "microfluidic_walls",
        "patterned_coverslip",
    }
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
        if pattern_model in PATTERN_DEFAULT_PRESETS and preset in {
            "",
            "default",
            pattern_model,
            "default_gold_holes",
            "default_nanopillars",
        }:
            preset = PATTERN_DEFAULT_PRESETS[pattern_model]
            params["sample_environment_pattern_preset"] = preset
        allowed_presets = {"empty_background"}
        if pattern_model in PATTERN_DEFAULT_PRESETS:
            allowed_presets.add(PATTERN_DEFAULT_PRESETS[pattern_model])
        if preset not in allowed_presets:
            raise ValueError(
                "PARAMS['sample_environment_pattern_preset'] is not valid for "
                f"{pattern_model!r}. Allowed values are {sorted(allowed_presets)}; "
                f"got {params.get('sample_environment_pattern_preset')!r}."
            )

    roughness_model_raw = params.get("sample_environment_pattern_roughness_model", "none")
    roughness_model = str(roughness_model_raw).strip().lower()
    if roughness_model not in ("none", "static", "flicker", "source_matched"):
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_model'] must be one of "
            "'none', 'static', 'flicker', or 'source_matched'; got "
            f"{roughness_model_raw!r}."
        )

    roughness_source = params.get("sample_environment_pattern_roughness_source", None)
    if roughness_model == "source_matched" and roughness_source is None:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_source'] must be provided "
            "when sample_environment_pattern_roughness_model='source_matched'."
        )

    if roughness_source is not None and not isinstance(
        roughness_source,
        (str, bytes, os.PathLike, list, tuple, np.ndarray),
    ):
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_source'] must be a file path "
            "or an array-like roughness field when set."
        )

    roughness_amplitude = _finite_float(
        "sample_environment_pattern_roughness_amplitude",
        nonnegative=True,
    )
    if roughness_model == "none" and roughness_amplitude > 0.0:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_amplitude'] can only be "
            "non-zero when sample_environment_pattern_roughness_model is "
            "'static', 'flicker', or 'source_matched'."
        )

    roughness_correlation_pixels = _finite_float(
        "sample_environment_pattern_roughness_correlation_pixels",
        nonnegative=True,
    )
    if 0.0 < roughness_correlation_pixels < 1.0:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_correlation_pixels'] "
            "must be at least 1.0 when positive."
        )

    roughness_phase_std = _finite_float(
        "sample_environment_pattern_roughness_phase_std",
        nonnegative=True,
    )
    if roughness_phase_std > np.pi:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_phase_std'] must be <= pi."
        )

    roughness_source_coupling = str(
        params.get(
            "sample_environment_pattern_roughness_source_coupling",
            PARAMS.get("sample_environment_pattern_roughness_source_coupling", "channel_weighted"),
        )
    ).strip().lower()
    if roughness_source_coupling not in {
        "independent",
        "coherent_amplitude",
        "field_weighted",
        "scene_weighted",
        "channel_weighted",
    }:
        raise ValueError(
            "PARAMS['sample_environment_pattern_roughness_source_coupling'] must be "
            "'independent', 'coherent_amplitude', 'field_weighted', "
            "'scene_weighted', or 'channel_weighted'; got "
            f"{roughness_source_coupling!r}."
        )

    detector_qe_public = _finite_float("detector_qe", nonnegative=True)
    if detector_qe_public > 1.0:
        raise ValueError("PARAMS['detector_qe'] must be <= 1.0.")
    _finite_float("emccd_excess_noise_factor", positive=True)
    if float(params.get("emccd_excess_noise_factor", PARAMS.get("emccd_excess_noise_factor"))) < 1.0:
        raise ValueError("PARAMS['emccd_excess_noise_factor'] must be >= 1.0.")
    if params.get("read_noise_e", PARAMS.get("read_noise_e")) is not None:
        _finite_float("read_noise_e", nonnegative=True)
    _finite_float("dark_current_e_per_pixel_per_s", nonnegative=True)
    _finite_float("exposure_time_s", positive=True)
    if params.get("saturation_e", PARAMS.get("saturation_e")) is not None:
        _finite_float("saturation_e", positive=True)

    for key in (
        "fixed_pattern_gain_map",
        "fixed_pattern_offset_map",
        "scmos_gain_map",
        "hot_pixel_mask",
        "scmos_variance_map",
        "scmos_read_noise_map",
        "flat_field_map",
        "dark_frame_map",
        "prnu_map",
        "dsnu_map",
    ):
        _validate_noise_map_value(f"PARAMS[{key!r}]", params.get(key, PARAMS.get(key)))

    noise_keys = {
        "shot_noise_enabled",
        "gaussian_noise_enabled",
        "camera_gain_e_per_count",
        "detector_qe",
        "detector_input_is_incident_quanta",
        "emccd_enabled",
        "emccd_gain",
        "emccd_excess_noise_factor",
        "read_noise_e",
        "read_noise_counts",
        "dark_current_e_per_pixel_per_s",
        "exposure_time_s",
        "saturation_level",
        "saturation_e",
        "adc_quantization",
        "adc_quantization_counts",
        "background_offset_counts",
        "fixed_pattern_gain_map",
        "fixed_pattern_offset_map",
        "scmos_gain_map",
        "scmos_read_noise_map",
        "hot_pixel_mask",
        "scmos_variance_map",
        "nonlinearity_calibration",
        "flat_field_map",
        "dark_frame_map",
        "prnu_map",
        "dsnu_map",
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
    for key in ("shot_noise_enabled", "gaussian_noise_enabled", "clip_output_to_nonnegative", "detector_input_is_incident_quanta", "emccd_enabled", "adc_quantization"):
        if key in noise_model:
            _bool_value(f"PARAMS['noise_model']['{key}']", noise_model[key])
    for key in ("fixed_pattern_gain_map", "fixed_pattern_offset_map", "scmos_gain_map", "hot_pixel_mask", "scmos_variance_map", "scmos_read_noise_map", "flat_field_map", "dark_frame_map", "prnu_map", "dsnu_map"):
        if key in noise_model:
            _validate_noise_map_value(f"PARAMS['noise_model']['{key}']", noise_model[key])
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
        for key in ("shot_noise_enabled", "gaussian_noise_enabled", "clip_output_to_nonnegative", "detector_input_is_incident_quanta", "emccd_enabled", "adc_quantization"):
            if key in overrides:
                _bool_value(
                    f"PARAMS['modality_noise'][{modality_name!r}]['{key}']",
                    overrides[key],
                )
        for key in ("fixed_pattern_gain_map", "fixed_pattern_offset_map", "scmos_gain_map", "hot_pixel_mask", "scmos_variance_map", "scmos_read_noise_map", "flat_field_map", "dark_frame_map", "prnu_map", "dsnu_map"):
            if key in overrides:
                _validate_noise_map_value(
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


def normalize_params(
    params: dict,
    *,
    allowed_extra_keys: set[str] | None = None,
    allowed_internal_keys: set[str] | None = None,
) -> dict:
    """
    Return a validated copy of ``params`` with canonical dependent values set.

    This is the entry point for runtime paths that need validation plus
    normalization without mutating the caller's dictionary.
    """
    normalized = deepcopy(params)
    _normalize_params_in_place(
        normalized,
        allowed_extra_keys=allowed_extra_keys,
        allowed_internal_keys=allowed_internal_keys,
    )
    return normalized


def validate_params(
    params: dict,
    *,
    allowed_extra_keys: set[str] | None = None,
    allowed_internal_keys: set[str] | None = None,
) -> None:
    """
    Validate the public PARAMS surface without mutating ``params``.

    Syniscopy v1 has one canonical public key per concept. Unknown keys raise
    immediately so aliases and typo-driven configuration drift do not enter
    generated datasets or manuscript artifacts.
    """
    _normalize_params_in_place(
        dict(params),
        allowed_extra_keys=allowed_extra_keys,
        allowed_internal_keys=allowed_internal_keys,
    )


__all__ = [
    "_normalize_params_in_place",
    "normalize_params",
    "validate_params",
]
