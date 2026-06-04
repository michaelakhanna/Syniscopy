from __future__ import annotations

from config import param_value
from config.runtime import (
    resolved_image_size_pixels,
    resolved_model_canvas_shape,
    resolved_pixel_size_nm,
    resolved_psf_oversampling_factor,
)
import json
from pathlib import Path
from typing import Any

from typing import Mapping

from backend_fidelity import extract_backend_fidelity_metadata
from experiment_contracts import (
    backend_contract_for_modality,
    detector_model_from_params,
    model_card_from_profile_card,
)
from imaging_models import get_imaging_model
from json_utils import json_safe_with_nonfinite_tags
from measurement_units import canonical_measurement_domain_and_signal_units_for_output
from modality_registry import (
    canonical_modality_name,
    is_electron_modality,
    modality_display_name,
)


PROFILE_SCHEMA_VERSION = "syniscopy-modality-profile-v1"


_PAPER_USE_CATEGORY_BY_DOMAIN = {
    "phase": "phase_domain_profile",
    "fringe_count": "scalar_proxy_profile",
    "electron_count": "simplified_electron_proxy",
}

_ELECTRON_PAPER_USE_CATEGORY_BY_FIDELITY = {
    "unknown": "unknown_backend_fidelity",
    "proxy": "simplified_electron_proxy",
    "simplified_proxy": "simplified_electron_proxy",
    "simplified_electron_proxy": "simplified_electron_proxy",
    "physics_based_unvalidated": "high_fidelity_electron_profile",
    "physics_based": "high_fidelity_electron_profile",
    "high_fidelity": "high_fidelity_electron_profile",
    "reference_validated": "reference_validated_electron_profile",
}


def _backend_fidelity_metadata_for_card(
    response: dict[str, Any], backend: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return extract_backend_fidelity_metadata(response, backend_contract=dict(backend or {}))


def _measurement_domain_and_units(
    modality: str,
    output_type: str,
    response: dict[str, Any],
) -> tuple[str, str]:
    return canonical_measurement_domain_and_signal_units_for_output(
        modality,
        output_type,
        response_function=response,
    )


def _forward_observable(modality: str, response: dict[str, Any]) -> str:
    kind = str(response.get("kind", "") or "").strip()
    if kind:
        return kind
    if modality.startswith("tem"):
        return "configured_tem_electron_contrast"
    if modality.startswith("sem"):
        return "configured_sem_secondary_electron_transport"
    if "fluorescence" in modality or "tirf" in modality:
        return "fluorophore_emission_counts"
    return "scalar_optical_contrast"


def _noise_model(params: dict, measurement_domain: str) -> str:
    if measurement_domain == "phase":
        return "phase_variance_readout_and_detected_quanta"
    try:
        from camera_noise import resolve_camera_noise_config

        cfg = resolve_camera_noise_config(params)
        shot_enabled = bool(cfg.shot_noise_enabled)
        gaussian_enabled = bool(cfg.gaussian_noise_enabled)
    except Exception:
        shot_enabled = bool(param_value(params, 'shot_noise_enabled'))
        gaussian_enabled = bool(param_value(params, 'gaussian_noise_enabled'))
    if shot_enabled and gaussian_enabled:
        return "poisson_shot_plus_gaussian_read_noise"
    if shot_enabled:
        return "poisson_shot_noise"
    if gaussian_enabled:
        return "gaussian_read_noise"
    return "deterministic_no_noise"


def _count_scaling_mode(params: dict, response: dict[str, Any], measurement_domain: str) -> str:
    if measurement_domain == "phase":
        return "phase_native_display_count_scaling_only"
    if response.get("count_scaling_mode"):
        return str(response.get("count_scaling_mode"))
    if measurement_domain == "electron_count":
        return "electron_dose_or_electrons_per_pixel"
    return str(response.get("count_scaling_mode", "detector_count_scaling"))


def _paper_use_category(modality: str, measurement_domain: str, response: dict[str, Any]) -> str:
    modality = canonical_modality_name(modality)
    if is_electron_modality(modality):
        level = str(response.get("backend_fidelity_level", "proxy")).strip().lower().replace(" ", "_")
        return _ELECTRON_PAPER_USE_CATEGORY_BY_FIDELITY.get(level, "simplified_electron_proxy")
    vectorial_markers = " ".join(
        str(response.get(key, ""))
        for key in (
            "fidelity_label",
            "backend_name",
            "backend_mode",
            "fluorescence_backend",
            "optical_field_backend",
            "equations_or_model_family",
            "implemented_approximation_level",
            "vectorial_detection_mode",
            "dpc_channel_model",
        )
    ).strip().lower()
    if "vectorial" in vectorial_markers or bool(response.get("uses_vectorial_field", False)):
        return "vectorial_optical_profile"
    if modality == "quantitative_phase":
        return "phase_domain_profile"
    if modality == "differential_phase_contrast":
        channel_model = str(response.get("dpc_channel_model", "")).strip().lower()
        if "vectorial" in channel_model or bool(response.get("dpc_vectorial_backend_enabled", False)):
            return "vectorial_optical_profile"
        return "scalar_proxy_profile"
    if modality in {"ricm", "zernike_phase_contrast"}:
        return "scalar_proxy_profile"
    if "diagnostic" in str(response.get("fidelity_label", "")).lower():
        return "implementation_sanity_check"
    return _PAPER_USE_CATEGORY_BY_DOMAIN.get(measurement_domain, "core_scalar_optical_profile")


def _active_parameters(params: dict, modality: str) -> dict[str, Any]:
    prefixes = (
        "imaging_model",
        "pixel_size_nm",
        "image_size_pixels",
        "wavelength_nm",
        "probe_wavelength_nm",
        "numerical_aperture",
        "refractive_index_medium",
        "background_intensity",
        "shot_noise_enabled",
        "gaussian_noise_enabled",
        "detector_",
        "emccd_",
        "read_noise_",
        "dark_",
        "fixed_pattern_",
        "hot_pixel_",
        "saturation_",
        "background_offset_counts",
        "flat_field_map",
        "dark_frame_map",
        "scan_line_",
        "noise_parameterization",
        "clip_output_to_nonnegative",
        "read_noise_counts",
        "camera_gain_e_per_count",
        "fisher_lateral_",
        "fisher_likelihood_model",
        "detected_quanta_derivative_target",
        "optical_field_backend",
        "spectral_integration_model",
        "illumination_spectrum_",
        "broadband_",
        "detector_spectral_response_model",
        "coverslip_",
        "scene_dimensionality",
        "volumetric_",
        "volume_",
        "confocal_",
        "light_sheet_",
        "holotomography_",
        "polarization_model",
        "vectorial_",
    )
    modality_prefixes_by_modality = {
        "tem_phase_contrast": ("tem_",),
        "sem_secondary_electron": ("sem_",),
        "quantitative_phase": ("qpi_",),
        "qpi": ("qpi_",),
        "ricm": ("ricm_",),
        "dpc": ("dpc_",),
        "differential_phase_contrast": ("dpc_",),
        "zernike_phase_contrast": ("zernike_",),
        "fluorescence_widefield": ("fluorescence_",),
        "tirf_fluorescence": ("fluorescence_", "tirf_"),
        "bright_field": ("kohler_",),
        "partially_coherent_bright_field": ("kohler_",),
        "dark_field": ("dark_field_",),
        "coherent_dark_field": ("dark_field_",),
        "off_axis_holography": ("off_axis_",),
    }
    modality_prefixes = modality_prefixes_by_modality.get(modality, ())
    selected = {
        key: value
        for key, value in params.items()
        if any(str(key).startswith(prefix) for prefix in prefixes)
        or any(str(key).startswith(prefix) for prefix in modality_prefixes)
    }
    selected["canonical_modality_name"] = modality
    return json_safe_with_nonfinite_tags(selected)


def _augment_profile_card(card: dict[str, Any], params: dict, modality: str) -> dict[str, Any]:
    backend = backend_contract_for_modality(modality, card.get("response_function", {})).to_dict()
    detector = detector_model_from_params(params).to_dict()
    card["profile_id"] = card.get("profile_id") or f"profile:{modality}:{card.get('fidelity_label', 'model_conditional')}"
    card["backend_contract"] = backend
    backend_fidelity_metadata = _backend_fidelity_metadata_for_card(
        card.get("response_function", {}), backend=backend
    )
    card["backend_fidelity_metadata"] = backend_fidelity_metadata
    if backend.get("backend_fidelity_level"):
        card["backend_fidelity_level"] = backend.get("backend_fidelity_level")
    if backend.get("reference_backend_metadata") is not None:
        card["reference_backend_metadata"] = backend.get("reference_backend_metadata")
    for key in (
        "backend_name",
        "equations_or_model_family",
        "implemented_approximation_level",
        "native_operating_assumptions",
        "validation_status",
        "convergence_status",
        "comparison_contract_id",
        "artifact_provenance_id",
        "known_omissions",
    ):
        if backend_fidelity_metadata.get(key) is not None:
            card[key] = json_safe_with_nonfinite_tags(backend_fidelity_metadata[key])
    for key in (
        "backend_id",
        "backend_family",
        "uses_scalar_scattered_field",
        "uses_vectorial_field",
        "uses_incoherent_source_map",
        "uses_electron_projected_potential",
        "uses_probe_scan",
        "uses_reference_interference",
        "uses_emission_psf",
        "axial_sensitivity_mechanism",
        "axial_sensitive",
        "source_input_kind",
        "source_map_ndim",
        "source_axis_order",
        "source_projection_policy",
        "backend_consumes_volume_source",
        "volume_transport_model",
        "known_omissions",
    ):
        if key in backend:
            card[key] = backend[key]
    if not card.get("contrast_frame_units") and backend.get("contrast_frame_units"):
        card["contrast_frame_units"] = str(backend["contrast_frame_units"])
    card["detector_model"] = detector
    nonlinear_detector = bool(detector["nonlinear_detector_effects_active"])
    deterministic_transfer = bool(detector["deterministic_detector_transfer_active"])
    linear_fisher_safe = bool(detector["safe_for_linear_fisher_variance"])
    card["detector_noise_input_domain"] = str(detector["detector_noise_input_domain"])
    card["nonlinear_detector_effects_active"] = nonlinear_detector
    card["deterministic_detector_transfer_active"] = deterministic_transfer
    card["safe_for_linear_fisher_variance"] = linear_fisher_safe
    card["fisher_variance_model_scope"] = (
        "linear_poisson_gaussian_only"
        if linear_fisher_safe
        else "diagnostic_only_linearized_detector_variance"
    )
    card["detector_likelihood_status"] = (
        "linear_poisson_gaussian_compatible"
        if linear_fisher_safe
        else "nonlinear_or_static_transfer_not_in_linear_fisher"
    )
    card["model_card"] = model_card_from_profile_card(card)
    return card


def profile_card_for_model(
    params: dict,
    imaging_model,
    modality_name: str | None = None,
    *,
    response_function: Mapping[str, Any] | None = None,
    model_canvas_shape: tuple[int, int] | None = None,
) -> dict:
    raw_modality = modality_name if modality_name is not None else param_value(params, 'imaging_model')
    modality = canonical_modality_name(str(raw_modality))
    if model_canvas_shape is not None:
        shape = (int(model_canvas_shape[0]), int(model_canvas_shape[1]))
    else:
        shape = resolved_model_canvas_shape(params)
    response = (
        dict(response_function)
        if response_function is not None
        else imaging_model.compute_response_function(shape, params)
    )
    output_type = str(getattr(imaging_model, "output_type", response.get("output_type", "intensity")))
    measurement_domain, signal_units = _measurement_domain_and_units(modality, output_type, response)
    detector_pixel_size_nm = float(resolved_pixel_size_nm(params))
    oversampling = float(resolved_psf_oversampling_factor(params))
    fidelity_label = str(response.get("fidelity_label", param_value(params, "profile_fidelity_label")))
    card = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "canonical_modality_name": modality,
        "display_name": modality_display_name(modality),
        "model_class": imaging_model.__class__.__name__,
        "forward_observable": _forward_observable(modality, response),
        "output_type": output_type,
        "measurement_domain": measurement_domain,
        "signal_units": signal_units,
        "noise_model": _noise_model(params, measurement_domain),
        "count_scaling_mode": _count_scaling_mode(params, response, measurement_domain),
        "derivative_validity_scope": (
            "stationary-shift derivatives require translationally invariant scenes; "
            "structured environments require rerendered_xy metadata"
        ),
        "active_parameters": _active_parameters(params, modality),
        "detector_parameters": {
            "detector_pixel_size_nm": detector_pixel_size_nm,
            "model_canvas_pixel_size_nm": detector_pixel_size_nm / oversampling,
            "psf_oversampling_factor": oversampling,
            "image_size_pixels": int(resolved_image_size_pixels(params)),
            "bit_depth": int(param_value(params, 'bit_depth')),
        },
        "sample_environment_usage": {
            "sample_environment_enabled": bool(param_value(params, 'sample_environment_enabled')),
            "sample_environment_pattern_enabled": bool(
                param_value(params, 'sample_environment_pattern_enabled')
            ),
            "uses_sample_environment_pattern": bool(
                getattr(imaging_model, "uses_sample_environment_pattern", False)
            ),
        },
        "uses_particle_material_sources": bool(getattr(imaging_model, "uses_particle_material_sources", False)),
        "requires_optical_scattered_field": bool(getattr(imaging_model, "requires_optical_scattered_field", True)),
        "requires_pre_crop_filtering": bool(getattr(imaging_model, "requires_pre_crop_optical_filtering", False)),
        "supports_spectral_channels": bool(getattr(imaging_model, "supports_spectral_channels", False)),
        "fidelity_label": fidelity_label,
        "backend_fidelity_level": str(response.get("backend_fidelity_level", "proxy")),
        "reference_backend_metadata": json_safe_with_nonfinite_tags(response.get("reference_backend_metadata")),
        "validity_scope": str(
            response.get(
                "validity_scope",
                "model-conditioned shared-scene diagnostic under the listed profile parameters",
            )
        ),
        "paper_use_category": _paper_use_category(modality, measurement_domain, response),
        "response_function": json_safe_with_nonfinite_tags(response),
    }
    return _augment_profile_card(card, params, modality)


def canonical_profile_card(params: dict, modality_name: str | None = None) -> dict:
    profile_params = dict(params)
    if modality_name is not None:
        profile_params["imaging_model"] = modality_name
    return profile_card_for_model(
        profile_params,
        get_imaging_model(profile_params),
        modality_name=modality_name,
    )


def write_profile_cards(modality_params_list, output_path) -> list[dict]:
    cards = [
        canonical_profile_card(params, param_value(params, "imaging_model"))
        for params in modality_params_list
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe_with_nonfinite_tags(cards), indent=2, sort_keys=True) + "\n")
    return cards
