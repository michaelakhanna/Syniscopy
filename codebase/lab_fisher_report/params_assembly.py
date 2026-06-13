"""Parameter assembly for lab Fisher reports."""

from __future__ import annotations
from configured_parameters import configured_assign

import argparse
from copy import deepcopy
from typing import Any, Mapping

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from config import (
    AcquisitionProfile,
    BackgroundSubtractionSettings,
    MicroscopeRuntimeSettings,
    ModalitySettings,
    MotionDynamicsSettings,
    default_params,
)
from modality_registry import (
    LAB_DEFAULT_MODALITIES,
    LAB_OPTICAL_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name,
    modality_comparison_identity,
)
from particle_specs import mutable_particle_scene_from_params
from particle_specs import normalize_particle_specs
from postprocessing import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)
from presets import apply_instrument_preset
from param_schema import PUBLIC_PARAM_KEYS

from .cli import TEMPLATE_OVERRIDES, _load_json
from .report_contracts import (
    REPORT_SHARED_RUN_PARAM_KEYS,
    assert_report_configured_profile_defaults,
    assert_report_configured_profile_particle_defaults,
)
from .scene_view import resolve_report_scene_state

__all__ = [
    "_apply_cli_overrides",
    "_make_microscope_base_and_shared_params",
    "_make_params",
    "_resolve_modalities",
]


def _resolve_modalities(spec: str) -> list[str]:
    text = str(spec).strip().lower()
    if text in {"lab-default", "default", "optical"}:
        modalities = list(LAB_OPTICAL_MODALITIES if text == "optical" else LAB_DEFAULT_MODALITIES)
    elif text == "all":
        modalities = list(SUPPORTED_MODALITIES)
    else:
        modalities = [
            canonical_modality_name(part.strip())
            for part in text.split(",")
            if part.strip()
        ]
    supported = set(SUPPORTED_MODALITIES)
    unsupported = sorted(set(modalities) - supported)
    if unsupported:
        raise ValueError(
            f"Unsupported modalities: {unsupported}. Supported: {sorted(supported)}"
        )
    # Public report comparisons are over physical modality profiles, not over
    # every accepted spelling. Exact-string de-duplication lets aliases such as
    # bright_field and partially_coherent_bright_field produce duplicate rows
    # with different user-facing ranks, so coalesce by registry-owned comparison
    # identity while preserving the first public spelling the user requested.
    unique: list[str] = []
    seen_comparison_identities: set[str] = set()
    for modality in modalities:
        identity = modality_comparison_identity(modality)
        if identity in seen_comparison_identities:
            continue
        seen_comparison_identities.add(identity)
        unique.append(modality)
    return unique




_DIRECT_CLI_OVERRIDE_KEYS = {
    "pixel_size_nm": "pixel_size_nm",
    "wavelength_nm": "wavelength_nm",
    "numerical_aperture": "na",
    "background_intensity": "background_counts",
    "read_noise_counts": "read_noise_counts",
    "camera_gain_e_per_count": "camera_gain_e_per_count",
    "image_size_pixels": "image_size_pixels",
    "pupil_samples": "pupil_samples",
}

def _make_template_base_params(*, instrument: str | None = None) -> dict[str, Any]:
    """Return the shared report defaults before run-level or microscope overlays."""

    params = default_params()
    params.update(deepcopy(TEMPLATE_OVERRIDES))
    assert_report_configured_profile_defaults(
        params,
        context="_make_template_base_params",
    )
    assert_report_configured_profile_particle_defaults(
        params,
        context="_make_template_base_params",
    )
    if instrument:
        params = apply_instrument_preset(params, instrument)
    return params


def _reject_unknown_public_keys(overlay: Mapping[str, Any], *, field_name: str) -> None:
    unknown = sorted(str(key) for key in overlay if str(key) not in PUBLIC_PARAM_KEYS)
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown parameter key(s) {unknown}. "
            "Use canonical public parameter-schema keys only."
        )


def _explicit_cli_override_keys(args: argparse.Namespace) -> set[str]:
    keys = {"random_seed"}
    for param_key, arg_name in _DIRECT_CLI_OVERRIDE_KEYS.items():
        if getattr(args, arg_name, None) is not None:
            keys.add(param_key)
    if (
        getattr(args, "diameter_nm", None) is not None
        or getattr(args, "material", None) is not None
        or getattr(args, "z_nm", None) is not None
    ):
        keys.add("particles")
        keys.add("initial_z_span_nm")
    return keys


def _make_microscope_base_and_shared_params(
    args: argparse.Namespace,
    *,
    microscope_shared_params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(instrument_base, shared_overlay)`` for explicit microscopes.

    Explicit microscope reports must not apply a microscope's named instrument
    preset after ``--params-json`` or CLI/shared-scene overrides.  The returned
    base is only parameters + template defaults + an optional run-level instrument;
    the sparse shared overlay contains public JSON/CLI/shared-scene keys that
    must be applied after each microscope's instrument preset and before that
    microscope's own params overlay.
    """

    instrument_base = _make_template_base_params(instrument=getattr(args, "instrument", None))
    assembled = deepcopy(instrument_base)
    shared_keys: set[str] = set()

    if getattr(args, "params_json", None):
        params_json = _load_json(args.params_json)
        _reject_unknown_public_keys(params_json, field_name="--params-json")
        assembled.update(deepcopy(params_json))
        shared_keys.update(str(key) for key in params_json)

    if microscope_shared_params:
        shared_mapping = dict(microscope_shared_params)
        _reject_unknown_public_keys(shared_mapping, field_name="microscope shared_params")
        assembled.update(deepcopy(shared_mapping))
        shared_keys.update(str(key) for key in shared_mapping)

    _apply_cli_overrides(assembled, args)
    # The resolved target particle and scene fingerprint are always shared, even
    # when they originated from the public template rather than params-json/CLI.
    # Otherwise explicit microscope candidates with different sampling/instrument
    # overlays would resample or recenter the scene independently and corrupt the
    # microscope_ranking.csv/report.md same-scene comparison contract.
    shared_keys.update({"particles", "initial_z_span_nm"})
    shared_keys.update(REPORT_SHARED_RUN_PARAM_KEYS)
    shared_keys.update(_explicit_cli_override_keys(args))

    shared_overlay = {
        key: deepcopy(assembled[key])
        for key in sorted(shared_keys)
        if key in assembled
    }
    return instrument_base, shared_overlay

def _apply_cli_overrides(params: dict[str, Any], args: argparse.Namespace) -> None:
    direct = {
        "pixel_size_nm": args.pixel_size_nm,
        "wavelength_nm": args.wavelength_nm,
        "numerical_aperture": args.na,
        "background_intensity": args.background_counts,
        "read_noise_counts": args.read_noise_counts,
        "camera_gain_e_per_count": args.camera_gain_e_per_count,
        "image_size_pixels": args.image_size_pixels,
        "pupil_samples": args.pupil_samples,
        "random_seed": args.seed,
    }
    for key, value in direct.items():
        if value is not None:
            configured_assign(params, key, value)

    num_frames = int(max(1, args.num_frames))
    fps = AcquisitionProfile.from_params(params).fps
    duration_seconds = AcquisitionProfile.duration_seconds_for_frame_count(fps, num_frames)

    configured_assign(params, 'return_ideal_float_frames', True)
    configured_assign(params, 'save_frame_sequence', False)
    configured_assign(params, 'save_raw_camera_video', False)
    configured_assign(params, 'save_raw_camera_frame_sequence', False)
    configured_assign(params, 'save_raw_frame_views', False)
    configured_assign(params, 'mask_generation_enabled', False)
    configured_assign(params, 'num_frames', num_frames)
    configured_assign(params, 'duration_seconds', duration_seconds)
    configured_assign(params, 'dynamic_bayesian_enabled', bool(args.dynamic_bayesian))
    dynamics = MotionDynamicsSettings.from_params(params)
    configured_assign(params, 'dynamic_process_noise_scale', float(args.dynamic_process_noise_scale)
        if args.dynamic_process_noise_scale is not None
        else dynamics.dynamic_process_noise_scale)
    configured_assign(params, 'dynamic_initial_variance_nm2', float(args.dynamic_initial_variance_nm2)
        if args.dynamic_initial_variance_nm2 is not None
        else dynamics.dynamic_initial_variance_nm2)
    configured_assign(params, 'dynamic_include_smoothing', bool(args.dynamic_include_smoothing))

    try:
        particles = mutable_particle_scene_from_params(params)
    except (TypeError, ValueError):
        particles = deepcopy(TEMPLATE_OVERRIDES["particles"])
        configured_assign(params, 'particles', particles)
    first = particles[0]
    first.setdefault("motion", {})
    first.setdefault("components", deepcopy(TEMPLATE_OVERRIDES["particles"][0]["components"]))
    components = first.get("components") or []
    if not components:
        components = deepcopy(TEMPLATE_OVERRIDES["particles"][0]["components"])
        first["components"] = components
    component = components[0]

    if args.diameter_nm is not None:
        component["diameter_nm"] = float(args.diameter_nm)
        first["motion"]["hydrodynamic_diameter_nm"] = float(args.diameter_nm)
    if args.material is not None:
        component["material"] = str(args.material)

    resolve_report_scene_state(params, z_nm_override=args.z_nm)


def _candidate_validation_modalities(
    params: dict[str, Any],
    resolved_modalities: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Return modality owners for lab-report base validation.

    In a single-modality simulation, parameters['imaging_model'] owns validation.
    In a generated lab Fisher comparison, the public --modality-sweep resolver
    owns the candidate set.  Using the stale base imaging_model here can reject
    a valid source-only modality before the report can render, rank, or emit
    microscope_ranking.csv.
    """

    if resolved_modalities is None:
        return [canonical_modality_name(ModalitySettings.from_params(params).modality)]
    if not resolved_modalities:
        raise ValueError("lab_fisher_report requires at least one resolved modality.")
    return [canonical_modality_name(modality) for modality in resolved_modalities]


def _validate_report_candidate_params(
    params: dict[str, Any],
    resolved_modalities: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Validate base report params under the requested candidate modalities.

    This is a user-value contract, not a cosmetic convenience: candidate
    modality validation decides whether a valid sample can reach rendering,
    Fisher/CRLB calculation, ranking, and report.md. The returned base params
    use the first candidate whose microscope runtime owners accept the shared
    report configuration.
    """

    validation_errors: dict[str, str] = {}
    for modality in _candidate_validation_modalities(params, resolved_modalities):
        candidate_params = deepcopy(params)
        configured_assign(candidate_params, 'imaging_model', modality)
        try:
            MicroscopeRuntimeSettings.from_params(candidate_params)
            return candidate_params
        except Exception as exc:  # defer per-modality failures to render loop when possible
            validation_errors[modality] = repr(exc)
    raise ValueError(
        "No requested lab Fisher modality has a valid base parameter set: "
        f"{validation_errors}"
    )


def _make_params(args: argparse.Namespace, resolved_modalities: list[str] | None = None) -> dict[str, Any]:
    params = default_params()
    params.update(deepcopy(TEMPLATE_OVERRIDES))
    assert_report_configured_profile_defaults(
        params,
        context="_make_params template defaults",
    )
    assert_report_configured_profile_particle_defaults(
        params,
        context="_make_params template defaults",
    )
    if args.instrument:
        params = apply_instrument_preset(params, args.instrument)
    if args.params_json:
        params_json = _load_json(args.params_json)
        _reject_unknown_public_keys(params_json, field_name="--params-json")
        params.update(params_json)
    _apply_cli_overrides(params, args)
    params = _validate_report_candidate_params(params, resolved_modalities)
    specs = normalize_particle_specs(params, mutate=False)
    if len(specs) != 1:
        raise ValueError(
            "lab_fisher_report currently expects exactly one logical particle. "
            "For multi-particle scenes, generate a dataset or run a targeted crop workflow."
        )
    method = BackgroundSubtractionSettings.from_params(params).method
    if method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "lab_fisher_report renders frame sequences; background_subtraction_method='video_median' "
            "is unsupported because it mixes temporal content into the contrast frames. "
            "Use 'reference_frame'."
        )
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "lab_fisher_report requires background_subtraction_method='reference_frame' "
            "so its contrast image and noise variance share the same analysis units."
        )
    if method not in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "Unsupported background_subtraction_method for lab_fisher_report: "
            f"{method!r}. Use 'reference_frame'."
        )
    return params
