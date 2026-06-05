"""Parameter assembly for lab Fisher reports."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

from bootstrap import ensure_codebase_on_path

ensure_codebase_on_path()

from config import PARAMS, normalize_params, param_value
from modality_registry import (
    LAB_DEFAULT_MODALITIES,
    LAB_OPTICAL_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name,
)
from particle_specs import normalize_particle_specs
from postprocessing import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)
from presets import apply_instrument_preset

from .cli import TEMPLATE_OVERRIDES, _load_json

__all__ = ["_apply_cli_overrides", "_make_params", "_resolve_modalities"]


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
    return list(dict.fromkeys(modalities))


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
            params[key] = value

    num_frames = int(max(1, args.num_frames))
    fps = float(param_value(params, "fps"))
    if fps <= 0.0:
        raise ValueError("PARAMS['fps'] must be positive for sequence rendering.")

    params["return_ideal_float_frames"] = True
    params["save_frame_sequence"] = False
    params["save_raw_camera_video"] = False
    params["save_raw_camera_frame_sequence"] = False
    params["save_raw_frame_views"] = False
    params["mask_generation_enabled"] = False
    params["num_frames"] = num_frames
    params["duration_seconds"] = float(num_frames) / fps
    params["dynamic_bayesian_enabled"] = bool(args.dynamic_bayesian)
    params["dynamic_process_noise_scale"] = (
        float(args.dynamic_process_noise_scale)
        if args.dynamic_process_noise_scale is not None
        else float(param_value(params, 'dynamic_process_noise_scale'))
    )
    params["dynamic_initial_variance_nm2"] = (
        float(args.dynamic_initial_variance_nm2)
        if args.dynamic_initial_variance_nm2 is not None
        else float(param_value(params, 'dynamic_initial_variance_nm2'))
    )
    params["dynamic_include_smoothing"] = bool(args.dynamic_include_smoothing)

    particles = param_value(params, "particles")
    if not isinstance(particles, list) or not particles:
        particles = deepcopy(TEMPLATE_OVERRIDES["particles"])
        params["particles"] = particles
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

    center_nm = 0.5 * (
        float(params["image_size_pixels"]) - 1.0
    ) * float(params["pixel_size_nm"])
    first["motion"]["initial_position_nm"] = [
        center_nm,
        center_nm,
        float(args.z_nm),
    ]
    params["initial_z_span_nm"] = max(
        float(param_value(params, "initial_z_span_nm")),
        2.0 * abs(float(args.z_nm)) + 1000.0,
    )


def _make_params(args: argparse.Namespace) -> dict[str, Any]:
    params = deepcopy(PARAMS)
    params.update(deepcopy(TEMPLATE_OVERRIDES))
    if args.instrument:
        params = apply_instrument_preset(params, args.instrument)
    if args.params_json:
        params.update(_load_json(args.params_json))
    _apply_cli_overrides(params, args)
    params = normalize_params(params)
    specs = normalize_particle_specs(params, mutate=False)
    if len(specs) != 1:
        raise ValueError(
            "lab_fisher_report currently expects exactly one logical particle. "
            "For multi-particle scenes, generate a dataset or run a targeted crop workflow."
        )
    method = str(params["background_subtraction_method"]).strip().lower()
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
