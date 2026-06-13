"""Command-line parsing and JSON/template helpers for lab Fisher reports."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from bootstrap import REPO_ROOT, ensure_codebase_on_path

ensure_codebase_on_path()

from json_utils import json_safe, load_typed_json
from config import default_params
from microscope_axes import SAMPLE, classify
from lab_fisher_report.report_contracts import (
    REPORT_CONFIGURED_PROFILE_DEFAULTS,
    REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS,
)
from modality_registry import LAB_DEFAULT_MODALITIES, REPORT_SHARED_PARAM_KEYS, require_modality_name
from modality_registry import modality_report_parameter_surface

TEMPLATE_OVERRIDES: dict[str, Any] = {
    **REPORT_CONFIGURED_PROFILE_DEFAULTS,
    "wavelength_nm": 532.0,
    "numerical_aperture": 1.0,
    "refractive_index_medium": 1.33,
    "refractive_index_immersion": 1.33,
    "background_intensity": 1000.0,
    "shot_noise_enabled": True,
    "gaussian_noise_enabled": True,
    "background_subtraction_method": "reference_frame",
    "read_noise_counts": 1.0,
    "camera_gain_e_per_count": 1.0,
    "sample_environment_enabled": False,
    "sample_environment_pattern_enabled": False,
    "sample_environment_pattern": "none",
    "particles": [
        {
            "name": "target_particle",
            "motion": {
                "hydrodynamic_diameter_nm": REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS[
                    "hydrodynamic_diameter_nm"
                ],
                "initial_position_nm": None,
            },
            "signal_multiplier": 1.0,
            "source_multiplier": 1.0,
            "components": [
                {
                    "shape": REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS["shape"],
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS[
                        "diameter_nm"
                    ],
                    "material": REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS[
                        "material"
                    ],
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        }
    ],
}

__all__ = ["TEMPLATE_OVERRIDES", "_load_json", "_parse_args", "_write_microscope_set_template"]

_TEMPLATE_ALWAYS_SHARED_KEYS = frozenset(
    {
        "particles",
        "pixel_size_nm",
        "image_size_pixels",
        "num_frames",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the same particle/scene through candidate microscopes and write a "
            "lab-facing Fisher/CRLB report."
        )
    )
    parser.add_argument(
        "--output",
        default="lab_reports/fisher_report",
        help="Output directory for CSVs, previews, maps, and report.md.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Path to a JSON file containing parameters overrides for the lab scenario.",
    )
    parser.add_argument(
        "--microscopes",
        default=None,
        help=(
            "Path to a syniscopy-microscope-set-v1 JSON file. "
            "When supplied, microscope names become the Fisher/CRLB comparison keys."
        ),
    )
    parser.add_argument(
        "--write-template",
        default=None,
        help=(
            "Write a syniscopy-microscope-set-v1 template and exit. The template "
            "uses microscope names as comparison keys and keeps modality as backend metadata."
        ),
    )
    parser.add_argument(
        "--template-modality",
        default=None,
        help=(
            "Optional modality for --write-template. When omitted, "
            "the template includes a small lab-default microscope set."
        ),
    )
    parser.add_argument(
        "--validate-microscopes",
        default=None,
        help=(
            "Validate a syniscopy-microscope-set-v1 JSON file, including fully "
            "resolved per-microscope parameters, and exit."
        ),
    )
    parser.add_argument(
        "--list-microscopes",
        default=None,
        help=(
            "Validate a syniscopy-microscope-set-v1 JSON file and print its "
            "microscope comparison keys with modality/backend metadata."
        ),
    )
    parser.add_argument(
        "--list-modalities",
        action="store_true",
        help="List supported modality names and exit.",
    )
    parser.add_argument(
        "--list-instruments",
        action="store_true",
        help="List available instrument preset names and exit.",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Optional instrument preset name from codebase/presets.py.",
    )
    parser.add_argument(
        "--modality-sweep",
        default="lab-default",
        help=(
            "Optional microscope generator: comma-separated modality names, or one of "
            "lab-default, optical, all. It emits one microscope per deduplicated "
            "modality on the shared report instrument; explicit --microscopes remains "
            "the primary path."
        ),
    )
    parser.add_argument("--diameter-nm", type=float, default=None, help="Target-particle component diameter in nm.")
    parser.add_argument("--material", default=None, help="Target-particle component material label.")
    parser.add_argument("--pixel-size-nm", type=float, default=None, help="Effective sample-plane pixel size in nm.")
    parser.add_argument("--wavelength-nm", type=float, default=None, help="Optical wavelength in nm.")
    parser.add_argument("--na", type=float, default=None, help="Numerical aperture.")
    parser.add_argument("--background-counts", type=float, default=None, help="Mean reference/background level in camera counts.")
    parser.add_argument("--read-noise-counts", type=float, default=None, help="Gaussian read-noise RMS in camera counts.")
    parser.add_argument("--camera-gain-e-per-count", type=float, default=None, help="Detected electrons per camera count/ADU.")
    parser.add_argument("--image-size-pixels", type=int, default=None, help="Square frame width/height in pixels.")
    parser.add_argument("--pupil-samples", type=int, default=None, help="Pupil samples used for optical PSF calculation.")
    parser.add_argument(
        "--z-nm",
        type=float,
        default=None,
        help=(
            "Override target-particle axial position in nm. When omitted, preserve "
            "particles[0].motion.initial_position_nm from --params-json; if no position "
            "is supplied, default to the centered template scene at z=0 nm."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to render per microscope. Multi-frame outputs enable dynamic Bayesian CRLB when enabled.",
    )
    parser.add_argument(
        "--dynamic-bayesian",
        action="store_true",
        help="Enable dynamic Bayesian CRLB estimation on the rendered frame sequence.",
    )
    parser.add_argument(
        "--dynamic-process-noise-scale",
        type=float,
        default=None,
        help="Scale factor for Brownian process prior used in dynamic Bayesian CRLB.",
    )
    parser.add_argument(
        "--dynamic-initial-variance-nm2",
        type=float,
        default=None,
        help="Initial state variance (diagonal element, nm^2) for dynamic Bayesian CRLB.",
    )
    parser.add_argument(
        "--dynamic-include-smoothing",
        action="store_true",
        help="Enable RTS-style Bayesian smoothing output for the dynamic path.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for run-scoped deterministic choices.")
    parser.add_argument(
        "--max-fusion-k",
        type=int,
        default=4,
        help="Largest subset size for best-k fusion search. Default: 4.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip preview PNGs and Fisher-density PNG/NPY writes.",
    )
    parser.add_argument(
        "--include-full-fusion",
        action="store_true",
        help="Also write the full-library fusion row when max-fusion-k is smaller than the modality count.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Return success even if one or more requested modalities fail. "
            "By default, partial reports write render_errors.json and exit nonzero."
        ),
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return load_typed_json(path, expected=dict, context="--params-json")



def _template_params_for_modality(modality: str) -> dict[str, Any]:
    canonical = require_modality_name(modality)
    surface = modality_report_parameter_surface(canonical)
    base = default_params()
    base.update(deepcopy(TEMPLATE_OVERRIDES))
    # A generated microscope entry is a sparse instrument/backend overlay.
    # The modality parameter-surface object is the single public owner for these
    # keys: template generation must not silently drop stale aliases or omit
    # canonical renderer knobs, because users build ranking-defining microscope
    # configurations from this JSON.
    local_keys = sorted(surface.template_keys(REPORT_SHARED_PARAM_KEYS))
    missing_defaults = [key for key in local_keys if key not in base]
    if missing_defaults:
        raise RuntimeError(
            f"Modality parameter surface for {canonical!r} contains key(s) "
            f"without concept-owned defaults: {missing_defaults!r}."
        )
    return {key: json_safe(base[key], nonfinite="string") for key in local_keys}


def _template_shared_params_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return shared-scene/run keys for a generated microscope-set template."""

    base = default_params()
    base.update(deepcopy(TEMPLATE_OVERRIDES))
    shared_keys = set(_TEMPLATE_ALWAYS_SHARED_KEYS)
    for modality in modalities:
        surface = modality_report_parameter_surface(require_modality_name(modality))
        shared_keys.update(
            key
            for key in surface.public_keys
            if classify(key) == SAMPLE and key != "imaging_model"
        )
    missing_defaults = [key for key in sorted(shared_keys) if key not in base]
    if missing_defaults:
        raise RuntimeError(
            "Generated microscope template shared surface contains key(s) "
            f"without concept-owned defaults: {missing_defaults!r}."
        )
    return {
        key: json_safe(base[key], nonfinite="string")
        for key in sorted(shared_keys)
    }


def _write_microscope_set_template(path: str | Path, *, modality: str | None = None) -> Path:
    """Write a public microscope-set template centered on microscope identity."""

    out = Path(path).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if modality:
        canonical = require_modality_name(modality)
        template_modalities = [canonical]
        microscope_entries = [
            {
                "name": f"{canonical}_microscope",
                "modality": canonical,
                "instrument": None,
                "params": _template_params_for_modality(canonical),
            }
        ]
    else:
        template_modalities = list(LAB_DEFAULT_MODALITIES[:3])
        microscope_entries = [
            {
                "name": f"{canonical}_microscope",
                "modality": canonical,
                "instrument": None,
                "params": _template_params_for_modality(canonical),
            }
            for canonical in template_modalities
        ]

    payload = {
        "schema": "syniscopy-microscope-set-v1",
        "shared_params": _template_shared_params_for_modalities(template_modalities),
        "microscopes": microscope_entries,
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return out
