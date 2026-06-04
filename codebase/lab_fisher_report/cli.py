"""Command-line parsing and JSON/template helpers for lab Fisher reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bootstrap import REPO_ROOT, ensure_codebase_on_path

ensure_codebase_on_path()

from json_utils import json_safe, load_typed_json

TEMPLATE_OVERRIDES: dict[str, Any] = {
    "image_size_pixels": 192,
    "pixel_size_nm": 65.0,
    "pupil_samples": 192,
    "psf_oversampling_factor": 2,
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
                "hydrodynamic_diameter_nm": 100.0,
                "initial_position_nm": None,
            },
            "signal_multiplier": 1.0,
            "source_multiplier": 1.0,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": 100.0,
                    "material": "fluorescent_polystyrene",
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        }
    ],
}

__all__ = ["TEMPLATE_OVERRIDES", "_load_json", "_parse_args", "_write_template"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the same particle through candidate modalities and write a "
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
        help="Path to a JSON file containing PARAMS overrides for the lab scenario.",
    )
    parser.add_argument(
        "--write-template",
        default=None,
        help="Write a small editable lab parameter template and exit.",
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
        "--modalities",
        default="lab-default",
        help=(
            "Comma-separated modality names, or one of: lab-default, optical, "
            "all. Default: lab-default."
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
    parser.add_argument("--z-nm", type=float, default=0.0, help="Target-particle axial position in nm; this report still computes lateral CRLBs.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to render per modality. Multi-frame outputs enable dynamic Bayesian CRLB when enabled.",
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


def _write_template(path: str | Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_safe(TEMPLATE_OVERRIDES, nonfinite="string"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return out


def _load_json(path: str | Path) -> dict[str, Any]:
    return load_typed_json(path, expected=dict, context="--params-json")
