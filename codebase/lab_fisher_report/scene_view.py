"""Shared scene-view ownership for lab Fisher microscope comparisons."""

from __future__ import annotations
from configured_parameters import configured_assign

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from config import MotionDynamicsSettings, SamplingGeometry
from json_utils import json_safe
from particle_specs import mutable_particle_scene_from_params
from simulation_runtime_state import runtime_state, runtime_state_or_default

REPORT_SCENE_VIEW_SCHEMA_VERSION = "syniscopy-lab-report-scene-view-v1"
REPORT_SCENE_COORDINATE_FRAME = "lab_report_shared_scene_xy_nm"


def _coerce_optional_position_nm(value: Any, *, field_name: str) -> list[float] | None:
    """Return a finite [x, y, z] position or None for template/default scenes."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be None or a three-element [x, y, z] sequence in nm.")
    out = [float(component) for component in value]
    if not all(math.isfinite(component) for component in out):
        raise ValueError(f"{field_name} must contain finite nanometer coordinates; got {value!r}.")
    return out


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serial = json.dumps(
        json_safe(payload, nonfinite="string"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]


def _first_particle(params: dict[str, Any]) -> dict[str, Any]:
    particles = mutable_particle_scene_from_params(params)
    first = particles[0]
    if not isinstance(first, dict):
        raise TypeError("parameters['particles'][0] must be a mapping for lab-report scene resolution.")
    first.setdefault("motion", {})
    if not isinstance(first["motion"], dict):
        raise TypeError("parameters['particles'][0]['motion'] must be a mapping.")
    return first


def scene_provenance_from_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the resolved report scene provenance carried by run state."""

    provenance = runtime_state_or_default(dict(params) if not isinstance(params, dict) else params).report_scene_provenance
    return dict(provenance) if isinstance(provenance, Mapping) else {}


def resolve_report_scene_state(
    params: dict[str, Any],
    *,
    z_nm_override: float | None = None,
) -> dict[str, Any]:
    """Resolve the shared lab-report scene before microscope overlays are applied.

    This is the canonical owner for report-scene defaulting.  Missing target
    positions are resolved once in a shared report scene coordinate frame, then
    copied into every microscope candidate as shared scene state.  Candidate
    pixel size, image size, backend, detector noise, or acquisition overlays must
    never resample or recenter this particle later, because that would rank
    different sample scenes while report.md and microscope_ranking.csv present a
    single best microscope/configuration recommendation.
    """

    first = _first_particle(params)
    sampling = SamplingGeometry.from_params(params)
    image_size_pixels = sampling.image_size_pixels
    pixel_size_nm = sampling.detector_pixel_size_nm

    center_nm = 0.5 * (float(image_size_pixels) - 1.0) * pixel_size_nm
    existing_position = _coerce_optional_position_nm(
        first["motion"].get("initial_position_nm"),
        field_name="particles[0].motion.initial_position_nm",
    )
    position_defaulted = existing_position is None
    resolved_position = [center_nm, center_nm, 0.0] if position_defaulted else list(existing_position)
    if z_nm_override is not None:
        z_value = float(z_nm_override)
        if not math.isfinite(z_value):
            raise ValueError(f"--z-nm must be finite; got {z_nm_override!r}.")
        resolved_position[2] = z_value

    first["motion"]["initial_position_nm"] = [float(v) for v in resolved_position]
    configured_assign(params, 'initial_z_span_nm', max(
        MotionDynamicsSettings.from_params(params).initial_z_span_nm,
        2.0 * abs(float(resolved_position[2])) + 1000.0,
    ))

    scene_basis = {
        "schema_version": REPORT_SCENE_VIEW_SCHEMA_VERSION,
        "coordinate_frame": REPORT_SCENE_COORDINATE_FRAME,
        "position_authority": "lab_report_shared_scene_before_candidate_overlays",
        "target_particle_index": 0,
        "target_particle_name": str(first.get("name", "target_particle")),
        "target_initial_position_nm": [float(v) for v in resolved_position],
        "target_position_defaulted": bool(position_defaulted),
        "shared_view_image_size_pixels": int(image_size_pixels),
        "shared_view_pixel_size_nm": float(pixel_size_nm),
        "shared_view_extent_nm": float(image_size_pixels) * float(pixel_size_nm),
        "shared_view_center_nm": [float(center_nm), float(center_nm)],
        "z_override_applied": z_nm_override is not None,
        "particle_scene": deepcopy(mutable_particle_scene_from_params(params)),
    }
    scene_basis["scene_fingerprint"] = _stable_hash(
        {
            key: scene_basis[key]
            for key in (
                "coordinate_frame",
                "target_particle_index",
                "target_initial_position_nm",
                "particle_scene",
            )
        }
    )
    state = runtime_state(params)
    state.report_scene_provenance = scene_basis
    state.report_scene_fingerprint = scene_basis["scene_fingerprint"]
    state.report_scene_coordinate_frame = REPORT_SCENE_COORDINATE_FRAME
    state.report_scene_position_authority = scene_basis["position_authority"]
    return scene_basis


__all__ = [
    "REPORT_SCENE_COORDINATE_FRAME",
    "REPORT_SCENE_VIEW_SCHEMA_VERSION",
    "resolve_report_scene_state",
    "scene_provenance_from_params",
]
