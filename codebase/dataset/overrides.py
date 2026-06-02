"""Dataset parameter override normalization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from config import PARAMS, normalize_params
from trajectory import resolve_public_num_frames
from particle_specs import normalize_particle_specs

_PARTICLE_OVERRIDE_KEYS = {"particles"}
_DATASET_MANAGED_OVERRIDE_KEYS = {
    "mask_output_directory",
    "multichannel_sidecar_directory",
    "output_filename",
    "random_seed",
}

def _normalize_num_frames_override(params, override_keys):
    """
    Make dataset-level num_frames overrides honor the renderer timebase.

    The renderer computes actual frame count as int(fps * duration_seconds).
    Explicit num_frames overrides therefore update duration_seconds so the
    renderer produces the requested count exactly.
    """
    if "num_frames" not in override_keys:
        return params

    resolve_public_num_frames(
        params,
        drop_num_frames=True,
        enforce_existing_duration="duration_seconds" in override_keys,
    )
    return params

def _reject_dataset_managed_overrides(
    overrides: Optional[Mapping[str, Any]],
    override_name: str,
) -> None:
    if not overrides:
        return
    managed = sorted(
        str(key)
        for key in overrides
        if str(key) in _DATASET_MANAGED_OVERRIDE_KEYS
    )
    if managed:
        raise ValueError(
            f"{override_name} contains dataset-managed key(s) {managed}. "
            "Pass dataset output and seed settings to generate_dataset() instead."
        )

def apply_parameter_overrides(
    params: Dict[str, Any],
    param_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return a copy of ``params`` with explicit user overrides applied.

    This is the supported programmatic entry point for notebooks and other UIs that
    need to expose dataset-generation knobs without editing the ``config`` package or
    module globals. Values are copied into the returned
    dictionary; the input dictionary is not modified.

    Particle identity, material, geometry, and motion are supplied through the
    canonical ``particles`` object. Other particle-list formats are not
    accepted here.
    """
    out = deepcopy(params)
    if not param_overrides:
        if out.get("num_frames") is not None:
            _normalize_num_frames_override(out, {"num_frames"})
        out = normalize_params(out, allowed_internal_keys=set(out))
        normalize_particle_specs(out, mutate=True)
        return out

    normalized_overrides: Dict[str, Any] = {}
    for raw_key, value in param_overrides.items():
        canonical_key = str(raw_key)
        allowed_extra_keys = {"num_frames", *_PARTICLE_OVERRIDE_KEYS}
        if canonical_key not in PARAMS and canonical_key not in allowed_extra_keys:
            raise ValueError(
                f"Unknown parameter override {canonical_key!r}. Use only keys from "
                "config.PARAMS plus the canonical particles object."
            )
        if canonical_key in _DATASET_MANAGED_OVERRIDE_KEYS:
            raise ValueError(
                f"Parameter override {canonical_key!r} is managed by dataset generation. "
                "Pass dataset output and seed settings to generate_dataset() instead."
            )
        if canonical_key in normalized_overrides:
            raise ValueError(
                f"Duplicate parameter override {canonical_key!r}; each "
                "override object key may be supplied only once."
            )
        normalized_overrides[canonical_key] = value

    override_keys = set(normalized_overrides)
    if out.get("num_frames") is not None:
        override_keys.add("num_frames")
    for key, value in normalized_overrides.items():
        out[key] = deepcopy(value)

    _normalize_num_frames_override(out, override_keys)
    out = normalize_params(
        out,
        allowed_extra_keys=_PARTICLE_OVERRIDE_KEYS,
        allowed_internal_keys=set(out),
    )
    normalize_particle_specs(out, mutate=True)
    return out

def _coerce_positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer.") from exc
    min_value = 0 if allow_zero else 1
    if int_value < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}.")
    return int_value

def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
