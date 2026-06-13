"""Typed run-local state kept out of the public parameters concept."""

from __future__ import annotations
from configured_parameters import configured_assign, configured_optional

from dataclasses import dataclass, field
from typing import Any, MutableMapping

from source_volume_support import SourceVolumeSupport


RUNTIME_STATE_PARAM_KEY = "_runtime_state"


@dataclass
class SimulationRuntimeState:
    """Derived/cache state owned by a simulation run, not by public parameters."""

    detector_static_seed: int | None = None
    substrate_pattern_layout_cache_token: str | None = None
    substrate_pattern_layout_extent_nm: float | None = None
    substrate_pattern_layout_rng: Any | None = None
    exposure_signal_scale: float = 1.0
    focus_plane_z_nm: float = 0.0
    particle_specs: Any | None = None
    particle_specs_fingerprint: str | None = None
    resolved_primary_component_refractive_indices_fingerprint: str | None = None
    resolved_primary_component_refractive_indices: Any | None = None
    resolved_particle_material_properties_fingerprint: str | None = None
    resolved_particle_material_properties: list[Any] | None = None
    resolved_particle_material_properties_metadata: list[Any] | None = None
    return_mask_arrays: bool = False
    write_mask_files: bool = True
    generated_spectral_channels: bool = False
    spectral_channel_count: int | None = None
    source_volume_supports: dict[str, SourceVolumeSupport] = field(default_factory=dict)
    report_scene_provenance: dict[str, Any] | None = None
    report_scene_fingerprint: str | None = None
    report_scene_coordinate_frame: str | None = None
    report_scene_position_authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_static_seed": self.detector_static_seed,
            "substrate_pattern_layout_cache_token": self.substrate_pattern_layout_cache_token,
            "substrate_pattern_layout_extent_nm": self.substrate_pattern_layout_extent_nm,
            "substrate_pattern_layout_rng_present": self.substrate_pattern_layout_rng is not None,
            "exposure_signal_scale": float(self.exposure_signal_scale),
            "focus_plane_z_nm": float(self.focus_plane_z_nm),
            "particle_specs": self.particle_specs,
            "particle_specs_fingerprint": self.particle_specs_fingerprint,
            "resolved_primary_component_refractive_indices_fingerprint": (
                self.resolved_primary_component_refractive_indices_fingerprint
            ),
            "resolved_primary_component_refractive_indices": self.resolved_primary_component_refractive_indices,
            "resolved_particle_material_properties_fingerprint": (
                self.resolved_particle_material_properties_fingerprint
            ),
            "resolved_particle_material_properties": self.resolved_particle_material_properties,
            "resolved_particle_material_properties_metadata": self.resolved_particle_material_properties_metadata,
            "return_mask_arrays": bool(self.return_mask_arrays),
            "write_mask_files": bool(self.write_mask_files),
            "generated_spectral_channels": bool(self.generated_spectral_channels),
            "spectral_channel_count": self.spectral_channel_count,
            "source_volume_supports": {
                str(prefix): (
                    support.to_metadata()
                    if hasattr(support, "to_metadata")
                    else support
                )
                for prefix, support in self.source_volume_supports.items()
            },
            "report_scene_provenance": self.report_scene_provenance,
            "report_scene_fingerprint": self.report_scene_fingerprint,
            "report_scene_coordinate_frame": self.report_scene_coordinate_frame,
            "report_scene_position_authority": self.report_scene_position_authority,
        }


def _state_from_payload(payload: Any) -> SimulationRuntimeState:
    if isinstance(payload, SimulationRuntimeState):
        return payload
    if payload is None:
        return SimulationRuntimeState()
    raise TypeError(
        f"{RUNTIME_STATE_PARAM_KEY} must be SimulationRuntimeState; "
        f"got {type(payload).__name__}."
    )


def runtime_state(params: MutableMapping[str, Any]) -> SimulationRuntimeState:
    """Return the mutable runtime state object attached to a params copy."""

    state = _state_from_payload(configured_optional(params, RUNTIME_STATE_PARAM_KEY))
    if configured_optional(params, RUNTIME_STATE_PARAM_KEY) is not state:
        configured_assign(params, RUNTIME_STATE_PARAM_KEY, state)
    return state


def runtime_state_or_default(params: dict[str, Any] | None) -> SimulationRuntimeState:
    if isinstance(params, dict) and RUNTIME_STATE_PARAM_KEY in params:
        return _state_from_payload(configured_optional(params, RUNTIME_STATE_PARAM_KEY))
    return SimulationRuntimeState()


def pop_runtime_state(params: MutableMapping[str, Any]) -> SimulationRuntimeState | None:
    """Remove and return attached run-local state from a config mapping."""

    if RUNTIME_STATE_PARAM_KEY not in params:
        return None
    return _state_from_payload(params.pop(RUNTIME_STATE_PARAM_KEY))


def attach_runtime_state(
    params: MutableMapping[str, Any],
    state: SimulationRuntimeState | None,
) -> None:
    """Attach run-local state to a config mapping after public validation."""

    if state is None:
        return
    if not isinstance(state, SimulationRuntimeState):
        raise TypeError(
            "attach_runtime_state requires a SimulationRuntimeState object; "
            f"got {type(state).__name__}."
        )
    configured_assign(params, RUNTIME_STATE_PARAM_KEY, state)


def config_without_runtime_state(params: MutableMapping[str, Any]) -> dict[str, Any]:
    """Return a plain config copy with no attached run-local state."""

    out = dict(params)
    out.pop(RUNTIME_STATE_PARAM_KEY, None)
    return out


def set_source_volume_support(
    params: MutableMapping[str, Any],
    prefix: str,
    support: SourceVolumeSupport,
) -> None:
    if not isinstance(support, SourceVolumeSupport):
        raise TypeError(
            "set_source_volume_support requires a SourceVolumeSupport object; "
            f"got {type(support).__name__}."
        )
    runtime_state(params).source_volume_supports[str(prefix)] = support


def get_source_volume_support(params: dict[str, Any] | None, prefix: str) -> SourceVolumeSupport | None:
    if not isinstance(params, dict):
        return None
    return runtime_state_or_default(params).source_volume_supports.get(str(prefix))


def clear_source_volume_support(params: MutableMapping[str, Any], prefix: str) -> None:
    runtime_state(params).source_volume_supports.pop(str(prefix), None)


__all__ = [
    "SimulationRuntimeState",
    "attach_runtime_state",
    "clear_source_volume_support",
    "config_without_runtime_state",
    "get_source_volume_support",
    "pop_runtime_state",
    "runtime_state_or_default",
    "runtime_state",
    "set_source_volume_support",
]
