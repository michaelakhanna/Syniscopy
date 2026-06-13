"""Source-volume coordinate and z-support ownership.

Physical source-volume support is a run-time source geometry concept, not an
imaging-model base-class concept.  This module owns the source-z frame labels
and the z-slab support descriptors used by renderers and backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SOURCE_Z_FRAME_CONTRACT_VERSION = "syniscopy-source-z-frame-v1"
SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD = "physical_sample_world"
SOURCE_Z_BASIS_FOCUS_RELATIVE = "focus_relative"
SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH = "entry_surface_depth"
SOURCE_Z_BASIS_PROJECTED_NO_Z = "projected_no_z"
VALID_SOURCE_Z_BASES = {
    SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD,
    SOURCE_Z_BASIS_FOCUS_RELATIVE,
    SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
    SOURCE_Z_BASIS_PROJECTED_NO_Z,
}


def normalize_source_z_basis(value: str, *, context: str = "source z basis") -> str:
    """Return a canonical source-z frame name and reject undeclared frames."""

    basis = str(value).strip().lower()
    if basis not in VALID_SOURCE_Z_BASES:
        raise ValueError(f"unknown {context}: {basis!r}.")
    return basis


def require_source_density_z_basis(
    source_density_z_basis: str,
    *,
    allowed_bases: set[str] | frozenset[str],
    source_input_kind: str,
    modality_name: str,
    backend_name: str | None = None,
) -> str:
    """Enforce the coordinate-frame contract at source-map boundaries."""

    basis = normalize_source_z_basis(source_density_z_basis, context="source_density_z_basis")
    allowed = {normalize_source_z_basis(item, context="allowed source_density_z_basis") for item in allowed_bases}
    if basis not in allowed:
        backend_text = "" if backend_name is None else f", backend={backend_name!r}"
        raise ValueError(
            "Source-density z frame violates the material/source coordinate contract: "
            f"modality={modality_name!r}{backend_text}, source_input_kind={source_input_kind!r}, "
            f"source_density_z_basis={basis!r}, allowed_bases={sorted(allowed)!r}. "
            "Focus-relative z is an imaging-response/defocus coordinate and must not "
            "index physical material source volumes."
        )
    return basis


def resolve_entry_surface_depth_nm(
    *,
    particle_world_z_nm: float | None,
    entry_surface_depth_nm: float | None = None,
) -> float | None:
    """Resolve a material-depth coordinate for entry-surface source volumes."""

    raw_depth = particle_world_z_nm if entry_surface_depth_nm is None else entry_surface_depth_nm
    if raw_depth is None:
        return None
    depth = float(raw_depth)
    if not np.isfinite(depth):
        raise ValueError(f"entry-surface source depth must be finite; got {raw_depth!r}.")
    return depth


@dataclass(frozen=True)
class SourceVolumeSupport:
    """Resolved z support for physical source-volume material maps."""

    modality: str
    source_z_basis: str
    policy: str
    configured_slice_count: int
    required_slice_count: int
    slice_count: int
    slice_thickness_nm: float
    z_center_nm: float
    z_min_nm: float
    z_max_nm: float
    envelope_min_nm: float
    envelope_max_nm: float
    preserved_configured_center: bool

    @property
    def source_z_planes_nm(self) -> np.ndarray:
        return (
            np.arange(int(self.slice_count), dtype=float)
            - 0.5 * float(int(self.slice_count) - 1)
        ) * float(self.slice_thickness_nm) + float(self.z_center_nm)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "source_z_basis": self.source_z_basis,
            "source_z_support_policy": self.policy,
            "source_volume_configured_slices": int(self.configured_slice_count),
            "source_volume_required_slices_for_rendered_z": int(self.required_slice_count),
            "source_volume_slices": int(self.slice_count),
            "source_slice_thickness_nm": float(self.slice_thickness_nm),
            "source_z_center_nm": float(self.z_center_nm),
            "source_z_min_nm": float(self.z_min_nm),
            "source_z_max_nm": float(self.z_max_nm),
            "source_z_envelope_min_nm": float(self.envelope_min_nm),
            "source_z_envelope_max_nm": float(self.envelope_max_nm),
            "source_z_preserved_configured_center": bool(self.preserved_configured_center),
            "source_z_planes_nm": self.source_z_planes_nm.astype(float).tolist(),
        }


def resolve_offset_source_volume_support(
    *,
    modality: str,
    source_z_basis: str,
    configured_slice_count: int,
    slice_thickness_nm: float,
    envelope_min_nm: float,
    envelope_max_nm: float,
    configured_min_nm: float = 0.0,
    min_allowed_z_nm: float | None = None,
    policy: str = "auto_from_rendered_offset_source_envelope",
) -> SourceVolumeSupport:
    """Resolve an offset-anchored physical-z source slab that covers material."""

    n_configured = int(configured_slice_count)
    if n_configured <= 0:
        raise ValueError(f"configured_slice_count must be positive; got {configured_slice_count!r}.")
    dz_nm = float(slice_thickness_nm)
    if not np.isfinite(dz_nm) or dz_nm <= 0.0:
        raise ValueError(f"slice_thickness_nm must be finite and positive; got {slice_thickness_nm!r}.")
    env_min = float(envelope_min_nm)
    env_max = float(envelope_max_nm)
    if not np.isfinite(env_min) or not np.isfinite(env_max):
        raise ValueError(
            "Source-volume support requires finite rendered source-envelope bounds; "
            f"got {envelope_min_nm!r}, {envelope_max_nm!r}."
        )
    if env_max < env_min:
        env_min, env_max = env_max, env_min

    configured_min = float(configured_min_nm)
    if not np.isfinite(configured_min):
        raise ValueError(f"configured_min_nm must be finite; got {configured_min_nm!r}.")
    if min_allowed_z_nm is not None:
        min_allowed = float(min_allowed_z_nm)
        if not np.isfinite(min_allowed):
            raise ValueError(f"min_allowed_z_nm must be finite when supplied; got {min_allowed_z_nm!r}.")
        tolerance = 1.0e-9 * max(abs(env_min), abs(env_max), abs(min_allowed), dz_nm, 1.0)
        if env_min < min_allowed - tolerance:
            raise ValueError(
                "Source-volume envelope extends outside the allowed physical z support: "
                f"envelope_min_nm={env_min:g}, min_allowed_z_nm={min_allowed:g}. "
                "For SEM entry-surface source volumes this means particle material was "
                "placed above the entry surface; use a non-negative entry-surface depth "
                "envelope or a projected SEM source representation."
            )
        configured_min = max(configured_min, min_allowed)

    configured_max = configured_min + float(n_configured) * dz_nm
    tolerance = 1.0e-9 * max(abs(configured_min), abs(configured_max), abs(env_min), abs(env_max), dz_nm, 1.0)
    envelope_span_slices = max(1, int(np.ceil(max(env_max - env_min, dz_nm) / dz_nm)))
    if env_min >= configured_min - tolerance and env_max <= configured_max + tolerance:
        return SourceVolumeSupport(
            modality=str(modality),
            source_z_basis=str(source_z_basis),
            policy=str(policy),
            configured_slice_count=n_configured,
            required_slice_count=int(envelope_span_slices),
            slice_count=n_configured,
            slice_thickness_nm=dz_nm,
            z_center_nm=float(configured_min + 0.5 * (configured_max - configured_min)),
            z_min_nm=float(configured_min),
            z_max_nm=float(configured_max),
            envelope_min_nm=env_min,
            envelope_max_nm=env_max,
            preserved_configured_center=True,
        )

    support_min = min(configured_min, env_min)
    if min_allowed_z_nm is not None:
        support_min = max(support_min, float(min_allowed_z_nm))
    required_slices = max(1, int(np.ceil(max(env_max - support_min, dz_nm) / dz_nm)))
    effective_slices = max(n_configured, required_slices)
    while support_min + float(effective_slices) * dz_nm + tolerance < env_max:
        effective_slices += 1
    support_max = support_min + float(effective_slices) * dz_nm
    return SourceVolumeSupport(
        modality=str(modality),
        source_z_basis=str(source_z_basis),
        policy=str(policy),
        configured_slice_count=n_configured,
        required_slice_count=int(required_slices),
        slice_count=int(effective_slices),
        slice_thickness_nm=dz_nm,
        z_center_nm=float(support_min + 0.5 * (support_max - support_min)),
        z_min_nm=float(support_min),
        z_max_nm=float(support_max),
        envelope_min_nm=env_min,
        envelope_max_nm=env_max,
        preserved_configured_center=False,
    )


def resolve_uniform_source_volume_support(
    *,
    modality: str,
    source_z_basis: str,
    configured_slice_count: int,
    slice_thickness_nm: float,
    envelope_min_nm: float,
    envelope_max_nm: float,
    configured_center_nm: float = 0.0,
    policy: str = "auto_from_rendered_trajectory_envelope",
) -> SourceVolumeSupport:
    """Resolve a uniform physical-z source slab that covers rendered material."""

    n_configured = int(configured_slice_count)
    if n_configured <= 0:
        raise ValueError(f"configured_slice_count must be positive; got {configured_slice_count!r}.")
    dz_nm = float(slice_thickness_nm)
    if not np.isfinite(dz_nm) or dz_nm <= 0.0:
        raise ValueError(f"slice_thickness_nm must be finite and positive; got {slice_thickness_nm!r}.")
    env_min = float(envelope_min_nm)
    env_max = float(envelope_max_nm)
    if not np.isfinite(env_min) or not np.isfinite(env_max):
        raise ValueError(
            "Source-volume support requires finite rendered source-envelope bounds; "
            f"got {envelope_min_nm!r}, {envelope_max_nm!r}."
        )
    if env_max < env_min:
        env_min, env_max = env_max, env_min

    configured_center = float(configured_center_nm)
    if not np.isfinite(configured_center):
        raise ValueError(f"configured_center_nm must be finite; got {configured_center_nm!r}.")
    configured_half_span = 0.5 * float(n_configured) * dz_nm
    configured_min = configured_center - configured_half_span
    configured_max = configured_center + configured_half_span
    tolerance = 1.0e-9 * max(abs(configured_min), abs(configured_max), abs(env_min), abs(env_max), dz_nm, 1.0)
    if env_min >= configured_min - tolerance and env_max <= configured_max + tolerance:
        required_slices = max(1, int(np.ceil(max(env_max - env_min, dz_nm) / dz_nm)))
        return SourceVolumeSupport(
            modality=str(modality),
            source_z_basis=str(source_z_basis),
            policy=str(policy),
            configured_slice_count=n_configured,
            required_slice_count=int(required_slices),
            slice_count=n_configured,
            slice_thickness_nm=dz_nm,
            z_center_nm=configured_center,
            z_min_nm=float(configured_min),
            z_max_nm=float(configured_max),
            envelope_min_nm=env_min,
            envelope_max_nm=env_max,
            preserved_configured_center=True,
        )

    required_span_nm = max(float(env_max - env_min), dz_nm)
    required_slices = max(1, int(np.ceil(required_span_nm / dz_nm)))
    effective_slices = max(n_configured, required_slices)
    while float(effective_slices) * dz_nm + tolerance < float(env_max - env_min):
        effective_slices += 1
    center_nm = 0.5 * (env_min + env_max)
    half_span_nm = 0.5 * float(effective_slices) * dz_nm
    return SourceVolumeSupport(
        modality=str(modality),
        source_z_basis=str(source_z_basis),
        policy=str(policy),
        configured_slice_count=n_configured,
        required_slice_count=int(required_slices),
        slice_count=int(effective_slices),
        slice_thickness_nm=dz_nm,
        z_center_nm=float(center_nm),
        z_min_nm=float(center_nm - half_span_nm),
        z_max_nm=float(center_nm + half_span_nm),
        envelope_min_nm=env_min,
        envelope_max_nm=env_max,
        preserved_configured_center=False,
    )


__all__ = [
    "SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH",
    "SOURCE_Z_BASIS_FOCUS_RELATIVE",
    "SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD",
    "SOURCE_Z_BASIS_PROJECTED_NO_Z",
    "SOURCE_Z_FRAME_CONTRACT_VERSION",
    "SourceVolumeSupport",
    "VALID_SOURCE_Z_BASES",
    "normalize_source_z_basis",
    "require_source_density_z_basis",
    "resolve_entry_surface_depth_nm",
    "resolve_offset_source_volume_support",
    "resolve_uniform_source_volume_support",
]
