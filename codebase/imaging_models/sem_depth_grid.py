"""Shared SEM source-depth grid contract helpers.

The SEM renderer, metadata, sample-environment source injection, and transport
backends must agree on the physical depth interval represented by source_stack[i].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from config.runtime import SemSettings
from simulation_runtime_state import get_source_volume_support
from source_volume_support import (
    SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
    SourceVolumeSupport,
    resolve_offset_source_volume_support,
)

SEM_DEPTH_GRID_CONTRACT_VERSION = "syniscopy-sem-depth-grid-v1"
SEM_DEPTH_GRID_OFFSET_POLICY = "slice_grid_origin_offset"

def _as_float(value: Any, *, key: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"parameters[{key!r}] must be finite; got {value!r}.")
    return out

def resolve_sem_volume_slice_thickness_nm(params: Mapping[str, Any], *, backend_name: str) -> float:
    return SemSettings.from_params(params).volume_slice_thickness_for_backend(backend_name)

@dataclass(frozen=True)
class SEMDepthGrid:
    """Physical entry-surface depth grid for SEM z-y-x source volumes."""
    slice_count: int
    slice_thickness_nm: float
    offset_nm: float
    origin: str = "entry_surface_depth"
    offset_policy: str = SEM_DEPTH_GRID_OFFSET_POLICY
    contract_version: str = SEM_DEPTH_GRID_CONTRACT_VERSION

    def __post_init__(self) -> None:
        thickness = float(self.slice_thickness_nm)
        offset = float(self.offset_nm)
        origin = str(self.origin).strip().lower()
        if int(self.slice_count) <= 0:
            raise ValueError("SEM depth grid requires at least one slice.")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("SEM depth-grid slice thickness must be positive and finite.")
        if not np.isfinite(offset):
            raise ValueError("SEM depth-grid offset must be finite.")
        if origin != "entry_surface_depth":
            raise ValueError("SEM volume depth grids require source_z_origin='entry_surface_depth'.")
        if offset < 0.0:
            raise ValueError("SEM volume depth grids use entry-surface depth; sem_source_z_offset_nm must be non-negative for volume SEM transport.")
        if str(self.offset_policy) != SEM_DEPTH_GRID_OFFSET_POLICY:
            raise ValueError(f"SEM depth-grid offset_policy must be {SEM_DEPTH_GRID_OFFSET_POLICY!r}.")
        object.__setattr__(self, "slice_count", int(self.slice_count))
        object.__setattr__(self, "slice_thickness_nm", thickness)
        object.__setattr__(self, "offset_nm", offset)
        object.__setattr__(self, "origin", origin)

    @property
    def edges_nm(self) -> list[float]:
        return [self.offset_nm + idx * self.slice_thickness_nm for idx in range(self.slice_count + 1)]

    @property
    def centers_nm(self) -> list[float]:
        return [self.offset_nm + (idx + 0.5) * self.slice_thickness_nm for idx in range(self.slice_count)]

    @property
    def depth_nm(self) -> float:
        return self.slice_count * self.slice_thickness_nm

    def slice_bounds_nm(self, index: int) -> tuple[float, float]:
        idx = int(index)
        if idx < 0 or idx >= self.slice_count:
            raise IndexError(f"SEM depth-grid slice index out of range: {index!r}.")
        z0 = self.offset_nm + idx * self.slice_thickness_nm
        return z0, z0 + self.slice_thickness_nm

    def slice_index_for_depth_nm(self, depth_nm: float) -> int | None:
        depth = float(depth_nm)
        if not np.isfinite(depth):
            raise ValueError(f"SEM depth must be finite; got {depth_nm!r}.")
        idx = int(np.floor((depth - self.offset_nm) / self.slice_thickness_nm))
        return None if idx < 0 or idx >= self.slice_count else idx

    def surface_slice_index(self) -> int:
        idx = self.slice_index_for_depth_nm(0.0)
        if idx is None:
            raise ValueError(
                "SEM sample-environment surface source is at entry-surface depth z=0 nm, "
                f"but the configured SEM volume depth grid starts at {self.offset_nm:g} nm. "
                "Use sem_source_z_offset_nm=0 for volume SEM sample-environment rendering or disable the sample environment."
            )
        return idx

    def metadata(self) -> dict[str, Any]:
        return {
            "source_depth_grid_contract_version": self.contract_version,
            "source_depth_grid_offset_policy": self.offset_policy,
            "source_z_origin": self.origin,
            "source_z_offset_nm": self.offset_nm,
            "source_slice_thickness_nm": self.slice_thickness_nm,
            "source_z_edges_nm": self.edges_nm,
            "source_z_planes_nm": self.centers_nm,
        }

def sem_source_volume_support_from_params(
    params: Mapping[str, Any],
    *,
    backend_name: str,
    envelope_min_nm: float,
    envelope_max_nm: float,
    policy: str = "auto_from_rendered_sem_material_envelope",
) -> SourceVolumeSupport:
    """Resolve SEM source-volume support before rasterization or backend setup.

    SEM volume transport has two depth-dependent consumers: the material source
    rasterizer and the electron-transport kernel stack.  Both must receive the
    same offset-anchored support; otherwise a local rasterization fix can stop
    clipping material while leaving the kernel indexed to a different physical
    depth interval.
    """
    settings = SemSettings.from_params(params)
    if settings.effective_source_representation != "volume":
        raise ValueError("SEM source-volume support resolution is only valid for volume SEM backends.")
    if settings.source_z_origin != "entry_surface_depth":
        raise ValueError("SEM source-volume support requires sem_source_z_origin='entry_surface_depth'.")
    return resolve_offset_source_volume_support(
        modality="sem_secondary_electron",
        source_z_basis=SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
        configured_slice_count=settings.volume_slices,
        slice_thickness_nm=settings.volume_slice_thickness_for_backend(backend_name),
        envelope_min_nm=float(envelope_min_nm),
        envelope_max_nm=float(envelope_max_nm),
        configured_min_nm=settings.source_z_offset_nm,
        min_allowed_z_nm=0.0,
        policy=policy,
    )


def sem_depth_grid_from_params(params: Mapping[str, Any], *, backend_name: str) -> SEMDepthGrid:
    settings = SemSettings.from_params(params)
    support = get_source_volume_support(dict(params) if not isinstance(params, dict) else params, "sem")
    slice_count = settings.volume_slices if support is None else int(support.slice_count)
    slice_thickness = (
        settings.volume_slice_thickness_for_backend(backend_name)
        if support is None
        else float(support.slice_thickness_nm)
    )
    offset_nm = settings.source_z_offset_nm if support is None else float(support.z_min_nm)
    return SEMDepthGrid(
        slice_count=slice_count,
        slice_thickness_nm=slice_thickness,
        offset_nm=offset_nm,
        origin=settings.source_z_origin,
    )
