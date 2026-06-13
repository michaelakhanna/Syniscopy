"""Shared fluorescence source-layer placement contracts.

The fluorescence and TIRF renderers both accept 2D sample-environment maps and
3D z-y-x emitter-density source stacks.  A 2D surface source cannot be inserted
into a 3D physical source stack by guessing that world z=0 is always the
correct layer: widefield and TIRF have different interface conventions.  This
module makes the source-layer physical z explicit before the vectorial PSF
backend consumes slice-specific defocus positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FLUORESCENCE_SOURCE_LAYER_CONTRACT_VERSION = "syniscopy-fluorescence-source-layer-v1"
FLUORESCENCE_SOURCE_LAYER_INSERTION_POLICY = "physical_world_z_to_configured_source_slice"
FLUORESCENCE_SOURCE_BASIS_EMITTER_DENSITY = "fluorescence_emitter_density"
FLUORESCENCE_SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD = "physical_sample_world"
FLUORESCENCE_SOURCE_ROLE_SAMPLE_ENVIRONMENT_AUTOFLUORESCENCE = "sample_environment_autofluorescence"


@dataclass(frozen=True)
class FluorescenceSourceLayerPlacement:
    """Resolved physical placement for a 2D fluorescence source layer.

    The layer remains an emitter-density source until the fluorescence backend
    applies the photon/count budget.  ``world_z_nm`` is the physical sample-world
    plane where that layer belongs in a 3D source stack; optical defocus remains
    a separate downstream transformation of ``world_z_nm - focus_plane_z_nm``.
    """

    role: str
    source_basis: str
    z_basis: str
    world_z_nm: float
    insertion_policy: str = FLUORESCENCE_SOURCE_LAYER_INSERTION_POLICY
    contract_version: str = FLUORESCENCE_SOURCE_LAYER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        world_z = float(self.world_z_nm)
        if not np.isfinite(world_z):
            raise ValueError(f"fluorescence source-layer world_z_nm must be finite; got {self.world_z_nm!r}.")
        if str(self.source_basis) != FLUORESCENCE_SOURCE_BASIS_EMITTER_DENSITY:
            raise ValueError(
                "fluorescence source layers currently support only emitter-density "
                f"sources; got source_basis={self.source_basis!r}."
            )
        if str(self.z_basis) != FLUORESCENCE_SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD:
            raise ValueError(
                "fluorescence source-layer z placement must be physical sample-world z; "
                f"got z_basis={self.z_basis!r}."
            )
        object.__setattr__(self, "world_z_nm", world_z)

    def volume_slice_index(
        self,
        source_z_planes_nm: np.ndarray | list[float] | tuple[float, ...],
        *,
        source_slice_thickness_nm: float,
    ) -> int:
        """Return the configured source slice containing this physical layer.

        Failing when the layer is outside the configured source support is
        intentional.  Reassigning an interface source to the nearest in-range
        slice silently changes its vectorial emission defocus and therefore the
        detector-count and Fisher/CRLB basis.
        """
        planes = np.asarray(source_z_planes_nm, dtype=float).reshape(-1)
        thickness = float(source_slice_thickness_nm)
        if planes.size == 0:
            raise ValueError("fluorescence source-layer placement requires at least one source z plane.")
        if not np.all(np.isfinite(planes)):
            raise ValueError("fluorescence source z planes must be finite for layer placement.")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "fluorescence source slice thickness must be finite and positive for layer placement; "
                f"got {source_slice_thickness_nm!r}."
            )
        half = 0.5 * thickness
        lower = planes - half
        upper = planes + half
        scale = max(float(np.max(np.abs(planes))) if planes.size else 0.0, abs(self.world_z_nm), thickness, 1.0)
        eps = 1.0e-12 * scale
        matches = np.flatnonzero((self.world_z_nm >= lower - eps) & (self.world_z_nm <= upper + eps))
        if matches.size == 0:
            raise ValueError(
                f"{self.role} layer at physical z={self.world_z_nm:g} nm is outside the configured "
                f"fluorescence source-stack support [{float(np.min(lower)):g}, {float(np.max(upper)):g}] nm. "
                "Increase the fluorescence/TIRF volume slice support or use projected_2d; a surface layer "
                "must not be silently reassigned to a different physical z plane."
            )
        nearest = matches[np.argmin(np.abs(planes[matches] - self.world_z_nm))]
        return int(nearest)

    def metadata(self) -> dict[str, Any]:
        return {
            "fluorescence_source_layer_contract_version": self.contract_version,
            "fluorescence_source_layer_role": self.role,
            "fluorescence_source_layer_basis": self.source_basis,
            "fluorescence_source_layer_z_basis": self.z_basis,
            "fluorescence_source_layer_world_z_nm": self.world_z_nm,
            "fluorescence_source_layer_insertion_policy": self.insertion_policy,
        }


__all__ = [
    "FLUORESCENCE_SOURCE_BASIS_EMITTER_DENSITY",
    "FLUORESCENCE_SOURCE_LAYER_CONTRACT_VERSION",
    "FLUORESCENCE_SOURCE_LAYER_INSERTION_POLICY",
    "FLUORESCENCE_SOURCE_ROLE_SAMPLE_ENVIRONMENT_AUTOFLUORESCENCE",
    "FLUORESCENCE_SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD",
    "FluorescenceSourceLayerPlacement",
]
