"""Canonical material-volume geometry for voxelized particle components."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _positive_float(value: Any, *, field_name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive; got {value!r}.")
    return out


@dataclass(frozen=True)
class MaterialVolumeGeometry:
    """Regular body-frame occupancy grid for arbitrary particle geometry."""

    occupancy: np.ndarray
    voxel_size_nm: float

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.occupancy, dtype=float)
        if occupancy.ndim != 3 or min(occupancy.shape) <= 0:
            raise ValueError(
                "MaterialVolumeGeometry.occupancy must be a non-empty 3D array "
                f"with axes (x, y, z); got shape {occupancy.shape!r}."
            )
        if np.any(~np.isfinite(occupancy)) or np.any(occupancy < 0.0):
            raise ValueError("MaterialVolumeGeometry occupancy must be finite and non-negative.")
        voxel_size_nm = _positive_float(self.voxel_size_nm, field_name="voxel_size_nm")
        total = float(np.sum(occupancy))
        if total <= 0.0:
            raise ValueError("MaterialVolumeGeometry occupancy must contain positive material volume.")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "voxel_size_nm", voxel_size_nm)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.occupancy.shape)

    @property
    def voxel_volume_nm3(self) -> float:
        return float(self.voxel_size_nm ** 3)

    @property
    def volume_nm3(self) -> float:
        return float(np.sum(self.occupancy) * self.voxel_volume_nm3)

    @property
    def equivalent_sphere_diameter_nm(self) -> float:
        return float((6.0 * self.volume_nm3 / np.pi) ** (1.0 / 3.0))

    @property
    def bounding_radius_nm(self) -> float:
        centers, _weights = self.occupied_voxels()
        corner_half_diag = 0.5 * np.sqrt(3.0) * float(self.voxel_size_nm)
        return float(np.max(np.linalg.norm(centers, axis=1)) + corner_half_diag)

    @property
    def source_normalization_length_nm(self) -> float:
        return float(max(self.grid_shape) * self.voxel_size_nm)

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(np.asarray(self.occupancy.shape, dtype=np.int64).tobytes())
        hasher.update(np.asarray([self.voxel_size_nm], dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(self.occupancy, dtype=np.float32).tobytes())
        return hasher.hexdigest()[:24]

    def axis_centers_nm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axes = []
        for size in self.grid_shape:
            center = 0.5 * (float(size) - 1.0)
            axes.append((np.arange(size, dtype=float) - center) * float(self.voxel_size_nm))
        return tuple(axes)  # type: ignore[return-value]

    def occupied_voxels(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self.occupancy > 0.0
        indices = np.argwhere(mask)
        if indices.size == 0:
            raise ValueError("MaterialVolumeGeometry has no occupied voxels.")
        axes = self.axis_centers_nm()
        centers = np.column_stack(
            [
                axes[0][indices[:, 0]],
                axes[1][indices[:, 1]],
                axes[2][indices[:, 2]],
            ]
        )
        weights = self.occupancy[mask].astype(float, copy=False)
        return centers.astype(float, copy=False), weights

    def axial_half_extent_nm(self, orientation_matrix: np.ndarray | None = None) -> float:
        centers, _weights = self.occupied_voxels()
        R = np.eye(3, dtype=float) if orientation_matrix is None else np.asarray(orientation_matrix, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("orientation_matrix must be 3x3.")
        z_centers = (R @ centers.T)[2]
        voxel_half_z = 0.5 * float(self.voxel_size_nm) * float(np.sum(np.abs(R[2, :])))
        return float(np.max(np.abs(z_centers)) + voxel_half_z)

    def normalized_form_factor(self, q_body: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
        q = np.asarray(q_body, dtype=float)
        if q.ndim < 2 or q.shape[0] != 3:
            raise ValueError("q_body must have shape (3, ...).")
        centers, weights = self.occupied_voxels()
        total = float(np.sum(weights))
        out = np.zeros(q.shape[1:], dtype=np.complex128)
        q_flat = q.reshape(3, -1)
        out_flat = out.reshape(-1)
        for start in range(0, centers.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), centers.shape[0])
            c = centers[start:stop]
            w = weights[start:stop]
            phase = c @ q_flat
            out_flat += np.sum(w[:, None] * np.exp(1j * phase), axis=0)
        out_flat /= total
        return out

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "voxel_size_nm": float(self.voxel_size_nm),
            "occupancy": self.occupancy.astype(float, copy=False).tolist(),
            "grid_shape": list(self.grid_shape),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_public_dict(cls, value: dict[str, Any]) -> "MaterialVolumeGeometry":
        if not isinstance(value, dict):
            raise TypeError("voxel_geometry must be a dictionary.")
        return cls(
            occupancy=np.asarray(value.get("occupancy"), dtype=float),
            voxel_size_nm=_positive_float(value.get("voxel_size_nm"), field_name="voxel_geometry.voxel_size_nm"),
        )


def voxelize_mesh_file(
    mesh_path: str | Path,
    *,
    voxel_size_nm: float,
    mesh_scale_nm_per_unit: float = 1.0,
) -> MaterialVolumeGeometry:
    """Load a mesh file and voxelize it into canonical body-frame occupancy."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Mesh particle components require the optional 'trimesh' package. "
            "Install trimesh or provide shape='voxel_volume' with voxel_geometry directly."
        ) from exc

    path = Path(mesh_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Mesh component file does not exist: {path}.")
    mesh = trimesh.load(path, force="mesh")
    if mesh is None or getattr(mesh, "vertices", None) is None or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a non-empty mesh from {path}.")
    scale = _positive_float(mesh_scale_nm_per_unit, field_name="mesh_scale_nm_per_unit")
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) * scale
    voxel_size = _positive_float(voxel_size_nm, field_name="mesh_voxel_size_nm")
    voxels = mesh.voxelized(pitch=voxel_size).fill()
    occupancy = np.asarray(voxels.matrix, dtype=float)
    return MaterialVolumeGeometry(occupancy=occupancy, voxel_size_nm=voxel_size)


def load_voxel_volume_file(
    voxel_path: str | Path,
    *,
    voxel_size_nm: float,
    voxel_array_key: str | None = None,
    occupancy_threshold: float | None = None,
) -> MaterialVolumeGeometry:
    """Load a measured or precomputed voxel occupancy volume as canonical geometry."""
    path = Path(voxel_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Voxel-volume component file does not exist: {path}.")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        occupancy = np.asarray(np.load(path), dtype=float)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if voxel_array_key is None:
                keys = list(archive.files)
                if len(keys) != 1:
                    raise ValueError(
                        "NPZ voxel-volume files with multiple arrays require voxel_array_key; "
                        f"available arrays are {keys!r}."
                    )
                voxel_array_key = keys[0]
            if voxel_array_key not in archive:
                raise KeyError(
                    f"voxel_array_key={voxel_array_key!r} was not found in {path}; "
                    f"available arrays are {list(archive.files)!r}."
                )
            occupancy = np.asarray(archive[voxel_array_key], dtype=float)
    else:
        raise ValueError(
            "Voxel-volume file components currently support .npy and .npz arrays; "
            f"got {path.name!r}."
        )

    if occupancy_threshold is not None:
        threshold = float(occupancy_threshold)
        if not np.isfinite(threshold):
            raise ValueError(
                f"occupancy_threshold must be finite when provided; got {occupancy_threshold!r}."
            )
        occupancy = np.where(occupancy >= threshold, occupancy, 0.0)
    return MaterialVolumeGeometry(
        occupancy=occupancy,
        voxel_size_nm=_positive_float(voxel_size_nm, field_name="voxel_size_nm"),
    )


__all__ = [
    "MaterialVolumeGeometry",
    "load_voxel_volume_file",
    "voxelize_mesh_file",
]
