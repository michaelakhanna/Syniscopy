"""
Factories for particle-object component lists.

Each function returns a list suitable for
parameters["particles"][i]["components"]. Offsets are in nanometres in the
particle body frame. The renderer rotates these offsets by the particle's
per-frame orientation matrix before placing component PSFs.
"""

from __future__ import annotations

import math
from typing import Any


def component(
    offset_nm: list[float],
    *,
    shape: str = "sphere",
    diameter_nm: float | None = None,
    axes_nm: list[float] | tuple[float, float, float] | None = None,
    length_nm: float | None = None,
    voxel_geometry: dict[str, Any] | None = None,
    voxel_path: str | None = None,
    voxel_size_nm: float | None = None,
    voxel_array_key: str | None = None,
    voxel_occupancy_threshold: float | None = None,
    mesh_path: str | None = None,
    mesh_voxel_size_nm: float | None = None,
    mesh_scale_nm_per_unit: float | None = None,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one primitive render component with a body-frame offset."""
    shape_name = str(shape).strip().lower()
    out = {
        "shape": shape_name,
        "offset_nm": [float(offset_nm[0]), float(offset_nm[1]), float(offset_nm[2])],
        "material": material,
        "refractive_index": refractive_index,
        "signal_multiplier": float(signal_multiplier),
        "source_multiplier": float(source_multiplier),
        "material_properties": material_properties,
    }
    if diameter_nm is not None:
        out["diameter_nm"] = float(diameter_nm)
    if axes_nm is not None:
        out["axes_nm"] = [float(axes_nm[0]), float(axes_nm[1]), float(axes_nm[2])]
    if length_nm is not None:
        out["length_nm"] = float(length_nm)
    if voxel_geometry is not None:
        out["voxel_geometry"] = voxel_geometry
    if voxel_path is not None:
        out["voxel_path"] = str(voxel_path)
    if voxel_size_nm is not None:
        out["voxel_size_nm"] = float(voxel_size_nm)
    if voxel_array_key is not None:
        out["voxel_array_key"] = str(voxel_array_key)
    if voxel_occupancy_threshold is not None:
        out["voxel_occupancy_threshold"] = float(voxel_occupancy_threshold)
    if mesh_path is not None:
        out["mesh_path"] = str(mesh_path)
    if mesh_voxel_size_nm is not None:
        out["mesh_voxel_size_nm"] = float(mesh_voxel_size_nm)
    if mesh_scale_nm_per_unit is not None:
        out["mesh_scale_nm_per_unit"] = float(mesh_scale_nm_per_unit)
    return out


def sphere_component(
    offset_nm: list[float] | None = None,
    *,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one exact sphere primitive component."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="sphere",
        diameter_nm=diameter_nm,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def ellipsoid_component(
    offset_nm: list[float] | None = None,
    *,
    axes_nm: list[float] | tuple[float, float, float],
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one ellipsoid primitive with body-frame axes [x, y, z]."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="ellipsoid",
        axes_nm=axes_nm,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def cylinder_component(
    offset_nm: list[float] | None = None,
    *,
    length_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one finite cylinder primitive along the body x axis."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="cylinder",
        length_nm=length_nm,
        diameter_nm=diameter_nm,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def spherocylinder_component(
    offset_nm: list[float] | None = None,
    *,
    length_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one capped-cylinder rod primitive along the body x axis."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="spherocylinder",
        length_nm=length_nm,
        diameter_nm=diameter_nm,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def voxel_volume_component(
    offset_nm: list[float] | None = None,
    *,
    voxel_geometry: dict[str, Any],
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one canonical voxel-volume geometry component."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="voxel_volume",
        voxel_geometry=voxel_geometry,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def mesh_component(
    offset_nm: list[float] | None = None,
    *,
    mesh_path: str,
    mesh_voxel_size_nm: float,
    mesh_scale_nm_per_unit: float = 1.0,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mesh producer component; normalization converts it to voxel_volume."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="mesh",
        mesh_path=mesh_path,
        mesh_voxel_size_nm=mesh_voxel_size_nm,
        mesh_scale_nm_per_unit=mesh_scale_nm_per_unit,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def voxel_file_component(
    offset_nm: list[float] | None = None,
    *,
    voxel_path: str,
    voxel_size_nm: float,
    voxel_array_key: str | None = None,
    voxel_occupancy_threshold: float | None = None,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a voxel-file producer; normalization converts it to voxel_volume."""
    return component(
        [0.0, 0.0, 0.0] if offset_nm is None else offset_nm,
        shape="voxel_file",
        voxel_path=voxel_path,
        voxel_size_nm=voxel_size_nm,
        voxel_array_key=voxel_array_key,
        voxel_occupancy_threshold=voxel_occupancy_threshold,
        material=material,
        refractive_index=refractive_index,
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=material_properties,
    )


def dimer(
    *,
    separation_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Return two equal spheres centered on the x axis."""
    half = 0.5 * float(separation_nm)
    return [
        component(
            [-half, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
        component(
            [half, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
    ]


def linear_trimer(
    *,
    separation_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Return three equal spheres in a straight x-axis chain."""
    sep = float(separation_nm)
    return [
        component(
            [-sep, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
        component(
            [0.0, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
        component(
            [sep, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
    ]


def bent_trimer(
    *,
    arm_separation_nm: float,
    bend_angle_deg: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Return a three-sphere bent chain with the bend centered on the origin."""
    theta = math.radians(float(bend_angle_deg))
    half = 0.5 * theta
    r = float(arm_separation_nm)
    return [
        component(
            [0.0, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
        component(
            [r * math.cos(half), r * math.sin(half), 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
        component(
            [r * math.cos(half), -r * math.sin(half), 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
            source_multiplier=source_multiplier,
        ),
    ]


def particle(
    *,
    name: str,
    components: list[dict[str, Any]],
    hydrodynamic_diameter_nm: float,
    initial_position_nm: list[float] | None = None,
    signal_multiplier: float = 1.0,
    source_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build a particle object from primitive components and motion size."""
    return {
        "name": str(name),
        "motion": {
            "hydrodynamic_diameter_nm": float(hydrodynamic_diameter_nm),
            "initial_position_nm": initial_position_nm,
        },
        "signal_multiplier": float(signal_multiplier),
        "source_multiplier": float(source_multiplier),
        "components": components,
    }
