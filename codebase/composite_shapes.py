"""
Factories for particle-object component lists.

Each function returns a list suitable for
PARAMS["particles"][i]["components"]. Offsets are in nanometres in the
particle body frame. The renderer rotates these offsets by the particle's
per-frame orientation matrix before placing component PSFs.
"""

from __future__ import annotations

import math
from typing import Any


def component(
    offset_nm: list[float],
    *,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
    material_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one spherical render component with a body-frame offset."""
    return {
        "shape": "sphere",
        "offset_nm": [float(offset_nm[0]), float(offset_nm[1]), float(offset_nm[2])],
        "diameter_nm": float(diameter_nm),
        "material": material,
        "refractive_index": refractive_index,
        "signal_multiplier": float(signal_multiplier),
        "material_properties": material_properties,
    }


def dimer(
    *,
    separation_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
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
        ),
        component(
            [half, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        ),
    ]


def linear_trimer(
    *,
    separation_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
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
        ),
        component(
            [0.0, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        ),
        component(
            [sep, 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        ),
    ]


def rod_stack(
    *,
    count: int,
    separation_nm: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Return ``count`` equal spheres in an x-axis stack."""
    count = int(count)
    if count <= 0:
        raise ValueError("rod_stack count must be positive.")
    center = 0.5 * (count - 1)
    return [
        component(
            [(i - center) * float(separation_nm), 0.0, 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        )
        for i in range(count)
    ]


def bent_trimer(
    *,
    arm_separation_nm: float,
    bend_angle_deg: float,
    diameter_nm: float,
    material: str | None = None,
    refractive_index: complex | dict[str, float] | None = None,
    signal_multiplier: float = 1.0,
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
        ),
        component(
            [r * math.cos(half), r * math.sin(half), 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        ),
        component(
            [r * math.cos(half), -r * math.sin(half), 0.0],
            diameter_nm=diameter_nm,
            material=material,
            refractive_index=refractive_index,
            signal_multiplier=signal_multiplier,
        ),
    ]


def particle(
    *,
    name: str,
    components: list[dict[str, Any]],
    hydrodynamic_diameter_nm: float,
    initial_position_nm: list[float] | None = None,
    signal_multiplier: float = 1.0,
    symmetry_class: str | None = None,
    continuous_rotational_symmetry_dim: int | None = None,
    singular_rotation_axes_body: list[str] | None = None,
) -> dict[str, Any]:
    """Build a particle object from render components and motion size."""
    out = {
        "name": str(name),
        "motion": {
            "hydrodynamic_diameter_nm": float(hydrodynamic_diameter_nm),
            "initial_position_nm": initial_position_nm,
        },
        "signal_multiplier": float(signal_multiplier),
        "components": components,
    }
    if symmetry_class is not None:
        out["symmetry_class"] = str(symmetry_class)
    if continuous_rotational_symmetry_dim is not None:
        out["continuous_rotational_symmetry_dim"] = int(continuous_rotational_symmetry_dim)
    if singular_rotation_axes_body is not None:
        out["singular_rotation_axes_body"] = [str(axis) for axis in singular_rotation_axes_body]
    return out
