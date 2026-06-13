"""Per-particle render state and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imaging_models.sem_source import (
    source_like_crop,
    source_like_normalize_exposure,
)
from imaging_models.source_rasterization import primitive_footprint_patch
from particle_model import ParticleInstance, ParticleType
from particle_specs import ParticleComponentSpec


@dataclass
class _ParticleFrameRenderState:
    """
    Render-time products for one particle in one output frame.

    Static identity lives in ParticleSpec / ParticleInstance. This object owns
    the derived frame-local quantities: the complex scattered-field canvas,
    optional material/source canvas, and the exposure-averaged rendered
    position used by masks and supervision.
    """

    field_canvas: np.ndarray
    source_canvas: np.ndarray | None
    geometry_canvas: np.ndarray
    rendered_position_sum_nm: np.ndarray
    particle_index: int
    rendered_position_count: int = 0

    def add_rendered_position(self, position_nm: np.ndarray) -> None:
        self.rendered_position_sum_nm += np.asarray(position_nm, dtype=float)
        self.rendered_position_count += 1

    def normalize_exposure(self, num_subsamples: int) -> None:
        self.field_canvas /= float(num_subsamples)
        if self.source_canvas is not None:
            source_like_normalize_exposure(self.source_canvas, num_subsamples)

    def rendered_position_nm(self, fallback_position_nm: np.ndarray) -> np.ndarray:
        if self.rendered_position_count <= 0:
            return np.asarray(fallback_position_nm, dtype=float)
        return self.rendered_position_sum_nm / float(self.rendered_position_count)

    def field_fov(self, crop_start: int, crop_end: int) -> np.ndarray:
        if self.field_canvas.ndim == 3:
            return self.field_canvas[
                :, crop_start:crop_end, crop_start:crop_end
            ]
        return self.field_canvas[crop_start:crop_end, crop_start:crop_end]

    def source_fov(self, crop_start: int, crop_end: int) -> np.ndarray | None:
        if self.source_canvas is None:
            return None
        return source_like_crop(self.source_canvas, crop_start, crop_end)


@dataclass(frozen=True)
class _ComponentRenderInfo:
    world_position_nm: np.ndarray
    ipsf_interpolator: object
    signal_multiplier: float
    source_multiplier: float
    diameter_nm: float
    refractive_index: complex
    material_properties: object
    component_geometry: ParticleComponentSpec
    orientation_matrix: np.ndarray | None


def _accumulate_projected_geometry_disk(
    canvas: np.ndarray,
    *,
    center_x_canvas: float,
    center_y_canvas: float,
    diameter_nm: float,
    pixel_size_nm: float,
    os_factor: int,
    component_geometry: ParticleComponentSpec | None = None,
    orientation_matrix: np.ndarray | None = None,
) -> None:
    if component_geometry is not None:
        patch = primitive_footprint_patch(
            component_geometry=component_geometry,
            center_x_canvas=float(center_x_canvas),
            center_y_canvas=float(center_y_canvas),
            pixel_size_nm=float(pixel_size_nm),
            os_factor=int(os_factor),
            canvas_shape=canvas.shape,
            orientation_matrix=orientation_matrix,
        )
        if patch is None:
            return
        occupied = patch.projected_chord_nm(normalize_total=False) > 0.0
        canvas[patch.y0:patch.y1, patch.x0:patch.x1] = np.maximum(
            canvas[patch.y0:patch.y1, patch.x0:patch.x1],
            occupied.astype(canvas.dtype, copy=False),
        )
        return

    radius_px = 0.5 * float(diameter_nm) / (float(pixel_size_nm) / float(os_factor))
    if not np.isfinite(radius_px) or radius_px <= 0.0:
        return
    H, W = canvas.shape
    x0 = max(0, int(np.floor(float(center_x_canvas) - radius_px - 1.0)))
    x1 = min(W, int(np.ceil(float(center_x_canvas) + radius_px + 1.0)) + 1)
    y0 = max(0, int(np.floor(float(center_y_canvas) - radius_px - 1.0)))
    y1 = min(H, int(np.ceil(float(center_y_canvas) + radius_px + 1.0)) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    dx = xx + x0 - float(center_x_canvas)
    dy = yy + y0 - float(center_y_canvas)
    disk = (dx * dx + dy * dy) <= radius_px * radius_px
    canvas[y0:y1, x0:x1] = np.maximum(
        canvas[y0:y1, x0:x1],
        disk.astype(canvas.dtype, copy=False),
    )


def _iter_subparticle_render_info(
    instance: ParticleInstance,
    base_position_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
) -> list[_ComponentRenderInfo]:
    """
    Compute the list of sub-particle render instructions for a given particle
    instance at a given (possibly interpolated) position and orientation.
    """
    ptype: ParticleType = instance.particle_type

    if not ptype.is_composite or not ptype.sub_particles:
        if instance.component_geometry is None:
            raise ValueError("ParticleInstance.component_geometry is required for rendering.")
        return [
            _ComponentRenderInfo(
                world_position_nm=np.asarray(base_position_nm, dtype=float),
                ipsf_interpolator=ptype.ipsf_interpolator,
                signal_multiplier=float(instance.component_signal_multiplier),
                source_multiplier=float(instance.component_source_multiplier),
                diameter_nm=float(ptype.diameter_nm),
                refractive_index=complex(ptype.refractive_index),
                material_properties=instance.material_properties,
                component_geometry=instance.component_geometry,
                orientation_matrix=orientation_matrix,
            )
        ]

    base_world_pos = np.asarray(base_position_nm, dtype=float)
    if base_world_pos.shape != (3,):
        raise ValueError(
            "base_position_nm must be a length-3 vector [x, y, z] in nm."
        )

    R = None
    if orientation_matrix is not None:
        R = np.asarray(orientation_matrix, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(
                "orientation_matrix must be a 3x3 rotation matrix when provided."
            )

    sub_infos: list[_ComponentRenderInfo] = []
    for sub in ptype.sub_particles:
        if sub.component_geometry is None:
            raise ValueError("SubParticle.component_geometry is required for rendering.")
        offset = np.asarray(sub.offset_nm, dtype=float)
        if offset.shape != (3,):
            raise ValueError(
                "SubParticle.offset_nm must be a length-3 vector [dx, dy, dz] in nm."
            )

        if R is not None:
            rotated_offset = R @ offset
        else:
            rotated_offset = offset

        sub_pos_world = base_world_pos + rotated_offset
        sub_infos.append(
            _ComponentRenderInfo(
                world_position_nm=sub_pos_world,
                ipsf_interpolator=sub.ipsf_interpolator,
                signal_multiplier=float(sub.signal_multiplier),
                source_multiplier=float(sub.source_multiplier),
                diameter_nm=float(sub.diameter_nm),
                refractive_index=complex(sub.refractive_index),
                material_properties=(
                    sub.material_properties
                    if sub.material_properties is not None
                    else instance.material_properties
                ),
                component_geometry=sub.component_geometry,
                orientation_matrix=R,
            )
        )

    return sub_infos
