"""Per-particle render state and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_model import ParticleInstance, ParticleType


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
            self.source_canvas /= float(num_subsamples)

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
        if self.source_canvas.ndim == 3:
            return self.source_canvas[
                :, crop_start:crop_end, crop_start:crop_end
            ]
        return self.source_canvas[crop_start:crop_end, crop_start:crop_end]


def _accumulate_projected_geometry_disk(
    canvas: np.ndarray,
    *,
    center_x_canvas: float,
    center_y_canvas: float,
    diameter_nm: float,
    pixel_size_nm: float,
    os_factor: int,
) -> None:
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
) -> list[tuple[np.ndarray, object, float, float, float, object]]:
    """
    Compute the list of sub-particle render instructions for a given particle
    instance at a given (possibly interpolated) position and orientation.
    """
    ptype: ParticleType = instance.particle_type

    if not ptype.is_composite or not ptype.sub_particles:
        return [
            (
                np.asarray(base_position_nm, dtype=float),
                ptype.ipsf_interpolator,
                float(instance.component_signal_multiplier),
                float(instance.component_source_multiplier),
                float(ptype.diameter_nm),
                instance.material_properties,
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

    sub_infos: list[tuple[np.ndarray, object, float, float, float, object]] = []
    for sub in ptype.sub_particles:
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
            (
                sub_pos_world,
                sub.ipsf_interpolator,
                float(sub.signal_multiplier),
                float(sub.source_multiplier),
                float(sub.diameter_nm),
                sub.material_properties if sub.material_properties is not None else instance.material_properties,
            )
        )

    return sub_infos
