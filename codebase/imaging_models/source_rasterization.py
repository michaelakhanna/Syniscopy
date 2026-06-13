"""Shared source-footprint rasterization helpers for material source maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class WeightedProjectedSourceIntegral:
    """Cell-averaged weighted source integral with an explicit basis contract.

    ``cell_integral_nm`` is the source-canvas quantity to multiply by a
    volume-density scale. It has length units because the source canvas stores
    a per-area density after axial integration. ``raw_cell_integral_nm`` is
    retained for diagnostics only; consumers should not use it as a physical
    source map unless they intentionally opt out of exact projected-source
    normalization.
    """

    cell_integral_nm: np.ndarray
    raw_cell_integral_nm: np.ndarray
    projected_chord_reference_nm: np.ndarray
    normalization_status: str
    source_basis: str


@dataclass(frozen=True)
class PrimitiveFootprintPatch:
    """Subpixel-integrated projected primitive footprint on a source canvas."""

    x0: int
    x1: int
    y0: int
    y1: int
    z_lower_samples_nm: np.ndarray
    z_upper_samples_nm: np.ndarray
    pitch_nm: float
    exact_volume_nm3: float
    full_primitive_inside_canvas: bool
    shape_name: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    def projected_chord_nm(self, *, normalize_total: bool = True) -> np.ndarray:
        chord_samples = np.maximum(
            np.asarray(self.z_upper_samples_nm, dtype=float)
            - np.asarray(self.z_lower_samples_nm, dtype=float),
            0.0,
        )
        chord_nm = np.mean(chord_samples, axis=0)
        if not normalize_total or not self.full_primitive_inside_canvas:
            return chord_nm
        exact_volume_nm3 = float(self.exact_volume_nm3)
        raw_volume_nm3 = float(np.sum(chord_nm) * self.pitch_nm ** 2)
        if exact_volume_nm3 > 0.0 and raw_volume_nm3 > 0.0 and np.isfinite(raw_volume_nm3):
            chord_nm = chord_nm * (exact_volume_nm3 / raw_volume_nm3)
        return chord_nm

    def average_over_samples(
        self,
        callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Average a z-interval-dependent quantity over source-cell area."""
        acc = None
        for z_lower_nm, z_upper_nm in zip(self.z_lower_samples_nm, self.z_upper_samples_nm):
            value = np.asarray(callback(z_lower_nm, z_upper_nm), dtype=float)
            if acc is None:
                acc = np.zeros_like(value, dtype=float)
            acc += value
        if acc is None:
            return np.zeros(self.shape, dtype=float)
        return acc / float(self.z_lower_samples_nm.shape[0])

    def weighted_projected_source_integral(
        self,
        callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        source_basis: str,
    ) -> WeightedProjectedSourceIntegral:
        raw = self.average_over_samples(callback)
        raw_projected_chord = self.projected_chord_nm(normalize_total=False)
        reference_projected_chord = self.projected_chord_nm(normalize_total=True)
        scale = np.zeros_like(reference_projected_chord, dtype=float)
        valid = raw_projected_chord > 0.0
        if np.any(valid):
            scale[valid] = reference_projected_chord[valid] / raw_projected_chord[valid]
        if raw.ndim == scale.ndim + 1:
            normalized = raw * scale[None, :, :]
        else:
            normalized = raw * scale
        status = (
            "exact_projected_primitive_reference"
            if self.full_primitive_inside_canvas
            else "clipped_projected_primitive_reference"
        )
        return WeightedProjectedSourceIntegral(
            cell_integral_nm=normalized,
            raw_cell_integral_nm=raw,
            projected_chord_reference_nm=reference_projected_chord,
            normalization_status=status,
            source_basis=str(source_basis),
        )


@dataclass(frozen=True)
class VoxelVolumeFootprintPatch:
    """Projected voxel-volume footprint implementing the source patch contract."""

    x0: int
    x1: int
    y0: int
    y1: int
    pitch_nm: float
    exact_volume_nm3: float
    full_primitive_inside_canvas: bool
    shape_name: str
    entry_y: np.ndarray
    entry_x: np.ndarray
    z_lower_nm: np.ndarray
    z_upper_nm: np.ndarray
    chord_nm: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    def projected_chord_nm(self, *, normalize_total: bool = True) -> np.ndarray:
        out = np.zeros(self.shape, dtype=float)
        np.add.at(out, (self.entry_y, self.entry_x), self.chord_nm)
        if not normalize_total or not self.full_primitive_inside_canvas:
            return out
        raw_volume_nm3 = float(np.sum(out) * self.pitch_nm ** 2)
        if self.exact_volume_nm3 > 0.0 and raw_volume_nm3 > 0.0 and np.isfinite(raw_volume_nm3):
            out = out * (float(self.exact_volume_nm3) / raw_volume_nm3)
        return out

    def average_over_samples(
        self,
        callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        sample_shape = self.shape
        acc = None
        for iy, ix, z0, z1, chord in zip(
            self.entry_y,
            self.entry_x,
            self.z_lower_nm,
            self.z_upper_nm,
            self.chord_nm,
        ):
            interval = max(float(z1) - float(z0), 1.0e-12)
            scale = float(chord) / interval
            z_lower = np.zeros(sample_shape, dtype=float)
            z_upper = np.zeros(sample_shape, dtype=float)
            z_lower[int(iy), int(ix)] = float(z0)
            z_upper[int(iy), int(ix)] = float(z1)
            value = np.asarray(callback(z_lower, z_upper), dtype=float)
            if acc is None:
                acc = np.zeros_like(value, dtype=float)
            if value.ndim == 2:
                acc[int(iy), int(ix)] += value[int(iy), int(ix)] * scale
            elif value.ndim == 3:
                acc[:, int(iy), int(ix)] += value[:, int(iy), int(ix)] * scale
            else:
                raise ValueError(
                    "Voxel volume source callback must return a 2D or 3D array; "
                    f"got shape {value.shape!r}."
                )
        if acc is None:
            return np.zeros(sample_shape, dtype=float)
        return acc

    def weighted_projected_source_integral(
        self,
        callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        source_basis: str,
    ) -> WeightedProjectedSourceIntegral:
        raw = self.average_over_samples(callback)
        raw_projected_chord = self.projected_chord_nm(normalize_total=False)
        reference_projected_chord = self.projected_chord_nm(normalize_total=True)
        scale = np.zeros_like(reference_projected_chord, dtype=float)
        valid = raw_projected_chord > 0.0
        if np.any(valid):
            scale[valid] = reference_projected_chord[valid] / raw_projected_chord[valid]
        normalized = raw * scale[None, :, :] if raw.ndim == scale.ndim + 1 else raw * scale
        status = (
            "exact_projected_voxel_volume_reference"
            if self.full_primitive_inside_canvas
            else "clipped_projected_voxel_volume_reference"
        )
        return WeightedProjectedSourceIntegral(
            cell_integral_nm=normalized,
            raw_cell_integral_nm=raw,
            projected_chord_reference_nm=reference_projected_chord,
            normalization_status=status,
            source_basis=str(source_basis),
        )


@dataclass(frozen=True)
class SphereFootprintPatch:
    """Subpixel-integrated projected sphere footprint on a source canvas."""

    x0: int
    x1: int
    y0: int
    y1: int
    half_chord_samples_nm: np.ndarray
    pitch_nm: float
    radius_nm: float
    full_sphere_inside_canvas: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    def projected_chord_nm(self, *, normalize_total: bool = True) -> np.ndarray:
        """Return the cell-average projected chord length in nanometers.

        The source canvas stores per-area source density.  Each canvas cell
        should therefore receive the lateral cell average of the sphere chord,
        not the chord sampled only at the cell center.  For fully in-frame
        spheres, rescale the quadrature result to the exact sphere volume so
        total source is independent of subpixel phase.
        """
        chord_nm = 2.0 * np.mean(self.half_chord_samples_nm, axis=0)
        if not normalize_total or not self.full_sphere_inside_canvas:
            return chord_nm
        exact_volume_nm3 = (4.0 / 3.0) * np.pi * self.radius_nm ** 3
        raw_volume_nm3 = float(np.sum(chord_nm) * self.pitch_nm ** 2)
        if raw_volume_nm3 > 0.0 and np.isfinite(raw_volume_nm3):
            chord_nm = chord_nm * (exact_volume_nm3 / raw_volume_nm3)
        return chord_nm

    def average_over_samples(
        self,
        callback: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Average a half-chord-dependent quantity over source-cell area."""
        acc = None
        for sample_half_chord_nm in self.half_chord_samples_nm:
            value = np.asarray(callback(sample_half_chord_nm), dtype=float)
            if acc is None:
                acc = np.zeros_like(value, dtype=float)
            acc += value
        if acc is None:
            return np.zeros(self.shape, dtype=float)
        return acc / float(self.half_chord_samples_nm.shape[0])

    def weighted_projected_source_integral(
        self,
        callback: Callable[[np.ndarray], np.ndarray],
        *,
        source_basis: str,
    ) -> WeightedProjectedSourceIntegral:
        """Return a normalized axial-weighted projected source integral.

        The TIRF and volume-source paths need more than a raw subpixel average:
        a constant axial weight must reduce exactly to ``projected_chord_nm()``
        for a fully in-frame sphere, otherwise source mass, photon counts, and
        Fisher/CRLB scale depend on pixel phase. The normalization below keeps
        the sampled axial-weight ratio in each cell but applies it to the same
        exact projected-chord reference used by the unweighted fluorescence path.
        """
        raw = self.average_over_samples(callback)
        raw_projected_chord = self.projected_chord_nm(normalize_total=False)
        reference_projected_chord = self.projected_chord_nm(normalize_total=True)
        normalized = np.zeros_like(raw, dtype=float)
        valid = raw_projected_chord > 0.0
        if np.any(valid):
            normalized[valid] = raw[valid] * (
                reference_projected_chord[valid] / raw_projected_chord[valid]
            )
        status = (
            "exact_projected_chord_reference"
            if self.full_sphere_inside_canvas
            else "clipped_projected_chord_reference"
        )
        return WeightedProjectedSourceIntegral(
            cell_integral_nm=normalized,
            raw_cell_integral_nm=raw,
            projected_chord_reference_nm=reference_projected_chord,
            normalization_status=status,
            source_basis=str(source_basis),
        )


def _adaptive_subpixel_samples(radius_px: float) -> int:
    if radius_px <= 0.0 or not np.isfinite(radius_px):
        return 8
    return int(max(8, min(128, np.ceil(1.5 / max(radius_px, 1.0e-12)))))


def sphere_footprint_patch(
    *,
    center_x_canvas: float,
    center_y_canvas: float,
    diameter_nm: float,
    pixel_size_nm: float,
    os_factor: int,
    canvas_shape: tuple[int, int],
    subpixel_samples: int | None = None,
) -> SphereFootprintPatch | None:
    """Return a deterministic cell-integrated sphere source footprint.

    Canvas coordinates are in model-canvas pixels whose cell centers have
    integer coordinates.  The returned half-chord samples are evaluated within
    each touched cell, preserving subpixel particles instead of dropping them
    when no cell center lies inside the projected sphere.
    """
    h, w = (int(canvas_shape[0]), int(canvas_shape[1]))
    if h <= 0 or w <= 0:
        return None
    pitch_nm = float(pixel_size_nm) / float(os_factor)
    radius_nm = 0.5 * float(diameter_nm)
    if not np.isfinite(pitch_nm) or pitch_nm <= 0.0:
        raise ValueError(f"Source canvas pitch must be positive; got {pitch_nm!r} nm.")
    if not np.isfinite(radius_nm) or radius_nm <= 0.0:
        return None
    radius_px = radius_nm / pitch_nm
    center_x = float(center_x_canvas)
    center_y = float(center_y_canvas)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError(
            "Source footprint center must be finite; got "
            f"({center_x_canvas!r}, {center_y_canvas!r})."
        )

    support_px = radius_px + np.sqrt(0.5)
    x0 = max(0, int(np.floor(center_x - support_px)))
    x1 = min(w, int(np.ceil(center_x + support_px)) + 1)
    y0 = max(0, int(np.floor(center_y - support_px)))
    y1 = min(h, int(np.ceil(center_y + support_px)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    q = _adaptive_subpixel_samples(radius_px) if subpixel_samples is None else int(subpixel_samples)
    if q <= 0:
        raise ValueError(f"subpixel_samples must be positive; got {subpixel_samples!r}.")
    offsets = (np.arange(q, dtype=float) + 0.5) / float(q) - 0.5
    sample_y, sample_x = np.meshgrid(offsets, offsets, indexing="ij")
    sample_offsets = np.column_stack([sample_y.reshape(-1), sample_x.reshape(-1)])

    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    xx = xx + float(x0)
    yy = yy + float(y0)
    half_samples = np.empty((sample_offsets.shape[0], y1 - y0, x1 - x0), dtype=float)
    for sample_index, (dy_sample, dx_sample) in enumerate(sample_offsets):
        dx_px = xx + dx_sample - center_x
        dy_px = yy + dy_sample - center_y
        lateral_nm = np.sqrt(dx_px * dx_px + dy_px * dy_px) * pitch_nm
        half_samples[sample_index] = np.sqrt(
            np.maximum(radius_nm * radius_nm - lateral_nm * lateral_nm, 0.0)
        )

    full_inside = (
        center_x - radius_px >= -0.5
        and center_x + radius_px <= float(w) - 0.5
        and center_y - radius_px >= -0.5
        and center_y + radius_px <= float(h) - 0.5
    )
    return SphereFootprintPatch(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        half_chord_samples_nm=half_samples,
        pitch_nm=pitch_nm,
        radius_nm=radius_nm,
        full_sphere_inside_canvas=bool(full_inside),
    )


def _quadratic_interval(
    A: float,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disc = B * B - 4.0 * float(A) * C
    valid = disc >= 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = 2.0 * float(A)
    lo = (-B - sqrt_disc) / denom
    hi = (-B + sqrt_disc) / denom
    return lo, hi, valid


def _slab_interval(
    q: np.ndarray,
    k: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(float(k)) <= 1e-14:
        valid = (q >= float(lower)) & (q <= float(upper))
        lo = np.full_like(q, -np.inf, dtype=float)
        hi = np.full_like(q, np.inf, dtype=float)
        return lo, hi, valid
    t0 = (float(lower) - q) / float(k)
    t1 = (float(upper) - q) / float(k)
    return np.minimum(t0, t1), np.maximum(t0, t1), np.ones_like(q, dtype=bool)


def _radial_cylinder_interval(
    qy: np.ndarray,
    qz: np.ndarray,
    ky: float,
    kz: float,
    radius_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = float(ky) ** 2 + float(kz) ** 2
    C = qy * qy + qz * qz - float(radius_nm) ** 2
    if A <= 1e-14:
        valid = C <= 0.0
        lo = np.full_like(qy, -np.inf, dtype=float)
        hi = np.full_like(qy, np.inf, dtype=float)
        return lo, hi, valid
    B = 2.0 * (qy * float(ky) + qz * float(kz))
    return _quadratic_interval(A, B, C)


def _merge_interval(
    union_lo: np.ndarray,
    union_hi: np.ndarray,
    union_valid: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid & (hi >= lo)
    out_lo = union_lo.copy()
    out_hi = union_hi.copy()
    out_valid = union_valid | valid
    out_lo[valid] = np.minimum(out_lo[valid], lo[valid])
    out_hi[valid] = np.maximum(out_hi[valid], hi[valid])
    return out_lo, out_hi, out_valid


def _primitive_z_intervals_nm(
    component_geometry,
    *,
    dx_nm: np.ndarray,
    dy_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    shape = str(component_geometry.shape).strip().lower()
    if orientation_matrix is None:
        R = np.eye(3, dtype=float)
    else:
        R = np.asarray(orientation_matrix, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("component orientation_matrix must be 3x3.")
    qx = R[0, 0] * dx_nm + R[1, 0] * dy_nm
    qy = R[0, 1] * dx_nm + R[1, 1] * dy_nm
    qz = R[0, 2] * dx_nm + R[1, 2] * dy_nm
    kx, ky, kz = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])

    if shape == "sphere":
        radius_nm = 0.5 * float(component_geometry.diameter_nm)
        C = dx_nm * dx_nm + dy_nm * dy_nm - radius_nm * radius_nm
        lo, hi, valid = _quadratic_interval(1.0, np.zeros_like(C), C)
    elif shape == "ellipsoid":
        axes = np.asarray(component_geometry.semi_axes_nm, dtype=float)
        A = (kx / axes[0]) ** 2 + (ky / axes[1]) ** 2 + (kz / axes[2]) ** 2
        B = 2.0 * (
            qx * kx / (axes[0] ** 2)
            + qy * ky / (axes[1] ** 2)
            + qz * kz / (axes[2] ** 2)
        )
        C = (qx / axes[0]) ** 2 + (qy / axes[1]) ** 2 + (qz / axes[2]) ** 2 - 1.0
        lo, hi, valid = _quadratic_interval(float(A), B, C)
    elif shape == "cylinder":
        half_length = 0.5 * float(component_geometry.length_nm)
        radius_nm = 0.5 * float(component_geometry.diameter_nm)
        x_lo, x_hi, x_valid = _slab_interval(qx, kx, -half_length, half_length)
        r_lo, r_hi, r_valid = _radial_cylinder_interval(qy, qz, ky, kz, radius_nm)
        lo = np.maximum(x_lo, r_lo)
        hi = np.minimum(x_hi, r_hi)
        valid = x_valid & r_valid & (hi >= lo)
    elif shape == "spherocylinder":
        radius_nm = 0.5 * float(component_geometry.diameter_nm)
        segment_half_length = 0.5 * max(float(component_geometry.length_nm) - float(component_geometry.diameter_nm), 0.0)
        union_lo = np.full_like(dx_nm, np.inf, dtype=float)
        union_hi = np.full_like(dx_nm, -np.inf, dtype=float)
        union_valid = np.zeros_like(dx_nm, dtype=bool)
        if segment_half_length > 0.0:
            x_lo, x_hi, x_valid = _slab_interval(
                qx,
                kx,
                -segment_half_length,
                segment_half_length,
            )
            r_lo, r_hi, r_valid = _radial_cylinder_interval(qy, qz, ky, kz, radius_nm)
            lo_body = np.maximum(x_lo, r_lo)
            hi_body = np.minimum(x_hi, r_hi)
            union_lo, union_hi, union_valid = _merge_interval(
                union_lo,
                union_hi,
                union_valid,
                lo_body,
                hi_body,
                x_valid & r_valid,
            )
            cap_centers = (-segment_half_length, segment_half_length)
        else:
            cap_centers = (0.0,)
        for cap_center in cap_centers:
            C = (
                (qx - float(cap_center)) ** 2
                + qy * qy
                + qz * qz
                - radius_nm * radius_nm
            )
            B = 2.0 * ((qx - float(cap_center)) * kx + qy * ky + qz * kz)
            lo_cap, hi_cap, cap_valid = _quadratic_interval(1.0, B, C)
            union_lo, union_hi, union_valid = _merge_interval(
                union_lo,
                union_hi,
                union_valid,
                lo_cap,
                hi_cap,
                cap_valid,
            )
        lo, hi, valid = union_lo, union_hi, union_valid
    else:
        raise ValueError(f"Unsupported primitive source footprint shape {shape!r}.")

    lower = np.where(valid, lo, 0.0)
    upper = np.where(valid, hi, 0.0)
    return lower, upper


def _voxel_volume_footprint_patch(
    *,
    component_geometry,
    center_x_canvas: float,
    center_y_canvas: float,
    pixel_size_nm: float,
    os_factor: int,
    canvas_shape: tuple[int, int],
    orientation_matrix: np.ndarray | None,
) -> VoxelVolumeFootprintPatch | None:
    volume = getattr(component_geometry, "voxel_geometry", None)
    if volume is None:
        raise ValueError("voxel_volume component requires voxel_geometry.")
    h, w = (int(canvas_shape[0]), int(canvas_shape[1]))
    if h <= 0 or w <= 0:
        return None
    pitch_nm = float(pixel_size_nm) / float(os_factor)
    if not np.isfinite(pitch_nm) or pitch_nm <= 0.0:
        raise ValueError(f"Source canvas pitch must be positive; got {pitch_nm!r} nm.")
    centers_body, weights = volume.occupied_voxels()
    R = np.eye(3, dtype=float) if orientation_matrix is None else np.asarray(orientation_matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("component orientation_matrix must be 3x3.")
    centers_world = (R @ centers_body.T).T
    x_canvas = float(center_x_canvas) + centers_world[:, 0] / pitch_nm
    y_canvas = float(center_y_canvas) + centers_world[:, 1] / pitch_nm
    ix_global = np.floor(x_canvas + 0.5).astype(int)
    iy_global = np.floor(y_canvas + 0.5).astype(int)
    inside = (
        (ix_global >= 0)
        & (ix_global < w)
        & (iy_global >= 0)
        & (iy_global < h)
        & np.isfinite(centers_world[:, 2])
    )
    if not np.any(inside):
        return None
    ix_inside = ix_global[inside]
    iy_inside = iy_global[inside]
    x0 = int(np.min(ix_inside))
    x1 = int(np.max(ix_inside)) + 1
    y0 = int(np.min(iy_inside))
    y1 = int(np.max(iy_inside)) + 1
    voxel_size_nm = float(volume.voxel_size_nm)
    voxel_half_z = 0.5 * voxel_size_nm * float(np.sum(np.abs(R[2, :])))
    z_center = centers_world[inside, 2]
    chord_nm = (
        weights[inside]
        * float(volume.voxel_volume_nm3)
        / (pitch_nm * pitch_nm)
    )
    full_inside = bool(np.all(inside))
    return VoxelVolumeFootprintPatch(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        pitch_nm=pitch_nm,
        exact_volume_nm3=float(volume.volume_nm3),
        full_primitive_inside_canvas=full_inside,
        shape_name=str(component_geometry.shape),
        entry_y=(iy_inside - y0).astype(int, copy=False),
        entry_x=(ix_inside - x0).astype(int, copy=False),
        z_lower_nm=(z_center - voxel_half_z).astype(float, copy=False),
        z_upper_nm=(z_center + voxel_half_z).astype(float, copy=False),
        chord_nm=chord_nm.astype(float, copy=False),
    )


def primitive_footprint_patch(
    *,
    component_geometry,
    center_x_canvas: float,
    center_y_canvas: float,
    pixel_size_nm: float,
    os_factor: int,
    canvas_shape: tuple[int, int],
    orientation_matrix: np.ndarray | None = None,
    subpixel_samples: int | None = None,
) -> PrimitiveFootprintPatch | None:
    """Return a deterministic cell-integrated primitive source footprint."""
    if str(component_geometry.shape).strip().lower() == "voxel_volume":
        return _voxel_volume_footprint_patch(
            component_geometry=component_geometry,
            center_x_canvas=float(center_x_canvas),
            center_y_canvas=float(center_y_canvas),
            pixel_size_nm=float(pixel_size_nm),
            os_factor=int(os_factor),
            canvas_shape=canvas_shape,
            orientation_matrix=orientation_matrix,
        )
    h, w = (int(canvas_shape[0]), int(canvas_shape[1]))
    if h <= 0 or w <= 0:
        return None
    pitch_nm = float(pixel_size_nm) / float(os_factor)
    if not np.isfinite(pitch_nm) or pitch_nm <= 0.0:
        raise ValueError(f"Source canvas pitch must be positive; got {pitch_nm!r} nm.")
    support_radius_nm = float(component_geometry.bounding_radius_nm)
    if not np.isfinite(support_radius_nm) or support_radius_nm <= 0.0:
        return None
    support_px = support_radius_nm / pitch_nm + np.sqrt(0.5)
    center_x = float(center_x_canvas)
    center_y = float(center_y_canvas)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError(
            "Source footprint center must be finite; got "
            f"({center_x_canvas!r}, {center_y_canvas!r})."
        )
    x0 = max(0, int(np.floor(center_x - support_px)))
    x1 = min(w, int(np.ceil(center_x + support_px)) + 1)
    y0 = max(0, int(np.floor(center_y - support_px)))
    y1 = min(h, int(np.ceil(center_y + support_px)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    q = _adaptive_subpixel_samples(support_radius_nm / pitch_nm) if subpixel_samples is None else int(subpixel_samples)
    if q <= 0:
        raise ValueError(f"subpixel_samples must be positive; got {subpixel_samples!r}.")
    offsets = (np.arange(q, dtype=float) + 0.5) / float(q) - 0.5
    sample_y, sample_x = np.meshgrid(offsets, offsets, indexing="ij")
    sample_offsets = np.column_stack([sample_y.reshape(-1), sample_x.reshape(-1)])
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    xx = xx + float(x0)
    yy = yy + float(y0)
    z_lower = np.empty((sample_offsets.shape[0], y1 - y0, x1 - x0), dtype=float)
    z_upper = np.empty_like(z_lower)
    for sample_index, (dy_sample, dx_sample) in enumerate(sample_offsets):
        dx_nm = (xx + dx_sample - center_x) * pitch_nm
        dy_nm = (yy + dy_sample - center_y) * pitch_nm
        lo, hi = _primitive_z_intervals_nm(
            component_geometry,
            dx_nm=dx_nm,
            dy_nm=dy_nm,
            orientation_matrix=orientation_matrix,
        )
        z_lower[sample_index] = lo
        z_upper[sample_index] = hi

    radius_px = support_radius_nm / pitch_nm
    full_inside = (
        center_x - radius_px >= -0.5
        and center_x + radius_px <= float(w) - 0.5
        and center_y - radius_px >= -0.5
        and center_y + radius_px <= float(h) - 0.5
    )
    return PrimitiveFootprintPatch(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        z_lower_samples_nm=z_lower,
        z_upper_samples_nm=z_upper,
        pitch_nm=pitch_nm,
        exact_volume_nm3=float(component_geometry.volume_nm3),
        full_primitive_inside_canvas=bool(full_inside),
        shape_name=str(component_geometry.shape),
    )


def normalize_sliced_source_to_projected_chord(
    sliced_source_nm: np.ndarray,
    projected_chord_nm: np.ndarray,
) -> np.ndarray:
    """Conserve each lateral cell's projected chord across z slices."""
    sliced = np.asarray(sliced_source_nm, dtype=float)
    projected = np.asarray(projected_chord_nm, dtype=float)
    total = np.sum(sliced, axis=0)
    out = np.zeros_like(sliced, dtype=float)
    valid = total > 0.0
    if np.any(valid):
        out[:, valid] = sliced[:, valid] * (projected[valid] / total[valid])[None, :]
    return out
