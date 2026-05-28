"""
Frame renderer for Syniscopy videos, masks, and supervision sidecars.

This module turns particle instances, trajectories, optical models, substrate
backgrounds, and camera noise into signal/reference frame sequences. When mask
generation is enabled it also writes the canonical supervision mask sidecars.
"""

import json
import logging
import numbers
import os
from dataclasses import dataclass, field

import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import j1

from camera_noise import contrast_noise_variance_counts
from mask_generation import save_mask
from mask_generation import generate_central_lobe_mask
from supervision_policy import (
    SupervisionAudit,
    SupervisionPolicy,
    build_policy_annotation_schema,
)
from substrate_pattern import (
    compute_contrast_scale_for_frame,
    generate_empirical_background_field,
    generate_reference_and_background_maps,
    resize_empirical_background_field,
)
from imaging_model import get_imaging_model
from substrate import sample_environment_from_params
from particle_model import ParticleInstance, ParticleType
from particle_specs import particle_count
from trajectory import resolve_num_frames


logger = logging.getLogger(__name__)

_AIRY_SUPPORT_RHO_MIN = 1e-4
_AIRY_SUPPORT_RHO_MAX = 200.0
_AIRY_SUPPORT_NUM_SAMPLES = 80000


@dataclass
class RenderedFrameSet:
    """Frame sequences produced by the renderer."""

    signal_frames: list[np.ndarray]
    reference_frames: list[np.ndarray]
    ideal_signal_frames: list[np.ndarray]
    ideal_reference_frames: list[np.ndarray]
    mask_arrays: list[dict] = field(default_factory=list)
    supervision_records: list[dict] = field(default_factory=list)
    supervision_audit_summary: dict | None = None


def _strict_json_safe(value):
    """Convert renderer metadata to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _strict_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _strict_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, complex):
        return {
            "real": _strict_json_safe(float(value.real)),
            "imag": _strict_json_safe(float(value.imag)),
        }
    return value


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
        return self.field_canvas[crop_start:crop_end, crop_start:crop_end]

    def source_fov(self, crop_start: int, crop_end: int) -> np.ndarray | None:
        if self.source_canvas is None:
            return None
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


def _airy_support_radius_pixels(
    params: dict,
    *,
    threshold_key: str | None,
    default_threshold: float,
    max_radius_fraction_of_fov: float,
) -> int:
    """
    Estimate a circular-pupil response support radius in oversampled pixels.

    This is a numerical guard-band calculation. It does not change the optical
    model; it determines how much scene context to include before cropping so
    finite FFTs do not wrap boundary content into the detector FOV.
    """
    img_size = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = int(params["psf_oversampling_factor"])
    NA = float(params["numerical_aperture"])
    n_medium = float(params["refractive_index_medium"])
    wavelength_nm = float(params["wavelength_nm"])

    if img_size <= 0 or pixel_size_nm <= 0 or os_factor <= 0:
        raise ValueError(
            "PARAMS['image_size_pixels'], PARAMS['pixel_size_nm'], and "
            "PARAMS['psf_oversampling_factor'] must all be positive."
        )

    if NA <= 0.0 or wavelength_nm <= 0.0 or n_medium <= 0.0:
        return 0

    threshold = (
        float(default_threshold)
        if threshold_key is None
        else float(params.get(threshold_key, default_threshold))
    )
    if not (0.0 < threshold < 1.0):
        if threshold_key is None:
            raise ValueError("Airy support threshold must be in the open interval (0, 1).")
        raise ValueError(f"PARAMS['{threshold_key}'] must be in the open interval (0, 1).")

    wavelength_medium_nm = wavelength_nm / n_medium

    # Scan Airy intensity over a large normalized-radius interval; rho=200
    # covers far sidelobes for the supported threshold range, and 80k samples
    # keeps the guard-band estimate stable without affecting frame rendering.
    rho = np.linspace(
        _AIRY_SUPPORT_RHO_MIN,
        _AIRY_SUPPORT_RHO_MAX,
        _AIRY_SUPPORT_NUM_SAMPLES,
    )
    x = np.pi * rho

    I_rel = (2.0 * j1(x) / x) ** 2

    indices_above = np.where(I_rel >= threshold)[0]
    if indices_above.size == 0:
        rho_crit = 0.0
    else:
        rho_crit = float(rho[indices_above[-1]])

    radius_nm = rho_crit * wavelength_medium_nm / NA

    psf_size_nm = img_size * pixel_size_nm
    max_radius_nm = float(max_radius_fraction_of_fov) * psf_size_nm
    if max_radius_nm > 0.0:
        radius_nm = min(radius_nm, max_radius_nm)

    radius_pixels_oversampled = radius_nm / pixel_size_nm * os_factor

    padding_pixels = int(np.ceil(radius_pixels_oversampled)) + 1
    return max(padding_pixels, 0)


def estimate_psf_padding_radius_pixels(params):
    """
    Estimate the extra padding radius (in oversampled pixels) required around
    the simulated field of view so that PSF contributions from particles
    located just outside the nominal FOV can be represented on the padded
    canvas without significant truncation in the central region that is
    ultimately written to the video.

    The estimate uses an Airy-pattern approximation for the PSF of a circular
    aperture. We compute the normalized intensity

        I_rel(r) = I(r) / I(0) ~= [2 J1(pi * rho) / (pi * rho)]^2,

    where rho = (NA * r) / lambda_medium is a dimensionless radial coordinate.

    We then find the largest radius r such that I_rel(r) is still above a
    user-controllable fraction:

        I_rel(r) >= psf_intensity_fraction_threshold,

    and treat everything beyond that radius as negligible. The corresponding
    physical radius is converted into oversampled pixels using the current
    imaging geometry.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS).

    Returns:
        int: Padding radius in oversampled pixels (>= 0).
    """
    return _airy_support_radius_pixels(
        params,
        threshold_key="psf_intensity_fraction_threshold",
        default_threshold=1e-4,
        max_radius_fraction_of_fov=0.5,
    )


def estimate_optical_filter_guard_radius_pixels(params):
    """
    Estimate the guard band for Fourier-domain optical filtering before crop.

    This uses a stricter Airy-tail threshold and a wider safety cap than the
    particle-placement padding because the coherent pupil filter is applied to
    the whole scene field, including empirical/background structure.
    """
    return _airy_support_radius_pixels(
        params,
        threshold_key=None,
        default_threshold=1e-5,
        max_radius_fraction_of_fov=1.0,
    )


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")

    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,) as [w, x, y, z].")

    w, x, y, z = q
    norm = np.linalg.norm(q)
    if norm == 0.0:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w /= norm
        x /= norm
        y /= norm
        z /= norm

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),   ww - xx - yy + zz],
        ],
        dtype=float,
    )
    return R


def _slerp_quaternions(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) between two unit quaternions.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("Quaternions must have shape (4,) as [w, x, y, z].")

    q0 = q0 / (np.linalg.norm(q0) or 1.0)
    q1 = q1 / (np.linalg.norm(q1) or 1.0)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(max(dot, -1.0), 1.0)

    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / (np.linalg.norm(q) or 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return q / (np.linalg.norm(q) or 1.0)


def _interpolate_orientation_for_instance(
    instance: ParticleInstance,
    time_index_float: float,
) -> np.ndarray | None:
    """
    Interpolate the orientation of a particle instance at a fractional frame index.
    """
    orientations = instance.orientation_matrices
    if orientations is None:
        return None

    num_frames = orientations.shape[0]
    if num_frames == 0:
        return None

    t = float(time_index_float)
    if t <= 0.0:
        return orientations[0]
    if t >= num_frames - 1:
        return orientations[-1]

    t_floor = int(np.floor(t))
    t_ceil = t_floor + 1
    alpha = t - t_floor

    if t_ceil >= num_frames:
        return orientations[-1]

    R0 = orientations[t_floor]
    R1 = orientations[t_ceil]

    q0 = _rotation_matrix_to_quaternion(R0)
    q1 = _rotation_matrix_to_quaternion(R1)
    q_interp = _slerp_quaternions(q0, q1, alpha)
    return _quaternion_to_rotation_matrix(q_interp)


def _iter_subparticle_render_info(
    instance: ParticleInstance,
    base_position_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
) -> list[tuple[np.ndarray, object, float, float, object]]:
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
                1.0,
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

    sub_infos: list[tuple[np.ndarray, object, float, float, object]] = []
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
                float(sub.diameter_nm),
                sub.material_properties if sub.material_properties is not None else instance.material_properties,
            )
        )

    return sub_infos




def generate_video_and_masks(params: dict, particle_instances: list[ParticleInstance]) -> RenderedFrameSet:
    """
    Generate video frames and, when enabled, segmentation masks.

    Particles are placed according to their trajectories, optical fields are
    accumulated with the configured motion-blur sampling, and the selected
    imaging model converts the scene into signal/reference frame sequences.

    The mask generation step uses detector-domain, noise-free particle
    contributions:

        - For each particle and frame, every rendered sub-exposure is evaluated
          with all particles and with that particle removed; the difference in
          detector counts is averaged over the exposure.
        - The saved geometry mask preserves the latent projected object
          footprint and, for non-composite particles, can include contrast lobes
          from the final detector-count contribution.
        - Composite particles use flood-fill contrast support around the rendered
          object geometry.

    Supervision policy operates on the same contrast images. The renderer emits
    a supervision annotation schema:

        mask_geometry   projected object/support mask before support gating
        mask_supported  geometry mask after configured support-factor gating
        ignore_mask     object pixels unsupported for the selected target
        loss_weight     per-pixel soft weight encoded as 0..255

    Side effects:
        When ``params["mask_generation_enabled"]`` is true, this function writes
        mask PNGs under ``params["mask_output_directory"]`` and writes the
        sidecar files ``supervision_records.jsonl``, ``supervision_audit.json``,
        and ``annotation_schema.json`` in that directory.

    Returns:
        ``RenderedFrameSet`` containing noisy signal/reference frames and,
        when ``params["return_ideal_float_frames"]`` is true, pre-noise float
        detector-count frames. The ideal-frame lists are empty when that option
        is false.

    """
    fps = float(params["fps"])
    num_frames = resolve_num_frames(params)
    if bool(params.get("supervision_stop_when_all_temporally_unsupported", False)) and bool(
        params.get("mask_generation_enabled", False)
    ):
        raise ValueError(
            "supervision_stop_when_all_temporally_unsupported is incompatible "
            "with fixed-length video and dataset manifests. Set it to False and "
            "use the emitted mask_supported/ignore_mask sidecars to exclude "
            "unsupported frames."
        )

    frame_interval_s = 1.0 / fps

    _raw_exposure = params.get("exposure_time_ms")
    exposure_time_ms = (
        1000.0 * frame_interval_s if _raw_exposure is None else float(_raw_exposure)
    )
    exposure_time_s = exposure_time_ms / 1000.0

    if exposure_time_s <= 0.0:
        raise ValueError("PARAMS['exposure_time_ms'] must be positive.")
    if exposure_time_s > frame_interval_s + 1e-12:
        raise ValueError(
            "PARAMS['exposure_time_ms'] must satisfy exposure_time_ms <= 1000 / fps "
            "so that the exposure window is contained within a single frame interval."
        )

    num_particles = len(particle_instances)
    expected_particle_count = particle_count(params)
    if num_particles != expected_particle_count:
        raise ValueError(
            "Number of ParticleInstance objects (%d) does not match "
            "the number of PARAMS['particles'] entries (%d)."
            % (num_particles, expected_particle_count)
        )

    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    # Instantiate before sizing the render canvas: Fourier-domain models need
    # a wider guard band than point-placement alone.
    imaging_model = get_imaging_model(params)
    pre_crop_optical_filtering = bool(
        getattr(imaging_model, "requires_pre_crop_optical_filtering", False)
    )
    render_guard_radius = estimate_psf_padding_radius_pixels(params)
    if pre_crop_optical_filtering:
        render_guard_radius = max(
            render_guard_radius,
            estimate_optical_filter_guard_radius_pixels(params),
        )
    os_canvas_size = os_size + 2 * render_guard_radius
    crop_start = render_guard_radius
    crop_end = crop_start + os_size

    # Full-canvas coordinate grids for sub-pixel particle placement. The
    # per-particle radial PSF profile is evaluated at every canvas pixel's
    # distance from the particle's actual float canvas coordinate (no
    # integer rounding), so the rendered field carries the particle's true
    # sub-pixel position rather than snapping to the oversampled-canvas
    # grid. These are computed once per run and reused across every
    # frame / particle / sub-particle.
    _yy_canvas_grid, _xx_canvas_grid = np.indices(
        (os_canvas_size, os_canvas_size), dtype=np.int32,
    )
    xx_canvas_full = _xx_canvas_grid.astype(np.float64)
    yy_canvas_full = _yy_canvas_grid.astype(np.float64)
    del _xx_canvas_grid, _yy_canvas_grid

    # Build reference maps after selecting the imaging model so modalities that
    # do not consume optical substrate patterns can explicitly bypass them while
    # leaving the caller's PARAMS untouched.
    pattern_requested = (
        bool(params.get("sample_environment_enabled", True))
        and bool(params.get("sample_environment_pattern_enabled", False))
    )
    pattern_active = (
        pattern_requested
        and bool(getattr(imaging_model, "uses_sample_environment_pattern", False))
    )
    reference_map_params = dict(params)
    if pattern_requested and not pattern_active:
        reference_map_params["sample_environment_pattern_enabled"] = False

    fov_shape_os = (os_size, os_size)
    model_shape_os = (
        (os_canvas_size, os_canvas_size)
        if pre_crop_optical_filtering
        else fov_shape_os
    )
    layout_extent_nm = (
        float(os_canvas_size) * float(pixel_size_nm) / float(os_factor)
        if pre_crop_optical_filtering
        else None
    )
    if pattern_active and layout_extent_nm is not None:
        reference_map_params["_substrate_pattern_layout_extent_nm"] = max(
            float(layout_extent_nm),
            float(reference_map_params.get("_substrate_pattern_layout_extent_nm", layout_extent_nm)),
        )
    sample_environment_model = (
        sample_environment_from_params(
            reference_map_params,
            model_shape_os,
            pixel_size_nm=float(pixel_size_nm) / float(os_factor),
        )
        if bool(params.get("sample_environment_enabled", True))
        else None
    )
    E_ref_os_base, E_ref_final_base, background_final_base = generate_reference_and_background_maps(
        reference_map_params,
        fov_shape_os=fov_shape_os,
        final_fov_shape=final_size,
        layout_extent_nm=layout_extent_nm,
    )
    E_ref_intensity_os_base = np.abs(E_ref_os_base) ** 2
    E_ref_intensity_final_base = np.abs(E_ref_final_base) ** 2
    if pre_crop_optical_filtering:
        E_ref_model_base, _, _ = generate_reference_and_background_maps(
            reference_map_params,
            fov_shape_os=model_shape_os,
            final_fov_shape=model_shape_os,
            layout_extent_nm=layout_extent_nm,
        )
    else:
        E_ref_model_base = E_ref_os_base
    E_ref_intensity_model_base = np.abs(E_ref_model_base) ** 2

    empirical_background_enabled = bool(
        params.get("empirical_background_enabled", False)
    )
    if empirical_background_enabled:
        if pre_crop_optical_filtering:
            empirical_background_model = generate_empirical_background_field(
                params,
                model_shape_os,
            )
            empirical_background_os = empirical_background_model[
                crop_start:crop_end, crop_start:crop_end
            ]
            empirical_background_final = cv2.resize(
                empirical_background_os,
                final_size,
                interpolation=cv2.INTER_AREA,
            )
            empirical_background_final = np.clip(
                empirical_background_final, 1e-6, None
            ).astype(float)
        else:
            empirical_background_final = generate_empirical_background_field(
                params,
                final_size,
            )
            empirical_background_os = resize_empirical_background_field(
                empirical_background_final,
                fov_shape_os,
            )
            empirical_background_model = empirical_background_os
        empirical_background_sqrt_os = np.sqrt(empirical_background_os)
        empirical_background_sqrt_model = np.sqrt(empirical_background_model)
    else:
        empirical_background_final = None
        empirical_background_os = None
        empirical_background_model = None
        empirical_background_sqrt_os = None
        empirical_background_sqrt_model = None

    contrast_model_raw = params.get("sample_environment_pattern_contrast_model", "static")
    contrast_model = str(contrast_model_raw).strip().lower()
    if contrast_model not in ("static", "time_dependent"):
        raise ValueError(
            "Unsupported sample_environment_pattern_contrast_model "
            f"'{contrast_model_raw}'. Supported values are 'static' and 'time_dependent'."
        )
    use_dynamic_contrast = (contrast_model == "time_dependent") and pattern_active

    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    if use_dynamic_contrast:
        if E_ref_amplitude > 0.0:
            pattern_os_base = E_ref_intensity_os_base / (E_ref_amplitude ** 2)
            pattern_model_base = E_ref_intensity_model_base / (E_ref_amplitude ** 2)
        else:
            pattern_os_base = np.ones_like(E_ref_intensity_os_base, dtype=float)
            pattern_model_base = np.ones_like(E_ref_intensity_model_base, dtype=float)

        if background_intensity > 0.0:
            pattern_final_base = background_final_base / background_intensity
        else:
            pattern_final_base = np.ones_like(background_final_base, dtype=float)

        mean_os = float(pattern_os_base.mean())
        if mean_os > 0.0:
            pattern_os_base /= mean_os

        mean_model = float(pattern_model_base.mean())
        if mean_model > 0.0:
            pattern_model_base /= mean_model

        mean_final = float(pattern_final_base.mean())
        if mean_final > 0.0:
            pattern_final_base /= mean_final

    bit_depth = params["bit_depth"]
    if (
        isinstance(bit_depth, bool)
        or not isinstance(bit_depth, numbers.Integral)
        or bit_depth <= 0
    ):
        raise ValueError("PARAMS['bit_depth'] must be a positive integer.")
    bit_depth = int(bit_depth)

    max_supported_bit_depth = 16
    if bit_depth > max_supported_bit_depth:
        raise ValueError(
            f"PARAMS['bit_depth']={bit_depth} exceeds the maximum supported bit depth "
            f"of {max_supported_bit_depth} for uint16 storage."
        )

    max_camera_count = (1 << bit_depth) - 1

    num_subsamples = params["motion_blur_subsamples"] if params["motion_blur_enabled"] else 1
    if not isinstance(num_subsamples, int) or num_subsamples <= 0:
        raise ValueError(
            "PARAMS['motion_blur_subsamples'] must be a positive integer."
        )
    sub_dt = exposure_time_s / num_subsamples

    all_signal_frames = []
    all_reference_frames = []
    return_ideal_float_frames = bool(params.get("return_ideal_float_frames", False))
    all_signal_ideal_frames = []
    all_reference_ideal_frames = []
    uses_particle_sources = bool(
        getattr(imaging_model, "uses_particle_material_sources", False)
    )

    def _new_particle_frame_states() -> list[_ParticleFrameRenderState]:
        states: list[_ParticleFrameRenderState] = []
        for _instance in particle_instances:
            source_canvas = (
                imaging_model.initialize_particle_source_canvas(
                    (os_canvas_size, os_canvas_size), params
                )
                if uses_particle_sources
                else None
            )
            states.append(
                _ParticleFrameRenderState(
                    field_canvas=np.zeros(
                        (os_canvas_size, os_canvas_size), dtype=np.complex128
                    ),
                    source_canvas=source_canvas,
                    geometry_canvas=np.zeros(
                        (os_canvas_size, os_canvas_size), dtype=np.float32
                    ),
                    rendered_position_sum_nm=np.zeros(3, dtype=float),
                )
            )
        return states

    def _model_arrays_from_states(
        states: list[_ParticleFrameRenderState],
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray] | None]:
        particle_field_canvases = [state.field_canvas for state in states]
        if particle_field_canvases:
            E_sca_total_canvas = np.sum(particle_field_canvases, axis=0)
        else:
            E_sca_total_canvas = np.zeros(
                (os_canvas_size, os_canvas_size), dtype=np.complex128
            )
        if pre_crop_optical_filtering:
            E_sca_total_for_model = E_sca_total_canvas
            particle_fields_for_model = particle_field_canvases
            particle_source_maps_for_model = None
            if uses_particle_sources:
                particle_source_maps_for_model = [
                    state.source_canvas for state in states
                ]
        else:
            E_sca_total_for_model = E_sca_total_canvas[
                crop_start:crop_end, crop_start:crop_end
            ]
            particle_fields_for_model = [
                state.field_fov(crop_start, crop_end) for state in states
            ]
            particle_source_maps_for_model = None
            if uses_particle_sources:
                particle_source_maps_for_model = [
                    state.source_fov(crop_start, crop_end) for state in states
                ]
        return (
            E_sca_total_for_model,
            particle_fields_for_model,
            particle_source_maps_for_model,
        )

    def _detector_intensity_os_from_states(
        states: list[_ParticleFrameRenderState],
        E_ref_model_current: np.ndarray,
        frame_index: int,
    ) -> np.ndarray:
        (
            E_sca_total_for_model,
            particle_fields_for_model,
            particle_source_maps_for_model,
        ) = _model_arrays_from_states(states)
        intensity_for_model = imaging_model.compute_scene_intensity(
            particle_fields_for_model,
            particle_instances,
            E_sca_total_for_model,
            E_ref_model_current,
            params,
            particle_source_maps=particle_source_maps_for_model,
            frame_index=frame_index,
        )
        intensity_for_model = imaging_model.apply_sample_environment(
            intensity=intensity_for_model,
            E_sca_total=E_sca_total_for_model,
            background_field=E_ref_model_current,
            params=params,
            sample_environment=sample_environment_model,
        )
        if pre_crop_optical_filtering:
            return intensity_for_model[crop_start:crop_end, crop_start:crop_end]
        return intensity_for_model

    def _detector_counts_from_states(
        states: list[_ParticleFrameRenderState],
        E_ref_model_current: np.ndarray,
        background_final_current: np.ndarray,
        E_ref_intensity_final_current: np.ndarray,
        frame_index: int,
    ) -> np.ndarray:
        intensity_os_current = _detector_intensity_os_from_states(
            states,
            E_ref_model_current,
            frame_index,
        )
        intensity_current = cv2.resize(
            intensity_os_current,
            final_size,
            interpolation=cv2.INTER_AREA,
        )
        return imaging_model.scale_intensity_to_counts(
            intensity=intensity_current,
            background_final=background_final_current,
            E_ref_intensity_final=E_ref_intensity_final_current,
            params=params,
        )

    return_mask_arrays = bool(params.get("_return_mask_arrays", False))
    write_mask_files = bool(params.get("_write_mask_files", True))
    returned_mask_arrays: list[dict] = []

    if params["mask_generation_enabled"]:
        supervision_policy = SupervisionPolicy(params, num_particles)
        supervision_audit = SupervisionAudit()
        supervision_records: list[dict] = []
        mask_root_dir = params["mask_output_directory"]
        if write_mask_files:
            for schema_name in (
                "mask_geometry",
                "mask_supported",
                "ignore_mask",
                "loss_weight",
            ):
                for i in range(num_particles):
                    os.makedirs(
                        os.path.join(mask_root_dir, schema_name, f"particle_{i + 1}"),
                        exist_ok=True,
                    )
    else:
        supervision_policy = None
        supervision_audit = None
        supervision_records = []

    logger.info("Generating video frames and masks...")
    for f in tqdm(range(num_frames), disable=not logger.isEnabledFor(logging.INFO)):
        if use_dynamic_contrast:
            alpha_f = compute_contrast_scale_for_frame(params, f, num_frames)

            pattern_os_f = 1.0 + alpha_f * (pattern_os_base - 1.0)
            pattern_model_f = 1.0 + alpha_f * (pattern_model_base - 1.0)
            pattern_final_f = 1.0 + alpha_f * (pattern_final_base - 1.0)

            pattern_os_f = np.maximum(pattern_os_f, 1e-8)
            pattern_model_f = np.maximum(pattern_model_f, 1e-8)
            pattern_final_f = np.maximum(pattern_final_f, 1e-8)

            E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os_f)).astype(np.complex128)
            E_ref_model = (E_ref_amplitude * np.sqrt(pattern_model_f)).astype(np.complex128)

            E_ref_intensity_final = (E_ref_amplitude ** 2) * pattern_final_f
            background_final = background_intensity * pattern_final_f
        else:
            E_ref_os = E_ref_os_base
            E_ref_model = E_ref_model_base
            E_ref_intensity_final = E_ref_intensity_final_base
            background_final = background_final_base

        if empirical_background_enabled:
            E_ref_os = (E_ref_os * empirical_background_sqrt_os).astype(np.complex128)
            E_ref_model = (E_ref_model * empirical_background_sqrt_model).astype(np.complex128)
            E_ref_intensity_final = E_ref_intensity_final * empirical_background_final
            background_final = background_final * empirical_background_final

        particle_frame_states = _new_particle_frame_states()

        drift_velocity = np.asarray(
            params.get("drift_velocity_nm_per_s", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if drift_velocity.size == 1:
            drift_velocity = np.array([float(drift_velocity), 0.0, 0.0])
        if drift_velocity.size != 3:
            raise ValueError("drift_velocity_nm_per_s must be a scalar or length-3 sequence.")

        frame_vibration_nm = np.zeros(3, dtype=float)
        vibration_std_nm = float(params.get("vibration_jitter_std_nm", 0.0))
        if vibration_std_nm > 0.0:
            frame_vibration_nm = np.random.normal(scale=vibration_std_nm, size=3)
            if not bool(params.get("vibration_include_axial", False)):
                frame_vibration_nm[2] = 0.0

        intensity_os_sum = None
        subsample_states_by_exposure: list[list[_ParticleFrameRenderState]] = []
        for s in range(num_subsamples):
            subsample_states = _new_particle_frame_states()
            frame_center_time = (f + 0.5) * frame_interval_s
            start_time = frame_center_time - 0.5 * exposure_time_s
            current_time = start_time + (s + 0.5) * sub_dt
            global_motion_shift_nm = drift_velocity * current_time + frame_vibration_nm

            # Trajectory samples are indexed by output frame number: trajectory[f]
            # is the particle state represented by frame f and by the masks saved
            # for frame f. Physical time starts at the beginning of frame 0, so
            # the center of frame f is at (f + 0.5) * frame_interval_s. Convert
            # physical sub-exposure times back to this frame-centered index.
            time_index_float = (current_time / frame_interval_s) - 0.5

            frame_idx_floor = int(np.floor(time_index_float))
            if frame_idx_floor < 0:
                frame_idx_floor = 0

            if frame_idx_floor >= num_frames - 1:
                frame_idx_floor = num_frames - 1
                frame_idx_ceil = num_frames - 1
                interp_factor = 0.0
            else:
                frame_idx_ceil = frame_idx_floor + 1
                interp_factor = time_index_float - frame_idx_floor
                if interp_factor < 0.0:
                    interp_factor = 0.0

            for i, instance in enumerate(particle_instances):
                frame_state = particle_frame_states[i]
                subsample_state = subsample_states[i]
                traj = instance.trajectory_nm
                if traj.shape[0] != num_frames or traj.shape[1] != 3:
                    raise ValueError(
                        "ParticleInstance %d has trajectory shape %s, expected (%d, 3)."
                        % (i, traj.shape, num_frames)
                    )

                pos_floor = traj[frame_idx_floor]
                pos_ceil = traj[frame_idx_ceil]
                current_pos_nm = (1.0 - interp_factor) * pos_floor + interp_factor * pos_ceil
                current_pos_nm = current_pos_nm + global_motion_shift_nm
                frame_state.add_rendered_position(current_pos_nm)

                orientation_matrix = _interpolate_orientation_for_instance(
                    instance=instance,
                    time_index_float=time_index_float,
                )

                sub_infos = _iter_subparticle_render_info(
                    instance=instance,
                    base_position_nm=current_pos_nm,
                    orientation_matrix=orientation_matrix,
                )

                for world_pos_nm, sub_interp, local_multiplier, sub_diameter_nm, sub_material in sub_infos:
                    px, py, pz = world_pos_nm
                    center_x_canvas = crop_start + px / pixel_size_nm * os_factor
                    center_y_canvas = crop_start + py / pixel_size_nm * os_factor
                    _accumulate_projected_geometry_disk(
                        frame_state.geometry_canvas,
                        center_x_canvas=center_x_canvas,
                        center_y_canvas=center_y_canvas,
                        diameter_nm=float(sub_diameter_nm),
                        pixel_size_nm=float(pixel_size_nm),
                        os_factor=int(os_factor),
                    )

                    E_sca_2D = sub_interp([pz])[0]

                    # Sub-pixel particle placement.
                    #
                    # The optics backend enforces radial symmetry on each
                    # complex PSF z-slice, so it is fully described by a 1-D
                    # radial profile E_radial(r). For each particle we
                    # evaluate that profile at every canvas pixel's distance
                    # from the particle's *actual sub-pixel* canvas
                    # coordinate (no integer rounding), then accumulate the
                    # result directly into the canvas. The rendered field
                    # therefore preserves the particle's continuous canvas
                    # position for localization-sensitive render paths.
                    pupil_samples = E_sca_2D.shape[0]
                    center_psf = pupil_samples // 2
                    E_radial_line = E_sca_2D[center_psf, center_psf:]
                    max_bin_psf = E_radial_line.size - 1

                    if max_bin_psf > 0:
                        # Spatial pitch of the precomputed PSF stack in
                        # nm per stack-pixel. For pupil_samples == img_size
                        # and os_factor=1 this equals pixel_size_nm; for
                        # other configurations it tracks the corresponding
                        # FFT-domain pitch chosen in
                        # optics.compute_complex_psf_stack.
                        nm_per_pixel_psf = (
                            img_size * pixel_size_nm
                        ) / (os_factor * pupil_samples)
                        r_bins_nm = np.arange(
                            max_bin_psf + 1, dtype=float,
                        ) * nm_per_pixel_psf

                        # Particle's sub-pixel canvas coordinate (float, no
                        # rounding). px, py are world-frame nm; convert to
                        # oversampled-canvas pixels and offset by the
                        # guard-band crop_start.
                        particle_x_canvas = (
                            float(center_x_canvas)
                        )
                        particle_y_canvas = (
                            float(center_y_canvas)
                        )
                        nm_per_pixel_canvas = pixel_size_nm / os_factor

                        # Per-canvas-pixel radial distance from the particle
                        # in nm. xx_canvas_full / yy_canvas_full were built
                        # once before the frame loop and span the full
                        # (guard-padded) canvas.
                        dx_canvas = xx_canvas_full - particle_x_canvas
                        dy_canvas = yy_canvas_full - particle_y_canvas
                        r_canvas_nm = np.sqrt(
                            dx_canvas * dx_canvas + dy_canvas * dy_canvas
                        ) * nm_per_pixel_canvas

                        # Interpolate the radial profile at each canvas
                        # pixel's distance. ``right=0.0`` truncates the PSF
                        # contribution past max_bin_psf, which is the
                        # finite-extent boundary of the precomputed PSF and
                        # corresponds physically to the optical cutoff.
                        E_real = np.interp(
                            r_canvas_nm.ravel(),
                            r_bins_nm,
                            E_radial_line.real,
                            right=0.0,
                        ).reshape(r_canvas_nm.shape)
                        E_imag = np.interp(
                            r_canvas_nm.ravel(),
                            r_bins_nm,
                            E_radial_line.imag,
                            right=0.0,
                        ).reshape(r_canvas_nm.shape)

                        amplitude_scale = (
                            instance.signal_multiplier * local_multiplier
                        )
                        # Accumulate directly into the canvas; the field is
                        # already correctly centred at the particle's
                        # sub-pixel canvas coordinate, so no integer-pixel
                        # shift is needed.
                        field_contribution = (
                            E_real + 1j * E_imag
                        ) * amplitude_scale
                        for target_state in (frame_state, subsample_state):
                            target_state.field_canvas += field_contribution

                    # Source-canvas accumulation uses the same sub-pixel canvas
                    # coordinate as the scattered-field placement.
                    if frame_state.source_canvas is not None:
                        for target_state in (frame_state, subsample_state):
                            imaging_model.accumulate_particle_source(
                                target_state.source_canvas,
                                center_x_canvas=center_x_canvas,
                                center_y_canvas=center_y_canvas,
                                diameter_nm=float(sub_diameter_nm),
                                pixel_size_nm=float(pixel_size_nm),
                                os_factor=int(os_factor),
                                material_properties=sub_material,
                                params=params,
                                particle_z_nm=float(pz),
                            )

            subsample_intensity_os = _detector_intensity_os_from_states(
                subsample_states, E_ref_model, f
            )
            subsample_states_by_exposure.append(subsample_states)
            if intensity_os_sum is None:
                intensity_os_sum = np.asarray(subsample_intensity_os, dtype=float)
            else:
                intensity_os_sum += np.asarray(subsample_intensity_os, dtype=float)

        for frame_state in particle_frame_states:
            frame_state.normalize_exposure(num_subsamples)

        if intensity_os_sum is None:
            raise RuntimeError("No detector-domain subexposure intensity was rendered.")
        intensity_os = intensity_os_sum / float(num_subsamples)
        intensity = cv2.resize(intensity_os, final_size, interpolation=cv2.INTER_AREA)

        intensity_scaled = imaging_model.scale_intensity_to_counts(
            intensity=intensity,
            background_final=background_final,
            E_ref_intensity_final=E_ref_intensity_final,
            params=params,
        )

        # Render a no-particle reference through the same model and sample
        # environment so background subtraction compares like with like.
        zero_field_for_model = np.zeros_like(E_ref_model, dtype=np.complex128)
        reference_intensity_for_model = imaging_model.compute_scene_intensity(
            [],
            [],
            zero_field_for_model,
            E_ref_model,
            params,
            particle_source_maps=[] if uses_particle_sources else None,
            frame_index=f,
        )
        reference_intensity_for_model = imaging_model.apply_sample_environment(
            intensity=reference_intensity_for_model,
            E_sca_total=zero_field_for_model,
            background_field=E_ref_model,
            params=params,
            sample_environment=sample_environment_model,
        )
        if pre_crop_optical_filtering:
            reference_intensity_os = reference_intensity_for_model[
                crop_start:crop_end, crop_start:crop_end
            ]
        else:
            reference_intensity_os = reference_intensity_for_model
        reference_intensity = cv2.resize(
            reference_intensity_os,
            final_size,
            interpolation=cv2.INTER_AREA,
        )
        reference_frame_ideal = imaging_model.scale_intensity_to_counts(
            intensity=reference_intensity,
            background_final=background_final,
            E_ref_intensity_final=E_ref_intensity_final,
            params=params,
        )
        supervision_noise_variance = contrast_noise_variance_counts(
            intensity_scaled,
            reference_frame_ideal,
            params,
            relative_reference=False,
        )
        finite_variance = supervision_noise_variance[
            np.isfinite(supervision_noise_variance)
        ]
        supervision_noise_std = (
            float(np.sqrt(np.median(finite_variance)))
            if finite_variance.size
            else float("inf")
        )

        if params["mask_generation_enabled"]:
            subsample_all_counts = [
                _detector_counts_from_states(
                    states,
                    E_ref_model,
                    background_final,
                    E_ref_intensity_final,
                    f,
                )
                for states in subsample_states_by_exposure
            ]
            mask_inputs = []
            for i, instance in enumerate(particle_instances):
                frame_state = particle_frame_states[i]
                contribution_sum = None
                for states, counts_all in zip(
                    subsample_states_by_exposure,
                    subsample_all_counts,
                    strict=False,
                ):
                    states_without_particle = [
                        state for idx, state in enumerate(states) if idx != i
                    ]
                    counts_without_particle = _detector_counts_from_states(
                        states_without_particle,
                        E_ref_model,
                        background_final,
                        E_ref_intensity_final,
                        f,
                    )
                    contribution = counts_all - counts_without_particle
                    if contribution_sum is None:
                        contribution_sum = np.asarray(contribution, dtype=float)
                    else:
                        contribution_sum += np.asarray(contribution, dtype=float)
                if contribution_sum is None:
                    contrast_final_counts = np.zeros(final_size[::-1], dtype=float)
                else:
                    contrast_final_counts = contribution_sum / float(
                        len(subsample_states_by_exposure)
                    )

                H, W = contrast_final_counts.shape
                geometry_os = frame_state.geometry_canvas[
                    crop_start:crop_end, crop_start:crop_end
                ]
                geometry_final = cv2.resize(
                    geometry_os, final_size, interpolation=cv2.INTER_AREA
                )
                projected_geometry_mask = (geometry_final > 0.0).astype(np.uint8) * 255
                if projected_geometry_mask.shape != (H, W):
                    raise RuntimeError(
                        "Latent geometry mask shape does not match contrast image shape."
                    )
                position_nm = frame_state.rendered_position_nm(instance.trajectory_nm[f, :])
                center_yx = (
                    int(round(float(position_nm[1]) / float(pixel_size_nm))),
                    int(round(float(position_nm[0]) / float(pixel_size_nm))),
                )
                is_composite = bool(
                    getattr(getattr(instance, "particle_type", None), "is_composite", False)
                )
                lobe_mask = generate_central_lobe_mask(
                    contrast_final_counts,
                    center_yx=center_yx,
                    outer_ring_count=0 if is_composite else int(params.get("mask_outer_ring_count", 0)),
                    use_floodfill=is_composite,
                    max_area_fraction=float(params.get("mask_max_area_fraction", 0.25)),
                )
                geometry_mask = np.maximum(projected_geometry_mask, lobe_mask).astype(np.uint8)

                mask_inputs.append(
                    {
                        "particle_index": i,
                        "frame_index": f,
                        "position_nm": position_nm,
                        "contrast_image": contrast_final_counts,
                        "geometry_mask": geometry_mask,
                    }
                )

            all_positions_nm = np.asarray(
                [item["position_nm"] for item in mask_inputs],
                dtype=float,
            )
            all_geometry_masks = [
                item["geometry_mask"] for item in mask_inputs
            ]
            for item in mask_inputs:
                policy_result = supervision_policy.evaluate(
                    particle_index=item["particle_index"],
                    frame_index=item["frame_index"],
                    position_nm=item["position_nm"],
                    contrast_image=item["contrast_image"],
                    geometry_mask=item["geometry_mask"],
                    all_positions_nm=all_positions_nm,
                    all_geometry_masks=all_geometry_masks,
                    noise_std=supervision_noise_std,
                    noise_variance_map=supervision_noise_variance,
                )

                masks = policy_result["masks"]
                record = policy_result["record"]
                supervision_records.append(record)
                supervision_audit.add(record)
                if return_mask_arrays:
                    returned_mask_arrays.append(
                        {
                            "particle_index": int(item["particle_index"]),
                            "frame_index": int(item["frame_index"]),
                            "masks": {
                                str(schema_name): np.asarray(mask_arr).copy()
                                for schema_name, mask_arr in masks.items()
                            },
                        }
                    )

                if write_mask_files:
                    for schema_name, mask_arr in masks.items():
                        save_mask(
                            mask_arr,
                            os.path.join(mask_root_dir, schema_name),
                            particle_index=item["particle_index"],
                            frame_index=item["frame_index"],
                        )

        if return_ideal_float_frames:
            all_signal_ideal_frames.append(
                np.where(np.isfinite(intensity_scaled), intensity_scaled, 0.0).astype(float, copy=False)
            )

        signal_frame_noisy = imaging_model.compute_noise(intensity_scaled, params)
        all_signal_frames.append(
            np.clip(signal_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        if return_ideal_float_frames:
            all_reference_ideal_frames.append(
                np.where(np.isfinite(reference_frame_ideal), reference_frame_ideal, 0.0).astype(float, copy=False)
            )
        reference_frame_noisy = imaging_model.compute_noise(reference_frame_ideal, params)
        all_reference_frames.append(
            np.clip(reference_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

    logger.info("Frame and mask generation complete.")
    if params["mask_generation_enabled"]:
        audit_path = os.path.join(params["mask_output_directory"], "supervision_audit.json")
        records_path = os.path.join(
            params["mask_output_directory"], "supervision_records.jsonl"
        )
        schema_path = os.path.join(params["mask_output_directory"], "annotation_schema.json")
        if write_mask_files:
            with open(records_path, "w", encoding="utf-8") as fh:
                for record in supervision_records:
                    fh.write(
                        json.dumps(
                            _strict_json_safe(record),
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
            with open(audit_path, "w", encoding="utf-8") as fh:
                json.dump(
                    _strict_json_safe(supervision_audit.summary()),
                    fh,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            with open(schema_path, "w", encoding="utf-8") as fh:
                json.dump(
                    build_policy_annotation_schema(params),
                    fh,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
    return RenderedFrameSet(
        signal_frames=all_signal_frames,
        reference_frames=all_reference_frames,
        ideal_signal_frames=all_signal_ideal_frames,
        ideal_reference_frames=all_reference_ideal_frames,
        mask_arrays=returned_mask_arrays,
        supervision_records=list(supervision_records),
        supervision_audit_summary=(
            _strict_json_safe(supervision_audit.summary())
            if supervision_audit is not None else None
        ),
    )
