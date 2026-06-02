"""Quaternion orientation interpolation helpers for rendering."""

from __future__ import annotations

import numpy as np

from particle_model import ParticleInstance

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

