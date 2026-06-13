from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for substrate-dependent bootstrap; missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _load_orientation_interpolation():
    _require_cv2_for_bootstrap()
    return importlib.import_module("rendering.orientation_interpolation")


_OI = _load_orientation_interpolation()
_slerp_quaternions = _OI._slerp_quaternions
_rotation_matrix_to_quaternion = _OI._rotation_matrix_to_quaternion
_quaternion_to_rotation_matrix = _OI._quaternion_to_rotation_matrix


def _rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("axis must be non-zero")
    axis = axis / norm
    x, y, z = axis
    half = 0.5 * angle_rad
    s = np.sin(half)
    w = np.cos(half)
    q = np.array([w, x * s, y * s, z * s], dtype=float)
    return _quaternion_to_rotation_matrix(q)


def _rotation_angle_from_matrix(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    val = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(val, -1.0, 1.0)))


def test_slerp_endpoints_and_unit_norm() -> None:
    R0 = np.eye(3)
    R1 = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)

    q0 = _rotation_matrix_to_quaternion(R0)
    q1 = _rotation_matrix_to_quaternion(R1)

    q_start = _slerp_quaternions(q0, q1, 0.0)
    q_end = _slerp_quaternions(q0, q1, 1.0)

    R_start = _quaternion_to_rotation_matrix(q_start)
    R_end = _quaternion_to_rotation_matrix(q_end)

    np.testing.assert_allclose(R_start, R0, atol=1e-12)
    np.testing.assert_allclose(R_end, R1, atol=1e-12)
    assert abs(np.linalg.norm(q_start) - 1.0) < 1e-12
    assert abs(np.linalg.norm(q_end) - 1.0) < 1e-12


def test_slerp_halfway_is_half_angle_for_perpendicular_axis() -> None:
    R0 = np.eye(3)
    R1 = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)

    q0 = _rotation_matrix_to_quaternion(R0)
    q1 = _rotation_matrix_to_quaternion(R1)
    q_mid = _slerp_quaternions(q0, q1, 0.5)
    R_mid = _quaternion_to_rotation_matrix(q_mid)

    angle_mid = _rotation_angle_from_matrix(R_mid)
    assert abs(angle_mid - (np.pi / 4)) < 3e-7


def test_rotation_matrix_quaternion_round_trip_identity() -> None:
    R = _rotation_from_axis_angle(np.array([1.0, 1.0, 0.2]), 0.72)
    q = _rotation_matrix_to_quaternion(R)
    R_round = _quaternion_to_rotation_matrix(q)

    np.testing.assert_allclose(R_round, R, atol=1e-12)


def test_slerp_is_shortest_path() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q1 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=float)

    q_mid = _slerp_quaternions(q0, q1, 0.5)

    assert abs(np.linalg.norm(q_mid) - 1.0) < 1e-12
    assert np.allclose(q_mid, np.array([1.0, 0.0, 0.0, 0.0]))
