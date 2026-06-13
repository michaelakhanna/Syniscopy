from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_AXES = ("x", "y", "z", "omega_x", "omega_y", "omega_z")


def _grid(size: int = 45, extent: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-extent, extent, int(size), dtype=float)
    return np.meshgrid(axis, axis, indexing="xy")


def _normalise_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr - float(np.mean(arr))
    scale = float(np.sqrt(np.mean(arr * arr)))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(arr)
    return arr / scale


def analytic_se3_renders(
    *,
    observable_rotations: set[str] | frozenset[str],
    size: int = 45,
    z_step_nm: float = 20.0,
    rotation_step_rad: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """Build independent pixelated SE(3) perturbation renders.

    The centre image supplies lateral x/y derivatives through Syniscopy's
    stationary-shift convention. The z and rotation perturbations are explicit
    symmetric finite-difference pairs.
    """
    xx, yy = _grid(size=size)
    rr = xx * xx + yy * yy
    g = np.exp(-0.5 * rr)
    centre = g * (1.0 + 0.08 * xx + 0.04 * (xx * xx - yy * yy))

    dz = _normalise_map((rr - 1.6) * g)
    drx = _normalise_map(xx * yy * g)
    dry = _normalise_map((xx * xx - yy * yy) * g)
    drz = _normalise_map((xx * xx * xx - 3.0 * xx * yy * yy) * g)
    derivs = {"rx": drx, "ry": dry, "rz": drz}

    renders: dict[str, np.ndarray] = {
        "centre": centre,
        "z_minus": centre - z_step_nm * dz,
        "z_plus": centre + z_step_nm * dz,
    }
    for short, key in (("rx", "rx"), ("ry", "ry"), ("rz", "rz")):
        deriv = derivs[short] if short in observable_rotations else np.zeros_like(centre)
        renders[f"{key}_minus"] = centre - rotation_step_rad * deriv
        renders[f"{key}_plus"] = centre + rotation_step_rad * deriv
    return renders


def featureless_sphere_renders(
    *,
    size: int = 45,
    z_step_nm: float = 20.0,
    rotation_step_rad: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    xx, yy = _grid(size=size)
    rr = xx * xx + yy * yy
    centre = np.exp(-0.5 * rr)
    dz = _normalise_map((rr - 1.5) * centre)
    renders = {
        "centre": centre,
        "z_minus": centre - z_step_nm * dz,
        "z_plus": centre + z_step_nm * dz,
    }
    for axis in ("rx", "ry", "rz"):
        renders[f"{axis}_minus"] = centre.copy()
        renders[f"{axis}_plus"] = centre.copy()
    return renders


def zero_signal_renders(size: int = 33) -> dict[str, np.ndarray]:
    zero = np.zeros((int(size), int(size)), dtype=float)
    return {
        "centre": zero.copy(),
        "z_minus": zero.copy(),
        "z_plus": zero.copy(),
        "rx_minus": zero.copy(),
        "rx_plus": zero.copy(),
        "ry_minus": zero.copy(),
        "ry_plus": zero.copy(),
        "rz_minus": zero.copy(),
        "rz_plus": zero.copy(),
    }


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@dataclass(frozen=True)
class Pose:
    translation_nm: np.ndarray
    rotation: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3, dtype=float), np.eye(3, dtype=float))

    def translated(self, delta_nm: tuple[float, float, float]) -> "Pose":
        return Pose(self.translation_nm + np.asarray(delta_nm, dtype=float), self.rotation)

    def body_rotated(self, axis: str, angle_rad: float) -> "Pose":
        if axis == "x":
            delta = _rot_x(angle_rad)
        elif axis == "y":
            delta = _rot_y(angle_rad)
        elif axis == "z":
            delta = _rot_z(angle_rad)
        else:
            raise ValueError(axis)
        return Pose(self.translation_nm, self.rotation @ delta)


class RigidProjector:
    """Small independent rigid-body image model for loop-invariance checks."""

    def __init__(self, *, size: int = 49, pixel_size_nm: float = 100.0) -> None:
        self.size = int(size)
        self.pixel_size_nm = float(pixel_size_nm)
        self.body_points_nm = np.asarray(
            [
                [-90.0, -20.0, -20.0],
                [70.0, 0.0, 15.0],
                [10.0, 75.0, 30.0],
            ],
            dtype=float,
        )
        self.weights = np.asarray([1.0, 0.75, 0.55], dtype=float)

    def render(self, pose: Pose) -> np.ndarray:
        yy, xx = np.indices((self.size, self.size), dtype=float)
        centre = 0.5 * (self.size - 1.0)
        image = np.zeros((self.size, self.size), dtype=float)
        world = (pose.rotation @ self.body_points_nm.T).T + pose.translation_nm
        for point, weight in zip(world, self.weights):
            cx = centre + point[0] / self.pixel_size_nm
            cy = centre + point[1] / self.pixel_size_nm
            z = point[2]
            sigma = 1.45 + 0.0015 * abs(z)
            amp = weight * (1.0 + 0.0008 * z)
            image += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        return image

    def se3_renders(
        self,
        pose: Pose,
        *,
        z_step_nm: float,
        rotation_step_rad: float,
    ) -> dict[str, np.ndarray]:
        return {
            "centre": self.render(pose),
            "z_minus": self.render(pose.translated((0.0, 0.0, -z_step_nm))),
            "z_plus": self.render(pose.translated((0.0, 0.0, z_step_nm))),
            "rx_minus": self.render(pose.body_rotated("x", -rotation_step_rad)),
            "rx_plus": self.render(pose.body_rotated("x", rotation_step_rad)),
            "ry_minus": self.render(pose.body_rotated("y", -rotation_step_rad)),
            "ry_plus": self.render(pose.body_rotated("y", rotation_step_rad)),
            "rz_minus": self.render(pose.body_rotated("z", -rotation_step_rad)),
            "rz_plus": self.render(pose.body_rotated("z", rotation_step_rad)),
        }

