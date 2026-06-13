from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend_fidelity import attach_backend_fidelity_metadata
# Single-source the elementary charge from the one electron-optics constants
# import the canonical value directly from shared constants to avoid duplication.
from electron_optics import (
    _ELEMENTARY_CHARGE_C as _E_CHARGE_C,
)


class SEMTransportBackendError(RuntimeError):
    """Raised when SEM transport/reference-kernel data cannot be used safely."""


SEM_REFERENCE_KERNEL_SCHEMA_VERSION = "syniscopy-sem-reference-kernel-v1"


@dataclass(frozen=True)
class SEMTransportMetadata:
    backend_mode: str
    backend_fidelity_level: str
    backend_name: str
    equations_or_model_family: str
    implemented_approximation_level: str
    native_operating_assumptions: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "backend_fidelity_level": self.backend_fidelity_level,
            "backend_name": self.backend_name,
            "equations_or_model_family": self.equations_or_model_family,
            "implemented_approximation_level": self.implemented_approximation_level,
            "native_operating_assumptions": self.native_operating_assumptions,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_nonnegative(
    name: str, value: Any, *, allow_none: bool = False, minimum: float = 0.0
) -> float:
    if value is None:
        if allow_none:
            return float("nan")
        raise SEMTransportBackendError(f"SEM transport parameter '{name}' is required.")
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise SEMTransportBackendError(f"SEM transport parameter '{name}' must be numeric; got {value!r}.") from exc
    if not np.isfinite(value_f):
        raise SEMTransportBackendError(f"SEM transport parameter '{name}' must be finite; got {value!r}.")
    if value_f < minimum:
        raise SEMTransportBackendError(
            f"SEM transport parameter '{name}' must be >= {minimum}; got {value_f}."
        )
    return value_f


def _electrons_from_beam_current(
    beam_current_nA: float,
    dwell_time_us: float,
) -> float | None:
    """Convert SEM beam current and dwell time to incident electrons."""
    if beam_current_nA <= 0.0 or dwell_time_us <= 0.0:
        return None
    charge_coulombs = (beam_current_nA * 1.0e-9) * (dwell_time_us * 1.0e-6)
    electrons = charge_coulombs / _E_CHARGE_C
    if electrons <= 0.0 or not np.isfinite(electrons):
        return None
    return float(electrons)


def _validate_takeoff_angle_deg(takeoff_angle_deg: float) -> float:
    angle = float(takeoff_angle_deg)
    if not np.isfinite(angle) or angle < 0.0 or angle > 90.0:
        raise SEMTransportBackendError(
            "SEM detector takeoff angle is measured above the specimen surface "
            f"and must be in [0, 90] degrees; got {takeoff_angle_deg!r}."
        )
    return angle


def _detector_takeoff_acceptance_gain(detector_acceptance: float, takeoff_angle_deg: float) -> float:
    """Return collection gain for takeoff measured above the specimen surface."""
    angle = _validate_takeoff_angle_deg(takeoff_angle_deg)
    acceptance = _finite_nonnegative(
        "sem_detector_acceptance",
        detector_acceptance,
        minimum=0.0,
    )
    return float(acceptance * max(np.sin(np.deg2rad(angle)), 0.0))


def _gaussian_kernel_1d(sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(int(4.0 * sigma_px), 1)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma_px)) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _convolve1d_reflect_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    row = np.asarray(values, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if row.size <= 1 or ker.size <= 1:
        return row * float(np.sum(ker))
    radius = ker.size // 2
    padded = np.pad(row, radius, mode="reflect")
    return np.convolve(padded, ker, mode="valid")


def _gaussian_blur(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return np.asarray(arr, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(np.asarray(arr, dtype=float), sigma=sigma_px)
    except ImportError:
        kernel = _gaussian_kernel_1d(sigma_px)
        out = np.asarray(arr, dtype=float, copy=True)
        out = np.apply_along_axis(_convolve1d_reflect_same, axis=0, arr=out, kernel=kernel)
        out = np.apply_along_axis(_convolve1d_reflect_same, axis=1, arr=out, kernel=kernel)
        return out


def _fft_convolve_centered(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear same-size convolution for centered, normalized kernels."""
    src = np.asarray(arr, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if src.ndim != 2 or ker.ndim != 2:
        raise SEMTransportBackendError(
            f"SEM convolution expects 2D arrays; got source={src.shape}, kernel={ker.shape}."
        )
    if ker.shape[0] > src.shape[0] or ker.shape[1] > src.shape[1]:
        raise SEMTransportBackendError(
            "SEM convolution kernel is larger than the guarded render canvas "
            f"(kernel={ker.shape}, canvas={src.shape}). Increase the SEM filter "
            "guard/canvas size instead of silently cropping the kernel."
        )
    try:
        from scipy.signal import fftconvolve

        return np.asarray(fftconvolve(src, ker, mode="same"), dtype=float)
    except ImportError:
        full_shape = (src.shape[0] + ker.shape[0] - 1, src.shape[1] + ker.shape[1] - 1)
        conv = np.fft.irfftn(
            np.fft.rfftn(src, full_shape) * np.fft.rfftn(ker, full_shape),
            full_shape,
        )
        y0 = (ker.shape[0] - 1) // 2
        x0 = (ker.shape[1] - 1) // 2
        return np.asarray(conv[y0:y0 + src.shape[0], x0:x0 + src.shape[1]], dtype=float)


def _gradient_components(arr: np.ndarray, spacing_nm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    spacing = float(spacing_nm)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SEMTransportBackendError(f"Gradient spacing must be positive and finite; got {spacing_nm!r}.")
    if min(arr.shape) < 2:
        zeros = np.zeros_like(arr, dtype=float)
        return zeros, zeros
    gy = np.empty_like(arr, dtype=float)
    gx = np.empty_like(arr, dtype=float)
    gy[1:-1, :] = 0.5 * (arr[2:, :] - arr[:-2, :])
    gy[0, :] = arr[1, :] - arr[0, :]
    gy[-1, :] = arr[-1, :] - arr[-2, :]
    gx[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2])
    gx[:, 0] = arr[:, 1] - arr[:, 0]
    gx[:, -1] = arr[:, -1] - arr[:, -2]
    return gx / spacing, gy / spacing


def _gradient_magnitude(arr: np.ndarray, spacing_nm: float = 1.0) -> np.ndarray:
    gx, gy = _gradient_components(arr, spacing_nm=spacing_nm)
    return np.sqrt(gx ** 2 + gy ** 2)



__all__ = [
    "Any",
    "Mapping",
    "Path",
    "SEMTransportBackendError",
    "SEMTransportMetadata",
    "SEM_REFERENCE_KERNEL_SCHEMA_VERSION",
    "attach_backend_fidelity_metadata",
    "json",
    "np",
    "_E_CHARGE_C",
    "_detector_takeoff_acceptance_gain",
    "_electrons_from_beam_current",
    "_fft_convolve_centered",
    "_finite_nonnegative",
    "_gaussian_blur",
    "_gaussian_kernel_1d",
    "_gradient_components",
    "_gradient_magnitude",
    "_sha256_file",
    "_validate_takeoff_angle_deg",
]
