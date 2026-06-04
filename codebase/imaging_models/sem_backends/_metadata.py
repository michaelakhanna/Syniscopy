from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend_fidelity import attach_backend_fidelity_metadata
# Single-source the elementary charge from the one electron-optics constants
# home instead of re-declaring the literal here (Phase 1: remove value shadows).
from imaging_models.electron_constants import (
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


def _gaussian_blur(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return np.asarray(arr, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(np.asarray(arr, dtype=float), sigma=sigma_px)
    except ImportError:
        kernel = _gaussian_kernel_1d(sigma_px)
        out = np.asarray(arr, dtype=float, copy=True)
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=0, arr=out)
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=out)
        return out


def _fft_convolve_centered(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular same-size convolution for centered, normalized kernels."""
    src = np.asarray(arr, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if ker.shape != src.shape:
        padded = np.zeros_like(src, dtype=float)
        h = min(src.shape[0], ker.shape[0])
        w = min(src.shape[1], ker.shape[1])
        sy = (ker.shape[0] - h) // 2
        sx = (ker.shape[1] - w) // 2
        dy = (src.shape[0] - h) // 2
        dx = (src.shape[1] - w) // 2
        padded[dy:dy + h, dx:dx + w] = ker[sy:sy + h, sx:sx + w]
        ker = padded
    centered_kernel = np.fft.ifftshift(ker)
    return np.real(np.fft.ifft2(np.fft.fft2(src) * np.fft.fft2(centered_kernel)))


def _gradient_components(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return gx, gy


def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gx, gy = _gradient_components(arr)
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
    "_electrons_from_beam_current",
    "_fft_convolve_centered",
    "_finite_nonnegative",
    "_gaussian_blur",
    "_gaussian_kernel_1d",
    "_gradient_components",
    "_gradient_magnitude",
    "_sha256_file",
]
