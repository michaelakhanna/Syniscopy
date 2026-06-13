"""Source-map convolution boundary contracts.

Finite source maps, guard-padded render canvases, and periodic tiles are
physically different objects even when they are all represented as ndarrays.
This module makes that boundary condition explicit before a PSF convolution can
produce detector-domain values for rendering or Fisher-facing direct products.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

SOURCE_CONVOLUTION_CONTRACT_ID = "syniscopy-source-convolution-boundary-v1"


class SourceConvolutionBoundaryMode(str, Enum):
    """Allowed boundary semantics for source-map PSF convolution."""

    LINEAR_ZERO_PADDED_SAME = "linear_zero_padded_same"
    PRE_CROP_GUARDED_FFT = "pre_crop_guarded_fft"
    EXPLICIT_PERIODIC_TILE = "explicit_periodic_tile"


def normalize_source_convolution_boundary_mode(value: Any) -> str:
    """Normalize and validate a source-convolution boundary mode."""

    raw = value.value if isinstance(value, SourceConvolutionBoundaryMode) else str(value).strip().lower()
    valid = {mode.value for mode in SourceConvolutionBoundaryMode}
    if raw not in valid:
        raise ValueError(
            f"unknown source-convolution boundary mode {raw!r}; expected one of {sorted(valid)!r}."
        )
    return raw


def _normalize_crop_slices(value: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("source convolution crop_slices must contain y and x ranges.")
    out: list[tuple[int, int]] = []
    for item in value:
        if len(item) != 2:
            raise ValueError("each source convolution crop slice must be a (start, end) pair.")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end <= start:
            raise ValueError(f"invalid source convolution crop slice {(start, end)!r}.")
        out.append((start, end))
    return (out[0], out[1])


@dataclass(frozen=True)
class SourceConvolutionContext:
    """Boundary semantics attached to a source map before PSF convolution.

    ``linear_zero_padded_same`` is the finite-FOV default for direct products.
    ``pre_crop_guarded_fft`` is allowed only for a render canvas whose wrapped
    region is outside a declared crop. ``explicit_periodic_tile`` is reserved for
    actual periodic samples, not as an FFT implementation shortcut.
    """

    boundary_mode: str = SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value
    source_extent_role: str = "finite_fov_source_map"
    producer: str = "unknown"
    guard_radius_pixels: int | None = None
    crop_slices: tuple[tuple[int, int], tuple[int, int]] | None = None
    periodic_declared: bool = False
    notes: str = ""
    contract_id: str = SOURCE_CONVOLUTION_CONTRACT_ID

    def __post_init__(self) -> None:
        mode = normalize_source_convolution_boundary_mode(self.boundary_mode)
        object.__setattr__(self, "boundary_mode", mode)
        crop = _normalize_crop_slices(self.crop_slices)
        object.__setattr__(self, "crop_slices", crop)
        if self.guard_radius_pixels is not None:
            guard = int(self.guard_radius_pixels)
            if guard < 0:
                raise ValueError("source convolution guard_radius_pixels must be non-negative.")
            object.__setattr__(self, "guard_radius_pixels", guard)
        if mode == SourceConvolutionBoundaryMode.PRE_CROP_GUARDED_FFT.value:
            if self.guard_radius_pixels is None or self.crop_slices is None:
                raise ValueError(
                    "pre_crop_guarded_fft requires guard_radius_pixels and crop_slices; "
                    "otherwise circular FFT wrapping can enter the saved field of view."
                )
        if mode == SourceConvolutionBoundaryMode.EXPLICIT_PERIODIC_TILE.value and not self.periodic_declared:
            raise ValueError(
                "explicit_periodic_tile requires periodic_declared=True so finite-FOV "
                "source maps cannot silently become periodic FFT domains."
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_convolution_contract_id": self.contract_id,
            "source_convolution_boundary_mode": self.boundary_mode,
            "source_extent_role": self.source_extent_role,
            "source_convolution_producer": self.producer,
            "source_convolution_guard_radius_pixels": self.guard_radius_pixels,
            "source_convolution_crop_slices": self.crop_slices,
            "source_convolution_periodic_declared": bool(self.periodic_declared),
            "source_convolution_notes": self.notes,
        }


def source_convolution_context_from(
    value: SourceConvolutionContext | Mapping[str, Any] | str | None,
    *,
    default_boundary_mode: str = SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value,
    source_extent_role: str = "finite_fov_source_map",
    producer: str = "unknown",
) -> SourceConvolutionContext:
    """Return a validated context from public/internal metadata forms."""

    if isinstance(value, SourceConvolutionContext):
        return value
    if value is None:
        return SourceConvolutionContext(
            boundary_mode=default_boundary_mode,
            source_extent_role=source_extent_role,
            producer=producer,
        )
    if isinstance(value, Mapping):
        return SourceConvolutionContext(
            boundary_mode=value.get("boundary_mode", value.get("source_convolution_boundary_mode", default_boundary_mode)),
            source_extent_role=value.get("source_extent_role", source_extent_role),
            producer=value.get("producer", value.get("source_convolution_producer", producer)),
            guard_radius_pixels=value.get("guard_radius_pixels", value.get("source_convolution_guard_radius_pixels")),
            crop_slices=value.get("crop_slices", value.get("source_convolution_crop_slices")),
            periodic_declared=bool(value.get("periodic_declared", value.get("source_convolution_periodic_declared", False))),
            notes=str(value.get("notes", value.get("source_convolution_notes", ""))),
        )
    return SourceConvolutionContext(
        boundary_mode=value,
        source_extent_role=source_extent_role,
        producer=producer,
    )


def _circular_convolve2d_fft(source: np.ndarray, psf: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=float)
    kernel = np.fft.ifftshift(np.asarray(psf, dtype=float))
    out = np.real(np.fft.ifft2(np.fft.fft2(src) * np.fft.fft2(kernel, s=src.shape)))
    return np.maximum(out, 0.0)


def _linear_zero_padded_convolve2d_same(source: np.ndarray, psf: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    if src.ndim != 2 or kernel.ndim != 2:
        raise ValueError("source-map PSF convolution requires 2D source and PSF arrays.")
    pad_shape = (src.shape[0] + kernel.shape[0] - 1, src.shape[1] + kernel.shape[1] - 1)
    full = np.real(
        np.fft.ifft2(
            np.fft.fft2(src, s=pad_shape) * np.fft.fft2(kernel, s=pad_shape)
        )
    )
    start_y = kernel.shape[0] // 2
    start_x = kernel.shape[1] // 2
    same = full[start_y:start_y + src.shape[0], start_x:start_x + src.shape[1]]
    return np.maximum(same, 0.0)


def convolve2d_with_source_boundary(
    source: np.ndarray,
    psf: np.ndarray,
    *,
    context: SourceConvolutionContext | Mapping[str, Any] | str | None,
    minimum_guard_radius_pixels: int | None = None,
) -> np.ndarray:
    """Convolve a 2D source map with a PSF under explicit boundary semantics."""

    ctx = source_convolution_context_from(context)
    src = np.asarray(source, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    if src.ndim != 2 or kernel.ndim != 2:
        raise ValueError("source-map PSF convolution requires 2D source and PSF arrays.")
    if ctx.boundary_mode == SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value:
        return _linear_zero_padded_convolve2d_same(src, kernel)
    if ctx.boundary_mode == SourceConvolutionBoundaryMode.PRE_CROP_GUARDED_FFT.value:
        if minimum_guard_radius_pixels is not None and ctx.guard_radius_pixels is not None:
            required = int(minimum_guard_radius_pixels)
            if ctx.guard_radius_pixels < required:
                raise ValueError(
                    "pre_crop_guarded_fft source convolution guard is too small: "
                    f"guard_radius_pixels={ctx.guard_radius_pixels}, required>={required}."
                )
        return _circular_convolve2d_fft(src, kernel)
    if ctx.boundary_mode == SourceConvolutionBoundaryMode.EXPLICIT_PERIODIC_TILE.value:
        return _circular_convolve2d_fft(src, kernel)
    raise AssertionError(f"unhandled source convolution boundary mode {ctx.boundary_mode!r}.")


__all__ = [
    "SOURCE_CONVOLUTION_CONTRACT_ID",
    "SourceConvolutionBoundaryMode",
    "SourceConvolutionContext",
    "convolve2d_with_source_boundary",
    "normalize_source_convolution_boundary_mode",
    "source_convolution_context_from",
]
