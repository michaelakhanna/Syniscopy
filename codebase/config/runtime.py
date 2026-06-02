"""Typed runtime views over validated simulation parameter dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderRuntimeConfig:
    """Core per-render settings consumed by the frame renderer."""

    fps: float
    random_seed: int | None
    image_size_pixels: int
    pixel_size_nm: float
    psf_oversampling_factor: int
    exposure_time_ms: float | None
    mask_generation_enabled: bool
    mask_output_directory: str
    bit_depth: int
    motion_blur_enabled: bool
    motion_blur_subsamples: int
    return_ideal_float_frames: bool
    reference_field_amplitude: float
    background_intensity: float

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "RenderRuntimeConfig":
        fps = float(params["fps"])
        if fps <= 0.0:
            raise ValueError("PARAMS['fps'] must be positive.")
        random_seed_raw = params.get("random_seed", None)
        random_seed = None if random_seed_raw is None else int(random_seed_raw)
        image_size_pixels = int(params["image_size_pixels"])
        if image_size_pixels <= 0:
            raise ValueError("PARAMS['image_size_pixels'] must be positive.")
        pixel_size_nm = float(params["pixel_size_nm"])
        if pixel_size_nm <= 0.0:
            raise ValueError("PARAMS['pixel_size_nm'] must be positive.")
        psf_oversampling_factor = int(params["psf_oversampling_factor"])
        if psf_oversampling_factor <= 0:
            raise ValueError("PARAMS['psf_oversampling_factor'] must be positive.")
        exposure_raw = params.get("exposure_time_ms")
        exposure_time_ms = None if exposure_raw is None else float(exposure_raw)
        bit_depth = int(params["bit_depth"])
        if bit_depth <= 0:
            raise ValueError("PARAMS['bit_depth'] must be positive.")
        motion_blur_subsamples = int(params["motion_blur_subsamples"])
        if motion_blur_subsamples <= 0:
            raise ValueError("PARAMS['motion_blur_subsamples'] must be positive.")
        reference_field_amplitude = float(params["reference_field_amplitude"])
        background_intensity = float(params["background_intensity"])

        return cls(
            fps=fps,
            random_seed=random_seed,
            image_size_pixels=image_size_pixels,
            pixel_size_nm=pixel_size_nm,
            psf_oversampling_factor=psf_oversampling_factor,
            exposure_time_ms=exposure_time_ms,
            mask_generation_enabled=bool(params.get("mask_generation_enabled", False)),
            mask_output_directory=str(params["mask_output_directory"]),
            bit_depth=bit_depth,
            motion_blur_enabled=bool(params["motion_blur_enabled"]),
            motion_blur_subsamples=motion_blur_subsamples,
            return_ideal_float_frames=bool(params.get("return_ideal_float_frames", False)),
            reference_field_amplitude=reference_field_amplitude,
            background_intensity=background_intensity,
        )


__all__ = ["RenderRuntimeConfig"]
