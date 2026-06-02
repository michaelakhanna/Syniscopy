"""Frame rendering package public API."""

from __future__ import annotations

from .airy_support import (
    estimate_optical_filter_guard_radius_pixels,
    estimate_psf_padding_radius_pixels,
)
from .canvas import resolve_render_canvas_geometry
from .frame_set import RenderedFrameSet
from .per_frame import generate_video_and_masks

__all__ = [
    "RenderedFrameSet",
    "estimate_optical_filter_guard_radius_pixels",
    "estimate_psf_padding_radius_pixels",
    "generate_video_and_masks",
    "resolve_render_canvas_geometry",
]
