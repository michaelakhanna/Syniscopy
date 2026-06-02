"""Compatibility wrapper for per-frame rendering orchestration."""

from __future__ import annotations

from particle_model import ParticleInstance

from .frame_loop import generate_video_and_masks as _generate_video_and_masks
from .frame_set import RenderedFrameSet


def generate_video_and_masks(params: dict, particle_instances: list[ParticleInstance]) -> RenderedFrameSet:
    """Generate video frames and masks using the focused frame-loop module."""
    return _generate_video_and_masks(params, particle_instances)


__all__ = ["RenderedFrameSet", "generate_video_and_masks"]
