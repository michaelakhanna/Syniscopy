"""Rendered frame sequence container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    render_metadata: dict = field(default_factory=dict)


__all__ = ["RenderedFrameSet"]
