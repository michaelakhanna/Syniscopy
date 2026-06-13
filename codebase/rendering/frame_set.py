"""Rendered frame sequence container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RenderedFrameSet:
    """
    Frame sequences produced by the renderer.

    signal/reference_frames are stochastic uint16 camera outputs.
    ideal_*_frames keep the model-scaled detector-input convention.
    detector_input_*_frames make that convention explicit for noise-variance
    propagation. detector_mean_*_frames are deterministic detector outputs
    after QE/offset/dark/static transfer but before stochastic noise.
    detector_object_field_frames stores the detector-grid complex coherent
    object/background field used by off-axis DHM demodulated Fisher to
    reconstruct the sideband into the shift-covariant complex field.
    analysis_noise_parameter_frames stores per-frame likelihood overlays, such
    as QPI detected-quanta maps, that cannot be reconstructed from display or
    phase frames after rendering.
    """

    signal_frames: list[np.ndarray]
    reference_frames: list[np.ndarray]
    ideal_signal_frames: list[np.ndarray]
    ideal_reference_frames: list[np.ndarray]
    detector_input_signal_frames: list[np.ndarray] = field(default_factory=list)
    detector_input_reference_frames: list[np.ndarray] = field(default_factory=list)
    detector_mean_signal_frames: list[np.ndarray] = field(default_factory=list)
    detector_mean_reference_frames: list[np.ndarray] = field(default_factory=list)
    detector_object_field_frames: list[np.ndarray] = field(default_factory=list)
    analysis_noise_parameter_frames: list[dict] = field(default_factory=list)
    rendered_trajectories_nm: np.ndarray | None = None
    mask_arrays: list[dict] = field(default_factory=list)
    supervision_records: list[dict] = field(default_factory=list)
    supervision_audit_summary: dict | None = None
    render_metadata: dict = field(default_factory=dict)


__all__ = ["RenderedFrameSet"]
