"""Interaction-volume projected-source SEM proxy backend."""

from __future__ import annotations

from ._metadata import _gaussian_blur, _gradient_components, np


class InteractionVolumeSEMProxyBackend:
    backend_mode = "interaction_volume_proxy"

    def __init__(
        self,
        *,
        probe_sigma_px: float,
        canvas_pitch_nm: float,
        interaction_volume_nm: float,
        detector_direction_xy: np.ndarray,
        edge_gain: float,
        bulk_gain: float,
        topography_gain: float,
        baseline: float,
    ) -> None:
        self.probe_sigma_px = float(probe_sigma_px)
        self.canvas_pitch_nm = float(canvas_pitch_nm)
        self.interaction_volume_nm = float(interaction_volume_nm)
        self.detector_direction_xy = np.asarray(detector_direction_xy, dtype=float)
        self.edge_gain = float(edge_gain)
        self.bulk_gain = float(bulk_gain)
        self.topography_gain = float(topography_gain)
        self.baseline = float(baseline)

    def _probe_blur(self, arr: np.ndarray) -> np.ndarray:
        return _gaussian_blur(np.asarray(arr, dtype=float), self.probe_sigma_px)

    def _interaction_blur(self, arr: np.ndarray) -> np.ndarray:
        if self.interaction_volume_nm <= 0.0:
            return self._probe_blur(arr)
        interaction_sigma_px = self.interaction_volume_nm / max(self.canvas_pitch_nm, 1e-12)
        sigma_px = float(np.sqrt(self.probe_sigma_px * self.probe_sigma_px + interaction_sigma_px * interaction_sigma_px))
        return _gaussian_blur(np.asarray(arr, dtype=float), sigma_px)

    def contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        source = np.maximum(np.asarray(source, dtype=float), 0.0)
        gx, gy = _gradient_components(source, self.canvas_pitch_nm)
        directed_edge = np.maximum(
            gx * self.detector_direction_xy[0] + gy * self.detector_direction_xy[1],
            0.0,
        )
        topography = np.sqrt(gx * gx + gy * gy)
        return np.maximum(
            self.bulk_gain * self._interaction_blur(source)
            + self.edge_gain * self._probe_blur(directed_edge)
            + self.topography_gain * self._probe_blur(topography),
            0.0,
        )

    def yield_from_source(self, source: np.ndarray, *, baseline: float | None = None) -> np.ndarray:
        base = self.baseline if baseline is None else float(baseline)
        return np.maximum(base + self.contrast_from_source(source), 0.0)


__all__ = ["InteractionVolumeSEMProxyBackend"]
