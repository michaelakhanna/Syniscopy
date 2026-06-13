"""Gaussian-probe projected-source SEM proxy backend."""

from __future__ import annotations

from ._metadata import _gaussian_blur, _gradient_magnitude, np


class GaussianProbeSEMProxyBackend:
    backend_mode = "gaussian_probe_proxy"

    def __init__(
        self,
        *,
        probe_sigma_px: float,
        canvas_pitch_nm: float,
        edge_gain: float,
        bulk_gain: float,
        baseline: float,
    ) -> None:
        self.probe_sigma_px = float(probe_sigma_px)
        self.canvas_pitch_nm = float(canvas_pitch_nm)
        self.edge_gain = float(edge_gain)
        self.bulk_gain = float(bulk_gain)
        self.baseline = float(baseline)

    def _probe_blur(self, arr: np.ndarray) -> np.ndarray:
        return _gaussian_blur(np.asarray(arr, dtype=float), self.probe_sigma_px)

    def contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        source = np.maximum(np.asarray(source, dtype=float), 0.0)
        edge = _gradient_magnitude(source, self.canvas_pitch_nm)
        source_blur = self._probe_blur(source)
        edge_blur = self._probe_blur(edge)
        se = self.edge_gain * edge_blur + self.bulk_gain * source_blur
        return np.maximum(se, 0.0)

    def yield_from_source(self, source: np.ndarray, *, baseline: float | None = None) -> np.ndarray:
        base = self.baseline if baseline is None else float(baseline)
        return np.maximum(base + self.contrast_from_source(source), 0.0)


__all__ = ["GaussianProbeSEMProxyBackend"]
