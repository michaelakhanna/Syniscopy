"""Array and clipping diagnostics for rendering."""

from __future__ import annotations

import numpy as np

def _array_diagnostics(arr: np.ndarray, *, prefix: str) -> dict[str, float]:
    values = np.asarray(arr, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_sum": float("nan"),
            f"{prefix}_nonzero_fraction": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_rms": float(np.sqrt(np.mean(finite * finite))),
        f"{prefix}_sum": float(np.sum(finite)),
        f"{prefix}_nonzero_fraction": float(np.count_nonzero(finite) / finite.size),
    }


def _clip_diagnostics(arr: np.ndarray, *, max_camera_count: int, prefix: str) -> dict[str, float]:
    values = np.asarray(arr, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_min_before_clip": float("nan"),
            f"{prefix}_max_before_clip": float("nan"),
            f"{prefix}_saturation_fraction": float("nan"),
            f"{prefix}_negative_fraction": float("nan"),
        }
    return {
        f"{prefix}_min_before_clip": float(np.min(finite)),
        f"{prefix}_max_before_clip": float(np.max(finite)),
        f"{prefix}_saturation_fraction": float(np.count_nonzero(finite > max_camera_count) / finite.size),
        f"{prefix}_negative_fraction": float(np.count_nonzero(finite < 0.0) / finite.size),
    }

