from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import builtins
import importlib
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for trajectory/substrate bootstrap; "
                f"missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _simulation_params() -> dict:
    from config import default_params

    params = deepcopy(default_params())
    params.update(
        {
            "temperature_K": 298.15,
            "viscosity_Pa_s": 1.0e-3,
            "fps": 1000.0,
            "num_frames": 1200,
            "random_seed": 2024,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "rotational_diffusion_enabled": False,
        }
    )

    base = deepcopy(params["particles"][0])
    base["motion"]["hydrodynamic_diameter_nm"] = 100.0
    base["motion"]["initial_position_nm"] = None
    params["particles"] = [base for _ in range(4)]
    return params


def _lateral_msd(pose: np.ndarray, *, num_lags: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(pose, dtype=float)[:, :, :2]
    max_lag = int(min(num_lags, lat.shape[1] - 1))
    lags = np.arange(1, max_lag + 1, dtype=int)
    msd = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        disp = lat[:, lag:] - lat[:, :-lag]
        msd[i] = float(np.mean(np.sum(disp * disp, axis=-1)))
    t = lags.astype(float) / float(fps)
    return t, msd


def _load_trajectory_module():
    _require_cv2_for_bootstrap()
    return importlib.import_module("trajectory")


def test_translational_msd_matches_stokes_einstein_free_diffusion() -> None:
    trajectory = _load_trajectory_module()

    params = _simulation_params()
    traj = trajectory.simulate_trajectories(params)
    traj = np.asarray(traj, dtype=float)
    assert traj.shape[1] == params["num_frames"]
    assert traj.shape[2] == 3

    t, msd = _lateral_msd(traj, num_lags=220, fps=params["fps"])
    # Use early-to-mid lags for linear-diffusion regime.
    fit = slice(1, 120)
    slope_nm2_per_s = float(np.polyfit(t[fit], msd[fit], 1)[0])

    d = params["particles"][0]["motion"]["hydrodynamic_diameter_nm"]
    expected_D_m2_s = trajectory.stokes_einstein_diffusion_coefficient(
        d,
        params["temperature_K"],
        params["viscosity_Pa_s"],
    )
    expected_slope = 4.0 * expected_D_m2_s * 1.0e18

    rel_error = abs(slope_nm2_per_s - expected_slope) / expected_slope
    assert rel_error < 0.30
