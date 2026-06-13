from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _temporal_only_params() -> dict:
    from config import default_params

    params = default_params()
    params.update(
        {
            "supervision_target": "mask_supported",
            "supervision_support_factors": ("temporal",),
            "supervision_temporal_support_enabled": True,
            "supervision_signal_support_enabled": False,
            "supervision_information_support_enabled": False,
            "supervision_ambiguity_support_enabled": False,
            "supervision_supported_threshold": 0.2,
        }
    )
    return params


def test_trackability_temporal_support_recomputes_after_implausible_step() -> None:
    from trackability import TrackabilityModel
    from trajectory import resolve_translational_diameters_nm

    params = _temporal_only_params()
    model = TrackabilityModel(
        params,
        num_particles=len(resolve_translational_diameters_nm(params)),
    )
    sigma_nm = float(model.r_sigma_nm[0])
    assert np.isfinite(sigma_nm) and sigma_nm > 0.0

    p0 = np.array([0.0, 0.0, 0.0], dtype=float)
    p1 = np.array([30.0 * sigma_nm, 0.0, 0.0], dtype=float)
    p2 = p1.copy()

    first = model.update_and_compute(0, 0, p0)
    implausible = model.update_and_compute(0, 1, p1)
    recovered = model.update_and_compute(0, 2, p2)

    assert first == 1.0
    assert implausible < 0.2
    assert recovered == 1.0


def test_supervision_temporal_mask_recovers_after_low_support_step() -> None:
    from supervision_policy import SupervisionPolicy
    from trajectory import resolve_translational_diameters_nm

    params = _temporal_only_params()
    policy = SupervisionPolicy(
        params,
        num_particles=len(resolve_translational_diameters_nm(params)),
    )
    sigma_nm = float(policy.temporal_model.r_sigma_nm[0])
    geom = np.ones((5, 5), dtype=np.uint8) * 255
    contrast = np.ones((5, 5), dtype=float)

    p0 = np.array([0.0, 0.0, 0.0], dtype=float)
    p1 = np.array([30.0 * sigma_nm, 0.0, 0.0], dtype=float)
    p2 = p1.copy()

    frame0 = policy.evaluate(
        particle_index=0,
        frame_index=0,
        position_nm=p0,
        contrast_image=contrast,
        geometry_mask=geom,
    )
    frame1 = policy.evaluate(
        particle_index=0,
        frame_index=1,
        position_nm=p1,
        contrast_image=contrast,
        geometry_mask=geom,
    )
    frame2 = policy.evaluate(
        particle_index=0,
        frame_index=2,
        position_nm=p2,
        contrast_image=contrast,
        geometry_mask=geom,
    )

    assert frame0["record"]["temporal_support"] == 1.0
    assert frame1["record"]["temporal_support"] < 0.2
    assert "implausible_brownian_step" in frame1["record"]["drop_reason"]
    assert np.count_nonzero(frame1["masks"]["mask_supported"]) == 0
    assert frame2["record"]["temporal_support"] == 1.0
    assert frame2["record"]["drop_reason"] == "supported"
    assert np.array_equal(frame2["masks"]["mask_supported"], geom)
