from __future__ import annotations

import numpy as np
import pytest

from syniscopy_verification.analytic_se3 import (
    Pose,
    RigidProjector,
    analytic_se3_renders,
    featureless_sphere_renders,
    zero_signal_renders,
)


pytestmark = pytest.mark.quick


PIXEL_SIZE_NM = 100.0
Z_STEP_NM = 20.0
ROT_STEP_RAD = 1.0e-3


def _se3_crlb(renders: dict[str, np.ndarray]) -> dict:
    from fisher.se3 import compute_localization_orientation_crlb

    return compute_localization_orientation_crlb(
        renders,
        noise_variance_map=1.0,
        pixel_size_nm=PIXEL_SIZE_NM,
        z_step_nm=Z_STEP_NM,
        rotation_step_rad=ROT_STEP_RAD,
    )


def test_featureless_sphere_reports_rotational_rank_deficit() -> None:
    crlb = _se3_crlb(featureless_sphere_renders(z_step_nm=Z_STEP_NM, rotation_step_rad=ROT_STEP_RAD))

    assert crlb["singular"] is True
    assert set(crlb["axes_singular"]) == {"omega_x", "omega_y", "omega_z"}
    assert crlb["rank"] == 3
    assert crlb["sigma_omega_x_rad"] == float("inf")
    assert crlb["sigma_omega_y_rad"] == float("inf")
    assert crlb["sigma_omega_z_rad"] == float("inf")


def test_zero_signal_se3_fim_is_zero_and_all_axes_singular() -> None:
    crlb = _se3_crlb(zero_signal_renders())
    F = np.asarray(crlb["fisher_matrix"], dtype=float)

    assert np.allclose(F, 0.0)
    assert crlb["rank"] == 0
    assert set(crlb["axes_singular"]) == {"x", "y", "z", "omega_x", "omega_y", "omega_z"}


def test_fusion_corollary_combines_complementary_rank_deficient_modalities() -> None:
    from fisher import (
        FisherMatrixCandidate,
        compute_candidate_fusion_crlb_from_fisher_matrices,
    )

    crlb_a = _se3_crlb(
        analytic_se3_renders(
            observable_rotations={"rx", "ry"},
            z_step_nm=Z_STEP_NM,
            rotation_step_rad=ROT_STEP_RAD,
        )
    )
    crlb_b = _se3_crlb(
        analytic_se3_renders(
            observable_rotations={"ry", "rz"},
            z_step_nm=Z_STEP_NM,
            rotation_step_rad=ROT_STEP_RAD,
        )
    )
    F_a = np.asarray(crlb_a["fisher_matrix"], dtype=float)
    F_b = np.asarray(crlb_b["fisher_matrix"], dtype=float)

    assert "omega_z" in crlb_a["axes_singular"]
    assert "omega_x" in crlb_b["axes_singular"]
    assert crlb_a["rank"] == 5
    assert crlb_b["rank"] == 5

    fused = compute_candidate_fusion_crlb_from_fisher_matrices(
        [
            FisherMatrixCandidate(key="candidate_A", fisher_matrix=F_a),
            FisherMatrixCandidate(key="candidate_B", fisher_matrix=F_b),
        ]
    )
    F_joint = F_a + F_b
    eig = np.linalg.eigvalsh(0.5 * (F_joint + F_joint.T))

    assert np.allclose(fused["fusion_fisher"], F_joint)
    assert np.min(eig) > 0.0
    assert fused["fusion_rank"] == 6
    assert fused["fusion_singular"] is False
    assert np.all(np.isfinite(np.linalg.inv(F_joint)))


def test_se3_loop_invariance_returns_same_fim_after_closed_motion_loop() -> None:
    projector = RigidProjector(size=49, pixel_size_nm=PIXEL_SIZE_NM)
    start_pose = Pose.identity()
    end_pose = (
        start_pose
        .body_rotated("y", 2.0 * np.pi)
        .translated((0.0, 0.0, 10_000.0))
        .body_rotated("y", -2.0 * np.pi)
        .translated((0.0, 0.0, -10_000.0))
    )

    start = _se3_crlb(projector.se3_renders(start_pose, z_step_nm=Z_STEP_NM, rotation_step_rad=ROT_STEP_RAD))
    end = _se3_crlb(projector.se3_renders(end_pose, z_step_nm=Z_STEP_NM, rotation_step_rad=ROT_STEP_RAD))

    assert np.allclose(start["fisher_matrix"], end["fisher_matrix"], rtol=0.0, atol=1.0e-7)
    assert np.allclose(
        np.linalg.eigvalsh(start["fisher_matrix"]),
        np.linalg.eigvalsh(end["fisher_matrix"]),
        rtol=0.0,
        atol=1.0e-7,
    )
