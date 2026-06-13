from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


pytestmark = pytest.mark.quick


def test_overlapping_identical_particles_drop_supervision_by_ambiguity() -> None:
    from config import PARAMS, normalize_params
    from supervision_policy import SupervisionPolicy

    params = deepcopy(PARAMS)
    params.update(
        {
            "image_size_pixels": 16,
            "pixel_size_nm": 100.0,
            "supervision_support_factors": ["ambiguity"],
            "supervision_ambiguity_support_enabled": True,
            "supervision_temporal_support_enabled": False,
            "supervision_signal_support_enabled": False,
            "supervision_information_support_enabled": False,
            "supervision_decision_rule": "product",
            "supervision_supported_threshold": 0.8,
            "supervision_target": "mask_supported",
        }
    )
    params = normalize_params(params)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:11, 5:11] = 255

    policy = SupervisionPolicy(params, num_particles=2)
    out = policy.evaluate(
        particle_index=0,
        frame_index=0,
        position_nm=np.asarray([0.0, 0.0, 0.0]),
        contrast_image=np.ones((16, 16), dtype=float),
        geometry_mask=mask,
        all_positions_nm=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        all_geometry_masks=[mask, mask],
    )

    masks = out["masks"]
    record = out["record"]
    assert np.count_nonzero(masks["mask_geometry"]) > 0
    assert np.count_nonzero(masks["mask_supported"]) == 0
    assert np.array_equal(masks["ignore_mask"] > 0, masks["mask_geometry"] > 0)
    assert np.count_nonzero(masks["loss_weight"]) == 0
    assert "ambiguous_assignment" in record["drop_reason"]
    assert record["ambiguity_support"] < params["supervision_supported_threshold"]


def test_matched_packet_builder_rejects_decoupled_coordinate_frame_metadata() -> None:
    from matched_microscope_packets import build_matched_microscope_packet

    images = {
        "bright_field": np.ones((8, 8), dtype=float),
        "sem_secondary_electron": np.ones((8, 8), dtype=float),
    }
    with pytest.raises(ValueError, match="shared_coordinate_frame"):
        build_matched_microscope_packet(
            latent_state={"frame_index": 0},
            images_by_microscope=images,
            candidate_by_microscope={name: name for name in images},
            metadata={},
        )


def test_packet_fishers_must_be_psd_and_symmetric() -> None:
    from matched_microscope_packets import build_matched_microscope_packet

    images = {
        "bright_field": np.ones((8, 8), dtype=float),
        "sem_secondary_electron": np.ones((8, 8), dtype=float),
    }
    metadata = {
        "shared_coordinate_frame": {
            "axes": ["x_nm", "y_nm"],
            "pixel_size_nm": 100.0,
            "fisher_frame": "shared_xy_detector_frame",
        }
    }
    with pytest.raises(ValueError, match="positive semidefinite"):
        build_matched_microscope_packet(
            latent_state={"frame_index": 0},
            images_by_microscope=images,
            candidate_by_microscope={name: name for name in images},
            fisher_by_microscope={
                "bright_field": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=float),
                "sem_secondary_electron": np.eye(2),
            },
            metadata=metadata,
        )
