from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


pytestmark = [pytest.mark.full, pytest.mark.visual]


def _rot_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _nanorod_instance():
    from composite_shapes import particle, rod_stack
    from config import PARAMS, normalize_params
    from particle_model import build_particle_types_and_instances

    components = rod_stack(
        count=5,
        separation_nm=30.0,
        diameter_nm=20.0,
        material="Gold",
    )
    params = deepcopy(PARAMS)
    params["particles"] = [
        particle(
            name="verification_visual_nanorod",
            components=components,
            hydrodynamic_diameter_nm=150.0,
            initial_position_nm=[0.0, 0.0, 0.0],
        )
    ]
    params = normalize_params(params)
    trajectories_nm = np.zeros((1, 1, 3), dtype=float)
    orientations = np.eye(3, dtype=float)[None, None, :, :]
    _types, instances = build_particle_types_and_instances(
        params,
        trajectories_nm,
        psf_interpolators_by_type={},
        orientations=orientations,
        require_optical_psf=False,
    )
    return instances[0]


def _subparticle_positions_nm(instance, rotation_matrix: np.ndarray, translation_nm=(0.0, 0.0, 0.0)) -> np.ndarray:
    from rendering.per_particle_state import _iter_subparticle_render_info

    infos = _iter_subparticle_render_info(
        instance,
        np.asarray(translation_nm, dtype=float),
        rotation_matrix,
    )
    return np.asarray([np.asarray(info[0], dtype=float) for info in infos], dtype=float)


def _render_nanorod_image(
    instance,
    *,
    rotation_matrix: np.ndarray,
    translation_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_size: int = 401,
    pixel_size_nm: float = 5.0,
    sigma_px: float = 4.0,
    flux: float = 1.0,
    exposure_s: float = 1.0,
) -> np.ndarray:
    yy, xx = np.indices((image_size, image_size), dtype=float)
    centre = 0.5 * (image_size - 1.0)
    image = np.zeros((image_size, image_size), dtype=float)
    for world_pos_nm in _subparticle_positions_nm(instance, rotation_matrix, translation_nm):
        cx = centre + world_pos_nm[0] / pixel_size_nm
        cy = centre + world_pos_nm[1] / pixel_size_nm
        image += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px * sigma_px))
    return float(flux) * float(exposure_s) * image


def _disk_mean(image: np.ndarray, *, x_px: float, y_px: float, radius_px: float) -> float:
    yy, xx = np.indices(image.shape, dtype=float)
    mask = ((xx - float(x_px)) ** 2 + (yy - float(y_px)) ** 2) <= float(radius_px) ** 2
    return float(np.mean(np.asarray(image, dtype=float)[mask]))


def _to_pixel(pos_nm: np.ndarray, *, image_size: int, pixel_size_nm: float) -> tuple[float, float]:
    centre = 0.5 * (image_size - 1.0)
    return centre + float(pos_nm[0]) / pixel_size_nm, centre + float(pos_nm[1]) / pixel_size_nm


def test_reciprocal_shutter_visual_linearity() -> None:
    instance = _nanorod_instance()
    first = _render_nanorod_image(
        instance,
        rotation_matrix=_rot_z(0.0),
        flux=100.0,
        exposure_s=1.0,
    )
    second = _render_nanorod_image(
        instance,
        rotation_matrix=_rot_z(0.0),
        flux=200.0,
        exposure_s=0.5,
    )
    residual = second - first

    assert np.allclose(residual, 0.0, rtol=0.0, atol=1.0e-12), (
        "Reciprocal shutter test failed: doubling flux and halving exposure "
        "changed the structural image array."
    )


def test_visual_energy_is_conserved_under_rigid_rotation_and_in_frame_translation() -> None:
    instance = _nanorod_instance()
    totals = []
    for angle_deg in (0.0, 30.0, 75.0, 120.0, 180.0, 270.0, 360.0):
        for translation in ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (-100.0, 50.0, 0.0)):
            image = _render_nanorod_image(
                instance,
                rotation_matrix=_rot_z(np.deg2rad(angle_deg)),
                translation_nm=translation,
            )
            totals.append(float(np.sum(image)))

    totals_arr = np.asarray(totals, dtype=float)
    relative_spread = float((np.max(totals_arr) - np.min(totals_arr)) / np.mean(totals_arr))

    assert relative_spread <= 1.0e-8, (
        "Energy conservation invariance failed: rigid in-frame motion changed "
        f"the total image sum by relative_spread={relative_spread:.6g}."
    )


def test_nanorod_one_degree_visual_difference_is_tip_weighted_not_center_shifted() -> None:
    instance = _nanorod_instance()
    image_size = 401
    pixel_size_nm = 5.0
    image_0 = _render_nanorod_image(
        instance,
        rotation_matrix=_rot_z(0.0),
        image_size=image_size,
        pixel_size_nm=pixel_size_nm,
    )
    image_1 = _render_nanorod_image(
        instance,
        rotation_matrix=_rot_z(np.deg2rad(1.0)),
        image_size=image_size,
        pixel_size_nm=pixel_size_nm,
    )
    diff = np.abs(image_1 - image_0)

    positions_0 = _subparticle_positions_nm(instance, _rot_z(0.0))
    tip_left = positions_0[np.argmin(positions_0[:, 0])]
    tip_right = positions_0[np.argmax(positions_0[:, 0])]
    center = np.asarray([0.0, 0.0, 0.0], dtype=float)
    left_px = _to_pixel(tip_left, image_size=image_size, pixel_size_nm=pixel_size_nm)
    right_px = _to_pixel(tip_right, image_size=image_size, pixel_size_nm=pixel_size_nm)
    center_px = _to_pixel(center, image_size=image_size, pixel_size_nm=pixel_size_nm)

    tip_signal = 0.5 * (
        _disk_mean(diff, x_px=left_px[0], y_px=left_px[1], radius_px=4.0)
        + _disk_mean(diff, x_px=right_px[0], y_px=right_px[1], radius_px=4.0)
    )
    center_signal = _disk_mean(diff, x_px=center_px[0], y_px=center_px[1], radius_px=4.0)
    diff_center_of_mass = np.asarray(np.unravel_index(np.argmax(diff), diff.shape), dtype=float)
    frame_center_yx = np.asarray([0.5 * (image_size - 1.0), 0.5 * (image_size - 1.0)], dtype=float)

    assert tip_signal > 5.0 * max(center_signal, 1.0e-15), (
        "Moment-of-inertia gradient test failed: a 1 degree rod rotation did "
        "not produce tip-dominated visual change."
    )
    assert np.linalg.norm(diff_center_of_mass - frame_center_yx) > 6.0, (
        "Moment-of-inertia gradient test failed: the largest visual difference "
        "is centered near the rod COM instead of the tips."
    )


@pytest.mark.output
def test_blank_defocused_visual_has_zero_fim_and_dropped_supervision_mask() -> None:
    from config import PARAMS, normalize_params
    from fisher import compute_fisher_information
    from supervision_policy import SupervisionPolicy

    params = deepcopy(PARAMS)
    params.update(
        {
            "image_size_pixels": 32,
            "pixel_size_nm": 100.0,
            "supervision_support_factors": ["signal", "information"],
            "supervision_temporal_support_enabled": False,
            "supervision_signal_support_enabled": True,
            "supervision_information_support_enabled": True,
            "supervision_ambiguity_support_enabled": False,
            "supervision_decision_rule": "product",
            "supervision_supported_threshold": 0.2,
            "supervision_target": "mask_supported",
        }
    )
    params = normalize_params(params)
    blank = np.zeros((32, 32), dtype=float)
    geometry = np.zeros((32, 32), dtype=np.uint8)
    geometry[12:20, 12:20] = 255
    fim = compute_fisher_information(blank, 1.0, pixel_size_nm=100.0)

    policy = SupervisionPolicy(params, num_particles=1)
    out = policy.evaluate(
        particle_index=0,
        frame_index=0,
        position_nm=np.asarray([0.0, 0.0, 10_000.0]),
        contrast_image=blank,
        geometry_mask=geometry,
        noise_std=1.0,
        noise_variance_map=np.ones_like(blank),
    )

    assert np.allclose(fim, 0.0)
    assert np.count_nonzero(out["masks"]["mask_supported"]) == 0
    assert np.count_nonzero(out["masks"]["loss_weight"]) == 0
    assert np.array_equal(out["masks"]["ignore_mask"] > 0, geometry > 0)
    assert "low_signal" in out["record"]["drop_reason"]
    assert "fisher_singular" in out["record"]["drop_reason"]


@pytest.mark.output
def test_identical_overlapping_nanorods_zero_confidence_assignment_mask() -> None:
    from config import PARAMS, normalize_params
    from supervision_policy import SupervisionPolicy

    params = deepcopy(PARAMS)
    params.update(
        {
            "image_size_pixels": 32,
            "pixel_size_nm": 100.0,
            "supervision_support_factors": ["ambiguity"],
            "supervision_temporal_support_enabled": False,
            "supervision_signal_support_enabled": False,
            "supervision_information_support_enabled": False,
            "supervision_ambiguity_support_enabled": True,
            "supervision_decision_rule": "product",
            "supervision_supported_threshold": 0.8,
            "supervision_target": "mask_supported",
        }
    )
    params = normalize_params(params)
    geometry = np.zeros((32, 32), dtype=np.uint8)
    geometry[14:18, 8:24] = 255

    policy = SupervisionPolicy(params, num_particles=2)
    out = policy.evaluate(
        particle_index=0,
        frame_index=0,
        position_nm=np.asarray([0.0, 0.0, 0.0]),
        contrast_image=np.ones((32, 32), dtype=float),
        geometry_mask=geometry,
        all_positions_nm=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        all_geometry_masks=[geometry, geometry],
    )

    assert np.count_nonzero(out["masks"]["mask_supported"]) == 0
    assert np.count_nonzero(out["masks"]["loss_weight"]) == 0
    assert "ambiguous_assignment" in out["record"]["drop_reason"]
    assert out["record"]["mask_overlap_pixels"] == int(np.count_nonzero(geometry))

