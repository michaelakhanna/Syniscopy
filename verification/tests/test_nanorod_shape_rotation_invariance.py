from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest


pytestmark = [pytest.mark.full, pytest.mark.shape]


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
            name="verification_nanorod_discrete_stack",
            components=components,
            hydrodynamic_diameter_nm=150.0,
            initial_position_nm=[0.0, 0.0, 0.0],
            symmetry_class="axisymmetric_discrete_rod_stack",
            continuous_rotational_symmetry_dim=1,
            singular_rotation_axes_body=["omega_x"],
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
    return components, instances[0]


def _render_component_stack_image(
    instance,
    rotation_matrix: np.ndarray,
    *,
    sigma_px: float,
    image_size: int,
    pixel_size_nm: float,
) -> np.ndarray:
    from rendering.per_particle_state import _iter_subparticle_render_info

    yy, xx = np.indices((image_size, image_size), dtype=float)
    centre = 0.5 * (image_size - 1.0)
    image = np.zeros((image_size, image_size), dtype=float)
    sub_infos = _iter_subparticle_render_info(
        instance,
        np.asarray([0.0, 0.0, 0.0], dtype=float),
        rotation_matrix,
    )
    for world_pos_nm, _interp, signal_mult, _source_mult, diameter_nm, _material in sub_infos:
        world_pos_nm = np.asarray(world_pos_nm, dtype=float)
        cx = centre + world_pos_nm[0] / pixel_size_nm
        cy = centre + world_pos_nm[1] / pixel_size_nm
        diameter_gain = float(diameter_nm) / 20.0
        amp = float(signal_mult) * diameter_gain
        image += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px * sigma_px))
    return image


def test_nanorod_generator_is_discrete_component_stack_not_analytic_mesh() -> None:
    components, instance = _nanorod_instance()

    assert len(components) == 5
    assert {component["shape"] for component in components} == {"sphere"}
    assert instance.particle_type.is_composite is True
    assert len(instance.particle_type.sub_particles) == 5
    offsets = np.asarray([sub.offset_nm for sub in instance.particle_type.sub_particles], dtype=float)
    assert np.allclose(offsets[:, 1:], 0.0)
    assert np.allclose(offsets[:, 0], [-60.0, -30.0, 0.0, 30.0, 60.0])


def test_nanorod_rigid_component_geometry_is_exact_through_360_degree_sweep() -> None:
    from rendering.per_particle_state import _iter_subparticle_render_info

    _components, instance = _nanorod_instance()
    base_infos = _iter_subparticle_render_info(
        instance,
        np.asarray([0.0, 0.0, 0.0], dtype=float),
        _rot_z(0.0),
    )
    base_positions = np.asarray([np.asarray(info[0], dtype=float) for info in base_infos])
    base_com = np.mean(base_positions, axis=0)
    base_radii = np.linalg.norm(base_positions - base_com, axis=1)
    base_pairwise = np.linalg.norm(
        base_positions[:, None, :] - base_positions[None, :, :],
        axis=2,
    )

    for angle_deg in np.arange(0.0, 360.0 + 5.0, 5.0):
        infos = _iter_subparticle_render_info(
            instance,
            np.asarray([0.0, 0.0, 0.0], dtype=float),
            _rot_z(np.deg2rad(angle_deg)),
        )
        positions = np.asarray([np.asarray(info[0], dtype=float) for info in infos])
        com = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions - com, axis=1)
        pairwise = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2,
        )

        assert np.allclose(com, base_com, rtol=0.0, atol=1.0e-12)
        assert np.allclose(radii, base_radii, rtol=0.0, atol=1.0e-12)
        assert np.allclose(pairwise, base_pairwise, rtol=0.0, atol=1.0e-12)


def test_nanorod_joint_fim_trace_has_only_bounded_pixel_grid_drift() -> None:
    from fisher import compute_fisher_information

    _components, instance = _nanorod_instance()
    image_size = 181
    pixel_size_nm = 10.0
    angle_grid_deg = np.arange(0.0, 360.0 + 5.0, 5.0)
    traces: list[float] = []

    for angle_deg in angle_grid_deg:
        R = _rot_z(np.deg2rad(angle_deg))
        candidate_a = _render_component_stack_image(
            instance,
            R,
            sigma_px=2.0,
            image_size=image_size,
            pixel_size_nm=pixel_size_nm,
        )
        candidate_b = _render_component_stack_image(
            instance,
            R,
            sigma_px=3.5,
            image_size=image_size,
            pixel_size_nm=pixel_size_nm,
        )
        joint_fim = (
            compute_fisher_information(candidate_a, 1.0, pixel_size_nm)
            + compute_fisher_information(candidate_b, 1.0, pixel_size_nm)
        )
        traces.append(float(np.trace(joint_fim)))

    traces_arr = np.asarray(traces, dtype=float)
    mean_trace = float(np.mean(traces_arr))
    min_index = int(np.argmin(traces_arr))
    max_index = int(np.argmax(traces_arr))
    relative_spread = float((np.max(traces_arr) - np.min(traces_arr)) / mean_trace)

    assert np.isclose(traces_arr[0], traces_arr[-1], rtol=0.0, atol=1.0e-12), (
        "Nanorod FIM trace does not return to its starting value after a full "
        "360 degree rotation, which indicates transform drift rather than "
        "ordinary finite-grid anisotropy."
    )
    assert relative_spread <= 2.0e-2, (
        "Nanorod joint FIM trace has excessive angle dependence across a 360 "
        "degree sweep. Some angle dependence is expected because this diagnostic "
        "samples a discrete sphere-stack rod on a Cartesian pixel grid and then "
        "uses finite-difference image gradients; this threshold is for large "
        "arbitrary precision spikes, not continuum rotational invariance. "
        f"relative_spread={relative_spread:.6g}, "
        f"min_trace={traces_arr[min_index]:.12g} at {angle_grid_deg[min_index]:.1f} deg, "
        f"max_trace={traces_arr[max_index]:.12g} at {angle_grid_deg[max_index]:.1f} deg."
    )
