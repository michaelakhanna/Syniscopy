from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


@dataclass(frozen=True)
class _MinimalOpticalComponent:
    component_geometry: Any
    world_position_nm: np.ndarray
    orientation_matrix: Any
    refractive_index: complex


def test_partitioning_distinguishes_near_and_far_pairs() -> None:
    import config
    from optical_cluster_scattering import partition_coupling_clusters

    params = config.default_params()
    from particle_specs import ParticleComponentSpec

    sphere = ParticleComponentSpec(shape="sphere", offset_nm=(0.0, 0.0, 0.0), diameter_nm=40.0)
    far_items = [
        _MinimalOpticalComponent(sphere, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(sphere, np.array([140.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
    ]

    near_items = [
        _MinimalOpticalComponent(sphere, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(
            sphere,
            np.array([45.0, 0.0, 0.0], dtype=float),
            None,
            1.59 + 0.0j,
        ),
    ]

    far_partition = partition_coupling_clusters(far_items, wavelength_nm=float(params["wavelength_nm"]))
    near_partition = partition_coupling_clusters(near_items, wavelength_nm=float(params["wavelength_nm"]))

    assert far_partition.significant_cluster_count == 0
    assert far_partition.clusters == ((0,), (1,))
    assert near_partition.significant_cluster_count == 1
    assert near_partition.clusters == ((0, 1),)


def test_near_field_overlap_is_rejected() -> None:
    import config
    from optical_cluster_scattering import partition_coupling_clusters
    from particle_specs import ParticleComponentSpec

    params = config.default_params()
    sphere = ParticleComponentSpec(shape="sphere", offset_nm=(0.0, 0.0, 0.0), diameter_nm=40.0)
    overlapping_items = [
        _MinimalOpticalComponent(sphere, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(sphere, np.array([35.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
    ]

    with pytest.raises(ValueError, match="intersect"):
        partition_coupling_clusters(overlapping_items, wavelength_nm=float(params["wavelength_nm"]))


def test_large_gap_cluster_returns_uncoupled_interaction_model() -> None:
    import config
    from optical_cluster_scattering import UNCOUPLED_CLUSTER_MODEL, coupled_cluster_scattering_result
    from particle_specs import ParticleComponentSpec

    params = config.default_params()
    sphere = ParticleComponentSpec(shape="sphere", offset_nm=(0.0, 0.0, 0.0), diameter_nm=40.0)
    far_items = [
        _MinimalOpticalComponent(sphere, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(sphere, np.array([140.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
    ]

    result = coupled_cluster_scattering_result(params, far_items)

    assert result.interaction_model == UNCOUPLED_CLUSTER_MODEL
    assert result.fidelity_level == "reference_validated_large_gap_limit"
    assert result.partition.significant_cluster_count == 0
    np.testing.assert_allclose(result.component_multipliers, (1.0 + 0.0j, 1.0 + 0.0j), rtol=1e-12, atol=0.0)


def test_near_field_override_independent_is_rejected() -> None:
    import config
    from optical_cluster_scattering import coupled_cluster_scattering_result
    from particle_specs import ParticleComponentSpec

    params = config.default_params()
    params["optical_cluster_scattering_model"] = "independent"

    sphere = ParticleComponentSpec(shape="sphere", offset_nm=(0.0, 0.0, 0.0), diameter_nm=40.0)
    near_items = [
        _MinimalOpticalComponent(sphere, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(sphere, np.array([45.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
    ]

    with pytest.raises(ValueError, match="independent.*near-field cluster"):
        coupled_cluster_scattering_result(params, near_items)


def test_nonspherical_near_cluster_uses_dipole_cluster_model() -> None:
    import config
    from optical_cluster_scattering import DISCRETE_DIPOLE_DDA_CLUSTER_MODEL, coupled_cluster_scattering_result
    from particle_specs import ParticleComponentSpec

    params = config.default_params()
    rod = ParticleComponentSpec(shape="spherocylinder", offset_nm=(0.0, 0.0, 0.0), diameter_nm=40.0, length_nm=120.0)
    rod_items = [
        _MinimalOpticalComponent(rod, np.array([0.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
        _MinimalOpticalComponent(rod, np.array([130.0, 0.0, 0.0], dtype=float), None, 1.59 + 0.0j),
    ]

    result = coupled_cluster_scattering_result(params, rod_items)

    assert result.interaction_model == DISCRETE_DIPOLE_DDA_CLUSTER_MODEL
    assert result.partition.significant_cluster_count == 1
    for multiplier in result.component_multipliers:
        assert np.isfinite(multiplier.real)
        assert np.isfinite(multiplier.imag)
        assert multiplier != 0.0 + 0.0j
