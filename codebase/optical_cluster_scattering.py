"""Cluster-level optical scattering geometry and coupled-dipole dispatch.

This module owns inter-component optical interaction. Per-component scattering
remains in :mod:`optical_scattering`; this layer decides when a set of
components is close enough that independent stamping is not the right physical
object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from config.runtime import OpticalInstrumentSettings, OpticalScatteringSettings
from optical_scattering import (
    analytic_polarizability_tensor_body_nm3,
    _coherent_incident_polarization,
    _detection_projection_vector,
)
from particle_specs import ParticleComponentSpec


COUPLED_DIPOLE_CLUSTER_MODEL = "coupled_dipole_quasistatic_cluster"
DISCRETE_DIPOLE_DDA_CLUSTER_MODEL = "discrete_dipole_dda_cluster"
MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL = "multi_sphere_t_matrix"
UNCOUPLED_CLUSTER_MODEL = "independent_component_superposition"
SINGLETON_CLUSTER_MODEL = "single_component"


@dataclass(frozen=True)
class ComponentGap:
    left_index: int
    right_index: int
    surface_gap_nm: float
    coupling_length_nm: float
    coupling_significant: bool


@dataclass(frozen=True)
class ClusterPartition:
    clusters: tuple[tuple[int, ...], ...]
    pair_gaps: tuple[ComponentGap, ...]
    minimum_surface_gap_nm: float | None
    coupling_length_nm: float
    significant_cluster_count: int


@dataclass(frozen=True)
class ClusterScatteringResult:
    component_multipliers: tuple[complex, ...]
    partition: ClusterPartition
    interaction_model: str
    fidelity_level: str
    approximation_label: str
    assumptions: tuple[str, ...]
    known_omissions: tuple[str, ...]


def _as_rotation(matrix: Any | None) -> np.ndarray:
    if matrix is None:
        return np.eye(3, dtype=float)
    out = np.asarray(matrix, dtype=float)
    if out.shape != (3, 3) or not np.all(np.isfinite(out)):
        raise ValueError("component orientation_matrix must be a finite 3x3 matrix.")
    return out


def _unit(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return d / n


def _support_body(component: ParticleComponentSpec, direction_body: np.ndarray) -> np.ndarray:
    d = _unit(direction_body)
    shape = component.shape_kind
    if shape == "sphere":
        return 0.5 * float(component.diameter_nm) * d
    if shape == "ellipsoid":
        axes = np.asarray(component.semi_axes_nm, dtype=float)
        weighted = axes * axes * d
        denom = float(np.sqrt(np.sum(axes * axes * d * d)))
        return weighted / max(denom, 1.0e-30)
    if shape == "cylinder":
        half_length = 0.5 * float(component.length_nm)
        radius = 0.5 * float(component.diameter_nm)
        radial = np.array([0.0, d[1], d[2]], dtype=float)
        radial_norm = float(np.linalg.norm(radial))
        out = np.array(
            [np.copysign(half_length, d[0]) if abs(d[0]) > 0.0 else 0.0, 0.0, 0.0],
            dtype=float,
        )
        if radial_norm > 0.0:
            out += radius * radial / radial_norm
        return out
    if shape == "spherocylinder":
        radius = 0.5 * float(component.diameter_nm)
        segment_half = 0.5 * max(float(component.length_nm) - float(component.diameter_nm), 0.0)
        return np.array(
            [np.copysign(segment_half, d[0]) if abs(d[0]) > 0.0 else 0.0, 0.0, 0.0],
            dtype=float,
        ) + radius * d
    if shape == "voxel_volume":
        if component.voxel_geometry is None:
            raise ValueError("voxel_volume component requires voxel_geometry.")
        centers, _weights = component.voxel_geometry.occupied_voxels()
        idx = int(np.argmax(centers @ d))
        half_box = 0.5 * float(component.voxel_geometry.voxel_size_nm) * np.sign(d)
        return centers[idx] + half_box
    raise ValueError(f"Unsupported component shape {component.shape!r}.")


def _support_world(
    component: ParticleComponentSpec,
    center_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
    direction_world: np.ndarray,
) -> np.ndarray:
    R = _as_rotation(orientation_matrix)
    d_world = _unit(direction_world)
    return (
        np.asarray(center_nm, dtype=float)
        + R @ _support_body(component, R.T @ d_world)
    )


def component_surface_gap_nm(
    left: ParticleComponentSpec,
    left_center_nm: Sequence[float],
    left_orientation_matrix: np.ndarray | None,
    right: ParticleComponentSpec,
    right_center_nm: Sequence[float],
    right_orientation_matrix: np.ndarray | None,
) -> float:
    """Surface-to-surface gap from support functions.

    Positive values are separation, zero is contact, and negative values mean
    overlap/intersection. Sphere-sphere is exact analytically; other analytic
    primitives use the convex support-function separation problem with bounded
    deterministic multi-start optimization.
    """

    left_center = np.asarray(left_center_nm, dtype=float)
    right_center = np.asarray(right_center_nm, dtype=float)
    if left_center.shape != (3,) or right_center.shape != (3,):
        raise ValueError("component centers must be length-3 vectors.")
    center_delta = right_center - left_center
    center_distance = float(np.linalg.norm(center_delta))
    if left.shape_kind == right.shape_kind == "sphere":
        return center_distance - 0.5 * float(left.diameter_nm) - 0.5 * float(right.diameter_nm)

    bounding_gap = center_distance - float(left.bounding_radius_nm) - float(right.bounding_radius_nm)
    starts = [
        _unit(center_delta),
        -_unit(center_delta),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    if bounding_gap > max(float(left.bounding_radius_nm), float(right.bounding_radius_nm), 1.0):
        starts = starts[:1]

    def separation(direction: np.ndarray) -> float:
        n = _unit(direction)
        left_support = _support_world(left, left_center, left_orientation_matrix, n)
        right_support = _support_world(right, right_center, right_orientation_matrix, -n)
        return float(np.dot(n, right_support - left_support))

    best = max(separation(start) for start in starts)

    def objective(x: np.ndarray) -> float:
        return -separation(x)

    constraint = {"type": "eq", "fun": lambda x: float(np.dot(x, x) - 1.0)}
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints=constraint,
            options={"maxiter": 80, "ftol": 1.0e-10, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            best = max(best, separation(result.x))
    return float(best)


def _coupling_length_nm(
    components: Sequence[Any],
    *,
    wavelength_nm: float,
) -> float:
    radii = [
        float(getattr(item.component_geometry, "bounding_radius_nm"))
        for item in components
    ]
    min_radius = min(radii) if radii else 0.0
    index_values = [complex(getattr(item, "refractive_index", 1.0 + 0.0j)) for item in components]
    plasmonic_or_lossy = any(abs(n.imag) > 0.05 or abs(n.real) > 2.0 for n in index_values)
    size_fraction = 0.60 if plasmonic_or_lossy else 0.35
    return float(max(1.0, min(0.20 * float(wavelength_nm), size_fraction * min_radius)))


def partition_coupling_clusters(
    components: Sequence[Any],
    *,
    wavelength_nm: float,
    overlap_tolerance_nm: float = 1.0e-6,
) -> ClusterPartition:
    n = len(components)
    if n <= 0:
        return ClusterPartition((), (), None, 0.0, 0)
    coupling_length = _coupling_length_nm(components, wavelength_nm=wavelength_nm)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    gaps: list[ComponentGap] = []
    minimum_gap = float("inf")
    for i, left in enumerate(components[:-1]):
        for j, right in enumerate(components[i + 1 :], start=i + 1):
            gap = component_surface_gap_nm(
                left.component_geometry,
                left.world_position_nm,
                left.orientation_matrix,
                right.component_geometry,
                right.world_position_nm,
                right.orientation_matrix,
            )
            if gap < -float(overlap_tolerance_nm):
                raise ValueError(
                    "Particle components overlap/intersect in optical geometry: "
                    f"component {i} and {j} have surface_gap_nm={gap:.6g}."
                )
            minimum_gap = min(minimum_gap, gap)
            significant = gap <= coupling_length
            gaps.append(
                ComponentGap(
                    left_index=i,
                    right_index=j,
                    surface_gap_nm=float(gap),
                    coupling_length_nm=float(coupling_length),
                    coupling_significant=bool(significant),
                )
            )
            if significant:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for idx in range(n):
        grouped.setdefault(find(idx), []).append(idx)
    clusters = tuple(tuple(group) for group in grouped.values())
    significant_clusters = sum(1 for cluster in clusters if len(cluster) > 1)
    return ClusterPartition(
        clusters=clusters,
        pair_gaps=tuple(gaps),
        minimum_surface_gap_nm=(None if minimum_gap == float("inf") else float(minimum_gap)),
        coupling_length_nm=float(coupling_length),
        significant_cluster_count=int(significant_clusters),
    )


def _alpha_world_for_component(params: dict, item: Any) -> np.ndarray:
    instrument = OpticalInstrumentSettings.from_params(params)
    geom = item.component_geometry
    n_particle = complex(getattr(item, "refractive_index", 1.0 + 0.0j))
    alpha_body = analytic_polarizability_tensor_body_nm3(
        component_geometry=geom,
        refractive_index=n_particle,
        medium_refractive_index=instrument.refractive_index_medium,
    )
    R = _as_rotation(getattr(item, "orientation_matrix", None))
    return R @ alpha_body @ R.T


def _project_dipole(params: dict, dipole: np.ndarray) -> complex:
    detector = _detection_projection_vector(params)
    if detector is None:
        return complex(np.linalg.norm(dipole))
    return complex(np.dot(detector.astype(np.complex128), dipole))


def _coupled_dipole_multipliers_for_cluster(
    params: dict,
    components: Sequence[Any],
    cluster: tuple[int, ...],
) -> dict[int, complex]:
    instrument = OpticalInstrumentSettings.from_params(params)
    k_medium = (
        2.0
        * np.pi
        * float(instrument.refractive_index_medium)
        / float(instrument.probe_wavelength_nm)
    )
    incident = _coherent_incident_polarization(params).astype(np.complex128)
    m = len(cluster)
    matrix = np.eye(3 * m, dtype=np.complex128)
    rhs = np.zeros(3 * m, dtype=np.complex128)
    alpha: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    isolated_projection: list[complex] = []
    for local_index, component_index in enumerate(cluster):
        item = components[component_index]
        alpha_i = _alpha_world_for_component(params, item)
        p_iso = alpha_i @ incident
        alpha.append(alpha_i)
        positions.append(np.asarray(item.world_position_nm, dtype=float))
        rhs[3 * local_index : 3 * local_index + 3] = p_iso
        isolated_projection.append(_project_dipole(params, p_iso))
    identity = np.eye(3, dtype=np.complex128)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            r = float(np.linalg.norm(r_vec))
            if not np.isfinite(r) or r <= 0.0:
                raise ValueError("Coupled optical scattering requires distinct component centers.")
            r_hat = r_vec / r
            rr = np.outer(r_hat, r_hat).astype(np.complex128)
            kr = k_medium * r
            phase = np.exp(1j * kr)
            green = phase * (
                ((k_medium * k_medium) / r) * (identity - rr)
                + ((1.0 / (r ** 3)) - 1j * k_medium / (r ** 2)) * (3.0 * rr - identity)
            )
            block = -alpha[i] @ green
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = block
    solution = np.linalg.solve(matrix, rhs).reshape(m, 3)
    out: dict[int, complex] = {}
    for local_index, component_index in enumerate(cluster):
        coupled_projection = _project_dipole(params, solution[local_index])
        iso = isolated_projection[local_index]
        if abs(iso) <= 1.0e-30:
            out[component_index] = 1.0 + 0.0j
        else:
            out[component_index] = complex(coupled_projection / iso)
    return out


def _component_contains_body(component: ParticleComponentSpec, points_body: np.ndarray) -> np.ndarray:
    shape = component.shape_kind
    p = np.asarray(points_body, dtype=float)
    if shape == "sphere":
        r = 0.5 * float(component.diameter_nm)
        return np.sum(p * p, axis=1) <= r * r
    if shape == "ellipsoid":
        axes = np.asarray(component.semi_axes_nm, dtype=float)
        return np.sum((p / axes[None, :]) ** 2, axis=1) <= 1.0
    if shape == "cylinder":
        half_length = 0.5 * float(component.length_nm)
        radius = 0.5 * float(component.diameter_nm)
        return (np.abs(p[:, 0]) <= half_length) & ((p[:, 1] ** 2 + p[:, 2] ** 2) <= radius * radius)
    if shape == "spherocylinder":
        radius = 0.5 * float(component.diameter_nm)
        segment_half = 0.5 * max(float(component.length_nm) - float(component.diameter_nm), 0.0)
        closest_x = np.clip(p[:, 0], -segment_half, segment_half)
        d2 = (p[:, 0] - closest_x) ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2
        return d2 <= radius * radius
    raise ValueError(f"Analytic DDA sampling does not support shape={shape!r}.")


def _analytic_component_dipoles_body(
    component: ParticleComponentSpec,
    *,
    spacing_nm: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius = float(component.bounding_radius_nm)
    spacing = float(spacing_nm)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("DDA spacing must be positive.")
    coords = np.arange(-radius, radius + 0.5 * spacing, spacing, dtype=float)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    mask = _component_contains_body(component, grid)
    points = grid[mask]
    if points.size == 0:
        points = np.zeros((1, 3), dtype=float)
    if points.shape[0] > int(max_points):
        step = int(np.ceil(points.shape[0] / int(max_points)))
        points = points[::step][: int(max_points)]
    volume_each = float(component.volume_nm3) / float(points.shape[0])
    return points, np.full(points.shape[0], volume_each, dtype=float)


def _component_dipoles_world(
    component: ParticleComponentSpec,
    center_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
    *,
    spacing_nm: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if component.shape_kind == "voxel_volume":
        if component.voxel_geometry is None:
            raise ValueError("voxel_volume component requires voxel_geometry.")
        body_points, weights = component.voxel_geometry.occupied_voxels()
        weights = np.asarray(weights, dtype=float) * float(component.voxel_geometry.voxel_volume_nm3)
        if body_points.shape[0] > int(max_points):
            step = int(np.ceil(body_points.shape[0] / int(max_points)))
            body_points = body_points[::step][: int(max_points)]
            weights = weights[::step][: int(max_points)]
            weights = weights * (float(component.volume_nm3) / max(float(np.sum(weights)), 1.0e-30))
    else:
        body_points, weights = _analytic_component_dipoles_body(
            component,
            spacing_nm=spacing_nm,
            max_points=max_points,
        )
    R = _as_rotation(orientation_matrix)
    world_points = np.asarray(center_nm, dtype=float)[None, :] + body_points @ R.T
    return world_points.astype(float, copy=False), np.asarray(weights, dtype=float)


def _cluster_dda_spacing_nm(
    settings: OpticalScatteringSettings,
    components: Sequence[Any],
    cluster: tuple[int, ...],
) -> float:
    if settings.dda_voxel_size_nm is not None:
        return float(settings.dda_voxel_size_nm)
    total_volume = sum(float(components[idx].component_geometry.volume_nm3) for idx in cluster)
    target = max(1, min(int(settings.dda_max_dipoles), 256))
    return float(max((total_volume / float(target)) ** (1.0 / 3.0), 0.5))


def _green_dyadic(k_medium: float, r_vec: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("DDA dipole positions must be distinct.")
    r_hat = np.asarray(r_vec, dtype=float) / r
    rr = np.outer(r_hat, r_hat).astype(np.complex128)
    identity = np.eye(3, dtype=np.complex128)
    kr = float(k_medium) * r
    phase = np.exp(1j * kr)
    return phase * (
        ((k_medium * k_medium) / r) * (identity - rr)
        + ((1.0 / (r ** 3)) - 1j * k_medium / (r ** 2)) * (3.0 * rr - identity)
    )


def _discrete_dipole_dda_multipliers_for_cluster(
    params: dict,
    components: Sequence[Any],
    cluster: tuple[int, ...],
) -> dict[int, complex]:
    settings = OpticalScatteringSettings.from_params(params)
    instrument = OpticalInstrumentSettings.from_params(params)
    k_medium = (
        2.0
        * np.pi
        * float(instrument.refractive_index_medium)
        / float(instrument.probe_wavelength_nm)
    )
    eps_medium = complex(float(instrument.refractive_index_medium)) ** 2
    spacing = _cluster_dda_spacing_nm(settings, components, cluster)
    max_total = int(settings.dda_max_dipoles)
    max_per_component = max(1, max_total // max(1, len(cluster)))
    incident = _coherent_incident_polarization(params).astype(np.complex128)

    dipole_positions: list[np.ndarray] = []
    dipole_alpha: list[complex] = []
    dipole_component_index: list[int] = []
    isolated_dipole_moment: list[np.ndarray] = []
    isolated_component_projection: dict[int, complex] = {idx: 0.0 + 0.0j for idx in cluster}
    for component_index in cluster:
        item = components[component_index]
        geom = item.component_geometry
        points, volumes = _component_dipoles_world(
            geom,
            np.asarray(item.world_position_nm, dtype=float),
            item.orientation_matrix,
            spacing_nm=spacing,
            max_points=max_per_component,
        )
        n_particle = complex(getattr(item, "refractive_index", 1.0 + 0.0j))
        eps_particle = n_particle * n_particle
        contrast = (eps_particle - eps_medium) / (eps_particle + 2.0 * eps_medium)
        for point, volume in zip(points, volumes):
            alpha = 3.0 * complex(volume) * eps_medium * contrast
            p_iso = alpha * incident
            dipole_positions.append(point)
            dipole_alpha.append(alpha)
            dipole_component_index.append(component_index)
            isolated_dipole_moment.append(p_iso)
            isolated_component_projection[component_index] += _project_dipole(params, p_iso)

    n_dipoles = len(dipole_positions)
    if n_dipoles <= 0:
        return {idx: 1.0 + 0.0j for idx in cluster}
    matrix = np.eye(3 * n_dipoles, dtype=np.complex128)
    rhs = np.zeros(3 * n_dipoles, dtype=np.complex128)
    for i, alpha in enumerate(dipole_alpha):
        rhs[3 * i : 3 * i + 3] = np.asarray(isolated_dipole_moment[i], dtype=np.complex128)
        for j in range(n_dipoles):
            if i == j:
                continue
            green = _green_dyadic(k_medium, dipole_positions[i] - dipole_positions[j])
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -complex(alpha) * green
    solution = np.linalg.solve(matrix, rhs).reshape(n_dipoles, 3)
    coupled_component_projection: dict[int, complex] = {idx: 0.0 + 0.0j for idx in cluster}
    for dipole_index, component_index in enumerate(dipole_component_index):
        coupled_component_projection[component_index] += _project_dipole(params, solution[dipole_index])
    out: dict[int, complex] = {}
    for component_index in cluster:
        iso = isolated_component_projection[component_index]
        out[component_index] = 1.0 + 0.0j if abs(iso) <= 1.0e-30 else complex(coupled_component_projection[component_index] / iso)
    return out


def _sphere_tmatrix_lmax(radius_nm: float, wavelength_nm: float, medium_index: float) -> int:
    x = 2.0 * np.pi * float(medium_index) * float(radius_nm) / float(wavelength_nm)
    return max(3, int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)))


def _plane_wave_swb_coefficients(basis: Any, *, k0: float, medium_index: float, poltype: str) -> np.ndarray:
    from treams import pw

    kvec = np.array([0.0, 0.0, float(k0) * float(medium_index)], dtype=float)
    coeffs = np.zeros(len(basis), dtype=np.complex128)
    # Use both transverse plane-wave polarizations with equal weight. The
    # renderer seam consumes a scalar amplitude correction; retaining both
    # polarizations avoids baking analyzer-specific choices into the exact
    # sphere-cluster interaction solve.
    for polpw in (0, 1):
        for idx, (pidx, ell, emm, polsw) in enumerate(
            zip(basis.pidx, basis.l, basis.m, basis.pol)
        ):
            value = pw.to_sw(
                int(ell),
                int(emm),
                int(polsw),
                kvec[0],
                kvec[1],
                kvec[2],
                int(polpw),
                poltype=poltype,
            )
            phase = np.exp(1j * float(np.dot(kvec, basis.positions[int(pidx)])))
            coeffs[idx] += complex(value) * complex(phase) / np.sqrt(2.0)
    return coeffs


def _multi_sphere_t_matrix_multipliers_for_cluster(
    params: dict,
    components: Sequence[Any],
    cluster: tuple[int, ...],
) -> dict[int, complex]:
    try:
        import treams
        from treams import Material
    except ImportError as exc:
        raise ValueError(
            "multi_sphere_t_matrix cluster scattering requires the optional "
            "'treams' package."
        ) from exc

    instrument = OpticalInstrumentSettings.from_params(params)
    wavelength_nm = float(instrument.probe_wavelength_nm)
    medium_index = float(instrument.refractive_index_medium)
    k0 = 2.0 * np.pi / wavelength_nm
    medium_material = Material(complex(medium_index) ** 2)
    tmats = []
    positions = []
    isolated_norms: dict[int, float] = {}
    for component_index in cluster:
        item = components[component_index]
        geom = item.component_geometry
        if geom.shape_kind != "sphere":
            raise ValueError("multi_sphere_t_matrix requires sphere components.")
        radius_nm = 0.5 * float(geom.diameter_nm)
        n_particle = complex(getattr(item, "refractive_index", 1.0 + 0.0j))
        lmax = _sphere_tmatrix_lmax(radius_nm, wavelength_nm, medium_index)
        tmat = treams.TMatrix.sphere(
            lmax,
            k0,
            radius_nm,
            [Material(n_particle * n_particle), medium_material],
            poltype="helicity",
        )
        tmats.append(tmat)
        positions.append(np.asarray(item.world_position_nm, dtype=float))
        incoming_single = _plane_wave_swb_coefficients(
            tmat.basis,
            k0=k0,
            medium_index=medium_index,
            poltype=tmat.poltype,
        )
        isolated = np.asarray(tmat @ incoming_single, dtype=np.complex128)
        isolated_norms[component_index] = float(np.linalg.norm(isolated))
    cluster_t = treams.TMatrix.cluster(tmats, positions)
    solved = cluster_t.interaction.solve()
    incoming_cluster = _plane_wave_swb_coefficients(
        solved.basis,
        k0=k0,
        medium_index=medium_index,
        poltype=solved.poltype,
    )
    outgoing = np.asarray(solved @ incoming_cluster, dtype=np.complex128)
    out: dict[int, complex] = {}
    for local_index, component_index in enumerate(cluster):
        mask = np.asarray(solved.basis.pidx) == int(local_index)
        coupled_norm = float(np.linalg.norm(outgoing[mask]))
        iso_norm = isolated_norms[component_index]
        out[component_index] = 1.0 + 0.0j if iso_norm <= 1.0e-30 else complex(coupled_norm / iso_norm)
    return out


def _cluster_solver_model(
    params: dict,
    components: Sequence[Any],
    cluster: tuple[int, ...],
) -> str:
    requested = OpticalScatteringSettings.from_params(params).cluster_model
    shapes = {str(components[idx].component_geometry.shape_kind) for idx in cluster}
    if requested == "independent":
        return UNCOUPLED_CLUSTER_MODEL
    if requested == "coupled_dipole":
        return COUPLED_DIPOLE_CLUSTER_MODEL
    if requested == "discrete_dipole_dda":
        return DISCRETE_DIPOLE_DDA_CLUSTER_MODEL
    if requested == "multi_sphere_t_matrix":
        if shapes != {"sphere"}:
            raise ValueError("optical_cluster_scattering_model='multi_sphere_t_matrix' requires all-sphere clusters.")
        return MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL
    if requested != "auto":
        raise ValueError(f"Unsupported optical_cluster_scattering_model={requested!r}.")
    if "voxel_volume" in shapes or len(shapes) > 1:
        return DISCRETE_DIPOLE_DDA_CLUSTER_MODEL
    if shapes == {"sphere"}:
        return MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL
    if shapes <= {"ellipsoid", "cylinder", "spherocylinder"}:
        return DISCRETE_DIPOLE_DDA_CLUSTER_MODEL
    return DISCRETE_DIPOLE_DDA_CLUSTER_MODEL


def coupled_cluster_scattering_result(
    params: dict,
    components: Sequence[Any],
) -> ClusterScatteringResult:
    if len(components) <= 1:
        return ClusterScatteringResult(
            component_multipliers=tuple(1.0 + 0.0j for _ in components),
            partition=partition_coupling_clusters(
                components,
                wavelength_nm=OpticalInstrumentSettings.from_params(params).probe_wavelength_nm,
            ),
            interaction_model=SINGLETON_CLUSTER_MODEL,
            fidelity_level="not_applicable",
            approximation_label="single_component_no_cluster_interaction",
            assumptions=(),
            known_omissions=(),
        )
    instrument = OpticalInstrumentSettings.from_params(params)
    partition = partition_coupling_clusters(
        components,
        wavelength_nm=instrument.probe_wavelength_nm,
    )
    multipliers = [1.0 + 0.0j for _ in components]
    if partition.significant_cluster_count == 0:
        return ClusterScatteringResult(
            component_multipliers=tuple(multipliers),
            partition=partition,
            interaction_model=UNCOUPLED_CLUSTER_MODEL,
            fidelity_level="reference_validated",
            approximation_label="large_gap_independent_component_limit",
            assumptions=("all inter-component gaps exceed the near-field coupling length",),
            known_omissions=(),
        )
    requested_cluster_model = OpticalScatteringSettings.from_params(params).cluster_model
    if requested_cluster_model == "independent":
        raise ValueError(
            "optical_cluster_scattering_model='independent' is invalid for a "
            "near-field cluster: at least one inter-component surface gap is "
            "within the optical coupling length."
        )
    active_models: set[str] = set()
    for cluster in partition.clusters:
        if len(cluster) <= 1:
            continue
        solver_model = _cluster_solver_model(params, components, cluster)
        active_models.add(solver_model)
        try:
            if solver_model == COUPLED_DIPOLE_CLUSTER_MODEL:
                cluster_multipliers = _coupled_dipole_multipliers_for_cluster(params, components, cluster)
            elif solver_model == DISCRETE_DIPOLE_DDA_CLUSTER_MODEL:
                cluster_multipliers = _discrete_dipole_dda_multipliers_for_cluster(params, components, cluster)
            elif solver_model == MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL:
                cluster_multipliers = _multi_sphere_t_matrix_multipliers_for_cluster(params, components, cluster)
            else:
                raise ValueError(f"Unsupported cluster scattering solver {solver_model!r}.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "Could not solve optical cluster scattering for "
                f"cluster {cluster!r} with model {solver_model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for idx, value in cluster_multipliers.items():
            multipliers[idx] = value
    if active_models == {MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL}:
        return ClusterScatteringResult(
            component_multipliers=tuple(multipliers),
            partition=partition,
            interaction_model=MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL,
            fidelity_level="reference_validated",
            approximation_label="multi_sphere_t_matrix_projected_to_component_amplitude",
            assumptions=(
                "all near-field clusters are solved with sphere T-matrices and cluster multiple scattering",
                "per-sphere outgoing coefficient block norms are projected to the renderer amplitude-correction seam",
            ),
            known_omissions=(
                "renderer uses per-component amplitude-corrected PSF stamping rather than a full cluster spherical-wave field image",
            ),
        )
    if active_models == {DISCRETE_DIPOLE_DDA_CLUSTER_MODEL}:
        return ClusterScatteringResult(
            component_multipliers=tuple(multipliers),
            partition=partition,
            interaction_model=DISCRETE_DIPOLE_DDA_CLUSTER_MODEL,
            fidelity_level="physics_based",
            approximation_label="discrete_dipole_volume_solver_component_amplitude_projection",
            assumptions=(
                "near-field clusters are discretized into material dipoles",
                "the coupled linear dipole system is solved per near-field cluster",
                "component field stamps receive cluster-coupled amplitude corrections",
            ),
            known_omissions=(
                "DDA convergence depends on optical_cluster_dda_voxel_size_nm and optical_cluster_dda_max_dipoles",
                "far-field angular redistribution is represented as component amplitude correction at the existing stamping seam",
            ),
        )
    return ClusterScatteringResult(
        component_multipliers=tuple(multipliers),
        partition=partition,
        interaction_model=COUPLED_DIPOLE_CLUSTER_MODEL,
        fidelity_level="physics_based",
        approximation_label="quasistatic_coupled_component_dipole",
        assumptions=(
            "cluster partition is determined by surface gap relative to optical coupling length",
            "component dipoles are solved in one linear coupled-dipole system per near-field cluster",
            "singletons and large-gap clusters use the existing per-component scattering path",
        ),
        known_omissions=(
            "not an exact multi-sphere T-matrix/GMM solver",
            "not a full DDA volume discretization for arbitrary imported geometry",
            "higher multipoles and retardation inside extended components are not represented",
        ),
    )


def static_component_interaction_metadata(
    params: dict,
    components: Sequence[ParticleComponentSpec],
) -> dict[str, Any]:
    """Static particle-type metadata from body-frame component geometry."""

    class _StaticItem:
        def __init__(self, component: ParticleComponentSpec):
            self.component_geometry = component
            self.world_position_nm = np.asarray(component.offset_nm, dtype=float)
            self.orientation_matrix = None
            self.refractive_index = complex(component.refractive_index or (1.0 + 0.0j))

    if len(components) <= 1:
        return {
            "optical_component_interaction_model": SINGLETON_CLUSTER_MODEL,
            "optical_component_interaction_fidelity_level": "not_applicable",
            "optical_component_interaction_approximation": "single_component_no_cluster_interaction",
            "component_count": int(len(components)),
            "minimum_component_surface_gap_nm": None,
            "optical_coupling_cluster_count": int(len(components)),
            "optical_coupling_significant_cluster_count": 0,
            "optical_component_interaction_assumptions": (),
            "optical_component_interaction_known_omissions": (),
        }
    items = tuple(_StaticItem(component) for component in components)
    partition = partition_coupling_clusters(
        items,
        wavelength_nm=OpticalInstrumentSettings.from_params(params).probe_wavelength_nm,
    )
    if partition.significant_cluster_count:
        requested = OpticalScatteringSettings.from_params(params).cluster_model
        significant_clusters = tuple(cluster for cluster in partition.clusters if len(cluster) > 1)
        static_model = _cluster_solver_model(params, items, significant_clusters[0])
        if static_model == COUPLED_DIPOLE_CLUSTER_MODEL:
            model = COUPLED_DIPOLE_CLUSTER_MODEL
            fidelity = "physics_based"
            approximation = "quasistatic_coupled_component_dipole"
            assumptions = (
                "near-field clusters are solved by coupled component dipoles at render time",
                "cluster membership is recomputed from exact frame-local oriented surface gaps",
            )
            omissions = (
                "not an exact multi-sphere T-matrix/GMM solver",
                "not a full arbitrary-geometry DDA/T-matrix implementation",
            )
        elif static_model == MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL:
            model = MULTI_SPHERE_T_MATRIX_CLUSTER_MODEL
            fidelity = "reference_validated"
            approximation = "multi_sphere_t_matrix_projected_to_component_amplitude"
            assumptions = (
                "near-field all-sphere clusters are solved by multi-sphere T-matrix at render time",
                "cluster membership is recomputed from exact frame-local oriented surface gaps",
            )
            omissions = (
                "renderer uses per-component amplitude-corrected PSF stamping rather than a full cluster spherical-wave field image",
            )
        else:
            model = DISCRETE_DIPOLE_DDA_CLUSTER_MODEL
            fidelity = "physics_based"
            approximation = "discrete_dipole_volume_solver_component_amplitude_projection"
            assumptions = (
                "near-field clusters are discretized into material dipoles at render time",
                "cluster membership is recomputed from exact frame-local oriented surface gaps",
            )
            omissions = (
                "DDA convergence depends on optical_cluster_dda_voxel_size_nm and optical_cluster_dda_max_dipoles",
                "far-field angular redistribution is represented as component amplitude correction at the existing stamping seam",
            )
    else:
        model = UNCOUPLED_CLUSTER_MODEL
        fidelity = "reference_validated"
        approximation = "large_gap_independent_component_limit"
        assumptions = ("all static body-frame component gaps exceed optical coupling length",)
        omissions = ()
    return {
        "optical_component_interaction_model": model,
        "optical_component_interaction_fidelity_level": fidelity,
        "optical_component_interaction_approximation": approximation,
        "component_count": int(len(components)),
        "minimum_component_surface_gap_nm": partition.minimum_surface_gap_nm,
        "optical_coupling_cluster_count": len(partition.clusters),
        "optical_coupling_significant_cluster_count": int(partition.significant_cluster_count),
        "optical_coupling_length_nm": float(partition.coupling_length_nm),
        "optical_component_interaction_assumptions": assumptions,
        "optical_component_interaction_known_omissions": omissions,
    }


__all__ = [
    "COUPLED_DIPOLE_CLUSTER_MODEL",
    "SINGLETON_CLUSTER_MODEL",
    "UNCOUPLED_CLUSTER_MODEL",
    "ClusterPartition",
    "ClusterScatteringResult",
    "ComponentGap",
    "component_surface_gap_nm",
    "coupled_cluster_scattering_result",
    "partition_coupling_clusters",
    "static_component_interaction_metadata",
]
