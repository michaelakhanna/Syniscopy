from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from optics import ComplexPSFZInterpolator
from materials import (
    resolve_component_material_properties,
    resolve_component_refractive_index,
)
from particle_specs import get_particle_specs, ParticleSpec
from substrate import MaterialProperties


@dataclass(frozen=True)
class SubParticle:
    """
    Describes a spherical sub-particle within a (potentially non-spherical)
    composite particle shape.

    Geometry:
        - offset_nm: 3D vector giving the body-fixed position of this
          sub-particle relative to the composite's reference point, in nm.

    Optics:
        - diameter_nm: physical diameter of the sub-particle, in nm.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: spherical iPSF interpolator for this
          sub-particle type.
        - signal_multiplier: local amplitude scaling applied on top of the
          parent ParticleInstance.signal_multiplier.
    """
    offset_nm: np.ndarray
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: ComplexPSFZInterpolator
    signal_multiplier: float = 1.0
    material_properties: MaterialProperties | None = None


@dataclass(frozen=True)
class ParticleType:
    """
    Describes an optical "particle type" in the simulation.

    Spherical case:
        - diameter_nm: physical diameter in nanometers.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: spherical iPSF Z-interpolator defined on a
          type-specific z-grid.

    Composite case:
        - is_composite:
            False -> the particle is treated as a single sphere; the renderer
                     uses ipsf_interpolator directly.
            True  -> the particle is a rigid composite; the renderer ignores
                     ipsf_interpolator and instead loops over sub_particles.

        - sub_particles:
            Tuple of SubParticle objects describing the rigid internal
            geometry in a body-fixed frame. For spherical particles this
            tuple is empty (is_composite=False). For non-spherical particles
            it lists all spherical sub-components that will be positioned
            by orientation and translation.
    """
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: ComplexPSFZInterpolator

    is_composite: bool = False
    sub_particles: Tuple[SubParticle, ...] = ()
    symmetry_class: str | None = None
    continuous_rotational_symmetry_dim: int | None = None
    singular_rotation_axes_body: tuple[str, ...] = ()

    @property
    def type_key(self) -> Tuple[float, float, float]:
        """
        Return a tuple that uniquely identifies this particle type within the
        current simulation:

            (diameter_nm, n.real, n.imag)

        This matches the key used in main.run_simulation when grouping
        particles by type.
        """
        n = self.refractive_index
        return (self.diameter_nm, float(n.real), float(n.imag))


@dataclass
class ParticleInstance:
    """
    Represents a single particle instance in the simulation.

    Each instance:
        - References exactly one ParticleType (optical behavior and iPSF).
        - Stores its full 3D trajectory in nanometers over all frames.
        - Stores its per-particle signal multiplier (scalar amplitude factor).
        - Optionally stores a per-frame orientation for non-spherical
          composite particles.

    Orientation representation:
        - orientation_matrices is either:
            * None (for spherical particles),
            * or a numpy array of shape (num_frames, 3, 3) where each 3x3
              matrix is a rotation mapping body-fixed coordinates into the
              lab/world frame at that frame index.

        - Rotations are used when:
            * params["rotational_diffusion_enabled"] is True, and
            * the particle is composite (ptype.is_composite=True).
          Spherical particles ignore orientation; their PSF is radially
          symmetric.
    """
    index: int
    particle_type: ParticleType
    trajectory_nm: np.ndarray
    signal_multiplier: float
    orientation_matrices: Optional[np.ndarray] = None
    material_properties: MaterialProperties | None = None


def build_particle_types_and_instances(
    params: dict,
    trajectories_nm: np.ndarray,
    psf_interpolators_by_type: Dict[Tuple[float, float, float], ComplexPSFZInterpolator],
    orientations: Optional[np.ndarray] = None,
) -> Tuple[Dict[Tuple[float, float, float], ParticleType], List[ParticleInstance]]:
    """
    Construct ParticleType and ParticleInstance objects for the current
    simulation run.

    This helper maps canonical ParticleSpec objects, trajectories, and
    precomputed component iPSF interpolators into render-ready ParticleInstance
    objects. A single-component particle becomes a spherical ParticleType.
    A multi-component particle becomes a composite ParticleType whose
    sub-particles come directly from the particle object's components.

    Orientation handling:
        - If `orientations` is None, all ParticleInstance objects are created
          with orientation_matrices=None (spherical / orientation-ignored).
        - If `orientations` is provided, it must have shape
          (num_particles, num_frames, 3, 3). The i-th ParticleInstance then
          receives orientations[i] as its orientation_matrices. Composite
          particles use these matrices to rotate sub-particle offsets during
          rendering; spherical particles ignore them.

    Args:
        params (dict):
            Global parameter dictionary (PARAMS) for this simulation.
            Must contain "particles".
        trajectories_nm (np.ndarray):
            Particle trajectories with shape (num_particles, num_frames, 3),
            as returned by trajectory.simulate_trajectories.
        psf_interpolators_by_type (dict):
            Mapping from type_key = (diameter_nm, n_real, n_imag) to the
            ComplexPSFZInterpolator computed for that type in main.run_simulation.
        orientations (Optional[np.ndarray]):
            Optional orientation array with shape
            (num_particles, num_frames, 3, 3). When provided, each particle's
            orientation_matrices field is populated from this array. When
            None, orientation_matrices is left as None for all particles.

    Returns:
        tuple:
            - A dictionary mapping type_key -> ParticleType.
            - A list of ParticleInstance objects of length num_particles.

    Raises:
        ValueError: If the lengths or shapes of the inputs are inconsistent
            with PARAMS["particles"] or the trajectory/orientation shapes.
    """
    trajectories_nm = np.asarray(trajectories_nm, dtype=float)
    particle_specs = get_particle_specs(params)
    num_particles = len(particle_specs)

    if (
        trajectories_nm.ndim != 3
        or trajectories_nm.shape[0] != num_particles
        or trajectories_nm.shape[2] != 3
    ):
        raise ValueError(
            "trajectories_nm must have shape (num_particles, num_frames, 3). "
            f"Got {trajectories_nm.shape} for num_particles={num_particles}."
        )

    num_frames = trajectories_nm.shape[1]

    if orientations is not None:
        orientations = np.asarray(orientations, dtype=float)
        if orientations.shape != (num_particles, num_frames, 3, 3):
            raise ValueError(
                "orientations must have shape (num_particles, num_frames, 3, 3) "
                f"when provided. Got {orientations.shape}."
            )

    # Build spherical optical type objects from every component type that was
    # collected upstream. Composite particle types reference these same
    # component iPSF interpolators.
    spherical_types: Dict[Tuple[float, float, float], ParticleType] = {}
    for type_key, interpolator in psf_interpolators_by_type.items():
        diam_nm, n_real, n_imag = type_key
        n_complex = complex(n_real, n_imag)
        spherical_types[type_key] = ParticleType(
            diameter_nm=float(diam_nm),
            refractive_index=n_complex,
            ipsf_interpolator=interpolator,
            is_composite=False,
            sub_particles=(),
        )

    particle_types: Dict[Tuple[Any, ...], ParticleType] = {}

    def _type_key_for_component(component) -> Tuple[float, float, float]:
        n_complex = resolve_component_refractive_index(params, component)
        return (
            float(component.diameter_nm),
            float(n_complex.real),
            float(n_complex.imag),
        )

    def _particle_type_from_spec(spec: ParticleSpec, p_index: int) -> ParticleType:
        primary = spec.primary_component
        primary_key = _type_key_for_component(primary)
        if primary_key not in spherical_types:
            raise KeyError(
                f"No ParticleType found for particle {p_index} primary component "
                f"type_key={primary_key}."
            )
        if spec.is_single_sphere:
            if (
                spec.symmetry_class is None
                and spec.continuous_rotational_symmetry_dim is None
                and not spec.singular_rotation_axes_body
            ):
                return spherical_types[primary_key]
            metadata_key = (
                "single_sphere_with_symmetry",
                primary_key,
                spec.symmetry_class,
                spec.continuous_rotational_symmetry_dim,
                tuple(spec.singular_rotation_axes_body),
            )
            if metadata_key not in particle_types:
                base_type = spherical_types[primary_key]
                particle_types[metadata_key] = ParticleType(
                    diameter_nm=base_type.diameter_nm,
                    refractive_index=base_type.refractive_index,
                    ipsf_interpolator=base_type.ipsf_interpolator,
                    is_composite=False,
                    sub_particles=(),
                    symmetry_class=spec.symmetry_class,
                    continuous_rotational_symmetry_dim=spec.continuous_rotational_symmetry_dim,
                    singular_rotation_axes_body=tuple(spec.singular_rotation_axes_body),
                )
            return particle_types[metadata_key]

        composite_key = (
            "composite",
            tuple(
                (
                    tuple(float(v) for v in component.offset_nm),
                    *_type_key_for_component(component),
                    float(component.signal_multiplier),
                )
                for component in spec.components
            ),
            spec.symmetry_class,
            spec.continuous_rotational_symmetry_dim,
            tuple(spec.singular_rotation_axes_body),
        )
        if composite_key in particle_types:
            return particle_types[composite_key]

        sub_particles: list[SubParticle] = []
        for c_idx, component in enumerate(spec.components):
            component_key = _type_key_for_component(component)
            if component_key not in spherical_types:
                raise KeyError(
                    f"No ParticleType found for particle {p_index} component {c_idx} "
                    f"type_key={component_key}."
                )
            n_complex = resolve_component_refractive_index(params, component)
            sub_particles.append(
                SubParticle(
                    offset_nm=np.asarray(component.offset_nm, dtype=float),
                    diameter_nm=float(component.diameter_nm),
                    refractive_index=n_complex,
                    ipsf_interpolator=spherical_types[component_key].ipsf_interpolator,
                    signal_multiplier=float(component.signal_multiplier),
                    material_properties=resolve_component_material_properties(params, component),
                )
            )

        primary_type = spherical_types[primary_key]
        composite_type = ParticleType(
            diameter_nm=primary_type.diameter_nm,
            refractive_index=primary_type.refractive_index,
            ipsf_interpolator=primary_type.ipsf_interpolator,
            is_composite=True,
            sub_particles=tuple(sub_particles),
            symmetry_class=spec.symmetry_class,
            continuous_rotational_symmetry_dim=spec.continuous_rotational_symmetry_dim,
            singular_rotation_axes_body=tuple(spec.singular_rotation_axes_body),
        )
        particle_types[composite_key] = composite_type
        return composite_type

    # Build ParticleInstance objects, one per particle, referencing the
    # appropriate ParticleType (spherical or composite) and its trajectory.
    instances: List[ParticleInstance] = []
    for i, spec in enumerate(particle_specs):
        ptype = _particle_type_from_spec(spec, i)

        if orientations is not None:
            orientation_matrices = orientations[i].copy()
        else:
            orientation_matrices = None

        instance = ParticleInstance(
            index=i,
            particle_type=ptype,
            trajectory_nm=trajectories_nm[i],
            signal_multiplier=float(spec.signal_multiplier),
            orientation_matrices=orientation_matrices,
            material_properties=resolve_component_material_properties(params, spec.primary_component),
        )
        instances.append(instance)

    # The returned type dictionary contains the spherical optical types required
    # by the renderer; composites reference those types through sub_particles.
    return spherical_types, instances
