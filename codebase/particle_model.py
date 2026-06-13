from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from optics import ComplexPSFZInterpolator
from optical_cluster_scattering import static_component_interaction_metadata
from optical_scattering import (
    optical_scattering_key_for_component,
    optical_scattering_reference_diameter_from_key,
    optical_scattering_refractive_index_from_key,
)
from particle_material_resolution import (
    resolve_component_material_properties,
    resolve_component_refractive_index,
)
from particle_specs import get_particle_specs, ParticleComponentSpec, ParticleSpec
from substrate import MaterialProperties


class NullPSFZInterpolator:
    """Placeholder for source-map modalities that never consume optical PSFs."""

    z_min = 0.0
    z_max = 0.0

    def __call__(self, z_nm):
        raise RuntimeError(
            "This particle instance was built for a source-map imaging model "
            "and has no complex optical PSF stack."
        )


@dataclass(frozen=True)
class SubParticle:
    """
    Describes one primitive component within a rigid particle shape.

    Geometry:
        - offset_nm: 3D vector giving the body-fixed position of this
          sub-particle relative to the composite's reference point, in nm.

    Optics:
        - component_geometry: canonical primitive geometry used by the
          scattering-keyed PSF cache and render-time amplitude model.
        - diameter_nm: sphere diameter or geometry-derived reference diameter.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: coherent optical field interpolator for this
          component scattering type.
        - signal_multiplier: local optical-field amplitude scaling applied on top of the
          parent ParticleInstance.signal_multiplier.
        - source_multiplier: local material-source scaling used by
          source-map modalities; kept separate so optical amplitude knobs do
          not change TEM/SEM/fluorescence material density/yield.
    """
    offset_nm: np.ndarray
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: ComplexPSFZInterpolator
    signal_multiplier: float = 1.0
    source_multiplier: float = 1.0
    material_properties: MaterialProperties | None = None
    component_geometry: ParticleComponentSpec | None = None


@dataclass(frozen=True)
class ParticleType:
    """
    Describes one render component type in the simulation.

    Single-component case:
        - diameter_nm: physical sphere diameter or primitive reference
          diameter in nanometers.
        - refractive_index: complex refractive index (n + i k) for optical
          scattering types, neutral placeholder for source-only types.
        - ipsf_interpolator: coherent-field Z-interpolator for optical
          scattering types, or a NullPSFZInterpolator for source-only types.

    Composite case:
        - is_composite:
            False -> the particle is treated as a single primitive; the
                     renderer uses ipsf_interpolator directly.
            True  -> the particle is a rigid composite; the renderer ignores
                     ipsf_interpolator and instead loops over sub_particles.

        - sub_particles:
            Tuple of SubParticle objects describing the rigid internal geometry
            in a body-fixed frame. For single primitives this tuple is empty
            (is_composite=False). Multi-component assemblies are rendered
            through their component geometry and render keys.
    """
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: ComplexPSFZInterpolator

    is_composite: bool = False
    sub_particles: Tuple[SubParticle, ...] = ()
    geometry_symmetry_class: str | None = None
    geometry_continuous_rotational_stabilizer_dim: int | None = None
    geometry_singular_rotation_axes_body: tuple[str, ...] = ()
    optical_component_interaction_model: str = "single_component"
    optical_component_interaction_fidelity_level: str = "not_applicable"
    optical_component_interaction_approximation: str = "single_component_no_cluster_interaction"
    component_count: int = 1
    minimum_component_surface_gap_nm: float | None = None
    optical_coupling_cluster_count: int = 1
    optical_coupling_significant_cluster_count: int = 0
    optical_coupling_length_nm: float | None = None
    optical_component_interaction_assumptions: tuple[str, ...] = ()
    optical_component_interaction_known_omissions: tuple[str, ...] = ()

@dataclass
class ParticleInstance:
    """
    Represents a single particle instance in the simulation.

    Each instance:
        - References exactly one ParticleType (optical behavior and iPSF).
        - Stores its full 3D trajectory in nanometers over all frames.
        - Stores particle-level optical/source multipliers from ParticleSpec.
        - Stores the single render-component multipliers separately for
          single primitives, so the renderer applies component semantics in
          the same place for both single-component and composite particles.
        - Optionally stores a per-frame orientation for non-spherical or
          composite particles.

    Orientation representation:
        - orientation_matrices is either:
            * None (for orientation-invariant particles),
            * or a numpy array of shape (num_frames, 3, 3) where each 3x3
              matrix is a rotation mapping body-fixed coordinates into the
              lab/world frame at that frame index.

        - Rotations are used when:
            * configured parameters["rotational_diffusion_enabled"] is True, and
            * the particle has orientation-dependent geometry.
          Spheres ignore orientation; anisotropic primitive amplitudes and
          composite offsets consume it during rendering.
    """
    index: int
    particle_type: ParticleType
    trajectory_nm: np.ndarray
    signal_multiplier: float
    source_multiplier: float = 1.0
    component_signal_multiplier: float = 1.0
    component_source_multiplier: float = 1.0
    orientation_matrices: Optional[np.ndarray] = None
    material_properties: MaterialProperties | None = None
    component_geometry: ParticleComponentSpec | None = None


def build_particle_types_and_instances(
    params: dict,
    trajectories_nm: np.ndarray,
    psf_interpolators_by_type: Dict[Tuple[Any, ...], ComplexPSFZInterpolator],
    orientations: Optional[np.ndarray] = None,
    *,
    require_optical_psf: bool = True,
) -> Tuple[Dict[Tuple[Any, ...], ParticleType], List[ParticleInstance]]:
    """
    Construct ParticleType and ParticleInstance objects for the current
    simulation run.

    This helper maps canonical ParticleSpec objects, trajectories, and
    precomputed component field interpolators into render-ready ParticleInstance
    objects. A single-component particle becomes a single primitive
    ParticleType. A multi-component particle becomes a composite ParticleType
    whose sub-particles come directly from the particle object's components.

    Orientation handling:
        - If `orientations` is None, all ParticleInstance objects are created
          with orientation_matrices=None.
        - If `orientations` is provided, it must have shape
          (num_particles, num_frames, 3, 3). The i-th ParticleInstance then
          receives orientations[i] as its orientation_matrices. Composite
          particles use these matrices to rotate sub-particle offsets during
          rendering; anisotropic primitive scattering consumes the same
          orientation during stamping.

    Args:
        params (dict):
            Global parameter dictionary (parameters) for this simulation.
            Must contain "particles".
        trajectories_nm (np.ndarray):
            Particle trajectories with shape (num_particles, num_frames, 3),
            as returned by trajectory.simulate_trajectories.
        psf_interpolators_by_type (dict):
            Mapping from optical scattering key to the ComplexPSFZInterpolator
            computed for that scattering type in the latent scene builder.
        orientations (Optional[np.ndarray]):
            Optional orientation array with shape
            (num_particles, num_frames, 3, 3). When provided, each particle's
            orientation_matrices field is populated from this array. When
            None, orientation_matrices is left as None for all particles.

    Returns:
        tuple:
            - A dictionary mapping optical scattering key -> ParticleType.
            - A list of ParticleInstance objects of length num_particles.

    Raises:
        ValueError: If the lengths or shapes of the inputs are inconsistent
            with parameters["particles"] or the trajectory/orientation shapes.
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

    def _component_material_properties(component):
        return resolve_component_material_properties(
            params,
            component,
            require_optical_refractive_index=require_optical_psf,
        )

    def _component_refractive_index(component) -> complex:
        if require_optical_psf:
            return resolve_component_refractive_index(params, component)
        return 1.0 + 0.0j

    # Build render type objects from every component type collected upstream.
    # Optical modalities are keyed by optical scattering identity. Source-map
    # modalities do not consume optical PSFs and are keyed by component geometry
    # so SEM/TEM/fluorescence never depend on coherent-scattering validity.
    render_types: Dict[Tuple[Any, ...], ParticleType] = {}
    if require_optical_psf:
        for type_key, interpolator in psf_interpolators_by_type.items():
            diam_nm = optical_scattering_reference_diameter_from_key(type_key)
            n_complex = optical_scattering_refractive_index_from_key(type_key)
            render_types[type_key] = ParticleType(
                diameter_nm=float(diam_nm),
                refractive_index=n_complex,
                ipsf_interpolator=interpolator,
                is_composite=False,
                sub_particles=(),
            )

    particle_types: Dict[Tuple[Any, ...], ParticleType] = {}

    def _type_key_for_component(component) -> Tuple[Any, ...]:
        if not require_optical_psf:
            return component.source_geometry_key
        n_complex = _component_refractive_index(component)
        return optical_scattering_key_for_component(params, component, n_complex)

    def _material_key_for_component(component) -> Tuple[Any, ...]:
        props = component.material_properties or {}
        props_key = tuple(sorted((str(key), repr(value)) for key, value in props.items()))
        return (
            None if component.material is None else str(component.material),
            props_key,
        )

    def _optical_interaction_fields(
        components: tuple[ParticleComponentSpec, ...],
    ) -> dict[str, Any]:
        if not require_optical_psf:
            return {
                "optical_component_interaction_model": "source_map_geometry_only",
                "optical_component_interaction_fidelity_level": "not_applicable",
                "optical_component_interaction_approximation": "source_map_geometry_only",
                "component_count": int(len(components)),
                "minimum_component_surface_gap_nm": None,
                "optical_coupling_cluster_count": int(len(components)),
                "optical_coupling_significant_cluster_count": 0,
                "optical_coupling_length_nm": None,
                "optical_component_interaction_assumptions": (),
                "optical_component_interaction_known_omissions": (),
            }
        return static_component_interaction_metadata(params, components)

    def _particle_type_from_spec(spec: ParticleSpec, p_index: int) -> ParticleType:
        primary = spec.primary_component
        primary_key = _type_key_for_component(primary)
        if primary_key not in render_types:
            if require_optical_psf:
                raise ValueError(f"Missing optical PSF interpolator for scattering key {primary_key!r}.")
            render_types[primary_key] = ParticleType(
                diameter_nm=float(primary.diameter_nm),
                refractive_index=_component_refractive_index(primary),
                ipsf_interpolator=NullPSFZInterpolator(),
                is_composite=False,
                sub_particles=(),
            )
        if spec.is_single_sphere:
            if (
                spec.geometry_symmetry_class is None
                and spec.geometry_continuous_rotational_stabilizer_dim is None
                and not spec.geometry_singular_rotation_axes_body
            ):
                return render_types[primary_key]
            metadata_key = (
                "single_sphere_with_geometry_symmetry",
                primary_key,
                spec.geometry_symmetry_class,
                spec.geometry_continuous_rotational_stabilizer_dim,
                tuple(spec.geometry_singular_rotation_axes_body),
            )
            if metadata_key not in particle_types:
                base_type = render_types[primary_key]
                particle_types[metadata_key] = ParticleType(
                    diameter_nm=base_type.diameter_nm,
                    refractive_index=base_type.refractive_index,
                    ipsf_interpolator=base_type.ipsf_interpolator,
                    is_composite=False,
                    sub_particles=(),
                    geometry_symmetry_class=spec.geometry_symmetry_class,
                    geometry_continuous_rotational_stabilizer_dim=(
                        spec.geometry_continuous_rotational_stabilizer_dim
                    ),
                    geometry_singular_rotation_axes_body=tuple(
                        spec.geometry_singular_rotation_axes_body
                    ),
                    **_optical_interaction_fields(spec.components),
                )
            return particle_types[metadata_key]

        composite_key = (
            "composite",
            tuple(
                (
                    str(component.shape),
                    tuple(float(v) for v in component.offset_nm),
                    None if component.axes_nm is None else tuple(float(v) for v in component.axes_nm),
                    None if component.length_nm is None else float(component.length_nm),
                    *_type_key_for_component(component),
                    _material_key_for_component(component),
                    float(component.signal_multiplier),
                    float(component.source_multiplier),
                )
                for component in spec.components
            ),
            spec.geometry_symmetry_class,
            spec.geometry_continuous_rotational_stabilizer_dim,
            tuple(spec.geometry_singular_rotation_axes_body),
        )
        if composite_key in particle_types:
            return particle_types[composite_key]

        sub_particles: list[SubParticle] = []
        for c_idx, component in enumerate(spec.components):
            component_key = _type_key_for_component(component)
            if component_key not in render_types:
                if require_optical_psf:
                    raise ValueError(f"Missing optical PSF interpolator for scattering key {component_key!r}.")
                render_types[component_key] = ParticleType(
                    diameter_nm=float(component.diameter_nm),
                    refractive_index=_component_refractive_index(component),
                    ipsf_interpolator=NullPSFZInterpolator(),
                    is_composite=False,
                    sub_particles=(),
                )
            n_complex = _component_refractive_index(component)
            sub_particles.append(
                SubParticle(
                    offset_nm=np.asarray(component.offset_nm, dtype=float),
                    diameter_nm=float(component.diameter_nm),
                    refractive_index=n_complex,
                    ipsf_interpolator=render_types[component_key].ipsf_interpolator,
                    signal_multiplier=float(component.signal_multiplier),
                    source_multiplier=float(component.source_multiplier),
                    material_properties=_component_material_properties(component),
                    component_geometry=component,
                )
            )

        primary_type = render_types[primary_key]
        composite_type = ParticleType(
            diameter_nm=primary_type.diameter_nm,
            refractive_index=primary_type.refractive_index,
            ipsf_interpolator=primary_type.ipsf_interpolator,
            is_composite=True,
            sub_particles=tuple(sub_particles),
            geometry_symmetry_class=spec.geometry_symmetry_class,
            geometry_continuous_rotational_stabilizer_dim=(
                spec.geometry_continuous_rotational_stabilizer_dim
            ),
            geometry_singular_rotation_axes_body=tuple(
                spec.geometry_singular_rotation_axes_body
            ),
            **_optical_interaction_fields(spec.components),
        )
        particle_types[composite_key] = composite_type
        return composite_type

    # Build ParticleInstance objects, one per particle, referencing the
    # appropriate ParticleType (single primitive or composite) and its trajectory.
    instances: List[ParticleInstance] = []
    for i, spec in enumerate(particle_specs):
        ptype = _particle_type_from_spec(spec, i)

        if orientations is not None:
            orientation_matrices = orientations[i].copy()
        else:
            orientation_matrices = None

        component_signal_multiplier = 1.0
        component_source_multiplier = 1.0
        if spec.is_single_sphere and not ptype.is_composite:
            primary_component = spec.primary_component
            component_signal_multiplier = float(primary_component.signal_multiplier)
            component_source_multiplier = float(primary_component.source_multiplier)

        instance = ParticleInstance(
            index=i,
            particle_type=ptype,
            trajectory_nm=trajectories_nm[i],
            signal_multiplier=float(spec.signal_multiplier),
            source_multiplier=float(spec.source_multiplier),
            component_signal_multiplier=component_signal_multiplier,
            component_source_multiplier=component_source_multiplier,
            orientation_matrices=orientation_matrices,
            material_properties=_component_material_properties(spec.primary_component),
            component_geometry=spec.primary_component,
        )
        instances.append(instance)

    # The returned type dictionary contains the render component types required
    # by the renderer; composites reference those types through sub_particles.
    return render_types, instances
