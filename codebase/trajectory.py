import numpy as np
from config import BOLTZMANN_CONSTANT
from particle_specs import (
    get_particle_specs,
    hydrodynamic_diameters_nm,
    initial_positions_from_specs_nm,
)
from substrate_pattern import (
    is_position_in_substrate_solid,
    project_position_to_fluid_region,
    reflect_position_across_substrate_boundary,
)

_INITIAL_POSITION_MAX_ATTEMPTS = 1000


def _positive_finite_param(params: dict, key: str) -> float:
    value = float(params[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"PARAMS['{key}'] must be finite and positive; got {value}.")
    return value


def resolve_num_frames(params: dict) -> int:
    """Resolve the positive frame count from ``num_frames`` or ``fps * duration_seconds``."""
    fps = _positive_finite_param(params, "fps")
    raw_num_frames = params.get("num_frames", None)
    if raw_num_frames is not None:
        if isinstance(raw_num_frames, bool):
            raise ValueError("PARAMS['num_frames'] must be a positive integer, not bool.")
        if isinstance(raw_num_frames, (float, np.floating)) and not float(raw_num_frames).is_integer():
            raise ValueError("PARAMS['num_frames'] must be an integer frame count.")
        try:
            num_frames = int(raw_num_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("PARAMS['num_frames'] must be a positive integer.") from exc
        if num_frames <= 0:
            raise ValueError("PARAMS['num_frames'] must be positive.")
        return num_frames

    duration_seconds = _positive_finite_param(params, "duration_seconds")
    num_frames = int(fps * duration_seconds)
    if num_frames <= 0:
        raise ValueError(
            "The product PARAMS['fps'] * PARAMS['duration_seconds'] must be "
            "positive to generate at least one frame."
        )
    return num_frames


def stokes_einstein_diffusion_coefficient(diameter_nm, temp_K, viscosity_Pa_s):
    """
    Calculate the diffusion coefficient for a spherical particle in a fluid
    using the Stokes–Einstein equation.

    Args:
        diameter_nm (float):
            Diameter of the particle in nanometers. This value is interpreted
            as a **translational equivalent diameter**
            when used for Brownian motion (i.e., the hydrodynamic diameter).
            Code that calls this function for translational diffusion must
            obtain diameters from the particle object's motion spec via
            resolve_translational_diameters_nm(params).
        temp_K (float): Absolute temperature of the fluid in Kelvin.
        viscosity_Pa_s (float): Dynamic viscosity of the fluid in Pascal-seconds.

    Returns:
        float: Diffusion coefficient in square meters per second (m^2/s).
    """
    radius_m = diameter_nm * 1e-9 / 2.0
    return (BOLTZMANN_CONSTANT * temp_K) / (6.0 * np.pi * viscosity_Pa_s * radius_m)

def resolve_translational_diameters_nm(params) -> np.ndarray:
    """
    Resolve the per-particle translational equivalent diameters (in nm) used
    for Brownian motion and any diffusion-based models such as trackability.

    Structural separation of optical vs translational size
    ------------------------------------------------------
    Each particle object owns a motion.hydrodynamic_diameter_nm value. This is
    conceptually separate from each renderable component's optical diameter.

    All components that depend on Brownian diffusion (e.g., simulate_trajectories,
    TrackabilityModel) must call this function and must read the hydrodynamic
    diameter from the particle object, not from renderable component size.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain:
                - "particles"

    Returns:
        np.ndarray: 1D float64 array of shape (num_particles,) containing the
        translational equivalent diameters in nanometers.

    Raises:
        ValueError: If provided arrays have inconsistent lengths or contain
            non-positive values.
    """
    diameters_nm = hydrodynamic_diameters_nm(params)

    if not np.all(diameters_nm > 0.0):
        raise ValueError("All particle motion hydrodynamic diameters must be positive.")

    return diameters_nm.astype(float)


def simulate_trajectories(params):
    """
    Simulate 3D Brownian motion trajectories for a set of particles.

    This implementation generates trajectories in a per-particle, per-frame
    update loop. The trajectory model is standard Stokes–Einstein Brownian
    motion, with two optional modifications:

        1. Substrate-pattern exclusion in the lateral (x, y) directions when a
           solid mounting-interface pattern is present.

        2. A configurable Z-axis motion constraint model that can enforce a
           non-penetrable planar surface.

    Lateral (x, y) substrate exclusion
    ----------------------------------
    When the substrate pattern configuration indicates a solid pattern, lateral
    positions whose projection lies in the solid region are not allowed.  The
    canonical PARAMS keys are ``sample_environment_pattern_enabled``,
    ``sample_environment_pattern`` and ``sample_environment_pattern_preset``.  The behavior is:

        - When ``sample_environment_pattern_enabled`` is False, or
          ``sample_environment_pattern_preset`` == "empty_background", or
          ``sample_environment_pattern`` == "none", the motion is fully
          unconstrained in x and y.

        - When a gold film with circular holes is enabled via
          ``sample_environment_pattern`` == "gold_holes" and
          ``sample_environment_pattern_preset`` == "default_gold_holes",
          lateral positions
          whose projection lies in the gold film are corrected after each
          Brownian step by the configured exclusion method. This enforces
          excluded volume without resampling steps.

        - When a nanopillar array is enabled via
          ``sample_environment_pattern`` == "nanopillars" and
          ``sample_environment_pattern_preset`` in {"nanopillars", "default_nanopillars"},
          lateral positions whose
          projection lies inside a nanopillar are deterministically mapped
          back into the nearest fluid region just outside the pillar boundary,
          again using project_position_to_fluid_region. This enforces excluded
          volume without introducing trapping or non-random motion.

    Z-axis motion constraint model
    ------------------------------
    The Z-axis behavior is controlled by PARAMS["z_motion_constraint_model"].

    Supported values and their semantics:

        - "unconstrained":
            Fully free 3D Brownian motion in z. The z-coordinate executes a
            standard random walk with no boundaries and no surface interaction.

        - "reflecting_floor_z0":
            Brownian motion in the half-space z >= 0 nm with a perfectly
            reflecting planar boundary at z = 0 nm. The interpretation is that
            z = 0 represents the sample-interface surface, and the particle
            center cannot cross into z < 0 (i.e., it cannot occupy the same
            space as the solid substrate in the axial direction). Any Brownian
            step that would place the particle at z < 0 is reflected across
            the plane.

            Initial z-positions for this model are sampled uniformly from the
            positive half of the configured initial z span when explicit
            positions are not provided, and user-specified initial positions
            must satisfy z >= 0 nm.

        - "reflecting_ceiling_z0":
            Brownian motion in the half-space z <= 0 nm with a perfectly
            reflecting planar boundary at z = 0 nm. The particle center lives
            below the plane and cannot cross into z > 0. Any Brownian step that
            would place the particle at z > 0 is reflected across the plane.

            Initial z-positions for this model are sampled uniformly from the
            negative half of the configured initial z span when explicit
            positions are not provided, and user-specified initial positions
            must satisfy z <= 0 nm.

    Translational equivalent diameter (separation from optical size)
    ----------------------------------------------------------------
    The translational diffusion coefficient for each particle is computed from
    a translational equivalent diameter:

        - Resolved via resolve_translational_diameters_nm(params).
        - Resolved from each particle object's motion.hydrodynamic_diameter_nm.
        - This value can differ from any component optical diameter without
          affecting optical PSF types or particle shapes.

    Composite particles can therefore have different optical component sizes
    and hydrodynamic motion size without coupling diffusion to rendered optics.

    Args:
        params (dict): Simulation parameter dictionary (PARAMS).

    Returns:
        numpy.ndarray: A 3D array of shape (num_particles, num_frames, 3)
            containing the [x, y, z] coordinates of each particle for each
            frame, in nanometers.
    """
    # --- Basic simulation timing and counts ---
    fps = _positive_finite_param(params, "fps")
    num_frames = resolve_num_frames(params)

    dt = 1.0 / fps
    particle_specs = get_particle_specs(params)
    num_particles = len(particle_specs)

    # --- Z-motion constraint model selection and validation ---
    z_model_raw = params.get("z_motion_constraint_model", "unconstrained")
    z_model_key = str(z_model_raw).strip().lower()

    if z_model_key == "unconstrained":
        z_model = "unconstrained"
    elif z_model_key == "reflecting_floor_z0":
        z_model = "reflecting_floor_z0"
    elif z_model_key == "reflecting_ceiling_z0":
        z_model = "reflecting_ceiling_z0"
    else:
        raise ValueError(
            f"Unsupported z_motion_constraint_model '{z_model_raw}'. "
            "Supported values are 'unconstrained', "
            "'reflecting_floor_z0', and 'reflecting_ceiling_z0'."
        )

    # Determine once per simulation whether to enforce lateral excluded volume
    # against solid regions of the configured substrate pattern.
    sample_environment_enabled = bool(params.get("sample_environment_enabled", True))
    substrate_enabled = bool(params.get("sample_environment_pattern_enabled", False))
    pattern_model_raw = params.get("sample_environment_pattern", "none")
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    apply_substrate_exclusion = sample_environment_enabled and substrate_enabled and (
        (
            pattern_model == "gold_holes"
            and substrate_preset == "default_gold_holes"
        )
        or (
            pattern_model == "nanopillars"
            and substrate_preset in ("nanopillars", "default_nanopillars")
        )
    )

    # Field-of-view extents used for sampling initial positions. These are
    # needed regardless of whether substrate exclusion is active.
    img_size_nm = float(params["image_size_pixels"]) * float(params["pixel_size_nm"])
    initial_z_span_nm = float(params["initial_z_span_nm"])
    if initial_z_span_nm <= 0.0:
        raise ValueError(
            "PARAMS['initial_z_span_nm'] must be positive."
        )

    # --- Initialize particle positions ---
    # Each particle object may provide its own initial position. Missing
    # positions are sampled independently from the same physical volume.
    explicit_initial_positions = initial_positions_from_specs_nm(params)
    initial_positions = np.empty((num_particles, 3), dtype=float)

    # --- Resolve translational equivalent diameters for diffusion and lateral
    # substrate clearance ---
    translational_diameters_nm = resolve_translational_diameters_nm(params)

    def _particle_clearance_nm(i: int) -> float:
        return 0.5 * float(translational_diameters_nm[i])

    def _validate_initial_position(i: int, x_nm: float, y_nm: float, z_nm: float) -> None:
        if apply_substrate_exclusion and is_position_in_substrate_solid(
            params,
            x_nm,
            y_nm,
            clearance_nm=_particle_clearance_nm(i),
        ):
            raise ValueError(
                f"Particle {i} initial position lies inside a solid sample-interface "
                "region or within one particle radius of it according to the "
                "current pattern configuration."
            )
        if z_model == "reflecting_floor_z0" and z_nm < 0.0:
            raise ValueError(
                f"Particle {i} initial z = {z_nm} nm is below the z = 0 surface "
                "for z_motion_constraint_model='reflecting_floor_z0'."
            )
        if z_model == "reflecting_ceiling_z0" and z_nm > 0.0:
            raise ValueError(
                f"Particle {i} initial z = {z_nm} nm is above the z = 0 surface "
                "for z_motion_constraint_model='reflecting_ceiling_z0'."
            )

    for i in range(num_particles):
        explicit_position = explicit_initial_positions[i]
        if explicit_position is not None:
            x_nm, y_nm, z_nm = (float(v) for v in explicit_position)
            _validate_initial_position(i, x_nm, y_nm, z_nm)
            initial_positions[i, :] = [x_nm, y_nm, z_nm]
            continue

        # Otherwise, sample x/y uniformly within the field of view and z
        # according to the selected z-motion model.
        if apply_substrate_exclusion:
            # Sample x, y until we find a position outside any solid region.
            # This implements free Brownian initial positions conditioned on
            # starting in the fluid region (holes or background between pillars).
            # The bounded retry count prevents dense or invalid pattern
            # geometries from hanging trajectory generation.
            max_attempts = _INITIAL_POSITION_MAX_ATTEMPTS
            for _ in range(max_attempts):
                x_nm = float(np.random.rand() * img_size_nm)
                y_nm = float(np.random.rand() * img_size_nm)
                if not is_position_in_substrate_solid(
                    params,
                    x_nm,
                    y_nm,
                    clearance_nm=_particle_clearance_nm(i),
                ):
                    initial_positions[i, 0] = x_nm
                    initial_positions[i, 1] = y_nm
                    break
            else:
                raise RuntimeError(
                    "Failed to sample a valid initial (x, y) position outside the "
                    "solid substrate-pattern region after many attempts. Please "
                    "verify the substrate pattern geometry parameters."
                )
        else:
            initial_positions[i, 0:2] = np.random.rand(2) * img_size_nm

        # Initialize z according to the selected z-motion model.
        if z_model == "unconstrained":
            # Symmetric distribution around z = 0. This is only an initial
            # sampling span, not a Brownian-motion confinement boundary.
            initial_positions[i, 2] = (float(np.random.rand()) - 0.5) * initial_z_span_nm
        elif z_model == "reflecting_floor_z0":
            # Start in the half-space z >= 0. For simplicity and consistency
            # with the PSF stack, sample z uniformly from the positive
            # half of the initial z span.
            initial_positions[i, 2] = float(np.random.rand()) * (initial_z_span_nm / 2.0)
        elif z_model == "reflecting_ceiling_z0":
            # Start in the half-space z <= 0 and sample z uniformly from the
            # negative half of the initial z span.
            initial_positions[i, 2] = -float(np.random.rand()) * (initial_z_span_nm / 2.0)
        else:
            # This should not be reachable due to earlier validation.
            raise RuntimeError(
                f"Unexpected z_motion_constraint_model '{z_model_raw}' encountered during initialization."
            )

    # --- Allocate trajectory array and set initial positions ---
    trajectories = np.zeros((num_particles, num_frames, 3), dtype=float)
    trajectories[:, 0, :] = initial_positions

    temp_K = _positive_finite_param(params, "temperature_K")
    viscosity_Pa_s = _positive_finite_param(params, "viscosity_Pa_s")

    # Loop over particles and generate their trajectories one time step at a time.
    for i in range(num_particles):
        diameter_nm = float(translational_diameters_nm[i])

        # Diffusion coefficient for this particle (m^2/s).
        D_m2_s = stokes_einstein_diffusion_coefficient(
            diameter_nm, temp_K, viscosity_Pa_s
        )

        # Standard deviation of displacement in each Cartesian dimension for
        # one time step, converted to nanometers.
        sigma_m = np.sqrt(2.0 * D_m2_s * dt)
        sigma_nm = float(sigma_m * 1e9)  # m -> nm

        # Generate the random walk over time.
        for frame_idx in range(1, num_frames):
            # Draw a 3D Brownian step [dx, dy, dz] in nanometers.
            step_nm = np.random.normal(loc=0.0, scale=sigma_nm, size=3)

            # Previous position at the last frame.
            prev_position_nm = trajectories[i, frame_idx - 1, :]

            # --- Lateral (x, y) update with optional substrate exclusion ---
            proposed_x_nm = float(prev_position_nm[0] + step_nm[0])
            proposed_y_nm = float(prev_position_nm[1] + step_nm[1])

            if apply_substrate_exclusion:
                # Hard-wall reflection is the physical boundary condition for
                # diffusion at an impenetrable feature. Projection is an
                # endpoint-clamping approximation and is kept as an explicit
                # alternative for users who want that geometry.
                exclusion_method = str(
                    params.get("sample_environment_exclusion_method", "reflection")
                ).strip().lower()
                if exclusion_method == "projection":
                    x_nm_new, y_nm_new = project_position_to_fluid_region(
                        params,
                        proposed_x_nm,
                        proposed_y_nm,
                        clearance_nm=_particle_clearance_nm(i),
                    )
                elif exclusion_method == "reflection":
                    x_nm_new, y_nm_new = reflect_position_across_substrate_boundary(
                        params,
                        float(prev_position_nm[0]),
                        float(prev_position_nm[1]),
                        proposed_x_nm,
                        proposed_y_nm,
                        clearance_nm=_particle_clearance_nm(i),
                    )
                else:
                    raise ValueError(
                        "PARAMS['sample_environment_exclusion_method'] must be either "
                        "'reflection' or 'projection'; got "
                        f"{params.get('sample_environment_exclusion_method')!r}."
                    )
            else:
                x_nm_new, y_nm_new = proposed_x_nm, proposed_y_nm

            # --- Z-axis update according to the chosen z-motion model ---
            prev_z_nm = float(prev_position_nm[2])
            dz_nm = float(step_nm[2])

            if z_model == "unconstrained":
                z_nm_new = prev_z_nm + dz_nm
            elif z_model == "reflecting_floor_z0":
                # Reflective boundary at z = 0 nm. If the proposed step would
                # cross into z < 0, reflect it across the plane so the particle
                # remains in the half-space z >= 0.
                z_candidate = prev_z_nm + dz_nm
                if z_candidate >= 0.0:
                    z_nm_new = z_candidate
                else:
                    z_nm_new = -z_candidate
            elif z_model == "reflecting_ceiling_z0":
                # Reflective boundary at z = 0 nm. If the proposed step would
                # cross into z > 0, reflect it across the plane so the particle
                # remains in the half-space z <= 0.
                z_candidate = prev_z_nm + dz_nm
                if z_candidate <= 0.0:
                    z_nm_new = z_candidate
                else:
                    z_nm_new = -z_candidate
            else:
                # This should not be reachable due to the earlier validation.
                raise RuntimeError(
                    f"Unexpected z_motion_constraint_model '{z_model_raw}' encountered during simulation."
                )

            trajectories[i, frame_idx, 0] = x_nm_new
            trajectories[i, frame_idx, 1] = y_nm_new
            trajectories[i, frame_idx, 2] = z_nm_new

    return trajectories


def _random_small_rotation_matrix(rng: np.random.Generator, std_angle_rad: float) -> np.ndarray:
    """
    Generate a random 3D rotation matrix from an isotropic Brownian increment.

    ``std_angle_rad`` is the per-component standard deviation of the rotation
    vector. For Stokes-Einstein rotational diffusion each component has variance
    ``2 * D_rot * dt``.

    Args:
        rng (np.random.Generator): Random number generator.
        std_angle_rad (float): Per-axis rotation-vector standard deviation in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    std_angle_rad = float(std_angle_rad)
    if std_angle_rad <= 0.0:
        # No rotation: identity.
        return np.eye(3, dtype=float)

    rotation_vector = rng.normal(loc=0.0, scale=std_angle_rad, size=3)
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 0.0:
        return np.eye(3, dtype=float)
    axis = rotation_vector / angle

    # Rodrigues' rotation formula.
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c

    R = np.array(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=float,
    )
    return R


def resolve_rotational_step_std_rad(params: dict, num_particles: int) -> np.ndarray:
    """
    Resolve the per-particle rotational step standard deviation in radians.

    Structural purpose
    ------------------
    This helper is the rotational analogue of resolve_translational_diameters_nm:
    it centralizes how the user-facing rotational configuration in PARAMS is
    turned into the per-axis rotation-vector step scale used by
    simulate_orientations.

    Supported semantics
    -------------------
    - ``rotational_diffusion_mode="empirical"`` reads a scalar
      ``rotational_step_std_deg`` as an RMS per-frame angle and converts it to
      an isotropic per-axis rotation-vector scale.

    - ``rotational_diffusion_mode="stokes_einstein"`` computes a per-particle
      per-axis rotation-vector standard deviation from Stokes-Einstein-Debye
      rotational diffusion using each particle object's hydrodynamic diameter,
      the configured temperature, viscosity, and frame interval.

    - If ``rotational_diffusion_enabled`` is false, orientation simulation is
      skipped and this resolver returns zeros.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS). May
            contain:
                - "rotational_diffusion_enabled"
                - "rotational_step_std_deg"
        num_particles (int): Number of particles for which step scales are
            required. Must be positive when rotational diffusion is enabled.

    Returns:
        np.ndarray: 1D float64 array of shape (num_particles,) with the per-
        particle standard deviation of the per-frame rotation angle in radians.
    """
    rotational_enabled = bool(params.get("rotational_diffusion_enabled", False))

    num_particles = int(num_particles)
    if num_particles <= 0:
        raise ValueError(
            "resolve_rotational_step_std_rad requires num_particles > 0, "
            f"got num_particles={num_particles}."
        )

    # Empirical scalar step standard deviation in degrees.
    step_std_deg = float(params.get("rotational_step_std_deg", 10.0))
    if not np.isfinite(step_std_deg) or step_std_deg < 0.0:
        raise ValueError(
            "PARAMS['rotational_step_std_deg'] must be finite and non-negative if provided."
        )

    # Convert to radians for the empirical mode.
    step_std_rad_scalar = float(np.deg2rad(step_std_deg))

    if not rotational_enabled:
        return np.zeros(num_particles, dtype=float)

    # --- Physics-derived mode (Stokes-Einstein-Debye) ----------------------
    #
    # PARAMS["rotational_diffusion_mode"] = "stokes_einstein" derives a
    # per-particle angular step standard deviation from the rotational diffusion
    # coefficient
    #
    #     D_rot = k_B T / (8 pi eta r^3),
    #
    # the per-frame time step dt = 1 / fps, and the Brownian relation
    #
    #     sigma_theta = sqrt(2 * D_rot * dt).
    #
    # Per-particle hydrodynamic diameters are resolved through
    # resolve_translational_diameters_nm so rotational and translational
    # diffusion use the same motion diameter.
    # "empirical" mode uses the configured scalar angular step.
    mode = str(params.get("rotational_diffusion_mode", "empirical")).lower()
    if mode == "stokes_einstein":
        temp_K = _positive_finite_param(params, "temperature_K")
        viscosity_Pa_s = _positive_finite_param(params, "viscosity_Pa_s")
        fps = _positive_finite_param(params, "fps")
        dt = 1.0 / fps

        diameters_nm = resolve_translational_diameters_nm(params)
        if diameters_nm.size != num_particles:
            if diameters_nm.size == 1:
                diameters_nm = np.full(num_particles, float(diameters_nm[0]))
            else:
                raise ValueError(
                    "Could not resolve a per-particle diameter list of length "
                    f"{num_particles} for rotational_diffusion_mode='stokes_einstein' "
                    f"(got {diameters_nm.size})."
                )

        # Vectorized closed-form: compute D_rot(d) for the full per-particle
        # diameter array in a single numpy expression, then sigma = sqrt(2*D*dt).
        # This is O(N) scalar work with zero Python-level iteration, which keeps
        # the cost negligible relative to a single frame's PSF render and avoids
        # paying a per-particle Python call in the hot setup path.
        diameters_m = np.asarray(diameters_nm, dtype=float) * 1.0e-9
        radius_m = 0.5 * diameters_m
        D_rot = (BOLTZMANN_CONSTANT * temp_K) / (
            8.0 * np.pi * viscosity_Pa_s * radius_m ** 3
        )
        return np.sqrt(2.0 * D_rot * dt)
    elif mode != "empirical":
        raise ValueError(
            f"Unknown PARAMS['rotational_diffusion_mode']={mode!r}; "
            "expected 'empirical' or 'stokes_einstein'."
        )

    return np.full(num_particles, step_std_rad_scalar / np.sqrt(3.0), dtype=float)


def simulate_orientations(params: dict, num_particles: int, num_frames: int) -> np.ndarray | None:
    """
    Simulate rotational Brownian motion (orientation trajectories) for a set
    of particles.

    This function provides the structural counterpart to simulate_trajectories
    for translation: it defines a per-particle, per-frame orientation timebase
    as a sequence of 3x3 rotation matrices. The representation is:

        orientations[i, t] -> 3x3 rotation matrix for particle i at frame t,

    where each matrix maps body-fixed coordinates into the lab frame.

    Orientation usage:
        - Spherical particles ignore orientation because their scalar PSF is
          radially symmetric.

        - Composite particles use these matrices during rendering to rotate
          component offsets before PSF placement.

    Configuration:
        - The model is controlled by the following optional PARAMS entries:

            "rotational_diffusion_enabled": bool
                Master switch. If False or absent, this function returns None
                and no orientations are simulated; ParticleInstance objects
                will then carry orientation_matrices=None.

            "rotational_step_std_deg": float
                Standard deviation (in degrees) of the per-frame rotation
                root-mean-square rotation angle. Typical values for small,
                smooth rotational Brownian motion are in the range 1–10
                degrees. Default: 10.0 degrees.

        - The Stokes-Einstein mode uses the frame interval dt = 1 / fps,
          matching the translational integration in simulate_trajectories.
          The empirical mode treats rotational_step_std_deg as the RMS
          per-frame angle and converts it to per-axis vector variance.

    RNG and reproducibility:
        - Rotational steps are driven by per-particle NumPy Generators whose
          seeds are drawn from the global np.random RNG. Since the dataset
          generator seeds np.random once per video, translational and
          rotational Brownian motion remain tied to the same per-video seed
          and are fully reproducible under that seeding scheme.

    Args:
        params (dict): Simulation parameter dictionary (PARAMS). Must contain
            "fps" when rotational_diffusion_enabled is True.
        num_particles (int): Number of particles being simulated.
        num_frames (int): Number of frames in the video.

    Returns:
        np.ndarray | None:
            - If rotational_diffusion_enabled is False (or not present),
              returns None.
            - Otherwise, returns a numpy array of shape
              (num_particles, num_frames, 3, 3) with dtype float, where each
              [i, t] entry is an SO(3) rotation matrix.
    """
    rotational_enabled = bool(params.get("rotational_diffusion_enabled", False))
    if not rotational_enabled:
        return None

    num_particles = int(num_particles)
    num_frames = int(num_frames)
    if num_particles <= 0 or num_frames <= 0:
        raise ValueError(
            "simulate_orientations requires positive num_particles and num_frames "
            f"(got num_particles={num_particles}, num_frames={num_frames})."
        )

    fps = float(params["fps"])
    if fps <= 0.0:
        raise ValueError("PARAMS['fps'] must be positive when simulating orientations.")

    # Resolve per-particle step standard deviations in radians.
    step_std_rad_per_particle = resolve_rotational_step_std_rad(params, num_particles)

    # Derive a deterministic set of per-particle seeds from the global
    # np.random RNG. The dataset generator seeds np.random once per video,
    # so drawing seeds here keeps rotational trajectories reproducible under
    # the same per-video seed used for translational trajectories and noise.
    #
    # We restrict seeds to a safe 32-bit range valid for default_rng.
    particle_seeds_int = np.random.randint(
        0,
        2**31,
        size=num_particles,
        dtype=np.int64,
    )

    # Allocate orientation array and initialize all particles to identity
    # orientation at frame 0.
    orientations = np.zeros((num_particles, num_frames, 3, 3), dtype=float)
    orientations[:, 0, :, :] = np.eye(3, dtype=float)

    # Perform a random walk on SO(3) for each particle using its own Generator
    # and its resolved per-particle angular step scale.
    #
    # Orientation matrices map body-fixed coordinates into the lab frame:
    # lab_vec = R_t @ body_vec. Body-frame Brownian increments compose by
    # right-multiplication, R_t = R_{t-1} @ R_step.
    for i in range(num_particles):
        rng_i = np.random.default_rng(int(particle_seeds_int[i]))
        std_rad_i = float(step_std_rad_per_particle[i])
        for t in range(1, num_frames):
            R_prev = orientations[i, t - 1]
            R_step = _random_small_rotation_matrix(rng_i, std_rad_i)
            orientations[i, t] = R_prev @ R_step

    return orientations
