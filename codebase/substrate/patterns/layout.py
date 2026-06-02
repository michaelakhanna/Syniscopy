"""layout substrate-pattern helpers."""

from __future__ import annotations

import hashlib

from ._shared import (
    Dict,
    Optional,
    Tuple,
    _LAYOUT_CACHE,
    _LAYOUT_CACHE_LOCK,
    _MAX_SHAPE_AXIS_DISTORTION_FRAC,
    _MIN_EDGE_RADIUS_FACTOR,
    _MIN_SHAPE_RADIUS_FACTOR,
    _param_default,
    math,
    np,
)

class _LatticeFeature:
    """
    Represents a single substrate feature (hole or pillar) in pattern coordinates.

    Attributes:
        center_x_um (float): Center x-coordinate in micrometers.
        center_y_um (float): Center y-coordinate in micrometers.
        r_x_um (float): Semi-axis length along feature's local x-axis (µm).
        r_y_um (float): Semi-axis length along feature's local y-axis (µm).
        theta_rad (float): Orientation of the ellipse in radians. The generated
            layouts use axis-aligned ellipses and set theta_rad = 0.0.
        edge_perturbation_enabled (bool): Whether per-feature edge perturbation
            should be applied when computing the boundary radius.
        edge_modes (Optional[np.ndarray]): 1D array of integer mode indices k
            used in the angular perturbation series δ(θ).
        edge_amp (Optional[np.ndarray]): 1D array of float amplitudes A_k
            (dimensionless, relative to baseline radius).
        edge_phase (Optional[np.ndarray]): 1D array of float phase offsets φ_k
            in radians.
    """
    __slots__ = (
        "center_x_um",
        "center_y_um",
        "r_x_um",
        "r_y_um",
        "theta_rad",
        "edge_perturbation_enabled",
        "edge_modes",
        "edge_amp",
        "edge_phase",
    )

    def __init__(
        self,
        center_x_um: float,
        center_y_um: float,
        r_x_um: float,
        r_y_um: float,
        theta_rad: float = 0.0,
        edge_perturbation_enabled: bool = False,
        edge_modes: Optional[np.ndarray] = None,
        edge_amp: Optional[np.ndarray] = None,
        edge_phase: Optional[np.ndarray] = None,
    ) -> None:
        self.center_x_um = float(center_x_um)
        self.center_y_um = float(center_y_um)
        self.r_x_um = float(r_x_um)
        self.r_y_um = float(r_y_um)
        self.theta_rad = float(theta_rad)
        self.edge_perturbation_enabled = bool(edge_perturbation_enabled)
        self.edge_modes = edge_modes
        self.edge_amp = edge_amp
        self.edge_phase = edge_phase

class _FeatureLayout:
    """
    Single source of truth for substrate-feature geometry (holes or pillars).

    This layout is:
        - Computed once per simulation (per parameter set).
        - Used by optical background generation and Brownian geometry checks.

    Attributes:
        pattern_model (str): "gold_holes" or "nanopillars".
        pitch_um (float): Lattice pitch in micrometers.
        nominal_radius_um (float): Nominal feature radius before distortion.
        features_by_cell (dict): Mapping (i, j) -> _LatticeFeature.
        i_min, i_max, j_min, j_max (int): Lattice index bounds that cover the
            full field-of-view (with margin) for the current run.
        offset_x_um, offset_y_um (float): Global pattern offset in micrometers
            applied to the entire lattice. These are sampled once per layout
            build and ensure that the substrate pattern is globally shifted relative
            to the camera FOV while preserving periodic tiling.
    """

    __slots__ = (
        "pattern_model",
        "pitch_um",
        "nominal_radius_um",
        "features_by_cell",
        "i_min",
        "i_max",
        "j_min",
        "j_max",
        "offset_x_um",
        "offset_y_um",
    )

    def __init__(
        self,
        pattern_model: str,
        pitch_um: float,
        nominal_radius_um: float,
        features_by_cell: Dict[Tuple[int, int], _LatticeFeature],
        i_min: int,
        i_max: int,
        j_min: int,
        j_max: int,
        offset_x_um: float,
        offset_y_um: float,
    ) -> None:
        self.pattern_model = pattern_model
        self.pitch_um = float(pitch_um)
        self.nominal_radius_um = float(nominal_radius_um)
        self.features_by_cell = features_by_cell
        self.i_min = int(i_min)
        self.i_max = int(i_max)
        self.j_min = int(j_min)
        self.j_max = int(j_max)
        # Global lattice shift in pattern coordinates (µm). This is applied
        # uniformly to all nominal feature centers when the layout is built,
        # so all subsequent geometry queries see the same offset implicitly.
        self.offset_x_um = float(offset_x_um)
        self.offset_y_um = float(offset_y_um)

def _effective_layout_extent_nm(
    params: dict,
    layout_extent_nm: float | None,
) -> float:
    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    if layout_extent_nm is None:
        layout_extent_nm = img_size_pixels * pixel_size_nm
    internal_extent = params.get("_substrate_pattern_layout_extent_nm", None)
    if internal_extent is not None:
        layout_extent_nm = max(float(layout_extent_nm), float(internal_extent))
    layout_extent_nm = float(layout_extent_nm)
    if layout_extent_nm <= 0.0:
        raise ValueError("layout_extent_nm must be positive.")
    return layout_extent_nm

def clear_sample_environment_pattern_layout_cache() -> None:
    """Clear cached randomized sample-environment feature layouts."""
    with _LAYOUT_CACHE_LOCK:
        _LAYOUT_CACHE.clear()

def _get_randomization_settings(params: dict) -> Tuple[bool, float, float]:
    """
    Extract and validate substrate pattern randomization settings.

    Returns:
        sample_environment_pattern_randomization_enabled (bool),
        position_jitter_std_um (float),
        shape_regularity (float)
    """
    enabled = bool(params.get(
        "sample_environment_pattern_randomization_enabled",
        _param_default("sample_environment_pattern_randomization_enabled"),
    ))
    jitter_nm = float(params.get(
        "sample_environment_pattern_position_jitter_std_nm",
        _param_default("sample_environment_pattern_position_jitter_std_nm"),
    ))
    shape_reg = float(params.get(
        "sample_environment_pattern_shape_regularity",
        _param_default("sample_environment_pattern_shape_regularity"),
    ))

    if jitter_nm < 0.0:
        raise ValueError(
            "PARAMS['sample_environment_pattern_position_jitter_std_nm'] must be non-negative."
        )
    if not (0.0 <= shape_reg <= 1.0):
        raise ValueError(
            "PARAMS['sample_environment_pattern_shape_regularity'] must be in the interval [0, 1]."
        )

    # Convert to micrometers for internal use.
    jitter_um = jitter_nm * 1e-3
    return enabled, jitter_um, shape_reg

def _get_edge_perturbation_settings(params: dict) -> Tuple[float, int]:
    """
    Extract and validate global edge perturbation settings for substrate features.

    Returns:
        max_rel_radius (float): Maximum relative radial deviation (delta_max).
        mode_count (int): Number of angular modes K used in the perturbation.
    """
    max_rel = float(params.get(
        "sample_environment_pattern_edge_perturbation_max_rel_radius",
        _param_default("sample_environment_pattern_edge_perturbation_max_rel_radius"),
    ))
    mode_count = int(params.get(
        "sample_environment_pattern_edge_perturbation_mode_count",
        _param_default("sample_environment_pattern_edge_perturbation_mode_count"),
    ))

    if max_rel < 0.0:
        raise ValueError(
            "PARAMS['sample_environment_pattern_edge_perturbation_max_rel_radius'] must be non-negative."
        )
    if mode_count < 0:
        raise ValueError(
            "PARAMS['sample_environment_pattern_edge_perturbation_mode_count'] must be non-negative."
        )

    return max_rel, mode_count

def _compute_lattice_bounds(
    img_size_pixels: int,
    pixel_size_nm: float,
    pitch_um: float,
    extent_nm: float | None = None,
) -> Tuple[int, int, int, int]:
    """
    Determine the lattice index bounds (i_min, i_max, j_min, j_max) that cover
    the requested physical extent (and a margin) in pattern coordinates.

    By default we treat the FOV as a square of side length:
        L_nm = img_size_pixels * pixel_size_nm
        L_um = L_nm * 1e-3

    A caller can pass extent_nm when the optical model operates on a larger
    padded canvas before the detector crop.

    We then compute the min/max lattice indices whose nominal centers fall
    within [-L_um/2 - margin, L_um/2 + margin] in both x and y.

    A small margin of one lattice period is used so that modest jitter and the
    global lattice offset cannot produce features that affect the FOV but fall
    outside the bounds. The bounds are computed for an unshifted grid; the
    global offset is applied later when building feature centers.
    """
    img_size_pixels = int(img_size_pixels)
    pixel_size_nm = float(pixel_size_nm)
    pitch_um = float(pitch_um)

    if img_size_pixels <= 0 or pixel_size_nm <= 0.0 or pitch_um <= 0.0:
        raise ValueError(
            "Image size, pixel_size_nm, and pitch_um must be positive."
        )

    if extent_nm is None:
        L_nm = img_size_pixels * pixel_size_nm
    else:
        L_nm = float(extent_nm)
        if L_nm <= 0.0:
            raise ValueError("extent_nm must be positive when provided.")
    # Physical FOV is independent of oversampling (os_factor is used elsewhere).
    L_um = (L_nm * 1e-3)

    half_L = 0.5 * L_um
    margin = pitch_um  # one extra lattice period in each direction

    x_min = -half_L - margin
    x_max = half_L + margin
    y_min = -half_L - margin
    y_max = half_L + margin

    i_min = int(math.floor(x_min / pitch_um))
    i_max = int(math.ceil(x_max / pitch_um))
    j_min = int(math.floor(y_min / pitch_um))
    j_max = int(math.ceil(y_max / pitch_um))

    return i_min, i_max, j_min, j_max

def _sample_edge_perturbation_coefficients(
    effective_amp_rel_max: float,
    mode_count: int,
    rng: np.random.Generator,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample per-feature edge perturbation coefficients for the angular series:

        δ(θ) = Σ_k A_k * cos(k θ + φ_k)

    The amplitudes A_k are drawn so that the sum of their absolute values is
    bounded by effective_amp_rel_max, ensuring that the perturbed radius stays
    within [1 - effective_amp_rel_max, 1 + effective_amp_rel_max] times the
    baseline radius in typical cases.

    Returns:
        enabled (bool): Whether perturbation is active (effective_amp_rel_max > 0 and mode_count > 0).
        modes (Optional[np.ndarray]): Integer mode indices k.
        amp (Optional[np.ndarray]): Amplitudes A_k (float32).
        phase (Optional[np.ndarray]): Phases φ_k in radians (float32).
    """
    if effective_amp_rel_max <= 0.0 or mode_count <= 0:
        return False, None, None, None

    # Simple strategy: distribute amplitude budget evenly across modes so that
    # sum(|A_k|) <= effective_amp_rel_max. We still randomize signs.
    per_mode_max = effective_amp_rel_max / float(mode_count)

    modes = np.arange(1, mode_count + 1, dtype=np.int16)
    amp = rng.uniform(
        low=-per_mode_max,
        high=per_mode_max,
        size=mode_count,
    ).astype(np.float32)
    phase = rng.uniform(
        low=0.0,
        high=2.0 * math.pi,
        size=mode_count,
    ).astype(np.float32)

    return True, modes, amp, phase

def _build_feature_layout(
    params: dict,
    pattern_model: str,
    pitch_um: float,
    nominal_radius_um: float,
    layout_extent_nm: float | None = None,
    rng: np.random.Generator | None = None,
) -> _FeatureLayout:
    """
    Construct a randomized or ideal lattice feature layout for the parameter set.

    Randomization is controlled by:
        - sample_environment_pattern_randomization_enabled
        - sample_environment_pattern_position_jitter_std_nm
        - sample_environment_pattern_shape_regularity

    The layout is built in pattern coordinates aligned with the FOV, using the
    same centered convention as _generate_gold_hole_pattern / optical maps.

    In addition to local feature randomization, every layout includes a global
    pattern offset (offset_x_um, offset_y_um) sampled once per layout build.
    This offset shifts the entire lattice uniformly relative to the camera
    center while preserving the periodicity of the pattern. The offset is
    always applied when substrate patterns are enabled and is independent of the
    sample_environment_pattern_randomization_enabled flag (which controls only local
    imperfections).

    Per-feature edge perturbation coefficients are generated for gold_holes
    when enabled via the global edge perturbation parameters and scaled by
    (1 - sample_environment_pattern_shape_regularity) so that highly regular shapes have
    minimal boundary roughness.
    """
    from .registry import _substrate_pattern_is_enabled

    rng = np.random.default_rng() if rng is None else rng
    substrate_enabled = _substrate_pattern_is_enabled(params)
    if not substrate_enabled:
        # Empty layout for disabled substrate-pattern rendering.
        features_by_cell: Dict[Tuple[int, int], _LatticeFeature] = {}
        return _FeatureLayout(
            pattern_model,
            pitch_um,
            nominal_radius_um,
            features_by_cell,
            0,
            -1,
            0,
            -1,
            0.0,
            0.0,
        )

    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    layout_extent_nm = _effective_layout_extent_nm(params, layout_extent_nm)

    i_min, i_max, j_min, j_max = _compute_lattice_bounds(
        img_size_pixels=img_size_pixels,
        pixel_size_nm=pixel_size_nm,
        pitch_um=pitch_um,
        extent_nm=layout_extent_nm,
    )

    randomization_enabled, jitter_std_um, shape_regularity = _get_randomization_settings(params)
    edge_amp_rel_max, edge_mode_count = _get_edge_perturbation_settings(params)

    # Axis distortion is capped at 25% of nominal radius so randomized holes and
    # pillars remain recognizable while still breaking perfect circular symmetry.
    distortion_frac = _MAX_SHAPE_AXIS_DISTORTION_FRAC * (1.0 - shape_regularity)

    # Effective edge perturbation amplitude: scale by (1 - shape_regularity)
    # so that shape_regularity = 1.0 -> perfectly smooth edges.
    effective_edge_amp_rel_max = edge_amp_rel_max * (1.0 - shape_regularity)

    features_by_cell: Dict[Tuple[int, int], _LatticeFeature] = {}

    # Global pattern offset: always applied when a substrate pattern is enabled.
    # We sample offsets uniformly over a single repeat unit in each direction,
    # [0, pitch_um). This is equivalent to wrapping the pattern relative to
    # the camera FOV and ensures that each video sees the grid in a different
    # lateral position while preserving periodic tiling.
    offset_x_um = float(rng.uniform(0.0, pitch_um))
    offset_y_um = float(rng.uniform(0.0, pitch_um))

    for i in range(i_min, i_max + 1):
        # Nominal center for this lattice index, including global offset.
        center_x_nominal_um = i * pitch_um + offset_x_um
        for j in range(j_min, j_max + 1):
            center_y_nominal_um = j * pitch_um + offset_y_um

            if randomization_enabled:
                # Gaussian jitter in position.
                dx_jitter = rng.normal(loc=0.0, scale=jitter_std_um)
                dy_jitter = rng.normal(loc=0.0, scale=jitter_std_um)
                center_x_um = center_x_nominal_um + dx_jitter
                center_y_um = center_y_nominal_um + dy_jitter

                if distortion_frac > 0.0:
                    delta_x = rng.uniform(-distortion_frac, distortion_frac)
                    delta_y = rng.uniform(-distortion_frac, distortion_frac)
                else:
                    delta_x = 0.0
                    delta_y = 0.0

                r_x_um = nominal_radius_um * (1.0 + delta_x)
                r_y_um = nominal_radius_um * (1.0 + delta_y)

                # Preserve at least half the nominal radius so randomized
                # ellipses cannot collapse into line-like features.
                r_x_um = max(r_x_um, nominal_radius_um * _MIN_SHAPE_RADIUS_FACTOR)
                r_y_um = max(r_y_um, nominal_radius_um * _MIN_SHAPE_RADIUS_FACTOR)

                theta_rad = 0.0
            else:
                # Ideal periodic circles, but the entire grid is globally shifted
                # by (offset_x_um, offset_y_um).
                center_x_um = center_x_nominal_um
                center_y_um = center_y_nominal_um
                r_x_um = nominal_radius_um
                r_y_um = nominal_radius_um
                theta_rad = 0.0

            # Edge perturbation is defined for the nanohole array. Nanopillars
            # and other pattern models remain smooth.
            if pattern_model == "gold_holes" and effective_edge_amp_rel_max > 0.0:
                enabled, modes, amp, phase = _sample_edge_perturbation_coefficients(
                    effective_amp_rel_max=effective_edge_amp_rel_max,
                    mode_count=edge_mode_count,
                    rng=rng,
                )
            else:
                enabled, modes, amp, phase = False, None, None, None

            features_by_cell[(i, j)] = _LatticeFeature(
                center_x_um=center_x_um,
                center_y_um=center_y_um,
                r_x_um=r_x_um,
                r_y_um=r_y_um,
                theta_rad=theta_rad,
                edge_perturbation_enabled=enabled,
                edge_modes=modes,
                edge_amp=amp,
                edge_phase=phase,
            )

    return _FeatureLayout(
        pattern_model=pattern_model,
        pitch_um=pitch_um,
        nominal_radius_um=nominal_radius_um,
        features_by_cell=features_by_cell,
        i_min=i_min,
        i_max=i_max,
        j_min=j_min,
        j_max=j_max,
        offset_x_um=offset_x_um,
        offset_y_um=offset_y_um,
    )

def _layout_rng_for_cache_key(
    params: dict,
    cache_key: tuple,
) -> np.random.Generator:
    explicit_rng = params.get("_substrate_pattern_layout_rng", None)
    if explicit_rng is not None:
        return explicit_rng
    has_seed_surface = (
        params.get("random_seed", None) is not None
        or params.get("_substrate_pattern_layout_cache_token", None) is not None
    )
    if not has_seed_surface:
        return np.random.default_rng()
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence(words))

def _get_feature_layout_for_params(
    params: dict,
    pattern_model: str,
    pitch_um: float,
    nominal_radius_um: float,
    layout_extent_nm: float | None = None,
) -> _FeatureLayout:
    """
    Retrieve (or build and cache) the feature layout corresponding to the
    current substrate-pattern configuration.

    The cache key uses only values that affect geometry deterministically for
    a given simulation run. The global pattern offset is *not* part of the
    cache key; it is sampled when the layout is first built and stored inside
    the layout. As long as the cache is not cleared, all callers in the same
    run see the same offset and the same feature centers.

    Note:
        The randomness used to build a layout (offset, jitter, shape
        distortion, edge perturbation) is driven by the global NumPy RNG. In
        the dataset generator, np.random.seed is set per video, so each video
    gets its own randomized layout (including global offset and edge
    shapes) in a deterministic way for a given seed.
    """
    from .registry import _substrate_pattern_is_enabled

    substrate_enabled = _substrate_pattern_is_enabled(params)
    if not substrate_enabled:
        # No substrate pattern = no layout; return an empty layout so callers can still run.
        empty_key = ("none", 0.0, 0.0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0)
        with _LAYOUT_CACHE_LOCK:
            layout = _LAYOUT_CACHE.get(empty_key)
            if layout is None:
                layout = _FeatureLayout("none", 1.0, 0.0, {}, 0, -1, 0, -1, 0.0, 0.0)
                _LAYOUT_CACHE[empty_key] = layout
        return layout

    layout_extent_nm = _effective_layout_extent_nm(params, layout_extent_nm)

    random_enabled, jitter_std_um, shape_reg = _get_randomization_settings(params)
    if random_enabled and pitch_um > 0.0 and jitter_std_um > 0.25 * float(pitch_um):
        raise ValueError(
            "PARAMS['sample_environment_pattern_position_jitter_std_nm'] is too large "
            "for the configured pattern pitch. Keep jitter standard deviation <= 25% "
            "of the pattern pitch so solid/fluid classification remains local."
        )
    edge_amp_rel_max, edge_mode_count = _get_edge_perturbation_settings(params)

    layout_cache_token = params.get(
        "_substrate_pattern_layout_cache_token",
        params.get("random_seed", None),
    )
    if layout_cache_token is not None:
        layout_cache_token = str(layout_cache_token)

    cache_key = (
        pattern_model,
        float(pitch_um),
        float(nominal_radius_um),
        float(layout_extent_nm),
        bool(random_enabled),
        float(jitter_std_um),
        float(shape_reg),
        float(edge_amp_rel_max),
        int(edge_mode_count),
        layout_cache_token,
    )

    with _LAYOUT_CACHE_LOCK:
        layout = _LAYOUT_CACHE.get(cache_key)
    if layout is None:
        layout_rng = _layout_rng_for_cache_key(params, cache_key)
        layout = _build_feature_layout(
            params=params,
            pattern_model=pattern_model,
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
            layout_extent_nm=layout_extent_nm,
            rng=layout_rng,
        )
        with _LAYOUT_CACHE_LOCK:
            existing = _LAYOUT_CACHE.get(cache_key)
            if existing is None:
                _LAYOUT_CACHE[cache_key] = layout
            else:
                layout = existing

    return layout

def _compute_feature_boundary_radius(
    feature: _LatticeFeature,
    dx_um: float,
    dy_um: float,
) -> float:
    """
    Compute the boundary radius (in micrometers) for a given feature in the
    direction specified by (dx_um, dy_um), which are the coordinates of the
    query point relative to the feature center.

    For features without edge perturbation enabled, the boundary is given by
    the ellipse defined by (r_x_um, r_y_um) and theta_rad.

    For features with edge perturbation enabled, the boundary radius is
    modulated by an angular perturbation series δ(θ) constructed from the
    per-feature coefficients stored on the feature instance.

    The returned value r_boundary_um is the radial distance from the feature
    center to the perturbed boundary along the direction of (dx_um, dy_um).
    """
    # If semi-axes are non-positive, treat as degenerate (no solid area).
    if feature.r_x_um <= 0.0 or feature.r_y_um <= 0.0:
        return 0.0

    # Rotate into the ellipse frame if needed.
    if feature.theta_rad != 0.0:
        ct = math.cos(-feature.theta_rad)
        st = math.sin(-feature.theta_rad)
        ex = ct * dx_um - st * dy_um
        ey = st * dx_um + ct * dy_um
    else:
        ex = dx_um
        ey = dy_um

    # Direction angle θ in the ellipse frame.
    theta = math.atan2(ey, ex)

    # Baseline ellipse boundary radius along direction θ.
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    denom = (cos_t / feature.r_x_um) ** 2 + (sin_t / feature.r_y_um) ** 2
    if denom <= 0.0:
        # Should not normally happen; fall back to min semi-axis.
        base_radius = min(feature.r_x_um, feature.r_y_um)
    else:
        base_radius = 1.0 / math.sqrt(denom)

    # Edge perturbation disabled or coefficients not present: return baseline.
    if not feature.edge_perturbation_enabled or feature.edge_modes is None:
        return base_radius

    modes = feature.edge_modes
    amp = feature.edge_amp
    phase = feature.edge_phase
    if modes is None or amp is None or phase is None:
        return base_radius

    # Evaluate δ(θ) = Σ_k A_k * cos(k θ + φ_k).
    # All arrays are small (mode_count ~ 3 by default), so a simple loop is fine.
    delta = 0.0
    for k, a_k, phi_k in zip(modes, amp, phase):
        delta += float(a_k) * math.cos(float(k) * theta + float(phi_k))

    factor = 1.0 + delta
    if factor <= 0.0:
        # Keep edge perturbations from inverting the boundary; 5% of the
        # baseline radius is a positive floor for pathological coefficient sums.
        factor = _MIN_EDGE_RADIUS_FACTOR

    return base_radius * factor

def _classify_point_against_layout(
    layout: _FeatureLayout,
    x_um: float,
    y_um: float,
    *,
    boundary_offset_um: float = 0.0,
) -> bool:
    """
    Classify a point (x_um, y_um) in pattern coordinates against a feature
    layout.

    Returns:
        inside_feature (bool): True if the point lies inside any feature
        (hole OR pillar, depending on pattern semantics).

    The global pattern offset is already baked into the feature centers stored
    in the layout. This function assumes x_um, y_um are in the same centered
    pattern coordinates as used for optical maps and trajectories.

    For gold_holes, the feature boundary may include per-hole edge
    perturbations; for other pattern models, the boundary remains the smooth
    ellipse defined by r_x_um, r_y_um, and theta_rad.
    """
    pitch_um = layout.pitch_um
    if pitch_um <= 0.0 or not layout.features_by_cell:
        return False

    # Approximate lattice indices of the nearest feature in the ideal grid.
    # Because the layout was built from integer indices (i, j) with a uniform
    # global offset, x_um / pitch_um is still close to the underlying index
    # even after the shift. The small 3x3 neighborhood is sufficient as long
    # as the jitter remains modest relative to the pitch.
    i0 = int(round(x_um / pitch_um))
    j0 = int(round(y_um / pitch_um))

    for di in (-1, 0, 1):
        i = i0 + di
        if i < layout.i_min or i > layout.i_max:
            continue
        for dj in (-1, 0, 1):
            j = j0 + dj
            if j < layout.j_min or j > layout.j_max:
                continue
            feature = layout.features_by_cell.get((i, j))
            if feature is None:
                continue

            dx = x_um - feature.center_x_um
            dy = y_um - feature.center_y_um

            # Compute boundary radius in this direction.
            r_boundary_um = _compute_feature_boundary_radius(feature, dx, dy)
            r_boundary_um = max(r_boundary_um + float(boundary_offset_um), 0.0)
            if r_boundary_um <= 0.0:
                continue

            r_um = math.hypot(dx, dy)
            if r_um <= r_boundary_um:
                return True

    return False

def _feature_radius_bound_um(feature: _LatticeFeature) -> float:
    radius = max(float(feature.r_x_um), float(feature.r_y_um), 0.0)
    if feature.edge_perturbation_enabled and feature.edge_amp is not None:
        radius *= 1.0 + float(np.sum(np.abs(feature.edge_amp)))
    return max(radius, 0.0)

def _classify_grid_against_feature(
    feature: _LatticeFeature,
    X_um: np.ndarray,
    Y_um: np.ndarray,
) -> np.ndarray:
    dx = np.asarray(X_um, dtype=float) - float(feature.center_x_um)
    dy = np.asarray(Y_um, dtype=float) - float(feature.center_y_um)
    if feature.r_x_um <= 0.0 or feature.r_y_um <= 0.0:
        return np.zeros_like(dx, dtype=bool)

    if feature.theta_rad != 0.0:
        ct = math.cos(-float(feature.theta_rad))
        st = math.sin(-float(feature.theta_rad))
        ex = ct * dx - st * dy
        ey = st * dx + ct * dy
    else:
        ex = dx
        ey = dy

    theta = np.arctan2(ey, ex)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    denom = (cos_t / float(feature.r_x_um)) ** 2 + (sin_t / float(feature.r_y_um)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = np.where(denom > 0.0, 1.0 / np.sqrt(denom), 0.0)

    if feature.edge_perturbation_enabled and feature.edge_modes is not None:
        if feature.edge_amp is not None and feature.edge_phase is not None:
            delta = np.zeros_like(boundary, dtype=float)
            for k, a_k, phi_k in zip(feature.edge_modes, feature.edge_amp, feature.edge_phase):
                delta += float(a_k) * np.cos(float(k) * theta + float(phi_k))
            boundary = boundary * np.maximum(1.0 + delta, _MIN_EDGE_RADIUS_FACTOR)

    r_um = np.hypot(dx, dy)
    return r_um <= boundary

def _nearest_feature_and_vector(
    layout: _FeatureLayout,
    x_um: float,
    y_um: float,
) -> Tuple[Optional[_LatticeFeature], float, float]:
    """
    Find the nearest feature center to (x_um, y_um) in pattern coordinates
    using the same local lattice neighborhood assumption as the classifier.

    Returns:
        feature (Optional[_LatticeFeature]): The nearest feature or None if
            no feature is found (should not happen in normal configurations).
        dx (float): x-offset from feature center to point (x_um - center_x_um).
        dy (float): y-offset from feature center to point.
    """
    pitch_um = layout.pitch_um
    if pitch_um <= 0.0 or not layout.features_by_cell:
        return None, 0.0, 0.0

    i0 = int(round(x_um / pitch_um))
    j0 = int(round(y_um / pitch_um))

    best_feature = None
    best_dx = 0.0
    best_dy = 0.0
    best_dist2 = float("inf")

    for di in (-1, 0, 1):
        i = i0 + di
        if i < layout.i_min or i > layout.i_max:
            continue
        for dj in (-1, 0, 1):
            j = j0 + dj
            if j < layout.j_min or j > layout.j_max:
                continue
            feature = layout.features_by_cell.get((i, j))
            if feature is None:
                continue

            dx = x_um - feature.center_x_um
            dy = y_um - feature.center_y_um
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_feature = feature
                best_dx = dx
                best_dy = dy

    return best_feature, best_dx, best_dy

__all__ = ['clear_sample_environment_pattern_layout_cache']
