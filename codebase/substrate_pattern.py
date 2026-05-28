"""
substrate_pattern.py — Substrate / surface-pattern geometry and optical maps.

This module owns the sample_environment_pattern_* parameter family and shared
feature layout used by both optical-background generation and Brownian
exclusion geometry.
"""

import math
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from config import PARAMS

_MAX_SHAPE_AXIS_DISTORTION_FRAC = 0.25
_MIN_SHAPE_RADIUS_FACTOR = 0.5
_MIN_EDGE_RADIUS_FACTOR = 0.05
_REFLECTION_BOUNDARY_BISECTION_STEPS = 20


def _param_default(key: str):
    return PARAMS[key]


def _normalize_to_unit_std(field: np.ndarray) -> np.ndarray:
    """
    Center a floating field and normalize it to unit standard deviation when
    possible. Degenerate fields return all zeros.
    """
    field = np.asarray(field, dtype=float)
    centered = field - float(np.mean(field))
    std = float(np.std(centered))
    if std <= 1e-12:
        return np.zeros_like(centered)
    return centered / std


def generate_empirical_background_field(
    params: dict,
    final_fov_shape: tuple,
) -> np.ndarray:
    """
    Generate a dimensionless, mean-one, spatially correlated nuisance field.

    This field represents residual illumination / flat-field variation,
    detector-offset nonuniformity, and other slowly varying empirical
    background terms. It is intentionally modeled as a smooth nuisance field,
    not as a latent population of out-of-focus particles.
    """
    shape = (int(final_fov_shape[0]), int(final_fov_shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("final_fov_shape must contain positive dimensions.")

    enabled = bool(params.get("empirical_background_enabled", False))
    if not enabled:
        return np.ones(shape, dtype=float)

    model = str(
        params.get("empirical_background_model", "multiscale_gaussian_field")
    ).strip().lower()
    if model not in ("multiscale_gaussian_field", "none"):
        raise ValueError(
            "Unsupported empirical_background_model "
            f"'{params.get('empirical_background_model')}'."
        )
    if model == "none":
        return np.ones(shape, dtype=float)

    relative_std = float(params.get("empirical_background_relative_std", 0.03))
    gradient_strength = float(
        params.get("empirical_background_gradient_relative_strength", 0.0)
    )
    if relative_std < 0.0:
        raise ValueError("empirical_background_relative_std must be non-negative.")
    if gradient_strength < 0.0:
        raise ValueError(
            "empirical_background_gradient_relative_strength must be non-negative."
        )

    scales = params.get("empirical_background_scales_px", [16.0, 64.0, 256.0])
    weights = params.get("empirical_background_scale_weights", [0.4, 0.35, 0.25])
    if len(scales) != len(weights):
        raise ValueError(
            "empirical_background_scales_px and "
            "empirical_background_scale_weights must have the same length."
        )
    if len(scales) == 0:
        return np.ones(shape, dtype=float)

    field = np.zeros(shape, dtype=float)
    for sigma_px, weight in zip(scales, weights):
        sigma = float(sigma_px)
        layer_weight = float(weight)
        if sigma <= 0.0 or layer_weight == 0.0:
            continue
        white = np.random.normal(loc=0.0, scale=1.0, size=shape)
        smooth = cv2.GaussianBlur(
            white,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        field += layer_weight * _normalize_to_unit_std(smooth)

    field = _normalize_to_unit_std(field)

    if gradient_strength > 0.0:
        yy, xx = np.indices(shape, dtype=float)
        x_norm = (xx - float(np.mean(xx))) / max(float(shape[1] - 1), 1.0)
        y_norm = (yy - float(np.mean(yy))) / max(float(shape[0] - 1), 1.0)
        angle = float(np.random.uniform(0.0, 2.0 * math.pi))
        gradient = math.cos(angle) * x_norm + math.sin(angle) * y_norm
        field += gradient_strength / max(relative_std, 1e-12) * _normalize_to_unit_std(
            gradient
        )
        field = _normalize_to_unit_std(field)

    if relative_std == 0.0:
        return np.ones(shape, dtype=float)

    nuisance = 1.0 + relative_std * field
    nuisance = np.clip(nuisance, 1e-6, None)
    mean_val = float(np.mean(nuisance))
    if mean_val > 0.0:
        nuisance /= mean_val
    return nuisance.astype(float)


def resize_empirical_background_field(
    nuisance_final: np.ndarray,
    target_shape: tuple,
) -> np.ndarray:
    """
    Resize a final-resolution empirical background field to another grid.
    """
    nuisance_final = np.asarray(nuisance_final, dtype=float)
    target = cv2.resize(
        nuisance_final,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(target, 1e-6, None).astype(float)


# -----------------------------------------------------------------------------
# Internal feature layout structures and cache
# -----------------------------------------------------------------------------

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


# Cache keyed by a simple signature so that all calls in a run share one layout.
_LAYOUT_CACHE: Dict[Tuple, _FeatureLayout] = {}


def _substrate_pattern_is_enabled(params: dict) -> bool:
    return (
        bool(params.get("sample_environment_enabled", True))
        and bool(params.get("sample_environment_pattern_enabled", False))
    )


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
    amp = np.random.uniform(
        low=-per_mode_max,
        high=per_mode_max,
        size=mode_count,
    ).astype(np.float32)
    phase = np.random.uniform(
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
    offset_x_um = float(np.random.uniform(0.0, pitch_um))
    offset_y_um = float(np.random.uniform(0.0, pitch_um))

    for i in range(i_min, i_max + 1):
        # Nominal center for this lattice index, including global offset.
        center_x_nominal_um = i * pitch_um + offset_x_um
        for j in range(j_min, j_max + 1):
            center_y_nominal_um = j * pitch_um + offset_y_um

            if randomization_enabled:
                # Gaussian jitter in position.
                dx_jitter = np.random.normal(loc=0.0, scale=jitter_std_um)
                dy_jitter = np.random.normal(loc=0.0, scale=jitter_std_um)
                center_x_um = center_x_nominal_um + dx_jitter
                center_y_um = center_y_nominal_um + dy_jitter

                if distortion_frac > 0.0:
                    delta_x = np.random.uniform(-distortion_frac, distortion_frac)
                    delta_y = np.random.uniform(-distortion_frac, distortion_frac)
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
    substrate_enabled = _substrate_pattern_is_enabled(params)
    if not substrate_enabled:
        # No substrate pattern = no layout; return an empty layout so callers can still run.
        empty_key = ("none", 0.0, 0.0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0)
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

    layout = _LAYOUT_CACHE.get(cache_key)
    if layout is None:
        layout = _build_feature_layout(
            params=params,
            pattern_model=pattern_model,
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
            layout_extent_nm=layout_extent_nm,
        )
        _LAYOUT_CACHE[cache_key] = layout

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

def _generate_gold_hole_pattern(
    shape: tuple,
    pixel_size_nm: float,
    hole_diameter_um: float,
    hole_edge_to_edge_spacing_um: float,
    hole_intensity_factor: float,
    gold_intensity_factor: float,
    params: Optional[dict] = None,
    layout_pattern_model: str = "gold_holes",
    layout_extent_nm: float | None = None,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a gold film with
    feature-layout holes.

    Behavior:
        - When a PARAMS dictionary is provided, the function uses the shared
          randomized feature layout so optical pattern geometry matches the
          Brownian exclusion geometry.
        - When params is None, the function uses an ideal circular,
          perfectly periodic centered grid for isolated uses.

    The global offset is applied when the layout is built and is invisible to
    callers of this function; here we only query the layout.
    """
    height, width = int(shape[0]), int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("Pattern shape must have positive height and width.")

    pixel_size_nm = float(pixel_size_nm)
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive for pattern generation.")

    hole_diameter_um = float(hole_diameter_um)
    hole_edge_to_edge_spacing_um = float(hole_edge_to_edge_spacing_um)
    if hole_diameter_um <= 0.0:
        raise ValueError("hole_diameter_um must be positive.")
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError("hole_edge_to_edge_spacing_um must be non-negative.")

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    radius_um = hole_diameter_um / 2.0

    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    hole_intensity_factor = float(hole_intensity_factor)
    gold_intensity_factor = float(gold_intensity_factor)
    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "hole_intensity_factor and gold_intensity_factor must be positive."
        )

    pixel_size_um = pixel_size_nm * 1e-3

    x_indices = np.arange(width, dtype=float)
    y_indices = np.arange(height, dtype=float)

    x_um = (x_indices - width / 2.0 + 0.5) * pixel_size_um
    y_um = (y_indices - height / 2.0 + 0.5) * pixel_size_um

    X_um, Y_um = np.meshgrid(x_um, y_um)

    if params is None:
        # Ideal, perfectly periodic circle model for isolated calls.
        half_pitch = pitch_um / 2.0
        dx_um = (X_um + half_pitch) % pitch_um - half_pitch
        dy_um = (Y_um + half_pitch) % pitch_um - half_pitch
        r_um = np.sqrt(dx_um * dx_um + dy_um * dy_um)

        pattern = np.full((height, width), gold_intensity_factor, dtype=float)
        hole_mask = r_um <= radius_um
        pattern[hole_mask] = hole_intensity_factor
    else:
        # Use shared feature layout, which includes the global lattice offset
        # and per-hole edge perturbations (if enabled).
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model=layout_pattern_model,
            pitch_um=pitch_um,
            nominal_radius_um=radius_um,
            layout_extent_nm=layout_extent_nm,
        )

        pattern = np.full((height, width), gold_intensity_factor, dtype=float)

        # Per-pixel classification using the layout. The layout stores feature
        # centers already shifted by the global offset, so X_um/Y_um can remain
        # in the standard centered FOV coordinates.
        for iy in range(height):
            for ix in range(width):
                inside_hole = _classify_point_against_layout(
                    layout,
                    X_um[iy, ix],
                    Y_um[iy, ix],
                )
                if inside_hole:
                    pattern[iy, ix] = hole_intensity_factor

    mean_val = float(pattern.mean())
    if mean_val > 0.0:
        pattern /= mean_val

    return pattern


def _generate_nanopillar_pattern(
    shape: tuple,
    pixel_size_nm: float,
    pillar_diameter_um: float,
    pillar_edge_to_edge_spacing_um: float,
    pillar_intensity_factor: float,
    background_intensity_factor: float,
    params: Optional[dict] = None,
    layout_extent_nm: float | None = None,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a nanopillar array.

    Behavior:
        - When a PARAMS dictionary is provided, the same shared feature layout
          used for Brownian dynamics defines the optical pattern, so the pattern
          (including the global lattice offset) matches the exclusion geometry.
        - When params is None, returns an ideal periodic circular-pillar pattern
          using the shared circular lattice rasterization path.
    """
    return _generate_gold_hole_pattern(
        shape=shape,
        pixel_size_nm=pixel_size_nm,
        hole_diameter_um=pillar_diameter_um,
        hole_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
        hole_intensity_factor=pillar_intensity_factor,
        gold_intensity_factor=background_intensity_factor,
        params=params,
        layout_pattern_model="nanopillars",
        layout_extent_nm=layout_extent_nm,
    )


def generate_sample_environment_pattern_maps(
    params: dict,
    shape: tuple,
    pixel_size_nm: float,
    layer_thickness_nm: float,
    layout_extent_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Generate height and material-fraction maps for the structured sample interface.

    This uses the same feature layout as the optical background generator and
    substrate-exclusion classifier. ``material_fraction_map`` is the patterned
    layer fraction: for ``gold_holes`` it is gold film outside the holes, and for
    ``nanopillars`` it is pillar material inside the pillars.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Pattern shape must have positive height and width.")

    pixel_size_nm = float(pixel_size_nm)
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive for sample-environment maps.")

    layer_thickness_nm = float(layer_thickness_nm)
    if not np.isfinite(layer_thickness_nm):
        raise ValueError("layer_thickness_nm must be finite for sample-environment maps.")

    pattern_model_raw = params.get("sample_environment_pattern", "none")
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    uniform_height = np.zeros((height, width), dtype=float)
    uniform_fraction = np.ones((height, width), dtype=float)
    if (
        not bool(params.get("sample_environment_enabled", True))
        or not _substrate_pattern_is_enabled(params)
        or substrate_preset == "empty_background"
        or pattern_model == "none"
    ):
        return uniform_height, uniform_fraction, "uniform"

    if pattern_model == "gold_holes":
        if substrate_preset != "default_gold_holes":
            raise ValueError(
                "sample_environment_pattern='gold_holes' supports presets "
                "'empty_background' and 'default_gold_holes'; got "
                f"{substrate_preset_raw!r}."
            )
        geom = _resolve_gold_hole_parameters(params)
        feature_is_material = False
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]
        kind = "gold_holes"
    elif pattern_model == "nanopillars":
        if substrate_preset not in ("nanopillars", "default_nanopillars"):
            raise ValueError(
                "sample_environment_pattern='nanopillars' supports presets "
                "'empty_background', 'nanopillars', and 'default_nanopillars'; got "
                f"{substrate_preset_raw!r}."
            )
        geom = _resolve_nanopillar_parameters(params)
        feature_is_material = True
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]
        kind = "nanopillars"
    else:
        raise ValueError(
            f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
            "Supported models are 'none', 'gold_holes', and 'nanopillars'."
        )

    layout = _get_feature_layout_for_params(
        params=params,
        pattern_model=pattern_model,
        pitch_um=pitch_um,
        nominal_radius_um=nominal_radius_um,
        layout_extent_nm=layout_extent_nm,
    )

    pixel_size_um = pixel_size_nm * 1e-3
    x_um = (np.arange(width, dtype=float) - width / 2.0 + 0.5) * pixel_size_um
    y_um = (np.arange(height, dtype=float) - height / 2.0 + 0.5) * pixel_size_um

    material_fraction = np.zeros((height, width), dtype=float)
    if not feature_is_material:
        material_fraction.fill(1.0)

    for iy, y_value_um in enumerate(y_um):
        for ix, x_value_um in enumerate(x_um):
            inside_feature = _classify_point_against_layout(
                layout,
                x_value_um,
                y_value_um,
            )
            if feature_is_material:
                material_fraction[iy, ix] = 1.0 if inside_feature else 0.0
            elif inside_feature:
                material_fraction[iy, ix] = 0.0

    height_map = layer_thickness_nm * material_fraction
    return height_map.astype(float), material_fraction.astype(float), kind


def _resolve_gold_hole_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for the gold film with
    circular holes from the global PARAMS dictionary.
    """
    dims = params.get("sample_environment_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['sample_environment_pattern_dimensions'] must be a dictionary when "
            "using sample_environment_pattern 'gold_holes'."
        )

    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )
    substrate_preset = str(substrate_preset_raw).strip().lower()

    hole_diameter_um = float(dims.get("hole_diameter_um", 15.0))
    hole_edge_to_edge_spacing_um = float(dims.get("hole_edge_to_edge_spacing_um", 2.0))

    if hole_diameter_um <= 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['hole_diameter_um'] must be positive."
        )
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['hole_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = hole_diameter_um / 2.0

    hole_intensity_factor = float(dims.get("hole_intensity_factor", 0.7))
    gold_intensity_factor = float(dims.get("gold_intensity_factor", 1.0))

    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['hole_intensity_factor'] and "
            "'gold_intensity_factor' must be positive."
        )

    return {
        "hole_diameter_um": hole_diameter_um,
        "hole_edge_to_edge_spacing_um": hole_edge_to_edge_spacing_um,
        "hole_intensity_factor": hole_intensity_factor,
        "gold_intensity_factor": gold_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _resolve_nanopillar_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for a circular nanopillar
    array from the global PARAMS dictionary.
    """
    dims = params.get("sample_environment_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['sample_environment_pattern_dimensions'] must be a dictionary when "
            "using sample_environment_pattern 'nanopillars'."
        )

    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )
    substrate_preset = str(substrate_preset_raw).strip().lower()

    pillar_diameter_um = float(dims.get("pillar_diameter_um", 1.0))
    pillar_edge_to_edge_spacing_um = float(
        dims.get("pillar_edge_to_edge_spacing_um", 2.0)
    )

    if pillar_diameter_um <= 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['pillar_diameter_um'] must be positive."
        )
    if pillar_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['pillar_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = pillar_diameter_um + pillar_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (pillar_diameter_um + pillar_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = pillar_diameter_um / 2.0

    pillar_intensity_factor = float(dims.get("pillar_intensity_factor", 1.3))
    background_intensity_factor = float(dims.get("background_intensity_factor", 1.0))

    if pillar_intensity_factor <= 0.0 or background_intensity_factor <= 0.0:
        raise ValueError(
            "sample_environment_pattern_dimensions['pillar_intensity_factor'] and "
            "sample_environment_pattern_dimensions['background_intensity_factor'] "
            "must be positive."
        )

    return {
        "pillar_diameter_um": pillar_diameter_um,
        "pillar_edge_to_edge_spacing_um": pillar_edge_to_edge_spacing_um,
        "pillar_intensity_factor": pillar_intensity_factor,
        "background_intensity_factor": background_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _map_position_nm_to_pattern_unit_cell(
    params: dict,
    x_nm: float,
    y_nm: float,
    pitch_um: float,
) -> tuple:
    """
    Convert a lateral position (x_nm, y_nm) in world coordinates to centered
    pattern coordinates (x_um, y_um) and return additional unit-cell helpers.

    The return values include modulo-based offsets within a square-lattice unit
    cell and centered coordinates used with the feature layout, which already
    includes the global pattern offset.

    Returns:
        dx_um, dy_um, r_um, x_um, y_um, center_nm
    """
    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    if img_size_pixels <= 0 or pixel_size_nm <= 0.0:
        raise ValueError(
            "PARAMS['image_size_pixels'] and PARAMS['pixel_size_nm'] must be "
            "positive when substrate exclusion is active."
        )

    # Renderer coordinates use integer pixel centers: world x=0 maps to the
    # center of pixel 0, world x=pixel_size maps to pixel 1, and so on. The
    # rasterized substrate maps use the same pixel-center convention via
    # (index - width/2 + 0.5) * pixel_size. Centering by (N - 1)/2 pixels keeps
    # point queries and raster pixels in the same coordinate system.
    center_nm = 0.5 * (img_size_pixels - 1) * pixel_size_nm
    x_nm_centered = float(x_nm) - center_nm
    y_nm_centered = float(y_nm) - center_nm

    x_um = x_nm_centered * 1e-3
    y_um = y_nm_centered * 1e-3

    half_pitch = pitch_um / 2.0
    dx_um = (x_um + half_pitch) % pitch_um - half_pitch
    dy_um = (y_um + half_pitch) % pitch_um - half_pitch
    r_um = math.hypot(dx_um, dy_um)

    return dx_um, dy_um, r_um, x_um, y_um, center_nm


def is_position_in_substrate_solid(
    params: dict,
    x_nm: float,
    y_nm: float,
    *,
    clearance_nm: float = 0.0,
) -> bool:
    """
    Determine whether a lateral position (x_nm, y_nm) lies inside a solid region
    of the configured substrate pattern.

    Behavior:
        - Uses the shared feature layout with imperfections, per-hole boundary
          perturbations (for gold_holes), and a per-layout global lattice
          offset, ensuring the geometry matches the optical substrate pattern.
        - Gold holes:
            solid = gold film (outside holes).
        - Nanopillars:
            solid = pillar interior.
    """
    substrate_enabled = _substrate_pattern_is_enabled(params)
    clearance_um = max(float(clearance_nm), 0.0) * 1e-3

    pattern_model_raw = params.get("sample_environment_pattern", "none"
    )
    pattern_model = str(pattern_model_raw).strip().lower()

    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )
    substrate_preset = str(substrate_preset_raw).strip().lower()

    if (
        not substrate_enabled
        or substrate_preset == "empty_background"
        or pattern_model == "none"
    ):
        return False

    # Gold film with circular holes: solid is gold (outside any hole feature).
    if pattern_model == "gold_holes":
        if substrate_preset != "default_gold_holes":
            raise ValueError(
                "sample_environment_pattern='gold_holes' supports presets "
                "'empty_background' and 'default_gold_holes'; got "
                f"{substrate_preset_raw!r}."
            )

        geom = _resolve_gold_hole_parameters(params)
        pitch_um = geom["pitch_um"]

        # Convert (x_nm, y_nm) to centered pattern coordinates (x_um, y_um).
        _, _, _, x_um, y_um, _ = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes",
            pitch_um=pitch_um,
            nominal_radius_um=geom["radius_um"],
        )

        inside_hole = _classify_point_against_layout(
            layout,
            x_um,
            y_um,
            boundary_offset_um=-clearance_um,
        )
        return not inside_hole

    # Nanopillars: solid is pillar interior.
    if pattern_model == "nanopillars":
        if substrate_preset not in ("nanopillars", "default_nanopillars"):
            raise ValueError(
                "sample_environment_pattern='nanopillars' supports presets "
                "'empty_background', 'nanopillars', and 'default_nanopillars'; got "
                f"{substrate_preset_raw!r}."
            )

        geom = _resolve_nanopillar_parameters(params)
        pitch_um = geom["pitch_um"]

        _, _, _, x_um, y_um, _ = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="nanopillars",
            pitch_um=pitch_um,
            nominal_radius_um=geom["radius_um"],
        )

        inside_pillar = _classify_point_against_layout(
            layout,
            x_um,
            y_um,
            boundary_offset_um=clearance_um,
        )
        return inside_pillar

    raise ValueError(
        f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
        "Supported models are 'none', 'gold_holes', and 'nanopillars'."
    )


def project_position_to_fluid_region(
    params: dict,
    x_nm: float,
    y_nm: float,
    *,
    clearance_nm: float = 0.0,
) -> tuple:
    """
    Given a lateral position (x_nm, y_nm), project it into the nearest fluid
    region of the patterned interface if it currently lies in a
    solid region.

    Behavior:
        - Uses the same feature layout (with imperfections, per-hole boundary
          perturbations, and global offset) as the classifier.
        - Gold holes:
            solid -> gold film. We move the point into the nearest hole
            interior by projecting toward the nearest feature's center and
            placing it just inside the perturbed feature boundary along that
            direction.
        - Nanopillars:
            solid -> pillar interior. We move the point outward to just
            outside the effective pillar boundary.

    The projection is approximate for elliptical features and then checked by
    the same classifier used for Brownian exclusion.
    """
    clearance_um = max(float(clearance_nm), 0.0) * 1e-3
    if not is_position_in_substrate_solid(
        params,
        x_nm,
        y_nm,
        clearance_nm=clearance_nm,
    ):
        return float(x_nm), float(y_nm)

    substrate_enabled = _substrate_pattern_is_enabled(params)
    pattern_model_raw = params.get("sample_environment_pattern", "none"
    )
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )
    substrate_preset = str(substrate_preset_raw).strip().lower()

    if not substrate_enabled:
        return float(x_nm), float(y_nm)

    # --- Gold film with circular holes: project from gold into nearest hole ---
    if (
        pattern_model == "gold_holes"
        and substrate_preset == "default_gold_holes"
    ):
        geom = _resolve_gold_hole_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]

        _, _, _, x_um, y_um, center_nm = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )

        feature, dx, dy = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is None:
            return float(x_nm), float(y_nm)

        # Direction from feature center to point.
        dist_um = math.hypot(dx, dy)
        if dist_um == 0.0:
            # If we are exactly at the feature center (unlikely for solid region),
            # choose an arbitrary direction along +x.
            dx = feature.r_x_um
            dy = 0.0
            dist_um = feature.r_x_um

        # Boundary radius in this direction, using the same perturbed geometry
        # as the classifier.
        r_boundary_um = _compute_feature_boundary_radius(feature, dx, dy)
        if r_boundary_um <= 0.0:
            # Degenerate case: fall back to minimal movement toward center.
            new_x_um = feature.center_x_um
            new_y_um = feature.center_y_um
        else:
            epsilon_um = 1e-3  # 1 nm
            r_target_um = max(r_boundary_um - clearance_um - epsilon_um, 0.0)
            scale = r_target_um / dist_um
            new_x_um = feature.center_x_um + dx * scale
            new_y_um = feature.center_y_um + dy * scale

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + center_nm
        new_y_nm = new_y_nm_centered + center_nm

        # Safety: re-check the projected position with the classifier.
        if is_position_in_substrate_solid(
            params,
            new_x_nm,
            new_y_nm,
            clearance_nm=clearance_nm,
        ):
            # As a fallback, place point at feature center minus epsilon in +x.
            fallback_dx = max(feature.r_x_um - clearance_um - 1e-3, 0.0)
            new_x_um = feature.center_x_um + fallback_dx
            new_y_um = feature.center_y_um
            new_x_nm_centered = new_x_um * 1e3
            new_y_nm_centered = new_y_um * 1e3
            new_x_nm = new_x_nm_centered + center_nm
            new_y_nm = new_y_nm_centered + center_nm

        return float(new_x_nm), float(new_y_nm)

    # --- Nanopillars: project from pillar interior to background fluid ---
    if (
        pattern_model == "nanopillars"
        and substrate_preset in ("nanopillars", "default_nanopillars")
    ):
        geom = _resolve_nanopillar_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]

        _, _, _, x_um, y_um, center_nm = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="nanopillars",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )

        feature, dx, dy = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is None:
            return float(x_nm), float(y_nm)

        dist_um = math.hypot(dx, dy)
        epsilon_um = 1e-3  # 1 nm

        if dist_um == 0.0:
            # If exactly at center, choose a direction along +x.
            # For nanopillars we still use the smooth ellipse boundary.
            r_boundary_um = _compute_feature_boundary_radius(feature, feature.r_x_um, 0.0)
            new_x_um = feature.center_x_um + r_boundary_um + clearance_um + epsilon_um
            new_y_um = feature.center_y_um
        else:
            # Move to just outside the boundary along the direction to the point.
            r_boundary_um = _compute_feature_boundary_radius(feature, dx, dy)
            if r_boundary_um <= 0.0:
                r_boundary_um = min(feature.r_x_um, feature.r_y_um)
            r_target_um = r_boundary_um + clearance_um + epsilon_um
            scale = r_target_um / dist_um
            new_x_um = feature.center_x_um + dx * scale
            new_y_um = feature.center_y_um + dy * scale

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + center_nm
        new_y_nm = new_y_nm_centered + center_nm

        if is_position_in_substrate_solid(
            params,
            new_x_nm,
            new_y_nm,
            clearance_nm=clearance_nm,
        ):
            # Fallback: step further outward along the same direction.
            dx2 = new_x_um - feature.center_x_um
            dy2 = new_y_um - feature.center_y_um
            norm2 = math.hypot(dx2, dy2) or 1.0
            step_um = min(feature.r_x_um, feature.r_y_um)
            new_x_um = feature.center_x_um + dx2 / norm2 * (
                r_boundary_um + clearance_um + step_um
            )
            new_y_um = feature.center_y_um + dy2 / norm2 * (
                r_boundary_um + clearance_um + step_um
            )
            new_x_nm_centered = new_x_um * 1e3
            new_y_nm_centered = new_y_um * 1e3
            new_x_nm = new_x_nm_centered + center_nm
            new_y_nm = new_y_nm_centered + center_nm

        return float(new_x_nm), float(new_y_nm)

    return float(x_nm), float(y_nm)


def reflect_position_across_substrate_boundary(
    params: dict,
    prev_x_nm: float,
    prev_y_nm: float,
    proposed_x_nm: float,
    proposed_y_nm: float,
    *,
    clearance_nm: float = 0.0,
) -> tuple:
    """
    Hard-wall reflection of a Brownian step against the substrate boundary.

    A step that lands in the solid is reflected across the boundary normal at
    the crossing point, preserving the step path length while bending the path
    at the wall. This implements a reflective-boundary approximation for the
    smooth circular pattern features. The projection fallback instead truncates
    the step at the boundary and can reduce apparent diffusion near walls.

    Algorithm:

      1. If ``(proposed_x_nm, proposed_y_nm)`` already lies in fluid, return it
         unchanged.
      2. Bisect along the segment ``prev -> proposed`` to find the crossing
         point ``B`` -- the last point still in fluid before the segment
         enters solid. ``_REFLECTION_BOUNDARY_BISECTION_STEPS`` sets the
         crossing precision.
      3. Determine the outward-pointing wall normal n_hat at B. For
         ``gold_holes`` (fluid = inside hole, solid = gold film outside)
         the wall normal points radially outward from the nearest hole
         center, so the inward normal (back into fluid) is the opposite. For
         ``nanopillars`` (fluid = outside pillar, solid = inside pillar)
         the wall normal points radially inward toward the pillar center.
      4. Reflect the remainder vector ``s = proposed - B`` across the plane
         through B with normal n_hat: ``s_reflected = s - 2 (s . n_hat) n_hat``.
         New position is ``B + s_reflected``.
      5. If the reflected position still lands in solid (e.g. the step is
         large enough to cross multiple feature boundaries), fall back to
         ``project_position_to_fluid_region`` and re-check the result.

    The radial-normal approximation is exact for the smooth circular feature
    boundary and is a good approximation for the lightly perturbed boundary
    used in the default gold-holes/nanopillars layouts. For the typical
    operating regime (Brownian step << feature size) reflection and
    projection agree to within numerical noise; the difference matters only
    when steps are comparable to or larger than the feature size.

    Returns
    -------
    (new_x_nm, new_y_nm) : tuple of float
        Reflected lateral position accepted by the substrate-region classifier.
    """
    if not is_position_in_substrate_solid(
        params,
        proposed_x_nm,
        proposed_y_nm,
        clearance_nm=clearance_nm,
    ):
        return float(proposed_x_nm), float(proposed_y_nm)

    # Bisection along the segment to find the last fluid point before entry.
    # If ``prev`` is already in solid, bisection cannot identify an entry
    # boundary; fall through to projection in that case.
    if is_position_in_substrate_solid(
        params,
        prev_x_nm,
        prev_y_nm,
        clearance_nm=clearance_nm,
    ):
        return project_position_to_fluid_region(
            params,
            proposed_x_nm,
            proposed_y_nm,
            clearance_nm=clearance_nm,
        )

    lo, hi = 0.0, 1.0
    dx_seg = proposed_x_nm - prev_x_nm
    dy_seg = proposed_y_nm - prev_y_nm
    for _ in range(_REFLECTION_BOUNDARY_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        x_mid = prev_x_nm + mid * dx_seg
        y_mid = prev_y_nm + mid * dy_seg
        if is_position_in_substrate_solid(
            params,
            x_mid,
            y_mid,
            clearance_nm=clearance_nm,
        ):
            hi = mid
        else:
            lo = mid
    t_boundary = lo
    bx_nm = prev_x_nm + t_boundary * dx_seg
    by_nm = prev_y_nm + t_boundary * dy_seg

    # Wall normal at B. Look up the nearest feature in the unit-cell frame and
    # compute the radial direction from feature center to B.
    pattern_model = str(params.get("sample_environment_pattern", "none")).strip().lower()
    substrate_preset = str(
        params.get("sample_environment_pattern_preset", "empty_background")
    ).strip().lower()

    n_x_world = 0.0
    n_y_world = 0.0
    if (
        pattern_model == "gold_holes"
        and substrate_preset == "default_gold_holes"
    ):
        geom = _resolve_gold_hole_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]
        _, _, _, x_um, y_um, _center_nm = _map_position_nm_to_pattern_unit_cell(
            params, bx_nm, by_nm, pitch_um,
        )
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )
        feature, dx_um, dy_um = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is not None:
            r = math.hypot(dx_um, dy_um)
            if r > 0.0:
                # Outward (into-solid) normal points radially outward from the
                # hole center: gold = outside hole.
                n_x_world = dx_um / r
                n_y_world = dy_um / r
    elif (
        pattern_model == "nanopillars"
        and substrate_preset in ("nanopillars", "default_nanopillars")
    ):
        geom = _resolve_nanopillar_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]
        _, _, _, x_um, y_um, _center_nm = _map_position_nm_to_pattern_unit_cell(
            params, bx_nm, by_nm, pitch_um,
        )
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="nanopillars",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )
        feature, dx_um, dy_um = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is not None:
            r = math.hypot(dx_um, dy_um)
            if r > 0.0:
                # Outward (into-solid) normal points radially inward toward the
                # pillar center: pillar = inside the boundary.
                n_x_world = -dx_um / r
                n_y_world = -dy_um / r

    # If the normal could not be determined, fall back to projection.
    if n_x_world == 0.0 and n_y_world == 0.0:
        return project_position_to_fluid_region(
            params,
            proposed_x_nm,
            proposed_y_nm,
            clearance_nm=clearance_nm,
        )

    # Reflect the remainder vector across the wall.
    s_x = proposed_x_nm - bx_nm
    s_y = proposed_y_nm - by_nm
    sn = s_x * n_x_world + s_y * n_y_world
    s_refl_x = s_x - 2.0 * sn * n_x_world
    s_refl_y = s_y - 2.0 * sn * n_y_world
    new_x_nm = bx_nm + s_refl_x
    new_y_nm = by_nm + s_refl_y

    # If the reflected point is still in solid (large step crossing multiple
    # features, or perturbed-boundary geometry pushing back into solid), fall
    # back to the projection helper.
    if is_position_in_substrate_solid(
        params,
        new_x_nm,
        new_y_nm,
        clearance_nm=clearance_nm,
    ):
        return project_position_to_fluid_region(
            params,
            new_x_nm,
            new_y_nm,
            clearance_nm=clearance_nm,
        )
    return float(new_x_nm), float(new_y_nm)


def generate_reference_and_background_maps(
    params: dict,
    fov_shape_os: tuple,
    final_fov_shape: tuple,
    layout_extent_nm: float | None = None,
):
    """
    Generate stationary reference field and background intensity maps for the
    simulated field of view.

    Behavior:
        - When a substrate pattern is enabled, the gold_holes and nanopillars
          generators use the same randomized feature layout that drives
          is_position_in_substrate_solid / project_position_to_fluid_region,
          including a per-layout global lattice offset and, for gold_holes,
          per-hole edge perturbations. Optical backgrounds and Brownian
          exclusion are therefore geometrically consistent.
    """
    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    substrate_enabled = _substrate_pattern_is_enabled(params)

    pattern_model_raw = params.get("sample_environment_pattern", "none"
    )
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )

    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset = str(substrate_preset_raw).strip().lower()

    use_uniform_background = (
        (not substrate_enabled)
        or (substrate_preset == "empty_background")
        or (pattern_model == "none")
    )

    if use_uniform_background:
        E_ref_os = np.full(fov_shape_os, E_ref_amplitude, dtype=np.complex128)
        E_ref_final = np.full(final_fov_shape, E_ref_amplitude, dtype=np.complex128)
        background_final = np.full(final_fov_shape, background_intensity, dtype=float)
        return E_ref_os, E_ref_final, background_final

    pixel_size_nm = float(params["pixel_size_nm"])
    if pixel_size_nm <= 0.0:
        raise ValueError("PARAMS['pixel_size_nm'] must be positive.")

    os_factor = float(params.get("psf_oversampling_factor", 1.0))
    if os_factor <= 0.0:
        raise ValueError("PARAMS['psf_oversampling_factor'] must be positive.")

    if pattern_model == "gold_holes":
        if substrate_preset != "default_gold_holes":
            raise ValueError(
                f"Unsupported sample_environment_pattern_preset '{substrate_preset_raw}' for "
                "sample_environment_pattern 'gold_holes'. Supported presets are "
                "'empty_background' and 'default_gold_holes'."
            )

        geom = _resolve_gold_hole_parameters(params)
        hole_diameter_um = geom["hole_diameter_um"]
        hole_edge_to_edge_spacing_um = geom["hole_edge_to_edge_spacing_um"]
        hole_intensity_factor = geom["hole_intensity_factor"]
        gold_intensity_factor = geom["gold_intensity_factor"]

        pattern_final = _generate_gold_hole_pattern(
            shape=final_fov_shape,
            pixel_size_nm=pixel_size_nm,
            hole_diameter_um=hole_diameter_um,
            hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
            hole_intensity_factor=hole_intensity_factor,
            gold_intensity_factor=gold_intensity_factor,
            params=params,
            layout_extent_nm=layout_extent_nm,
        )

        pattern_os = _generate_gold_hole_pattern(
            shape=fov_shape_os,
            pixel_size_nm=pixel_size_nm / os_factor,
            hole_diameter_um=hole_diameter_um,
            hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
            hole_intensity_factor=hole_intensity_factor,
            gold_intensity_factor=gold_intensity_factor,
            params=params,
            layout_extent_nm=layout_extent_nm,
        )

    elif pattern_model == "nanopillars":
        if substrate_preset not in ("nanopillars", "default_nanopillars"):
            raise ValueError(
                f"Unsupported sample_environment_pattern_preset '{substrate_preset_raw}' for "
                "sample_environment_pattern 'nanopillars'. Supported presets are "
                "'empty_background', 'nanopillars', and 'default_nanopillars'."
            )

        geom = _resolve_nanopillar_parameters(params)
        pillar_diameter_um = geom["pillar_diameter_um"]
        pillar_edge_to_edge_spacing_um = geom["pillar_edge_to_edge_spacing_um"]
        pillar_intensity_factor = geom["pillar_intensity_factor"]
        background_intensity_factor = geom["background_intensity_factor"]

        pattern_final = _generate_nanopillar_pattern(
            shape=final_fov_shape,
            pixel_size_nm=pixel_size_nm,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
            params=params,
            layout_extent_nm=layout_extent_nm,
        )

        pattern_os = _generate_nanopillar_pattern(
            shape=fov_shape_os,
            pixel_size_nm=pixel_size_nm / os_factor,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
            params=params,
            layout_extent_nm=layout_extent_nm,
        )

    else:
        raise ValueError(
            f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
                "Supported models are 'none', 'gold_holes', and 'nanopillars'."
        )

    E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os)).astype(np.complex128)
    E_ref_final = (E_ref_amplitude * np.sqrt(pattern_final)).astype(np.complex128)

    background_final = (background_intensity * pattern_final).astype(float)

    return E_ref_os, E_ref_final, background_final


def compute_contrast_scale_for_frame(
    params: dict,
    frame_index: int,
    num_frames: int,
) -> float:
    """
    Return the multiplicative substrate-pattern contrast scale for a frame.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive when computing contrast scale.")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(
            f"frame_index={frame_index} is out of range for num_frames={num_frames}."
        )

    model_raw = params.get("sample_environment_pattern_contrast_model", "static"
    )
    model = str(model_raw).strip().lower()

    if model == "static":
        return 1.0

    if model == "time_dependent":
        amplitude = float(params.get("sample_environment_pattern_contrast_amplitude", 0.0,
        ))
        if amplitude <= 0.0:
            return 1.0
        if amplitude > 1.0:
            amplitude = 1.0

        if num_frames == 1:
            t_frac = 0.0
        else:
            t_frac = frame_index / float(num_frames - 1)

        alpha = 1.0 - amplitude * t_frac
        return float(alpha)

    raise ValueError(
        f"Unsupported sample_environment_pattern_contrast_model '{model_raw}'. "
        "Supported models are 'static' and 'time_dependent'."
    )
