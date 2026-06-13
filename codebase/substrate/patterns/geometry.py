"""geometry substrate-pattern helpers."""

from __future__ import annotations
from config import SampleEnvironmentSettings, SamplingGeometry
from param_schema.sample_environment import PATTERN_DEFAULT_PRESETS

from ._shared import (
    _BAR_MATERIAL_PATTERNS,
    _CIRCULAR_MATERIAL_PATTERNS,
    _CIRCULAR_VOID_PATTERNS,
    _REFLECTION_BOUNDARY_BISECTION_STEPS,
    _bar_solid_mask_from_coordinates,
    _canonical_bar_orientation,
    _substrate_pattern_is_enabled,
    canonical_sample_environment_pattern_and_preset,
    math,
    np,
)

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
    sampling = SamplingGeometry.from_params(params)
    img_size_pixels = sampling.image_size_pixels
    pixel_size_nm = sampling.detector_pixel_size_nm

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
    from .gold_holes import _circular_feature_geometry
    from .layout import _classify_point_against_layout, _get_feature_layout_for_params
    from .nanopillars import _bar_geometry

    substrate_enabled = _substrate_pattern_is_enabled(params)
    clearance_um = max(float(clearance_nm), 0.0) * 1e-3

    sample_environment = SampleEnvironmentSettings.from_params(params)
    pattern_model, substrate_preset = canonical_sample_environment_pattern_and_preset(
        sample_environment.pattern,
        sample_environment.pattern_preset,
    )

    if (
        not substrate_enabled
        or substrate_preset == "empty_background"
        or pattern_model == "none"
    ):
        return False

    if pattern_model in _CIRCULAR_VOID_PATTERNS | _CIRCULAR_MATERIAL_PATTERNS:
        geom = _circular_feature_geometry(params, pattern_model)
        _, _, _, x_um, y_um, _ = _map_position_nm_to_pattern_unit_cell(
            params,
            x_nm,
            y_nm,
            float(geom["pitch_um"]),
        )
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model=pattern_model,
            pitch_um=float(geom["pitch_um"]),
            nominal_radius_um=float(geom["radius_um"]),
        )
        if bool(geom["feature_is_material"]):
            return _classify_point_against_layout(
                layout,
                x_um,
                y_um,
                boundary_offset_um=clearance_um,
            )
        inside_void = _classify_point_against_layout(
            layout,
            x_um,
            y_um,
            boundary_offset_um=-clearance_um,
        )
        return not inside_void

    if pattern_model in _BAR_MATERIAL_PATTERNS:
        geom = _bar_geometry(params, pattern_model)
        _, _, _, x_um, y_um, _ = _map_position_nm_to_pattern_unit_cell(
            params,
            x_nm,
            y_nm,
            float(geom["pitch_um"]),
        )
        return bool(
            _bar_solid_mask_from_coordinates(
                np.asarray(x_um),
                np.asarray(y_um),
                pitch_um=float(geom["pitch_um"]),
                width_um=float(geom["width_um"]),
                orientation=str(geom["orientation"]),
                clearance_um=clearance_um,
            )
        )

    raise ValueError(
        f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
        "Supported models are 'none', 'gold_holes', 'nanopillars', "
                "'fiducial_dots', 'grid_bars', 'holey_carbon', "
                "'microfluidic_walls', and 'patterned_coverslip'."
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
    from .gold_holes import _circular_feature_geometry, _resolve_gold_hole_parameters
    from .layout import (
        _compute_feature_boundary_radius,
        _get_feature_layout_for_params,
        _nearest_feature_and_vector,
    )
    from .nanopillars import _bar_geometry, _resolve_nanopillar_parameters

    clearance_um = max(float(clearance_nm), 0.0) * 1e-3
    if not is_position_in_substrate_solid(
        params,
        x_nm,
        y_nm,
        clearance_nm=clearance_nm,
    ):
        return float(x_nm), float(y_nm)

    substrate_enabled = _substrate_pattern_is_enabled(params)
    sample_environment = SampleEnvironmentSettings.from_params(params)
    pattern_model, substrate_preset = canonical_sample_environment_pattern_and_preset(
        sample_environment.pattern,
        sample_environment.pattern_preset,
    )

    if not substrate_enabled:
        return float(x_nm), float(y_nm)

    if pattern_model in _CIRCULAR_VOID_PATTERNS | _CIRCULAR_MATERIAL_PATTERNS:
        expected_preset = PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset != expected_preset:
            return float(x_nm), float(y_nm)
        geom = _circular_feature_geometry(params, pattern_model)
        pitch_um = float(geom["pitch_um"])
        nominal_radius_um = float(geom["radius_um"])
        feature_is_material = bool(geom["feature_is_material"])

        _, _, _, x_um, y_um, center_nm = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model=pattern_model,
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )
        feature, dx, dy = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is None:
            return float(x_nm), float(y_nm)

        dist_um = math.hypot(dx, dy)
        if dist_um == 0.0:
            dx = feature.r_x_um
            dy = 0.0
            dist_um = max(feature.r_x_um, 1.0e-9)
        r_boundary_um = _compute_feature_boundary_radius(feature, dx, dy)
        if r_boundary_um <= 0.0:
            r_boundary_um = min(feature.r_x_um, feature.r_y_um)

        epsilon_um = 1.0e-3
        if feature_is_material:
            r_target_um = r_boundary_um + clearance_um + epsilon_um
        else:
            r_target_um = max(r_boundary_um - clearance_um - epsilon_um, 0.0)
        scale = r_target_um / dist_um
        new_x_um = feature.center_x_um + dx * scale
        new_y_um = feature.center_y_um + dy * scale
        new_x_nm = new_x_um * 1.0e3 + center_nm
        new_y_nm = new_y_um * 1.0e3 + center_nm

        if is_position_in_substrate_solid(
            params,
            new_x_nm,
            new_y_nm,
            clearance_nm=clearance_nm,
        ):
            if feature_is_material:
                r_target_um = r_boundary_um + clearance_um + max(nominal_radius_um, 1.0e-3)
                scale = r_target_um / dist_um
                new_x_um = feature.center_x_um + dx * scale
                new_y_um = feature.center_y_um + dy * scale
            else:
                new_x_um = feature.center_x_um
                new_y_um = feature.center_y_um
            new_x_nm = new_x_um * 1.0e3 + center_nm
            new_y_nm = new_y_um * 1.0e3 + center_nm

        return float(new_x_nm), float(new_y_nm)

    if pattern_model in _BAR_MATERIAL_PATTERNS:
        expected_preset = PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset != expected_preset:
            return float(x_nm), float(y_nm)
        geom = _bar_geometry(params, pattern_model)
        pitch_um = float(geom["pitch_um"])
        width_um = float(geom["width_um"])
        orientation = _canonical_bar_orientation(geom["orientation"])
        _, _, _, x_um, y_um, center_nm = _map_position_nm_to_pattern_unit_cell(
            params, x_nm, y_nm, pitch_um
        )
        half_width = 0.5 * width_um + clearance_um
        epsilon_um = 1.0e-3

        def _project_axis(value_um: float) -> float:
            rel = ((float(value_um) + 0.5 * pitch_um) % pitch_um) - 0.5 * pitch_um
            bar_center = float(value_um) - rel
            sign = 1.0 if rel >= 0.0 else -1.0
            if abs(rel) <= half_width:
                return bar_center + sign * (half_width + epsilon_um)
            return float(value_um)

        new_x_um = float(x_um)
        new_y_um = float(y_um)
        if orientation in {"vertical", "both"}:
            new_x_um = _project_axis(new_x_um)
        if orientation in {"horizontal", "both"}:
            new_y_um = _project_axis(new_y_um)
        new_x_nm = new_x_um * 1.0e3 + center_nm
        new_y_nm = new_y_um * 1.0e3 + center_nm
        return float(new_x_nm), float(new_y_nm)

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
        and substrate_preset == "default_nanopillars"
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
    from .gold_holes import _circular_feature_geometry, _resolve_gold_hole_parameters
    from .layout import _get_feature_layout_for_params, _nearest_feature_and_vector
    from .nanopillars import _bar_geometry, _resolve_nanopillar_parameters

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
    sample_environment = SampleEnvironmentSettings.from_params(params)
    pattern_model, substrate_preset = canonical_sample_environment_pattern_and_preset(
        sample_environment.pattern,
        sample_environment.pattern_preset,
    )

    n_x_world = 0.0
    n_y_world = 0.0
    if pattern_model in _CIRCULAR_VOID_PATTERNS | _CIRCULAR_MATERIAL_PATTERNS:
        expected_preset = PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset == expected_preset:
            geom = _circular_feature_geometry(params, pattern_model)
            pitch_um = float(geom["pitch_um"])
            nominal_radius_um = float(geom["radius_um"])
            _, _, _, x_um, y_um, _center_nm = _map_position_nm_to_pattern_unit_cell(
                params, bx_nm, by_nm, pitch_um,
            )
            layout = _get_feature_layout_for_params(
                params=params,
                pattern_model=pattern_model,
                pitch_um=pitch_um,
                nominal_radius_um=nominal_radius_um,
            )
            feature, dx_um, dy_um = _nearest_feature_and_vector(layout, x_um, y_um)
            if feature is not None:
                r = math.hypot(dx_um, dy_um)
                if r > 0.0:
                    if bool(geom["feature_is_material"]):
                        n_x_world = -dx_um / r
                        n_y_world = -dy_um / r
                    else:
                        n_x_world = dx_um / r
                        n_y_world = dy_um / r
    elif pattern_model in _BAR_MATERIAL_PATTERNS:
        expected_preset = PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset == expected_preset:
            geom = _bar_geometry(params, pattern_model)
            pitch_um = float(geom["pitch_um"])
            half_width = 0.5 * float(geom["width_um"]) + max(float(clearance_nm), 0.0) * 1.0e-3
            orientation = _canonical_bar_orientation(geom["orientation"])
            _, _, _, x_um, y_um, _center_nm = _map_position_nm_to_pattern_unit_cell(
                params, bx_nm, by_nm, pitch_um,
            )

            def _bar_rel(value_um: float) -> float:
                return ((float(value_um) + 0.5 * pitch_um) % pitch_um) - 0.5 * pitch_um

            rel_x = _bar_rel(x_um)
            rel_y = _bar_rel(y_um)
            candidates = []
            if orientation in {"vertical", "both"} and abs(rel_x) <= half_width + 1.0e-6:
                candidates.append((abs(abs(rel_x) - half_width), -1.0 if rel_x >= 0.0 else 1.0, 0.0))
            if orientation in {"horizontal", "both"} and abs(rel_y) <= half_width + 1.0e-6:
                candidates.append((abs(abs(rel_y) - half_width), 0.0, -1.0 if rel_y >= 0.0 else 1.0))
            if candidates:
                _, n_x_world, n_y_world = min(candidates, key=lambda item: item[0])
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
        and substrate_preset == "default_nanopillars"
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

__all__ = ['is_position_in_substrate_solid', 'project_position_to_fluid_region', 'reflect_position_across_substrate_boundary']
