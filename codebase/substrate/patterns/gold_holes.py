"""gold holes substrate-pattern helpers."""

from __future__ import annotations

from config import param_value

from ._shared import (
    Optional,
    _dimension_factor,
    _dimension_um,
    _dimension_um_from_keys,
    _pattern_dimensions,
    np,
)
from .circular_params import _resolve_nanopillar_parameters

def _circular_feature_geometry(params: dict, pattern_model: str) -> dict:
    if pattern_model in {"gold_holes", "holey_carbon"}:
        geom = _resolve_gold_hole_parameters(params)
        return {
            "pitch_um": float(geom["pitch_um"]),
            "radius_um": float(geom["radius_um"]),
            "feature_is_material": False,
            "feature_factor": float(geom["hole_intensity_factor"]),
            "background_factor": float(geom["gold_intensity_factor"]),
        }
    if pattern_model == "nanopillars":
        geom = _resolve_nanopillar_parameters(params)
        return {
            "pitch_um": float(geom["pitch_um"]),
            "radius_um": float(geom["radius_um"]),
            "feature_is_material": True,
            "feature_factor": float(geom["pillar_intensity_factor"]),
            "background_factor": float(geom["background_intensity_factor"]),
        }
    if pattern_model == "fiducial_dots":
        diameter_um = _dimension_um_from_keys(
            params,
            "fiducial_dot_diameter_um",
            "dot_diameter_nm",
            _dimension_um(params, "pillar_diameter_um"),
        )
        if "fiducial_dot_pitch_um" in _pattern_dimensions(params) or "pitch_nm" in _pattern_dimensions(params):
            pitch_um = _dimension_um_from_keys(params, "fiducial_dot_pitch_um", "pitch_nm", diameter_um + _dimension_um(params, "fiducial_dot_edge_to_edge_spacing_um"))
        else:
            pitch_um = None
        spacing_um = float(_dimension_um(params, "fiducial_dot_edge_to_edge_spacing_um"))
        if not np.isfinite(spacing_um) or spacing_um < 0.0:
            raise ValueError("fiducial_dot_edge_to_edge_spacing_um must be finite and non-negative.")
        return {
            "pitch_um": float(pitch_um if pitch_um is not None else diameter_um + spacing_um),
            "radius_um": 0.5 * diameter_um,
            "feature_is_material": True,
            "feature_factor": _dimension_factor(params, "fiducial_dot_intensity_factor"),
            "background_factor": _dimension_factor(params, "fiducial_background_intensity_factor"),
        }
    if pattern_model == "patterned_coverslip":
        diameter_um = _dimension_um_from_keys(params, "coverslip_patch_diameter_um", "coverslip_patch_diameter_nm", _dimension_um(params, "coverslip_patch_diameter_um"))
        if "coverslip_patch_pitch_um" in _pattern_dimensions(params) or "coverslip_patch_pitch_nm" in _pattern_dimensions(params):
            pitch_um = _dimension_um_from_keys(params, "coverslip_patch_pitch_um", "coverslip_patch_pitch_nm", diameter_um + _dimension_um(params, "coverslip_patch_edge_to_edge_spacing_um"))
        else:
            pitch_um = None
        spacing_um = float(_pattern_dimensions(params)["coverslip_patch_edge_to_edge_spacing_um"])
        if not np.isfinite(spacing_um) or spacing_um < 0.0:
            raise ValueError("coverslip_patch_edge_to_edge_spacing_um must be finite and non-negative.")
        return {
            "pitch_um": float(pitch_um if pitch_um is not None else diameter_um + spacing_um),
            "radius_um": 0.5 * diameter_um,
            "feature_is_material": True,
            "feature_factor": _dimension_factor(params, "coverslip_patch_intensity_factor"),
            "background_factor": _dimension_factor(params, "coverslip_background_intensity_factor"),
        }
    raise ValueError(f"Pattern {pattern_model!r} is not a circular-feature pattern.")

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
    from .layout import (
        _classify_grid_against_feature,
        _feature_radius_bound_um,
        _get_feature_layout_for_params,
    )

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

        # Rasterize per feature over local bounding boxes instead of scanning
        # every pixel against a 3x3 lattice neighborhood.
        for feature in layout.features_by_cell.values():
            radius_bound_um = _feature_radius_bound_um(feature)
            if radius_bound_um <= 0.0:
                continue
            x0 = int(np.searchsorted(x_um, float(feature.center_x_um) - radius_bound_um, side="left"))
            x1 = int(np.searchsorted(x_um, float(feature.center_x_um) + radius_bound_um, side="right"))
            y0 = int(np.searchsorted(y_um, float(feature.center_y_um) - radius_bound_um, side="left"))
            y1 = int(np.searchsorted(y_um, float(feature.center_y_um) + radius_bound_um, side="right"))
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            if x0 >= x1 or y0 >= y1:
                continue
            local_mask = _classify_grid_against_feature(
                feature,
                X_um[y0:y1, x0:x1],
                Y_um[y0:y1, x0:x1],
            )
            pattern[y0:y1, x0:x1][local_mask] = hole_intensity_factor

    mean_val = float(pattern.mean())
    if mean_val > 0.0:
        pattern /= mean_val

    return pattern

def _resolve_gold_hole_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for the gold film with
    circular holes from the global PARAMS dictionary.
    """
    dims = param_value(params, "sample_environment_pattern_dimensions")
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['sample_environment_pattern_dimensions'] must be a dictionary when "
            "using sample_environment_pattern 'gold_holes'."
        )

    substrate_preset_raw = param_value(params, "sample_environment_pattern_preset"
    )
    substrate_preset = str(substrate_preset_raw).strip().lower()

    hole_diameter_um = float(dims["hole_diameter_um"])
    hole_edge_to_edge_spacing_um = float(dims["hole_edge_to_edge_spacing_um"])

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

    hole_intensity_factor = float(dims["hole_intensity_factor"])
    gold_intensity_factor = float(dims["gold_intensity_factor"])

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

__all__ = []
