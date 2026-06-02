"""registry substrate-pattern helpers."""

from __future__ import annotations

from ._shared import (
    _BAR_MATERIAL_PATTERNS,
    _CIRCULAR_MATERIAL_PATTERNS,
    _CIRCULAR_VOID_PATTERNS,
    _PATTERN_DEFAULT_PRESETS,
    _bar_solid_mask_from_coordinates,
    _centered_pattern_grid,
    np,
)

def _substrate_pattern_is_enabled(params: dict) -> bool:
    return (
        bool(params.get("sample_environment_enabled", True))
        and bool(params.get("sample_environment_pattern_enabled", False))
    )

def canonical_sample_environment_pattern_and_preset(pattern: object, preset: object = "empty_background") -> tuple[str, str]:
    """Normalize public sample-environment preset spellings without collapsing geometry."""
    p = str(pattern).strip().lower()
    q = str(preset).strip().lower()
    if p in _PATTERN_DEFAULT_PRESETS and q in {
        "",
        "default",
        p,
        "default_gold_holes",
        "default_nanopillars",
    }:
        q = _PATTERN_DEFAULT_PRESETS[p]
    return p, q

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
    from .gold_holes import _circular_feature_geometry
    from .layout import (
        _classify_grid_against_feature,
        _feature_radius_bound_um,
        _get_feature_layout_for_params,
    )
    from .nanopillars import _bar_geometry

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
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background")
    pattern_model, substrate_preset = canonical_sample_environment_pattern_and_preset(
        pattern_model_raw, substrate_preset_raw
    )

    uniform_height = np.zeros((height, width), dtype=float)
    uniform_fraction = np.ones((height, width), dtype=float)
    if (
        not bool(params.get("sample_environment_enabled", True))
        or not _substrate_pattern_is_enabled(params)
        or substrate_preset == "empty_background"
        or pattern_model == "none"
    ):
        return uniform_height, uniform_fraction, "uniform"

    if pattern_model in _CIRCULAR_VOID_PATTERNS | _CIRCULAR_MATERIAL_PATTERNS:
        expected_preset = _PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset != expected_preset:
            raise ValueError(
                f"sample_environment_pattern={pattern_model!r} received invalid "
                f"preset {substrate_preset_raw!r}."
            )
        geom = _circular_feature_geometry(params, pattern_model)
        feature_is_material = bool(geom["feature_is_material"])
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model=pattern_model,
            pitch_um=float(geom["pitch_um"]),
            nominal_radius_um=float(geom["radius_um"]),
            layout_extent_nm=layout_extent_nm,
        )
        pixel_size_um = pixel_size_nm * 1e-3
        x_um = (np.arange(width, dtype=float) - width / 2.0 + 0.5) * pixel_size_um
        y_um = (np.arange(height, dtype=float) - height / 2.0 + 0.5) * pixel_size_um
        material_fraction = np.zeros((height, width), dtype=float)
        if not feature_is_material:
            material_fraction.fill(1.0)
        X_um, Y_um = np.meshgrid(x_um, y_um)
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
            if feature_is_material:
                material_fraction[y0:y1, x0:x1][local_mask] = 1.0
            else:
                material_fraction[y0:y1, x0:x1][local_mask] = 0.0
        height_map = layer_thickness_nm * material_fraction
        return height_map.astype(float), material_fraction.astype(float), pattern_model

    if pattern_model in _BAR_MATERIAL_PATTERNS:
        expected_preset = _PATTERN_DEFAULT_PRESETS.get(pattern_model)
        if substrate_preset != expected_preset:
            raise ValueError(
                f"sample_environment_pattern={pattern_model!r} received invalid "
                f"preset {substrate_preset_raw!r}."
            )
        geom = _bar_geometry(params, pattern_model)
        X_um, Y_um = _centered_pattern_grid((height, width), pixel_size_nm)
        material_fraction = _bar_solid_mask_from_coordinates(
            X_um,
            Y_um,
            pitch_um=float(geom["pitch_um"]),
            width_um=float(geom["width_um"]),
            orientation=str(geom["orientation"]),
        ).astype(float)
        height_map = layer_thickness_nm * material_fraction
        return height_map.astype(float), material_fraction.astype(float), pattern_model

    raise ValueError(
        f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
        "Supported models are 'none', 'gold_holes', 'nanopillars', "
        "'fiducial_dots', 'grid_bars', 'holey_carbon', "
        "'microfluidic_walls', and 'patterned_coverslip'."
    )

__all__ = ['canonical_sample_environment_pattern_and_preset', 'generate_sample_environment_pattern_maps']
