"""nanopillars substrate-pattern helpers."""

from __future__ import annotations

from ._shared import (
    Optional,
    _dimension_factor,
    _dimension_um_from_keys,
    _pattern_dimensions,
    np,
)

def _bar_geometry(params: dict, pattern_model: str) -> dict:
    if pattern_model == "grid_bars":
        return {
            "pitch_um": _dimension_um_from_keys(params, "grid_pitch_um", "pitch_nm", 5.0),
            "width_um": _dimension_um_from_keys(params, "grid_bar_width_um", "bar_width_nm", 0.5),
            "orientation": "both",
            "feature_factor": _dimension_factor(params, "grid_bar_intensity_factor", 1.25),
            "background_factor": _dimension_factor(params, "grid_background_intensity_factor", 1.0),
        }
    if pattern_model == "microfluidic_walls":
        return {
            "pitch_um": _dimension_um_from_keys(params, "microfluidic_channel_pitch_um", "microfluidic_channel_pitch_nm", 10.0),
            "width_um": _dimension_um_from_keys(params, "microfluidic_wall_width_um", "microfluidic_wall_width_nm", 1.0),
            "orientation": str(_pattern_dimensions(params).get("microfluidic_wall_orientation", "vertical")).strip().lower(),
            "feature_factor": _dimension_factor(params, "microfluidic_wall_intensity_factor", 1.2),
            "background_factor": _dimension_factor(params, "microfluidic_channel_intensity_factor", 1.0),
        }
    raise ValueError(f"Pattern {pattern_model!r} is not a bar/wall pattern.")

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
    from .gold_holes import _generate_gold_hole_pattern

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

__all__ = []
