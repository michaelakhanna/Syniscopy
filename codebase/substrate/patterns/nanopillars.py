"""nanopillars substrate-pattern helpers."""

from __future__ import annotations

from ._shared import (
    Optional,
    _dimension_factor,
    _dimension_um,
    _dimension_um_from_keys,
    _pattern_dimensions,
    np,
)
from .circular_params import _resolve_nanopillar_parameters

def _bar_geometry(params: dict, pattern_model: str) -> dict:
    if pattern_model == "grid_bars":
        return {
            "pitch_um": _dimension_um_from_keys(params, "grid_pitch_um", "pitch_nm", _dimension_um(params, "grid_pitch_um")),
            "width_um": _dimension_um_from_keys(params, "grid_bar_width_um", "bar_width_nm", _dimension_um(params, "grid_bar_width_um")),
            "orientation": "both",
            "feature_factor": _dimension_factor(params, "grid_bar_intensity_factor"),
            "background_factor": _dimension_factor(params, "grid_background_intensity_factor"),
        }
    if pattern_model == "microfluidic_walls":
        return {
            "pitch_um": _dimension_um_from_keys(params, "microfluidic_channel_pitch_um", "microfluidic_channel_pitch_nm", _dimension_um(params, "microfluidic_channel_pitch_um")),
            "width_um": _dimension_um_from_keys(params, "microfluidic_wall_width_um", "microfluidic_wall_width_nm", _dimension_um(params, "microfluidic_wall_width_um")),
            "orientation": str(_pattern_dimensions(params)["microfluidic_wall_orientation"]).strip().lower(),
            "feature_factor": _dimension_factor(params, "microfluidic_wall_intensity_factor"),
            "background_factor": _dimension_factor(params, "microfluidic_channel_intensity_factor"),
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
        - When a parameters dictionary is provided, the same shared feature layout
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

__all__ = []
