"""Shared circular-feature substrate-pattern parameter resolvers."""

from __future__ import annotations

import numpy as np

from config import SampleEnvironmentSettings


def _resolve_nanopillar_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for a circular nanopillar
    array from the global parameters dictionary.
    """
    sample_environment = SampleEnvironmentSettings.from_params(params)
    dims = sample_environment.pattern_dimensions
    substrate_preset = sample_environment.pattern_preset

    pillar_diameter_um = float(dims["pillar_diameter_um"])
    pillar_edge_to_edge_spacing_um = float(dims["pillar_edge_to_edge_spacing_um"])

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

    pillar_intensity_factor = float(dims["pillar_intensity_factor"])
    background_intensity_factor = float(dims["background_intensity_factor"])
    if not np.isfinite(pillar_intensity_factor) or pillar_intensity_factor <= 0.0:
        raise ValueError("pillar_intensity_factor must be finite and positive.")
    if not np.isfinite(background_intensity_factor) or background_intensity_factor <= 0.0:
        raise ValueError("background_intensity_factor must be finite and positive.")

    return {
        "pillar_diameter_um": pillar_diameter_um,
        "pillar_edge_to_edge_spacing_um": pillar_edge_to_edge_spacing_um,
        "pillar_intensity_factor": pillar_intensity_factor,
        "background_intensity_factor": background_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": 0.5 * pillar_diameter_um,
        "substrate_preset": substrate_preset,
    }


__all__ = ["_resolve_nanopillar_parameters"]
