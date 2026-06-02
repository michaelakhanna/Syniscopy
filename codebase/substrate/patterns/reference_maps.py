"""reference maps substrate-pattern helpers."""

from __future__ import annotations

from ._shared import (
    _BAR_MATERIAL_PATTERNS,
    _CIRCULAR_MATERIAL_PATTERNS,
    _CIRCULAR_VOID_PATTERNS,
    _pattern_intensity_from_material_fraction,
    np,
)

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
    from .gold_holes import _circular_feature_geometry
    from .nanopillars import _bar_geometry
    from .registry import (
        _substrate_pattern_is_enabled,
        canonical_sample_environment_pattern_and_preset,
        generate_sample_environment_pattern_maps,
    )

    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    substrate_enabled = _substrate_pattern_is_enabled(params)

    pattern_model_raw = params.get("sample_environment_pattern", "none"
    )
    substrate_preset_raw = params.get("sample_environment_pattern_preset", "empty_background"
    )

    pattern_model, substrate_preset = canonical_sample_environment_pattern_and_preset(
        pattern_model_raw, substrate_preset_raw
    )

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

    if pattern_model in _CIRCULAR_VOID_PATTERNS | _CIRCULAR_MATERIAL_PATTERNS:
        geom = _circular_feature_geometry(params, pattern_model)
        if bool(geom["feature_is_material"]):
            material_factor = float(geom["feature_factor"])
            background_factor = float(geom["background_factor"])
        else:
            material_factor = float(geom["background_factor"])
            background_factor = float(geom["feature_factor"])
    elif pattern_model in _BAR_MATERIAL_PATTERNS:
        geom = _bar_geometry(params, pattern_model)
        material_factor = float(geom["feature_factor"])
        background_factor = float(geom["background_factor"])
    else:
        raise ValueError(
            f"Unsupported sample_environment_pattern '{pattern_model_raw}'. "
                "Supported models are 'none', 'gold_holes', 'nanopillars', "
                "'fiducial_dots', 'grid_bars', 'holey_carbon', "
                "'microfluidic_walls', and 'patterned_coverslip'."
        )
    _, fraction_final, _ = generate_sample_environment_pattern_maps(
        params,
        final_fov_shape,
        pixel_size_nm,
        layer_thickness_nm=1.0,
        layout_extent_nm=layout_extent_nm,
    )
    _, fraction_os, _ = generate_sample_environment_pattern_maps(
        params,
        fov_shape_os,
        pixel_size_nm / os_factor,
        layer_thickness_nm=1.0,
        layout_extent_nm=layout_extent_nm,
    )
    pattern_final = _pattern_intensity_from_material_fraction(
        fraction_final,
        material_factor=material_factor,
        background_factor=background_factor,
    )
    pattern_os = _pattern_intensity_from_material_fraction(
        fraction_os,
        material_factor=material_factor,
        background_factor=background_factor,
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

__all__ = ['generate_reference_and_background_maps', 'compute_contrast_scale_for_frame']
