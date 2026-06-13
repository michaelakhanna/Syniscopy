"""Axis decomposition of the microscope parameter surface.

Why this exists
---------------
The recurring modelling error in this codebase is collapsing a *product of
independent axes* into one named entity. ``modality`` was secretly
``contrast-mechanism x instrument x representation``; the fix was to make the
microscope the unit. But the microscope's parameter surface is itself a product
of independent axes that the report still treats as one flat dict:

* **SAMPLE**       -- the shared physical specimen, environment, medium, scene
                      (shared across microscopes by contract).
* **INSTRUMENT**   -- optical/electron hardware: NA, wavelength, objective,
                      immersion, illumination geometry, backend selection.
* **OPERATING_POINT** -- how the instrument is *run*: dose / photon budget,
                      exposure, frame rate, frame count, detected-quanta budget.
* **DETECTOR**     -- the readout chain: read noise, gain, QE, bit depth, dark
                      current, fixed-pattern terms.
* **SAMPLING**     -- the physical detector grid: pixel pitch, field size.
* **ANALYSIS**     -- numerical / inference conventions that are *not* physics:
                      pupil samples, PSF oversampling, derivative step/mode,
                      random seed, background-subtraction method.

The in-flight microscope refactor already separates SAMPLE (shared scene) from
"microscope-local" params, but it folds OPERATING_POINT, DETECTOR, SAMPLING, and
ANALYSIS together with INSTRUMENT. That makes first-class experimental questions
awkward to express: "same instrument, twice the dose", "same instrument and
dose, finer derivative step", "same everything but a different camera". Naming
the axes makes those questions one-liners and makes the Cramer--Rao bound's
dependence on each axis explicit.

This module is the source owner for report-facing parameter-axis classification.
It classifies via a small explicit canonical map plus ordered pattern rules, and
anything it cannot place lands in ``UNCLASSIFIED`` rather than being silently
mis-assigned. Lab Fisher microscope overlays consume this taxonomy directly, so
sample/scene authority is not duplicated in report-local key lists.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# -- axis names -------------------------------------------------------------
SAMPLE = "sample"
INSTRUMENT = "instrument"
OPERATING_POINT = "operating_point"
DETECTOR = "detector"
SAMPLING = "sampling"
ANALYSIS = "analysis"
UNCLASSIFIED = "unclassified"

ALL_AXES = (SAMPLE, INSTRUMENT, OPERATING_POINT, DETECTOR, SAMPLING, ANALYSIS)

# Axes a per-microscope overlay may legitimately set. SAMPLE is shared scene and
# must be supplied once; this generalizes the report's shared-scene guard.
SHARED_AXES = frozenset({SAMPLE})
MICROSCOPE_LOCAL_AXES = frozenset({INSTRUMENT, OPERATING_POINT, DETECTOR, SAMPLING, ANALYSIS})

# -- explicit canonical keys (the ones users actually set) ------------------
EXPLICIT_AXIS: dict[str, str] = {
    # sample / scene (shared)
    "particles": SAMPLE, "components": SAMPLE, "material": SAMPLE,
    "diameter_nm": SAMPLE, "refractive_index": SAMPLE,
    "medium_material": SAMPLE, "refractive_index_medium": SAMPLE,
    "viscosity_Pa_s": SAMPLE, "temperature_K": SAMPLE,
    # instrument (optical / electron hardware)
    "numerical_aperture": INSTRUMENT, "wavelength_nm": INSTRUMENT,
    "magnification": INSTRUMENT, "objective_model": INSTRUMENT,
    "objective_focal_length_mm": INSTRUMENT, "refractive_index_immersion": INSTRUMENT,
    "optical_field_backend": INSTRUMENT, "optical_scattering_model": INSTRUMENT,
    "optical_cluster_scattering_model": INSTRUMENT,
    "optical_cluster_dda_voxel_size_nm": ANALYSIS,
    "optical_cluster_dda_max_dipoles": ANALYSIS,
    "imaging_model": INSTRUMENT,
    "tem_acceleration_kV": INSTRUMENT, "fluorescence_backend": INSTRUMENT,
    "tem_backend": INSTRUMENT, "sem_backend": INSTRUMENT, "sem_model": INSTRUMENT,
    # operating point (how the instrument is run)
    "background_intensity": OPERATING_POINT, "fps": OPERATING_POINT,
    "duration_seconds": OPERATING_POINT, "num_frames": OPERATING_POINT,
    "fluorescence_photon_count_scale": OPERATING_POINT,
    "tem_dose_per_pixel": OPERATING_POINT, "sem_electrons_per_pixel": OPERATING_POINT,
    "channels": OPERATING_POINT,
    # detector (readout chain)
    "read_noise_counts": DETECTOR, "camera_gain_e_per_count": DETECTOR,
    "detector_qe": DETECTOR, "bit_depth": DETECTOR,
    "emccd_excess_noise_factor": DETECTOR, "emccd_gain": DETECTOR,
    "dark_current_e_per_pixel_per_s": DETECTOR,
    "fixed_pattern_gain_std": DETECTOR, "fixed_pattern_offset_counts": DETECTOR,
    "fixed_pattern_gain_map": DETECTOR, "fixed_pattern_offset_map": DETECTOR,
    # sampling (physical detector grid)
    "pixel_size_nm": SAMPLING, "image_size_pixels": SAMPLING,
    # analysis (numerical / inference convention, not physics)
    "pupil_samples": ANALYSIS, "psf_oversampling_factor": ANALYSIS,
    "random_seed": ANALYSIS,
    "background_subtraction_method": ANALYSIS, "dynamic_bayesian_enabled": ANALYSIS,
}

# -- specific pattern rules (override even the schema group default) ---------
# These encode sub-group distinctions a coarse ``group`` would get wrong, e.g.
# operating-point and analysis keys that live in instrument/optics groups, or
# the detector-vs-operating split inside the "Noise" group. First match wins.
_SPECIFIC_PATTERN_RULES: tuple[tuple[str, str], ...] = (
    # sample / scene
    ("substrate", SAMPLE), ("sample_environment", SAMPLE), ("coverslip", SAMPLE),
    ("source_volume", SAMPLE), ("source_z", SAMPLE), ("motion", SAMPLE),
    ("trajectory", SAMPLE), ("initial_position", SAMPLE), ("particle_spec", SAMPLE),
    ("bulk_substrate", SAMPLE), ("carbon_film", SAMPLE), ("bar_height", SAMPLE),
    ("dot_height", SAMPLE), ("medium", SAMPLE), ("viscosity", SAMPLE),
    ("fiducial", SAMPLE), ("drift", SAMPLE), ("vibration", SAMPLE),
    ("mounting_interface", SAMPLE), ("initial_z_span", SAMPLE), ("z_motion", SAMPLE),
    ("z_span", SAMPLE), ("hole_", SAMPLE), ("grid_", SAMPLE), ("gold_", SAMPLE), ("nanopillar", SAMPLE),
    ("pillar", SAMPLE), ("roughness", SAMPLE), ("speckle", SAMPLE), ("pattern_", SAMPLE),
    ("autofluorescence", SAMPLE),
    # operating point (run-time budget / exposure / scheduling) -- must beat
    # the "Advanced modality"->INSTRUMENT and "Noise"->DETECTOR defaults.
    ("dose", OPERATING_POINT), ("photon_count", OPERATING_POINT),
    ("electrons_per_pixel", OPERATING_POINT), ("exposure", OPERATING_POINT),
    ("detected_quanta", OPERATING_POINT), ("broadband", OPERATING_POINT),
    ("dynamic_process_noise", OPERATING_POINT), ("dynamic_initial_variance", OPERATING_POINT),
    ("background_intensity", OPERATING_POINT),
    # detector readout specifics (some live outside the "Noise" group)
    ("read_noise", DETECTOR), ("camera_gain", DETECTOR), ("emccd", DETECTOR),
    ("dark_current", DETECTOR), ("dark_offset", DETECTOR), ("dark_frame", DETECTOR),
    ("adc_quantization", DETECTOR), ("background_offset", DETECTOR),
    ("fixed_pattern", DETECTOR), ("hot_pixel", DETECTOR), ("scan_line", DETECTOR),
    ("scanline", DETECTOR), ("_qe", DETECTOR), ("gain_map", DETECTOR),
    ("empirical_background", DETECTOR), ("flat_field", DETECTOR), ("shading", DETECTOR),
    ("gaussian_noise", DETECTOR), ("poisson_noise", DETECTOR), ("read_out", DETECTOR),
    ("detector_input", DETECTOR), ("detector_noise", DETECTOR), ("detector_spectral", DETECTOR),
    # analysis / numerical convention / supervision / IO -- must beat the
    # "Optics"->INSTRUMENT default for pupil_samples/derivative/oversampling.
    ("oversampling", ANALYSIS), ("pupil_samples", ANALYSIS), ("derivative", ANALYSIS),
    ("fisher_", ANALYSIS), ("_seed", ANALYSIS), ("random_seed", ANALYSIS),
    ("dynamic_bayesian", ANALYSIS), ("dynamic_include_smoothing", ANALYSIS),
    ("background_subtraction", ANALYSIS), ("convergence", ANALYSIS),
    ("mask", ANALYSIS), ("supervision", ANALYSIS), ("loss_weight", ANALYSIS),
    ("ignore", ANALYSIS), ("clip_output", ANALYSIS), ("annotation", ANALYSIS),
)

# -- broad family fallbacks (only consulted after the schema group default) --
_BROAD_PATTERN_RULES: tuple[tuple[str, str], ...] = (
    ("detector_", DETECTOR),
    ("annular_dark_field", INSTRUMENT), ("dark_field", INSTRUMENT),
    ("bright_field", INSTRUMENT), ("dpc_", INSTRUMENT), ("zernike", INSTRUMENT),
    ("confocal", INSTRUMENT), ("illumination", INSTRUMENT), ("condenser", INSTRUMENT),
    ("objective", INSTRUMENT), ("aberration", INSTRUMENT), ("defocus", INSTRUMENT),
    ("apodization", INSTRUMENT), ("wavelength", INSTRUMENT), ("numerical_aperture", INSTRUMENT),
    ("tem_", INSTRUMENT), ("sem_", INSTRUMENT), ("fluorescence_excitation", INSTRUMENT),
    ("fluorescence_emission", INSTRUMENT), ("spectral_response", INSTRUMENT),
)


# -- schema-backed coarse default: PARAM_SCHEMA's own ``group`` -> axis -------
# The param schema already groups keys; this maps each group to its dominant
# axis as a *coarse default*. Explicit keys and specific patterns above override
# it for the known exceptions (e.g. pixel_size_nm in the "Optics" group is
# SAMPLING, num_frames in "Workflow" is OPERATING_POINT), so a mixed group is
# only the fallback, never the final word for a key we can place precisely.
GROUP_AXIS_DEFAULT: dict[str, str] = {
    "Imaging": INSTRUMENT,
    "Optics": INSTRUMENT,
    "Advanced modality": INSTRUMENT,
    "Fluorescence": SAMPLE,
    "Noise": DETECTOR,
    "Workflow": ANALYSIS,
    "Sample environment": SAMPLE,
    "Particle": SAMPLE,
    "Dynamics": SAMPLE,
    "Advanced supervision": ANALYSIS,
    "Mask": ANALYSIS,
    "Advanced Fisher": ANALYSIS,
}

_SCHEMA_GROUPS_CACHE: dict[str, str] | None = None


def param_schema_groups() -> dict[str, str]:
    """Best-effort ``{key: group}`` from PARAM_SCHEMA; ``{}`` if unavailable.

    Defensive: importing the param schema is optional, so the classifier keeps
    working standalone (pattern-only) when the schema cannot be imported (e.g.
    mid-refactor). The result is cached after the first successful load.
    """
    global _SCHEMA_GROUPS_CACHE
    if _SCHEMA_GROUPS_CACHE is not None:
        return _SCHEMA_GROUPS_CACHE
    groups: dict[str, str] = {}
    try:  # pragma: no cover - environment dependent
        from param_schema import PARAM_SCHEMA  # type: ignore

        for key, spec in PARAM_SCHEMA.items():
            group = spec.get("group") if isinstance(spec, dict) else getattr(spec, "group", None)
            if group:
                groups[str(key)] = str(group)
    except Exception:
        groups = {}
    _SCHEMA_GROUPS_CACHE = groups
    return groups


def classify(key: str, *, schema_groups: Mapping[str, str] | None = None) -> str:
    """Return the axis a parameter key belongs to.

    Precedence: explicit canonical map -> specific pattern rules -> schema
    ``group`` default -> broad pattern families -> :data:`UNCLASSIFIED`. Internal
    keys (leading underscore) are classified by their stem so derived mirrors
    share their public key's axis. ``schema_groups`` defaults to
    :func:`param_schema_groups`; pass ``{}`` to force pattern-only behavior.
    """
    name = str(key)
    probe = name[1:] if name.startswith("_") else name
    if probe in EXPLICIT_AXIS:
        return EXPLICIT_AXIS[probe]
    for pattern, axis in _SPECIFIC_PATTERN_RULES:
        if pattern in probe:
            return axis
    groups = param_schema_groups() if schema_groups is None else schema_groups
    group = groups.get(probe) or groups.get(name)
    if group and group in GROUP_AXIS_DEFAULT:
        return GROUP_AXIS_DEFAULT[group]
    for pattern, axis in _BROAD_PATTERN_RULES:
        if pattern in probe:
            return axis
    return UNCLASSIFIED


def decompose(params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Split a flat params dict into ``{axis: {key: value}}`` buckets."""
    out: dict[str, dict[str, Any]] = {axis: {} for axis in (*ALL_AXES, UNCLASSIFIED)}
    for key, value in params.items():
        out[classify(key)][key] = value
    return out


def axis_keys(params: Mapping[str, Any], axis: str) -> dict[str, Any]:
    """Return only the params on a given axis."""
    if axis not in (*ALL_AXES, UNCLASSIFIED):
        raise ValueError(f"unknown axis {axis!r}; valid axes: {(*ALL_AXES, UNCLASSIFIED)}.")
    return {k: v for k, v in params.items() if classify(k) == axis}


def shared_scene_keys(overlay: Mapping[str, Any]) -> list[str]:
    """SAMPLE-axis keys present in an overlay (the shared-scene authorities)."""
    return sorted(k for k in overlay if classify(k) == SAMPLE)


def assert_microscope_overlay(overlay: Mapping[str, Any], *, field_name: str = "overlay") -> None:
    """Reject a per-microscope overlay that sets shared SAMPLE-axis keys.

    Generalizes the report's hand-listed shared-scene guard: any SAMPLE-axis key
    in a microscope-local overlay means two microscopes would rank *different*
    scenes while the report still presents one recommendation.
    """
    blocked = shared_scene_keys(overlay)
    if blocked:
        raise ValueError(
            f"{field_name} sets shared SAMPLE-axis key(s) {blocked!r}. "
            "Sample/specimen/environment/medium parameters are shared scene and "
            "must be supplied once; a microscope overlay may only vary the "
            f"{sorted(MICROSCOPE_LOCAL_AXES)} axes."
        )


def derive_shared_scene_param_keys(param_keys: Sequence[str]) -> frozenset[str]:
    """Derive the shared-scene (SAMPLE-axis) key set from the taxonomy.

    Pass the full ``parameters`` key list and the shared-scene set is computed, so
    adding a new sample/environment parameter cannot silently become
    microscope-local because a report-local static list was not updated.
    """
    return frozenset(k for k in param_keys if classify(str(k)) == SAMPLE)


def vary(base: Mapping[str, Any], axis: str, changes: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``changes`` applied, asserting all changes are on ``axis``.

    Enables clean single-axis experiments, e.g. ``vary(p, OPERATING_POINT,
    {"tem_dose_per_pixel": 2.0 * p["tem_dose_per_pixel"]})`` ("same instrument,
    twice the dose"). Misattributed keys raise rather than silently crossing
    axes.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"cannot vary unknown axis {axis!r}.")
    misattributed = {k: classify(k) for k in changes if classify(k) != axis}
    if misattributed:
        raise ValueError(
            f"vary({axis!r}) received keys on other axes: {misattributed}. "
            "Cross-axis edits must be made explicitly, one axis at a time."
        )
    updated = dict(base)
    updated.update(changes)
    return updated


def coverage(params: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostic: per-axis key counts plus any UNCLASSIFIED keys to triage."""
    buckets = decompose(params)
    return {
        "counts": {axis: len(buckets[axis]) for axis in (*ALL_AXES, UNCLASSIFIED)},
        "unclassified_keys": sorted(buckets[UNCLASSIFIED]),
    }


__all__ = [
    "SAMPLE", "INSTRUMENT", "OPERATING_POINT", "DETECTOR", "SAMPLING", "ANALYSIS",
    "UNCLASSIFIED", "ALL_AXES", "SHARED_AXES", "MICROSCOPE_LOCAL_AXES",
    "EXPLICIT_AXIS", "GROUP_AXIS_DEFAULT", "param_schema_groups",
    "classify", "decompose", "axis_keys", "shared_scene_keys",
    "assert_microscope_overlay", "derive_shared_scene_param_keys",
    "vary", "coverage",
]
