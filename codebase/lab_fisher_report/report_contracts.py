"""Shared ownership contracts for the lab Fisher report workflow."""

from __future__ import annotations

from typing import Any, Mapping

# Paper-facing configured-profile shared-grid defaults.  These are the
# fixed-instrument Contract-LP template values described in the manuscript; user
# overlays may intentionally define other experiments, but the default lab
# report path must not drift away from this contract silently.
REPORT_CONFIGURED_PROFILE_DEFAULTS = {
    "image_size_pixels": 192,
    "pixel_size_nm": 65.0,
    "pupil_samples": 384,
    "psf_oversampling_factor": 2,
}

REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS = {
    "particle_count": 1,
    "component_count": 1,
    "shape": "sphere",
    "diameter_nm": 100.0,
    "hydrodynamic_diameter_nm": 100.0,
    "material": "gold",
}


def _matches_report_contract_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 0.0
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return actual == expected


def assert_report_configured_profile_defaults(
    params: Mapping[str, Any],
    *,
    context: str = "lab Fisher configured-profile defaults",
) -> None:
    """Assert that the shared configured-profile template matches the paper."""

    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, expected in REPORT_CONFIGURED_PROFILE_DEFAULTS.items():
        actual = params.get(key)
        if not _matches_report_contract_value(actual, expected):
            mismatches[key] = (actual, expected)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} expected {expected!r}"
            for key, (actual, expected) in sorted(mismatches.items())
        )
        raise AssertionError(f"{context} do not match the paper contract: {details}")


def assert_report_configured_profile_particle_defaults(
    params: Mapping[str, Any],
    *,
    context: str = "lab Fisher configured-profile particle defaults",
) -> None:
    """Assert that the default shared target particle matches the paper."""

    expected = REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS
    particles = params.get("particles")
    mismatches: dict[str, tuple[Any, Any]] = {}
    if not isinstance(particles, list):
        raise AssertionError(
            f"{context} do not match the paper contract: "
            f"particles={particles!r} expected a one-particle list"
        )
    if len(particles) != int(expected["particle_count"]):
        mismatches["particle_count"] = (len(particles), expected["particle_count"])
    first = particles[0] if particles else {}
    if not isinstance(first, Mapping):
        raise AssertionError(
            f"{context} do not match the paper contract: "
            f"particles[0]={first!r} expected a particle mapping"
        )
    motion = first.get("motion", {})
    if not isinstance(motion, Mapping):
        motion = {}
    components = first.get("components")
    if not isinstance(components, list):
        raise AssertionError(
            f"{context} do not match the paper contract: "
            f"particles[0].components={components!r} expected a one-component list"
        )
    if len(components) != int(expected["component_count"]):
        mismatches["component_count"] = (len(components), expected["component_count"])
    component = components[0] if components else {}
    if not isinstance(component, Mapping):
        raise AssertionError(
            f"{context} do not match the paper contract: "
            f"particles[0].components[0]={component!r} expected a component mapping"
        )

    checks = {
        "shape": component.get("shape"),
        "diameter_nm": component.get("diameter_nm"),
        "hydrodynamic_diameter_nm": motion.get("hydrodynamic_diameter_nm"),
        "material": str(component.get("material", "")).strip().lower(),
    }
    for key, actual in checks.items():
        expected_value = expected[key]
        if not _matches_report_contract_value(actual, expected_value):
            mismatches[key] = (actual, expected_value)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} expected {expected_value!r}"
            for key, (actual, expected_value) in sorted(mismatches.items())
        )
        raise AssertionError(f"{context} do not match the paper contract: {details}")


# Keys that must be carried from the run/shared layer over each candidate's
# instrument preset before microscope-local params are applied.  The separate
# forbidden-local set below is intentionally narrower: the microscope-axis
# taxonomy owns sample/scene rejection, and explicit microscope
# comparisons may vary frame count/timing as an acquisition strategy, but the
# dynamic-estimator toggles, output flags, random seed, and scene geometry are
# owned by the report run and cannot be microscope-local without corrupting the
# meaning of microscope_ranking.csv, sequence_fisher_summary.csv, manifest.json,
# and report.md.
REPORT_SHARED_RUN_PARAM_KEYS = frozenset(
    {
        "return_ideal_float_frames",
        "save_frame_sequence",
        "save_raw_camera_video",
        "save_raw_camera_frame_sequence",
        "save_raw_frame_views",
        "mask_generation_enabled",
        "num_frames",
        "duration_seconds",
        "dynamic_bayesian_enabled",
        "dynamic_process_noise_scale",
        "dynamic_initial_variance_nm2",
        "dynamic_include_smoothing",
        "random_seed",
        "particles",
        "initial_z_span_nm",
    }
)

# These keys have global/report-run consumers.  Do not add
# num_frames/fps/duration_seconds here: per-microscope frame/timing variants are
# an explicit acquisition-configuration axis and are surfaced through resolved
# params plus sequence_frames_by_microscope.  Sample/scene authority is rejected
# separately through microscope_axes.assert_microscope_overlay.
MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS = frozenset(
    {
        "return_ideal_float_frames",
        "save_frame_sequence",
        "save_raw_camera_video",
        "save_raw_camera_frame_sequence",
        "save_raw_frame_views",
        "mask_generation_enabled",
        "dynamic_bayesian_enabled",
        "dynamic_process_noise_scale",
        "dynamic_initial_variance_nm2",
        "dynamic_include_smoothing",
        "random_seed",
    }
)

__all__ = [
    "REPORT_CONFIGURED_PROFILE_DEFAULTS",
    "REPORT_CONFIGURED_PROFILE_PARTICLE_DEFAULTS",
    "assert_report_configured_profile_defaults",
    "assert_report_configured_profile_particle_defaults",
    "MICROSCOPE_LOCAL_FORBIDDEN_REPORT_PARAM_KEYS",
    "REPORT_SHARED_RUN_PARAM_KEYS",
]
