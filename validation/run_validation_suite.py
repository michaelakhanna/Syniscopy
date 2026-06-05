#!/usr/bin/env python3
"""Run Syniscopy scientific validation checks with machine-readable output.

The default profile is intentionally fast enough to run before notebook
generation. It validates the core equations and runtime contracts that the
paper/notebooks rely on, while marking heavier reference checks as skipped or
external-artifact-required instead of silently treating them as passed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


@dataclass
class ValidationResult:
    name: str
    status: str
    level: str
    surface: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "skip"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": str(value)}
    return value


def _result(
    name: str,
    status: str,
    *,
    level: str,
    surface: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
    started: float | None = None,
) -> ValidationResult:
    return ValidationResult(
        name=name,
        status=status,
        level=level,
        surface=surface,
        detail=detail,
        metrics=_json_safe(metrics or {}),
        duration_s=0.0 if started is None else time.perf_counter() - started,
    )


def _run_check(
    name: str,
    level: str,
    surface: str,
    check: Callable[[], tuple[bool, str, dict[str, Any]]],
) -> ValidationResult:
    started = time.perf_counter()
    try:
        ok, detail, metrics = check()
    except Exception as exc:  # validation failures should be explicit rows
        return _result(
            name,
            "fail",
            level=level,
            surface=surface,
            detail=f"{type(exc).__name__}: {exc}",
            metrics={},
            started=started,
        )
    return _result(
        name,
        "pass" if ok else "fail",
        level=level,
        surface=surface,
        detail=detail,
        metrics=metrics,
        started=started,
    )


def _missing_optional(name: str, package: str, *, level: str, surface: str, require: bool) -> ValidationResult:
    status = "fail" if require else "skip"
    return _result(
        name,
        status,
        level=level,
        surface=surface,
        detail=f"optional dependency {package!r} is not installed",
    )


def check_compile_core() -> tuple[bool, str, dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "codebase", "validation", "scripts"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode == 0, proc.stdout.strip() or "compileall passed", {"returncode": proc.returncode}


def check_imports_and_defaults() -> tuple[bool, str, dict[str, Any]]:
    modules = [
        "config",
        "simulation",
        "rendering",
        "fisher",
        "imaging_models",
        "trajectory",
        "camera_noise",
        "metadata",
        "counterfactual_packets",
    ]
    loaded = []
    for module in modules:
        importlib.import_module(module)
        loaded.append(module)
    from config import PARAMS, validate_params

    params = dict(PARAMS)
    validate_params(params)
    expected = {
        "tem_model": "multislice_physical",
        "tem_backend": "multislice_physical",
        "sem_model": "physical_electron_transport",
        "sem_backend": "monte_carlo_physical",
        "sem_physical_elastic_model": "screened_rutherford",
    }
    mismatches = {
        key: {"observed": params.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if params.get(key) != expected_value
    }
    return (
        not mismatches,
        "core imports and real-physics defaults validated" if not mismatches else "default mismatch",
        {"modules": loaded, "defaults": {key: params.get(key) for key in expected}, "mismatches": mismatches},
    )


def check_parameter_contracts() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS, validate_params
    from param_schema import PARAM_SCHEMA

    exponent_keys = ("sem_transport_source_exponent", "sem_transport_topography_exponent")
    schema_mins = {key: PARAM_SCHEMA[key]["min"] for key in exponent_keys}
    rejected: dict[str, bool] = {}
    for key in exponent_keys:
        params = dict(PARAMS)
        params[key] = 0.049
        try:
            validate_params(params)
        except ValueError:
            rejected[key] = True
        else:
            rejected[key] = False
    ok = all(value == 0.05 for value in schema_mins.values()) and all(rejected.values())
    return (
        ok,
        "SEM transport exponent schema minima match config validation",
        {"schema_mins": schema_mins, "below_min_rejected": rejected},
    )


def check_sem_response_contract_metadata() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS, validate_params
    from imaging_models.sem import ScanningElectronMicroscopyImagingModel

    cases = {
        "gaussian": {
            "sem_model": "gaussian_probe_secondary_yield",
            "sem_backend": "gaussian_probe_proxy",
            "sem_source_representation": "projected",
            "expected": {
                "sem_backend_consumes_topography_gain": False,
                "sem_backend_consumes_detector_direction": False,
                "sem_backend_topography_convention": "not_supported",
                "sem_sample_environment_z_policy": "projected_surface_source",
                "sem_sample_environment_uses_particle_world_z": False,
            },
        },
        "interaction_volume": {
            "sem_model": "interaction_volume_proxy",
            "sem_backend": "interaction_volume_proxy",
            "sem_source_representation": "projected",
            "sem_topography_contrast_gain": 1.0,
            "expected": {
                "sem_backend_consumes_topography_gain": True,
                "sem_backend_consumes_detector_direction": True,
                "sem_backend_edge_convention": "positive_directed_detector_gradient",
                "sem_backend_topography_convention": "gradient_magnitude",
            },
        },
        "physical_volume": {
            "sem_model": "physical_electron_transport",
            "sem_backend": "monte_carlo_physical",
            "sem_source_representation": "volume",
            "expected": {
                "sem_backend_consumes_topography_gain": True,
                "sem_backend_consumes_detector_direction": True,
                "sem_backend_edge_convention": "gradient_magnitude",
                "sem_backend_topography_convention": "absolute_directed_detector_gradient",
                "sem_sample_environment_z_policy": "surface_source_first_slice",
                "sem_sample_environment_uses_particle_world_z": False,
            },
        },
    }
    observed: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    for name, spec in cases.items():
        params = dict(PARAMS)
        expected = dict(spec["expected"])
        params.update({key: value for key, value in spec.items() if key != "expected"})
        validate_params(params)
        response = ScanningElectronMicroscopyImagingModel(params).compute_response_function((32, 32), params)
        observed[name] = {key: response.get(key) for key in expected}
        missing = {
            key: {"observed": observed[name][key], "expected": value}
            for key, value in expected.items()
            if observed[name][key] != value
        }
        if missing:
            mismatches[name] = missing
    return (
        not mismatches,
        "SEM response metadata reports backend-active terms and sample-environment z policy",
        {"observed": observed, "mismatches": mismatches},
    )


def check_material_contracts() -> tuple[bool, str, dict[str, Any]]:
    from material_optical_catalog import MATERIAL_ELECTRON_DEFAULTS, sem_transport_material

    sem_materials = ("carbon", "aluminum", "silicon", "copper", "silver", "gold", "water", "polystyrene")
    missing: dict[str, list[str]] = {}
    for material in sem_materials:
        values = MATERIAL_ELECTRON_DEFAULTS.get(material, {})
        absent = [
            field
            for field in ("mean_inner_potential_V", "density_g_cm3", "atomic_number", "atomic_weight_g_mol")
            if values.get(field) is None
        ]
        if absent:
            missing[material] = absent
        if material in {"carbon", "aluminum", "silicon", "copper", "silver", "gold"}:
            sem_transport_material(material)
    return not missing, "material electron/SEM constants present", {"missing": missing}


def check_material_single_source() -> tuple[bool, str, dict[str, Any]]:
    from material_optical_catalog import MATERIAL_REFRACTIVE_INDEX, lookup_refractive_index

    legacy_path = CODEBASE / "materials.py"
    stale_imports: list[dict[str, Any]] = []
    for path in sorted(CODEBASE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(CODEBASE)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("from materials import") or stripped == "import materials" or stripped.startswith("import materials,"):
                stale_imports.append({"file": str(rel), "line": lineno, "text": stripped})

    lookup_failures: list[dict[str, Any]] = []
    wavelengths_nm = (405.0, 488.0, 532.0, 635.0, 785.0)
    for material in sorted(MATERIAL_REFRACTIVE_INDEX):
        for wavelength_nm in wavelengths_nm:
            try:
                value = lookup_refractive_index(material, wavelength_nm=wavelength_nm)
            except Exception as exc:
                lookup_failures.append(
                    {
                        "material": material,
                        "wavelength_nm": wavelength_nm,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not (np.isfinite(value.real) and np.isfinite(value.imag)):
                lookup_failures.append(
                    {
                        "material": material,
                        "wavelength_nm": wavelength_nm,
                        "error": f"non-finite refractive index {value!r}",
                    }
                )

    ok = not legacy_path.exists() and not stale_imports and not lookup_failures
    return ok, "material optical/electron data resolved through material_optical_catalog only", {
        "legacy_materials_py_exists": legacy_path.exists(),
        "stale_top_level_materials_imports": stale_imports,
        "catalog_material_count": len(MATERIAL_REFRACTIVE_INDEX),
        "catalog_lookup_count": len(MATERIAL_REFRACTIVE_INDEX) * len(wavelengths_nm),
        "lookup_failures": lookup_failures,
    }


def check_electron_constants() -> tuple[bool, str, dict[str, Any]]:
    from imaging_models.electron_constants import electron_wavelength_m, scherzer_defocus_m

    published_pm = {100.0: 3.7014, 200.0: 2.5079, 300.0: 1.9687}
    wavelength_errors = {}
    for kv, expected_pm in published_pm.items():
        observed_pm = electron_wavelength_m(kv) * 1.0e12
        wavelength_errors[str(int(kv))] = abs(observed_pm / expected_pm - 1.0)
    scherzer_errors = {}
    for kv, cs_mm in ((200.0, 1.0), (300.0, 2.0)):
        observed = scherzer_defocus_m(kv, cs_mm)
        expected = math.sqrt(1.5 * electron_wavelength_m(kv) * cs_mm * 1.0e-3)
        scherzer_errors[f"{int(kv)}kV_{cs_mm:g}mm"] = abs(observed / expected - 1.0)
    ok = max(wavelength_errors.values()) <= 1.0e-3 and max(scherzer_errors.values()) <= 1.0e-12
    return ok, "electron wavelength and Scherzer formulas checked", {
        "wavelength_relative_errors": wavelength_errors,
        "scherzer_relative_errors": scherzer_errors,
    }


def check_thinfilm_fresnel() -> tuple[bool, str, dict[str, Any]]:
    from substrate import material_from_name
    from thinfilm import normal_incidence_thinfilm_reflection

    wavelength_nm = 550.0
    n0 = complex(material_from_name("water").n_complex(wavelength_nm))
    ns = complex(material_from_name("glass").n_complex(wavelength_nm))
    observed = normal_incidence_thinfilm_reflection("water", "glass", None, wavelength_nm)
    expected = (n0 - ns) / (n0 + ns)
    error = abs(observed - expected)
    return error <= 1.0e-12, "no-layer thin-film reflection equals Fresnel", {"abs_error": error}


def check_tmm_external(require: bool) -> ValidationResult:
    try:
        import tmm  # type: ignore
    except Exception:
        return _missing_optional(
            "thinfilm_tmm_external",
            "tmm",
            level="A",
            surface="thin-film reflection",
            require=require,
        )

    started = time.perf_counter()
    try:
        from substrate import material_from_name
        from thinfilm import normal_incidence_thinfilm_reflection

        wavelength_nm = 550.0
        n0 = complex(material_from_name("water").n_complex(wavelength_nm))
        n1 = complex(material_from_name("silica").n_complex(wavelength_nm))
        ns = complex(material_from_name("glass").n_complex(wavelength_nm))
        thickness_nm = 120.0
        observed = normal_incidence_thinfilm_reflection(
            "water",
            "glass",
            [{"n_complex": {"real": n1.real, "imag": n1.imag}, "thickness_nm": thickness_nm}],
            wavelength_nm,
        )
        expected = tmm.coh_tmm("s", [n0, n1, ns], [np.inf, thickness_nm, np.inf], 0.0, wavelength_nm)["r"]
        complex_error = abs(observed - expected)
        conjugate_error = abs(observed - np.conj(expected))
        reflectance_error = abs(abs(observed) ** 2 - abs(expected) ** 2)
        return _result(
            "thinfilm_tmm_external",
            "pass" if reflectance_error <= 1.0e-9 else "fail",
            level="A",
            surface="thin-film reflection",
            detail="single-layer reflectance compared with tmm; complex amplitude convention recorded separately",
            metrics={
                "complex_amplitude_abs_error": complex_error,
                "complex_conjugate_abs_error": conjugate_error,
                "reflectance_abs_error": reflectance_error,
            },
            started=started,
        )
    except Exception as exc:
        return _result(
            "thinfilm_tmm_external",
            "fail",
            level="A",
            surface="thin-film reflection",
            detail=f"{type(exc).__name__}: {exc}",
            started=started,
        )


def check_mie_external(require: bool) -> ValidationResult:
    try:
        import miepython  # type: ignore
    except Exception:
        return _missing_optional(
            "miepython_external",
            "miepython",
            level="A",
            surface="Mie angular amplitude",
            require=require,
        )

    started = time.perf_counter()
    try:
        from mie_scattering import mie_S1_S2_from_coefficients, mie_an_bn

        m = complex(1.59, 0.0)
        x = 3.0
        mus = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        a_n, b_n = mie_an_bn(m, x)
        _, s2_syn = mie_S1_S2_from_coefficients(a_n, b_n, mus)
        fn = None
        for name in ("S1_S2", "mie_S1_S2", "mie_s1_s2"):
            if hasattr(miepython, name):
                fn = getattr(miepython, name)
                break
        if fn is None:
            raise RuntimeError("miepython has no S1/S2 API")
        try:
            _, s2_ref = fn(m, x, mus, norm="wiscombe")
        except TypeError:
            _, s2_ref = fn(m, x, mus)
        ratios = np.abs(np.asarray(s2_syn)) / np.maximum(np.abs(np.asarray(s2_ref)), 1.0e-300)
        rel_std = float(np.std(ratios) / max(float(np.mean(ratios)), 1.0e-300))
        return _result(
            "miepython_external",
            "pass" if rel_std <= 1.0e-6 else "fail",
            level="A",
            surface="Mie angular amplitude",
            detail="S2 angular dependence agrees with miepython up to normalization",
            metrics={"ratio_mean": float(np.mean(ratios)), "ratio_relative_std": rel_std},
            started=started,
        )
    except Exception as exc:
        return _result(
            "miepython_external",
            "fail",
            level="A",
            surface="Mie angular amplitude",
            detail=f"{type(exc).__name__}: {exc}",
            started=started,
        )


def check_noise() -> tuple[bool, str, dict[str, Any]]:
    import camera_noise as cn
    from config import PARAMS

    analytic_errors = {}
    for gain in (1.0, 2.0, 5.0):
        params = dict(PARAMS)
        params.update({"shot_noise_enabled": True, "camera_gain_e_per_count": gain})
        signal = np.array([100.0, 400.0, 2500.0, 10000.0])
        observed = np.asarray(cn.shot_noise_std_counts(signal, params), dtype=float)
        expected = np.sqrt(signal * gain) / gain
        analytic_errors[str(gain)] = float(np.max(np.abs(observed / expected - 1.0)))

    slopes = {}
    for gain in (1.0, 4.0):
        params = dict(PARAMS)
        params.update(
            {
                "shot_noise_enabled": True,
                "gaussian_noise_enabled": False,
                "camera_gain_e_per_count": gain,
                "emccd_enabled": False,
                "read_noise_counts": 0.0,
                "read_noise_e": None,
                "dark_current_e_per_pixel_per_s": 0.0,
                "fixed_pattern_gain_std": 0.0,
                "fixed_pattern_offset_counts": 0.0,
                "hot_pixel_fraction": 0.0,
                "scan_line_noise_counts": 0.0,
                "fixed_pattern_gain_map": None,
                "scmos_gain_map": None,
                "scmos_read_noise_map": None,
                "random_seed": 7,
            }
        )
        means = []
        variances = []
        for level in (100.0, 400.0, 1600.0, 6400.0):
            image = np.full((240, 240), level, dtype=float)
            out = np.asarray(cn.apply_camera_noise_counts(image, params), dtype=float)
            means.append(float(out.mean()))
            variances.append(float(out.var()))
        slope = float(np.polyfit(np.asarray(means), np.asarray(variances), 1)[0])
        slopes[str(gain)] = slope
    slope_errors = {gain: abs(slope - 1.0 / float(gain)) for gain, slope in slopes.items()}
    ok = max(analytic_errors.values()) <= 1.0e-12 and max(slope_errors.values()) <= 0.02
    return ok, "shot-noise analytic and variance/mean checks", {
        "analytic_relative_errors": analytic_errors,
        "variance_mean_slopes": slopes,
        "slope_abs_errors": slope_errors,
    }


def _gaussian_signal(amplitude: float, sigma_px: float, size: int) -> np.ndarray:
    center = (size - 1.0) / 2.0
    y, x = np.indices((size, size), dtype=float)
    r2 = (x - center) ** 2 + (y - center) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma_px * sigma_px))


def check_lateral_fisher() -> tuple[bool, str, dict[str, Any]]:
    from fisher.lateral import compute_localization_crlb

    amplitude = 1.0
    variance = 1.0
    pixel_nm = 100.0
    expected_sigma = pixel_nm * math.sqrt(variance) / amplitude * math.sqrt(2.0 / math.pi)
    errors = {}
    for sigma_px in (3.0, 4.0, 6.0, 8.0):
        contrast = _gaussian_signal(amplitude, sigma_px, 121)
        result = compute_localization_crlb(contrast, variance, pixel_nm)
        observed = float(result["sigma_x_nm"])
        errors[str(sigma_px)] = abs(observed / expected_sigma - 1.0)
    scaling_errors = {}
    for amp, var in ((2.0, 1.0), (1.0, 4.0), (0.5, 1.0), (1.0, 0.25)):
        contrast = _gaussian_signal(amp, 5.0, 121)
        result = compute_localization_crlb(contrast, var, pixel_nm)
        observed = float(result["sigma_x_nm"])
        expected = pixel_nm * math.sqrt(var) / amp * math.sqrt(2.0 / math.pi)
        scaling_errors[f"A={amp},var={var}"] = abs(observed / expected - 1.0)
    ok = max(errors.values()) <= 0.035 and max(scaling_errors.values()) <= 0.015
    return ok, "lateral Fisher/CRLB matches Gaussian closed-form tolerance", {
        "absolute_relative_errors": errors,
        "scaling_relative_errors": scaling_errors,
    }


def check_trajectory() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS
    from trajectory import simulate_trajectories, stokes_einstein_diffusion_coefficient

    diameter_nm, temperature_k, viscosity, fps = 100.0, 298.0, 1.0e-3, 1000.0
    particle_count, frames, seed = 32, 2500, 12345
    dt = 1.0 / fps
    params = dict(PARAMS)
    params.update(
        {
            "random_seed": seed,
            "fps": fps,
            "num_frames": frames,
            "temperature_K": temperature_k,
            "viscosity_Pa_s": viscosity,
            "z_motion_constraint_model": "unconstrained",
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "sample_environment_pattern": "none",
            "sample_environment_pattern_preset": "empty_background",
            "image_size_pixels": 256,
            "pixel_size_nm": 100.0,
            "initial_z_span_nm": 4000.0,
        }
    )
    one = {
        "name": "p",
        "motion": {"hydrodynamic_diameter_nm": diameter_nm, "initial_position_nm": None},
        "signal_multiplier": 1.0,
        "source_multiplier": 1.0,
        "components": [
            {
                "shape": "sphere",
                "offset_nm": [0.0, 0.0, 0.0],
                "diameter_nm": diameter_nm,
                "material": "polystyrene",
                "refractive_index": None,
                "signal_multiplier": 1.0,
                "source_multiplier": 1.0,
                "material_properties": None,
            }
        ],
    }
    import copy

    params["particles"] = [copy.deepcopy(one) for _ in range(particle_count)]
    diffusion = stokes_einstein_diffusion_coefficient(diameter_nm, temperature_k, viscosity)
    expected_step_nm = math.sqrt(2.0 * diffusion * dt) * 1.0e9
    trajectory = simulate_trajectories(params)
    reproducible = np.array_equal(trajectory, simulate_trajectories(params))
    steps = np.diff(trajectory, axis=1)
    std_nm = steps.reshape(-1, 3).std(axis=0)
    step_rel_error = np.abs(std_nm / expected_step_nm - 1.0)
    lags = np.unique(np.round(np.geomspace(1, frames - 1, 20)).astype(int))
    msd = []
    for lag in lags:
        disp = trajectory[:, lag:, :] - trajectory[:, :-lag, :]
        msd.append((disp * disp).sum(axis=2).mean())
    slope = float(np.polyfit(lags * dt, np.asarray(msd), 1)[0])
    diffusion_msd = (slope * 1.0e-18) / 6.0
    msd_rel_error = abs(diffusion_msd / diffusion - 1.0)
    mean_nm = steps.reshape(-1, 3).mean(axis=0)
    mean_rel_error = np.abs(mean_nm / expected_step_nm)
    ok = (
        reproducible
        and float(np.max(step_rel_error)) <= 0.05
        and float(np.max(mean_rel_error)) <= 0.03
        and msd_rel_error <= 0.25
    )
    return ok, "Brownian one-step Gaussian statistics, MSD diagnostic, and seed determinism", {
        "expected_step_nm": expected_step_nm,
        "observed_step_mean_nm": mean_nm.tolist(),
        "observed_step_std_nm": std_nm.tolist(),
        "mean_relative_errors": mean_rel_error.tolist(),
        "step_relative_errors": step_rel_error.tolist(),
        "diffusion_m2_s": diffusion,
        "diffusion_from_msd_m2_s": diffusion_msd,
        "msd_relative_error": msd_rel_error,
        "tolerances": {
            "max_step_std_relative_error": 0.05,
            "max_step_mean_relative_error": 0.03,
            "max_msd_relative_error": 0.25,
        },
        "seed_reproducible": reproducible,
    }


def check_rotational_diffusion() -> tuple[bool, str, dict[str, Any]]:
    from config import BOLTZMANN_CONSTANT, PARAMS
    from trajectory import resolve_rotational_step_std_rad, simulate_orientations

    diameter_nm = 120.0
    temperature_k = 298.0
    viscosity = 1.0e-3
    fps = 200.0
    particle_count = 4
    frames = 12
    params = dict(PARAMS)
    params.update(
        {
            "random_seed": 2468,
            "fps": fps,
            "num_frames": frames,
            "temperature_K": temperature_k,
            "viscosity_Pa_s": viscosity,
            "rotational_diffusion_enabled": True,
            "rotational_diffusion_mode": "stokes_einstein",
            "particles": [
                {
                    "name": f"p{i}",
                    "motion": {"hydrodynamic_diameter_nm": diameter_nm, "initial_position_nm": None},
                    "signal_multiplier": 1.0,
                    "source_multiplier": 1.0,
                    "components": [
                        {
                            "shape": "sphere",
                            "offset_nm": [0.0, 0.0, 0.0],
                            "diameter_nm": diameter_nm,
                            "material": "polystyrene",
                            "refractive_index": None,
                            "signal_multiplier": 1.0,
                            "source_multiplier": 1.0,
                            "material_properties": None,
                        }
                    ],
                }
                for i in range(particle_count)
            ],
        }
    )
    radius_m = 0.5 * diameter_nm * 1.0e-9
    d_rot = BOLTZMANN_CONSTANT * temperature_k / (8.0 * math.pi * viscosity * radius_m ** 3)
    expected_step = math.sqrt(2.0 * d_rot / fps)
    observed_steps = resolve_rotational_step_std_rad(params, particle_count)
    step_errors = np.abs(observed_steps / expected_step - 1.0)

    first = simulate_orientations(params, particle_count, frames)
    second = simulate_orientations(params, particle_count, frames)
    reproducible = bool(np.array_equal(first, second))
    orthogonality_errors = []
    determinant_errors = []
    for matrix in np.asarray(first).reshape(-1, 3, 3):
        orthogonality_errors.append(float(np.max(np.abs(matrix.T @ matrix - np.eye(3)))))
        determinant_errors.append(abs(float(np.linalg.det(matrix)) - 1.0))
    max_orthogonality_error = max(orthogonality_errors)
    max_determinant_error = max(determinant_errors)
    ok = (
        float(np.max(step_errors)) <= 1.0e-12
        and reproducible
        and max_orthogonality_error <= 1.0e-12
        and max_determinant_error <= 1.0e-12
    )
    return ok, "rotational Stokes-Einstein-Debye step scale and SO(3) determinism", {
        "expected_step_std_rad": expected_step,
        "observed_step_std_rad": observed_steps.tolist(),
        "step_relative_errors": step_errors.tolist(),
        "seed_reproducible": reproducible,
        "max_orthogonality_error": max_orthogonality_error,
        "max_determinant_error": max_determinant_error,
    }


def check_manifest_packet_roundtrip() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS
    from counterfactual_packets import (
        load_counterfactual_modality_packet,
        save_counterfactual_modality_packet,
        validate_counterfactual_modality_packet,
    )
    from metadata import build_dataset_index_entry, build_video_manifest, save_dataset_manifest

    metrics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="syniscopy_validation_") as tmp:
        tmp_path = Path(tmp)
        params = dict(PARAMS)
        params.update(
            {
                "output_filename": str(tmp_path / "video_0000.avi"),
                "mask_output_directory": str(tmp_path / "masks" / "video_0000"),
                "fps": 10.0,
                "duration_seconds": 0.1,
                "num_frames": 1,
                "image_size_pixels": 8,
                "pixel_size_nm": 100.0,
            }
        )
        manifest = build_video_manifest(
            params,
            str(tmp_path),
            video_index=0,
            dataset_preset="validation",
            instrument_preset=None,
            video_seed=123,
            result_metadata={"validation": True, "frames_shape": [1, 1, 8, 8]},
        )
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
        entry = build_dataset_index_entry(manifest)
        dataset_path = save_dataset_manifest(str(tmp_path), [entry], source_provenance=manifest["source_provenance"])
        dataset_manifest = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        manifest_ok = (
            bool(encoded)
            and dataset_manifest["num_videos"] == 1
            and dataset_manifest["videos"][0]["video_index"] == 0
            and dataset_manifest["videos"][0]["source_provenance_fingerprint"]
            == manifest["source_provenance"]["fingerprint"]
        )
        metrics["manifest_keys"] = sorted(manifest.keys())
        metrics["dataset_manifest_path_exists"] = Path(dataset_path).exists()

        images = {
            "coherent_bright_field": np.arange(16, dtype=float).reshape(4, 4),
            "differential_phase_contrast": np.arange(16, dtype=float).reshape(4, 4) + 1.0,
        }
        packet_path = save_counterfactual_modality_packet(
            str(tmp_path / "packet.npz"),
            latent_state={"frame_index": 0, "particle_count": 1},
            images_by_modality=images,
            metadata={
                "shared_coordinate_frame": {
                    "axis_order": "yx",
                    "pixel_size_nm": 100.0,
                    "origin": "validation_fixture",
                }
            },
            compressed=False,
        )
        loaded = load_counterfactual_modality_packet(packet_path)
        validate_counterfactual_modality_packet(loaded)
        packet_modalities = sorted(loaded["images_by_modality"])
        packet_ok = packet_modalities == sorted(images) and all(
            np.array_equal(loaded["images_by_modality"][name], images[name])
            for name in images
        )
        metrics["packet_modalities"] = packet_modalities
        metrics["packet_path_exists"] = Path(packet_path).exists()

    return manifest_ok and packet_ok, "manifest JSON and counterfactual packet round trip", metrics


def check_iscat() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS
    from imaging_models.interferometric import InterferometricImagingModel

    rng = np.random.default_rng(0)
    h = w = 48
    e_sca = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))) * 0.05
    background = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))) * 1.0
    errors = {}
    for phase, amp in ((0.0, 1.0), (np.pi / 3.0, 1.0), (0.0, 1.3), (np.pi / 2.0, 0.7)):
        params = dict(PARAMS)
        params.update(
            {
                "imaging_model": "interferometric",
                "reference_field_amplitude": 1.0,
                "iscat_reference_model": "renderer",
                "iscat_reference_phase_rad": phase,
                "iscat_reference_amplitude_scale": amp,
                "iscat_collection_model": "scalar",
            }
        )
        model = InterferometricImagingModel(params)
        contrast = model.compute_per_particle_contrast(e_sca, background, params)
        ref = background * (amp * np.exp(1j * phase))
        expected = 2.0 * np.real(np.conj(ref) * e_sca) + np.abs(e_sca) ** 2
        errors[f"phase={phase:.3f},amp={amp}"] = float(np.max(np.abs(contrast - expected)))
    return max(errors.values()) <= 1.0e-12, "iSCAT contrast identity", {"max_abs_errors": errors}


def check_flagship() -> tuple[bool, str, dict[str, Any]]:
    from validation.flagship_validation import run_validation

    ok = bool(run_validation(tolerance=1.0e-12, gradient_relative_tolerance=1.0e-12))
    return ok, "COBRI and DPC flagship checks", {}


def check_psf_airy() -> tuple[bool, str, dict[str, Any]]:
    from config import PARAMS
    from optics import compute_complex_psf_stack

    na, n_medium, wavelength_nm = 0.30, 1.0, 550.0
    pixel_nm, oversampling, pupil_samples = 100.0, 2, 256
    params = dict(PARAMS)
    params.update(
        {
            "numerical_aperture": na,
            "refractive_index_medium": n_medium,
            "wavelength_nm": wavelength_nm,
            "pupil_samples": pupil_samples,
            "psf_oversampling_factor": oversampling,
            "pixel_size_nm": pixel_nm,
            "apodization_factor": 0.0,
            "spherical_aberration_strength": 0.0,
            "random_aberration_strength": 0.0,
            "optical_field_backend": "scalar_paraxial",
            "coverslip_correction_enabled": False,
        }
    )
    interp = compute_complex_psf_stack(
        params,
        particle_diameter_nm=10.0,
        particle_refractive_index=complex(1.59, 0.0),
        z_values_nm=np.array([0.0]),
    )
    field = np.asarray(interp(0.0))
    intensity = np.abs(field) ** 2
    intensity /= float(intensity.max())
    canvas_pitch = pixel_nm / oversampling
    center = pupil_samples // 2
    yy, xx = np.indices(intensity.shape)
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    bins = np.arange(0, int(radius.max()))
    profile = np.array([intensity[(radius >= k) & (radius < k + 1)].mean() for k in bins])
    first_min = None
    for k in range(2, len(profile) - 1):
        if profile[k] < profile[k - 1] and profile[k] <= profile[k + 1] and profile[k] < 0.2:
            first_min = k
            break
    measured_nm = float("nan") if first_min is None else first_min * canvas_pitch
    expected_nm = 0.61 * wavelength_nm / na
    rel_error = abs(measured_nm / expected_nm - 1.0)
    ok = first_min is not None and rel_error <= 0.03
    return ok, "scalar low-NA PSF Airy first dark ring", {
        "measured_first_zero_nm": measured_nm,
        "expected_first_zero_nm": expected_nm,
        "relative_error": rel_error,
        "first_min_bin": first_min,
    }


def check_tem_internal() -> tuple[bool, str, dict[str, Any]]:
    from validation.tem_multislice_validation import run_validation

    ok = bool(
        run_validation(
            tolerance=1.0e-10,
            weak_relative_tolerance=2.0e-3,
            weak_absolute_tolerance=1.0e-12,
            require_abtem=False,
        )
    )
    return ok, "TEM multislice internal physics anchors", {}


def check_modality_equations() -> tuple[bool, str, dict[str, Any]]:
    from validation.modality_equation_validation import run_validation

    ok = bool(run_validation())
    return ok, "dark-field, QPI, RICM, holography, fluorescence, TIRF, Zernike, TEM CTF, and Kohler limits", {}


def check_vectorial_debye() -> tuple[bool, str, dict[str, Any]]:
    from validation.vectorial_debye_validation import run_validation

    ok = bool(run_validation())
    return ok, "Richards-Wolf rotation covariance, on-axis polarization, and low-NA Airy limit", {}


def check_sem_smoke() -> tuple[bool, str, dict[str, Any]]:
    from imaging_models.sem_backends.physical_transport import simulate_sem_transport_observables
    from material_optical_catalog import sem_transport_material

    metrics: dict[str, Any] = {}
    for name in ("carbon", "gold"):
        observables, _, _, _, _ = simulate_sem_transport_observables(
            material=sem_transport_material(name),
            acceleration_keV=20.0,
            histories=600,
            seed=11,
            max_steps=384,
            cutoff_keV=0.05,
            elastic_model="screened_rutherford",
        )
        metrics[name] = observables.to_dict()
    carbon_eta = metrics["carbon"]["backscatter_coefficient"]
    gold_eta = metrics["gold"]["backscatter_coefficient"]
    ok = (
        0.0 <= carbon_eta <= 1.0
        and 0.0 <= gold_eta <= 1.0
        and gold_eta > carbon_eta
        and metrics["carbon"]["active_histories_at_limit"] == 0
    )
    return ok, "SEM physical transport smoke: finite BSE and high-Z trend", metrics


def check_tem_abtem_external(require: bool) -> ValidationResult:
    from validation.tem_multislice_validation import (
        ABTEM_REFERENCE_PATH,
        compare_abtem_reference_fixture,
    )

    started = time.perf_counter()
    if not ABTEM_REFERENCE_PATH.exists():
        return _result(
            "tem_abtem_external_fixture",
            "fail" if require else "skip",
            level="A/B",
            surface="TEM physical multislice",
            detail=(
                f"missing generated abTEM fixture at {ABTEM_REFERENCE_PATH}; "
                "run `python validation/run_abtem_reference.py` from the repository root"
            ),
            metrics={"fixture_path": str(ABTEM_REFERENCE_PATH)},
            started=started,
        )
    try:
        ok, detail, metrics = compare_abtem_reference_fixture(ABTEM_REFERENCE_PATH)
    except Exception as exc:
        return _result(
            "tem_abtem_external_fixture",
            "fail",
            level="A/B",
            surface="TEM physical multislice",
            detail=f"{type(exc).__name__}: {exc}",
            metrics={"fixture_path": str(ABTEM_REFERENCE_PATH)},
            started=started,
        )
    return _result(
        "tem_abtem_external_fixture",
        "pass" if ok else "fail",
        level="A/B",
        surface="TEM physical multislice",
        detail=detail,
        metrics=metrics,
        started=started,
    )


def _run_subprocess_check(
    name: str,
    args: list[str],
    *,
    level: str,
    surface: str,
    timeout_s: int | None = None,
) -> ValidationResult:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            name,
            "fail",
            level=level,
            surface=surface,
            detail=f"timed out after {timeout_s}s",
            metrics={"partial_output": exc.output},
            started=started,
        )
    tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
    return _result(
        name,
        "pass" if proc.returncode == 0 else "fail",
        level=level,
        surface=surface,
        detail=tail or f"returncode={proc.returncode}",
        metrics={"returncode": proc.returncode},
        started=started,
    )


def build_results(profile: str, *, require_external: bool) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    checks: list[tuple[str, str, str, Callable[[], tuple[bool, str, dict[str, Any]]]]] = [
        ("compile_core", "runtime", "all Python", check_compile_core),
        ("imports_and_defaults", "runtime", "core runtime", check_imports_and_defaults),
        ("parameter_contracts", "runtime", "config/schema", check_parameter_contracts),
        ("sem_response_contract_metadata", "runtime", "SEM metadata", check_sem_response_contract_metadata),
        ("material_contracts", "runtime", "material constants", check_material_contracts),
        ("material_single_source", "runtime", "material constants", check_material_single_source),
        ("electron_constants", "B", "TEM/electron constants", check_electron_constants),
        ("thinfilm_fresnel", "B", "thin-film reflection", check_thinfilm_fresnel),
        ("noise_poisson", "B", "camera noise", check_noise),
        ("lateral_fisher_gaussian", "B", "Fisher/CRLB", check_lateral_fisher),
        ("brownian_trajectory", "B", "trajectory", check_trajectory),
        ("rotational_diffusion", "B", "orientation trajectory", check_rotational_diffusion),
        ("manifest_packet_roundtrip", "internal", "metadata/packets", check_manifest_packet_roundtrip),
        ("iscat_identity", "B", "iSCAT contrast", check_iscat),
        ("flagship_cobri_dpc", "B", "COBRI/DPC contrast", check_flagship),
        ("psf_airy_limit", "B-", "scalar PSF", check_psf_airy),
        ("tem_multislice_internal", "B/internal", "TEM physical multislice", check_tem_internal),
        ("modality_equation_suite", "B/B-", "modality contrast equations", check_modality_equations),
        ("vectorial_debye_suite", "B/C", "vectorial optical PSF", check_vectorial_debye),
        ("sem_physical_smoke", "C/smoke", "SEM physical transport", check_sem_smoke),
    ]
    for name, level, surface, check in checks:
        results.append(_run_check(name, level, surface, check))

    if profile in {"external", "release"}:
        results.append(check_mie_external(require=require_external))
        results.append(check_tmm_external(require=require_external))
        results.append(check_tem_abtem_external(require=require_external))

    if profile in {"sem", "external"}:
        results.append(
            _run_subprocess_check(
                "sem_transport_reference_sweep",
                [
                    "validation/sem_transport_validation.py",
                    "--histories",
                    "100000",
                    "--max-steps",
                    "4096",
                    "--elastic-model",
                    "screened_rutherford",
                ],
                level="C/reference",
                surface="SEM physical transport",
                timeout_s=None,
            )
        )

    return results


def summarize(results: list[ValidationResult], *, fail_on_skip: bool) -> tuple[int, dict[str, Any]]:
    failed = [row for row in results if row.status == "fail"]
    skipped = [row for row in results if row.status == "skip"]
    exit_code = 1 if failed or (fail_on_skip and skipped) else 0
    summary = {
        "status": "pass" if exit_code == 0 else "fail",
        "counts": {
            "pass": sum(row.status == "pass" for row in results),
            "fail": len(failed),
            "skip": len(skipped),
        },
        "results": [asdict(row) for row in results],
    }
    return exit_code, _json_safe(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["fast", "release", "sem", "external"],
        default="fast",
        help=(
            "fast: core checks only; release: fast + optional external checks; "
            "sem: core + long SEM reference sweep; external: optional checks + long SEM sweep"
        ),
    )
    parser.add_argument("--require-external", action="store_true", help="fail if optional external validators are unavailable")
    parser.add_argument("--fail-on-skip", action="store_true", help="treat skipped optional checks as failures")
    parser.add_argument("--json-output", type=Path, default=None, help="write JSON result manifest to this path")
    args = parser.parse_args()

    started = time.perf_counter()
    results = build_results(args.profile, require_external=args.require_external)
    exit_code, summary = summarize(results, fail_on_skip=args.fail_on_skip)
    summary["profile"] = args.profile
    summary["duration_s"] = time.perf_counter() - started

    print(f"Syniscopy validation profile={args.profile}")
    for row in results:
        print(
            f"{row.status.upper():4} {row.name:34} "
            f"[{row.level}] {row.surface} ({row.duration_s:.2f}s)"
        )
        if row.status != "pass":
            print(f"     {row.detail}")
    print(
        "Summary: "
        f"{summary['counts']['pass']} pass, "
        f"{summary['counts']['fail']} fail, "
        f"{summary['counts']['skip']} skip"
    )

    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
