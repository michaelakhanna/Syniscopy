"""
camera_noise.py - Counts-domain camera-noise model for Syniscopy.

The user-facing image level is camera counts/ADU. The model keeps shot noise
in physical count units without making users tune an unobserved photon/electron
budget:

    signal_counts -> signal_electrons = signal_counts * camera_gain_e_per_count
    Poisson sample in electrons
    convert back to counts

camera_gain_e_per_count is a camera-conversion/calibration parameter. For a
calibrated reproduction workflow it should be estimated from the real video or
left at the documented default; it is not treated as a free signal-fitting knob.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
from threading import RLock
from typing import Any

import numpy as np

from config.runtime import (
    param_value,
    resolved_background_intensity,
    resolved_dark_field_background_count,
    resolved_dark_field_illumination_count,
    resolved_detector_qe,
    resolved_modality,
    resolved_qpi_phase_to_count_scale,
)
from measurement_units import normalize_detector_noise_input_domain
from modality_registry import canonical_modality_name, is_electron_modality, is_fluorescence_modality
from shared_constants import RAW_BACKGROUND_SUBTRACTION_METHODS


@dataclass
class DetectorNoiseRuntime:
    """Run-scoped detector-noise state: random source plus resolved map caches."""

    rng: Any = field(default_factory=np.random.default_rng)
    static_detector_map_cache: dict[tuple, dict[str, np.ndarray]] = field(default_factory=dict)
    camera_noise_map_cache: dict[tuple[str, tuple[int, ...], Any], np.ndarray] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)

    def clear_static_detector_maps(self) -> None:
        with self.lock:
            self.static_detector_map_cache.clear()

    def clear_resolved_maps(self) -> None:
        with self.lock:
            self.camera_noise_map_cache.clear()

    def clear(self) -> None:
        with self.lock:
            self.static_detector_map_cache.clear()
            self.camera_noise_map_cache.clear()


_DEFAULT_DETECTOR_NOISE_RUNTIME = DetectorNoiseRuntime()


def _noise_runtime(runtime: DetectorNoiseRuntime | None = None) -> DetectorNoiseRuntime:
    return runtime if runtime is not None else _DEFAULT_DETECTOR_NOISE_RUNTIME


def detector_noise_runtime_from_seed(seed: int | None = None, *, rng: Any | None = None) -> DetectorNoiseRuntime:
    """Return an isolated detector-noise runtime for deterministic direct calls."""
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng to detector_noise_runtime_from_seed(), not both.")
    resolved_rng = rng if rng is not None else np.random.default_rng(None if seed is None else int(seed))
    return DetectorNoiseRuntime(rng=resolved_rng)


def clear_detector_static_noise_cache(runtime: DetectorNoiseRuntime | None = None) -> None:
    """Clear cached fixed-pattern/hot-pixel maps for long dataset runs."""
    _noise_runtime(runtime).clear_static_detector_maps()


@dataclass(frozen=True)
class CameraNoiseConfig:
    shot_noise_enabled: bool
    gaussian_noise_enabled: bool
    camera_gain_e_per_count: float
    detector_qe: float
    detector_input_is_incident_quanta: bool
    detector_noise_input_domain: str
    emccd_enabled: bool
    emccd_gain: float
    emccd_excess_noise_factor: float
    read_noise_e: float | None
    dark_current_e_per_pixel_per_s: float
    exposure_time_s: float
    saturation_level: float | None
    saturation_e: float | None
    adc_quantization: bool
    adc_quantization_counts: float
    background_offset_counts: float
    read_noise_counts: float
    dark_offset_counts: float
    fixed_pattern_gain_std: float
    fixed_pattern_offset_counts: float
    hot_pixel_fraction: float
    hot_pixel_value_counts: float | None
    fixed_pattern_gain_map: str | None
    fixed_pattern_offset_map: str | None
    scmos_gain_map: str | None
    scmos_variance_map: str | None
    scmos_read_noise_map: str | None
    read_noise_map_mode: str
    hot_pixel_mask: str | None
    nonlinearity_calibration: str | None
    flat_field_map: str | None
    dark_frame_map: str | None
    scan_line_noise_counts: float
    clip_output_to_nonnegative: bool
    noise_parameterization: str
    nonlinear_detector_effects_active: bool
    deterministic_detector_transfer_active: bool
    safe_for_linear_fisher_variance: bool


def _normalise_noise_key(name: Any) -> str:
    return str(name).strip().lower()


def _explicit_noise_input_domain(noise_model: dict[str, Any]) -> Any | None:
    if "detector_noise_input_domain" in noise_model:
        return noise_model["detector_noise_input_domain"]
    return None


def _resolved_detector_noise_input_domain(params: dict[str, Any]) -> str:
    noise_model = _resolved_noise_model(params)
    explicit = _explicit_noise_input_domain(noise_model)
    if explicit is not None:
        domain = explicit
    else:
        configured = param_value(params, "detector_noise_input_domain")
        if configured is not None:
            domain = configured
        else:
            modality = resolved_modality(params)
            domain = "electron_count" if is_electron_modality(modality) else "camera_counts"
    return normalize_detector_noise_input_domain(domain)


def _map_cache_key(kind: str, raw: Any) -> tuple[str, tuple[int, ...], Any]:
    if isinstance(raw, (str, Path)):
        path = Path(raw).expanduser()
        try:
            stat = path.stat()
            mtime = int(stat.st_mtime)
            size = int(stat.st_size)
        except OSError:
            mtime = 0
            size = 0
        return (kind, (0,), (str(path), size, mtime,))
    if isinstance(raw, np.ndarray):
        fingerprint = hashlib.blake2b(raw.tobytes(), digest_size=8).hexdigest()
        return (kind, tuple(raw.shape), (raw.dtype.str, fingerprint,))
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw)
        fingerprint = hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()
        return (kind, tuple(arr.shape), (arr.dtype.str, fingerprint,))
    return (kind, tuple(), (type(raw).__name__, raw,))


def _load_map_array(path: str) -> np.ndarray:
    path_obj = Path(path).expanduser()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Noise map file not found: {path_obj}")
    ext = path_obj.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path_obj))
        return np.asarray(arr)
    if ext == ".npz":
        with np.load(str(path_obj)) as data:
            if len(data.files) == 0:
                raise ValueError(f"Noise map .npz archive is empty: {path_obj}")
            arr = data[data.files[0]]
        return np.asarray(arr)
    if ext in {".json", ".jsn"}:
        raw = json.loads(path_obj.read_text(encoding="utf-8"))
        return np.asarray(raw)
    try:
        return np.loadtxt(str(path_obj))
    except ValueError as exc:
        raise ValueError(f"Noise map file format not supported: {path_obj}") from exc


def _coerce_map_value(
    raw: Any,
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | float | int | bool | None, str]:
    if raw is None:
        return None, ""
    if isinstance(raw, (str, bytes, Path)):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        text = str(raw).strip()
    else:
        text = None
    if text is not None:
        if not text:
            raise TypeError(f"Unsupported detector noise map value type: {type(raw)!r}.")
        path = Path(text).expanduser()
        if path.exists():
            cache_key = _map_cache_key("file", path)
            active_runtime = _noise_runtime(runtime)
            with active_runtime.lock:
                if cache_key not in active_runtime.camera_noise_map_cache:
                    active_runtime.camera_noise_map_cache[cache_key] = np.asarray(_load_map_array(str(path)), dtype=dtype)
                return active_runtime.camera_noise_map_cache[cache_key], str(path)
        try:
            return np.array(float(text), dtype=dtype), "inline_scalar"
        except ValueError as exc:
            raise FileNotFoundError(f"Noise map file not found and not parseable as a scalar: {path}") from exc
    if isinstance(raw, bool):
        return np.array(float(raw), dtype=float), "inline"
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return np.array(float(raw), dtype=dtype), "inline_scalar"
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=dtype), "inline_array"
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=dtype)
        key = _map_cache_key("inline", arr)
        active_runtime = _noise_runtime(runtime)
        with active_runtime.lock:
            if key not in active_runtime.camera_noise_map_cache:
                active_runtime.camera_noise_map_cache[key] = arr
            return active_runtime.camera_noise_map_cache[key], f"inline_array{arr.shape}"
    raise TypeError(f"Unsupported detector noise map value type: {type(raw)!r}.")


def _resolve_map(
    raw: Any,
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | None, str]:
    value, summary = _coerce_map_value(raw, dtype=dtype, runtime=runtime)
    if value is None:
        return None, ""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr), summary
    return np.asarray(arr, dtype=dtype), summary


def _resolve_boolean_mask(
    name: str,
    raw: Any,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | None, str]:
    if raw is None:
        return None, ""
    if isinstance(raw, (str, bytes, Path)):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        arr = _coerce_map_value(raw, dtype=float, runtime=runtime)[0]
        if np.asarray(arr).ndim == 0:
            return np.asarray(bool(np.asarray(arr) > 0.5)), str(raw)
        return np.asarray(arr, dtype=float) > 0.5, str(raw)
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.asarray(bool(arr > 0.5)), f"inline_array{arr.shape}"
        return (arr > 0.5), f"inline_array{arr.shape}"
    if isinstance(raw, (int, float, bool, np.integer, np.floating)):
        value = float(raw)
        if not np.isfinite(value):
            raise TypeError(f"Unsupported detector noise boolean map value type: {type(raw)!r}.")
        return np.asarray(value > 0.5), "inline_scalar"
    if np.asarray(raw).ndim == 0:
        return np.asarray(float(np.asarray(raw))) > 0.5, "inline_scalar"
    raise TypeError(f"Unsupported detector noise boolean map value type: {type(raw)!r}.")


def _resolve_map_shape(
    name: str,
    raw: Any,
    shape: tuple[int, ...],
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | float | None, str]:
    value, summary = _resolve_map(raw, dtype=dtype, runtime=runtime)
    if value is None:
        return None, summary
    arr = np.asarray(value)
    if not shape:
        if arr.ndim == 0:
            return float(arr), summary
        return float(np.mean(arr)), summary
    if arr.ndim > 0 and tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"{name} map shape mismatch: expected {tuple(shape)} but got {tuple(arr.shape)}. "
            "Respecify the map to match the frame shape or supply a scalar override."
        )
    if np.any(~np.isfinite(np.asarray(arr, dtype=float))):
        raise ValueError(f"{name} map must contain finite values; got non-finite entries.")
    return value, summary


def _map_to_array(value: np.ndarray | float | int, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    return arr


def _mask_to_array(value: np.ndarray | bool, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(shape, bool(arr), dtype=bool)
    return arr.astype(bool)


def _resolved_noise_model(params: dict[str, Any]) -> dict[str, Any]:
    """Return the effective camera-noise configuration for this modality.

    Precedence is: base params < params["noise_model"] <
    params["modality_noise"][imaging_model]. Only noise_model and
    modality_noise are public configuration containers.
    """
    cfg: dict[str, Any] = {}
    noise_model = param_value(params, 'noise_model') or {}
    if isinstance(noise_model, dict):
        cfg.update(noise_model)

    modality_key = _normalise_noise_key(resolved_modality(params))
    per_modality = param_value(params, 'modality_noise') or {}
    if isinstance(per_modality, dict) and modality_key:
        for raw_key, override in per_modality.items():
            raw_norm = _normalise_noise_key(canonical_modality_name(raw_key))
            if raw_norm == modality_key:
                if not isinstance(override, dict):
                    raise TypeError(
                        "PARAMS['modality_noise'][imaging_model] must be a dictionary."
                    )
                cfg.update(override)
                break
    return cfg


def _cfg_value(params: dict[str, Any], key: str) -> Any:
    noise_model = _resolved_noise_model(params)
    if key in noise_model:
        return noise_model[key]
    return param_value(params, key)


def resolve_camera_noise_config(params: dict[str, Any] | None = None) -> CameraNoiseConfig:
    """
    Return the effective counts-domain camera-noise configuration.

    Values are resolved from PARAMS plus the optional ``noise_model`` and
    ``modality_noise`` override containers. Numeric noise amplitudes are in
    camera counts unless the field name states otherwise.
    """
    params = dict(params or {})
    detector_noise_input_domain = _resolved_detector_noise_input_domain(params)

    def _cfg_bool(key: str) -> bool:
        value = _cfg_value(params, key)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean true/false value; got {value!r}.")
        return bool(value)

    gain = float(_cfg_value(params, "camera_gain_e_per_count"))
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"camera_gain_e_per_count must be finite and positive; got {gain}.")
    if detector_noise_input_domain == "electron_count":
        gain = 1.0

    noise_model = _resolved_noise_model(params)
    if "detector_qe" in noise_model:
        detector_qe = float(noise_model["detector_qe"])
    elif "fluorescence_detector_qe" in noise_model:
        detector_qe = float(noise_model["fluorescence_detector_qe"])
    else:
        detector_qe = resolved_detector_qe(
            params,
            fluorescence=is_fluorescence_modality(resolved_modality(params)),
        )
    if not np.isfinite(detector_qe) or detector_qe < 0.0 or detector_qe > 1.0:
        raise ValueError(f"detector_qe must be finite and in [0, 1]; got {detector_qe}.")
    emccd_enabled = _cfg_bool("emccd_enabled")
    emccd_gain = float(_cfg_value(params, "emccd_gain"))
    if not np.isfinite(emccd_gain) or emccd_gain <= 0.0:
        raise ValueError(f"emccd_gain must be finite and positive; got {emccd_gain}.")
    emccd_excess = float(_cfg_value(params, "emccd_excess_noise_factor"))
    if not np.isfinite(emccd_excess) or emccd_excess < 1.0:
        raise ValueError(f"emccd_excess_noise_factor must be finite and >= 1; got {emccd_excess}.")
    read_noise_e = _cfg_value(params, "read_noise_e")
    read_noise = float(_cfg_value(params, "read_noise_counts"))
    if read_noise_e is not None:
        read_noise = float(read_noise_e) / gain
    if not np.isfinite(read_noise) or read_noise < 0.0:
        raise ValueError(f"read_noise_counts must be finite and non-negative; got {read_noise}.")

    dark_offset = float(_cfg_value(params, "dark_offset_counts"))
    background_offset = float(_cfg_value(params, "background_offset_counts"))
    dark_current_e = float(_cfg_value(params, "dark_current_e_per_pixel_per_s"))
    exposure_time_s = float(_cfg_value(params, "exposure_time_s"))
    saturation_level = _cfg_value(params, "saturation_level")
    saturation_level = None if saturation_level is None else float(saturation_level)
    saturation_e = _cfg_value(params, "saturation_e")
    saturation_e = None if saturation_e is None else float(saturation_e)
    adc_quantization = _cfg_bool("adc_quantization")
    adc_quantization_counts = float(_cfg_value(params, "adc_quantization_counts"))
    fpn_gain = float(_cfg_value(params, "fixed_pattern_gain_std"))
    fpn_offset = float(_cfg_value(params, "fixed_pattern_offset_counts"))
    hot_fraction = float(_cfg_value(params, "hot_pixel_fraction"))
    hot_value = _cfg_value(params, "hot_pixel_value_counts")
    hot_value = None if hot_value is None else float(hot_value)
    line_noise = float(_cfg_value(params, "scan_line_noise_counts"))
    fixed_pattern_gain_map = _cfg_value(params, "fixed_pattern_gain_map")
    fixed_pattern_offset_map = _cfg_value(params, "fixed_pattern_offset_map")
    scmos_variance_map = _cfg_value(params, "scmos_variance_map")
    scmos_gain_map = _cfg_value(params, "scmos_gain_map")
    scmos_read_noise_map = _cfg_value(params, "scmos_read_noise_map")
    read_noise_map_mode = str(_cfg_value(params, "read_noise_map_mode")).strip().lower()
    if read_noise_map_mode not in {"replace", "add"}:
        raise ValueError(
            "read_noise_map_mode must be 'replace' or 'add'; "
            f"got {read_noise_map_mode!r}."
        )
    hot_pixel_mask = _cfg_value(params, "hot_pixel_mask")
    flat_field_map = _cfg_value(params, "flat_field_map")
    dark_frame_map = _cfg_value(params, "dark_frame_map")
    if fixed_pattern_gain_map is not None:
        _coerce_map_value(fixed_pattern_gain_map, dtype=float)
    if fixed_pattern_offset_map is not None:
        _coerce_map_value(fixed_pattern_offset_map, dtype=float)
    if scmos_variance_map is not None:
        _coerce_map_value(scmos_variance_map, dtype=float)
    if scmos_gain_map is not None:
        _coerce_map_value(scmos_gain_map, dtype=float)
    if scmos_read_noise_map is not None:
        _coerce_map_value(scmos_read_noise_map, dtype=float)
    if flat_field_map is not None:
        _coerce_map_value(flat_field_map, dtype=float)
    if dark_frame_map is not None:
        _coerce_map_value(dark_frame_map, dtype=float)
    if hot_pixel_mask is not None:
        _resolve_boolean_mask("hot_pixel_mask", hot_pixel_mask)

    # Base validation: required non-negative scalar noise terms.
    for key, value in {
        "fixed_pattern_gain_std": fpn_gain,
        "fixed_pattern_offset_counts": fpn_offset,
        "hot_pixel_fraction": hot_fraction,
        "scan_line_noise_counts": line_noise,
        "background_offset_counts": background_offset,
        "dark_current_e_per_pixel_per_s": dark_current_e,
        "exposure_time_s": exposure_time_s,
        "adc_quantization_counts": adc_quantization_counts,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and non-negative; got {value}.")

    # Hot pixel fraction is additionally bounded above by one.
    if hot_fraction > 1.0:
        raise ValueError(f"hot_pixel_fraction must be <= 1.0; got {hot_fraction}.")

    # ADC quantization must be strictly positive when enabled.
    if adc_quantization and adc_quantization_counts <= 0.0:
        raise ValueError("adc_quantization_counts must be positive when adc_quantization is enabled.")
    if saturation_level is not None and (not np.isfinite(saturation_level) or saturation_level <= 0.0):
        raise ValueError(f"saturation_level must be finite and positive when supplied; got {saturation_level}.")
    if saturation_e is not None and (not np.isfinite(saturation_e) or saturation_e <= 0.0):
        raise ValueError(f"saturation_e must be finite and positive when supplied; got {saturation_e}.")

    noise_parameterization = str(_cfg_value(params, "noise_parameterization")).strip().lower()
    if noise_parameterization != "camera_counts":
        raise ValueError(
            "noise_parameterization must be 'camera_counts'; "
            f"got {_cfg_value(params, 'noise_parameterization')!r}."
        )
    detector_input_is_incident_quanta = _cfg_bool("detector_input_is_incident_quanta")
    if detector_noise_input_domain == "electron_count":
        detector_input_is_incident_quanta = False

    nonlinearity_calibration = _cfg_value(params, "nonlinearity_calibration")
    if nonlinearity_calibration is not None:
        nonlinearity_calibration = str(nonlinearity_calibration)

    nonlinear_active = bool(
        saturation_level is not None
        or saturation_e is not None
        or adc_quantization
        or nonlinearity_calibration is not None
    )
    deterministic_transfer_active = bool(
        fixed_pattern_gain_map is not None
        or fixed_pattern_offset_map is not None
        or scmos_gain_map is not None
        or flat_field_map is not None
        or dark_frame_map is not None
        or hot_pixel_mask is not None
        or fpn_gain > 0.0
        or fpn_offset > 0.0
        or hot_fraction > 0.0
    )
    safe_for_linear_fisher_variance = not (nonlinear_active or deterministic_transfer_active)

    return CameraNoiseConfig(
        shot_noise_enabled=_cfg_bool("shot_noise_enabled"),
        gaussian_noise_enabled=_cfg_bool("gaussian_noise_enabled"),
        camera_gain_e_per_count=gain,
        detector_qe=detector_qe,
        detector_input_is_incident_quanta=detector_input_is_incident_quanta,
        detector_noise_input_domain=detector_noise_input_domain,
        emccd_enabled=emccd_enabled,
        emccd_gain=emccd_gain,
        emccd_excess_noise_factor=emccd_excess,
        read_noise_e=None if read_noise_e is None else float(read_noise_e),
        read_noise_counts=read_noise,
        dark_current_e_per_pixel_per_s=dark_current_e,
        exposure_time_s=exposure_time_s,
        saturation_level=saturation_level,
        saturation_e=saturation_e,
        adc_quantization=adc_quantization,
        adc_quantization_counts=adc_quantization_counts,
        background_offset_counts=background_offset,
        dark_offset_counts=dark_offset,
        fixed_pattern_gain_std=fpn_gain,
        fixed_pattern_offset_counts=fpn_offset,
        fixed_pattern_gain_map=fixed_pattern_gain_map,
        fixed_pattern_offset_map=fixed_pattern_offset_map,
        scmos_variance_map=scmos_variance_map,
        scmos_gain_map=scmos_gain_map,
        scmos_read_noise_map=scmos_read_noise_map,
        read_noise_map_mode=read_noise_map_mode,
        hot_pixel_mask=hot_pixel_mask,
        nonlinearity_calibration=nonlinearity_calibration,
        flat_field_map=flat_field_map,
        dark_frame_map=dark_frame_map,
        hot_pixel_fraction=hot_fraction,
        hot_pixel_value_counts=hot_value,
        scan_line_noise_counts=line_noise,
        clip_output_to_nonnegative=_cfg_bool("clip_output_to_nonnegative"),
        noise_parameterization=noise_parameterization,
        nonlinear_detector_effects_active=nonlinear_active,
        deterministic_detector_transfer_active=deterministic_transfer_active,
        safe_for_linear_fisher_variance=safe_for_linear_fisher_variance,
    )


def camera_noise_metadata(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved camera-noise configuration as JSON-safe metadata."""
    cfg = resolve_camera_noise_config(params)
    meta = asdict(cfg)
    meta["scan_line_noise_in_fisher_variance"] = bool(cfg.scan_line_noise_counts > 0.0)
    meta["noise_covariance_kind"] = (
        "row_correlated_scan_lines_diagonalized"
        if cfg.scan_line_noise_counts > 0.0
        else "independent_pixels"
    )
    meta["noise_independence_assumption"] = (
        "row-correlated scan-line noise is included as diagonal variance; "
        "off-diagonal row covariance is not represented"
        if cfg.scan_line_noise_counts > 0.0
        else "independent pixel variance"
    )
    meta["emccd_gain_applied_stage"] = (
        "input_referred_read_noise_reduction"
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0
        else "not_applied"
    )
    meta["fisher_variance_model_scope"] = (
        "linear_poisson_gaussian_only"
        if cfg.safe_for_linear_fisher_variance
        else "diagnostic_only_linearized_detector_variance"
    )
    meta["detector_likelihood_status"] = (
        "linear_poisson_gaussian_compatible"
        if cfg.safe_for_linear_fisher_variance
        else "nonlinear_or_static_transfer_not_in_linear_fisher"
    )
    return meta


def shot_noise_std_counts(signal_counts: np.ndarray | float, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return shot-noise standard deviation in camera counts."""
    cfg = resolve_camera_noise_config(params)
    counts = np.asarray(signal_counts, dtype=float)
    if not cfg.shot_noise_enabled:
        return np.zeros_like(counts, dtype=float)
    counts_pos = np.where(np.isfinite(counts) & (counts > 0.0), counts, 0.0)
    if cfg.detector_input_is_incident_quanta:
        counts_pos = counts_pos * cfg.detector_qe
    shot_variance = counts_pos / cfg.camera_gain_e_per_count
    if cfg.emccd_enabled:
        shot_variance = shot_variance * cfg.emccd_excess_noise_factor
    return np.sqrt(np.maximum(shot_variance, 0.0))


def total_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return combined shot and read-noise standard deviation in camera counts."""
    cfg = resolve_camera_noise_config(params)
    shot = shot_noise_std_counts(signal_counts, params)
    if not cfg.gaussian_noise_enabled:
        read = np.zeros_like(np.asarray(signal_counts, dtype=float), dtype=float)
    else:
        shape = np.asarray(signal_counts, dtype=float).shape
        scmos_variance_map, _ = _resolve_map_shape(
            "scmos_variance_map",
            cfg.scmos_variance_map,
            shape,
            dtype=float,
            runtime=runtime,
        )
        scmos_read_noise_map, _ = _resolve_map_shape(
            "scmos_read_noise_map",
            cfg.scmos_read_noise_map,
            shape,
            dtype=float,
            runtime=runtime,
        )
        has_calibrated_read_map = scmos_variance_map is not None or scmos_read_noise_map is not None
        if has_calibrated_read_map and cfg.read_noise_map_mode == "replace":
            read_var = np.zeros(shape, dtype=float)
        else:
            read_var = np.full(shape, cfg.read_noise_counts, dtype=float) ** 2
        if scmos_variance_map is not None:
            read_var = read_var + _map_to_array(scmos_variance_map, shape)
        if scmos_read_noise_map is not None:
            read_var = read_var + np.square(_map_to_array(scmos_read_noise_map, shape))
        if cfg.scan_line_noise_counts > 0.0:
            read_var = read_var + cfg.scan_line_noise_counts ** 2
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0:
            read_var = read_var / (cfg.emccd_gain ** 2)
        read = np.sqrt(np.maximum(read_var, 0.0))
    return np.sqrt(shot * shot + read * read)


def total_noise_variance_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return combined shot and read-noise variance in camera-count units."""
    std = total_noise_std_counts(signal_counts, params, runtime=runtime)
    return std * std


def contrast_noise_variance_counts(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Propagate detector noise into the contrast image used by Fisher diagnostics.

    For raw/no-subtraction views this returns the signal-frame count variance.
    For additive reference subtraction, ``C = S - R`` and
    ``Var(C) = Var(S) + Var(R)``. For relative reference contrast,
    ``C = (S - R) / R = S/R - 1`` and first-order propagation gives
    ``Var(C) = Var(S)/R^2 + S^2 Var(R)/R^4``.
    """
    params = dict(params or {})
    method = str(param_value(params, 'background_subtraction_method')).strip().lower()
    signal = np.asarray(signal_counts, dtype=float)
    var_signal = total_noise_variance_counts(signal, params, runtime=runtime)
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS or reference_counts is None:
        return np.maximum(var_signal, float(variance_floor))

    reference = np.asarray(reference_counts, dtype=float)
    if signal.shape != reference.shape:
        raise ValueError(
            "signal_counts and reference_counts must have the same shape for "
            f"contrast-noise propagation; got {signal.shape} and {reference.shape}."
        )
    var_reference = total_noise_variance_counts(reference, params, runtime=runtime)

    if relative_reference is None:
        imaging_model_name = resolved_modality(params)
        from imaging_models import modality_uses_relative_reference_contrast

        relative_reference = modality_uses_relative_reference_contrast(imaging_model_name)

    if bool(relative_reference):
        ref_safe = np.maximum(np.abs(reference), 1e-12)
        variance = var_signal / (ref_safe ** 2) + (signal ** 2) * var_reference / (ref_safe ** 4)
    else:
        variance = var_signal + var_reference
    return np.maximum(variance, float(variance_floor))


def analysis_contrast_noise_variance(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Propagate detector noise into the analysis contrast returned by public views.

    Most modalities use count-domain contrast conventions and can delegate
    directly to :func:`contrast_noise_variance_counts`. Phase-output modalities
    such as QPI expose phase contrast in radians, while their rendered signal
    and reference frames are count-like display images. For those modes, count
    variance is converted to phase variance by the square of the configured
    phase-to-count scale unless an explicit phase-noise standard deviation is
    supplied.
    """
    params = dict(params or {})
    imaging_model_name = resolved_modality(params)
    from imaging_models import get_imaging_model_class

    output_type = getattr(get_imaging_model_class(imaging_model_name), "output_type", "intensity")
    if output_type == "phase":
        phase_noise = param_value(params, 'qpi_phase_noise_std_rad')
        signal = np.asarray(signal_counts, dtype=float)
        if phase_noise is not None:
            sigma = float(phase_noise)
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError(
                    "qpi_phase_noise_std_rad must be non-negative and finite "
                    f"when supplied; got {phase_noise!r}."
                )
            return np.maximum(
                np.full(signal.shape, sigma * sigma, dtype=float),
                float(variance_floor),
            )
        phase_to_count = resolved_qpi_phase_to_count_scale(params)
        if not np.isfinite(phase_to_count) or phase_to_count <= 0.0:
            raise ValueError(
                "qpi_phase_to_count_scale must be positive and finite for "
                "phase-domain noise propagation."
            )
        count_variance = contrast_noise_variance_counts(
            signal_counts,
            reference_counts,
            params,
            relative_reference=False if relative_reference is None else relative_reference,
            variance_floor=variance_floor,
            runtime=runtime,
        )
        return np.maximum(count_variance / (phase_to_count * phase_to_count), float(variance_floor))

    return contrast_noise_variance_counts(
        signal_counts,
        reference_counts,
        params,
        relative_reference=relative_reference,
        variance_floor=variance_floor,
        runtime=runtime,
    )


def estimate_detector_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Return detector-noise standard deviation in camera counts/ADU.

    This is the counts-domain helper expected by supervision and audit code.
    It delegates to the canonical total_noise_std_counts implementation so
    rendering, supervision, and metadata use the same camera-noise model.
    """
    return total_noise_std_counts(signal_counts, params, runtime=runtime)


def _seed_component(value: float) -> int:
    return int(round(abs(float(value)) * 1_000_000_000.0)) % (2**32)


def _detector_static_noise_seed(
    params: dict[str, Any] | None,
    runtime: DetectorNoiseRuntime | None = None,
) -> int:
    """
    Return a simulation-stable seed for static detector maps.

    Fixed-pattern gain/offset maps and hot-pixel locations are detector
    properties: they must persist across frames and across signal/reference
    renders within one simulated video. The public random_seed gives
    reproducible per-video detector maps; if no seed is supplied, a run-scoped
    internal seed is assigned once to the active parameter dictionary.
    """
    if not isinstance(params, dict):
        return 0
    internal_key = "_camera_noise_static_seed"
    if internal_key in params:
        return int(params[internal_key]) % (2**32)
    if "random_seed" in params and params["random_seed"] is not None:
        seed = int(params["random_seed"]) % (2**32)
    else:
        seed = int(float(_noise_runtime(runtime).rng.random()) * (2**32 - 1)) % (2**32)
    params[internal_key] = seed
    return seed


def _static_detector_maps(
    shape: tuple[int, ...],
    cfg: CameraNoiseConfig,
    params: dict[str, Any] | None,
    runtime: DetectorNoiseRuntime | None = None,
) -> dict[str, np.ndarray]:
    active_runtime = _noise_runtime(runtime)
    seed = _detector_static_noise_seed(params, active_runtime)
    key = (
        seed,
        tuple(int(x) for x in shape),
        _seed_component(cfg.fixed_pattern_gain_std),
        _seed_component(cfg.fixed_pattern_offset_counts),
        _seed_component(cfg.hot_pixel_fraction),
    )
    with active_runtime.lock:
        cached = active_runtime.static_detector_map_cache.get(key)
        if cached is not None:
            return cached

    seed_sequence = np.random.SeedSequence(
        [
            seed,
            len(shape),
            *[int(x) % (2**32) for x in shape],
            key[2],
            key[3],
            key[4],
        ]
    )
    static_rng = np.random.default_rng(seed_sequence)
    maps: dict[str, np.ndarray] = {}
    if cfg.fixed_pattern_gain_std > 0.0:
        maps["gain"] = static_rng.normal(
            loc=1.0,
            scale=cfg.fixed_pattern_gain_std,
            size=shape,
        )
    if cfg.fixed_pattern_offset_counts > 0.0:
        maps["offset"] = static_rng.normal(
            loc=0.0,
            scale=cfg.fixed_pattern_offset_counts,
            size=shape,
        )
    if cfg.hot_pixel_fraction > 0.0:
        maps["hot_mask"] = static_rng.random(size=shape) < cfg.hot_pixel_fraction

    with active_runtime.lock:
        active_runtime.static_detector_map_cache[key] = maps
    return maps



def apply_camera_noise_counts(
    frame_counts: np.ndarray,
    params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Apply camera noise to a frame already expressed in camera counts/ADU."""
    if rng is not None and random_seed is not None:
        raise ValueError("Pass either rng or random_seed to apply_camera_noise_counts(), not both.")
    if runtime is not None and random_seed is not None:
        raise ValueError("Pass either runtime or random_seed to apply_camera_noise_counts(), not both.")
    active_runtime = (
        detector_noise_runtime_from_seed(random_seed)
        if random_seed is not None
        else _noise_runtime(runtime)
    )
    cfg = resolve_camera_noise_config(params)
    rng = rng if rng is not None else active_runtime.rng

    clean = np.asarray(frame_counts, dtype=float)
    shape = clean.shape
    noisy = clean.copy()
    fixed_pattern_gain_map, _ = _resolve_map_shape(
        "fixed_pattern_gain_map",
        cfg.fixed_pattern_gain_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    fixed_pattern_offset_map, _ = _resolve_map_shape(
        "fixed_pattern_offset_map",
        cfg.fixed_pattern_offset_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    scmos_gain_map, _ = _resolve_map_shape("scmos_gain_map", cfg.scmos_gain_map, shape, dtype=float, runtime=active_runtime)
    scmos_variance_map, _ = _resolve_map_shape(
        "scmos_variance_map",
        cfg.scmos_variance_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    scmos_read_noise_map, _ = _resolve_map_shape(
        "scmos_read_noise_map",
        cfg.scmos_read_noise_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    flat_field_map, _ = _resolve_map_shape("flat_field_map", cfg.flat_field_map, shape, dtype=float, runtime=active_runtime)
    dark_frame_map, _ = _resolve_map_shape("dark_frame_map", cfg.dark_frame_map, shape, dtype=float, runtime=active_runtime)
    hot_pixel_mask, _ = _resolve_boolean_mask("hot_pixel_mask", cfg.hot_pixel_mask, runtime=active_runtime)

    static_maps: dict[str, np.ndarray] = {}
    if (
        (fixed_pattern_gain_map is None and cfg.fixed_pattern_gain_std > 0.0)
        or (fixed_pattern_offset_map is None and cfg.fixed_pattern_offset_counts > 0.0)
        or (hot_pixel_mask is None and cfg.hot_pixel_fraction > 0.0)
    ):
        static_maps = _static_detector_maps(clean.shape, cfg, params, runtime=active_runtime)

    if fixed_pattern_gain_map is None and "gain" in static_maps:
        fixed_pattern_gain_map = static_maps["gain"]
    if fixed_pattern_offset_map is None and "offset" in static_maps:
        fixed_pattern_offset_map = static_maps["offset"]
    if hot_pixel_mask is None and "hot_mask" in static_maps:
        hot_pixel_mask = static_maps["hot_mask"]

    if fixed_pattern_gain_map is not None:
        fixed_pattern_gain_map = _map_to_array(fixed_pattern_gain_map, shape)
    if fixed_pattern_offset_map is not None:
        fixed_pattern_offset_map = _map_to_array(fixed_pattern_offset_map, shape)
    if scmos_variance_map is not None:
        scmos_variance_map = _map_to_array(scmos_variance_map, shape)
    if scmos_gain_map is not None:
        scmos_gain_map = _map_to_array(scmos_gain_map, shape)
    if scmos_read_noise_map is not None:
        scmos_read_noise_map = _map_to_array(scmos_read_noise_map, shape)
    if flat_field_map is not None:
        flat_field_map = _map_to_array(flat_field_map, shape)
    if dark_frame_map is not None:
        dark_frame_map = _map_to_array(dark_frame_map, shape)
    if hot_pixel_mask is not None:
        hot_pixel_mask = _mask_to_array(hot_pixel_mask, shape)

    if cfg.background_offset_counts:
        noisy = noisy + cfg.background_offset_counts
    if cfg.dark_current_e_per_pixel_per_s and cfg.exposure_time_s:
        noisy = noisy + (cfg.dark_current_e_per_pixel_per_s * cfg.exposure_time_s / cfg.camera_gain_e_per_count)
    if cfg.dark_offset_counts:
        noisy = noisy + cfg.dark_offset_counts
    if dark_frame_map is not None:
        noisy = noisy + dark_frame_map

    if cfg.shot_noise_enabled:
        counts_pos = np.where(np.isfinite(noisy) & (noisy > 0.0), noisy, 0.0)
        detected_counts_mean = counts_pos * cfg.detector_qe if cfg.detector_input_is_incident_quanta else counts_pos
        electron_mean = detected_counts_mean * cfg.camera_gain_e_per_count
        electron_sample = rng.poisson(electron_mean).astype(float)
        poisson_counts = electron_sample / cfg.camera_gain_e_per_count
        shot_residual = poisson_counts - detected_counts_mean
        noisy = detected_counts_mean + shot_residual
        if cfg.emccd_enabled and cfg.emccd_excess_noise_factor > 1.0:
            excess_var = np.maximum(detected_counts_mean / cfg.camera_gain_e_per_count, 0.0) * (cfg.emccd_excess_noise_factor - 1.0)
            noisy = noisy + rng.normal(scale=np.sqrt(excess_var), size=clean.shape)

    if flat_field_map is not None:
        noisy = noisy * flat_field_map

    if fixed_pattern_gain_map is not None:
        noisy = noisy * fixed_pattern_gain_map
    elif cfg.fixed_pattern_gain_std > 0.0:
        noisy = noisy * static_maps["gain"]

    if scmos_gain_map is not None:
        noisy = noisy * scmos_gain_map

    if fixed_pattern_offset_map is not None:
        noisy = noisy + fixed_pattern_offset_map
    elif cfg.fixed_pattern_offset_counts > 0.0:
        noisy = noisy + static_maps["offset"]

    if hot_pixel_mask is not None:
        hot_mask = hot_pixel_mask
        if cfg.hot_pixel_value_counts is None:
            finite = noisy[np.isfinite(noisy)]
            hot_value = float(finite.max()) if finite.size else 0.0
        else:
            hot_value = cfg.hot_pixel_value_counts
        noisy[hot_mask] = hot_value

    if cfg.scan_line_noise_counts > 0.0:
        noisy = noisy + rng.normal(scale=cfg.scan_line_noise_counts, size=(clean.shape[0], 1))

    if cfg.gaussian_noise_enabled:
        has_calibrated_read_map = scmos_variance_map is not None or scmos_read_noise_map is not None
        if has_calibrated_read_map and cfg.read_noise_map_mode == "replace":
            read_variance = np.zeros(shape, dtype=float)
        else:
            read_variance = np.full(shape, cfg.read_noise_counts, dtype=float) ** 2
        if scmos_variance_map is not None:
            read_variance = read_variance + scmos_variance_map
        if scmos_read_noise_map is not None:
            read_variance = read_variance + np.square(scmos_read_noise_map)
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0:
            read_variance = read_variance / (cfg.emccd_gain ** 2)
        read_sigma = np.sqrt(np.maximum(read_variance, 0.0))
        if np.any(read_sigma > 0.0):
            noisy = noisy + rng.normal(scale=read_sigma, size=clean.shape)

    if cfg.saturation_e is not None:
        noisy = np.minimum(noisy, cfg.saturation_e / cfg.camera_gain_e_per_count)
    if cfg.saturation_level is not None:
        noisy = np.minimum(noisy, cfg.saturation_level)
    if cfg.adc_quantization:
        noisy = np.round(noisy / cfg.adc_quantization_counts) * cfg.adc_quantization_counts

    if cfg.clip_output_to_nonnegative:
        noisy = np.where(np.isfinite(noisy), np.maximum(noisy, 0.0), 0.0)

    return noisy.astype(float, copy=False)



def estimate_contrast_noise_std_from_params(params: dict[str, Any]) -> float:
    """
    Estimate noise in contrast units for supervision support gates.

    For count images, contrast noise is noise_counts / reference_counts. This is
    deliberately tied to the same camera-count model used by rendering.
    """
    imaging_model = resolved_modality(params)

    if imaging_model in {"dark_field", "coherent_dark_field"}:
        background_counts = max(resolved_dark_field_background_count(params), 1e-9)
        normalization_counts = max(resolved_dark_field_illumination_count(params), 1e-9)
    else:
        background_counts = max(resolved_background_intensity(params), 1e-9)
        normalization_counts = background_counts

    noise_counts = float(total_noise_std_counts(background_counts, params))
    return float(noise_counts / normalization_counts)


def calibrate_camera_gain_e_per_count_from_video(
    video_path: str,
    *,
    max_frames: int = 80,
    sample_stride: int = 1,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    min_gain: float = 0.25,
    max_gain: float = 256.0,
) -> float:
    """
    Estimate an effective camera_gain_e_per_count from a real video.

    This is a practical count-domain calibration for dataset matching. It uses
    temporal variance over relatively flat pixels and estimates

        variance_counts ~= mean_counts / camera_gain_e_per_count.

    The result is an effective camera-conversion number for simulation matching,
    not a claim about the physical camera manual.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video calibration.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for noise calibration: {video_path}")

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % sample_stride == 0:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray.astype(np.float32))
        idx += 1
    cap.release()

    if len(frames) < 8:
        return 1.0

    stack = np.stack(frames, axis=0)
    temporal_mean = np.mean(stack, axis=0)
    temporal_var = np.var(stack, axis=0, ddof=1)

    # Reject saturated/extreme pixels and high-structure pixels so sample
    # features and background gradients do not dominate the noise estimate.
    lo = np.quantile(temporal_mean, lower_quantile)
    hi = np.quantile(temporal_mean, upper_quantile)
    gx = np.zeros_like(temporal_mean)
    gy = np.zeros_like(temporal_mean)
    gx[:, 1:-1] = np.abs(temporal_mean[:, 2:] - temporal_mean[:, :-2])
    gy[1:-1, :] = np.abs(temporal_mean[2:, :] - temporal_mean[:-2, :])
    grad = gx + gy
    grad_cut = np.quantile(grad, 0.50)
    mask = (
        np.isfinite(temporal_mean)
        & np.isfinite(temporal_var)
        & (temporal_mean > lo)
        & (temporal_mean < hi)
        & (temporal_var > 1e-6)
        & (grad <= grad_cut)
    )

    if np.count_nonzero(mask) < 100:
        mask = np.isfinite(temporal_mean) & np.isfinite(temporal_var) & (temporal_var > 1e-6)

    mean_counts = float(np.median(temporal_mean[mask]))
    var_counts = float(np.median(temporal_var[mask]))
    if not np.isfinite(mean_counts) or not np.isfinite(var_counts) or mean_counts <= 0.0 or var_counts <= 0.0:
        return 1.0

    gain = mean_counts / var_counts
    gain = max(float(min_gain), min(float(max_gain), float(gain)))
    return float(gain)
