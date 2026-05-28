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

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


class _GlobalNumpyRandomAdapter:
    """Adapter exposing Generator-like methods backed by np.random's global RNG."""

    @staticmethod
    def poisson(lam):
        return np.random.poisson(lam)

    @staticmethod
    def normal(*args, **kwargs):
        return np.random.normal(*args, **kwargs)

    @staticmethod
    def random(*args, **kwargs):
        return np.random.random(*args, **kwargs)


_GLOBAL_NUMPY_RNG = _GlobalNumpyRandomAdapter()
_STATIC_DETECTOR_MAP_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


@dataclass(frozen=True)
class CameraNoiseConfig:
    shot_noise_enabled: bool = True
    gaussian_noise_enabled: bool = True
    camera_gain_e_per_count: float = 1.0
    read_noise_counts: float = 1.0
    dark_offset_counts: float = 0.0
    fixed_pattern_gain_std: float = 0.0
    fixed_pattern_offset_counts: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_value_counts: float | None = None
    scan_line_noise_counts: float = 0.0
    clip_output_to_nonnegative: bool = True
    noise_parameterization: str = "camera_counts"


def _normalise_noise_key(name: Any) -> str:
    return str(name).strip().lower()


def _resolved_noise_model(params: dict[str, Any]) -> dict[str, Any]:
    """Return the effective camera-noise configuration for this modality.

    Precedence is: base params < params["noise_model"] <
    params["modality_noise"][imaging_model]. Only noise_model and
    modality_noise are public configuration containers.
    """
    cfg: dict[str, Any] = {}
    noise_model = params.get("noise_model", {}) or {}
    if isinstance(noise_model, dict):
        cfg.update(noise_model)

    modality_key = _normalise_noise_key(params.get("imaging_model", ""))
    per_modality = params.get("modality_noise", {}) or {}
    if isinstance(per_modality, dict) and modality_key:
        for raw_key, override in per_modality.items():
            if _normalise_noise_key(raw_key) == modality_key:
                if not isinstance(override, dict):
                    raise TypeError(
                        "PARAMS['modality_noise'][imaging_model] must be a dictionary."
                    )
                cfg.update(override)
                break
    return cfg


def _cfg_value(params: dict[str, Any], key: str, default: Any) -> Any:
    noise_model = _resolved_noise_model(params)
    if key in noise_model:
        return noise_model[key]
    return params.get(key, default)


def resolve_camera_noise_config(params: dict[str, Any] | None = None) -> CameraNoiseConfig:
    """
    Return the effective counts-domain camera-noise configuration.

    Values are resolved from PARAMS plus the optional ``noise_model`` and
    ``modality_noise`` override containers. Numeric noise amplitudes are in
    camera counts unless the field name states otherwise.
    """
    params = dict(params or {})

    def _cfg_bool(key: str, default: bool) -> bool:
        value = _cfg_value(params, key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean true/false value; got {value!r}.")
        return bool(value)

    gain = float(_cfg_value(params, "camera_gain_e_per_count", 1.0))
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"camera_gain_e_per_count must be finite and positive; got {gain}.")

    read_noise = float(_cfg_value(params, "read_noise_counts", 1.0))
    if not np.isfinite(read_noise) or read_noise < 0.0:
        raise ValueError(f"read_noise_counts must be finite and non-negative; got {read_noise}.")

    dark_offset = float(_cfg_value(params, "dark_offset_counts", 0.0))
    fpn_gain = float(_cfg_value(params, "fixed_pattern_gain_std", 0.0))
    fpn_offset = float(_cfg_value(params, "fixed_pattern_offset_counts", 0.0))
    hot_fraction = float(_cfg_value(params, "hot_pixel_fraction", 0.0))
    hot_value = _cfg_value(params, "hot_pixel_value_counts", None)
    hot_value = None if hot_value is None else float(hot_value)
    line_noise = float(_cfg_value(params, "scan_line_noise_counts", 0.0))

    for key, value in {
        "fixed_pattern_gain_std": fpn_gain,
        "fixed_pattern_offset_counts": fpn_offset,
        "hot_pixel_fraction": hot_fraction,
        "scan_line_noise_counts": line_noise,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and non-negative; got {value}.")
    if hot_fraction > 1.0:
        raise ValueError(f"hot_pixel_fraction must be <= 1.0; got {hot_fraction}.")

    noise_parameterization = str(_cfg_value(params, "noise_parameterization", "camera_counts")).strip().lower()
    if noise_parameterization != "camera_counts":
        raise ValueError(
            "noise_parameterization must be 'camera_counts'; "
            f"got {_cfg_value(params, 'noise_parameterization', 'camera_counts')!r}."
        )

    return CameraNoiseConfig(
        shot_noise_enabled=_cfg_bool("shot_noise_enabled", True),
        gaussian_noise_enabled=_cfg_bool("gaussian_noise_enabled", True),
        camera_gain_e_per_count=gain,
        read_noise_counts=read_noise,
        dark_offset_counts=dark_offset,
        fixed_pattern_gain_std=fpn_gain,
        fixed_pattern_offset_counts=fpn_offset,
        hot_pixel_fraction=hot_fraction,
        hot_pixel_value_counts=hot_value,
        scan_line_noise_counts=line_noise,
        clip_output_to_nonnegative=_cfg_bool("clip_output_to_nonnegative", True),
        noise_parameterization=noise_parameterization,
    )


def camera_noise_metadata(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved camera-noise configuration as JSON-safe metadata."""
    return asdict(resolve_camera_noise_config(params))


def shot_noise_std_counts(signal_counts: np.ndarray | float, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return shot-noise standard deviation in camera counts."""
    cfg = resolve_camera_noise_config(params)
    counts = np.asarray(signal_counts, dtype=float)
    if not cfg.shot_noise_enabled:
        return np.zeros_like(counts, dtype=float)
    counts_pos = np.where(np.isfinite(counts) & (counts > 0.0), counts, 0.0)
    return np.sqrt(counts_pos / cfg.camera_gain_e_per_count)


def total_noise_std_counts(signal_counts: np.ndarray | float, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return combined shot and read-noise standard deviation in camera counts."""
    cfg = resolve_camera_noise_config(params)
    shot = shot_noise_std_counts(signal_counts, params)
    read = cfg.read_noise_counts if cfg.gaussian_noise_enabled else 0.0
    return np.sqrt(shot * shot + read * read)


def total_noise_variance_counts(signal_counts: np.ndarray | float, params: dict[str, Any] | None = None) -> np.ndarray:
    """Return combined shot and read-noise variance in camera-count units."""
    std = total_noise_std_counts(signal_counts, params)
    return std * std


def contrast_noise_variance_counts(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
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
    method = str(params.get("background_subtraction_method", "video_median")).strip().lower()
    raw_methods = {"none", "raw", "raw_signal", "off", "disabled", "no_subtraction"}
    signal = np.asarray(signal_counts, dtype=float)
    var_signal = total_noise_variance_counts(signal, params)
    if method in raw_methods or reference_counts is None:
        return np.maximum(var_signal, float(variance_floor))

    reference = np.asarray(reference_counts, dtype=float)
    if signal.shape != reference.shape:
        raise ValueError(
            "signal_counts and reference_counts must have the same shape for "
            f"contrast-noise propagation; got {signal.shape} and {reference.shape}."
        )
    var_reference = total_noise_variance_counts(reference, params)

    if relative_reference is None:
        imaging_model_name = str(params.get("imaging_model", "bright_field")).strip().lower()
        from imaging_model import modality_uses_relative_reference_contrast

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
    imaging_model_name = str(params.get("imaging_model", "bright_field")).strip().lower()
    from imaging_model import get_imaging_model_class

    output_type = getattr(get_imaging_model_class(imaging_model_name), "output_type", "intensity")
    if output_type == "phase":
        phase_noise = params.get("qpi_phase_noise_std_rad", None)
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
        phase_to_count = float(
            params.get(
                "qpi_phase_to_count_scale",
                params.get("background_intensity", 100.0),
            )
        )
        if not np.isfinite(phase_to_count) or phase_to_count <= 0.0:
            raise ValueError(
                "qpi_phase_to_count_scale must be positive and finite for "
                "phase-domain noise propagation."
            )
        count_variance = contrast_noise_variance_counts(
            signal_counts,
            reference_counts,
            params,
            relative_reference=False,
            variance_floor=variance_floor,
        )
        return np.maximum(count_variance / (phase_to_count * phase_to_count), float(variance_floor))

    return contrast_noise_variance_counts(
        signal_counts,
        reference_counts,
        params,
        relative_reference=relative_reference,
        variance_floor=variance_floor,
    )


def estimate_detector_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Return detector-noise standard deviation in camera counts/ADU.

    This is the counts-domain helper expected by supervision and audit code.
    It delegates to the canonical total_noise_std_counts implementation so
    rendering, supervision, and metadata use the same camera-noise model.
    """
    return total_noise_std_counts(signal_counts, params)


def _seed_component(value: float) -> int:
    return int(round(abs(float(value)) * 1_000_000_000.0)) % (2**32)


def _detector_static_noise_seed(params: dict[str, Any] | None) -> int:
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
        seed = int(np.random.randint(0, 2**32 - 1))
    params[internal_key] = seed
    return seed


def _static_detector_maps(
    shape: tuple[int, ...],
    cfg: CameraNoiseConfig,
    params: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    seed = _detector_static_noise_seed(params)
    key = (
        seed,
        tuple(int(x) for x in shape),
        _seed_component(cfg.fixed_pattern_gain_std),
        _seed_component(cfg.fixed_pattern_offset_counts),
        _seed_component(cfg.hot_pixel_fraction),
    )
    cached = _STATIC_DETECTOR_MAP_CACHE.get(key)
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

    _STATIC_DETECTOR_MAP_CACHE[key] = maps
    return maps



def apply_camera_noise_counts(
    frame_counts: np.ndarray,
    params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply camera noise to a frame already expressed in camera counts/ADU."""
    cfg = resolve_camera_noise_config(params)
    rng = rng if rng is not None else _GLOBAL_NUMPY_RNG

    clean = np.asarray(frame_counts, dtype=float)
    noisy = clean.copy()
    static_maps: dict[str, np.ndarray] = {}
    if (
        cfg.fixed_pattern_gain_std > 0.0
        or cfg.fixed_pattern_offset_counts > 0.0
        or cfg.hot_pixel_fraction > 0.0
    ):
        static_maps = _static_detector_maps(clean.shape, cfg, params)

    if cfg.dark_offset_counts:
        noisy = noisy + cfg.dark_offset_counts

    if cfg.shot_noise_enabled:
        counts_pos = np.where(np.isfinite(noisy) & (noisy > 0.0), noisy, 0.0)
        electron_mean = counts_pos * cfg.camera_gain_e_per_count
        electron_sample = rng.poisson(electron_mean).astype(float)
        poisson_counts = electron_sample / cfg.camera_gain_e_per_count
        shot_residual = poisson_counts - counts_pos
        noisy = noisy + shot_residual

    if cfg.fixed_pattern_gain_std > 0.0:
        noisy = noisy * static_maps["gain"]

    if cfg.fixed_pattern_offset_counts > 0.0:
        noisy = noisy + static_maps["offset"]

    if cfg.hot_pixel_fraction > 0.0:
        hot_mask = static_maps["hot_mask"]
        if cfg.hot_pixel_value_counts is None:
            finite = noisy[np.isfinite(noisy)]
            hot_value = float(finite.max()) if finite.size else 0.0
        else:
            hot_value = cfg.hot_pixel_value_counts
        noisy[hot_mask] = hot_value

    if cfg.scan_line_noise_counts > 0.0:
        noisy = noisy + rng.normal(scale=cfg.scan_line_noise_counts, size=(clean.shape[0], 1))

    if cfg.gaussian_noise_enabled and cfg.read_noise_counts > 0.0:
        noisy = noisy + rng.normal(scale=cfg.read_noise_counts, size=clean.shape)

    if cfg.clip_output_to_nonnegative:
        noisy = np.where(np.isfinite(noisy), np.maximum(noisy, 0.0), 0.0)

    return noisy.astype(float, copy=False)



def estimate_contrast_noise_std_from_params(params: dict[str, Any]) -> float:
    """
    Estimate noise in contrast units for supervision support gates.

    For count images, contrast noise is noise_counts / reference_counts. This is
    deliberately tied to the same camera-count model used by rendering.
    """
    imaging_model = str(params.get("imaging_model", "bright_field")).strip().lower()

    if imaging_model in {"dark_field", "coherent_dark_field"}:
        background_counts = max(float(params.get("dark_field_background_count", 1.0)), 1e-9)
        normalization_counts = max(float(params.get("dark_field_illumination_count", background_counts)), 1e-9)
    else:
        background_counts = max(float(params.get("background_intensity", 1.0)), 1e-9)
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
    except Exception as exc:  # pragma: no cover
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
