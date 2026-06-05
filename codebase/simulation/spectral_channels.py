from __future__ import annotations

import os
from copy import deepcopy

import numpy as np

from camera_noise import DetectorNoiseRuntime, apply_camera_noise_counts
from config import normalize_params
from config.runtime import internal_param_value, param_value, resolved_modality, resolved_random_seed
from imaging_models import get_imaging_model
from modality_registry import is_fluorescence_modality
from postprocessing import (
    apply_background_subtraction,
    normalize_raw_camera_frames,
    save_video,
)

_VISIBLE_WAVELENGTH_MIN_NM = 380.0
_VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM = 420.0
_VISIBLE_WAVELENGTH_VIOLET_BLUE_NM = 440.0
_VISIBLE_WAVELENGTH_BLUE_GREEN_NM = 490.0
_VISIBLE_WAVELENGTH_GREEN_CYAN_NM = 510.0
_VISIBLE_WAVELENGTH_YELLOW_RED_NM = 580.0
_VISIBLE_WAVELENGTH_RED_EDGE_NM = 645.0
_VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM = 700.0
_VISIBLE_WAVELENGTH_MAX_NM = 780.0
_WAVELENGTH_RGB_EDGE_FACTOR = 0.3
_WAVELENGTH_RGB_GAMMA = 0.8

from .latent_scene import _simulate_latent_scene
from .output import (
    _RUNTIME_PARAM_KEYS,
    _multichannel_output_mode,
    _raw_signal_video_filename,
    _safe_channel_filename,
    _simulation_result,
)
from .scene_render import _render_scene_with_params

def _channel_spec_to_params(base_params: dict, channel, channel_index: int) -> tuple[str, dict, np.ndarray, float]:
    """
    Resolve one spectral sample.

    Returns:
        channel_name:
            Human-readable channel label.
        channel_params:
            PARAMS clone with wavelength/probe overrides for this spectral sample.
        detector_weights_rgb:
            Length-3 detector response weights [R, G, B] for this sample.
        spectral_weight:
            Scalar spectral source/integration weight for this sample.
    """
    channel_params = deepcopy(base_params)
    channel_params.pop("channels", None)
    explicit_keys: set[str] = set(channel) if isinstance(channel, dict) else set()

    if isinstance(channel, dict):
        channel_name = str(channel.get("name", f"ch{channel_index + 1}"))
        if "imaging_model" in channel:
            requested_model = str(channel["imaging_model"]).strip()
            if requested_model and resolved_modality({"imaging_model": requested_model}) != resolved_modality(base_params):
                raise ValueError(
                    "Spectral channel entries may not override PARAMS['imaging_model']; "
                    "use matched_modalities for multi-modality packets instead."
                )
        channel_params.update(
            {k: v for k, v in channel.items()
             if k not in {"name", "rgb", "detector_weights_rgb", "detector_weights", "weight", "spectral_weight"}}
        )
        wavelength_nm = float(param_value(channel_params, "wavelength_nm"))
        spectral_weight = float(channel.get("spectral_weight", channel.get("weight", 1.0)))
        weights = None
        for weights_key in ("detector_weights_rgb", "detector_weights", "rgb"):
            if weights_key in channel and channel[weights_key] is not None:
                weights = channel[weights_key]
                break
        if weights is None:
            detector_weights_rgb = _wavelength_to_rgb_weights(wavelength_nm)
        else:
            detector_weights_rgb = np.asarray(weights, dtype=float)
    else:
        wavelength_nm = float(channel)
        channel_name = f"{wavelength_nm:.0f}nm"
        channel_params["wavelength_nm"] = wavelength_nm
        channel_params["probe_wavelength_nm"] = wavelength_nm
        spectral_weight = 1.0
        detector_weights_rgb = _wavelength_to_rgb_weights(wavelength_nm)

    if not np.isfinite(spectral_weight) or spectral_weight < 0.0:
        raise ValueError(
            "Each channel spectral_weight/weight must be finite and non-negative; "
            f"got {spectral_weight!r} for channel {channel_name!r}."
        )

    if detector_weights_rgb.shape != (3,):
        raise ValueError(
            "Each channel detector weight must be length 3, ordered [R, G, B]. "
            f"Got shape {detector_weights_rgb.shape} for channel {channel_name!r}."
        )
    if not np.all(np.isfinite(detector_weights_rgb)):
        raise ValueError(
            "Each channel detector weight must contain only finite values; "
            f"got {detector_weights_rgb!r} for channel {channel_name!r}."
        )
    if np.any(detector_weights_rgb < 0.0):
        raise ValueError(
            "Each channel detector weight must be non-negative; "
            f"got {detector_weights_rgb!r} for channel {channel_name!r}."
        )

    channel_params["wavelength_nm"] = wavelength_nm
    probe_wavelength_nm = param_value(channel_params, 'probe_wavelength_nm')
    if probe_wavelength_nm is None:
        probe_wavelength_nm = wavelength_nm
    channel_params["probe_wavelength_nm"] = float(probe_wavelength_nm)
    modality = resolved_modality(channel_params)
    if (
        is_fluorescence_modality(modality)
        and "fluorescence_emission_wavelength_nm" not in explicit_keys
    ):
        channel_params["fluorescence_emission_wavelength_nm"] = float(wavelength_nm)
    if modality == "tirf_fluorescence" and "fluorescence_excitation_wavelength_nm" not in explicit_keys:
        channel_params["fluorescence_excitation_wavelength_nm"] = float(wavelength_nm)
    if modality == "ricm" and "ricm_wavelength_nm" not in explicit_keys:
        channel_params["ricm_wavelength_nm"] = float(wavelength_nm)
    channel_params = normalize_params(channel_params, allowed_internal_keys=_RUNTIME_PARAM_KEYS)
    return channel_name, channel_params, detector_weights_rgb.astype(float), spectral_weight


def _wavelength_to_rgb_weights(wavelength_nm: float) -> np.ndarray:
    """
    Approximate visible-wavelength display/detector response.

    This piecewise display approximation is deterministic and preserves
    visible-wavelength ordering for RGB sidecars. Scientific spectral work
    should pass explicit
    detector_weights_rgb and spectral_weight per sample.
    """
    wl = float(wavelength_nm)
    if wl < _VISIBLE_WAVELENGTH_MIN_NM or wl > _VISIBLE_WAVELENGTH_MAX_NM:
        return np.zeros(3, dtype=float)

    if wl < _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM:
        r = -(
            wl - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM
        ) / (_VISIBLE_WAVELENGTH_VIOLET_BLUE_NM - _VISIBLE_WAVELENGTH_MIN_NM)
        g = 0.0
        b = 1.0
    elif wl < _VISIBLE_WAVELENGTH_BLUE_GREEN_NM:
        r = 0.0
        g = (
            wl - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM
        ) / (_VISIBLE_WAVELENGTH_BLUE_GREEN_NM - _VISIBLE_WAVELENGTH_VIOLET_BLUE_NM)
        b = 1.0
    elif wl < _VISIBLE_WAVELENGTH_GREEN_CYAN_NM:
        r = 0.0
        g = 1.0
        b = -(
            wl - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM
        ) / (_VISIBLE_WAVELENGTH_GREEN_CYAN_NM - _VISIBLE_WAVELENGTH_BLUE_GREEN_NM)
    elif wl < _VISIBLE_WAVELENGTH_YELLOW_RED_NM:
        r = (
            wl - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM
        ) / (_VISIBLE_WAVELENGTH_YELLOW_RED_NM - _VISIBLE_WAVELENGTH_GREEN_CYAN_NM)
        g = 1.0
        b = 0.0
    elif wl < _VISIBLE_WAVELENGTH_RED_EDGE_NM:
        r = 1.0
        g = -(
            wl - _VISIBLE_WAVELENGTH_RED_EDGE_NM
        ) / (_VISIBLE_WAVELENGTH_RED_EDGE_NM - _VISIBLE_WAVELENGTH_YELLOW_RED_NM)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if wl < _VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM:
        factor = _WAVELENGTH_RGB_EDGE_FACTOR + (
            1.0 - _WAVELENGTH_RGB_EDGE_FACTOR
        ) * (wl - _VISIBLE_WAVELENGTH_MIN_NM) / (
            _VISIBLE_WAVELENGTH_FULL_INTENSITY_MIN_NM - _VISIBLE_WAVELENGTH_MIN_NM
        )
    elif wl <= _VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM:
        factor = 1.0
    else:
        factor = _WAVELENGTH_RGB_EDGE_FACTOR + (
            1.0 - _WAVELENGTH_RGB_EDGE_FACTOR
        ) * (_VISIBLE_WAVELENGTH_MAX_NM - wl) / (
            _VISIBLE_WAVELENGTH_MAX_NM - _VISIBLE_WAVELENGTH_FULL_INTENSITY_MAX_NM
        )

    return np.asarray([
        (max(r, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
        (max(g, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
        (max(b, 0.0) * factor) ** _WAVELENGTH_RGB_GAMMA,
    ], dtype=float)


def _disable_detector_noise_for_spectral_component(params: dict) -> dict:
    """
    Return a params clone for deterministic spectral rendering.

    For broadband/RGB rendering, noise belongs after spectral integration into
    detector channels. Rendering each wavelength with independent shot/readout
    noise and then summing channels is not the correct measurement model.

    This disables all noise by setting the canonical camera_noise.py toggles
    and per-pixel artefact controls to their "off" states.
    """
    p = deepcopy(params)
    # Disable via canonical camera_noise.py parameter names.
    p["shot_noise_enabled"] = False
    p["gaussian_noise_enabled"] = False
    p["fixed_pattern_gain_std"] = 0.0
    p["fixed_pattern_offset_counts"] = 0.0
    p["hot_pixel_fraction"] = 0.0
    p["scan_line_noise_counts"] = 0.0
    p["dark_offset_counts"] = 0.0
    # Also clear override containers so no per-modality detector noise can be
    # reintroduced before the post-integration RGB noise pass.
    noise_model = dict(p.get("noise_model", {}) or {})
    noise_model.update(
        shot_noise_enabled=False,
        gaussian_noise_enabled=False,
        fixed_pattern_gain_std=0.0,
        fixed_pattern_offset_counts=0.0,
        hot_pixel_fraction=0.0,
        scan_line_noise_counts=0.0,
        dark_offset_counts=0.0,
    )
    p["noise_model"] = noise_model
    p["modality_noise"] = {}
    return p


def _save_rgb_video(path: str, frames_rgb: list[np.ndarray], fps: float) -> None:
    """Save RGB frames through the canonical video writer."""
    if not frames_rgb:
        return
    save_video(path, frames_rgb, fps, color_order="rgb")


def _save_channel_videos(params: dict, spectral_items: list[dict], fps: float) -> list[str]:
    """
    Save per-channel grayscale sidecar videos.

    These sidecars preserve individual channel renderings. They are separate
    from the RGB visualization composite and are only written when
    ``multichannel_output_mode`` is ``"channels"`` or ``"both"``.
    """
    sidecar_dir = param_value(params, 'multichannel_sidecar_directory')
    if not sidecar_dir:
        stem, _ = os.path.splitext(params["output_filename"])
        sidecar_dir = stem + "_channels"

    os.makedirs(sidecar_dir, exist_ok=True)

    written = []
    used_names: set[str] = set()
    for channel_index, item in enumerate(spectral_items):
        base_name = _safe_channel_filename(item.get("name", "channel"))
        name = base_name
        if name in used_names:
            suffix = f"_ch{channel_index + 1}"
            name = f"{base_name}{suffix}"
            suffix_count = 2
            while name in used_names:
                name = f"{base_name}{suffix}_{suffix_count}"
                suffix_count += 1
        used_names.add(name)
        frame_meta = _channel_result_metadata(item)
        frames = list(
            frame_meta.get("background_subtracted_frames", [])
            or []
        )
        if not frames:
            continue

        first = np.asarray(frames[0])
        if first.ndim != 2:
            raise ValueError(
                "Channel sidecar frames must be grayscale arrays with shape (H, W); "
                f"got {first.shape} for channel {name!r}."
            )

        h, w = first.shape
        out_path = os.path.join(sidecar_dir, f"{name}.avi")
        save_video(out_path, frames, fps, (w, h))
        written.append(out_path)

    return written


def _channel_result_metadata(item: dict) -> dict:
    result = item.get("frames", {})
    if not isinstance(result, dict):
        raise TypeError(
            "Each spectral channel item must store the _simulation_result dict "
            f"under item['frames']; got {type(result).__name__}."
        )
    return dict(result.get("metadata", {}) or {})


def _apply_detector_noise_to_rgb_raw_frames(
    frames_rgb_float: list[np.ndarray],
    params: dict,
    *,
    detector_noise_runtime: DetectorNoiseRuntime | None = None,
) -> list[np.ndarray]:
    if detector_noise_runtime is None:
        rng_seed = param_value(params, 'random_seed')
        detector_noise_runtime = DetectorNoiseRuntime(
            rng=np.random.default_rng(None if rng_seed is None else int(rng_seed))
        )
    active_runtime = detector_noise_runtime
    out = []
    for frame_rgb in frames_rgb_float:
        frame_rgb = np.asarray(frame_rgb, dtype=float)
        noisy_channels = []
        for c in range(3):
            noisy_channels.append(
                apply_camera_noise_counts(
                    frame_rgb[:, :, c],
                    params,
                    runtime=active_runtime,
                )
            )
        out.append(np.stack(noisy_channels, axis=-1))
    return out


def _background_subtract_rgb(
    signal_rgb_frames: list[np.ndarray],
    reference_rgb_frames: list[np.ndarray],
    params: dict,
) -> list[np.ndarray]:
    final_channels = []
    for c in range(3):
        sig_c = [np.asarray(f, dtype=float)[:, :, c] for f in signal_rgb_frames]
        ref_c = [np.asarray(f, dtype=float)[:, :, c] for f in reference_rgb_frames]
        final_c = apply_background_subtraction(sig_c, ref_c, params)
        final_channels.append([
            np.clip(np.asarray(x, dtype=float), 0.0, 255.0).astype(np.uint8)
            for x in final_c
        ])

    n_frames = min(len(final_channels[0]), len(final_channels[1]), len(final_channels[2]))
    return [
        np.stack(
            [final_channels[0][i], final_channels[1][i], final_channels[2][i]],
            axis=-1,
        ).astype(np.uint8)
        for i in range(n_frames)
    ]


def _validate_multichannel_frame_arrays(
    raw_signal_arrays: list[np.ndarray],
    raw_reference_arrays: list[np.ndarray],
) -> tuple[int, int, int]:
    if not raw_signal_arrays:
        raise ValueError("Multichannel rendering produced no channel frame arrays.")
    if len(raw_signal_arrays) != len(raw_reference_arrays):
        raise ValueError(
            "Multichannel signal/reference channel counts differ: "
            f"{len(raw_signal_arrays)} signal vs {len(raw_reference_arrays)} reference."
        )

    expected_shape: tuple[int, int, int] | None = None
    for channel_index, (sig_arr, ref_arr) in enumerate(zip(raw_signal_arrays, raw_reference_arrays)):
        if sig_arr.size == 0 or ref_arr.size == 0:
            raise ValueError(
                f"Multichannel channel {channel_index} produced empty signal/reference frames."
            )
        if sig_arr.ndim != 3 or ref_arr.ndim != 3:
            raise ValueError(
                "Multichannel channel arrays must have shape (T, H, W); "
                f"channel {channel_index} has signal {sig_arr.shape} and reference {ref_arr.shape}."
            )
        if sig_arr.shape != ref_arr.shape:
            raise ValueError(
                f"Multichannel channel {channel_index} signal/reference shapes differ: "
                f"{sig_arr.shape} vs {ref_arr.shape}."
            )
        if expected_shape is None:
            expected_shape = tuple(int(v) for v in sig_arr.shape)
        elif sig_arr.shape != expected_shape:
            raise ValueError(
                "All multichannel channel arrays must have the same shape; "
                f"channel 0 has {expected_shape}, channel {channel_index} has {sig_arr.shape}."
            )
        if not np.all(np.isfinite(sig_arr)) or not np.all(np.isfinite(ref_arr)):
            raise ValueError(
                f"Multichannel channel {channel_index} contains non-finite signal/reference values."
            )

    assert expected_shape is not None
    return expected_shape


def _run_multichannel_simulation(
    params: dict,
    channels,
    *,
    return_frames: bool = False,
):
    """
    Render one latent scene through multiple spectral samples and integrate into RGB.

    Important invariant:
        trajectories, orientations, particle identities, masks, and sample geometry
        are simulated once. Each spectral sample rebuilds wavelength-dependent
        materials/PSFs against that same latent scene. Detector noise is applied
        only after RGB channel integration.
    """
    if not isinstance(channels, (list, tuple)) or len(channels) == 0:
        raise ValueError("PARAMS['channels'] must be a non-empty list when set.")
    base_model = get_imaging_model(params)
    if not bool(getattr(base_model, "supports_spectral_channels", True)):
        raise ValueError(
            f"PARAMS['imaging_model']={param_value(params, 'imaging_model')!r} does not support "
            "PARAMS['channels']; use matched_modalities or a modality-specific loop instead."
        )

    latent_scene = _simulate_latent_scene(params)

    spectral_items = []
    for channel_index, channel in enumerate(channels):
        channel_name, channel_params, detector_weights_rgb, spectral_weight = _channel_spec_to_params(
            params,
            channel,
            channel_index,
        )
        deterministic_params = _disable_detector_noise_for_spectral_component(channel_params)
        deterministic_params["return_ideal_float_frames"] = True
        deterministic_params["mask_generation_enabled"] = bool(channel_index == 0 and param_value(params, 'mask_generation_enabled'))

        frames = _render_scene_with_params(
            deterministic_params,
            latent_scene,
            save_video_output=False,
            return_frames=True,
        ) or {}

        spectral_items.append(
            {
                "name": channel_name,
                "params": channel_params,
                "detector_weights_rgb": detector_weights_rgb,
                "spectral_weight": float(spectral_weight),
                "frames": frames,
            }
        )

    ideal_signal_arrays = [
        np.asarray(_channel_result_metadata(item).get("ideal_signal_frames", []), dtype=float)
        for item in spectral_items
    ]
    ideal_reference_arrays = [
        np.asarray(_channel_result_metadata(item).get("ideal_reference_frames", []), dtype=float)
        for item in spectral_items
    ]

    n_frames, h, w = _validate_multichannel_frame_arrays(
        ideal_signal_arrays,
        ideal_reference_arrays,
    )

    signal_rgb_float = []
    reference_rgb_float = []
    for t in range(n_frames):
        sig_rgb = np.zeros((h, w, 3), dtype=float)
        ref_rgb = np.zeros((h, w, 3), dtype=float)
        for item, sig_arr, ref_arr in zip(spectral_items, ideal_signal_arrays, ideal_reference_arrays):
            weights = item["detector_weights_rgb"] * item["spectral_weight"]
            for c in range(3):
                sig_rgb[:, :, c] += sig_arr[t] * weights[c]
                ref_rgb[:, :, c] += ref_arr[t] * weights[c]
        signal_rgb_float.append(sig_rgb)
        reference_rgb_float.append(ref_rgb)

    rng_seed = resolved_random_seed(params)
    detector_noise_runtime = DetectorNoiseRuntime(
        rng=np.random.default_rng(None if rng_seed is None else int(rng_seed))
    )
    signal_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(
        signal_rgb_float,
        params,
        detector_noise_runtime=detector_noise_runtime,
    )
    reference_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(
        reference_rgb_float,
        params,
        detector_noise_runtime=detector_noise_runtime,
    )

    final_rgb = _background_subtract_rgb(signal_rgb_noisy, reference_rgb_noisy, params)
    output_mode = _multichannel_output_mode(params)
    written_channel_sidecars = []
    raw_signal_video_path = None
    if output_mode in {"rgb", "both"}:
        _save_rgb_video(params["output_filename"], final_rgb, float(params["fps"]))
        if bool(param_value(params, "save_raw_camera_video")):
            raw_signal_video_path = _raw_signal_video_filename(params)
            raw_rgb_preview = normalize_raw_camera_frames(signal_rgb_noisy, params)
            _save_rgb_video(raw_signal_video_path, raw_rgb_preview, float(params["fps"]))
    if output_mode in {"channels", "both"}:
        written_channel_sidecars = _save_channel_videos(
            params,
            spectral_items,
            float(params["fps"]),
        )


    if return_frames:
        return _simulation_result(final_rgb, ["red", "green", "blue"], {
            "spectral_channels": [item["name"] for item in spectral_items],
            "spectral_items": spectral_items,
            "spectral_integration_model": str(
                param_value(params, "spectral_integration_model")
            ),
            "generated_spectral_channels": bool(internal_param_value(params, "_generated_spectral_channels")),
            "spectral_channel_count": int(len(spectral_items)),
            "ideal_signal_frames_by_spectral_sample": ideal_signal_arrays,
            "ideal_reference_frames_by_spectral_sample": ideal_reference_arrays,
            "raw_signal_frames_rgb": signal_rgb_noisy,
            "raw_reference_frames_rgb": reference_rgb_noisy,
            "background_subtracted_frames_rgb": final_rgb,
            "analysis_video_path": str(params["output_filename"]) if output_mode in {"rgb", "both"} else None,
            "raw_signal_video_path": raw_signal_video_path,
            "analysis_video_semantics": "background_subtracted_contrast_normalized_rgb_uint8",
            "raw_signal_video_semantics": "windowed_raw_detector_count_preview_rgb_uint8",
            "multichannel_output_mode": output_mode,
            "channel_sidecar_videos": written_channel_sidecars,
        })

    return None


__all__ = [
    "_apply_detector_noise_to_rgb_raw_frames",
    "_background_subtract_rgb",
    "_channel_result_metadata",
    "_channel_spec_to_params",
    "_disable_detector_noise_for_spectral_component",
    "_run_multichannel_simulation",
    "_safe_channel_filename",
    "_save_channel_videos",
    "_save_rgb_video",
    "_validate_multichannel_frame_arrays",
    "_wavelength_to_rgb_weights",
]
