from __future__ import annotations
from configured_parameters import configured_assign

import os
from copy import deepcopy

import numpy as np

from camera_noise import (
    DetectorNoiseRuntime,
    apply_camera_noise_counts,
    noise_model_overrides_from_params,
)
from array_representation import STAGE_DISPLAY
from config.runtime import (
    AcquisitionProfile,
    FluorescenceSettings,
    MaskGenerationSettings,
    ModalitySettings,
    OpticalInstrumentSettings,
    QpiReadoutSettings,
    RicmSettings,
    SimulationOutputSettings,
    SpectralIntegrationSettings,
    TirfSettings,
)
from imaging_models import get_imaging_model
from modality_registry import is_fluorescence_modality
from postprocessing import (
    compute_contrast_frames,
    normalize_raw_camera_frames,
    save_video,
)
from simulation_runtime_state import runtime_state
from stochastic_runtime import rng_from_seed

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
_SPECTRAL_CHANNEL_METADATA_KEYS = frozenset(
    {
        "name",
        "rgb",
        "detector_weights_rgb",
        "detector_weights",
        "weight",
        "spectral_weight",
    }
)
_SPECTRAL_CHANNEL_PARAM_KEYS = frozenset(
    {
        "wavelength_nm",
        "probe_wavelength_nm",
        "fluorescence_excitation_wavelength_nm",
        "fluorescence_emission_wavelength_nm",
        "ricm_wavelength_nm",
        "qpi_detected_quanta_per_pixel",
        "qpi_phase_to_count_scale",
    }
)

from .latent_scene import _simulate_latent_scene
from .output import (
    _multichannel_output_mode,
    _raw_signal_video_filename,
    _safe_channel_filename,
    _simulation_result,
)
from .scene_render import _render_scene_with_params


def _force_detected_count_detector_domain(params: dict) -> None:
    configured_assign(params, 'detector_input_is_incident_quanta', False)
    noise_model = noise_model_overrides_from_params(params)
    noise_model["detector_input_is_incident_quanta"] = False
    configured_assign(params, 'noise_model', noise_model)


def _model_owns_rgb_noise(params: dict) -> bool:
    model = get_imaging_model(params)
    return getattr(model, "output_type", "intensity") == "phase"


def _validate_spectral_channel_keys(channel: dict, channel_index: int) -> set[str]:
    keys = {str(key) for key in channel}
    forbidden = sorted(keys - _SPECTRAL_CHANNEL_METADATA_KEYS - _SPECTRAL_CHANNEL_PARAM_KEYS)
    if forbidden:
        raise ValueError(
            "Spectral channel entries are spectral samples, not parameters overlays. "
            f"channels[{channel_index}] contains unsupported key(s) {forbidden}; "
            f"allowed parameter keys are {sorted(_SPECTRAL_CHANNEL_PARAM_KEYS)}."
        )
    return keys


def _validate_spectral_channel_runtime(channel_params: dict, *, channel_name: str) -> None:
    modality = ModalitySettings.from_params(channel_params).modality
    OpticalInstrumentSettings.from_params(channel_params)
    QpiReadoutSettings.from_params(channel_params)
    if modality == "ricm":
        RicmSettings.from_params(channel_params)
    if is_fluorescence_modality(modality):
        if modality == "tirf_fluorescence":
            TirfSettings.from_params(channel_params)
        else:
            FluorescenceSettings.from_params(channel_params)
    if ModalitySettings.from_params(channel_params).modality != modality:
        raise RuntimeError(
            f"Spectral channel {channel_name!r} changed modality during validation."
        )


def _channel_spec_to_params(base_params: dict, channel, channel_index: int) -> tuple[str, dict, np.ndarray, float]:
    """
    Resolve one spectral sample.

    Returns:
        channel_name:
            Human-readable channel label.
        channel_params:
            parameters clone with wavelength/probe overrides for this spectral sample.
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
        explicit_keys = _validate_spectral_channel_keys(channel, channel_index)
        channel_params.update(
            {k: v for k, v in channel.items()
             if str(k) in _SPECTRAL_CHANNEL_PARAM_KEYS}
        )
        wavelength_nm = OpticalInstrumentSettings.from_params(channel_params).wavelength_nm
        if "wavelength_nm" in explicit_keys and "probe_wavelength_nm" not in explicit_keys:
            configured_assign(channel_params, 'probe_wavelength_nm', float(wavelength_nm))
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
        configured_assign(channel_params, 'wavelength_nm', wavelength_nm)
        configured_assign(channel_params, 'probe_wavelength_nm', wavelength_nm)
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

    configured_assign(channel_params, 'wavelength_nm', wavelength_nm)
    instrument = OpticalInstrumentSettings.from_params(channel_params)
    configured_assign(channel_params, 'probe_wavelength_nm', float(
        instrument.probe_wavelength_nm
        if instrument.probe_wavelength_nm_is_explicit
        else wavelength_nm
    ))
    modality = ModalitySettings.from_params(channel_params).modality
    if (
        is_fluorescence_modality(modality)
        and "fluorescence_emission_wavelength_nm" not in explicit_keys
    ):
        configured_assign(channel_params, 'fluorescence_emission_wavelength_nm', float(wavelength_nm))
    if modality == "ricm" and "ricm_wavelength_nm" not in explicit_keys:
        configured_assign(channel_params, 'ricm_wavelength_nm', float(wavelength_nm))
    _validate_spectral_channel_runtime(channel_params, channel_name=channel_name)
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
    configured_assign(p, "shot_noise_enabled", False)
    configured_assign(p, "gaussian_noise_enabled", False)
    configured_assign(p, "fixed_pattern_gain_std", 0.0)
    configured_assign(p, "fixed_pattern_offset_counts", 0.0)
    configured_assign(p, "hot_pixel_fraction", 0.0)
    configured_assign(p, "scan_line_noise_counts", 0.0)
    configured_assign(p, "dark_offset_counts", 0.0)
    # Also clear override containers so no per-modality detector noise can be
    # reintroduced before the post-integration RGB noise pass.
    noise_model = noise_model_overrides_from_params(p)
    noise_model.update(
        shot_noise_enabled=False,
        gaussian_noise_enabled=False,
        fixed_pattern_gain_std=0.0,
        fixed_pattern_offset_counts=0.0,
        hot_pixel_fraction=0.0,
        scan_line_noise_counts=0.0,
        dark_offset_counts=0.0,
    )
    configured_assign(p, "noise_model", noise_model)
    configured_assign(p, "modality_noise", {})
    if is_fluorescence_modality(ModalitySettings.from_params(p).modality):
        _force_detected_count_detector_domain(p)
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
    sidecar_dir = SimulationOutputSettings.from_params(params).multichannel_sidecar_output_directory

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
    active_rgb_channels: np.ndarray | None = None,
    rgb_noise_params: dict | None = None,
    qpi_quanta_scale_rgb: np.ndarray | None = None,
    qpi_phase_to_count_scale_rgb: np.ndarray | None = None,
) -> list[np.ndarray]:
    noise_params = (
        rgb_noise_params
        if rgb_noise_params is not None
        else _rgb_noise_params_with_effective_exposure(params)
    )
    if detector_noise_runtime is None:
        rng_seed = AcquisitionProfile.from_params(noise_params).random_seed
        detector_noise_runtime = DetectorNoiseRuntime(
            rng=rng_from_seed(
                None if rng_seed is None else int(rng_seed),
                stream="spectral_channel_detector_noise",
            )
        )
    active_runtime = detector_noise_runtime
    model = get_imaging_model(noise_params)
    use_model_noise = _model_owns_rgb_noise(noise_params)
    if active_rgb_channels is None:
        active_channels = np.ones(3, dtype=bool)
    else:
        active_channels = np.asarray(active_rgb_channels, dtype=bool)
        if active_channels.shape != (3,):
            raise ValueError(
                "active_rgb_channels must have shape (3,), ordered [R, G, B]; "
                f"got {active_channels.shape}."
            )
    if qpi_quanta_scale_rgb is None:
        qpi_scales = None
    else:
        qpi_scales = np.asarray(qpi_quanta_scale_rgb, dtype=float)
        if qpi_scales.shape != (3,):
            raise ValueError(
                "qpi_quanta_scale_rgb must have shape (3,), ordered [R, G, B]; "
                f"got {qpi_scales.shape}."
            )
        if not np.all(np.isfinite(qpi_scales)) or np.any(qpi_scales < 0.0):
            raise ValueError(
                "qpi_quanta_scale_rgb must contain finite non-negative RGB scales."
            )
    if qpi_phase_to_count_scale_rgb is None:
        qpi_display_scales = None
    else:
        qpi_display_scales = np.asarray(qpi_phase_to_count_scale_rgb, dtype=float)
        if qpi_display_scales.shape != (3,):
            raise ValueError(
                "qpi_phase_to_count_scale_rgb must have shape (3,), ordered [R, G, B]; "
                f"got {qpi_display_scales.shape}."
            )
        if not np.all(np.isfinite(qpi_display_scales)) or np.any(qpi_display_scales < 0.0):
            raise ValueError(
                "qpi_phase_to_count_scale_rgb must contain finite non-negative RGB scales."
            )
    out = []
    for frame_rgb in frames_rgb_float:
        frame_rgb = np.asarray(frame_rgb, dtype=float)
        noisy_channels = []
        for c in range(3):
            plane = frame_rgb[:, :, c]
            if use_model_noise:
                if not active_channels[c]:
                    noisy_channels.append(np.zeros_like(plane, dtype=float))
                else:
                    plane_noise_params = noise_params
                    if (
                        (qpi_scales is not None and qpi_scales[c] > 0.0)
                        or qpi_display_scales is not None
                    ):
                        plane_noise_params = dict(noise_params)
                    if qpi_scales is not None and qpi_scales[c] > 0.0:
                        detected_quanta_raw = (
                            QpiReadoutSettings.from_params(
                                plane_noise_params
                            ).configured_detected_quanta_per_pixel
                        )
                        configured_assign(plane_noise_params, 'qpi_detected_quanta_per_pixel', float(detected_quanta_raw) * float(qpi_scales[c]))
                    if qpi_display_scales is not None:
                        configured_assign(plane_noise_params, 'qpi_phase_to_count_scale', float(
                            qpi_display_scales[c]
                        ))
                    noisy_channels.append(
                        model.compute_noise(
                            plane,
                            plane_noise_params,
                            rng=active_runtime.rng,
                            detector_noise_runtime=active_runtime,
                        )
                    )
            else:
                noisy_channels.append(
                    apply_camera_noise_counts(
                        plane,
                        noise_params,
                        runtime=active_runtime,
                    )
                )
        out.append(np.stack(noisy_channels, axis=-1))
    return out


def _rgb_noise_params_with_effective_exposure(params: dict) -> dict:
    """Mirror frame_loop.py's exposure-time override for integrated RGB noise."""
    out = dict(params)
    acquisition = AcquisitionProfile.from_params(out)
    out["exposure_time_s"] = acquisition.exposure_time_s
    runtime_state(out).exposure_signal_scale = acquisition.exposure_signal_scale
    if is_fluorescence_modality(ModalitySettings.from_params(out).modality):
        _force_detected_count_detector_domain(out)
    return out


def _rgb_contrast_frames_from_detector_counts(
    signal_rgb_frames: list[np.ndarray],
    reference_rgb_frames: list[np.ndarray],
    params: dict,
    *,
    qpi_phase_to_count_scale_rgb: np.ndarray | None = None,
    active_rgb_channels: np.ndarray | None = None,
) -> list[np.ndarray]:
    qpi_scales = None
    if qpi_phase_to_count_scale_rgb is not None:
        qpi_scales = np.asarray(qpi_phase_to_count_scale_rgb, dtype=float)
        if qpi_scales.shape != (3,):
            raise ValueError(
                "qpi_phase_to_count_scale_rgb must have shape (3,), ordered [R, G, B]; "
                f"got {qpi_scales.shape}."
            )
        if not np.all(np.isfinite(qpi_scales)) or np.any(qpi_scales < 0.0):
            raise ValueError(
                "qpi_phase_to_count_scale_rgb must contain finite non-negative RGB scales."
            )
    if active_rgb_channels is None:
        active_channels = np.ones(3, dtype=bool)
    else:
        active_channels = np.asarray(active_rgb_channels, dtype=bool)
        if active_channels.shape != (3,):
            raise ValueError(
                "active_rgb_channels must have shape (3,), ordered [R, G, B]; "
                f"got {active_channels.shape}."
            )

    per_channel_contrast = []
    for c in range(3):
        sig_c = [np.asarray(f, dtype=float)[:, :, c] for f in signal_rgb_frames]
        ref_c = [np.asarray(f, dtype=float)[:, :, c] for f in reference_rgb_frames]
        if qpi_scales is not None:
            if not active_channels[c]:
                per_channel_contrast.append([
                    np.zeros_like(np.asarray(sig_frame, dtype=float))
                    for sig_frame in sig_c
                ])
            else:
                scale = float(qpi_scales[c])
                if scale <= 0.0:
                    raise ValueError(
                        "Active QPI spectral RGB channels require a positive "
                        f"qpi_phase_to_count_scale_rgb entry; channel {c} has {scale!r}."
                    )
                per_channel_contrast.append([
                    (np.asarray(sig_frame, dtype=float) - np.asarray(ref_frame, dtype=float)) / scale
                    for sig_frame, ref_frame in zip(sig_c, ref_c)
                ])
        else:
            per_channel_contrast.append([
                np.asarray(frame, dtype=float)
                for frame in compute_contrast_frames(sig_c, ref_c, params)
            ])

    if not per_channel_contrast or any(len(channel) == 0 for channel in per_channel_contrast):
        return []

    n_frames = min(len(per_channel_contrast[0]), len(per_channel_contrast[1]), len(per_channel_contrast[2]))
    rgb_contrast = [
        np.stack(
            [
                per_channel_contrast[0][i],
                per_channel_contrast[1][i],
                per_channel_contrast[2][i],
            ],
            axis=-1,
        )
        for i in range(n_frames)
    ]
    return rgb_contrast


def _background_subtract_rgb(
    signal_rgb_frames: list[np.ndarray],
    reference_rgb_frames: list[np.ndarray],
    params: dict,
    *,
    qpi_phase_to_count_scale_rgb: np.ndarray | None = None,
    active_rgb_channels: np.ndarray | None = None,
) -> list[np.ndarray]:
    rgb_contrast = _rgb_contrast_frames_from_detector_counts(
        signal_rgb_frames,
        reference_rgb_frames,
        params,
        qpi_phase_to_count_scale_rgb=qpi_phase_to_count_scale_rgb,
        active_rgb_channels=active_rgb_channels,
    )
    if not rgb_contrast:
        return []

    stack = np.stack([np.asarray(frame, dtype=float) for frame in rgb_contrast], axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return [np.zeros_like(frame, dtype=np.uint8) for frame in rgb_contrast]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [np.zeros_like(frame, dtype=np.uint8) for frame in rgb_contrast]
    return [
        np.clip(255.0 * (np.asarray(frame, dtype=float) - lo) / (hi - lo), 0.0, 255.0).astype(np.uint8)
        for frame in rgb_contrast
    ]


def _zero_inactive_rgb_detector_channels(
    frames_rgb: list[np.ndarray],
    spectral_items: list[dict],
) -> list[np.ndarray]:
    response_totals = np.zeros(3, dtype=float)
    for item in spectral_items:
        response_totals += (
            np.asarray(item["detector_weights_rgb"], dtype=float)
            * float(item["spectral_weight"])
        )
    inactive = response_totals == 0.0
    if not np.any(inactive):
        return frames_rgb
    out = []
    for frame in frames_rgb:
        frame_arr = np.asarray(frame, dtype=np.uint8).copy()
        frame_arr[:, :, inactive] = 0
        out.append(frame_arr)
    return out


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
        raise ValueError("parameters['channels'] must be a non-empty list when set.")
    base_model = get_imaging_model(params)
    if not bool(getattr(base_model, "supports_spectral_channels", True)):
        raise ValueError(
            f"parameters['imaging_model']={ModalitySettings.from_params(params).modality!r} does not support "
            "parameters['channels']; use matched_microscopes or a microscope-specific loop instead."
        )

    latent_scene = _simulate_latent_scene(params)

    mask_settings = MaskGenerationSettings.from_params(params)
    spectral_items = []
    for channel_index, channel in enumerate(channels):
        channel_name, channel_params, detector_weights_rgb, spectral_weight = _channel_spec_to_params(
            params,
            channel,
            channel_index,
        )
        deterministic_params = _disable_detector_noise_for_spectral_component(channel_params)
        configured_assign(deterministic_params, 'return_ideal_float_frames', True)
        configured_assign(deterministic_params, 'mask_generation_enabled', bool(
            channel_index == 0 and mask_settings.enabled
        ))

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

    def _metadata_frame_sequence(metadata: dict, primary_key: str, fallback_key: str):
        frames = metadata.get(primary_key)
        if frames is None or len(frames) == 0:
            frames = metadata.get(fallback_key, [])
        return frames

    detector_input_signal_arrays = [
        np.asarray(
            _metadata_frame_sequence(
                _channel_result_metadata(item),
                "detector_input_signal_frames",
                "ideal_signal_frames",
            ),
            dtype=float,
        )
        for item in spectral_items
    ]
    detector_input_reference_arrays = [
        np.asarray(
            _metadata_frame_sequence(
                _channel_result_metadata(item),
                "detector_input_reference_frames",
                "ideal_reference_frames",
            ),
            dtype=float,
        )
        for item in spectral_items
    ]
    ideal_signal_arrays = [
        np.asarray(_channel_result_metadata(item).get("ideal_signal_frames", []), dtype=float)
        for item in spectral_items
    ]
    ideal_reference_arrays = [
        np.asarray(_channel_result_metadata(item).get("ideal_reference_frames", []), dtype=float)
        for item in spectral_items
    ]
    detector_mean_signal_arrays = [
        np.asarray(_channel_result_metadata(item).get("detector_mean_signal_frames", []), dtype=float)
        for item in spectral_items
    ]
    detector_mean_reference_arrays = [
        np.asarray(_channel_result_metadata(item).get("detector_mean_reference_frames", []), dtype=float)
        for item in spectral_items
    ]

    n_frames, h, w = _validate_multichannel_frame_arrays(
        detector_input_signal_arrays,
        detector_input_reference_arrays,
    )

    signal_rgb_float = []
    reference_rgb_float = []
    for t in range(n_frames):
        sig_rgb = np.zeros((h, w, 3), dtype=float)
        ref_rgb = np.zeros((h, w, 3), dtype=float)
        for item, sig_arr, ref_arr in zip(
            spectral_items,
            detector_input_signal_arrays,
            detector_input_reference_arrays,
        ):
            weights = item["detector_weights_rgb"] * item["spectral_weight"]
            for c in range(3):
                sig_rgb[:, :, c] += sig_arr[t] * weights[c]
                ref_rgb[:, :, c] += ref_arr[t] * weights[c]
        signal_rgb_float.append(sig_rgb)
        reference_rgb_float.append(ref_rgb)

    acquisition = AcquisitionProfile.from_params(params)
    rng_seed = acquisition.random_seed
    detector_noise_runtime = DetectorNoiseRuntime(
        rng=rng_from_seed(
            None if rng_seed is None else int(rng_seed),
            stream="spectral_channel_detector_noise",
        )
    )
    active_rgb_channels = np.zeros(3, dtype=bool)
    qpi_quanta_scale_rgb = np.zeros(3, dtype=float)
    qpi_phase_to_count_scale_rgb = np.zeros(3, dtype=float)
    qpi_rgb_noise_is_phase_domain = getattr(base_model, "output_type", "intensity") == "phase"
    for item in spectral_items:
        detector_response_rgb = (
            np.asarray(item["detector_weights_rgb"], dtype=float)
            * float(item["spectral_weight"])
        )
        active_rgb_channels |= detector_response_rgb > 0.0
        qpi_quanta_scale_rgb += detector_response_rgb
        if qpi_rgb_noise_is_phase_domain:
            item_phase_to_count = float(
                QpiReadoutSettings.from_params(
                    item["params"]
                ).phase_to_count_scale
            )
            if not np.isfinite(item_phase_to_count) or item_phase_to_count < 0.0:
                raise ValueError(
                    "QPI spectral qpi_phase_to_count_scale must be finite and "
                    f"non-negative; got {item_phase_to_count!r} for channel "
                    f"{item.get('name', '<unnamed>')!r}."
                )
            qpi_phase_to_count_scale_rgb += detector_response_rgb * item_phase_to_count
    rgb_noise_params = _rgb_noise_params_with_effective_exposure(params)
    signal_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(
        signal_rgb_float,
        params,
        detector_noise_runtime=detector_noise_runtime,
        active_rgb_channels=active_rgb_channels,
        rgb_noise_params=rgb_noise_params,
        qpi_quanta_scale_rgb=qpi_quanta_scale_rgb,
        qpi_phase_to_count_scale_rgb=(
            qpi_phase_to_count_scale_rgb if qpi_rgb_noise_is_phase_domain else None
        ),
    )
    reference_rgb_noisy = _apply_detector_noise_to_rgb_raw_frames(
        reference_rgb_float,
        params,
        detector_noise_runtime=detector_noise_runtime,
        active_rgb_channels=active_rgb_channels,
        rgb_noise_params=rgb_noise_params,
        qpi_quanta_scale_rgb=qpi_quanta_scale_rgb,
        qpi_phase_to_count_scale_rgb=(
            qpi_phase_to_count_scale_rgb if qpi_rgb_noise_is_phase_domain else None
        ),
    )

    final_rgb = _background_subtract_rgb(
        signal_rgb_noisy,
        reference_rgb_noisy,
        params,
        qpi_phase_to_count_scale_rgb=(
            qpi_phase_to_count_scale_rgb if qpi_rgb_noise_is_phase_domain else None
        ),
        active_rgb_channels=active_rgb_channels,
    )
    final_rgb = _zero_inactive_rgb_detector_channels(final_rgb, spectral_items)
    output_settings = SimulationOutputSettings.from_params(params)
    output_mode = output_settings.multichannel_output_mode
    written_channel_sidecars = []
    raw_signal_video_path = None
    if output_mode in {"rgb", "both"}:
        _save_rgb_video(output_settings.output_filename, final_rgb, acquisition.fps)
        if output_settings.save_raw_camera_video:
            raw_signal_video_path = _raw_signal_video_filename(params)
            raw_rgb_preview = normalize_raw_camera_frames(signal_rgb_noisy, params)
            _save_rgb_video(raw_signal_video_path, raw_rgb_preview, acquisition.fps)
    if output_mode in {"channels", "both"}:
        written_channel_sidecars = _save_channel_videos(
            params,
            spectral_items,
            acquisition.fps,
        )


    if return_frames:
        primary_channel_metadata = _channel_result_metadata(spectral_items[0])
        qpi_effective_quanta_rgb = None
        qpi_phase_to_count_scale_rgb_record = None
        if getattr(base_model, "output_type", "intensity") == "phase":
            qpi_quanta_raw = (
                QpiReadoutSettings.from_params(params)
                .configured_detected_quanta_per_pixel
            )
            qpi_effective_quanta_rgb = [
                float(qpi_quanta_raw) * float(scale)
                for scale in qpi_quanta_scale_rgb
            ]
            qpi_phase_to_count_scale_rgb_record = [
                float(scale) for scale in qpi_phase_to_count_scale_rgb
            ]
        return _simulation_result(final_rgb, ["red", "green", "blue"], {
            "spectral_channels": [item["name"] for item in spectral_items],
            "spectral_items": spectral_items,
            "spectral_integration_model": SpectralIntegrationSettings.from_params(params).model,
            "qpi_quanta_scale_rgb": [float(scale) for scale in qpi_quanta_scale_rgb],
            "qpi_effective_detected_quanta_per_pixel_rgb": qpi_effective_quanta_rgb,
            "qpi_phase_to_count_scale_rgb": qpi_phase_to_count_scale_rgb_record,
            "generated_spectral_channels": bool(runtime_state(params).generated_spectral_channels),
            "spectral_channel_count": int(len(spectral_items)),
            "trajectories_nm": np.asarray(latent_scene.get("trajectories_nm", [])),
            "rendered_trajectories_nm": np.asarray(
                primary_channel_metadata.get("rendered_trajectories_nm", [])
            ),
            "trajectory_semantics": primary_channel_metadata.get(
                "trajectory_semantics",
                {
                    "trajectories_nm": (
                        "latent Brownian particle centers before render-time "
                        "rigid drift/vibration"
                    ),
                    "rendered_trajectories_nm": (
                        "exposure-averaged particle centers actually rendered "
                        "into output frames"
                    ),
                },
            ),
            "ideal_signal_frames_by_spectral_sample": ideal_signal_arrays,
            "ideal_reference_frames_by_spectral_sample": ideal_reference_arrays,
            "detector_input_signal_frames_by_spectral_sample": detector_input_signal_arrays,
            "detector_input_reference_frames_by_spectral_sample": detector_input_reference_arrays,
            "detector_mean_signal_frames_by_spectral_sample": detector_mean_signal_arrays,
            "detector_mean_reference_frames_by_spectral_sample": detector_mean_reference_arrays,
            "raw_signal_frames_rgb": signal_rgb_noisy,
            "raw_reference_frames_rgb": reference_rgb_noisy,
            "background_subtracted_frames_rgb": final_rgb,
            "background_subtracted_frames_rgb_frame_basis": STAGE_DISPLAY,
            "background_subtracted_frames_rgb_contrast_basis": "display_only",
            "background_subtracted_frames_rgb_quantitative": False,
            "background_subtracted_frames_rgb_units": "display_only",
            "quantitative_contrast_frames_rgb": [],
            "quantitative_contrast_frames_rgb_quantitative": False,
            "quantitative_contrast_frames_rgb_status": (
                "not_constructed; RGB products are display composites. "
                "Use spectral sample detector-input/mean arrays for quantitative "
                "per-channel analysis."
            ),
            "analysis_video_path": output_settings.output_filename if output_mode in {"rgb", "both"} else None,
            "raw_signal_video_path": raw_signal_video_path,
            "analysis_video_semantics": "display_normalized_background_subtracted_rgb_uint8",
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
    "_rgb_contrast_frames_from_detector_counts",
    "_run_multichannel_simulation",
    "_safe_channel_filename",
    "_save_channel_videos",
    "_save_rgb_video",
    "_validate_multichannel_frame_arrays",
    "_rgb_noise_params_with_effective_exposure",
    "_wavelength_to_rgb_weights",
]
