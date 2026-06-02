from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CODEBASE = Path(__file__).resolve().parents[1] / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def test_runtime_param_keys_are_imported_by_split_simulation_modules() -> None:
    import simulation.scene_render as scene_render
    import simulation.spectral_channels as spectral_channels

    assert "_RUNTIME_PARAM_KEYS" in vars(scene_render)
    assert "_RUNTIME_PARAM_KEYS" in vars(spectral_channels)


def test_phase_noise_variance_respects_explicit_relative_reference() -> None:
    from camera_noise import analysis_contrast_noise_variance

    params = {
        "imaging_model": "quantitative_phase",
        "background_subtraction_method": "reference_frame",
        "shot_noise_enabled": True,
        "gaussian_noise_enabled": False,
        "camera_gain_e_per_count": 1.0,
        "qpi_phase_to_count_scale": 10.0,
    }
    signal = np.asarray([[20.0]])
    reference = np.asarray([[10.0]])

    additive = analysis_contrast_noise_variance(
        signal,
        reference,
        params,
        relative_reference=False,
    )
    relative = analysis_contrast_noise_variance(
        signal,
        reference,
        params,
        relative_reference=True,
    )

    assert np.allclose(additive, [[0.3]])
    assert np.allclose(relative, [[0.006]])


def test_frames_to_channel_first_preserves_channel_first_with_small_width() -> None:
    from simulation.output import _frames_to_channel_first

    frames = np.arange(2 * 3 * 5 * 3, dtype=np.uint8).reshape(2, 3, 5, 3)

    converted = _frames_to_channel_first(frames, channel_count=3)

    assert converted.shape == (2, 3, 5, 3)
    assert np.array_equal(converted, frames)


def test_frames_to_channel_first_converts_channels_last() -> None:
    from simulation.output import _frames_to_channel_first

    frames = np.arange(2 * 5 * 7 * 3, dtype=np.uint8).reshape(2, 5, 7, 3)

    converted = _frames_to_channel_first(frames, channel_count=3)

    assert converted.shape == (2, 3, 5, 7)
    assert np.array_equal(converted, np.moveaxis(frames, -1, 1))


def test_single_frame_viewer_center_uses_pixel_center_coordinates() -> None:
    from single_frame_viewer import _compute_center_position_nm

    center_x, center_y = _compute_center_position_nm(
        {"image_size_pixels": 4, "pixel_size_nm": 10.0}
    )

    assert center_x == 15.0
    assert center_y == 15.0


def test_multichannel_channel_result_metadata_uses_simulation_result_dict() -> None:
    from simulation.spectral_channels import _channel_result_metadata

    item = {
        "frames": {
            "frames": np.zeros((1, 1, 2, 2), dtype=np.uint8),
            "metadata": {"ideal_signal_frames": [np.ones((2, 2))]},
        }
    }

    metadata = _channel_result_metadata(item)

    assert "ideal_signal_frames" in metadata


def test_multichannel_channel_result_metadata_rejects_frame_array_payload() -> None:
    from simulation.spectral_channels import _channel_result_metadata

    try:
        _channel_result_metadata({"frames": np.zeros((1, 2, 2), dtype=np.uint8)})
    except TypeError as exc:
        assert "_simulation_result dict" in str(exc)
    else:
        raise AssertionError("Expected TypeError for non-dict channel frame payload.")
