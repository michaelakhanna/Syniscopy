from __future__ import annotations

from copy import deepcopy
import os

import numpy as np
import pytest


pytestmark = [pytest.mark.renderer, pytest.mark.electron]


def _renderer_enabled() -> bool:
    return os.environ.get("SYNISCOPY_VERIFY_RUN_RENDERER") == "1"


def _compact_params() -> dict:
    from config import PARAMS, normalize_params

    params = deepcopy(PARAMS)
    params.update(
        {
            "image_size_pixels": 16,
            "pixel_size_nm": 50.0,
            "psf_oversampling_factor": 1,
            "pupil_samples": 16,
            "vectorial_pupil_samples": 16,
            "max_psf_z_slices": 64,
            "z_stack_step_nm": 500.0,
            "num_frames": 1,
            "duration_seconds": 1.0 / 24.0,
            "background_subtraction_method": "reference_frame",
            "mask_generation_enabled": False,
            "sample_environment_enabled": False,
            "sample_environment_pattern_enabled": False,
            "return_ideal_float_frames": True,
            "motion_blur_enabled": False,
            "save_frame_sequence": False,
            "save_raw_camera_video": False,
            "save_raw_camera_frame_sequence": False,
            "save_raw_frame_views": False,
            "fluorescence_backend": "parametric_psf",
            "tem_model": "weak_phase_ctf",
            "tem_backend": "ctf_proxy",
            "sem_model": "interaction_volume_proxy",
            "sem_backend": "interaction_volume_proxy",
            "sem_source_representation": "projected",
            "random_seed": 123,
        }
    )
    params["particles"] = [deepcopy(PARAMS["particles"][0])]
    params["particles"][0]["motion"]["initial_position_nm"] = [0.0, 0.0, 0.0]
    return normalize_params(params)


def _single_frame_trace(params: dict, modality: str) -> float:
    from lab_fisher_report.microscopes import MicroscopeSpec
    from lab_fisher_report.render import _render_microscope

    rendered, _ = _render_microscope(
        params,
        MicroscopeSpec(name=modality, modality=modality),
    )
    F = np.asarray(rendered["fisher_matrices"][0], dtype=float)
    return float(np.trace(F))


def test_tem_near_zero_voltage_erases_live_rendered_fim() -> None:
    if not _renderer_enabled():
        pytest.skip("set SYNISCOPY_VERIFY_RUN_RENDERER=1 or pass --include-renderer")

    high = _compact_params()
    high["tem_acceleration_kV"] = 300.0
    low = deepcopy(high)
    low["tem_acceleration_kV"] = 1.0e-6

    high_trace = _single_frame_trace(high, "tem_phase_contrast")
    low_trace = _single_frame_trace(low, "tem_phase_contrast")

    assert high_trace > 0.0
    assert low_trace <= high_trace * 1.0e-6


def test_sem_near_zero_voltage_erases_live_rendered_fim() -> None:
    if not _renderer_enabled():
        pytest.skip("set SYNISCOPY_VERIFY_RUN_RENDERER=1 or pass --include-renderer")

    high = _compact_params()
    high["sem_acceleration_kV"] = 20.0
    low = deepcopy(high)
    low["sem_acceleration_kV"] = 1.0e-6

    high_trace = _single_frame_trace(high, "sem_secondary_electron")
    low_trace = _single_frame_trace(low, "sem_secondary_electron")

    assert high_trace > 0.0
    assert low_trace <= high_trace * 1.0e-6
