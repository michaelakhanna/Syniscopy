from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _sideband_mask(
    shape: tuple[int, int],
    *,
    carrier_bin_yx: tuple[int, int],
    radius_bins: float,
) -> np.ndarray:
    h, w = shape
    yy = np.fft.fftfreq(h) * h
    xx = np.fft.fftfreq(w) * w
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    cy, cx = carrier_bin_yx
    return np.hypot(grid_y - float(cy), grid_x - float(cx)) <= float(radius_bins)


def _complex_coordinate_derivatives(
    field: np.ndarray,
    pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(field, dtype=np.complex128)
    h, w = z.shape
    fx = np.fft.fftfreq(w, d=float(pixel_size_nm))
    fy = np.fft.fftfreq(h, d=float(pixel_size_nm))
    spectrum = np.fft.fft2(z)
    gx = np.fft.ifft2(spectrum * (1j * 2.0 * np.pi * fx)[None, :])
    gy = np.fft.ifft2(spectrum * (1j * 2.0 * np.pi * fy)[:, None])
    return -gx, -gy


def _complex_inline_reference_fisher(
    field: np.ndarray,
    *,
    pixel_size_nm: float,
    complex_variance: float,
) -> np.ndarray:
    """Fisher for a proper-complex full-field reference on [Re, Im]."""

    dx, dy = _complex_coordinate_derivatives(field, pixel_size_nm)
    grads = (dx, dy)
    fisher = np.zeros((2, 2), dtype=float)
    for i, gi in enumerate(grads):
        for j, gj in enumerate(grads[i:], start=i):
            value = float(2.0 * np.real(np.vdot(gi, gj)) / float(complex_variance))
            fisher[i, j] = value
            fisher[j, i] = value
    return fisher


def _sigma_xy_nm(fisher: np.ndarray) -> float:
    cov = np.linalg.inv(np.asarray(fisher, dtype=float))
    return float(np.sqrt(cov[0, 0] + cov[1, 1]))


def _synthetic_shift_covariant_complex_field(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=float) - 0.5 * h,
        np.arange(w, dtype=float) - 0.5 * w,
        indexing="ij",
    )
    envelope = np.exp(-(xx * xx + yy * yy) / (2.0 * 4.5 * 4.5))
    low_frequency_object = (1.0 + 0.35j) * envelope
    high_frequency_structure = 0.30 * envelope * np.exp(
        1j * 2.0 * np.pi * (6.0 * xx / float(w) + 4.0 * yy / float(h))
    )
    return low_frequency_object + high_frequency_structure


def _dhm_test_params() -> dict:
    from config import default_params

    params = default_params()
    params.update(
        {
            "imaging_model": "off_axis_holography",
            "background_subtraction_method": "reference_frame",
            "shot_noise_enabled": True,
            "gaussian_noise_enabled": False,
            "camera_gain_e_per_count": 1.0,
            "detector_qe": 1.0,
            "detector_noise_input_domain": "camera_counts",
            "read_noise_counts": 0.0,
            "dark_current_e_per_pixel_per_s": 0.0,
            "exposure_time_s": 1.0,
            "scan_line_noise_counts": 0.0,
            "background_offset_counts": 0.0,
            "dark_offset_counts": 0.0,
            "fixed_pattern_gain_std": 0.0,
            "fixed_pattern_offset_counts": 0.0,
            "hot_pixel_fraction": 0.0,
            "adc_quantization": False,
            "nonlinearity_calibration": None,
            "flat_field_map": None,
            "dark_frame_map": None,
        }
    )
    return params


def test_off_axis_demodulated_sideband_bound_is_worse_than_inline_reference_same_photon_budget() -> None:
    """DHM sideband demodulation must not beat an inline complex-field reference.

    The comparison is deliberately below the renderer layer.  Both observables
    use the same shift-covariant complex object field and the same unit raw
    count variance.  The inline reference sees the whole complex field.  The
    off-axis path sees only the declared +1 Fourier sideband and must invert the
    correlated covariance produced by that projection.  Therefore it cannot
    have a smaller localization CRLB than the inline full-field likelihood.
    """

    from fisher import compute_off_axis_demodulated_localization_crlb_from_field
    from noise_contracts import fourier_sideband_demodulated_noise_model

    shape = (32, 32)
    pixel_size_nm = 25.0
    raw_variance = np.ones(shape, dtype=float)
    carrier_bin_yx = (0, 8)
    sideband_shift = (0, -8)
    mask = _sideband_mask(shape, carrier_bin_yx=carrier_bin_yx, radius_bins=2.5)
    field = _synthetic_shift_covariant_complex_field(shape)

    dhm_noise = fourier_sideband_demodulated_noise_model(
        diagonal_variance=np.ones(shape, dtype=float),
        raw_variance=raw_variance,
        sideband_mask=mask,
        sideband_shift=sideband_shift,
        context="test_off_axis_demodulated_sideband_bound",
    )
    dhm = compute_off_axis_demodulated_localization_crlb_from_field(
        field,
        dhm_noise,
        pixel_size_nm,
    )

    inline_fisher = _complex_inline_reference_fisher(
        field,
        pixel_size_nm=pixel_size_nm,
        complex_variance=1.0,
    )
    inline_sigma_xy = _sigma_xy_nm(inline_fisher)
    dhm_sigma_xy = float(dhm["sigma_xy_nm"])

    assert str(dhm["analysis_noise_covariance_kind"]) == "fourier_sideband_demodulated_complex_field"
    assert np.all(np.isfinite(inline_fisher))
    assert np.isfinite(inline_sigma_xy)
    assert np.isfinite(dhm_sigma_xy)
    assert dhm_sigma_xy > 1.05 * inline_sigma_xy


def test_off_axis_demodulator_uses_detector_grid_period_for_detector_frames() -> None:
    from fisher.dhm_demodulated import (
        OFF_AXIS_DEMODULATION_GRID_DETECTOR,
        build_off_axis_demodulated_observation,
    )

    shape = (64, 128)
    x = np.arange(shape[1], dtype=float)[None, :]
    signal = 100.0 + 10.0 * np.cos(2.0 * np.pi * x / 8.0)
    signal = np.repeat(signal, shape[0], axis=0)
    reference = np.full(shape, 100.0, dtype=float)
    params = _dhm_test_params()
    response = {
        "off_axis_fringe_period_detector_px": 8.0,
        "off_axis_fringe_period_canvas_px": 16.0,
        "off_axis_fringe_angle_rad": 0.0,
    }

    observation = build_off_axis_demodulated_observation(
        signal,
        reference,
        params,
        response_function=response,
        object_field_detector=np.ones(shape, dtype=np.complex128),
    )

    assert observation.metadata["off_axis_demodulation_input_grid"] == OFF_AXIS_DEMODULATION_GRID_DETECTOR
    assert observation.metadata["off_axis_demodulation_period_basis"] == "off_axis_fringe_period_detector_px"
    assert observation.metadata["off_axis_carrier_bin_yx"] == [0, 16]
    assert observation.metadata["off_axis_sideband_shift_yx"] == [0, -16]
    assert abs(float(np.mean(np.abs(observation.field_contrast))) - 5.0) < 0.25


def test_off_axis_demodulated_covariance_uses_raw_signal_noise_not_reference_subtracted_noise() -> None:
    from fisher.dhm_demodulated import build_off_axis_demodulated_observation

    shape = (32, 32)
    signal = np.full(shape, 100.0, dtype=float)
    reference = np.full(shape, 100.0, dtype=float)
    params = _dhm_test_params()
    response = {
        "off_axis_fringe_period_detector_px": 8.0,
        "off_axis_fringe_period_canvas_px": 16.0,
        "off_axis_fringe_angle_rad": 0.0,
    }

    observation = build_off_axis_demodulated_observation(
        signal,
        reference,
        params,
        response_function=response,
        object_field_detector=np.ones(shape, dtype=np.complex128),
    )
    raw_variance = np.asarray(observation.noise_model.fourier_sideband_raw_variance, dtype=float)

    assert observation.metadata["off_axis_noise_source"] == "raw_signal_interferogram_only"
    assert observation.metadata["off_axis_reference_treatment"] == "deterministic_demodulated_empty_reference_centering"
    assert np.allclose(raw_variance, 100.0)
    assert not np.allclose(raw_variance, 200.0)


def test_off_axis_demodulator_reconstructs_field_contrast_not_object_product_or_conjugate() -> None:
    from fisher.dhm_demodulated import build_off_axis_demodulated_observation

    shape = (64, 64)
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        indexing="ij",
    )
    object_field = (
        1.0
        + 0.20 * np.sin(2.0 * np.pi * xx / float(shape[1]))
        + 0.10j * np.cos(2.0 * np.pi * yy / float(shape[0]))
    ).astype(np.complex128)
    scattered_contrast = 1.0 + 2.0j
    carrier = np.exp(1j * 2.0 * np.pi * xx / 8.0)
    reference_amplitude = 1.0
    signal = np.abs(object_field + scattered_contrast + reference_amplitude * object_field * carrier) ** 2
    reference = np.abs(object_field + reference_amplitude * object_field * carrier) ** 2
    params = _dhm_test_params()
    response = {
        "off_axis_fringe_period_detector_px": 8.0,
        "off_axis_fringe_period_canvas_px": 16.0,
        "off_axis_fringe_angle_rad": 0.0,
        "off_axis_reference_amplitude_scale": reference_amplitude,
    }

    observation = build_off_axis_demodulated_observation(
        signal,
        reference,
        params,
        response_function=response,
        object_field_detector=object_field,
    )
    reconstructed = np.asarray(observation.field_contrast, dtype=np.complex128)

    assert observation.metadata["off_axis_sideband_conjugated_for_field_convention"] is True
    assert observation.metadata["off_axis_reconstruction_normalization"] == (
        "conj(centered_plus_one_sideband)/conj(a_E_obj_detector)"
    )
    assert abs(complex(np.mean(reconstructed)) - scattered_contrast) < 2.0e-12
    assert np.std(reconstructed - scattered_contrast) < 2.0e-12
    assert abs(complex(np.mean(reconstructed)) - np.conj(scattered_contrast)) > 3.0


def test_off_axis_demodulator_removes_fractional_detector_carrier_residual() -> None:
    from fisher.dhm_demodulated import (
        _carrier_demodulation_plan,
        _complex_spectral_coordinate_derivatives,
    )

    shape = (96, 192)
    response = {
        "off_axis_fringe_period_detector_px": 10.0,
        "off_axis_fringe_period_canvas_px": 20.0,
        "off_axis_fringe_angle_rad": 0.0,
    }
    plan = _carrier_demodulation_plan(shape, response)
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        indexing="ij",
    )
    residual_y, residual_x = plan.residual_cycles_per_pixel_yx
    integer_shifted_residual = np.exp(1j * 2.0 * np.pi * (residual_x * xx + residual_y * yy))
    corrected = plan.phase_correction * integer_shifted_residual
    old_dx, old_dy = _complex_spectral_coordinate_derivatives(integer_shifted_residual, 65.0)
    new_dx, new_dy = _complex_spectral_coordinate_derivatives(corrected, 65.0)

    assert plan.carrier_bin == (0, 19)
    assert np.isclose(plan.exact_bin_yx[1], 19.2)
    assert np.isclose(plan.residual_cycles_per_pixel_yx[1], 1.0 / 960.0)
    assert np.max(np.abs(corrected - 1.0)) < 1.0e-12
    assert np.linalg.norm(new_dx) + np.linalg.norm(new_dy) < 1.0e-10 * (
        np.linalg.norm(old_dx) + np.linalg.norm(old_dy)
    )


def test_off_axis_sideband_fisher_matches_real_augmented_covariance() -> None:
    from fisher.precision import (
        _project_demodulated_sideband,
        _sideband_forward_from_raw,
        compute_fisher_from_complex_fourier_sideband_gradients,
    )
    from noise_contracts import fourier_sideband_demodulated_noise_model

    shape = (4, 4)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 1] = True
    sideband_shift = (0, -1)
    baseband_mask = np.roll(mask.astype(float), sideband_shift, axis=(0, 1))
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        indexing="ij",
    )
    phase_correction = np.exp(-1j * 2.0 * np.pi * (0.125 * xx / float(shape[1]) + 0.25 * yy / float(shape[0])))
    raw_variance = 0.5 + np.arange(1, 17, dtype=float).reshape(shape) / 10.0
    grads = (
        _project_demodulated_sideband(
            np.ones(shape, dtype=np.complex128) * (1.0 + 2.0j),
            baseband_mask,
            phase_correction,
        ),
        _project_demodulated_sideband(
            np.ones(shape, dtype=np.complex128) * (0.5 - 0.25j),
            baseband_mask,
            phase_correction,
        ),
    )
    noise_model = fourier_sideband_demodulated_noise_model(
        diagonal_variance=np.ones(shape, dtype=float),
        raw_variance=raw_variance,
        sideband_mask=mask,
        sideband_shift=sideband_shift,
        sideband_phase_correction=phase_correction,
        context="test_off_axis_sideband_fisher_matches_real_augmented_covariance",
    )

    fisher, metadata = compute_fisher_from_complex_fourier_sideband_gradients(
        grads,
        noise_model,
        tolerance=1.0e-12,
        max_iterations=512,
    )

    pixel_count = int(np.prod(shape))
    dense_operator = np.zeros((2 * pixel_count, pixel_count), dtype=float)
    for raw_index in range(pixel_count):
        raw_basis = np.zeros(shape, dtype=float)
        raw_basis.ravel()[raw_index] = 1.0
        sideband_field = _project_demodulated_sideband(
            _sideband_forward_from_raw(
                raw_basis,
                carrier_mask=mask.astype(float),
                sideband_shift=sideband_shift,
                phase_correction=phase_correction,
            ),
            baseband_mask,
            phase_correction,
        )
        dense_operator[:pixel_count, raw_index] = sideband_field.real.ravel()
        dense_operator[pixel_count:, raw_index] = sideband_field.imag.ravel()

    real_covariance = (
        dense_operator
        @ np.diag(raw_variance.ravel())
        @ dense_operator.T
    )
    precision = np.linalg.pinv(real_covariance, rcond=1.0e-12)
    dense_grads = [
        np.concatenate([grad.real.ravel(), grad.imag.ravel()])
        for grad in grads
    ]
    expected = np.array(
        [
            [
                float(dense_grads[i] @ precision @ dense_grads[j])
                for j in range(len(dense_grads))
            ]
            for i in range(len(dense_grads))
        ]
    )

    assert metadata["precision_complex_observation_model"] == "real_augmented_re_im_covariance"
    assert metadata["precision_sideband_fractional_phase_correction"] is True
    assert np.allclose(fisher, expected, rtol=1.0e-10, atol=1.0e-10)
