"""Off-axis DHM demodulated-field lateral Fisher owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from noise_contracts import (
    AnalysisNoiseModel,
    fourier_sideband_demodulated_noise_model,
)

from ._metadata_helpers import _localization_derivative_metadata
from .lateral import _localization_crlb_from_fisher
from .precision import compute_fisher_from_complex_fourier_sideband_gradients
from .spectral_fisher import boundary_energy_fraction, nyquist_band_fraction


OFF_AXIS_HOLOGRAPHY_MODALITY = "off_axis_holography"
OFF_AXIS_DEMODULATED_COVARIANCE_KIND = "fourier_sideband_demodulated_complex_field"
OFF_AXIS_DEMODULATION_GRID_DETECTOR = "detector_frame"
OFF_AXIS_DEMODULATION_GRID_CANVAS = "model_canvas"


@dataclass(frozen=True)
class OffAxisDemodulatedObservation:
    """Complex reconstructed field and propagated Fisher likelihood."""

    field_contrast: np.ndarray
    raw_interferogram: np.ndarray
    noise_model: AnalysisNoiseModel
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _OffAxisCarrierDemodulation:
    carrier_bin: tuple[int, int]
    sideband_shift: tuple[int, int]
    exact_bin_yx: tuple[float, float]
    residual_bin_yx: tuple[float, float]
    residual_cycles_per_pixel_yx: tuple[float, float]
    phase_correction: np.ndarray


def is_off_axis_holography_modality(modality: Any) -> bool:
    return str(modality or "").strip().lower() == OFF_AXIS_HOLOGRAPHY_MODALITY


def _carrier_demodulation_plan(
    shape: tuple[int, int],
    response_function: Mapping[str, Any],
    *,
    input_grid: str = OFF_AXIS_DEMODULATION_GRID_DETECTOR,
) -> _OffAxisCarrierDemodulation:
    h, w = (int(shape[0]), int(shape[1]))
    grid = str(input_grid or "").strip().lower()
    if grid == OFF_AXIS_DEMODULATION_GRID_DETECTOR:
        period_key = "off_axis_fringe_period_detector_px"
    elif grid == OFF_AXIS_DEMODULATION_GRID_CANVAS:
        period_key = "off_axis_fringe_period_canvas_px"
    else:
        raise ValueError(
            "off-axis demodulated Fisher input_grid must be "
            f"{OFF_AXIS_DEMODULATION_GRID_DETECTOR!r} or "
            f"{OFF_AXIS_DEMODULATION_GRID_CANVAS!r}; got {input_grid!r}."
        )
    period = float(response_function.get(period_key, 0.0))
    angle = float(response_function.get("off_axis_fringe_angle_rad", 0.0))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(
            "off-axis demodulated Fisher requires response metadata field "
            f"{period_key!r} for {grid!r} inputs."
        )
    exact_x = float(w * np.cos(angle) / period)
    exact_y = float(h * np.sin(angle) / period)
    kx = int(round(exact_x))
    ky = int(round(exact_y))
    if kx == 0 and ky == 0:
        raise ValueError(
            "off-axis demodulated Fisher carrier rounds to the DC Fourier bin; "
            "increase image size or decrease off_axis_fringe_period_px."
        )
    if abs(kx) >= w // 2 or abs(ky) >= h // 2:
        raise ValueError(
            "off-axis demodulated Fisher carrier is at or beyond Nyquist; "
            f"got carrier bins (ky={ky}, kx={kx}) for shape {(h, w)}."
        )
    carrier_bin = (ky, kx)
    sideband_shift = (-ky, -kx)
    residual_bin_y = exact_y - float(ky)
    residual_bin_x = exact_x - float(kx)
    residual_cycles_y = residual_bin_y / float(h)
    residual_cycles_x = residual_bin_x / float(w)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=float),
        np.arange(w, dtype=float),
        indexing="ij",
    )
    phase = -2.0 * np.pi * (residual_cycles_x * xx + residual_cycles_y * yy)
    phase_correction = np.exp(1j * phase)
    return _OffAxisCarrierDemodulation(
        carrier_bin=carrier_bin,
        sideband_shift=sideband_shift,
        exact_bin_yx=(exact_y, exact_x),
        residual_bin_yx=(residual_bin_y, residual_bin_x),
        residual_cycles_per_pixel_yx=(residual_cycles_y, residual_cycles_x),
        phase_correction=phase_correction.astype(np.complex128),
    )


def _carrier_bin_and_shift(
    shape: tuple[int, int],
    response_function: Mapping[str, Any],
    *,
    input_grid: str = OFF_AXIS_DEMODULATION_GRID_DETECTOR,
) -> tuple[tuple[int, int], tuple[int, int]]:
    plan = _carrier_demodulation_plan(
        shape,
        response_function,
        input_grid=input_grid,
    )
    return plan.carrier_bin, plan.sideband_shift


def _wrapped_bin_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = (int(shape[0]), int(shape[1]))
    yy = np.fft.fftfreq(h) * h
    xx = np.fft.fftfreq(w) * w
    return np.meshgrid(yy, xx, indexing="ij")


def _sideband_mask(
    shape: tuple[int, int],
    carrier_bin: tuple[int, int],
) -> tuple[np.ndarray, float]:
    ky, kx = carrier_bin
    yy, xx = _wrapped_bin_grid(shape)
    distance = np.hypot(yy - float(ky), xx - float(kx))
    carrier_distance = float(np.hypot(float(ky), float(kx)))
    if not np.isfinite(carrier_distance) or carrier_distance <= 0.0:
        raise ValueError("off-axis demodulated Fisher carrier distance must be positive.")
    radius = max(
        1.0,
        min(
            0.45 * carrier_distance,
            0.25 * float(min(shape)),
        ),
    )
    mask = distance <= radius
    if not np.any(mask):
        raise ValueError("off-axis demodulated Fisher sideband mask is empty.")
    return mask, float(radius)


def _demodulate_with_sideband(
    raw_frame: np.ndarray,
    *,
    sideband_mask: np.ndarray,
    sideband_shift: tuple[int, int],
    phase_correction: np.ndarray | None = None,
) -> np.ndarray:
    spectrum = np.fft.fft2(np.asarray(raw_frame, dtype=float))
    shifted = np.roll(spectrum * sideband_mask, sideband_shift, axis=(0, 1))
    field = np.fft.ifft2(shifted)
    if phase_correction is None:
        return field
    correction = np.asarray(phase_correction, dtype=np.complex128)
    if correction.shape != field.shape:
        raise ValueError(
            "off-axis demodulated Fisher phase correction shape "
            f"{correction.shape} does not match field shape {field.shape}."
        )
    return correction * field


def _sideband_diagonal_variance_summary(
    raw_variance: np.ndarray,
    *,
    sideband_mask: np.ndarray,
    sideband_shift: tuple[int, int],
    output_normalization: np.ndarray | None = None,
) -> np.ndarray:
    baseband_mask = np.roll(sideband_mask.astype(float), sideband_shift, axis=(0, 1))
    kernel_power = np.abs(np.fft.ifft2(baseband_mask)) ** 2
    diagonal = np.fft.ifft2(np.fft.fft2(raw_variance) * np.fft.fft2(kernel_power)).real
    if output_normalization is not None:
        norm = np.asarray(output_normalization, dtype=np.complex128)
        if norm.shape != diagonal.shape:
            raise ValueError(
                "off-axis sideband output-normalization shape "
                f"{norm.shape} does not match variance shape {diagonal.shape}."
            )
        diagonal = diagonal * np.square(np.abs(norm))
    return np.maximum(diagonal, 1e-30)


def _reconstruction_output_normalization(
    object_field_detector: np.ndarray,
    reference_amplitude_scale: float,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    object_field = np.asarray(object_field_detector, dtype=np.complex128)
    if object_field.shape != expected_shape:
        raise ValueError(
            "off-axis demodulated Fisher requires object_field_detector on the "
            f"same detector grid as the raw interferogram; got {object_field.shape} "
            f"for raw shape {expected_shape}."
        )
    if (
        np.any(~np.isfinite(object_field.real))
        or np.any(~np.isfinite(object_field.imag))
    ):
        raise ValueError("off-axis object_field_detector must contain only finite values.")
    scale = float(reference_amplitude_scale)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(
            "off-axis demodulated Fisher requires finite nonzero "
            "off_axis_reference_amplitude_scale."
        )
    reference_factor = scale * object_field
    if np.any(np.abs(reference_factor) <= 0.0):
        raise ValueError(
            "off-axis demodulated Fisher cannot reconstruct through zero "
            "object/reference field samples."
        )
    # The rendered carrier is exp(+iK.r), so the selected +1 sideband contains
    # a * E_obj * F^*.  The reconstructed field contrast is therefore obtained
    # by conjugating the centered sideband and dividing by conj(a * E_obj).
    return (1.0 / np.conj(reference_factor)).astype(np.complex128)


def _reconstruct_field_contrast_from_plus_sideband(
    signal_field: np.ndarray,
    reference_field: np.ndarray,
    output_normalization: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(signal_field, dtype=np.complex128) - np.asarray(
        reference_field,
        dtype=np.complex128,
    )
    return np.asarray(output_normalization, dtype=np.complex128) * np.conj(centered)


def _complex_spectral_coordinate_derivatives(
    field: np.ndarray,
    pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(field, dtype=np.complex128)
    if z.ndim != 2:
        raise ValueError(f"off-axis demodulated field must be 2D; got {z.shape}.")
    if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)):
        raise ValueError("off-axis demodulated field must contain only finite values.")
    if not np.isfinite(pixel_size_nm) or float(pixel_size_nm) <= 0.0:
        raise ValueError(f"pixel_size_nm must be positive and finite; got {pixel_size_nm!r}.")
    h, w = z.shape
    fx = np.fft.fftfreq(w, d=float(pixel_size_nm))
    fy = np.fft.fftfreq(h, d=float(pixel_size_nm))
    spectrum = np.fft.fft2(z)
    gx = np.fft.ifft2(spectrum * (1j * 2.0 * np.pi * fx)[None, :])
    gy = np.fft.ifft2(spectrum * (1j * 2.0 * np.pi * fy)[:, None])
    return -gx, -gy


def build_off_axis_demodulated_observation(
    signal_counts: np.ndarray,
    reference_counts: np.ndarray | None,
    params: Mapping[str, Any],
    *,
    response_function: Mapping[str, Any],
    object_field_detector: np.ndarray | None = None,
) -> OffAxisDemodulatedObservation:
    """Return the demodulated DHM field and its propagated covariance model."""

    from camera_noise import (
        CameraNoiseConfig,
        detector_mean_frames_for_analysis,
        total_noise_variance_counts,
    )

    params_dict = dict(params or {})
    signal = np.asarray(signal_counts, dtype=float)
    reference = None if reference_counts is None else np.asarray(reference_counts, dtype=float)
    if reference is not None and reference.shape != signal.shape:
        raise ValueError(
            "off-axis demodulated Fisher requires signal/reference frames with "
            f"matching shapes; got {signal.shape} and {reference.shape}."
        )
    signal_mean, reference_mean = detector_mean_frames_for_analysis(
        signal,
        reference,
        params_dict,
    )
    raw_signal = np.asarray(signal_mean, dtype=float)
    if raw_signal.ndim != 2:
        raise ValueError(f"off-axis demodulated Fisher requires a 2D raw frame; got {raw_signal.shape}.")
    response = dict(response_function or {})
    if object_field_detector is None:
        raise ValueError(
            "off-axis demodulated Fisher requires object_field_detector, the "
            "detector-grid complex object/background field used to render the "
            "off-axis reference arm. Without it the +1 sideband is only "
            "a*E_obj*E_sca^*, not the reconstructed field contrast."
        )
    output_normalization = _reconstruction_output_normalization(
        object_field_detector,
        float(response.get("off_axis_reference_amplitude_scale", 1.0)),
        raw_signal.shape,
    )
    input_grid = OFF_AXIS_DEMODULATION_GRID_DETECTOR
    carrier_plan = _carrier_demodulation_plan(
        raw_signal.shape,
        response,
        input_grid=input_grid,
    )
    mask, radius = _sideband_mask(raw_signal.shape, carrier_plan.carrier_bin)
    signal_field = _demodulate_with_sideband(
        raw_signal,
        sideband_mask=mask,
        sideband_shift=carrier_plan.sideband_shift,
        phase_correction=carrier_plan.phase_correction,
    )
    reference_field = (
        np.zeros_like(signal_field)
        if reference_mean is None
        else _demodulate_with_sideband(
            np.asarray(reference_mean, dtype=float),
            sideband_mask=mask,
            sideband_shift=carrier_plan.sideband_shift,
            phase_correction=carrier_plan.phase_correction,
        )
    )
    field = _reconstruct_field_contrast_from_plus_sideband(
        signal_field,
        reference_field,
        output_normalization,
    )

    raw_variance = np.asarray(total_noise_variance_counts(signal, params_dict), dtype=float)
    if raw_variance.shape != raw_signal.shape:
        raise ValueError(
            "off-axis demodulated Fisher raw variance shape "
            f"{raw_variance.shape} does not match raw signal shape {raw_signal.shape}."
        )
    cfg = CameraNoiseConfig.from_params(params_dict)
    raw_row_variance = float(cfg.scan_line_noise_counts) ** 2
    diagonal_summary = _sideband_diagonal_variance_summary(
        raw_variance,
        sideband_mask=mask,
        sideband_shift=carrier_plan.sideband_shift,
        output_normalization=output_normalization,
    )
    noise_model = fourier_sideband_demodulated_noise_model(
        diagonal_variance=diagonal_summary,
        raw_variance=raw_variance,
        sideband_mask=mask,
        sideband_shift=carrier_plan.sideband_shift,
        sideband_phase_correction=carrier_plan.phase_correction,
        sideband_output_normalization=output_normalization,
        sideband_output_conjugate=True,
        raw_row_correlated_variance=raw_row_variance,
        measurement_domain="demodulated_reconstructed_complex_field_contrast",
        signal_units="complex_detector_count_scaled_field",
        noise_variance_units="complex_detector_count_scaled_field_squared",
        status_reason=(
            "off-axis DHM raw count noise propagated through +1 Fourier-sideband "
            "demodulation from the single-shot signal interferogram, fractional "
            "detector-carrier correction, deterministic empty-reference centering, "
            "division by the detector-grid complex object/reference factor, and "
            "the conjugation required by the exp(+iK.r) carrier convention; "
            "diagonal variance is a summary and Fisher uses the real-augmented "
            "sideband covariance operator"
        ),
        context="build_off_axis_demodulated_observation",
    )
    metadata = {
        "off_axis_demodulation_contract": "plus_one_sideband_object_normalized_conjugated_reconstruction_v1",
        "off_axis_demodulation_input_grid": input_grid,
        "off_axis_demodulation_period_basis": "off_axis_fringe_period_detector_px",
        "off_axis_demodulation_period_px": float(
            response.get("off_axis_fringe_period_detector_px", np.nan)
        ),
        "off_axis_demodulated_observable": "reconstructed_complex_object_field_contrast",
        "off_axis_reconstruction_normalization": "conj(centered_plus_one_sideband)/conj(a_E_obj_detector)",
        "off_axis_sideband_conjugated_for_field_convention": True,
        "off_axis_raw_observable": "single_shot_detector_count_interferogram",
        "off_axis_reference_treatment": "deterministic_demodulated_empty_reference_centering",
        "off_axis_noise_source": "raw_signal_interferogram_only",
        "off_axis_raw_row_correlated_variance": raw_row_variance,
        "off_axis_carrier_bin_yx": [
            int(carrier_plan.carrier_bin[0]),
            int(carrier_plan.carrier_bin[1]),
        ],
        "off_axis_carrier_exact_bin_yx": [
            float(carrier_plan.exact_bin_yx[0]),
            float(carrier_plan.exact_bin_yx[1]),
        ],
        "off_axis_carrier_residual_bin_yx": [
            float(carrier_plan.residual_bin_yx[0]),
            float(carrier_plan.residual_bin_yx[1]),
        ],
        "off_axis_carrier_residual_cycles_per_pixel_yx": [
            float(carrier_plan.residual_cycles_per_pixel_yx[0]),
            float(carrier_plan.residual_cycles_per_pixel_yx[1]),
        ],
        "off_axis_sideband_shift_yx": [
            int(carrier_plan.sideband_shift[0]),
            int(carrier_plan.sideband_shift[1]),
        ],
        "off_axis_sideband_radius_bins": float(radius),
        "off_axis_sideband_selected_coefficient_count": int(np.count_nonzero(mask)),
        "off_axis_demodulation_is_linear": True,
        "off_axis_fractional_carrier_phase_correction": True,
        "off_axis_demodulation_noise_covariance": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
        "fisher_observable_preprocessing": "raw_interferogram_plus_one_sideband_demodulated_centered_object_normalized_and_conjugated_to_complex_field_contrast",
        "raw_frame_has_detector_fixed_carrier": True,
        "demodulated_field_has_detector_fixed_carrier": False,
    }
    return OffAxisDemodulatedObservation(
        field_contrast=field,
        raw_interferogram=raw_signal,
        noise_model=noise_model,
        metadata=metadata,
    )


def compute_off_axis_demodulated_fisher_information(
    field_contrast: np.ndarray,
    noise_model: AnalysisNoiseModel,
    pixel_size_nm: float,
    *,
    context: str = "off-axis demodulated lateral Fisher",
) -> tuple[np.ndarray, dict[str, Any]]:
    d_dx0, d_dy0 = _complex_spectral_coordinate_derivatives(field_contrast, pixel_size_nm)
    return compute_fisher_from_complex_fourier_sideband_gradients(
        (d_dx0, d_dy0),
        noise_model,
        context=context,
    )


def compute_off_axis_demodulated_localization_crlb(
    signal_counts: np.ndarray,
    reference_counts: np.ndarray | None,
    params: Mapping[str, Any],
    pixel_size_nm: float,
    *,
    response_function: Mapping[str, Any],
    object_field_detector: np.ndarray | None = None,
) -> tuple[dict[str, Any], OffAxisDemodulatedObservation]:
    observation = build_off_axis_demodulated_observation(
        signal_counts,
        reference_counts,
        params,
        response_function=response_function,
        object_field_detector=object_field_detector,
    )
    fisher, precision_metadata = compute_off_axis_demodulated_fisher_information(
        observation.field_contrast,
        observation.noise_model,
        float(pixel_size_nm),
    )
    derivative_metadata = _localization_derivative_metadata(
        float(pixel_size_nm),
        signal_units=observation.noise_model.signal_units,
        measurement_domain=observation.noise_model.measurement_domain,
        noise_variance_units=observation.noise_model.noise_variance_units,
    )
    field_magnitude = np.abs(observation.field_contrast)
    derivative_metadata.update(
        {
            "derivative_basis": "off_axis_demodulated_complex_spectral_band_limited",
            "lateral_step_note": (
                "FFT spectral derivative of the demodulated off-axis holography "
                "sideband; covariance is propagated from the raw interferogram"
            ),
            "fisher_noise_covariance_model": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
            "boundary_energy_fraction": boundary_energy_fraction(field_magnitude),
            "nyquist_band_fraction": nyquist_band_fraction(field_magnitude),
            "bandlimit_validity_basis": "demodulated_sideband_sampling_and_fft_periodicity_diagnostics",
            **observation.metadata,
            **precision_metadata,
        }
    )
    crlb = _localization_crlb_from_fisher(fisher, derivative_metadata)
    crlb.update(
        {
            "fisher_noise_covariance_model": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
            "analysis_noise_covariance_kind": observation.noise_model.covariance_kind,
            "analysis_noise_status_reason": observation.noise_model.status_reason,
            "safe_for_covariance_fisher_variance": True,
            **observation.metadata,
            **precision_metadata,
        }
    )
    return crlb, observation


def is_off_axis_demodulated_fisher_payload(
    field_contrast: Any,
    noise_model: Any,
) -> bool:
    return (
        np.iscomplexobj(np.asarray(field_contrast))
        and isinstance(noise_model, AnalysisNoiseModel)
        and str(noise_model.covariance_kind) == OFF_AXIS_DEMODULATED_COVARIANCE_KIND
    )


def compute_off_axis_demodulated_localization_crlb_from_field(
    field_contrast: np.ndarray,
    noise_model: AnalysisNoiseModel,
    pixel_size_nm: float,
) -> dict[str, Any]:
    """CRLB for callers that already carry the demodulated DHM field."""

    if not is_off_axis_demodulated_fisher_payload(field_contrast, noise_model):
        raise ValueError(
            "off-axis demodulated CRLB from field requires a complex field and "
            "an AnalysisNoiseModel with covariance_kind="
            f"{OFF_AXIS_DEMODULATED_COVARIANCE_KIND!r}."
        )
    fisher, precision_metadata = compute_off_axis_demodulated_fisher_information(
        field_contrast,
        noise_model,
        float(pixel_size_nm),
        context="off-axis demodulated lateral Fisher from candidate field",
    )
    derivative_metadata = _localization_derivative_metadata(
        float(pixel_size_nm),
        signal_units=noise_model.signal_units,
        measurement_domain=noise_model.measurement_domain,
        noise_variance_units=noise_model.noise_variance_units,
    )
    magnitude = np.abs(np.asarray(field_contrast, dtype=np.complex128))
    derivative_metadata.update(
        {
            "derivative_basis": "off_axis_demodulated_complex_spectral_band_limited",
            "lateral_step_note": (
                "FFT spectral derivative of a caller-supplied demodulated "
                "off-axis holography sideband field"
            ),
            "fisher_noise_covariance_model": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
            "boundary_energy_fraction": boundary_energy_fraction(magnitude),
            "nyquist_band_fraction": nyquist_band_fraction(magnitude),
            "bandlimit_validity_basis": "demodulated_sideband_sampling_and_fft_periodicity_diagnostics",
            "off_axis_demodulated_observable": "reconstructed_complex_object_field_contrast",
            "demodulated_field_has_detector_fixed_carrier": False,
            **precision_metadata,
        }
    )
    crlb = _localization_crlb_from_fisher(fisher, derivative_metadata)
    crlb.update(
        {
            "fisher_noise_covariance_model": OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
            "analysis_noise_covariance_kind": noise_model.covariance_kind,
            "analysis_noise_status_reason": noise_model.status_reason,
            "safe_for_covariance_fisher_variance": True,
            "off_axis_demodulated_observable": "reconstructed_complex_object_field_contrast",
            **precision_metadata,
        }
    )
    return crlb


__all__ = [
    "OFF_AXIS_DEMODULATED_COVARIANCE_KIND",
    "OFF_AXIS_HOLOGRAPHY_MODALITY",
    "OffAxisDemodulatedObservation",
    "build_off_axis_demodulated_observation",
    "compute_off_axis_demodulated_fisher_information",
    "compute_off_axis_demodulated_localization_crlb",
    "compute_off_axis_demodulated_localization_crlb_from_field",
    "is_off_axis_demodulated_fisher_payload",
    "is_off_axis_holography_modality",
]
