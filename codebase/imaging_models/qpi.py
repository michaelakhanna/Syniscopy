"""qpi imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    coherent_phase_from_reference,
    np,
    reference_vector_for_scattered,
)
from config.runtime import CountBudgetSettings, OpticalModeSettings, param_value

class QuantitativePhaseImagingModel(ImagingModel):
    """
    Quantitative phase imaging (QPI) contrast model.

    Rather than photon counts, the detector-domain quantity here is the
    optical phase of the transmitted (or reflected) field, expressed in
    radians.  This corresponds physically to a phase-shifting interferometer
    or an off-axis holographic setup that has already been demodulated.

    The computed phase is the argument of the complex field normalised by
    the reference (so the idle frame is zero-phase everywhere):

        φ(r, t) = arg( E_ref(r) + E_sca(r, t) ) − arg( E_ref(r) )
                = arg( 1 + E_sca(r, t) / E_ref(r) )

    Per-particle contrast is the same quantity computed using only that
    particle's scattered field.  Both outputs are real 2D float arrays in
    radians, and may be negative.  The returned arrays are wrapped to
    (−π, π] by ``np.angle``; 2D unwrap is the caller's responsibility and
    is *not* needed for visualisation.

    Small-signal approximation:
        For |E_sca| ≪ |E_ref|, φ ≈ Im(E_sca / E_ref), which is the quantity
        typically reported in nanoscale biology / cell thickness mapping.

    Validation:
        reference_field_amplitude must be > 0 (phase is only defined
        relative to a nonzero reference).
    """

    output_type = "phase"
    uses_sample_environment_pattern = True  # Phase is referenced to the structured coherent background field.
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "QuantitativePhaseImagingModel (imaging_model='quantitative_phase'). "
                "Phase is only defined relative to a nonzero reference field."
            )

    @staticmethod
    def _phase(E_sum: np.ndarray, E_ref: np.ndarray) -> np.ndarray:
        """Compute arg(E_sum) − arg(E_ref), wrapped to (−π, π]."""
        return coherent_phase_from_reference(E_sum, E_ref)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Full-frame phase: arg(E_ref + E_sca_total) − arg(E_ref), radians.

        This method implements the shared imaging-model entry point.
        ``output_type = "phase"`` declares that the return value is a phase map.
        """
        E_ref = reference_vector_for_scattered(background_field, E_sca_total, params)
        return self._phase(E_ref + E_sca_total, E_ref)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Per-particle phase shift: arg(E_ref + E_sca_i) − arg(E_ref).
        """
        E_ref = reference_vector_for_scattered(background_field, E_sca_particle, params)
        return self._phase(E_ref + E_sca_particle, E_ref)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        phase_noise = param_value(params, 'qpi_phase_noise_std_rad')
        visibility = float(param_value(params, 'qpi_visibility'))
        detected_quanta_raw = param_value(params, 'qpi_detected_quanta_per_pixel')
        if detected_quanta_raw is None:
            detected_quanta_raw = CountBudgetSettings.from_params(params).background_intensity
        configured_detected_quanta = float(detected_quanta_raw)
        exposure_scale = float(params.get("_exposure_signal_scale", 1.0))
        detected_quanta = configured_detected_quanta * exposure_scale
        readout_variance = 0.0 if phase_noise is None else float(phase_noise) ** 2
        shot_variance = (
            1.0 / (visibility * visibility * detected_quanta)
            if visibility > 0.0 and detected_quanta > 0.0
            else float("inf")
        )
        response.update(
            kind="quantitative_phase",
            count_scaling_mode="display_phase_offset_counts",
            output_units="radian",
            signal_units="radian",
            display_count_scaling="display_only",
            qpi_visibility=visibility,
            qpi_detected_quanta_per_pixel=detected_quanta,
            qpi_configured_detected_quanta_per_pixel=configured_detected_quanta,
            qpi_detected_quanta_exposure_scale=exposure_scale,
            qpi_phase_readout_variance_rad2=readout_variance,
            qpi_phase_variance_rad2=float(shot_variance + readout_variance),
            phase_noise_model="1/(V^2 nQ)+sigma_phi_readout^2",
            qpi_phase_to_count_scale=CountBudgetSettings.from_params(params).qpi_phase_to_count_scale,
            qpi_phase_noise_std_rad=(
                None if phase_noise is None else float(phase_noise)
            ),
            qpi_display_counts_clipped_to_nonnegative=True,
        )
        return response

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Convert a demodulated phase map into a count-like video image.

        QPI's physical output is phase in radians, not optical intensity, so
        the base interferometric |E|^2-to-count scaling is invalid here. For
        rendered videos we store phase contrast around the detector background
        with a configurable radians-to-counts display scale. The CRLB and mask
        paths continue to use the actual phase contrast image.
        """
        del E_ref_intensity_final
        phase_to_count = CountBudgetSettings.from_params(params).qpi_phase_to_count_scale
        return np.asarray(background_final, dtype=float) + phase_to_count * np.asarray(intensity, dtype=float)

    def compute_noise(
        self,
        frame_counts: np.ndarray,
        params: dict,
        rng: np.random.Generator | None = None,
        *,
        detector_noise_runtime=None,
    ) -> np.ndarray:
        """Apply QPI phase-domain noise, then return display-count values."""
        del detector_noise_runtime
        counts = np.asarray(frame_counts, dtype=float)
        settings = CountBudgetSettings.from_params(params)
        phase_to_count = settings.qpi_phase_to_count_scale
        from camera_noise import qpi_phase_noise_variance_rad2

        variance = qpi_phase_noise_variance_rad2(counts, params)
        generator = rng if rng is not None else np.random.default_rng()
        phase_noise = generator.normal(
            loc=0.0,
            scale=np.sqrt(np.maximum(variance, 0.0)),
            size=counts.shape,
        )
        return counts + phase_to_count * phase_noise

__all__ = ['QuantitativePhaseImagingModel']
