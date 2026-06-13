"""qpi imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    coherent_phase_from_reference,
    np,
    reference_vector_for_scattered,
)
from config.runtime import OpticalModeSettings, QpiReadoutSettings
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_PHASE_DISPLAY_COUNTS,
    MODEL_OUTPUT_DOMAIN_PHASE_RADIANS,
    REFERENCE_BASIS_NONE,
    VALUE_FORM_DISPLAY,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)
from stochastic_runtime import rng_from_seed

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
    sample_environment_reference_field_only = True
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "parameters['reference_field_amplitude'] must be positive for "
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
        from camera_noise import QPI_PHASE_LIKELIHOOD_CONTRACT_ID, qpi_phase_noise_components_rad2

        settings = QpiReadoutSettings.from_params(params)
        noise_components = qpi_phase_noise_components_rad2(params)
        response.update(
            kind="quantitative_phase",
            count_scaling_mode="display_phase_offset_counts",
            output_units="radian",
            signal_units="radian",
            display_count_scaling="display_only",
            qpi_visibility=float(noise_components["visibility"]),
            qpi_detected_quanta_per_pixel=float(noise_components["detected_quanta_per_pixel"]),
            qpi_configured_detected_quanta_per_pixel=float(
                noise_components["configured_detected_quanta_per_pixel"]
            ),
            qpi_detected_quanta_exposure_scale=float(
                noise_components["detected_quanta_exposure_scale"]
            ),
            qpi_phase_shot_variance_rad2=float(noise_components["shot_variance_rad2"]),
            qpi_phase_readout_variance_rad2=float(noise_components["readout_variance_rad2"]),
            qpi_phase_variance_rad2=float(noise_components["total_variance_rad2"]),
            qpi_phase_shot_noise_enabled=bool(noise_components["shot_noise_enabled"]),
            qpi_phase_gaussian_noise_enabled=bool(noise_components["gaussian_noise_enabled"]),
            qpi_phase_likelihood_contract_id=QPI_PHASE_LIKELIHOOD_CONTRACT_ID,
            qpi_detected_quanta_basis="uniform_scalar_or_per_frame_detected_quanta_map",
            phase_noise_model="1/(V^2 N(r))+sigma_phi_readout^2",
            qpi_phase_to_count_scale=settings.phase_to_count_scale,
            qpi_phase_noise_std_rad=(
                None
                if settings.phase_noise_std_rad is None
                else float(settings.phase_noise_std_rad)
            ),
            qpi_display_counts_clipped_to_nonnegative=True,
        )
        return response

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
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
        phase_to_count = QpiReadoutSettings.from_params(params).phase_to_count_scale
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=background_final,
            reference_intensity_frame=None,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_PHASE_RADIANS,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_PHASE_DISPLAY_COUNTS,
                value_form=VALUE_FORM_DISPLAY,
                reference_basis=REFERENCE_BASIS_NONE,
                scale=phase_to_count,
                offset=background_final,
                require_nonnegative=False,
            ),
            params=params,
            context="QuantitativePhaseImagingModel.convert_model_output_to_detector_frame",
        )

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
        settings = QpiReadoutSettings.from_params(params)
        phase_to_count = settings.phase_to_count_scale
        from camera_noise import qpi_phase_noise_variance_rad2

        variance = qpi_phase_noise_variance_rad2(counts, params, variance_floor=0.0)
        if not np.any(variance > 0.0):
            return counts.copy()
        generator = rng if rng is not None else rng_from_seed(None, stream="qpi_phase_noise")
        phase_noise = generator.normal(
            loc=0.0,
            scale=np.sqrt(np.maximum(variance, 0.0)),
            size=counts.shape,
        )
        return counts + phase_to_count * phase_noise

__all__ = ['QuantitativePhaseImagingModel']
