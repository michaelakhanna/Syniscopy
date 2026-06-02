"""coherent darkfield imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    _mean_normalized_map,
    field_intensity,
    np,
)

class CoherentDarkFieldImagingModel(ImagingModel):
    """
    Dark-field (reference-free) imaging model.

    No reference beam reaches the detector.  The signal is purely the
    scattered intensity:

        Intensity  = |E_sca_total|²
        Contrast_i = |E_sca_i|²

    Note: ``PARAMS['reference_field_amplitude']`` is ignored in this mode: the
    reference field has no role in the dark-field forward model.
    """

    uses_sample_environment_pattern = True  # Patterned substrates scatter into the dark-field stop.
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        # Dark-field accepts the shared constructor signature used by the
        # imaging-model factory.
        self._field_gain = float(params.get("dark_field_field_gain", 1.0))
        if self._field_gain <= 0.0:
            raise ValueError("PARAMS['dark_field_field_gain'] must be positive.")

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field intensity: |E_sca_total|².

        ``background_field`` is accepted as part of the shared imaging-model
        interface and is not used by dark-field intensity.
        """
        return field_intensity(self._field_gain * E_sca_total)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field per-particle contrast: |E_sca_i|².

        ``background_field`` is not used.
        """
        return field_intensity(self._field_gain * E_sca_particle)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field intensity-to-counts conversion.

        Physical rationale
        ------------------
        In dark-field there is no reference beam to divide by. The base-class
        count conversion is ill-conditioned when ``|E_ref|^2`` is small, which
        is the dark-field regime. This model converts the dimensionless
        |E_sca|^2 to photon counts by multiplying by an illumination-level
        scale and adding a detector / stray-light pedestal.

        Why a pedestal is required
        --------------------------
        Far from any particle ``intensity = |E_sca|^2`` is exactly zero.
        Without a pedestal the downstream noise pipeline does
        ``Poisson(0) = 0`` plus ``Gaussian(0, sigma_read)``, and the
        ``np.clip(., 0, max).astype(uint16)`` step at the end of the
        rendering loop half-clips the Gaussian to zero, producing a
        visually noise-free black background that does not match real
        dark-field detectors.  A small pedestal corresponds to the physical
        reality of dark-field imaging --- residual stray light reaching the
        detector, plus camera dark current --- and gives read noise a
        non-zero baseline to fluctuate around.

        Parameter resolution
        --------------------
        The illumination-level scale is taken from
        ``PARAMS['dark_field_illumination_count']`` if set, and otherwise
        falls back to ``PARAMS['background_intensity']`` (which is the
        count-domain reference-beam brightness used by the other modalities,
        so in the default configuration the dark-field peak will land at a
        comparable fraction of the camera's dynamic range to the other
        modalities' reference-beam intensity).

        The pedestal is taken from ``PARAMS['dark_field_background_count']``.
        The default is zero, which preserves the ideal zero-baseline dark-field
        model; callers can set a positive pedestal to represent stray light or
        dark current.

        Result
        ------
        - Away from the particle: output is approximately the pedestal,
          so read + shot noise is visible at the expected level.
        - Near the particle: the peak |E_sca|^2 times the illumination
          count adds on top of the pedestal, producing a proportional
          bright spot.
        - Shot noise is applied downstream by the camera-noise model using the
          returned count values as Poisson rates.
        """
        illumination_count = float(params.get(
            "dark_field_illumination_count",
            float(params.get("background_intensity", 1.0)),
        ))
        background_count = float(params.get(
            "dark_field_background_count",
            0.0,
        ))
        return illumination_count * intensity + background_count

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        """Coherent dark-field uses the shared scalar incident-field interface."""
        return super().illumination_field(shape, params)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="coherent_dark_field_scattered_intensity",
            zero_order_reference_blocked=True,
        )
        return response

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        del background_field
        if sample_environment is None:
            return intensity
        edge = sample_environment.substrate.topography_gradient()
        edge = _mean_normalized_map(edge + 1e-12) - 1.0
        gain = float(params.get("dark_field_sample_environment_edge_gain", 0.02))
        pedestal = float(params.get("dark_field_sample_environment_scatter_pedestal", 0.0))
        return np.maximum(intensity + gain * np.maximum(edge, 0.0) + pedestal, 0.0)

__all__ = ['CoherentDarkFieldImagingModel']
