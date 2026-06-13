"""off axis holography imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    field_intensity,
    np,
    reference_vector_for_scattered,
)
from config.runtime import (
    OffAxisHolographySettings,
    OpticalModeSettings,
    SamplingGeometry,
)
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
    MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
    REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY,
    VALUE_FORM_ABSOLUTE,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)

class OffAxisHolographyImagingModel(ImagingModel):
    """
    Off-axis digital holographic microscopy (DHM).

    A tilted copy of the structured object/reference background is introduced
    at angle θ relative to the sample beam, producing a detector fringe pattern:

        I(r, t) = | E_obj(r) + E_sca(r, t)
                    + a E_obj(r) · e^{i K · r} |²

    where K = (2π / T) · (cos θ, sin θ) is the tilt wavevector, T is the
    fringe period in pixels, θ is the fringe orientation in radians, and a is
    the reference-arm amplitude ratio.

    Off-axis holography's distinguishing feature: a single raw frame
    encodes the full complex field of the sample, recoverable by Fourier-
    domain bandpass-demodulation around K.  This is what makes it appealing
    for dynamic / single-shot QPI of live samples.  We render the raw
    fringe frame.  Lateral Fisher diagnostics demodulate this raw frame through
    the +1 Fourier sideband and localize on the reconstructed complex field,
    because the raw detector carrier is fixed in detector coordinates.

    Parameters (all taken from parameters with nominal defaults):
        off_axis_fringe_period_px   (default 10.0)
        off_axis_fringe_angle_rad   (default 0.0, fringes run along y)
        off_axis_reference_amplitude_scale (default 1.0)

    Validation:
        reference_field_amplitude must be > 0.
        off_axis_fringe_period_px must be >= 2 (so K is within Nyquist).
    """

    output_type = "fringe"
    supports_spectral_channels = True
    uses_sample_environment_pattern = True  # The tilted reference is derived from the structured background field.
    sample_environment_reference_field_only = True
    # The model renders a raw carrier interferogram, but the lateral Fisher
    # observable is the demodulated reconstructed field owned by
    # fisher.dhm_demodulated, not an x/y rerender exception.
    stationary_lateral_fisher_safe_for_single_uniform_scene = True
    has_detector_fixed_lateral_carrier = False
    requires_rerendered_lateral_fisher = False
    raw_interferogram_has_detector_fixed_carrier = True

    def __init__(self, params: dict) -> None:
        settings = OffAxisHolographySettings.from_params(params)
        E_ref_amplitude = settings.optical.reference_field_amplitude
        if not np.isfinite(E_ref_amplitude) or E_ref_amplitude <= 0.0:
            raise ValueError(
                "parameters['reference_field_amplitude'] must be positive for "
                "OffAxisHolographyImagingModel (imaging_model='off_axis_holography')."
            )
        self._reference_amplitude = float(E_ref_amplitude)
        self._reference_amplitude_scale = settings.reference_amplitude_scale
        self._period_detector_px = settings.fringe_period_detector_px
        oversampling_factor = float(SamplingGeometry.from_params(params).psf_oversampling_factor)
        if not np.isfinite(oversampling_factor) or oversampling_factor <= 0.0:
            raise ValueError(
                "psf_oversampling_factor must be finite and positive for "
                "OffAxisHolographyImagingModel; got "
                f"{oversampling_factor}."
            )
        self._period_canvas_px = self._period_detector_px * oversampling_factor
        if (
            not np.isfinite(self._period_canvas_px)
            or self._period_canvas_px <= 0.0
            or not np.isfinite(2.0 * np.pi / self._period_canvas_px)
        ):
            raise ValueError(
                "off-axis carrier period on the model canvas must produce a "
                "finite carrier wavevector; got "
                f"{self._period_canvas_px}."
            )
        self._angle_rad = settings.fringe_angle_rad
        # Lazily-initialised fringe phase cache keyed by shape, since the
        # array is identical across frames.
        self._tilt_phase_cache: dict = {}

    def _tilt_field(self, shape: tuple) -> np.ndarray:
        """Return a unit-amplitude carrier e^{i K·r} of shape ``shape``."""
        spatial_shape = tuple(shape[-2:])
        cached = self._tilt_phase_cache.get(spatial_shape)
        if cached is not None:
            return cached
        H, W = spatial_shape
        # Pixel grid: row 0 at y=0, col 0 at x=0.
        yy, xx = np.meshgrid(
            np.arange(H, dtype=float),
            np.arange(W, dtype=float),
            indexing="ij",
        )
        K = 2.0 * np.pi / self._period_canvas_px
        phase = K * (xx * np.cos(self._angle_rad) + yy * np.sin(self._angle_rad))
        carrier = np.exp(1j * phase)
        self._tilt_phase_cache[spatial_shape] = carrier
        return carrier

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Off-axis camera frame:

            |E_obj + E_sca_total + a E_obj · exp(iK·r)|².

        The unscattered object/background field remains in the signal frame and
        its tilted copy forms the off-axis reference, so an empty scene has
        carrier fringes as in a raw off-axis hologram. Downstream single-frame
        contrast subtracts the no-particle reference frame when an analysis
        contrast image is requested.
        """
        carrier = self._tilt_field(E_sca_total.shape)
        object_field = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128),
            E_sca_total,
            params,
        )
        reference_field = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128)
            * self._reference_amplitude_scale
            * carrier,
            E_sca_total,
            params,
        )
        return field_intensity(object_field + E_sca_total + reference_field)

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Scale raw off-axis interferograms to detector counts.

        The empty-scene interferogram is
        ``|E_obj + a E_obj exp(iK.r)|^2``. Its carrier-averaged intensity is
        ``|E_obj|^2 * (1 + a^2)`` for reference-arm amplitude ratio ``a``.
        Dividing by that factor keeps ``background_intensity`` as the mean
            empty-scene count level while preserving the raw carrier modulation.
        """
        del params
        reference_scale = float(self._reference_amplitude_scale)
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=background_final,
            reference_intensity_frame=E_ref_intensity_final,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
                value_form=VALUE_FORM_ABSOLUTE,
                reference_basis=REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY,
                reference_scale=1.0 + reference_scale * reference_scale,
            ),
            params=None,
            context="OffAxisHolographyImagingModel.convert_model_output_to_detector_frame",
        )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            {
                "off_axis_fringe_period_detector_px": self._period_detector_px,
                "off_axis_fringe_period_canvas_px": self._period_canvas_px,
                "off_axis_fringe_angle_rad": self._angle_rad,
                "off_axis_reference_amplitude_scale": self._reference_amplitude_scale,
                "off_axis_carrier_units": "model_canvas_pixels",
                "observable_subtype": "off_axis_fringe_interferogram",
                "off_axis_frame_contract": "raw_object_reference_carrier_interferogram",
                "off_axis_reference_arm_source": "structured_background_field",
                "off_axis_empty_scene_contains_carrier_fringes": True,
                "off_axis_count_scaling": "empty_interferogram_mean_preserving",
                "raw_interferogram_has_detector_fixed_carrier": True,
                "lateral_fisher_observable": "demodulated_reconstructed_complex_field",
                "lateral_fisher_demodulation": "plus_one_fourier_sideband_to_baseband_then_object_normalized_conjugation",
                "lateral_fisher_reconstruction": "conj(centered_plus_one_sideband)/conj(a_E_obj_detector)",
                "lateral_fisher_noise_covariance_model": "fourier_sideband_demodulated_complex_field",
                "stationary_lateral_fisher_safe_for_single_uniform_scene": True,
                "has_detector_fixed_lateral_carrier": False,
                "requires_rerendered_lateral_fisher": False,
            }
        )
        return response

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Per-particle fringe contrast:

            |E_obj + E_sca_i + a E_obj exp(iK·r)|²
            − |E_obj + a E_obj exp(iK·r)|².
        """
        carrier = self._tilt_field(E_sca_particle.shape)
        object_field = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128),
            E_sca_particle,
            params,
        )
        reference_field = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128)
            * self._reference_amplitude_scale
            * carrier,
            E_sca_particle,
            params,
        )
        empty_frame = object_field + reference_field
        return field_intensity(empty_frame + E_sca_particle) - field_intensity(empty_frame)

__all__ = ['OffAxisHolographyImagingModel']
