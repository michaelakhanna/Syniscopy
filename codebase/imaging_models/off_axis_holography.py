"""off axis holography imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    field_intensity,
    np,
    reference_vector_for_scattered,
)
from config.runtime import (
    OpticalModeSettings,
    SamplingGeometry,
    param_value,
)

class OffAxisHolographyImagingModel(ImagingModel):
    """
    Off-axis digital holographic microscopy (DHM).

    A tilted reference beam is introduced at angle θ relative to the sample
    beam, producing a detector fringe pattern:

        I(r, t) = | E_ref(r) · e^{i K · r} + E_sca(r, t) |²
                = |E_ref|² + |E_sca|²
                  + 2 · Re( E_ref*(r) · e^{-i K·r} · E_sca(r, t) )

    where K = (2π / T) · (cos θ, sin θ) is the tilt wavevector, T is the
    fringe period in pixels, and θ is the fringe orientation in radians.

    Off-axis holography's distinguishing feature: a single raw frame
    encodes the full complex field of the sample, recoverable by Fourier-
    domain bandpass-demodulation around K.  This is what makes it appealing
    for dynamic / single-shot QPI of live samples.  We render the raw
    fringe frame; demodulation is a post-processing step outside the
    simulator's scope.

    Parameters (all taken from PARAMS with nominal defaults):
        off_axis_fringe_period_px   (default 10.0)
        off_axis_fringe_angle_rad   (default 0.0, fringes run along y)

    Validation:
        reference_field_amplitude must be > 0.
        off_axis_fringe_period_px must be >= 2 (so K is within Nyquist).
    """

    output_type = "fringe"
    supports_spectral_channels = True
    uses_sample_environment_pattern = True  # Off-axis DHM has a coherent reference arm that can carry substrate structure.

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        if not np.isfinite(E_ref_amplitude) or E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "OffAxisHolographyImagingModel (imaging_model='off_axis_holography')."
            )
        self._period_detector_px = float(param_value(params, "off_axis_fringe_period_px"))
        if not np.isfinite(self._period_detector_px) or self._period_detector_px < 2.0:
            raise ValueError(
                "off_axis_fringe_period_px must be >= 2.0 (Nyquist); got "
                f"{self._period_detector_px}."
            )
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
        self._angle_rad = float(param_value(params, "off_axis_fringe_angle_rad"))
        if not np.isfinite(self._angle_rad):
            raise ValueError(
                "off_axis_fringe_angle_rad must be finite for "
                "OffAxisHolographyImagingModel; got "
                f"{self._angle_rad}."
            )
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
        Off-axis fringe frame: |E_ref · e^{iK·r} + E_sca_total|².
        """
        carrier = self._tilt_field(E_sca_total.shape)
        E_ref = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128) * carrier,
            E_sca_total,
            params,
        )
        return field_intensity(E_ref + E_sca_total)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            {
                "off_axis_fringe_period_detector_px": self._period_detector_px,
                "off_axis_fringe_period_canvas_px": self._period_canvas_px,
                "off_axis_fringe_angle_rad": self._angle_rad,
                "off_axis_carrier_units": "model_canvas_pixels",
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
        Per-particle fringe contrast: |E_ref · e^{iK·r} + E_sca_i|² − |E_ref|².
        """
        carrier = self._tilt_field(E_sca_particle.shape)
        E_ref = reference_vector_for_scattered(
            np.asarray(background_field, dtype=np.complex128) * carrier,
            E_sca_particle,
            params,
        )
        ref_intensity = field_intensity(E_ref)
        return field_intensity(E_ref + E_sca_particle) - ref_intensity

__all__ = ['OffAxisHolographyImagingModel']
