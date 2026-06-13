"""zernike phase imaging model."""

from __future__ import annotations

from ._shared import (
    _optical_pupil_frequency_grid,
    field_intensity,
    is_vectorial_field,
    np,
)
from .coherent_brightfield import CoherentBrightfieldImagingModel
from config.runtime import SamplingGeometry, ZernikePhaseSettings

class ZernikePhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Zernike phase contrast using a pupil-domain phase-ring transfer."""

    uses_sample_environment_pattern = True

    @staticmethod
    def _phase_ring_field(field: np.ndarray, params: dict) -> np.ndarray:
        settings = ZernikePhaseSettings.from_params(params)
        arr = np.asarray(field, dtype=np.complex128)
        H, W = arr.shape[-2:]
        if settings.model == "pupil_phase_ring":
            pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
            _, _, rho, _ = _optical_pupil_frequency_grid((H, W), pixel_size_px_nm, params)
        else:
            fy = np.fft.fftfreq(H)
            fx = np.fft.fftfreq(W)
            FX, FY = np.meshgrid(fx, fy, indexing="xy")
            rho = np.sqrt(FX * FX + FY * FY) / max(float(np.max(np.sqrt(FX * FX + FY * FY))), 1e-30)
        F = np.fft.fft2(arr, axes=(-2, -1))
        ring = (rho >= settings.inner_fraction) & (rho <= settings.outer_fraction)
        transfer = np.ones_like(rho, dtype=np.complex128)
        transfer[ring] = settings.amplitude * np.exp(1j * settings.shift_rad)
        if is_vectorial_field(arr):
            F_shifted = F * transfer[np.newaxis, :, :]
        else:
            F_shifted = F * transfer
        shifted = np.fft.ifft2(F_shifted, axes=(-2, -1))
        return settings.bias * arr + settings.gain * (shifted - arr)

    def _intensity_from_total_field(
        self,
        total_field: np.ndarray,
        incident_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        field = self._phase_ring_field(total_field, params)
        return field_intensity(field)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._incident_field_for_scattered(
            E_sca_total,
            params,
            base_field=background_field,
        )
        return self._intensity_from_total_field(E_inc + E_sca_total, E_inc, params)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._incident_field_for_scattered(
            E_sca_particle,
            params,
            base_field=background_field,
        )
        return (
            self._intensity_from_total_field(E_inc + E_sca_particle, E_inc, params)
            - self._intensity_from_total_field(E_inc, E_inc, params)
        )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        settings = ZernikePhaseSettings.from_params(params)
        response.update(
            kind="zernike_phase_ring",
            zernike_model=settings.model,
            zernike_phase_ring_coordinate_system=settings.coordinate_system,
            phase_ring_inner_fraction=settings.inner_fraction,
            phase_ring_outer_fraction=settings.outer_fraction,
            phase_shift_rad=settings.shift_rad,
            phase_ring_amplitude=settings.amplitude,
            phase_ring_gain=settings.gain,
            phase_bias=settings.bias,
        )
        return response

__all__ = ['ZernikePhaseContrastImagingModel']
