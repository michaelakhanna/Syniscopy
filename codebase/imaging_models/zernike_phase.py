"""zernike phase imaging model."""

from __future__ import annotations

from ._shared import (
    _optical_pupil_frequency_grid,
    coherent_phase_from_reference,
    field_intensity,
    is_vectorial_field,
    np,
)
from .coherent_brightfield import CoherentBrightfieldImagingModel

class ZernikePhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Zernike phase contrast using a pupil-domain phase-ring transfer."""

    uses_sample_environment_pattern = True

    @staticmethod
    def _phase_ring_field(field: np.ndarray, params: dict) -> np.ndarray:
        model = str(params.get("zernike_model", "pupil_phase_ring")).strip().lower()
        if model == "legacy_scalar_phase_proxy":
            arr = np.asarray(field, dtype=np.complex128)
            if is_vectorial_field(arr):
                raise ValueError(
                    "zernike_model='legacy_scalar_phase_proxy' is a scalar phase "
                    "proxy and cannot consume a full-vector field. Use "
                    "zernike_model='pupil_phase_ring' or a scalar/analyzer "
                    "vectorial_detection_mode."
                )
            return arr
        if model not in {"pupil_phase_ring", "fourier_phase_ring_proxy"}:
            raise ValueError(
                "PARAMS['zernike_model'] must be 'pupil_phase_ring', "
                "'fourier_phase_ring_proxy', or "
                f"'legacy_scalar_phase_proxy'; got {model!r}."
            )
        arr = np.asarray(field, dtype=np.complex128)
        H, W = arr.shape[-2:]
        if model == "pupil_phase_ring":
            pixel_size_px_nm = float(params["pixel_size_nm"]) / max(
                float(params.get("psf_oversampling_factor", 1)),
                1.0,
            )
            _, _, rho, _ = _optical_pupil_frequency_grid((H, W), pixel_size_px_nm, params)
        else:
            fy = np.fft.fftfreq(H)
            fx = np.fft.fftfreq(W)
            FX, FY = np.meshgrid(fx, fy, indexing="xy")
            rho = np.sqrt(FX * FX + FY * FY) / max(float(np.max(np.sqrt(FX * FX + FY * FY))), 1e-30)
        inner = float(params.get("zernike_phase_ring_inner_fraction", 0.0))
        outer = float(params.get("zernike_phase_ring_outer_fraction", 0.15))
        shift = float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0))
        amplitude = float(params.get("zernike_phase_ring_amplitude", 1.0))
        if inner < 0.0 or outer <= inner or outer > 1.0:
            raise ValueError(
                "Zernike phase ring fractions must satisfy 0 <= inner < outer <= 1."
            )
        F = np.fft.fft2(arr, axes=(-2, -1))
        ring = (rho >= inner) & (rho <= outer)
        F_shifted = F.copy()
        if is_vectorial_field(arr):
            F_shifted[:, ring] *= amplitude * np.exp(1j * shift)
        else:
            F_shifted[ring] *= amplitude * np.exp(1j * shift)
        return np.fft.ifft2(F_shifted, axes=(-2, -1))

    def _intensity_from_total_field(
        self,
        total_field: np.ndarray,
        incident_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        if str(params.get("zernike_model", "pupil_phase_ring")).strip().lower() == "legacy_scalar_phase_proxy":
            if is_vectorial_field(total_field):
                raise ValueError(
                    "zernike_model='legacy_scalar_phase_proxy' cannot be used with "
                    "vectorial_detection_mode='full_vector'."
                )
            phase = coherent_phase_from_reference(total_field, incident_field)
            gain = float(params.get("zernike_phase_ring_gain", 0.35))
            phase_shift = float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0))
            bias = float(params.get("zernike_phase_bias", 1.0))
            phase_gain = gain * np.sin(phase_shift)
            return np.maximum((self._E_inc_amplitude ** 2) * (bias + phase_gain * phase), 0.0)
        field = self._phase_ring_field(total_field, params)
        return field_intensity(field)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._incident_field_for_scattered(E_sca_total, params)
        return self._intensity_from_total_field(E_inc + E_sca_total, E_inc, params)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._incident_field_for_scattered(E_sca_particle, params)
        return (
            self._intensity_from_total_field(E_inc + E_sca_particle, E_inc, params)
            - self._intensity_from_total_field(E_inc, E_inc, params)
        )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="zernike_phase_ring",
            zernike_model=str(params.get("zernike_model", "pupil_phase_ring")),
            zernike_phase_ring_coordinate_system=(
                "objective_pupil_na_over_wavelength"
                if str(params.get("zernike_model", "pupil_phase_ring")).strip().lower() == "pupil_phase_ring"
                else "fft_nyquist_normalized"
            ),
            phase_ring_inner_fraction=float(params.get("zernike_phase_ring_inner_fraction", 0.0)),
            phase_ring_outer_fraction=float(params.get("zernike_phase_ring_outer_fraction", 0.15)),
            phase_shift_rad=float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0)),
            phase_ring_amplitude=float(params.get("zernike_phase_ring_amplitude", 1.0)),
        )
        return response

__all__ = ['ZernikePhaseContrastImagingModel']
