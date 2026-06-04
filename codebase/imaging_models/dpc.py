"""dpc imaging model."""

from __future__ import annotations

from ._shared import (
    _optical_pupil_frequency_grid,
    field_intensity,
    is_vectorial_field,
    np,
)
from .coherent_brightfield import CoherentBrightfieldImagingModel
from config.runtime import (
    DpcSettings,
    SamplingGeometry,
)

class DifferentialPhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Differential phase contrast with pupil-domain asymmetric illumination."""

    uses_sample_environment_pattern = True
    _DPC_FULL_VECTOR_DETECTION = "full_vector"

    _DPC_CHANNEL_SCALAR = "two_axis_scalar_asymmetric_illumination"
    _DPC_CHANNEL_VECTORIAL = "vectorial_debye_asymmetric_illumination"
    _DPC_TRANSFER_PUPIL_HALF_PLANE = "pupil_half_plane_intensity"
    _DPC_TRANSFER_PHASE_GRADIENT = "phase_gradient_proxy"

    @classmethod
    def _resolve_dpc_channel_model(cls, params: dict) -> str:
        raw = DpcSettings.from_params(params).channel_model
        if raw in {cls._DPC_CHANNEL_SCALAR, cls._DPC_CHANNEL_VECTORIAL}:
            return raw
        raise ValueError(
            "PARAMS['dpc_channel_model'] must be "
            "'two_axis_scalar_asymmetric_illumination' or "
            f"'vectorial_debye_asymmetric_illumination'; got {raw!r}."
        )

    @classmethod
    def _vectorial_dpc_enabled(cls, params: dict) -> bool:
        return cls._resolve_dpc_channel_model(params) == cls._DPC_CHANNEL_VECTORIAL

    @classmethod
    def _resolve_dpc_transfer_model(cls, params: dict) -> str:
        raw = DpcSettings.from_params(params).transfer_model
        if raw in {cls._DPC_TRANSFER_PUPIL_HALF_PLANE, cls._DPC_TRANSFER_PHASE_GRADIENT}:
            return raw
        raise ValueError(
            "PARAMS['dpc_transfer_model'] must be 'pupil_half_plane_intensity' "
            f"or 'phase_gradient_proxy'; got {raw!r}."
        )

    @staticmethod
    def _dpc_components(field: np.ndarray, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
        phase = np.unwrap(np.unwrap(np.angle(field), axis=0), axis=1)
        dphi_dy, dphi_dx = np.gradient(phase, pixel_size_nm)
        return dphi_dx, dphi_dy

    @staticmethod
    def _dpc_components_vector(
        field: np.ndarray,
        pixel_size_nm: float,
        vectorial_detection_mode: str = "full_vector",
    ) -> tuple[np.ndarray, np.ndarray]:
        E = np.asarray(field, dtype=np.complex128)
        detection_mode = str(vectorial_detection_mode).strip().lower()

        if E.ndim != 3 or E.shape[0] != 3:
            raise ValueError(
                "Vectorial DPC requires field shape (3, H, W) when detector is vectorial; "
                f"got shape {E.shape!r}."
            )

        if detection_mode == "analyzer_x":
            return DifferentialPhaseContrastImagingModel._dpc_components(E[0], pixel_size_nm)
        elif detection_mode == "analyzer_y":
            return DifferentialPhaseContrastImagingModel._dpc_components(E[1], pixel_size_nm)
        elif detection_mode == "incoherent_sum":
            dx_x, dy_x = DifferentialPhaseContrastImagingModel._dpc_components(E[0], pixel_size_nm)
            dx_y, dy_y = DifferentialPhaseContrastImagingModel._dpc_components(E[1], pixel_size_nm)
            return 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
        elif detection_mode == "unpolarized":
            dx_x, dy_x = DifferentialPhaseContrastImagingModel._dpc_components(E[0], pixel_size_nm)
            dx_y, dy_y = DifferentialPhaseContrastImagingModel._dpc_components(E[1], pixel_size_nm)
            return 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
        elif detection_mode == "full_vector":
            dE_dy, dE_dx = np.gradient(E, pixel_size_nm, axis=(1, 2))
            denominator = np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2 + np.abs(E[2]) ** 2
            eps = float(np.finfo(float).eps)
            denominator = np.maximum(denominator, eps)
            dphi_dx = np.imag(np.sum(np.conj(E) * dE_dx, axis=0)) / denominator
            dphi_dy = np.imag(np.sum(np.conj(E) * dE_dy, axis=0)) / denominator
            return dphi_dx, dphi_dy
        raise ValueError(
            "Differential-phase-contrast vectorial backend received "
            "unsupported vectorial_detection_mode "
            f"{vectorial_detection_mode!r}."
        )

    @staticmethod
    def _masked_pupil_intensity(field: np.ndarray, pupil_weight: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, dtype=np.complex128)
        F = np.fft.fft2(arr, axes=(-2, -1))
        filtered = np.fft.ifft2(F * np.asarray(pupil_weight, dtype=float), axes=(-2, -1))
        return field_intensity(filtered)

    @classmethod
    def _asymmetric_pupil_dpc_components(
        cls,
        field: np.ndarray,
        pixel_size_nm: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = tuple(np.asarray(field).shape[-2:])
        FX, FY, rho, _ = _optical_pupil_frequency_grid(shape, pixel_size_nm, params)
        pupil = (rho <= 1.0).astype(float)
        center_x = np.isclose(FX, 0.0)
        center_y = np.isclose(FY, 0.0)
        pos_x = pupil * ((FX > 0.0).astype(float) + 0.5 * center_x.astype(float))
        neg_x = pupil * ((FX < 0.0).astype(float) + 0.5 * center_x.astype(float))
        pos_y = pupil * ((FY > 0.0).astype(float) + 0.5 * center_y.astype(float))
        neg_y = pupil * ((FY < 0.0).astype(float) + 0.5 * center_y.astype(float))
        i_pos_x = cls._masked_pupil_intensity(field, pos_x)
        i_neg_x = cls._masked_pupil_intensity(field, neg_x)
        i_pos_y = cls._masked_pupil_intensity(field, pos_y)
        i_neg_y = cls._masked_pupil_intensity(field, neg_y)
        eps = float(np.finfo(float).eps)
        dpc_x = (i_pos_x - i_neg_x) / np.maximum(i_pos_x + i_neg_x, eps)
        dpc_y = (i_pos_y - i_neg_y) / np.maximum(i_pos_y + i_neg_y, eps)
        return dpc_x, dpc_y

    @classmethod
    def _dpc_signal(cls, field: np.ndarray, pixel_size_nm: float, params: dict) -> tuple[np.ndarray, float]:
        transfer_model = cls._resolve_dpc_transfer_model(params)
        settings = DpcSettings.from_params(params)
        if cls._vectorial_dpc_enabled(params):
            detection_mode = settings.optical.vectorial_detection_mode
            if detection_mode not in {
                "analyzer_x",
                "analyzer_y",
                "incoherent_sum",
                "unpolarized",
                cls._DPC_FULL_VECTOR_DETECTION,
            }:
                raise ValueError(
                    "Differential-phase-contrast vectorial backend requires "
                    "vectorial_detection_mode='analyzer_x', 'analyzer_y', "
                    "'incoherent_sum', 'unpolarized', or 'full_vector'; got "
                    f"{detection_mode!r}."
                )
            optical_backend = settings.optical.optical_field_backend
            if optical_backend != "vectorial_debye":
                raise ValueError(
                    "Differential-phase-contrast vectorial backend requires "
                    f"optical_field_backend='vectorial_debye'; got {optical_backend!r}."
                )
            use_vector_signal = True
        else:
            use_vector_signal = False

        if transfer_model == cls._DPC_TRANSFER_PUPIL_HALF_PLANE:
            arr = np.asarray(field, dtype=np.complex128)
            if use_vector_signal and arr.ndim == 3 and arr.shape[0] == 3:
                if detection_mode == "analyzer_x":
                    signal_field = arr[0]
                    dphi_dx, dphi_dy = cls._asymmetric_pupil_dpc_components(
                        signal_field,
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode == "analyzer_y":
                    signal_field = arr[1]
                    dphi_dx, dphi_dy = cls._asymmetric_pupil_dpc_components(
                        signal_field,
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode in {"incoherent_sum", "unpolarized"}:
                    dx_x, dy_x = cls._asymmetric_pupil_dpc_components(
                        arr[0],
                        pixel_size_nm,
                        params,
                    )
                    dx_y, dy_y = cls._asymmetric_pupil_dpc_components(
                        arr[1],
                        pixel_size_nm,
                        params,
                    )
                    dphi_dx, dphi_dy = 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
                else:
                    dphi_dx, dphi_dy = cls._asymmetric_pupil_dpc_components(
                        arr,
                        pixel_size_nm,
                        params,
                    )
            else:
                dphi_dx, dphi_dy = cls._asymmetric_pupil_dpc_components(
                    arr,
                    pixel_size_nm,
                    params,
                )
        elif use_vector_signal:
            dphi_dx, dphi_dy = cls._dpc_components_vector(
                field,
                pixel_size_nm,
                vectorial_detection_mode=detection_mode,
            )
        else:
            dphi_dx, dphi_dy = cls._dpc_components(field, pixel_size_nm)
        channel = settings.output_channel
        if transfer_model == cls._DPC_TRANSFER_PUPIL_HALF_PLANE:
            gain_x = settings.intensity_gain_x
            gain_y = settings.intensity_gain_y
        else:
            gain_x = settings.phase_gradient_gain_x
            gain_y = settings.phase_gradient_gain_y
        if channel == "x":
            return dphi_dx, gain_x
        if channel == "y":
            return dphi_dy, gain_y
        if channel == "diagonal":
            return (dphi_dx + dphi_dy) / np.sqrt(2.0), 0.5 * (gain_x + gain_y)
        if channel == "magnitude":
            return np.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy), 0.5 * (gain_x + gain_y)
        raise ValueError("dpc_output_channel must be x, y, diagonal, or magnitude.")

    def _intensity_from_total_field(
        self,
        total_field: np.ndarray,
        incident_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        dpc, gain = self._dpc_signal(total_field, pixel_size_px_nm, params)
        return np.maximum(field_intensity(incident_field) * (1.0 + gain * dpc), 0.0)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        if self._vectorial_dpc_enabled(params) and is_vectorial_field(E_sca_total):
            detection_mode = DpcSettings.from_params(params).optical.vectorial_detection_mode
            E_inc = np.zeros_like(E_sca_total, dtype=np.complex128)
            if detection_mode == "analyzer_x":
                E_inc[0, :, :] = self._E_inc_amplitude
            elif detection_mode == "analyzer_y":
                E_inc[1, :, :] = self._E_inc_amplitude
            elif detection_mode == "incoherent_sum":
                E_inc[0, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
                E_inc[1, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
            elif detection_mode == self._DPC_FULL_VECTOR_DETECTION:
                E_inc = self._incident_field_for_scattered(E_sca_total, params)
            elif detection_mode == "unpolarized":
                e_inc_x = np.zeros_like(E_inc)
                e_inc_y = np.zeros_like(E_inc)
                e_inc_x[0, :, :] = self._E_inc_amplitude
                e_inc_y[1, :, :] = self._E_inc_amplitude
                dpc_x, gain_x = self._dpc_signal(
                    e_inc_x + E_sca_total,
                    pixel_size_px_nm,
                    dict(params, vectorial_detection_mode="analyzer_x"),
                )
                dpc_y, gain_y = self._dpc_signal(
                    e_inc_y + E_sca_total,
                    pixel_size_px_nm,
                    dict(params, vectorial_detection_mode="analyzer_y"),
                )
                dpc = 0.5 * (dpc_x + dpc_y)
                gain = 0.5 * (gain_x + gain_y)
                return np.maximum(
                    (self._E_inc_amplitude ** 2) * (1.0 + gain * dpc),
                    0.0,
                )
        else:
            E_inc = self._uniform_field(E_sca_total.shape)
        del pixel_size_px_nm
        return self._intensity_from_total_field(E_inc + E_sca_total, E_inc, params)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        if self._vectorial_dpc_enabled(params) and is_vectorial_field(E_sca_particle):
            detection_mode = DpcSettings.from_params(params).optical.vectorial_detection_mode
            E_inc = np.zeros_like(E_sca_particle, dtype=np.complex128)
            if detection_mode == "analyzer_x":
                E_inc[0, :, :] = self._E_inc_amplitude
            elif detection_mode == "analyzer_y":
                E_inc[1, :, :] = self._E_inc_amplitude
            elif detection_mode == "incoherent_sum":
                E_inc[0, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
                E_inc[1, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
            elif detection_mode == self._DPC_FULL_VECTOR_DETECTION:
                E_inc = self._incident_field_for_scattered(E_sca_particle, params)
            elif detection_mode == "unpolarized":
                e_inc_x = np.zeros_like(E_inc)
                e_inc_y = np.zeros_like(E_inc)
                e_inc_x[0, :, :] = self._E_inc_amplitude
                e_inc_y[1, :, :] = self._E_inc_amplitude
                dpc_x, gain_x = self._dpc_signal(
                    e_inc_x + E_sca_particle,
                    pixel_size_px_nm,
                    dict(params, vectorial_detection_mode="analyzer_x"),
                )
                dpc_y, gain_y = self._dpc_signal(
                    e_inc_y + E_sca_particle,
                    pixel_size_px_nm,
                    dict(params, vectorial_detection_mode="analyzer_y"),
                )
                dpc = 0.5 * (dpc_x + dpc_y)
                gain = 0.5 * (gain_x + gain_y)
                return (self._E_inc_amplitude ** 2) * gain * dpc
        else:
            E_inc = self._uniform_field(E_sca_particle.shape)
        del pixel_size_px_nm
        return (
            self._intensity_from_total_field(E_inc + E_sca_particle, E_inc, params)
            - self._intensity_from_total_field(E_inc, E_inc, params)
        )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        settings = DpcSettings.from_params(params)
        channel_model = self._resolve_dpc_channel_model(params)
        transfer_model = self._resolve_dpc_transfer_model(params)
        if channel_model == self._DPC_CHANNEL_VECTORIAL:
            detection_mode = settings.optical.vectorial_detection_mode
            if detection_mode == "analyzer_x":
                forward_observable = "vectorial Debye analyzer-x pupil half-plane DPC"
            elif detection_mode == "analyzer_y":
                forward_observable = "vectorial Debye analyzer-y pupil half-plane DPC"
            elif detection_mode == "incoherent_sum":
                forward_observable = "vectorial Debye incoherent-sum pupil half-plane DPC"
            elif detection_mode == "unpolarized":
                forward_observable = "vectorial Debye unpolarized average pupil half-plane DPC"
            else:
                forward_observable = "vectorial Debye full-vector pupil half-plane DPC"
            vectorial_detection_mode = settings.optical.vectorial_detection_mode
        else:
            forward_observable = "scalar two-axis pupil half-plane DPC"
            vectorial_detection_mode = "incoherent_sum"
        if transfer_model == self._DPC_TRANSFER_PHASE_GRADIENT:
            forward_observable = forward_observable.replace("pupil half-plane", "phase-gradient proxy")
        response.update(
            kind="asymmetric_illumination_dpc",
            forward_observable=forward_observable,
            dpc_channel_model=channel_model,
            dpc_transfer_model=transfer_model,
            dpc_output_channel=settings.output_channel,
            dpc_vectorial_detection_mode=vectorial_detection_mode,
            intensity_gain_x=float(
                settings.intensity_gain_x
            ),
            intensity_gain_y=float(
                settings.intensity_gain_y
            ),
            phase_gradient_gain_x=float(
                settings.phase_gradient_gain_x
            ),
            phase_gradient_gain_y=float(
                settings.phase_gradient_gain_y
            ),
            dpc_vectorial_backend_enabled=bool(channel_model == self._DPC_CHANNEL_VECTORIAL),
        )
        return response

__all__ = ['DifferentialPhaseContrastImagingModel']
