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
from vectorial_detection_contracts import (
    VALID_VECTORIAL_DETECTION_MODES,
    VECTORIAL_DETECTION_ANALYZER_X,
    VECTORIAL_DETECTION_ANALYZER_Y,
    VECTORIAL_DETECTION_FULL_VECTOR,
    VECTORIAL_DETECTION_INCOHERENT_SUM,
    VECTORIAL_DETECTION_UNPOLARIZED,
    vectorial_detection_contract_for_mode,
)

class DifferentialPhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Differential phase contrast with illumination-side and detector-side modes."""

    uses_sample_environment_pattern = True
    _DPC_FULL_VECTOR_DETECTION = VECTORIAL_DETECTION_FULL_VECTOR
    _DPC_INCOHERENT_DETECTION_MODES = {
        VECTORIAL_DETECTION_INCOHERENT_SUM,
        VECTORIAL_DETECTION_UNPOLARIZED,
    }

    _DPC_CHANNEL_SCALAR_SPLIT = "two_axis_scalar_split_pupil_detection"
    _DPC_CHANNEL_VECTORIAL_SPLIT = "vectorial_debye_split_pupil_detection"
    _DPC_CHANNEL_SCALAR_ASYMMETRIC = "two_axis_scalar_asymmetric_illumination"
    _DPC_CHANNEL_VECTORIAL_ASYMMETRIC = "vectorial_debye_asymmetric_illumination"
    _DPC_TRANSFER_ASYMMETRIC_ILLUMINATION = "asymmetric_illumination"
    _DPC_TRANSFER_SPLIT_PUPIL_DETECTION = "split_pupil_detection"
    _DPC_TRANSFER_PHASE_GRADIENT = "phase_gradient_proxy"

    @classmethod
    def _resolve_dpc_channel_model(cls, params: dict) -> str:
        raw = DpcSettings.from_params(params).channel_model
        if raw in {
            cls._DPC_CHANNEL_SCALAR_SPLIT,
            cls._DPC_CHANNEL_VECTORIAL_SPLIT,
            cls._DPC_CHANNEL_SCALAR_ASYMMETRIC,
            cls._DPC_CHANNEL_VECTORIAL_ASYMMETRIC,
        }:
            return raw
        raise ValueError(
            "parameters['dpc_channel_model'] must be "
            "'two_axis_scalar_asymmetric_illumination', "
            "'vectorial_debye_asymmetric_illumination', "
            "'two_axis_scalar_split_pupil_detection', or "
            f"'vectorial_debye_split_pupil_detection'; got {raw!r}."
        )

    @classmethod
    def _vectorial_dpc_enabled(cls, params: dict) -> bool:
        return cls._resolve_dpc_channel_model(params) in {
            cls._DPC_CHANNEL_VECTORIAL_SPLIT,
            cls._DPC_CHANNEL_VECTORIAL_ASYMMETRIC,
        }

    @classmethod
    def _resolve_dpc_transfer_model(cls, params: dict) -> str:
        raw = DpcSettings.from_params(params).transfer_model
        if raw in {
            cls._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION,
            cls._DPC_TRANSFER_SPLIT_PUPIL_DETECTION,
            cls._DPC_TRANSFER_PHASE_GRADIENT,
        }:
            return raw
        raise ValueError(
            "parameters['dpc_transfer_model'] must be 'asymmetric_illumination', "
            "'split_pupil_detection', or 'phase_gradient_proxy'; "
            f"got {raw!r}."
        )

    @staticmethod
    def _hex_disc_samples(n_target: int) -> np.ndarray:
        if n_target <= 1:
            return np.zeros((1, 2), dtype=float)
        n_rings = max(1, int(round(np.sqrt(max(n_target, 1) / np.pi))))
        pts = [(0.0, 0.0)]
        for ring in range(1, n_rings + 1):
            radius = ring / n_rings
            n_in_ring = 6 * ring
            for k in range(n_in_ring):
                theta = 2.0 * np.pi * k / n_in_ring
                pts.append((radius * np.cos(theta), radius * np.sin(theta)))
        return np.asarray(pts, dtype=float)

    @classmethod
    def _half_disc_source_points(
        cls,
        params: dict,
        *,
        axis: str,
        sign: float,
    ) -> np.ndarray:
        pts, weights = cls._half_disc_source_quadrature(params, axis=axis, sign=sign)
        return pts[weights > 0.0]

    @classmethod
    def _half_disc_source_quadrature(
        cls,
        params: dict,
        *,
        axis: str,
        sign: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        settings = DpcSettings.from_params(params)
        n_target = settings.source_samples
        sigma = settings.illumination_sigma
        pts = sigma * cls._hex_disc_samples(n_target)
        axis_index = 0 if axis == "x" else 1
        signed_coord = float(sign) * pts[:, axis_index]
        positive = signed_coord > 1e-12
        boundary = np.abs(pts[:, axis_index]) <= 1e-12
        selected_mask = positive | boundary
        if not np.any(selected_mask):
            return np.zeros((1, 2), dtype=float), np.ones(1, dtype=float)
        selected = pts[selected_mask]
        weights = np.where(boundary[selected_mask], 0.5, 1.0).astype(float)
        return selected, weights

    @staticmethod
    def _source_point_chunks(pts: np.ndarray, shape: tuple[int, int], field: np.ndarray):
        components = 3 if is_vectorial_field(field) else 1
        pixels_per_source = max(1, int(shape[0]) * int(shape[1]) * components)
        chunk_size = max(1, min(int(pts.shape[0]), 8_000_000 // pixels_per_source))
        for start in range(0, int(pts.shape[0]), chunk_size):
            yield pts[start : start + chunk_size]

    @staticmethod
    def _shifted_objective_pupil_filter_batch(
        F_field: np.ndarray,
        shape: tuple[int, int],
        pts: np.ndarray,
        pixel_size_nm: float,
        params: dict,
    ) -> np.ndarray:
        FX, FY, _rho, _cutoff = _optical_pupil_frequency_grid(shape, pixel_size_nm, params)
        sx = pts[:, 0][:, None, None]
        sy = pts[:, 1][:, None, None]
        pupil = ((FX[None, :, :] - sx) ** 2 + (FY[None, :, :] - sy) ** 2) <= 1.0
        if is_vectorial_field(F_field):
            return np.fft.ifft2(F_field[None, :, :, :] * pupil[:, None, :, :], axes=(-2, -1))
        return np.fft.ifft2(F_field[None, :, :] * pupil, axes=(-2, -1))

    @classmethod
    def _source_half_intensity(
        cls,
        field: np.ndarray,
        pixel_size_nm: float,
        params: dict,
        *,
        axis: str,
        sign: float,
    ) -> np.ndarray:
        arr = np.asarray(field, dtype=np.complex128)
        shape = tuple(arr.shape[-2:])
        pts, weights = cls._half_disc_source_quadrature(params, axis=axis, sign=sign)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("DPC half-disc source quadrature has zero total weight.")
        F_field = np.fft.fft2(arr, axes=(-2, -1))
        out = np.zeros(shape, dtype=float)
        offset = 0
        for chunk in cls._source_point_chunks(pts, shape, arr):
            chunk_weights = weights[offset : offset + chunk.shape[0]]
            offset += chunk.shape[0]
            filtered = cls._shifted_objective_pupil_filter_batch(
                F_field,
                shape,
                chunk,
                pixel_size_nm,
                params,
            )
            out += np.tensordot(chunk_weights, field_intensity(filtered), axes=(0, 0))
        return out / weight_sum

    @classmethod
    def _asymmetric_illumination_dpc_components(
        cls,
        field: np.ndarray,
        pixel_size_nm: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        i_pos_x = cls._source_half_intensity(field, pixel_size_nm, params, axis="x", sign=1.0)
        i_neg_x = cls._source_half_intensity(field, pixel_size_nm, params, axis="x", sign=-1.0)
        i_pos_y = cls._source_half_intensity(field, pixel_size_nm, params, axis="y", sign=1.0)
        i_neg_y = cls._source_half_intensity(field, pixel_size_nm, params, axis="y", sign=-1.0)
        eps = float(np.finfo(float).eps)
        dpc_x = (i_pos_x - i_neg_x) / np.maximum(i_pos_x + i_neg_x, eps)
        dpc_y = (i_pos_y - i_neg_y) / np.maximum(i_pos_y + i_neg_y, eps)
        return dpc_x, dpc_y

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

        if detection_mode == VECTORIAL_DETECTION_ANALYZER_X:
            return DifferentialPhaseContrastImagingModel._dpc_components(E[0], pixel_size_nm)
        elif detection_mode == VECTORIAL_DETECTION_ANALYZER_Y:
            return DifferentialPhaseContrastImagingModel._dpc_components(E[1], pixel_size_nm)
        elif detection_mode == VECTORIAL_DETECTION_INCOHERENT_SUM:
            dx_x, dy_x = DifferentialPhaseContrastImagingModel._dpc_components(E[0], pixel_size_nm)
            dx_y, dy_y = DifferentialPhaseContrastImagingModel._dpc_components(E[1], pixel_size_nm)
            return 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
        elif detection_mode == VECTORIAL_DETECTION_UNPOLARIZED:
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
    def _combine_dpc_components(
        dphi_dx: np.ndarray,
        dphi_dy: np.ndarray,
        gain_x: float,
        gain_y: float,
        channel: str,
    ) -> tuple[np.ndarray, float]:
        """Apply axis gains before forming any composite DPC observable."""
        if channel == "x":
            return dphi_dx, float(gain_x)
        if channel == "y":
            return dphi_dy, float(gain_y)

        x = float(gain_x) * np.asarray(dphi_dx, dtype=float)
        y = float(gain_y) * np.asarray(dphi_dy, dtype=float)
        if channel == "diagonal":
            return (x + y) / np.sqrt(2.0), 1.0
        if channel == "magnitude":
            return np.sqrt(x * x + y * y), 1.0
        raise ValueError("dpc_output_channel must be x, y, diagonal, or magnitude.")

    @staticmethod
    def _masked_pupil_intensity(field: np.ndarray, pupil_weight: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, dtype=np.complex128)
        F = np.fft.fft2(arr, axes=(-2, -1))
        filtered = np.fft.ifft2(F * np.asarray(pupil_weight, dtype=float), axes=(-2, -1))
        return field_intensity(filtered)

    @classmethod
    def _split_pupil_detection_dpc_components(
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
            if detection_mode not in VALID_VECTORIAL_DETECTION_MODES:
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
            if settings.optical.polarization_model == "unpolarized":
                raise ValueError(
                    "Vectorial DPC requires a coherent input polarization before "
                    "the renderer applies its detection reduction. "
                    "polarization_model='unpolarized' is an incoherent source average; "
                    "use linear_x or linear_y, then choose full_vector, analyzer_x/"
                    "analyzer_y, or an incoherent detection reduction explicitly."
                )
            use_vector_signal = True
        else:
            use_vector_signal = False

        if transfer_model == cls._DPC_TRANSFER_SPLIT_PUPIL_DETECTION:
            arr = np.asarray(field, dtype=np.complex128)
            if use_vector_signal and arr.ndim == 3 and arr.shape[0] == 3:
                if detection_mode == VECTORIAL_DETECTION_ANALYZER_X:
                    signal_field = arr[0]
                    dphi_dx, dphi_dy = cls._split_pupil_detection_dpc_components(
                        signal_field,
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode == VECTORIAL_DETECTION_ANALYZER_Y:
                    signal_field = arr[1]
                    dphi_dx, dphi_dy = cls._split_pupil_detection_dpc_components(
                        signal_field,
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode in cls._DPC_INCOHERENT_DETECTION_MODES:
                    dx_x, dy_x = cls._split_pupil_detection_dpc_components(
                        arr[0],
                        pixel_size_nm,
                        params,
                    )
                    dx_y, dy_y = cls._split_pupil_detection_dpc_components(
                        arr[1],
                        pixel_size_nm,
                        params,
                    )
                    dphi_dx, dphi_dy = 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
                else:
                    dphi_dx, dphi_dy = cls._split_pupil_detection_dpc_components(
                        arr,
                        pixel_size_nm,
                        params,
                    )
            else:
                dphi_dx, dphi_dy = cls._split_pupil_detection_dpc_components(
                    arr,
                    pixel_size_nm,
                    params,
                )
        elif transfer_model == cls._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION:
            arr = np.asarray(field, dtype=np.complex128)
            if use_vector_signal and arr.ndim == 3 and arr.shape[0] == 3:
                if detection_mode == VECTORIAL_DETECTION_ANALYZER_X:
                    dphi_dx, dphi_dy = cls._asymmetric_illumination_dpc_components(
                        arr[0],
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode == VECTORIAL_DETECTION_ANALYZER_Y:
                    dphi_dx, dphi_dy = cls._asymmetric_illumination_dpc_components(
                        arr[1],
                        pixel_size_nm,
                        params,
                    )
                elif detection_mode in cls._DPC_INCOHERENT_DETECTION_MODES:
                    dx_x, dy_x = cls._asymmetric_illumination_dpc_components(
                        arr[0],
                        pixel_size_nm,
                        params,
                    )
                    dx_y, dy_y = cls._asymmetric_illumination_dpc_components(
                        arr[1],
                        pixel_size_nm,
                        params,
                    )
                    dphi_dx, dphi_dy = 0.5 * (dx_x + dx_y), 0.5 * (dy_x + dy_y)
                else:
                    dphi_dx, dphi_dy = cls._asymmetric_illumination_dpc_components(
                        arr,
                        pixel_size_nm,
                        params,
                    )
            else:
                dphi_dx, dphi_dy = cls._asymmetric_illumination_dpc_components(
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
        if transfer_model in {
            cls._DPC_TRANSFER_SPLIT_PUPIL_DETECTION,
            cls._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION,
        }:
            gain_x = settings.intensity_gain_x
            gain_y = settings.intensity_gain_y
        else:
            gain_x = settings.phase_gradient_gain_x
            gain_y = settings.phase_gradient_gain_y
        return cls._combine_dpc_components(
            dphi_dx,
            dphi_dy,
            gain_x,
            gain_y,
            channel,
        )

    def _intensity_from_total_field(
        self,
        total_field: np.ndarray,
        incident_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        dpc, gain = self._dpc_signal(total_field, pixel_size_px_nm, params)
        return np.maximum(field_intensity(incident_field) * (1.0 + gain * dpc), 0.0)

    def _incoherent_analyzer_average_signal(
        self,
        E_sca: np.ndarray,
        params: dict,
        *,
        include_background: bool,
    ) -> np.ndarray:
        """Average analyzer-resolved DPC signals in the intensity domain.

        ``vectorial_detection_mode`` is a renderer-owned physical contract for
        vectorial DPC.  Incoherent/unpolarized detection must not be implemented
        as a coherent field sum or by applying a future gain after channel
        averaging; each analyzer branch is reduced first and only then averaged
        as an intensity/count contribution.  This makes the unpolarized path
        stable if analyzer branches ever acquire distinct calibration factors.
        """

        arr = np.asarray(E_sca, dtype=np.complex128)
        pixel_size_px_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        e_inc_x = np.zeros_like(arr)
        e_inc_y = np.zeros_like(arr)
        e_inc_x[0, :, :] = self._E_inc_amplitude
        e_inc_y[1, :, :] = self._E_inc_amplitude
        dpc_x, gain_x = self._dpc_signal(
            e_inc_x + arr,
            pixel_size_px_nm,
            dict(params, vectorial_detection_mode=VECTORIAL_DETECTION_ANALYZER_X),
        )
        dpc_y, gain_y = self._dpc_signal(
            e_inc_y + arr,
            pixel_size_px_nm,
            dict(params, vectorial_detection_mode=VECTORIAL_DETECTION_ANALYZER_Y),
        )
        contrast_term = 0.5 * (
            float(gain_x) * np.asarray(dpc_x, dtype=float)
            + float(gain_y) * np.asarray(dpc_y, dtype=float)
        )
        incident_intensity = self._E_inc_amplitude ** 2
        if include_background:
            return np.maximum(incident_intensity * (1.0 + contrast_term), 0.0)
        return incident_intensity * contrast_term

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
            if detection_mode == VECTORIAL_DETECTION_ANALYZER_X:
                E_inc[0, :, :] = self._E_inc_amplitude
            elif detection_mode == VECTORIAL_DETECTION_ANALYZER_Y:
                E_inc[1, :, :] = self._E_inc_amplitude
            elif detection_mode == VECTORIAL_DETECTION_INCOHERENT_SUM:
                E_inc[0, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
                E_inc[1, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
            elif detection_mode == self._DPC_FULL_VECTOR_DETECTION:
                E_inc = self._incident_field_for_scattered(
                    E_sca_total,
                    params,
                    base_field=background_field,
                )
            elif detection_mode == VECTORIAL_DETECTION_UNPOLARIZED:
                return self._incoherent_analyzer_average_signal(
                    E_sca_total,
                    params,
                    include_background=True,
                )
        else:
            E_inc = self._incident_field_for_scattered(
                E_sca_total,
                params,
                base_field=background_field,
            )
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
            if detection_mode == VECTORIAL_DETECTION_ANALYZER_X:
                E_inc[0, :, :] = self._E_inc_amplitude
            elif detection_mode == VECTORIAL_DETECTION_ANALYZER_Y:
                E_inc[1, :, :] = self._E_inc_amplitude
            elif detection_mode == VECTORIAL_DETECTION_INCOHERENT_SUM:
                E_inc[0, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
                E_inc[1, :, :] = self._E_inc_amplitude / np.sqrt(2.0)
            elif detection_mode == self._DPC_FULL_VECTOR_DETECTION:
                E_inc = self._incident_field_for_scattered(
                    E_sca_particle,
                    params,
                    base_field=background_field,
                )
            elif detection_mode == VECTORIAL_DETECTION_UNPOLARIZED:
                return self._incoherent_analyzer_average_signal(
                    E_sca_particle,
                    params,
                    include_background=False,
                )
        else:
            E_inc = self._incident_field_for_scattered(
                E_sca_particle,
                params,
                base_field=background_field,
            )
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
        vectorial_backend_enabled = channel_model in {
            self._DPC_CHANNEL_VECTORIAL_SPLIT,
            self._DPC_CHANNEL_VECTORIAL_ASYMMETRIC,
        }
        if channel_model in {
            self._DPC_CHANNEL_VECTORIAL_SPLIT,
            self._DPC_CHANNEL_VECTORIAL_ASYMMETRIC,
        }:
            detection_mode = settings.optical.vectorial_detection_mode
            detection_label = {
                "analyzer_x": "analyzer-x",
                "analyzer_y": "analyzer-y",
                "incoherent_sum": "incoherent-sum",
                "unpolarized": "unpolarized average",
            }.get(detection_mode, "full-vector")
            forward_observable = f"vectorial Debye {detection_label} split-pupil detection DPC"
            vectorial_detection_mode = settings.optical.vectorial_detection_mode
        else:
            forward_observable = "scalar two-axis split-pupil detection DPC"
            vectorial_detection_mode = "incoherent_sum"
        if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION:
            forward_observable = forward_observable.replace(
                "split-pupil detection",
                "asymmetric half-disc illumination",
            )
        elif transfer_model == self._DPC_TRANSFER_PHASE_GRADIENT:
            forward_observable = forward_observable.replace("split-pupil detection", "phase-gradient proxy")
        if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION:
            fidelity_label = (
                "vectorial_asymmetric_illumination_dpc"
                if vectorial_backend_enabled
                else "scalar_asymmetric_illumination_dpc"
            )
            validity_scope = (
                "Illumination-side DPC using opposite condenser half-disc source "
                "images, shifted objective pupils, normalized intensity "
                "differences, and the listed vectorial/scalar detection mode."
            )
        elif transfer_model == self._DPC_TRANSFER_SPLIT_PUPIL_DETECTION:
            fidelity_label = (
                "vectorial_split_pupil_detection_dpc"
                if vectorial_backend_enabled
                else "scalar_split_pupil_detection_dpc"
            )
            validity_scope = (
                "Detector-side split-pupil/Foucault DPC diagnostic; not the "
                "illumination-side asymmetric DPC default."
            )
        else:
            fidelity_label = "phase_gradient_proxy_dpc"
            validity_scope = (
                "Local phase-gradient proxy retained for diagnostics; not the "
                "illumination-side asymmetric DPC default."
            )
        kind = (
            "phase_gradient_proxy_dpc"
            if transfer_model == self._DPC_TRANSFER_PHASE_GRADIENT
            else (
                "asymmetric_illumination_dpc"
                if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                else "split_pupil_detection_dpc"
            )
        )
        if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION:
            source_counts = {}
            source_weight_sums = {}
            source_boundary_counts = {}
            for key, axis, sign in (
                ("x_positive", "x", 1.0),
                ("x_negative", "x", -1.0),
                ("y_positive", "y", 1.0),
                ("y_negative", "y", -1.0),
            ):
                pts, weights = self._half_disc_source_quadrature(
                    params,
                    axis=axis,
                    sign=sign,
                )
                source_counts[key] = int(pts.shape[0])
                source_weight_sums[key] = float(np.sum(weights))
                source_boundary_counts[key] = int(np.count_nonzero(np.isclose(weights, 0.5)))
        else:
            source_counts = None
            source_weight_sums = None
            source_boundary_counts = None
        detection_contract = vectorial_detection_contract_for_mode(vectorial_detection_mode)
        response.update(
            kind=kind,
            fidelity_label=fidelity_label,
            validity_scope=validity_scope,
            forward_observable=forward_observable,
            dpc_model_identity=(
                "phase_gradient_proxy"
                if transfer_model == self._DPC_TRANSFER_PHASE_GRADIENT
                else (
                    "illumination_side_asymmetric_kohler_dpc"
                    if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                    else "split_pupil_foucault_detection"
                )
            ),
            dpc_illumination_side_model=(
                "opposite_half_disc_condenser_source"
                if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                else "not_asymmetric_illumination"
            ),
            dpc_channel_model=channel_model,
            dpc_transfer_model=transfer_model,
            dpc_output_channel=settings.output_channel,
            dpc_vectorial_detection_mode=vectorial_detection_mode,
            dpc_vectorial_detection_reduction_kind=detection_contract.reduction_kind,
            dpc_vectorial_detection_field_or_intensity_basis=detection_contract.field_or_intensity_basis,
            dpc_illumination_sigma=(
                settings.illumination_sigma
                if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                else None
            ),
            dpc_source_samples_requested=(
                settings.source_samples
                if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                else None
            ),
            dpc_source_half_counts=source_counts,
            dpc_source_half_weight_sums=source_weight_sums,
            dpc_source_half_boundary_counts=source_boundary_counts,
            dpc_source_sampling_scheme=(
                "deterministic_hex_half_disc_with_half_weight_boundary"
                if transfer_model == self._DPC_TRANSFER_ASYMMETRIC_ILLUMINATION
                else None
            ),
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
            dpc_axis_gain_combination="componentwise_before_channel_projection",
            dpc_vectorial_backend_enabled=bool(vectorial_backend_enabled),
        )
        return response

__all__ = ['DifferentialPhaseContrastImagingModel']
