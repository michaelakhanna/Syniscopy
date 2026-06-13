from __future__ import annotations
from configured_parameters import configured_assign

from dataclasses import dataclass
from typing import Any

import numpy as np
from .kinetics import resolve_fluorescence_state_kinetics
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import (
    FluorescenceSettings,
    OpticalInstrumentSettings,
    OpticalPsfSupportSettings,
)
from source_convolution_contracts import (
    SOURCE_CONVOLUTION_CONTRACT_ID,
    SourceConvolutionBoundaryMode,
    SourceConvolutionContext,
    convolve2d_with_source_boundary,
    source_convolution_context_from,
)

try:
    import vectorial_optics
except ImportError:  # pragma: no cover - optional vectorial backend dependency.
    vectorial_optics = None


class FluorescencePhotophysicsError(RuntimeError):
    """Raised when vectorial/photophysics fluorescence configuration is invalid."""


@dataclass(frozen=True)
class FluorescenceBackendMetadata:
    backend_mode: str
    backend_fidelity_level: str
    algorithm: str
    excitation_wavelength_nm: float
    emission_wavelength_nm: float
    quantum_yield: float
    excitation_scale: float
    collection_efficiency: float
    detector_qe: float
    absorbed_excitation_photons_per_fluorophore_per_frame: float
    uniform_background_counts_per_pixel: float
    effective_detection_factor: float
    detector_pixel_area_nm2: float
    areal_density_count_scale: float
    bright_to_dark_rate_per_frame: float
    dark_to_bright_rate_per_frame: float
    bleaching_rate_per_frame: float
    tirf_evanescent_depth_nm: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fluorescence_backend": self.backend_mode,
            "backend_mode": self.backend_mode,
            "backend_fidelity_level": self.backend_fidelity_level,
            "algorithm": self.algorithm,
            "excitation_wavelength_nm": self.excitation_wavelength_nm,
            "emission_wavelength_nm": self.emission_wavelength_nm,
            "quantum_yield": self.quantum_yield,
            "excitation_scale": self.excitation_scale,
            "collection_efficiency": self.collection_efficiency,
            "detector_qe": self.detector_qe,
            "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": (
                self.absorbed_excitation_photons_per_fluorophore_per_frame
            ),
            "uniform_background_counts_per_pixel": self.uniform_background_counts_per_pixel,
            "effective_detection_factor": self.effective_detection_factor,
            "fluorescence_detector_pixel_area_nm2": self.detector_pixel_area_nm2,
            "fluorescence_areal_density_count_scale": self.areal_density_count_scale,
            "fluorescence_bright_to_dark_rate_per_frame": self.bright_to_dark_rate_per_frame,
            "fluorescence_dark_to_bright_rate_per_frame": self.dark_to_bright_rate_per_frame,
            "bleaching_rate_per_frame": self.bleaching_rate_per_frame,
            "tirf_evanescent_depth_nm": self.tirf_evanescent_depth_nm,
        }


class VectorialPhotophysicsFluorescenceBackend:
    """Syniscopy-owned vectorial PSF + fluorophore-state fluorescence backend."""

    backend_mode = "vectorial_photophysics"

    def __init__(
        self,
        params: dict,
        *,
        fluorescence_settings: FluorescenceSettings,
        canvas_pitch_nm: float,
        detector_pixel_area_nm2: float,
        base_emission_sigma_px: float,
        quantum_yield: float,
        excitation_scale: float,
        collection_efficiency: float,
        detector_qe: float,
        absorbed_excitation_photons_per_fluorophore: float,
        uniform_background: float,
        vectorial_numerical_aperture: float | None = None,
    ) -> None:
        self.params = dict(params)
        self.fluorescence_settings = fluorescence_settings
        self.excitation_wavelength_nm = fluorescence_settings.excitation_wavelength_nm
        self.emission_wavelength_nm = fluorescence_settings.emission_wavelength_nm
        self.canvas_pitch_nm = float(canvas_pitch_nm)
        self.detector_pixel_area_nm2 = float(detector_pixel_area_nm2)
        self.base_emission_sigma_px = float(base_emission_sigma_px)
        self.quantum_yield = float(quantum_yield)
        self.excitation_scale = float(excitation_scale)
        self.collection_efficiency = float(collection_efficiency)
        self.detector_qe = float(detector_qe)
        self.absorbed_excitation_photons_per_fluorophore = float(
            absorbed_excitation_photons_per_fluorophore
        )
        self.uniform_background = float(uniform_background)
        self.vectorial_numerical_aperture = (
            None
            if vectorial_numerical_aperture is None
            else float(vectorial_numerical_aperture)
        )
        if self.vectorial_numerical_aperture is not None and (
            not np.isfinite(self.vectorial_numerical_aperture)
            or self.vectorial_numerical_aperture <= 0.0
        ):
            raise ValueError(
                "vectorial_numerical_aperture must be finite and positive when set; "
                f"got {vectorial_numerical_aperture!r}."
            )
        if (
            not np.isfinite(self.absorbed_excitation_photons_per_fluorophore)
            or self.absorbed_excitation_photons_per_fluorophore < 0.0
        ):
            raise ValueError(
                "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame "
                "must be finite and non-negative; got "
                f"{self.absorbed_excitation_photons_per_fluorophore!r}."
            )
        if (
            not np.isfinite(self.detector_pixel_area_nm2)
            or self.detector_pixel_area_nm2 <= 0.0
        ):
            raise ValueError(
                "detector_pixel_area_nm2 must be finite and positive; got "
                f"{self.detector_pixel_area_nm2!r}."
            )
        # Kinetics contract: resolve bright/dark/tau/bleach semantics once at
        # the backend boundary so runtime occupancy and metadata use one basis.
        self.kinetics = resolve_fluorescence_state_kinetics(params)
        self.bright_to_dark_rate_per_frame = self.kinetics.bright_to_dark_rate_per_frame
        self.dark_to_bright_rate_per_frame = self.kinetics.dark_to_bright_rate_per_frame
        self.bleaching_rate = self.kinetics.bleaching_rate_per_frame
        if not np.isfinite(self.bright_to_dark_rate_per_frame):
            raise ValueError(
                "resolved bright_to_dark rate must be finite; got "
                f"{self.bright_to_dark_rate_per_frame!r}."
            )
        if not np.isfinite(self.dark_to_bright_rate_per_frame):
            raise ValueError(
                "resolved dark_to_bright rate must be finite; got "
                f"{self.dark_to_bright_rate_per_frame!r}."
            )
        if not np.isfinite(self.bleaching_rate) or self.bleaching_rate < 0.0:
            raise ValueError(
                f"parameters['fluorescence_bleaching_rate_per_frame'] must be finite and "
                f"non-negative; got {self.bleaching_rate!r}."
            )
        self.reference_status = fluorescence_settings.reference_status
        self.validation_status = (
            "external_artifact_required"
            if self.reference_status == "reference_validated"
            else "diagnostic_only"
        )
        self._psf_cache: dict[tuple[int, int, float], np.ndarray] = {}
        self.allow_psf_fallback = fluorescence_settings.allow_psf_fallback
        self._last_psf_backend = "not_evaluated"
        self._last_psf_error: str | None = None
        self._last_source_ndim = 2
        self._last_convolution_context = SourceConvolutionContext(
            boundary_mode=SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value,
            source_extent_role="finite_fov_source_map",
            producer=f"{type(self).__name__}.source_to_emission_density",
        )

    def _fallback_gaussian_psf(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = int(shape[-2]), int(shape[-1])
        y = np.arange(h, dtype=float) - h // 2
        x = np.arange(w, dtype=float) - w // 2
        yy, xx = np.meshgrid(y, x, indexing="ij")
        sigma = max(self.base_emission_sigma_px, 0.5)
        psf = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
        total = float(psf.sum())
        return psf / total if total > 0.0 else np.full((h, w), 1.0 / float(h * w))

    def _tail_fraction_threshold(self) -> float:
        return OpticalPsfSupportSettings.from_params(self.params).intensity_fraction_threshold

    @staticmethod
    def _centered_l1_support_radius_pixels(
        psf: np.ndarray,
        *,
        tail_fraction: float,
    ) -> int:
        arr = np.maximum(np.asarray(psf, dtype=float), 0.0)
        total = float(np.sum(arr))
        if total <= 0.0 or not np.isfinite(total):
            return 0
        h, w = arr.shape[-2:]
        yy, xx = np.indices((h, w), dtype=float)
        cy = h // 2
        cx = w // 2
        radius = np.sqrt((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2).ravel()
        values = arr.ravel()
        order = np.argsort(radius)
        cumulative = np.cumsum(values[order])
        target = total * (1.0 - float(np.clip(tail_fraction, 0.0, 1.0)))
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), order.size - 1)
        return int(np.ceil(float(radius[order[idx]])))

    def _normalize_and_validate_psf(
        self,
        psf: np.ndarray,
        target_shape: tuple[int, int],
        *,
        z_nm: float,
        validate_support: bool,
    ) -> np.ndarray:
        target = (int(target_shape[-2]), int(target_shape[-1]))
        psf = np.maximum(np.asarray(psf, dtype=float), 0.0)
        threshold = self._tail_fraction_threshold()
        raw_total = float(psf.sum())
        if psf.shape != target:
            out = np.zeros(target, dtype=float)
            h = min(target[0], psf.shape[0])
            w = min(target[1], psf.shape[1])
            sy = (psf.shape[0] - h) // 2
            sx = (psf.shape[1] - w) // 2
            dy = (target[0] - h) // 2
            dx = (target[1] - w) // 2
            out[dy:dy + h, dx:dx + w] = psf[sy:sy + h, sx:sx + w]
            kept_total = float(out.sum())
            if validate_support and raw_total > 0.0:
                lost_fraction = max(0.0, 1.0 - kept_total / raw_total)
                if lost_fraction > threshold:
                    raise FluorescencePhotophysicsError(
                        "Vectorial fluorescence PSF support exceeds the guarded "
                        f"render canvas at z={float(z_nm):.6g} nm "
                        f"(lost_fraction={lost_fraction:.3g}, threshold={threshold:.3g}, "
                        f"psf_shape={psf.shape}, canvas_shape={target}). Increase the "
                        "fluorescence/TEM-style filter guard or render canvas instead of "
                        "cropping and renormalizing the PSF tail."
                    )
            psf = out
        total = float(psf.sum())
        if total <= 0.0 or not np.isfinite(total):
            if not self.allow_psf_fallback:
                raise FluorescencePhotophysicsError(
                    "Vectorial fluorescence PSF had non-positive or non-finite "
                    "energy. Set fluorescence_allow_psf_fallback=True only for "
                    "an explicit Gaussian-proxy diagnostic run."
                )
            self._last_psf_backend = "fallback_gaussian"
            self._last_psf_error = "vectorial PSF had non-positive or non-finite energy"
            psf = self._fallback_gaussian_psf(target)
            total = float(psf.sum())
        return psf / total

    def psf_support_radius_pixels(
        self,
        shape: tuple[int, int],
        *,
        z_positions_nm: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> int:
        if z_positions_nm is None:
            z_values = np.array([0.0], dtype=float)
        else:
            z_values = np.asarray(z_positions_nm, dtype=float).reshape(-1)
            z_values = z_values[np.isfinite(z_values)]
            if z_values.size == 0:
                z_values = np.array([0.0], dtype=float)
        threshold = self._tail_fraction_threshold()
        support = 0
        for z_nm in np.unique(np.round(z_values, 9)):
            psf = self._vectorial_psf(
                shape,
                z_nm=float(z_nm),
                validate_support=False,
            )
            support = max(
                support,
                self._centered_l1_support_radius_pixels(
                    psf,
                    tail_fraction=threshold,
                ),
            )
        return int(support)

    def _vectorial_psf(
        self,
        shape: tuple[int, int],
        *,
        z_nm: float = 0.0,
        validate_support: bool = True,
    ) -> np.ndarray:
        key = (int(shape[-2]), int(shape[-1]), round(float(z_nm), 9))
        cached = self._psf_cache.get(key) if validate_support else None
        if cached is not None:
            return cached
        if vectorial_optics is not None:
            try:
                params = dict(self.params)
                configured_assign(params, 'wavelength_nm', self.emission_wavelength_nm)
                if self.vectorial_numerical_aperture is not None:
                    configured_assign(params, 'numerical_aperture', self.vectorial_numerical_aperture)
                samples = OpticalInstrumentSettings.from_params(params).vectorial_pupil_samples
                configured_assign(params, 'vectorial_pupil_samples', max(samples, key[0], key[1]))
                stack = vectorial_optics.compute_isotropic_dipole_emission_psf(
                    params,
                    [float(z_nm)],
                )
                intensity = stack["intensity"][0]
                psf = np.asarray(intensity, dtype=float)
                self._last_psf_backend = "isotropic_dipole_vectorial_debye"
                self._last_psf_error = None
            except Exception as exc:
                if not self.allow_psf_fallback:
                    raise FluorescencePhotophysicsError(
                        "Vectorial fluorescence PSF construction failed. Set "
                        "fluorescence_allow_psf_fallback=True only for an explicit "
                        "Gaussian-proxy diagnostic run."
                    ) from exc
                self._last_psf_backend = "fallback_gaussian"
                self._last_psf_error = f"{type(exc).__name__}: {exc}"
                psf = self._fallback_gaussian_psf(key[:2])
        else:
            if not self.allow_psf_fallback:
                raise FluorescencePhotophysicsError(
                    "Vectorial fluorescence backend requires vectorial_optics. Set "
                    "fluorescence_allow_psf_fallback=True only for an explicit "
                    "Gaussian-proxy diagnostic run."
                )
            self._last_psf_backend = "fallback_gaussian"
            self._last_psf_error = "vectorial_optics module unavailable"
            psf = self._fallback_gaussian_psf(key[:2])
        psf = self._normalize_and_validate_psf(
            psf,
            key[:2],
            z_nm=float(z_nm),
            validate_support=validate_support,
        )
        if validate_support:
            self._psf_cache[key] = psf
        return psf

    def _convolve_source_with_psf(
        self,
        source: np.ndarray,
        psf: np.ndarray,
        *,
        convolution_context: SourceConvolutionContext | dict[str, Any] | str | None = None,
    ) -> np.ndarray:
        # Source-boundary contract fix: fluorescence source maps are finite FOVs
        # unless a render canvas explicitly proves it is guard-padded/cropped or
        # a caller declares a periodic tile.  Same-size FFT multiplication is a
        # circular convolution, so it must not be the implicit default for direct
        # Fisher-facing source products.
        ctx = source_convolution_context_from(
            convolution_context,
            default_boundary_mode=SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value,
            source_extent_role="finite_fov_source_map",
            producer=f"{type(self).__name__}.source_to_emission_density",
        )
        required_guard = None
        if ctx.boundary_mode == SourceConvolutionBoundaryMode.PRE_CROP_GUARDED_FFT.value:
            required_guard = self._centered_l1_support_radius_pixels(
                psf,
                tail_fraction=self._tail_fraction_threshold(),
            )
        out = convolve2d_with_source_boundary(
            source,
            psf,
            context=ctx,
            minimum_guard_radius_pixels=required_guard,
        )
        self._last_convolution_context = ctx
        return np.maximum(out, 0.0)

    def _state_factor(self, frame_index: int) -> float:
        # Kinetics contract: the shared resolver/dataclass owns the complete
        # bright/dark plus bleaching occupancy law. The backend only applies the
        # resolved state factor once before detector-count scaling.
        return self.kinetics.state_factor(frame_index)

    def _physical_count_scale(self) -> float:
        return float(
            self.absorbed_excitation_photons_per_fluorophore
            * self.excitation_scale
            * self.quantum_yield
            * self.collection_efficiency
            * self.detector_qe
        )

    def _areal_density_count_scale(self) -> float:
        return float(self._physical_count_scale() * self.detector_pixel_area_nm2)

    def source_to_emission_density(
        self,
        source: np.ndarray,
        *,
        frame_index: int = 0,
        z_positions_nm: np.ndarray | list[float] | tuple[float, ...] | None = None,
        convolution_context: SourceConvolutionContext | dict[str, Any] | str | None = None,
    ) -> np.ndarray:
        src = np.maximum(np.asarray(source, dtype=float), 0.0)
        self._last_source_ndim = int(src.ndim)
        if src.ndim == 2:
            psf = self._vectorial_psf(src.shape, z_nm=0.0)
            emission_density = self._convolve_source_with_psf(
                src,
                psf,
                convolution_context=convolution_context,
            )
        elif src.ndim == 3:
            if z_positions_nm is None:
                raise ValueError(
                    "3D fluorescence source stacks require z_positions_nm with "
                    "one axial position per source slice."
                )
            z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
            if z_positions.shape[0] != src.shape[0] or not np.all(np.isfinite(z_positions)):
                raise ValueError(
                    "z_positions_nm must be finite and have length equal to the "
                    f"3D source stack depth ({src.shape[0]})."
                )
            emission_density = np.zeros(src.shape[-2:], dtype=float)
            for zi, z_nm in enumerate(z_positions):
                psf = self._vectorial_psf(src.shape[-2:], z_nm=float(z_nm))
                emission_density += self._convolve_source_with_psf(
                    src[zi],
                    psf,
                    convolution_context=convolution_context,
                )
        else:
            raise ValueError(
                "Fluorescence source maps must be 2D (y, x) or 3D (z, y, x); "
                f"got shape {src.shape!r}."
            )
        emission_density = emission_density * self._state_factor(frame_index)
        if not np.all(np.isfinite(emission_density)):
            raise FloatingPointError(
                "vectorial photophysics fluorescence produced non-finite emission density."
            )
        return np.maximum(emission_density, 0.0)

    def source_to_detector_counts(
        self,
        source: np.ndarray,
        *,
        frame_index: int = 0,
        z_positions_nm: np.ndarray | list[float] | tuple[float, ...] | None = None,
        include_background: bool = True,
        convolution_context: SourceConvolutionContext | dict[str, Any] | str | None = None,
    ) -> np.ndarray:
        detected = (
            self.source_to_emission_density(
                source,
                frame_index=frame_index,
                z_positions_nm=z_positions_nm,
                convolution_context=convolution_context,
            )
            * self._areal_density_count_scale()
        )
        if include_background:
            detected = detected + self.uniform_background
        if not np.all(np.isfinite(detected)):
            raise FloatingPointError("vectorial photophysics fluorescence produced non-finite counts.")
        return np.maximum(detected, 0.0)

    def metadata(
        self,
        params: dict | None = None,
        *,
        tirf_depth_nm: float | None = None,
        source_ndim: int | None = None,
    ) -> dict[str, Any]:
        p = self.params if params is None else params
        fidelity = "high_fidelity"
        validation = self.validation_status
        meta = FluorescenceBackendMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level=fidelity,
            algorithm="isotropic_dipole_vectorial_debye_psf_with_mean_field_fluorophore_state_model",
            excitation_wavelength_nm=self.excitation_wavelength_nm,
            emission_wavelength_nm=self.emission_wavelength_nm,
            quantum_yield=self.quantum_yield,
            excitation_scale=self.excitation_scale,
            collection_efficiency=self.collection_efficiency,
            detector_qe=self.detector_qe,
            absorbed_excitation_photons_per_fluorophore_per_frame=(
                self.absorbed_excitation_photons_per_fluorophore
            ),
            uniform_background_counts_per_pixel=self.uniform_background,
            effective_detection_factor=self._physical_count_scale(),
            detector_pixel_area_nm2=self.detector_pixel_area_nm2,
            areal_density_count_scale=self._areal_density_count_scale(),
            bright_to_dark_rate_per_frame=self.bright_to_dark_rate_per_frame,
            dark_to_bright_rate_per_frame=self.dark_to_bright_rate_per_frame,
            bleaching_rate_per_frame=self.bleaching_rate,
            tirf_evanescent_depth_nm=tirf_depth_nm,
        ).to_dict()
        meta.update(
            kind="vectorial_photophysics_fluorescence",
            fidelity_label=(
                "vectorial_photophysics_external_artifact_required"
                if self.reference_status == "reference_validated"
                else "vectorial_photophysics_fidelity_based"
            ),
            forward_observable="vectorial source map PSF with fluorophore-state Markov kinetics",
            photophysics_state_model="deterministic_mean_occupancy",
            fluorescence_photon_budget_source=(
                "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"
            ),
            fluorescence_photon_budget_semantics="absorbed_excitation_photons_before_quantum_yield",
            fluorescence_physical_count_scale=self._physical_count_scale(),
            count_scale=self._areal_density_count_scale(),
            count_scaling_mode="areal_density_times_detector_pixel_area_times_photon_budget",
            **self.kinetics.to_metadata(),
            fluorescence_emitter_orientation_model="isotropic_dipole_incoherent_average_xyz",
            emission_psf_vectorial_model="isotropic_dipole_vectorial_debye_detection_intensity",
            emission_psf_axial_model=(
                "z_sliced_source_specific_psf"
                if int(source_ndim if source_ndim is not None else self._last_source_ndim) == 3
                else "single_focal_plane_z0"
            ),
            reference_backend_metadata={
                "reference_status": self.reference_status,
                "reference_validation_hash": self.fluorescence_settings.reference_validation_hash,
                "claim_maturity_gate": validation,
            },
            validation_status=validation,
            comparison_contract_id=p.get("comparison_contract_id", "Contract-NR"),
            fluorescence_absolute_scale="physical_absorbed_excitation_photon_budget",
            fluorescence_background_units="detected_counts_per_pixel",
            emission_psf_backend=self._last_psf_backend,
            source_convolution_contract_id=SOURCE_CONVOLUTION_CONTRACT_ID,
            emission_psf_boundary_mode=self._last_convolution_context.boundary_mode,
            source_convolution_context=self._last_convolution_context.to_metadata(),
            emission_psf_fallback_error=self._last_psf_error,
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=p,
            backend_name=self.backend_mode,
            equations_or_model_family="vectorial_photophysics_psf_with_mean_field_fluorophore_kinetics",
            implemented_approximation_level=fidelity,
            native_operating_assumptions="incoherent isotropic-dipole vectorial emission PSF with deterministic mean-field fluorophore occupancy",
            comparison_contract_id=p.get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=p.get("artifact_provenance_id", None),
        )


__all__ = ["FluorescencePhotophysicsError", "VectorialPhotophysicsFluorescenceBackend"]
