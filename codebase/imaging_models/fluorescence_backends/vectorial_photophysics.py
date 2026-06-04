from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import param_value

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
    collection_efficiency: float
    detector_qe: float
    photons_per_fluorophore_per_frame: float | None
    uniform_background_counts_per_pixel: float
    effective_detection_factor: float
    blinking_rate_per_frame: float
    recovery_rate_per_frame: float
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
            "collection_efficiency": self.collection_efficiency,
            "detector_qe": self.detector_qe,
            "photons_per_fluorophore_per_frame": self.photons_per_fluorophore_per_frame,
            "uniform_background_counts_per_pixel": self.uniform_background_counts_per_pixel,
            "effective_detection_factor": self.effective_detection_factor,
            "blinking_rate_per_frame": self.blinking_rate_per_frame,
            "recovery_rate_per_frame": self.recovery_rate_per_frame,
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
        canvas_pitch_nm: float,
        base_emission_sigma_px: float,
        quantum_yield: float,
        excitation_scale: float,
        collection_efficiency: float,
        detector_qe: float,
        photons_per_fluorophore: float | None,
        uniform_background: float,
    ) -> None:
        self.params = dict(params)
        self.canvas_pitch_nm = float(canvas_pitch_nm)
        self.base_emission_sigma_px = float(base_emission_sigma_px)
        self.quantum_yield = float(quantum_yield)
        self.excitation_scale = float(excitation_scale)
        self.collection_efficiency = float(collection_efficiency)
        self.detector_qe = float(detector_qe)
        self.photons_per_fluorophore = None if photons_per_fluorophore is None else float(photons_per_fluorophore)
        self.uniform_background = float(uniform_background)
        legacy_scale = param_value(params, "fluorescence_photon_count_scale")
        strict_budget = bool(param_value(params, 'fluorescence_require_physical_photon_budget'))
        if self.photons_per_fluorophore is not None and (
            not np.isfinite(self.photons_per_fluorophore) or self.photons_per_fluorophore < 0.0
        ):
            raise ValueError(
                "fluorescence_photons_per_fluorophore_per_frame must be finite and non-negative "
                f"when supplied; got {self.photons_per_fluorophore!r}."
            )
        if strict_budget and self.photons_per_fluorophore is None:
            raise ValueError(
                "vectorial_photophysics fluorescence was configured with "
                "fluorescence_require_physical_photon_budget=True but no "
                "fluorescence_photons_per_fluorophore_per_frame value. "
                "fluorescence_photon_count_scale is a legacy proxy-scale field and "
                "does not define the physical vectorial photon budget."
            )
        if legacy_scale is None:
            self.legacy_photon_count_scale = None
        else:
            self.legacy_photon_count_scale = float(legacy_scale)
            if not np.isfinite(self.legacy_photon_count_scale) or self.legacy_photon_count_scale < 0.0:
                raise ValueError(
                    "fluorescence_photon_count_scale must be finite and non-negative "
                    f"when supplied; got {self.legacy_photon_count_scale!r}."
                )
        self.legacy_photon_count_scale_supplied = legacy_scale is not None
        self.blinking_rate = float(param_value(params, 'fluorescence_blinking_rate_per_frame'))
        self.recovery_rate = float(param_value(params, 'fluorescence_recovery_rate_per_frame'))
        bleaching_raw = param_value(params, 'fluorescence_bleaching_rate_per_frame')
        self.bleaching_rate = float(bleaching_raw)
        tau_raw = param_value(params, 'fluorescence_photobleach_tau_frames')
        if self.bleaching_rate == 0.0 and tau_raw is not None:
            tau_frames = float(tau_raw)
            if not np.isfinite(tau_frames) or tau_frames <= 0.0:
                raise ValueError(
                    "fluorescence_photobleach_tau_frames must be positive when supplied."
                )
            self.bleaching_rate = 1.0 / tau_frames
        for name, val in (
            ("fluorescence_blinking_rate_per_frame", self.blinking_rate),
            ("fluorescence_recovery_rate_per_frame", self.recovery_rate),
            ("fluorescence_bleaching_rate_per_frame", self.bleaching_rate),
        ):
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"PARAMS[{name!r}] must be finite and non-negative; got {val!r}.")
        status = str(param_value(params, 'fluorescence_reference_status')).strip().lower()
        if status not in {"physics_based_unvalidated", "reference_validated"}:
            raise ValueError("fluorescence_reference_status must be physics_based_unvalidated or reference_validated.")
        if status == "reference_validated" and not param_value(params, "fluorescence_reference_validation_hash"):
            raise ValueError("reference_validated fluorescence requires fluorescence_reference_validation_hash.")
        self.reference_status = status
        self.validation_status = (
            "external_artifact_required"
            if status == "reference_validated"
            else "diagnostic_only"
        )
        self._psf_cache: dict[tuple[int, int], np.ndarray] = {}
        self.allow_psf_fallback = bool(param_value(params, 'fluorescence_allow_psf_fallback'))
        self._last_psf_backend = "not_evaluated"
        self._last_psf_error: str | None = None

    def _fallback_gaussian_psf(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = int(shape[-2]), int(shape[-1])
        y = np.arange(h, dtype=float) - h // 2
        x = np.arange(w, dtype=float) - w // 2
        yy, xx = np.meshgrid(y, x, indexing="ij")
        sigma = max(self.base_emission_sigma_px, 0.5)
        psf = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
        total = float(psf.sum())
        return psf / total if total > 0.0 else np.full((h, w), 1.0 / float(h * w))

    def _vectorial_psf(self, shape: tuple[int, int]) -> np.ndarray:
        key = (int(shape[-2]), int(shape[-1]))
        cached = self._psf_cache.get(key)
        if cached is not None:
            return cached
        if vectorial_optics is not None:
            try:
                params = dict(self.params)
                params["wavelength_nm"] = float(param_value(params, "fluorescence_emission_wavelength_nm"))
                configured_samples = param_value(params, "vectorial_pupil_samples")
                samples = int(configured_samples if configured_samples is not None else param_value(params, "pupil_samples"))
                params["vectorial_pupil_samples"] = max(samples, max(key))
                stack = vectorial_optics.compute_vectorial_debye_psf(params, [0.0])
                intensity = sum(np.abs(stack[name][0]) ** 2 for name in ("Ex", "Ey", "Ez"))
                psf = np.asarray(intensity, dtype=float)
                self._last_psf_backend = "vectorial_debye"
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
                psf = self._fallback_gaussian_psf(key)
        else:
            if not self.allow_psf_fallback:
                raise FluorescencePhotophysicsError(
                    "Vectorial fluorescence backend requires vectorial_optics. Set "
                    "fluorescence_allow_psf_fallback=True only for an explicit "
                    "Gaussian-proxy diagnostic run."
                )
            self._last_psf_backend = "fallback_gaussian"
            self._last_psf_error = "vectorial_optics module unavailable"
            psf = self._fallback_gaussian_psf(key)
        psf = np.maximum(np.asarray(psf, dtype=float), 0.0)
        if psf.shape != key:
            # Center-crop or pad to the requested render shape without adding a scipy dependency.
            out = np.zeros(key, dtype=float)
            h = min(key[0], psf.shape[0]); w = min(key[1], psf.shape[1])
            sy = (psf.shape[0] - h) // 2; sx = (psf.shape[1] - w) // 2
            dy = (key[0] - h) // 2; dx = (key[1] - w) // 2
            out[dy:dy+h, dx:dx+w] = psf[sy:sy+h, sx:sx+w]
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
            psf = self._fallback_gaussian_psf(key)
            total = float(psf.sum())
        psf = psf / total
        self._psf_cache[key] = psf
        return psf

    def _fft_convolve_reflect_safe(self, source: np.ndarray, psf: np.ndarray) -> np.ndarray:
        src = np.asarray(source, dtype=float)
        kernel = np.fft.ifftshift(np.asarray(psf, dtype=float))
        out = np.real(np.fft.ifft2(np.fft.fft2(src) * np.fft.fft2(kernel, s=src.shape)))
        return np.maximum(out, 0.0)

    def _state_factor(self, frame_index: int) -> float:
        t = max(float(frame_index), 0.0)
        bleached_survival = np.exp(-self.bleaching_rate * t)
        if self.blinking_rate <= 0.0:
            emitting_fraction = 1.0
        else:
            total = max(self.blinking_rate + self.recovery_rate, 1e-12)
            dark_equilibrium = self.blinking_rate / total
            emitting_fraction = 1.0 - dark_equilibrium * (1.0 - np.exp(-total * t))
        return float(np.clip(bleached_survival * emitting_fraction, 0.0, 1.0))

    def source_to_detector_counts(
        self,
        source: np.ndarray,
        *,
        frame_index: int = 0,
        tirf_excitation_factor: float | None = None,
        include_background: bool = True,
    ) -> np.ndarray:
        src = np.maximum(np.asarray(source, dtype=float), 0.0)
        psf = self._vectorial_psf(src.shape)
        emission_density = self._fft_convolve_reflect_safe(src, psf)
        photons = emission_density * self.quantum_yield * self.excitation_scale * self._state_factor(frame_index)
        if self.photons_per_fluorophore is not None:
            photons = photons * self.photons_per_fluorophore
        elif self.legacy_photon_count_scale is not None:
            photons = photons * self.legacy_photon_count_scale
        if tirf_excitation_factor is not None:
            photons = photons * float(tirf_excitation_factor)
        detected = photons * self.collection_efficiency * self.detector_qe
        if include_background:
            detected = detected + self.uniform_background
        if not np.all(np.isfinite(detected)):
            raise FloatingPointError("vectorial photophysics fluorescence produced non-finite counts.")
        return np.maximum(detected, 0.0)

    def metadata(self, params: dict | None = None, *, tirf_depth_nm: float | None = None) -> dict[str, Any]:
        p = self.params if params is None else params
        if self.photons_per_fluorophore is not None:
            fidelity = "high_fidelity"
        else:
            fidelity = "physics_based"
        validation = self.validation_status
        meta = FluorescenceBackendMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level=fidelity,
            algorithm="vectorial_debye_psf_with_mean_field_fluorophore_state_model",
            excitation_wavelength_nm=float(param_value(p, "fluorescence_excitation_wavelength_nm")),
            emission_wavelength_nm=float(param_value(p, "fluorescence_emission_wavelength_nm")),
            quantum_yield=self.quantum_yield,
            collection_efficiency=self.collection_efficiency,
            detector_qe=self.detector_qe,
            photons_per_fluorophore_per_frame=self.photons_per_fluorophore,
            uniform_background_counts_per_pixel=self.uniform_background,
            effective_detection_factor=float(
                self.quantum_yield
                * self.excitation_scale
                * self.collection_efficiency
                * self.detector_qe
                * (
                    self.photons_per_fluorophore
                    if self.photons_per_fluorophore is not None
                    else (
                        self.legacy_photon_count_scale
                        if self.legacy_photon_count_scale is not None
                        else 1.0
                    )
                )
            ),
            blinking_rate_per_frame=self.blinking_rate,
            recovery_rate_per_frame=self.recovery_rate,
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
            fluorescence_emitter_orientation_model="isotropic_scalar_density",
            emission_psf_vectorial_model="vectorial_debye_detection_intensity",
            emission_psf_axial_model="single_focal_plane_z0",
            reference_backend_metadata={
                "reference_status": self.reference_status,
                "reference_validation_hash": p.get("fluorescence_reference_validation_hash"),
                "claim_maturity_gate": validation,
            },
            validation_status=validation,
            comparison_contract_id=p.get("comparison_contract_id", "Contract-NR"),
            fluorescence_legacy_photon_count_scale_supplied=bool(self.legacy_photon_count_scale_supplied),
            fluorescence_legacy_photon_count_scale=self.legacy_photon_count_scale,
            fluorescence_background_units="detected_counts_per_pixel",
            fluorescence_photon_count_scale_contract="emitted_photon_scale_before_collection_and_qe",
            emission_psf_backend=self._last_psf_backend,
            emission_psf_boundary_mode="circular_fft_convolution",
            emission_psf_fallback_error=self._last_psf_error,
            fluorescence_photobleach_tau_frames_consumed=bool(
                p.get("fluorescence_photobleach_tau_frames", None) is not None
            ),
            fluorescence_photon_budget_source=(
                "fluorescence_photons_per_fluorophore_per_frame"
                if self.photons_per_fluorophore is not None
                else (
                    "fluorescence_photon_count_scale"
                    if self.legacy_photon_count_scale is not None
                    else "source_map_density_only"
                )
            ),
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=p,
            backend_name=self.backend_mode,
            equations_or_model_family="vectorial_photophysics_psf_with_mean_field_fluorophore_kinetics",
            implemented_approximation_level=fidelity,
            native_operating_assumptions="vectorial detection PSF with scalar isotropic emitter density and deterministic mean-field fluorophore occupancy",
            comparison_contract_id=p.get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=p.get("artifact_provenance_id", None),
        )


__all__ = ["FluorescencePhotophysicsError", "VectorialPhotophysicsFluorescenceBackend"]
