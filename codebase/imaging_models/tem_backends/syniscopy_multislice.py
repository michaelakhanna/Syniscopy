"""Syniscopy-native multislice TEM backend hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from backend_fidelity import attach_backend_fidelity_metadata
from config import param_value


class HighFidelityTEMBackendError(RuntimeError):
    """Raised when a requested high-fidelity TEM backend cannot run honestly."""


@dataclass(frozen=True)
class TEMBackendMetadata:
    backend_mode: str
    backend_fidelity_level: str
    algorithm: str
    voltage_kV: float
    dose_per_pixel: float
    slice_thickness_nm: float | None
    potential_source: str
    reference_status: str
    reference_validation_hash: str | None = None
    objective_aperture_mrad: float | None = None
    multislice_convergence_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tem_backend": self.backend_mode,
            "backend_mode": self.backend_mode,
            "backend_fidelity_level": self.backend_fidelity_level,
            "algorithm": self.algorithm,
            "voltage_kV": self.voltage_kV,
            "dose_per_pixel": self.dose_per_pixel,
            "slice_thickness_nm": self.slice_thickness_nm,
            "objective_aperture_mrad": self.objective_aperture_mrad,
            "potential_source": self.potential_source,
            "reference_status": self.reference_status,
            "reference_validation_hash": self.reference_validation_hash,
            "multislice_convergence_diagnostics": self.multislice_convergence_diagnostics,
            "multislice_slices_input": None,
        }


def _require_validation_for_reference_status(params: dict, *, key_prefix: str) -> tuple[str, str | None]:
    status = str(param_value(params, f"{key_prefix}_reference_status")).strip().lower()
    if status not in {"physics_based_unvalidated", "reference_validated"}:
        raise ValueError(
            f"PARAMS['{key_prefix}_reference_status'] must be 'physics_based_unvalidated' "
            f"or 'reference_validated'; got {status!r}."
        )
    validation_hash = param_value(params, f"{key_prefix}_reference_validation_hash")
    if status == "reference_validated" and not validation_hash:
        raise ValueError(
            f"PARAMS['{key_prefix}_reference_status']='reference_validated' requires "
            f"PARAMS['{key_prefix}_reference_validation_hash']."
        )
    return status, None if validation_hash is None else str(validation_hash)


class SyniscopyMultisliceTEMBackend:
    """Syniscopy-owned split-step multislice TEM backend.

    This backend implements the forward model in Syniscopy source code rather than
    importing an external simulator. It is a physics-based multislice backend:
    projected electrostatic phase is split over slices, Fresnel propagated, and
    read out through a coherent objective transfer function. It is labelled
    ``physics_based_unvalidated`` unless explicit reference validation metadata is
    supplied by the caller.
    """

    backend_mode = "syniscopy_multislice"

    def __init__(
        self,
        params: dict,
        *,
        pixel_size_m: float,
        electron_wavelength_m: float,
        Cs_mm: float,
        defocus_m: float,
        partial_coherence_alpha_mrad: float,
        dose_per_pixel: float,
        default_slice_count: int,
        default_slice_thickness_nm: float | None,
        default_objective_aperture_mrad: float | None = None,
    ) -> None:
        self.pixel_size_m = float(pixel_size_m)
        self.lambda_m = float(electron_wavelength_m)
        self.Cs_m = 1.0e-3 * float(Cs_mm)
        self.defocus_m = float(defocus_m)
        self.alpha_rad = 1.0e-3 * float(partial_coherence_alpha_mrad)
        self.dose_per_pixel = float(dose_per_pixel)
        self.n_slices = int(param_value(params, "tem_multislice_slices"))
        if self.n_slices <= 0:
            raise ValueError("PARAMS['tem_multislice_slices'] must be positive for syniscopy_multislice.")
        raw_slice = param_value(params, "tem_slice_thickness_nm")
        self.slice_thickness_nm = None if raw_slice is None else float(raw_slice)
        if self.slice_thickness_nm is not None and self.slice_thickness_nm <= 0.0:
            raise ValueError("PARAMS['tem_slice_thickness_nm'] must be positive for syniscopy_multislice.")
        raw_aperture_mrad = param_value(params, "tem_objective_aperture_mrad")
        self.objective_aperture_mrad = None if raw_aperture_mrad is None else float(raw_aperture_mrad)
        if self.objective_aperture_mrad is not None:
            if not np.isfinite(self.objective_aperture_mrad) or self.objective_aperture_mrad <= 0.0:
                raise ValueError("PARAMS['tem_objective_aperture_mrad'] must be positive when set.")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0.0:
            raise ValueError("pixel_size_m must be positive and finite for syniscopy_multislice.")
        if not np.isfinite(self.lambda_m) or self.lambda_m <= 0.0:
            raise ValueError("electron wavelength must be positive and finite for syniscopy_multislice.")
        if not np.isfinite(self.dose_per_pixel) or self.dose_per_pixel < 0.0:
            raise ValueError("tem_dose_per_pixel must be finite and non-negative for syniscopy_multislice.")
        self.potential_source = str(param_value(params, 'tem_potential_source'))
        self.reference_status, self.reference_validation_hash = _require_validation_for_reference_status(params, key_prefix="tem")
        self.validation_status = (
            "external_artifact_required"
            if self.reference_status == "reference_validated"
            else "diagnostic_only"
        )
        self._transfer_cache: dict[tuple[int, int], np.ndarray] = {}

    def _frequency_grid(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = int(shape[-2]), int(shape[-1])
        fx = np.fft.fftfreq(w, d=self.pixel_size_m)
        fy = np.fft.fftfreq(h, d=self.pixel_size_m)
        kx, ky = np.meshgrid(fx, fy, indexing="xy")
        k = np.sqrt(kx * kx + ky * ky)
        return kx, ky, k

    def _objective_transfer(self, shape: tuple[int, int]) -> np.ndarray:
        key = (int(shape[-2]), int(shape[-1]))
        cached = self._transfer_cache.get(key)
        if cached is not None:
            return cached
        _, _, k = self._frequency_grid(key)
        chi = (np.pi * (self.lambda_m ** 3) * self.Cs_m * 0.5) * k ** 4 - (np.pi * self.lambda_m * self.defocus_m) * k ** 2
        if self.alpha_rad == 0.0:
            envelope = np.ones_like(k)
        else:
            arg = (self.Cs_m * self.lambda_m ** 2) * k ** 3 - self.defocus_m * k
            envelope = np.exp(-(np.pi * self.alpha_rad) ** 2 * arg ** 2)
        if self.objective_aperture_mrad is None:
            aperture = np.ones_like(k)
        else:
            k_max = (1.0e-3 * self.objective_aperture_mrad) / self.lambda_m
            aperture = (k <= k_max).astype(float)
            if np.all(~np.asarray(aperture, dtype=bool)):
                raise ValueError("tem_objective_aperture_mrad is too small for this sampling grid.")
        transfer = aperture * envelope * np.exp(-1j * chi)
        self._transfer_cache[key] = transfer
        return transfer

    def _prepare_potential_slices(self, projected_phase: np.ndarray) -> tuple[np.ndarray, float]:
        source = np.asarray(projected_phase, dtype=float)
        if source.ndim not in (2, 3):
            raise ValueError("projected_phase must be 2D projected-phase or 3D potential-stack.")
        if not np.all(np.isfinite(source)):
            raise ValueError("projected_phase contains non-finite values.")
        if source.ndim == 2:
            if self.n_slices <= 0:
                raise ValueError("n_slices must be positive.")
            dz_nm = 0.0 if self.slice_thickness_nm is None else float(self.slice_thickness_nm)
            return np.broadcast_to((source / float(self.n_slices)), (self.n_slices, *source.shape)), dz_nm
        # 3D potential stack: each slice holds integrated projected potential (sigma * V * dz).
        if self.slice_thickness_nm is None:
            raise ValueError(
                "PARAMS['tem_slice_thickness_nm'] must be set when syniscopy_multislice "
                "receives a 3D potential stack."
            )
        if source.shape[0] <= 0:
            raise ValueError("projected_phase stack must contain at least one slice.")
        if source.shape[0] != self.n_slices:
            raise ValueError(
                "projected_phase stack depth must match PARAMS['tem_multislice_slices']; "
                f"got {source.shape[0]} but n_slices={self.n_slices}."
            )
        return source, float(self.slice_thickness_nm)

    def _convergence_diagnostics(self, shape: tuple[int, int]) -> dict[str, Any]:
        return {
            "backend_name": self.backend_mode,
            "shape_pixels": list(map(int, shape)),
            "pixel_pitch_nm": float(self.pixel_size_m * 1.0e9),
            "slices": int(self.n_slices),
            "slice_thickness_nm": None if self.slice_thickness_nm is None else float(self.slice_thickness_nm),
            "objective_aperture_mrad": self.objective_aperture_mrad,
            "recommended_slices_for_stability": max(4, int(np.clip(self.n_slices, 1, np.iinfo(np.int32).max))),
        }

    @staticmethod
    def _shape_from_params(params: dict | None) -> tuple[int, int]:
        source = params or {}
        py = source.get("render_shape_y", None)
        px = source.get("render_shape_x", None)
        if py is None or px is None:
            return (0, 0)
        try:
            return (int(py), int(px))
        except (TypeError, ValueError):
            return (0, 0)

    def _fresnel_propagate(self, psi: np.ndarray, dz_nm: float) -> np.ndarray:
        dz_m = 1.0e-9 * float(dz_nm)
        if dz_m == 0.0:
            return psi
        kx, ky, _ = self._frequency_grid(psi.shape)
        kernel = np.exp(-1j * np.pi * self.lambda_m * dz_m * (kx * kx + ky * ky))
        return np.fft.ifft2(np.fft.fft2(psi) * kernel)

    def exit_wave_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        source_stack, dz_nm = self._prepare_potential_slices(projected_phase)
        shape = tuple(source_stack.shape[-2:])
        if len(shape) != 2:
            raise ValueError("TEM potential stack must expose 2D transverse shape.")
        psi = np.ones(shape, dtype=np.complex128)
        for slice_phase in source_stack:
            psi *= np.exp(1j * np.asarray(slice_phase, dtype=float))
            psi = self._fresnel_propagate(psi, dz_nm)
        return psi

    def intensity_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        psi = self.exit_wave_from_projected_phase(projected_phase)
        image_wave = np.fft.ifft2(np.fft.fft2(psi) * self._objective_transfer(psi.shape))
        intensity = np.abs(image_wave) ** 2
        if not np.all(np.isfinite(intensity)):
            raise FloatingPointError("syniscopy_multislice produced non-finite TEM intensity.")
        return np.maximum(intensity, 0.0)

    def contrast_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        return self.intensity_from_projected_phase(projected_phase) - 1.0

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        voltage = float((params or {}).get("tem_acceleration_kV", np.nan))
        if not np.isfinite(voltage):
            voltage = float("nan")
        shape_pixels = self._shape_from_params(params)
        fidelity = "high_fidelity"
        validation = self.validation_status
        meta = TEMBackendMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level=fidelity,
            algorithm="syniscopy_split_step_multislice_with_coherent_objective_transfer",
            voltage_kV=voltage,
            dose_per_pixel=self.dose_per_pixel,
            slice_thickness_nm=self.slice_thickness_nm,
            objective_aperture_mrad=self.objective_aperture_mrad,
            potential_source=self.potential_source,
            reference_status=self.reference_status,
            reference_validation_hash=self.reference_validation_hash,
            multislice_convergence_diagnostics=self._convergence_diagnostics(shape_pixels),
        ).to_dict()
        meta["multislice_slices_input"] = int(self.n_slices)
        meta.update(
            kind="tem_multislice",
            backend_fidelity_level=fidelity,
            fidelity_label=(
                "syniscopy_multislice_physics_based"
                if self.reference_status != "reference_validated"
                else "syniscopy_multislice_external_artifact_required"
            ),
            forward_observable="|objective-transfer(multislice exit wave)|^2",
            multislice_slices=self.n_slices,
            convergence_status="production_grid_only",
            objective_aperture_mrad=self.objective_aperture_mrad,
            convergence_diagnostics=self._convergence_diagnostics(shape_pixels),
            validation_status=validation,
            comparison_contract_id=(params or {}).get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=(params or {}).get("artifact_provenance_id", None),
            reference_backend_metadata={
                "reference_status": self.reference_status,
                "reference_validation_hash": self.reference_validation_hash,
                "claim_maturity_gate": validation,
            },
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=params,
            backend_name=self.backend_mode,
            equations_or_model_family="syniscopy_split_step_multislice_electron_transfer",
            implemented_approximation_level=fidelity,
            native_operating_assumptions="electron split-step propagation in weak-phase projected potential",
            comparison_contract_id=(params or {}).get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=(params or {}).get("artifact_provenance_id", None),
        )


__all__ = [
    "HighFidelityTEMBackendError",
    "TEMBackendMetadata",
    "SyniscopyMultisliceTEMBackend",
]
