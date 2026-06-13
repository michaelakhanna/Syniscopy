"""Physical multislice TEM backend.

This backend consumes Syniscopy's projected electrostatic phase source. A 2D
source is split evenly across slices; a 3D source is interpreted as per-slice
integrated phase. Each slice applies Cowley-Moodie transmission followed by
Fresnel free-space propagation. A 2/3 reciprocal-space bandlimit is enforced
after every transmission/propagation step before the objective transfer is
applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import TemSettings

from .ctf_proxy import CTFProxyTEMBackend
from .syniscopy_multislice import TEMBackendMetadata


class PhysicalMultisliceTEMBackend:
    backend_mode = "multislice_physical"

    def __init__(
        self,
        *,
        ctf_backend: CTFProxyTEMBackend,
        tem_settings: TemSettings,
        electron_wavelength_m: float,
        pixel_size_m: float,
        dose_per_pixel: float,
        default_slice_count: int,
    ) -> None:
        self.ctf_backend = ctf_backend
        self.lambda_m = float(electron_wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.dose_per_pixel = float(dose_per_pixel)
        self.voltage_kV = tem_settings.acceleration_kV
        self.n_slices = int(default_slice_count)
        if self.n_slices <= 0:
            raise ValueError("Effective multislice slice count must be positive for multislice_physical.")
        self.slice_thickness_nm = tem_settings.slice_thickness_nm
        if self.slice_thickness_nm is not None and (
            not np.isfinite(self.slice_thickness_nm) or self.slice_thickness_nm <= 0.0
        ):
            raise ValueError("parameters['tem_slice_thickness_nm'] must be positive for multislice_physical.")
        self.objective_aperture_mrad = tem_settings.objective_aperture_mrad
        if self.objective_aperture_mrad is not None:
            if not np.isfinite(self.objective_aperture_mrad) or self.objective_aperture_mrad <= 0.0:
                raise ValueError("parameters['tem_objective_aperture_mrad'] must be positive when set.")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0.0:
            raise ValueError("pixel_size_m must be positive and finite for multislice_physical.")
        if not np.isfinite(self.lambda_m) or self.lambda_m <= 0.0:
            raise ValueError("electron wavelength must be positive and finite for multislice_physical.")
        if not np.isfinite(self.dose_per_pixel) or self.dose_per_pixel < 0.0:
            raise ValueError("tem_dose_per_pixel must be finite and non-negative for multislice_physical.")
        self.potential_source = tem_settings.potential_source
        self.reference_status = tem_settings.reference_status
        self.reference_validation_hash = tem_settings.reference_validation_hash
        self.validation_status = (
            "validated"
            if self.reference_status == "reference_validated"
            else "diagnostic_only"
        )
        self._fresnel_cache: dict[tuple[int, int, float], np.ndarray] = {}
        self._bandlimit_cache: dict[tuple[int, int], np.ndarray] = {}
        self._objective_transfer_cache: dict[tuple[int, int], np.ndarray] = {}

    def _frequency_grid(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = int(shape[-2]), int(shape[-1])
        fx = np.fft.fftfreq(w, d=self.pixel_size_m)
        fy = np.fft.fftfreq(h, d=self.pixel_size_m)
        kx, ky = np.meshgrid(fx, fy, indexing="xy")
        k = np.sqrt(kx * kx + ky * ky)
        return kx, ky, k

    def _bandlimit_mask(self, shape: tuple[int, int]) -> np.ndarray:
        key = (int(shape[-2]), int(shape[-1]))
        cached = self._bandlimit_cache.get(key)
        if cached is not None:
            return cached
        kx, ky, k = self._frequency_grid(key)
        del kx, ky
        k_nyquist = 0.5 / self.pixel_size_m
        mask = k <= (2.0 / 3.0) * k_nyquist
        self._bandlimit_cache[key] = mask
        return mask

    def _bandlimit_wave(self, psi: np.ndarray) -> np.ndarray:
        mask = self._bandlimit_mask(tuple(psi.shape[-2:]))
        return np.fft.ifft2(np.fft.fft2(psi) * mask)

    def _fresnel_kernel(self, shape: tuple[int, int], dz_nm: float) -> np.ndarray:
        dz_m = 1.0e-9 * float(dz_nm)
        key = (int(shape[-2]), int(shape[-1]), dz_m)
        cached = self._fresnel_cache.get(key)
        if cached is not None:
            return cached
        kx, ky, _ = self._frequency_grid((key[0], key[1]))
        kernel = np.exp(-1j * np.pi * self.lambda_m * dz_m * (kx * kx + ky * ky))
        self._fresnel_cache[key] = kernel
        return kernel

    def _fresnel_propagate(self, psi: np.ndarray, dz_nm: float) -> np.ndarray:
        dz_nm = float(dz_nm)
        if dz_nm == 0.0:
            return self._bandlimit_wave(psi)
        kernel = self._fresnel_kernel(tuple(psi.shape[-2:]), dz_nm)
        return np.fft.ifft2(np.fft.fft2(psi) * kernel * self._bandlimit_mask(tuple(psi.shape[-2:])))

    def _objective_transfer(self, shape: tuple[int, int]) -> np.ndarray:
        key = (int(shape[-2]), int(shape[-1]))
        cached = self._objective_transfer_cache.get(key)
        if cached is not None:
            return cached
        transfer = self.ctf_backend.objective_transfer(
            key,
            objective_aperture_mrad=self.objective_aperture_mrad,
        )
        self._objective_transfer_cache[key] = transfer
        return transfer

    def _prepare_phase_slices(self, projected_phase: np.ndarray) -> tuple[np.ndarray, float]:
        source = np.asarray(projected_phase, dtype=float)
        if source.ndim not in {2, 3}:
            raise ValueError("projected_phase must be a 2D phase map or a 3D per-slice phase stack.")
        if not np.all(np.isfinite(source)):
            raise ValueError("projected_phase contains non-finite values.")
        if source.ndim == 2:
            dz_nm = 0.0 if self.slice_thickness_nm is None else float(self.slice_thickness_nm)
            return np.broadcast_to(source / float(self.n_slices), (self.n_slices, *source.shape)), dz_nm
        if source.shape[0] != self.n_slices:
            raise ValueError(
                "projected_phase stack depth must match the effective multislice slice count; "
                f"got {source.shape[0]} but n_slices={self.n_slices}."
            )
        if self.slice_thickness_nm is None:
            raise ValueError(
                "parameters['tem_slice_thickness_nm'] must be set for 3D multislice_physical input."
            )
        return source, float(self.slice_thickness_nm)

    def exit_wave_after_transmission_only(self, projected_phase: np.ndarray) -> np.ndarray:
        source_stack, _ = self._prepare_phase_slices(projected_phase)
        return np.exp(1j * np.sum(source_stack, axis=0))

    def exit_wave_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        source_stack, dz_nm = self._prepare_phase_slices(projected_phase)
        shape = tuple(source_stack.shape[-2:])
        psi = np.ones(shape, dtype=np.complex128)
        if dz_nm != 0.0:
            psi = self._fresnel_propagate(psi, 0.5 * dz_nm)
        for idx, slice_phase in enumerate(source_stack):
            psi *= np.exp(1j * np.asarray(slice_phase, dtype=float))
            psi = self._bandlimit_wave(psi)
            step_nm = 0.5 * dz_nm if idx == source_stack.shape[0] - 1 else dz_nm
            psi = self._fresnel_propagate(psi, step_nm)
        if not np.all(np.isfinite(psi.real)) or not np.all(np.isfinite(psi.imag)):
            raise FloatingPointError("multislice_physical produced non-finite exit wave.")
        return psi

    def image_wave_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        psi = self.exit_wave_from_projected_phase(projected_phase)
        image_wave = np.fft.ifft2(np.fft.fft2(psi) * self._objective_transfer(tuple(psi.shape[-2:])))
        if not np.all(np.isfinite(image_wave.real)) or not np.all(np.isfinite(image_wave.imag)):
            raise FloatingPointError("multislice_physical produced non-finite image wave.")
        return image_wave

    def intensity_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        intensity = np.abs(self.image_wave_from_projected_phase(projected_phase)) ** 2
        if not np.all(np.isfinite(intensity)):
            raise FloatingPointError("multislice_physical produced non-finite TEM intensity.")
        return np.maximum(intensity, 0.0)

    def contrast_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        return self.intensity_from_projected_phase(projected_phase) - 1.0

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

    def _convergence_diagnostics(self, shape: tuple[int, int]) -> dict[str, Any]:
        return {
            "backend_name": self.backend_mode,
            "shape_pixels": list(map(int, shape)),
            "pixel_pitch_nm": float(self.pixel_size_m * 1.0e9),
            "slices": int(self.n_slices),
            "slice_thickness_nm": None if self.slice_thickness_nm is None else float(self.slice_thickness_nm),
            "bandlimit_policy": "two_thirds_circular_nyquist",
            "objective_aperture_mrad": self.objective_aperture_mrad,
            "recommended_slices_for_stability": max(8, int(self.n_slices)),
        }

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        params_source = params or {}
        shape_pixels = self._shape_from_params(params_source)
        fidelity = (
            "reference_validated"
            if self.reference_status == "reference_validated"
            else "physics_based"
        )
        meta = TEMBackendMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level=fidelity,
            algorithm="cowley_moodie_multislice_projected_phase_with_fresnel_propagation_and_ctf_readout",
            voltage_kV=self.voltage_kV,
            dose_per_pixel=self.dose_per_pixel,
            slice_thickness_nm=self.slice_thickness_nm,
            objective_aperture_mrad=self.objective_aperture_mrad,
            potential_source=self.potential_source,
            reference_status=self.reference_status,
            reference_validation_hash=self.reference_validation_hash,
            multislice_convergence_diagnostics=self._convergence_diagnostics(shape_pixels),
        ).to_dict()
        meta.update(
            kind="tem_multislice_physical",
            backend_fidelity_level=fidelity,
            fidelity_label="multislice_physical_reference_validated"
            if self.reference_status == "reference_validated"
            else "multislice_physical_unvalidated",
            forward_observable="|objective-transfer(multislice exit wave)|^2",
            multislice_slices=self.n_slices,
            convergence_status="production_grid_only",
            objective_aperture_mrad=self.objective_aperture_mrad,
            convergence_diagnostics=self._convergence_diagnostics(shape_pixels),
            validation_status=self.validation_status,
            comparison_contract_id=params_source.get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=params_source.get("artifact_provenance_id", None),
            reference_backend_metadata={
                "reference_status": self.reference_status,
                "reference_validation_hash": self.reference_validation_hash,
                "claim_maturity_gate": self.validation_status,
            },
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=params_source,
            backend_name=self.backend_mode,
            equations_or_model_family="cowley_moodie_kirkland_multislice_electron_transfer",
            implemented_approximation_level=fidelity,
            native_operating_assumptions=(
                "amorphous projected MIP phase slices; Fresnel propagation; "
                "two-thirds reciprocal-space bandlimit; coherent objective transfer"
            ),
            comparison_contract_id=params_source.get("comparison_contract_id", "Contract-NR"),
            artifact_provenance_id=params_source.get("artifact_provenance_id", None),
        )


__all__ = ["PhysicalMultisliceTEMBackend"]
