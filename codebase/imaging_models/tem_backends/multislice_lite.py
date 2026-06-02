"""Reduced split-step multislice-lite TEM backend."""

from __future__ import annotations

import numpy as np

from .ctf_proxy import CTFProxyTEMBackend


class MultisliceLiteTEMBackend:
    backend_mode = "multislice_lite"

    def __init__(
        self,
        *,
        ctf_backend: CTFProxyTEMBackend,
        electron_wavelength_m: float,
        pixel_size_m: float,
        multislice_slices: int,
        slice_thickness_nm: float | None,
    ) -> None:
        self.ctf_backend = ctf_backend
        self.electron_wavelength_m = float(electron_wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.multislice_slices = max(int(multislice_slices), 1)
        self.slice_thickness_nm = None if slice_thickness_nm is None else float(slice_thickness_nm)

    def _fresnel_propagate_exit_wave(self, psi: np.ndarray, dz_nm: float) -> np.ndarray:
        dz_m = 1.0e-9 * float(dz_nm)
        if dz_m == 0.0:
            return psi
        h, w = psi.shape[-2], psi.shape[-1]
        fx = np.fft.fftfreq(w, d=self.pixel_size_m)
        fy = np.fft.fftfreq(h, d=self.pixel_size_m)
        kx, ky = np.meshgrid(fx, fy, indexing="xy")
        kernel = np.exp(-1j * np.pi * self.electron_wavelength_m * dz_m * (kx * kx + ky * ky))
        return np.fft.ifft2(np.fft.fft2(psi) * kernel)

    def intensity_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        source = np.asarray(projected_phase, dtype=float)
        dz_nm = 0.0 if self.slice_thickness_nm is None else float(self.slice_thickness_nm)
        psi = np.ones(source.shape, dtype=complex)
        phase_slice = source / float(self.multislice_slices)
        for _ in range(self.multislice_slices):
            psi *= np.exp(1j * phase_slice)
            psi = self._fresnel_propagate_exit_wave(psi, dz_nm)
        projected_exit_phase = np.angle(psi)
        return np.maximum(1.0 + self.ctf_backend.apply_ctf(projected_exit_phase), 0.0)

    def contrast_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        return self.intensity_from_projected_phase(projected_phase) - 1.0


__all__ = ["MultisliceLiteTEMBackend"]
