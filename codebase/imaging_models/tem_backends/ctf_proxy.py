"""Weak-phase CTF proxy TEM backend."""

from __future__ import annotations

import numpy as np


class CTFProxyTEMBackend:
    backend_mode = "ctf_proxy"

    def __init__(
        self,
        *,
        pixel_size_m: float,
        electron_wavelength_m: float,
        Cs_mm: float,
        defocus_m: float,
        partial_coherence_alpha_mrad: float,
        objective_aperture_mrad: float | None = None,
    ) -> None:
        self.pixel_size_m = float(pixel_size_m)
        self.electron_wavelength_m = float(electron_wavelength_m)
        self.Cs_mm = float(Cs_mm)
        self.defocus_m = float(defocus_m)
        self.partial_coherence_alpha_mrad = float(partial_coherence_alpha_mrad)
        self.objective_aperture_mrad = None if objective_aperture_mrad is None else float(objective_aperture_mrad)
        if self.objective_aperture_mrad is not None and (
            not np.isfinite(self.objective_aperture_mrad) or self.objective_aperture_mrad <= 0.0
        ):
            raise ValueError("objective_aperture_mrad must be positive and finite when set.")
        self._ctf_cache: dict[tuple, np.ndarray] = {}

    def _chi(self, k: np.ndarray) -> np.ndarray:
        lam = self.electron_wavelength_m
        Cs_m = 1.0e-3 * self.Cs_mm
        df_m = self.defocus_m
        return (np.pi * (lam ** 3) * Cs_m * 0.5) * k ** 4 - (np.pi * lam * df_m) * k ** 2

    def _envelope(self, k: np.ndarray) -> np.ndarray:
        alpha_rad = 1.0e-3 * self.partial_coherence_alpha_mrad
        if alpha_rad == 0.0:
            return np.ones_like(k)
        lam = self.electron_wavelength_m
        Cs_m = 1.0e-3 * self.Cs_mm
        df_m = self.defocus_m
        arg = (Cs_m * lam ** 2) * k ** 3 - df_m * k
        return np.exp(-(np.pi * alpha_rad) ** 2 * arg ** 2)

    def _objective_aperture(
        self,
        k: np.ndarray,
        objective_aperture_mrad: float | None = None,
    ) -> np.ndarray:
        aperture_mrad = self.objective_aperture_mrad if objective_aperture_mrad is None else objective_aperture_mrad
        if aperture_mrad is None:
            return np.ones_like(k)
        aperture_mrad = float(aperture_mrad)
        if not np.isfinite(aperture_mrad) or aperture_mrad <= 0.0:
            raise ValueError("objective_aperture_mrad must be positive and finite when set.")
        k_max = (1.0e-3 * aperture_mrad) / self.electron_wavelength_m
        aperture = (k <= k_max).astype(float)
        if not np.any(aperture):
            raise ValueError("objective_aperture_mrad is too small for this TEM sampling grid.")
        return aperture

    def ctf(self, shape: tuple) -> np.ndarray:
        cached = self._ctf_cache.get(shape)
        if cached is not None:
            return cached
        H, W = shape[-2], shape[-1]
        fx = np.fft.fftfreq(W, d=self.pixel_size_m)
        fy = np.fft.fftfreq(H, d=self.pixel_size_m)
        KX, KY = np.meshgrid(fx, fy, indexing="xy")
        k = np.sqrt(KX ** 2 + KY ** 2)
        ctf = 2.0 * np.sin(self._chi(k)) * self._envelope(k) * self._objective_aperture(k)
        if shape != (H, W):
            ctf = ctf.reshape(shape)
        self._ctf_cache[shape] = ctf
        return ctf

    def objective_transfer(
        self,
        shape: tuple[int, int],
        *,
        objective_aperture_mrad: float | None = None,
    ) -> np.ndarray:
        H, W = int(shape[-2]), int(shape[-1])
        fx = np.fft.fftfreq(W, d=self.pixel_size_m)
        fy = np.fft.fftfreq(H, d=self.pixel_size_m)
        KX, KY = np.meshgrid(fx, fy, indexing="xy")
        k = np.sqrt(KX ** 2 + KY ** 2)
        aperture = self._objective_aperture(k, objective_aperture_mrad)
        return aperture * self._envelope(k) * np.exp(-1j * self._chi(k))

    def apply_ctf(self, projected_phase: np.ndarray) -> np.ndarray:
        source = np.asarray(projected_phase, dtype=float)
        ctf = self.ctf(source.shape)
        return np.real(np.fft.ifft2(ctf * np.fft.fft2(source)))

    def intensity_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        return np.maximum(1.0 + self.apply_ctf(projected_phase), 0.0)

    def contrast_from_projected_phase(self, projected_phase: np.ndarray) -> np.ndarray:
        return self.apply_ctf(projected_phase)

    def diagnostics(self, shape: tuple[int, int]) -> dict[str, float]:
        ctf = np.asarray(self.ctf(tuple(shape)), dtype=float)
        abs_ctf = np.abs(ctf)
        return {
            "ctf_min": float(np.nanmin(ctf)),
            "ctf_max": float(np.nanmax(ctf)),
            "ctf_rms": float(np.sqrt(np.nanmean(ctf * ctf))),
            "ctf_abs_mean": float(np.nanmean(abs_ctf)),
            "ctf_abs_max": float(np.nanmax(abs_ctf)),
        }


__all__ = ["CTFProxyTEMBackend"]
