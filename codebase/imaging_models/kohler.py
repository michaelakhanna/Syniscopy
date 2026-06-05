"""
imaging_models/kohler.py - Real partially-coherent Köhler-illumination forward models.

These are the partially coherent bright-field and dark-field imaging models
for Syniscopy. They are physically distinct from the coherent COBRI model
(``CoherentBrightfieldImagingModel`` in imaging_models.py) and from the
coherent zero-order-blocked dark-field model (``CoherentDarkFieldImagingModel``).

Implementation: Abbe sum-of-coherent-systems decomposition (Born & Wolf
section 7.6.5; Goodman section 6.5; Hopkins 1953). The Köhler condenser is
sampled into N_s discrete plane-wave components covering either:
    - a disc of radius sigma * NA_obj (bright-field, sigma in [0, 1]), or
    - an annulus with inner radius sigma_inner * NA_obj and outer radius
      sigma_outer * NA_obj (annular dark-field, both > 1).

For each source point s = (sx, sy):
    1. The complex scattered field E_sca and substrate-modulated background
       field E_bg are projected through the shifted-pupil collection cone
       of the objective via FFT-multiply-IFFT.
    2. The coherent intensity |E_bg_eff + E_sca_eff|^2 (bright-field) or
       |field_gain * E_sca_eff + E_bg_eff|^2 (dark-field, direct on-axis
       illumination blocked but substrate-coupled background retained) is
       accumulated.

The final intensity is the average over source points (incoherent
superposition across condenser angles, as required for an extended Köhler
source). In the single on-axis source-point limit, objective-bandlimited input
fields reduce to the coherent bright-field composition rule. Arbitrary
unfiltered fields can differ because this model applies the objective pupil in
the Abbe decomposition while the coherent bright-field model consumes an
already propagated/scattered field. sigma -> 1 produces the fully partially
coherent limit of bench bright-field microscopes.
"""

from __future__ import annotations

import numpy as np

from config.runtime import (
    AnnularDarkFieldSettings,
    KohlerBrightFieldSettings,
    OpticalModeSettings,
    SamplingGeometry,
    param_value,
)
from .base import (
    ImagingModel,
    field_intensity,
    is_vectorial_field,
    reference_vector_for_scattered,
)
from substrate import SampleEnvironment


def _uses_full_vector_field(params: dict) -> bool:
    return OpticalModeSettings.from_params(params).uses_full_vector_field


# ---------------------------------------------------------------------------
# Source-point samplers
# ---------------------------------------------------------------------------

def _hex_disc_samples(n_target: int) -> np.ndarray:
    """Hex-ring source-point set filling the unit disc.

    Returns an (N, 2) array. The center is always the first sample. The
    actual sample count is approximately n_target (rounded to the nearest
    closed hex tiling).
    """
    if n_target <= 1:
        return np.zeros((1, 2), dtype=float)
    n_rings = max(1, int(round(np.sqrt(max(n_target, 1) / np.pi))))
    pts = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        r = ring / n_rings
        n_in_ring = max(6, int(round(2.0 * np.pi * ring)))
        for k in range(n_in_ring):
            theta = 2.0 * np.pi * k / n_in_ring
            pts.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(pts, dtype=float)


def _annulus_samples(n_target: int, r_inner: float, r_outer: float) -> np.ndarray:
    """Source points distributed inside an annulus of radii [r_inner, r_outer]."""
    if r_outer <= r_inner:
        raise ValueError(
            f"annular dark-field outer radius ({r_outer}) must be > inner radius ({r_inner})."
        )
    n_target = max(int(n_target), 6)
    width = r_outer - r_inner
    mid = 0.5 * (r_outer + r_inner)
    n_rings = max(1, int(round(np.sqrt(n_target * width / max(mid, 1e-6)))))
    pts: list[tuple[float, float]] = []
    for i in range(n_rings):
        r = r_inner + width * (i + 0.5) / n_rings
        n_in_ring = max(6, int(round(2.0 * np.pi * r * n_target / (n_rings * 2.0 * mid))))
        for k in range(n_in_ring):
            theta = 2.0 * np.pi * k / n_in_ring
            pts.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# Abbe-decomposition base
# ---------------------------------------------------------------------------

class _AbbeKohlerBase(ImagingModel):
    """Shared infrastructure for Abbe-decomposed partially-coherent imaging."""

    uses_sample_environment_pattern = True
    output_type = "intensity"
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        E_amp = OpticalModeSettings.from_params(params).reference_field_amplitude
        if E_amp <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                f"{type(self).__name__}."
            )
        self._E_inc_amplitude = E_amp

    # --- helpers ---

    def _physical_pixel_size_nm(self, params: dict) -> float:
        return SamplingGeometry.from_params(params).model_canvas_pixel_size_nm

    @staticmethod
    def _frequency_grids(shape: tuple[int, int], dx_m: float):
        H, W = shape
        kx = np.fft.fftfreq(W, d=dx_m)
        ky = np.fft.fftfreq(H, d=dx_m)
        return np.meshgrid(kx, ky, indexing="xy")

    def _shifted_pupil_mask(
        self,
        shape: tuple[int, int],
        sx_norm: float,
        sy_norm: float,
        cutoff_cycles_per_m: float,
        dx_m: float,
    ) -> np.ndarray:
        KX, KY = self._frequency_grids(shape, dx_m)
        return (
            (KX - sx_norm * cutoff_cycles_per_m) ** 2
            + (KY - sy_norm * cutoff_cycles_per_m) ** 2
        ) <= cutoff_cycles_per_m ** 2

    def _filter_field(
        self,
        F_field: np.ndarray,
        shape: tuple[int, int],
        sx_norm: float,
        sy_norm: float,
        cutoff_cycles_per_m: float,
        dx_m: float,
    ) -> np.ndarray:
        mask = self._shifted_pupil_mask(shape, sx_norm, sy_norm, cutoff_cycles_per_m, dx_m)
        return np.fft.ifft2(F_field * mask)

    # --- subclass interface ---

    def _source_points(self, params: dict) -> np.ndarray:
        raise NotImplementedError

    def _coherent_intensity_at_source(
        self,
        E_sca_eff: np.ndarray,
        E_bg_eff: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        return field_intensity(E_bg_eff + E_sca_eff)

    def _coherent_no_particle_intensity(
        self,
        E_bg_eff: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        return field_intensity(E_bg_eff)

    # --- public API ---

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        wavelength_m = self.probe_wavelength_nm(params) * 1e-9
        NA_obj = float(param_value(params, "numerical_aperture"))
        cutoff = NA_obj / wavelength_m
        dx_m = self._physical_pixel_size_nm(params) * 1e-9
        E_sca_total = np.asarray(E_sca_total, dtype=np.complex128)
        shape = E_sca_total.shape[-2:] if is_vectorial_field(E_sca_total) else E_sca_total.shape

        E_bg = (
            reference_vector_for_scattered(background_field, E_sca_total, params)
            if background_field is not None
            else None
        )
        F_sca = np.fft.fft2(E_sca_total, axes=(-2, -1))
        F_bg = np.fft.fft2(E_bg, axes=(-2, -1)) if E_bg is not None else None

        pts = self._source_points(params)
        if pts.shape[0] == 0:
            pts = np.zeros((1, 2), dtype=float)

        I_total = np.zeros(shape, dtype=float)
        for sx, sy in pts:
            E_sca_eff = self._filter_field(F_sca, shape, sx, sy, cutoff, dx_m)
            if F_bg is not None:
                E_bg_eff = self._filter_field(F_bg, shape, sx, sy, cutoff, dx_m)
            else:
                E_bg_eff = reference_vector_for_scattered(
                    np.full(shape, self._E_inc_amplitude, dtype=np.complex128),
                    E_sca_total,
                    params,
                )
            I_total += self._coherent_intensity_at_source(E_sca_eff, E_bg_eff, params)
        return I_total / float(pts.shape[0])

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        wavelength_m = self.probe_wavelength_nm(params) * 1e-9
        NA_obj = float(param_value(params, "numerical_aperture"))
        cutoff = NA_obj / wavelength_m
        dx_m = self._physical_pixel_size_nm(params) * 1e-9
        E_sca_particle = np.asarray(E_sca_particle, dtype=np.complex128)
        shape = E_sca_particle.shape[-2:] if is_vectorial_field(E_sca_particle) else E_sca_particle.shape

        E_bg = (
            reference_vector_for_scattered(background_field, E_sca_particle, params)
            if background_field is not None
            else None
        )
        F_sca = np.fft.fft2(E_sca_particle, axes=(-2, -1))
        F_bg = np.fft.fft2(E_bg, axes=(-2, -1)) if E_bg is not None else None

        pts = self._source_points(params)
        if pts.shape[0] == 0:
            pts = np.zeros((1, 2), dtype=float)

        I_with = np.zeros(shape, dtype=float)
        I_without = np.zeros(shape, dtype=float)
        for sx, sy in pts:
            E_sca_eff = self._filter_field(F_sca, shape, sx, sy, cutoff, dx_m)
            if F_bg is not None:
                E_bg_eff = self._filter_field(F_bg, shape, sx, sy, cutoff, dx_m)
            else:
                E_bg_eff = reference_vector_for_scattered(
                    np.full(shape, self._E_inc_amplitude, dtype=np.complex128),
                    E_sca_particle,
                    params,
                )
            I_with += self._coherent_intensity_at_source(E_sca_eff, E_bg_eff, params)
            I_without += self._coherent_no_particle_intensity(E_bg_eff, params)
        return (I_with - I_without) / float(pts.shape[0])

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        pts = self._source_points(params)
        response.update(
            scalar_vectorial_backend=OpticalModeSettings.from_params(params).optical_field_backend,
            objective_NA=float(param_value(params, "numerical_aperture")),
            wavelength_nm=float(self.probe_wavelength_nm(params)),
            source_point_count_actual=int(pts.shape[0]),
            source_sampling_scheme="deterministic_hex_disc_or_annulus",
        )
        return response


# ---------------------------------------------------------------------------
# Partially-coherent (Köhler) bright-field
# ---------------------------------------------------------------------------

class PartiallyCoherentBrightfieldImagingModel(_AbbeKohlerBase):
    """
    Real partially-coherent Köhler bright-field for the ``bright_field``
    modality.

    Distinct from coherent brightfield imaging (COBRI), registered as
    ``coherent_bright_field``. Under Köhler
    illumination the condenser is an extended source filling a disc of
    radius sigma * NA_obj (sigma in [0, 1], typically 0.5-0.9 in real
    systems). Each source point produces a tilted plane-wave illumination;
    the detector integrates intensity incoherently across source points.

    Differences from COBRI:
        - frequency-dependent partial-coherence transfer that materially
          differs from COBRI for small particles near the resolution limit;
        - non-zero intensity contrast for phase-only objects (e.g. thin
          substrate transmission patterns) that vanishes in the COBRI
          limit;
        - depth-of-field broadening and contrast roll-off characteristic
          of bench bright-field microscopes.

    Parameters (with defaults):
        ``kohler_coherence_factor`` (sigma): 0.7
        ``kohler_source_samples`` (target N_s): 19
    """

    def _source_points(self, params: dict) -> np.ndarray:
        settings = KohlerBrightFieldSettings.from_params(params)
        sigma = settings.coherence_factor
        sigma = max(0.0, min(sigma, 1.0))
        n_target = settings.source_samples
        return sigma * _hex_disc_samples(n_target)

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        # Substrate modulation enters via background_field and is propagated
        # through the partial-coherence transfer by the Abbe integration in
        # compute_intensity / compute_per_particle_contrast. Adding another
        # transmission factor here would double-count it.
        del E_sca_total, background_field, sample_environment, params
        return intensity

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="abbe_kohler_bright_field",
            condenser_sigma=KohlerBrightFieldSettings.from_params(params).coherence_factor,
            kohler_source_points=KohlerBrightFieldSettings.from_params(params).source_samples,
            field_representation=(
                "vectorial_full_field"
                if _uses_full_vector_field(params)
                else "scalar_or_analyzer_component_field"
            ),
            fidelity_label=(
                "vectorial_abbe_kohler_bright_field"
                if _uses_full_vector_field(params)
                else "scalar_abbe_kohler_bright_field"
            ),
        )
        return response


# ---------------------------------------------------------------------------
# Annular Köhler dark-field
# ---------------------------------------------------------------------------

class AnnularDarkFieldImagingModel(_AbbeKohlerBase):
    """
    Real annular Köhler dark-field for the ``dark_field`` modality.

    The condenser illuminates the sample from incident angles strictly
    outside the objective collection cone. For an objective of NA_obj,
    illumination originates in an annulus with inner radius
    sigma_inner * NA_obj and outer radius sigma_outer * NA_obj, with
    sigma_inner > 1 by construction. Light reaching the objective is only
    that scattered from the sample (particles AND substrate edges).

    Distinct from the coherent zero-order-blocked dark-field model registered
    as ``coherent_dark_field``.

    Parameters (with defaults):
        ``annular_dark_field_inner_sigma``: 1.02
        ``annular_dark_field_outer_sigma``: 1.08
        ``annular_dark_field_source_samples``: 24
    """

    def _source_points(self, params: dict) -> np.ndarray:
        settings = AnnularDarkFieldSettings.from_params(params)
        return _annulus_samples(settings.source_samples, settings.inner_sigma, settings.outer_sigma)

    def _coherent_intensity_at_source(self, E_sca_eff, E_bg_eff, params):
        # Direct illumination is OUTSIDE the objective NA, so the unscattered
        # zero-order is rejected. Only the scattered + substrate-edge field
        # that couples through the objective pupil is detected.
        field_gain = AnnularDarkFieldSettings.from_params(params).dark_field.field_gain
        if field_gain <= 0.0:
            raise ValueError("PARAMS['dark_field_field_gain'] must be positive.")
        return field_intensity(field_gain * E_sca_eff + E_bg_eff)

    def _coherent_no_particle_intensity(self, E_bg_eff, params):
        return field_intensity(E_bg_eff)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        settings = AnnularDarkFieldSettings.from_params(params).dark_field
        illumination_count = settings.illumination_count
        background_count = settings.background_count
        return illumination_count * intensity + background_count

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="abbe_annular_kohler_dark_field",
            annulus_inner_sigma=AnnularDarkFieldSettings.from_params(params).inner_sigma,
            annulus_outer_sigma=AnnularDarkFieldSettings.from_params(params).outer_sigma,
            kohler_source_points=AnnularDarkFieldSettings.from_params(params).source_samples,
            direct_beam_blocked=True,
            substrate_background_handling="substrate_edge_field_retained",
            field_representation=(
                "vectorial_full_field"
                if _uses_full_vector_field(params)
                else "scalar_or_analyzer_component_field"
            ),
            fidelity_label=(
                "vectorial_abbe_annular_kohler_dark_field"
                if _uses_full_vector_field(params)
                else "scalar_abbe_annular_kohler_dark_field"
            ),
        )
        return response

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        # Topography enters through the substrate-modulated background field
        # and is filtered correctly by the off-axis pupil integration.
        del E_sca_total, background_field, sample_environment, params
        return intensity


__all__ = [
    "PartiallyCoherentBrightfieldImagingModel",
    "AnnularDarkFieldImagingModel",
]
