"""interferometric imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    _complex_from_param,
    field_intensity,
    fresnel_reflection_amplitude,
    np,
    reference_vector_for_scattered,
)

class InterferometricImagingModel(ImagingModel):
    """
    Standard interferometric scattering contrast model.

    The reference field E_ref (a spatially structured or uniform complex
    amplitude) interferes with the scattered field E_sca on the detector.
    Intensity and per-particle contrast are computed as:

        Intensity   = |E_ref + E_sca_total|²
        Contrast_i  = |E_ref + E_sca_i|² − |E_ref|²

    The reference field is taken from ``background_field`` which is the
    oversampled E_ref array already constructed by the renderer (including
    any substrate pattern modulation).

    Validation:
        reference_field_amplitude must be > 0.  An error is raised here so
        that it is associated with the imaging model, not the renderer.
    """

    uses_sample_environment_pattern = True  # Reference-arm modality; patterned E_ref is physical.
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = float(params.get("reference_field_amplitude", 0.0))
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "InterferometricImagingModel (imaging_model='interferometric'). "
                "A nonzero reference field is required for interferometric contrast."
            )

    @staticmethod
    def _fresnel_reference_coefficient(params: dict) -> complex:
        """Normal-incidence Fresnel amplitude for optional iSCAT calibration profiles."""
        wavelength_nm = float(params.get("wavelength_nm", 532.0))
        top_name = params.get(
            "iscat_reference_medium_material",
            params.get("medium_material", "water"),
        )
        bottom_name = params.get(
            "iscat_reference_substrate_material",
            params.get("bulk_substrate_material", "glass"),
        )
        return fresnel_reflection_amplitude(top_name, bottom_name, wavelength_nm)

    @classmethod
    def _reference_field_scale(cls, params: dict) -> complex:
        """Return the opt-in complex scale applied to the renderer reference field."""
        model = str(params.get("iscat_reference_model", "renderer")).strip().lower()
        phase = float(params.get("iscat_reference_phase_rad", 0.0))
        amplitude_scale = _complex_from_param(
            params.get("iscat_reference_amplitude_scale", 1.0),
            default=1.0 + 0.0j,
        )
        phase_scale = np.exp(1j * phase)
        if model in {"renderer", "rendered", "none", "uniform"}:
            return amplitude_scale * phase_scale
        if model in {"fresnel", "fresnel_normal", "normal_incidence_fresnel"}:
            coeff = cls._fresnel_reference_coefficient(params)
            if bool(params.get("iscat_reference_normalize_fresnel_phase_only", False)):
                mag = abs(coeff)
                coeff = 1.0 + 0.0j if mag <= 1e-12 else coeff / mag
            return amplitude_scale * phase_scale * coeff
        if model in {"explicit", "complex"}:
            coeff = _complex_from_param(
                params.get("iscat_reference_coefficient", 1.0 + 0.0j),
                default=1.0 + 0.0j,
            )
            return amplitude_scale * phase_scale * coeff
        raise ValueError(
            "Unsupported PARAMS['iscat_reference_model'] "
            f"{params.get('iscat_reference_model')!r}. Supported values are "
            "'renderer', 'fresnel', and 'explicit'."
        )

    @staticmethod
    def _dipole_collection_fraction(params: dict) -> float:
        """Collected fraction for a transverse electric dipole over a cone."""
        NA = float(params.get("numerical_aperture", 1.0))
        n_medium = float(params.get("refractive_index_medium", 1.33))
        if n_medium <= 0.0:
            raise ValueError("PARAMS['refractive_index_medium'] must be positive.")
        sin_theta = float(np.clip(NA / n_medium, 0.0, 1.0))
        cos_theta = float(np.sqrt(max(0.0, 1.0 - sin_theta * sin_theta)))
        fraction = (4.0 - 3.0 * cos_theta - cos_theta ** 3) / 8.0
        return float(np.clip(fraction, 0.0, 1.0))

    @classmethod
    def _scattered_field_scale(cls, params: dict) -> float:
        """Return optional collected-field scaling for native iSCAT profiles."""
        model = str(params.get("iscat_collection_model", "scalar")).strip().lower()
        if model in {"scalar", "renderer", "none"}:
            return 1.0
        if model in {"dipole", "dipole_high_na", "rayleigh_dipole"}:
            fraction = cls._dipole_collection_fraction(params)
            reference_fraction = float(
                params.get("iscat_collection_reference_fraction", 1.0)
            )
            if not np.isfinite(reference_fraction) or reference_fraction <= 0.0:
                raise ValueError(
                    "PARAMS['iscat_collection_reference_fraction'] must be positive."
                )
            return float(np.sqrt(max(fraction, 1e-30) / reference_fraction))
        raise ValueError(
            "Unsupported PARAMS['iscat_collection_model'] "
            f"{params.get('iscat_collection_model')!r}. Supported values are "
            "'scalar' and 'dipole_high_na'."
        )

    @classmethod
    def _effective_fields(
        cls,
        E_sca: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        E_sca_eff = np.asarray(E_sca, dtype=np.complex128) * cls._scattered_field_scale(params)
        E_ref = reference_vector_for_scattered(
            background_field,
            E_sca_eff,
            params,
            scale=cls._reference_field_scale(params),
        )
        return E_ref, E_sca_eff

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Full-frame intensity: |E_ref + E_sca_total|².

        Args:
            E_sca_total: Complex 2D scattered-field array (oversampled FOV).
            background_field: Complex 2D reference-field array E_ref (same shape).
            params: Shared imaging-model interface dictionary; this method does
                not read additional parameters.

        Returns:
            Real 2D intensity array.
        """
        E_ref, E_sca_eff = self._effective_fields(E_sca_total, background_field, params)
        return field_intensity(E_ref + E_sca_eff)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Per-particle interferometric contrast:
            C_i = |E_ref + E_sca_i|² − |E_ref|²

        Args:
            E_sca_particle: Complex 2D scattered field for one particle.
            background_field: Complex 2D E_ref array (oversampled FOV).
            params: Shared imaging-model interface dictionary; this method does
                not read additional parameters.

        Returns:
            Real 2D contrast array (un-normalized).
        """
        E_ref, E_sca_eff = self._effective_fields(E_sca_particle, background_field, params)
        E_ref_intensity = field_intensity(E_ref)
        contrast = field_intensity(E_ref + E_sca_eff) - E_ref_intensity
        return contrast

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        ref_scale_intensity = abs(self._reference_field_scale(params)) ** 2
        ref_scale_intensity = max(float(ref_scale_intensity), 1e-30)
        E_ref_intensity_safe = np.maximum(
            E_ref_intensity_final * ref_scale_intensity,
            1e-12,
        )
        return background_final * (intensity / E_ref_intensity_safe)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        ref_scale = self._reference_field_scale(params)
        collection_scale = self._scattered_field_scale(params)
        response = super().compute_response_function(shape, params)
        response.update(
            kind="interferometric_scattering",
            iscat_reference_model=str(params.get("iscat_reference_model", "renderer")),
            iscat_reference_scale_real=float(np.real(ref_scale)),
            iscat_reference_scale_imag=float(np.imag(ref_scale)),
            iscat_reference_intensity_scale=float(abs(ref_scale) ** 2),
            iscat_collection_model=str(params.get("iscat_collection_model", "scalar")),
            iscat_scattered_field_scale=float(collection_scale),
        )
        if str(params.get("iscat_reference_model", "renderer")).strip().lower() in {
            "fresnel",
            "fresnel_normal",
            "normal_incidence_fresnel",
        }:
            response.update(
                iscat_reference_medium_material=str(
                    params.get("iscat_reference_medium_material", params.get("medium_material", "water"))
                ),
                iscat_reference_substrate_material=str(
                    params.get("iscat_reference_substrate_material", params.get("bulk_substrate_material", "glass"))
                ),
            )
        return response

__all__ = ['InterferometricImagingModel']
