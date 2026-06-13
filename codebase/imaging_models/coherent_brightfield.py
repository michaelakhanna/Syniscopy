"""coherent brightfield imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    _mean_normalized_map,
    field_intensity,
    is_vectorial_field,
    np,
    reference_vector_for_scattered,
)
from config.runtime import OpticalModeSettings, SampleEnvironmentSettings

class CoherentBrightfieldImagingModel(ImagingModel):
    """
    Coherent bright-field imaging model with a uniform incident field.

    The sample is fully illuminated by a spatially uniform beam E_inc of
    amplitude reference_field_amplitude.  Transmitted intensity is:

        I = |E_inc + E_sca_total|²

    Under the scalar plane-wave assumption this is coherent brightfield
    imaging (COBRI): the transmitted incident beam is the reference.  A real
    patterned substrate modulates the transmitted field once, in
    ``compute_scene_intensity_with_sample_environment``. The renderer must not
    also pre-pattern the coherent reference map for this modality. The
    per-particle contrast is:

        C_i = |E_inc + E_sca_i|² − |E_inc|²

    Validation:
        reference_field_amplitude must be > 0 (just as for interferometric).
    """

    uses_sample_environment_pattern = True
    sample_environment_field_domain_transmission = True
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "parameters['reference_field_amplitude'] must be positive for "
                "CoherentBrightfieldImagingModel (imaging_model='coherent_bright_field')."
            )
        self._E_inc_amplitude = E_ref_amplitude

    def _uniform_field(self, shape: tuple) -> np.ndarray:
        """Return a uniform complex reference field of shape ``shape``."""
        return np.full(shape, self._E_inc_amplitude, dtype=np.complex128)

    def _incident_field_for_scattered(
        self,
        E_sca: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None = None,
        base_field: np.ndarray | None = None,
        *,
        sample_environment_contrast_scale: float = 1.0,
    ) -> np.ndarray:
        sca = np.asarray(E_sca, dtype=np.complex128)
        spatial_shape = sca.shape[-2:] if is_vectorial_field(sca) else sca.shape
        if base_field is None:
            scalar_incident = self._uniform_field(spatial_shape)
        else:
            scalar_incident = np.asarray(base_field, dtype=np.complex128)
            if scalar_incident.shape != tuple(spatial_shape):
                raise ValueError(
                    "Coherent bright-field incident/reference field must match "
                    f"image shape; got {scalar_incident.shape}, expected {tuple(spatial_shape)}."
                )
        if sample_environment is not None:
            wavelength_nm = self.probe_wavelength_nm(params)
            t_sub = np.asarray(
                sample_environment.substrate.transmission_phase(wavelength_nm),
                dtype=np.complex128,
            )
            if t_sub.shape != tuple(spatial_shape):
                raise ValueError(
                    "Coherent bright-field sample-environment transmission must "
                    f"match image shape; got {t_sub.shape}, expected {tuple(spatial_shape)}."
                )
            alpha = float(sample_environment_contrast_scale)
            if not np.isfinite(alpha):
                raise ValueError(
                    "sample_environment_contrast_scale must be finite for "
                    "field-domain coherent bright-field sample environments."
                )
            t_sub = 1.0 + alpha * (t_sub - 1.0)
            scalar_incident = scalar_incident * t_sub
        return reference_vector_for_scattered(scalar_incident, sca, params)

    def _intensity_from_total_field(
        self,
        total_field: np.ndarray,
        incident_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del incident_field, params
        return field_intensity(total_field)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Bright-field intensity: |E_inc + E_sca_total|².

        ``background_field`` supplies the renderer's structured coherent
        reference map when available.
        """
        E_inc = self._incident_field_for_scattered(
            E_sca_total,
            params,
            base_field=background_field,
        )
        return self._intensity_from_total_field(E_inc + E_sca_total, E_inc, params)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Bright-field per-particle contrast: |E_inc + E_sca_i|² − |E_inc|².

        ``background_field`` supplies the renderer's structured coherent
        reference map when available.
        """
        E_inc = self._incident_field_for_scattered(
            E_sca_particle,
            params,
            base_field=background_field,
        )
        E_inc_intensity = field_intensity(E_inc)
        contrast = (
            self._intensity_from_total_field(E_inc + E_sca_particle, E_inc, params)
            - E_inc_intensity
        )
        return contrast

    def compute_scene_intensity_with_sample_environment(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
        sample_environment: SampleEnvironment | None = None,
        sample_environment_contrast_scale: float = 1.0,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, particle_source_maps, frame_index
        E_inc = self._incident_field_for_scattered(
            E_sca_total,
            params,
            sample_environment,
            base_field=background_field,
            sample_environment_contrast_scale=sample_environment_contrast_scale,
        )
        return self._intensity_from_total_field(E_inc + E_sca_total, E_inc, params)

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        del E_sca_total, background_field
        if sample_environment is None:
            return intensity
        wavelength_nm = self.probe_wavelength_nm(params)
        t_sub = sample_environment.substrate.transmission_phase(wavelength_nm)
        transmission = _mean_normalized_map(np.abs(t_sub) ** 2)
        sample_environment_settings = SampleEnvironmentSettings.from_params(params)
        gain = sample_environment_settings.bright_field_gain
        phase = np.unwrap(np.unwrap(np.angle(t_sub), axis=0), axis=1)
        phase_contrast = phase - float(np.mean(phase))
        phase_gain = sample_environment_settings.bright_field_phase_gain
        modulation = 1.0 + gain * (transmission - 1.0) + phase_gain * phase_contrast
        return np.maximum(intensity * modulation, 0.0)

__all__ = ['CoherentBrightfieldImagingModel']
