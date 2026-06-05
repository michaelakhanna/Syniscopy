"""ricm imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    _ricm_particle_reflection_material,
    field_intensity,
    fresnel_reflection_amplitude,
    np,
    reference_vector_for_scattered,
)
from config.runtime import RicmSettings, param_value
from thinfilm import normal_incidence_thinfilm_reflection

class ReflectionInterferenceContrastImagingModel(ImagingModel):
    """
    Reflection Interference Contrast Microscopy (RICM).

    Common in cell-substrate adhesion imaging: light reflects from the
    glass/water interface (the "reference" reflection) and from the
    lower surface of the sample (the "particle" reflection).  The two
    reflected paths interfere on the detector, with a characteristic
    π phase shift between them due to the opposite orderings of
    dielectric indices at the two interfaces.

    The intensity is

        I(r, t) = | r_s · E_ref(r)
                   + r_p · e^{i φ_interface} · E_sca(r, t) |²

    where ``r_s`` is the substrate-reflection amplitude (typical ~0.2
    for glass/water), ``r_p`` is the particle-reflection amplitude
    (typical ~0.04 for biological material / water), and
    ``φ_interface`` is the interface phase shift (default π).

    The per-particle contrast subtracts the substrate-only baseline:

        C_i = | r_s · E_ref + r_p · e^{i φ} · E_sca_i |² − | r_s · E_ref |²

    Why this is different from iSCAT:
        * RICM's two interfering beams are both reflected paths with
          different Fresnel coefficients, not an incident reference and
          a forward-scattered secondary.
        * The π phase shift is a hallmark of the glass/water → water/
          sample reflection geometry and produces a sign flip that does
          not appear in iSCAT geometry.
        * RICM directly exploits patterned-interface reflection, so substrate
          structure is part of the modality response rather than a generic
          post-hoc background.

    Validation:
        reference_field_amplitude must be > 0.
        Substrate and particle reflection amplitudes must be positive.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True  # RICM contrast directly depends on substrate reflection structure.
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        settings = RicmSettings.from_params(params)
        E_ref_amplitude = settings.reference_field_amplitude
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "ReflectionInterferenceContrastImagingModel (imaging_model='ricm')."
            )
        self._interface_reflection_model = settings.interface_reflection_model
        self._particle_reflection_model = settings.particle_reflection_model
        if self._interface_reflection_model == "fresnel":
            self._r_s = fresnel_reflection_amplitude(
                settings.interface_medium_material,
                settings.interface_substrate_material,
                self.probe_wavelength_nm(params),
            )
        elif self._interface_reflection_model == "thin_film_stack":
            self._r_s = normal_incidence_thinfilm_reflection(
                settings.interface_medium_material,
                settings.interface_substrate_material,
                settings.thinfilm_layers,
                self.probe_wavelength_nm(params),
            )
        elif self._interface_reflection_model == "param":
            self._r_s = complex(settings.interface_reflection_coefficient)
        else:
            raise ValueError(
                "PARAMS['ricm_interface_reflection_model'] must be 'param', 'fresnel', "
                "or 'thin_film_stack'; "
                f"got {self._interface_reflection_model!r}."
            )
        if self._particle_reflection_model == "fresnel":
            self._r_p = fresnel_reflection_amplitude(
                settings.particle_medium_material,
                _ricm_particle_reflection_material(params),
                self.probe_wavelength_nm(params),
            )
        elif self._particle_reflection_model == "param":
            self._r_p = complex(settings.particle_reflection_coefficient)
        else:
            raise ValueError(
                "PARAMS['ricm_particle_reflection_model'] must be 'param' or 'fresnel'; "
                f"got {self._particle_reflection_model!r}."
            )
        self._phi = settings.interface_phase_shift_rad
        self._gap_nm = settings.gap_nm
        self._use_particle_z_as_gap = settings.use_particle_z_as_gap
        if abs(self._r_s) <= 0.0 or abs(self._r_p) <= 0.0:
            raise ValueError(
                "RICM requires positive interface and particle reflection "
                f"coefficients; got r_s={self._r_s}, r_p={self._r_p}."
            )

    def _gap_phase_rad(self, gap_nm: float, params: dict) -> float:
        """Round-trip optical-path phase for a particle/interface gap."""
        wavelength_nm = self.probe_wavelength_nm(params)
        n_medium = float(param_value(params, "refractive_index_medium"))
        gap = max(float(gap_nm), 0.0)
        return float(4.0 * np.pi * n_medium * gap / wavelength_nm)

    def _effective_gap_nm(self, params: dict, rendered_position_nm: np.ndarray | None = None) -> float:
        gap_nm = float(self._gap_nm)
        if self._use_particle_z_as_gap and rendered_position_nm is not None:
            position = np.asarray(rendered_position_nm, dtype=float).reshape(-1)
            if position.size >= 3 and np.isfinite(position[2]):
                gap_nm += float(position[2])
        return max(gap_nm, 0.0)

    def _sca_prefactor(self, params: dict, *, gap_nm: float | None = None) -> complex:
        """Complex prefactor r_p · exp(i (φ_interface + φ_gap)) applied to E_sca."""
        gap = self._gap_nm if gap_nm is None else float(gap_nm)
        phase = self._phi + self._gap_phase_rad(gap, params)
        return self._r_p * np.exp(1j * phase)

    def probe_wavelength_nm(self, params: dict) -> float:
        return float(param_value(params, "ricm_wavelength_nm"))

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        RICM intensity: |r_s · E_ref + r_p · e^{i φ} · E_sca_total|².
        """
        pref = self._sca_prefactor(params)
        E_ref = reference_vector_for_scattered(
            background_field,
            E_sca_total,
            params,
            scale=self._r_s,
        )
        return field_intensity(E_ref + pref * E_sca_total)

    def compute_scene_intensity_from_render_states(
        self,
        E_sca_particles: list[np.ndarray],
        render_states: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        RICM scene intensity with per-particle gap-dependent round-trip phase.
        """
        del particle_source_maps, frame_index
        E_ref = reference_vector_for_scattered(
            background_field,
            E_sca_total,
            params,
            scale=self._r_s,
        )
        if not E_sca_particles:
            return field_intensity(E_ref)
        E_sca_weighted = np.zeros_like(E_sca_total, dtype=np.complex128)
        for field, state in zip(E_sca_particles, render_states, strict=False):
            rendered_position = state.rendered_position_nm(np.zeros(3, dtype=float))
            gap_nm = self._effective_gap_nm(params, rendered_position)
            E_sca_weighted = E_sca_weighted + self._sca_prefactor(params, gap_nm=gap_nm) * field
        return field_intensity(E_ref + E_sca_weighted)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        RICM per-particle contrast: subtract the substrate-only baseline
        so the returned array is zero where the particle contributes nothing.
        """
        pref = self._sca_prefactor(params)
        E_ref = reference_vector_for_scattered(
            background_field,
            E_sca_particle,
            params,
            scale=self._r_s,
        )
        baseline_intensity = field_intensity(E_ref)
        with_particle = field_intensity(E_ref + pref * E_sca_particle)
        return with_particle - baseline_intensity

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        pref = self._sca_prefactor(params)
        response.update(
            kind="ricm_reflection_interference",
            forward_observable="|r_s E_ref + sum_i r_p exp(i (phi_interface + 4 pi n_medium gap_i / lambda)) E_sca_i|^2",
            interface_reflection_model=self._interface_reflection_model,
            particle_reflection_model=self._particle_reflection_model,
            r_s_real=float(np.real(self._r_s)),
            r_s_imag=float(np.imag(self._r_s)),
            r_s_abs=float(abs(self._r_s)),
            r_p_real=float(np.real(self._r_p)),
            r_p_imag=float(np.imag(self._r_p)),
            r_p_abs=float(abs(self._r_p)),
            thinfilm_layers=list(param_value(params, 'ricm_thinfilm_layers')),
            scatter_prefactor_real=float(np.real(pref)),
            scatter_prefactor_imag=float(np.imag(pref)),
            scatter_prefactor_abs=float(abs(pref)),
            interface_phase_shift_rad=float(self._phi),
            gap_nm=float(self._gap_nm),
            use_particle_z_as_gap=bool(self._use_particle_z_as_gap),
            gap_phase_model="round_trip_normal_incidence_4pi_n_gap_over_lambda",
            gap_phase_rad_at_baseline=float(self._gap_phase_rad(self._gap_nm, params)),
            count_scaling_mode="incident_background_counts_scaled_by_reflection_intensity",
        )
        return response

__all__ = ['ReflectionInterferenceContrastImagingModel']
