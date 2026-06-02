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
        E_ref_amplitude = float(params.get("reference_field_amplitude", 0.0))
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "ReflectionInterferenceContrastImagingModel (imaging_model='ricm')."
            )
        self._interface_reflection_model = str(params.get("ricm_interface_reflection_model", "param")).lower()
        self._particle_reflection_model = str(params.get("ricm_particle_reflection_model", "param")).lower()
        if self._interface_reflection_model == "fresnel":
            self._r_s = fresnel_reflection_amplitude(
                params.get("ricm_interface_medium_material", "water"),
                params.get("ricm_interface_substrate_material", "glass"),
                self.probe_wavelength_nm(params),
            )
        elif self._interface_reflection_model == "thin_film_stack":
            self._r_s = normal_incidence_thinfilm_reflection(
                params.get("ricm_interface_medium_material", "water"),
                params.get("ricm_interface_substrate_material", "glass"),
                params.get("ricm_thinfilm_layers", []),
                self.probe_wavelength_nm(params),
            )
        elif self._interface_reflection_model == "param":
            self._r_s = complex(float(params.get("ricm_interface_reflection_coefficient", 0.20)))
        else:
            raise ValueError(
                "PARAMS['ricm_interface_reflection_model'] must be 'param', 'fresnel', "
                "or 'thin_film_stack'; "
                f"got {self._interface_reflection_model!r}."
            )
        if self._particle_reflection_model == "fresnel":
            self._r_p = fresnel_reflection_amplitude(
                params.get("ricm_particle_medium_material", "water"),
                _ricm_particle_reflection_material(params),
                self.probe_wavelength_nm(params),
            )
        elif self._particle_reflection_model == "param":
            self._r_p = complex(float(params.get("ricm_particle_reflection_coefficient", 0.04)))
        else:
            raise ValueError(
                "PARAMS['ricm_particle_reflection_model'] must be 'param' or 'fresnel'; "
                f"got {self._particle_reflection_model!r}."
            )
        self._phi = float(params.get("ricm_interface_phase_shift_rad", np.pi))
        if abs(self._r_s) <= 0.0 or abs(self._r_p) <= 0.0:
            raise ValueError(
                "RICM requires positive interface and particle reflection "
                f"coefficients; got r_s={self._r_s}, r_p={self._r_p}."
            )

    def _sca_prefactor(self) -> complex:
        """Complex prefactor r_p · exp(i φ_interface) applied to E_sca."""
        return self._r_p * np.exp(1j * self._phi)

    def probe_wavelength_nm(self, params: dict) -> float:
        return float(params.get("ricm_wavelength_nm", params.get("wavelength_nm", 532.0)))

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        RICM intensity: |r_s · E_ref + r_p · e^{i φ} · E_sca_total|².
        """
        pref = self._sca_prefactor()
        E_ref = reference_vector_for_scattered(
            background_field,
            E_sca_total,
            params,
            scale=self._r_s,
        )
        return field_intensity(E_ref + pref * E_sca_total)

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
        pref = self._sca_prefactor()
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
        pref = self._sca_prefactor()
        response.update(
            kind="ricm_reflection_interference",
            forward_observable="|r_s E_ref + r_p exp(i phi_interface) E_sca|^2",
            interface_reflection_model=self._interface_reflection_model,
            particle_reflection_model=self._particle_reflection_model,
            r_s_real=float(np.real(self._r_s)),
            r_s_imag=float(np.imag(self._r_s)),
            r_s_abs=float(abs(self._r_s)),
            r_p_real=float(np.real(self._r_p)),
            r_p_imag=float(np.imag(self._r_p)),
            r_p_abs=float(abs(self._r_p)),
            thinfilm_layers=list(params.get("ricm_thinfilm_layers", [])),
            scatter_prefactor_real=float(np.real(pref)),
            scatter_prefactor_imag=float(np.imag(pref)),
            scatter_prefactor_abs=float(abs(pref)),
            interface_phase_shift_rad=float(self._phi),
            count_scaling_mode="incident_background_counts_scaled_by_reflection_intensity",
        )
        return response

__all__ = ['ReflectionInterferenceContrastImagingModel']
