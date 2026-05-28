"""
imaging_model.py — Pluggable imaging contrast models for Syniscopy.

Each imaging model converts the complex scattered field and scene background
into a detector-domain image plus per-particle contrast images for supervision.
The registry maps public modality names to concrete model classes:
bright_field and dark_field use Köhler implementations in kohler_imaging.py,
while coherent_* names expose scalar coherent-limit models. Registry helpers
normalize accepted alias spellings to a consistent modality vocabulary for
configs and manifests.

Supported models cover bright-field, fluorescence, dark-field, phase/interference
contrast, interferometric scattering, and simplified electron-style contrast.
"""

from __future__ import annotations

import numpy as np

from substrate import MaterialProperties, SampleEnvironment, fresnel_reflection_amplitude
from imaging_base import ImagingModel


LABEL_FREE_OPTICAL_MODALITIES = (
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "dark_field",
    "coherent_dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
)

CANONICAL_COHERENT_MODALITIES = (
    "coherent_bright_field",
    "coherent_dark_field",
)

RELATIVE_REFERENCE_CONTRAST_MODALITIES = (
    "interferometric",
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
)


def _ricm_particle_reflection_material(params: dict) -> str | MaterialProperties:
    explicit = params.get("ricm_particle_material", None)
    if isinstance(explicit, MaterialProperties):
        return explicit
    explicit_text = "" if explicit is None else str(explicit).strip()
    if explicit_text.lower() not in ("", "none", "particle_material", "primary_particle"):
        return explicit_text

    from particle_specs import get_particle_specs
    from materials import resolve_component_material_properties

    specs = get_particle_specs(params)
    primary = specs[0].primary_component
    if (
        primary.material not in (None, "")
        or primary.refractive_index is not None
        or primary.material_properties is not None
    ):
        return resolve_component_material_properties(params, primary)
    legacy_material = params.get("particle_material", None)
    if legacy_material not in (None, ""):
        return str(legacy_material)
    return "polystyrene"

SUPPORTED_MODALITIES = (
    "bright_field",
    "fluorescence_widefield",
    "tirf_fluorescence",
    "dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
    "tem_phase_contrast",
    "sem_secondary_electron",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "coherent_dark_field",
)

MODALITY_ALIASES = {
    "partially_coherent_brightfield": "partially_coherent_bright_field",
    "coherent_brightfield": "coherent_bright_field",
}


def canonical_modality_name(model_name: str) -> str:
    """Return the canonical public spelling for an imaging-model name."""
    key = str(model_name).strip().lower()
    return MODALITY_ALIASES.get(key, key)


def _mean_normalized_map(arr: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Return ``arr`` divided by its positive finite mean."""
    out = np.asarray(arr, dtype=float)
    finite = np.isfinite(out)
    mean = float(out[finite].mean()) if np.any(finite) else 0.0
    if abs(mean) <= floor:
        return np.ones_like(out, dtype=float)
    return out / mean


def _complex_from_param(value, *, default: complex = 1.0 + 0.0j) -> complex:
    """Coerce a config value into a complex scalar."""
    if value is None:
        return complex(default)
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, np.number)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        text = value.strip().replace("i", "j")
        return complex(text)
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise TypeError(f"Cannot interpret {value!r} as a complex scalar.")


# ---------------------------------------------------------------------------
# Interferometric imaging model
# ---------------------------------------------------------------------------

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
        E_ref = np.asarray(background_field, dtype=np.complex128) * cls._reference_field_scale(params)
        E_sca_eff = np.asarray(E_sca, dtype=np.complex128) * cls._scattered_field_scale(params)
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
        return np.abs(E_ref + E_sca_eff) ** 2

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
        E_ref_intensity = np.abs(E_ref) ** 2
        contrast = np.abs(E_ref + E_sca_eff) ** 2 - E_ref_intensity
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


# ---------------------------------------------------------------------------
# Dark-field imaging model
# ---------------------------------------------------------------------------

class CoherentDarkFieldImagingModel(ImagingModel):
    """
    Dark-field (reference-free) imaging model.

    No reference beam reaches the detector.  The signal is purely the
    scattered intensity:

        Intensity  = |E_sca_total|²
        Contrast_i = |E_sca_i|²

    Note: ``PARAMS['reference_field_amplitude']`` is ignored in this mode: the
    reference field has no role in the dark-field forward model.
    """

    uses_sample_environment_pattern = True  # Patterned substrates scatter into the dark-field stop.

    def __init__(self, params: dict) -> None:
        # Dark-field accepts the shared constructor signature used by the
        # imaging-model factory.
        self._field_gain = float(params.get("dark_field_field_gain", 1.0))
        if self._field_gain <= 0.0:
            raise ValueError("PARAMS['dark_field_field_gain'] must be positive.")

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field intensity: |E_sca_total|².

        ``background_field`` is accepted as part of the shared imaging-model
        interface and is not used by dark-field intensity.
        """
        return np.abs(self._field_gain * E_sca_total) ** 2

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field per-particle contrast: |E_sca_i|².

        ``background_field`` is not used.
        """
        return np.abs(self._field_gain * E_sca_particle) ** 2

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Dark-field intensity-to-counts conversion.

        Physical rationale
        ------------------
        In dark-field there is no reference beam to divide by. The base-class
        count conversion is ill-conditioned when ``|E_ref|^2`` is small, which
        is the dark-field regime. This model converts the dimensionless
        |E_sca|^2 to photon counts by multiplying by an illumination-level
        scale and adding a detector / stray-light pedestal.

        Why a pedestal is required
        --------------------------
        Far from any particle ``intensity = |E_sca|^2`` is exactly zero.
        Without a pedestal the downstream noise pipeline does
        ``Poisson(0) = 0`` plus ``Gaussian(0, sigma_read)``, and the
        ``np.clip(., 0, max).astype(uint16)`` step at the end of the
        rendering loop half-clips the Gaussian to zero, producing a
        visually noise-free black background that does not match real
        dark-field detectors.  A small pedestal corresponds to the physical
        reality of dark-field imaging --- residual stray light reaching the
        detector, plus camera dark current --- and gives read noise a
        non-zero baseline to fluctuate around.

        Parameter resolution
        --------------------
        The illumination-level scale is taken from
        ``PARAMS['dark_field_illumination_count']`` if set, and otherwise
        falls back to ``PARAMS['background_intensity']`` (which is the
        count-domain reference-beam brightness used by the other modalities,
        so in the default configuration the dark-field peak will land at a
        comparable fraction of the camera's dynamic range to the other
        modalities' reference-beam intensity).

        The pedestal is taken from ``PARAMS['dark_field_background_count']``.
        The default is zero, which preserves the ideal zero-baseline dark-field
        model; callers can set a positive pedestal to represent stray light or
        dark current.

        Result
        ------
        - Away from the particle: output is approximately the pedestal,
          so read + shot noise is visible at the expected level.
        - Near the particle: the peak |E_sca|^2 times the illumination
          count adds on top of the pedestal, producing a proportional
          bright spot.
        - Shot noise is applied downstream by the camera-noise model using the
          returned count values as Poisson rates.
        """
        illumination_count = float(params.get(
            "dark_field_illumination_count",
            float(params.get("background_intensity", 1.0)),
        ))
        background_count = float(params.get(
            "dark_field_background_count",
            0.0,
        ))
        return illumination_count * intensity + background_count

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        """Coherent dark-field uses the shared scalar incident-field interface."""
        return super().illumination_field(shape, params)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="coherent_dark_field_scattered_intensity",
            zero_order_reference_blocked=True,
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
        del E_sca_total, background_field
        if sample_environment is None:
            return intensity
        edge = sample_environment.substrate.topography_gradient()
        edge = _mean_normalized_map(edge + 1e-12) - 1.0
        gain = float(params.get("dark_field_sample_environment_edge_gain", 0.02))
        pedestal = float(params.get("dark_field_sample_environment_scatter_pedestal", 0.0))
        return np.maximum(intensity + gain * np.maximum(edge, 0.0) + pedestal, 0.0)


# ---------------------------------------------------------------------------
# Coherent bright-field / COBRI imaging model
# ---------------------------------------------------------------------------

class CoherentBrightfieldImagingModel(ImagingModel):
    """
    Coherent bright-field imaging model with a uniform incident field.

    The sample is fully illuminated by a spatially uniform beam E_inc of
    amplitude reference_field_amplitude.  Transmitted intensity is:

        I = |E_inc + E_sca_total|²

    Under the scalar plane-wave assumption this is coherent brightfield
    imaging (COBRI): the transmitted incident beam is the reference.  A real
    patterned substrate modulates the transmitted field; that
    modulation is applied by ``apply_sample_environment`` so bright-field
    includes substrates/patterns. The per-particle contrast is:

        C_i = |E_inc + E_sca_i|² − |E_inc|²

    Validation:
        reference_field_amplitude must be > 0 (just as for interferometric).
    """

    uses_sample_environment_pattern = True

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = float(params.get("reference_field_amplitude", 0.0))
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "CoherentBrightfieldImagingModel (imaging_model='coherent_bright_field')."
            )
        self._E_inc_amplitude = E_ref_amplitude

    def _uniform_field(self, shape: tuple) -> np.ndarray:
        """Return a uniform complex reference field of shape ``shape``."""
        return np.full(shape, self._E_inc_amplitude, dtype=np.complex128)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Bright-field intensity: |E_inc + E_sca_total|².

        ``background_field`` is not used; E_inc is spatially uniform.
        """
        E_inc = self._uniform_field(E_sca_total.shape)
        return np.abs(E_inc + E_sca_total) ** 2

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Bright-field per-particle contrast: |E_inc + E_sca_i|² − |E_inc|².

        ``background_field`` is not used.
        """
        E_inc = self._uniform_field(E_sca_particle.shape)
        E_inc_intensity = self._E_inc_amplitude ** 2
        contrast = np.abs(E_inc + E_sca_particle) ** 2 - E_inc_intensity
        return contrast

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
        gain = float(params.get("bright_field_sample_environment_gain", 1.0))
        phase = np.unwrap(np.unwrap(np.angle(t_sub), axis=0), axis=1)
        phase_contrast = phase - float(np.mean(phase))
        phase_gain = float(params.get("bright_field_sample_environment_phase_gain", 0.05))
        modulation = 1.0 + gain * (transmission - 1.0) + phase_gain * phase_contrast
        return np.maximum(intensity * modulation, 0.0)


# Import after the coherent scalar classes to avoid circular imports while
# building the registry.
from kohler_imaging import (
    PartiallyCoherentBrightfieldImagingModel as _PartiallyCoherentBrightfieldImagingModel,
    AnnularDarkFieldImagingModel as _AnnularDarkFieldImagingModel,
)


# ---------------------------------------------------------------------------
# Zernike phase contrast
# ---------------------------------------------------------------------------

class ZernikePhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Scalar Zernike phase-contrast approximation using the recovered phase."""

    uses_sample_environment_pattern = True

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._uniform_field(E_sca_total.shape)
        phase = np.angle(E_inc + E_sca_total) - np.angle(E_inc)
        gain = float(params.get("zernike_phase_ring_gain", 0.35))
        phase_shift = float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0))
        bias = float(params.get("zernike_phase_bias", 1.0))
        phase_gain = gain * np.sin(phase_shift)
        return np.maximum((self._E_inc_amplitude ** 2) * (bias + phase_gain * phase), 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._uniform_field(E_sca_particle.shape)
        phase = np.angle(E_inc + E_sca_particle) - np.angle(E_inc)
        gain = float(params.get("zernike_phase_ring_gain", 0.35))
        phase_shift = float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0))
        phase_gain = gain * np.sin(phase_shift)
        return (self._E_inc_amplitude ** 2) * phase_gain * phase

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="zernike_phase_ring",
            phase_shift_rad=float(params.get("zernike_phase_ring_shift_rad", np.pi / 2.0)),
        )
        return response


# ---------------------------------------------------------------------------
# Differential phase contrast
# ---------------------------------------------------------------------------

class DifferentialPhaseContrastImagingModel(CoherentBrightfieldImagingModel):
    """Scalar differential phase-contrast approximation from phase gradients."""

    uses_sample_environment_pattern = True

    @staticmethod
    def _dpc_signal(field: np.ndarray, pixel_size_nm: float) -> np.ndarray:
        phase = np.unwrap(np.unwrap(np.angle(field), axis=0), axis=1)
        dphi_dy, dphi_dx = np.gradient(phase, pixel_size_nm)
        return dphi_dx + dphi_dy

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._uniform_field(E_sca_total.shape)
        dpc = self._dpc_signal(
            E_inc + E_sca_total,
            float(params["pixel_size_nm"]) / max(float(params.get("psf_oversampling_factor", 1)), 1.0),
        )
        gain = float(params.get("dpc_phase_gradient_gain", 2500.0))
        return np.maximum((self._E_inc_amplitude ** 2) * (1.0 + gain * dpc), 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        E_inc = self._uniform_field(E_sca_particle.shape)
        dpc = self._dpc_signal(
            E_inc + E_sca_particle,
            float(params["pixel_size_nm"]) / max(float(params.get("psf_oversampling_factor", 1)), 1.0),
        )
        gain = float(params.get("dpc_phase_gradient_gain", 2500.0))
        return (self._E_inc_amplitude ** 2) * gain * dpc

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(kind="asymmetric_illumination_dpc")
        return response


# ---------------------------------------------------------------------------
# Quantitative phase imaging (QPI)
# ---------------------------------------------------------------------------

class QuantitativePhaseImagingModel(ImagingModel):
    """
    Quantitative phase imaging (QPI) contrast model.

    Rather than photon counts, the detector-domain quantity here is the
    optical phase of the transmitted (or reflected) field, expressed in
    radians.  This corresponds physically to a phase-shifting interferometer
    or an off-axis holographic setup that has already been demodulated.

    The computed phase is the argument of the complex field normalised by
    the reference (so the idle frame is zero-phase everywhere):

        φ(r, t) = arg( E_ref(r) + E_sca(r, t) ) − arg( E_ref(r) )
                = arg( 1 + E_sca(r, t) / E_ref(r) )

    Per-particle contrast is the same quantity computed using only that
    particle's scattered field.  Both outputs are real 2D float arrays in
    radians, and may be negative.  The returned arrays are wrapped to
    (−π, π] by ``np.angle``; 2D unwrap is the caller's responsibility and
    is *not* needed for visualisation.

    Small-signal approximation:
        For |E_sca| ≪ |E_ref|, φ ≈ Im(E_sca / E_ref), which is the quantity
        typically reported in nanoscale biology / cell thickness mapping.

    Validation:
        reference_field_amplitude must be > 0 (phase is only defined
        relative to a nonzero reference).
    """

    output_type = "phase"
    uses_sample_environment_pattern = True  # Phase is referenced to the structured coherent background field.

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = float(params.get("reference_field_amplitude", 0.0))
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "QuantitativePhaseImagingModel (imaging_model='quantitative_phase'). "
                "Phase is only defined relative to a nonzero reference field."
            )

    @staticmethod
    def _phase(E_sum: np.ndarray, E_ref: np.ndarray) -> np.ndarray:
        """Compute arg(E_sum) − arg(E_ref), wrapped to (−π, π]."""
        # Guard against zero-valued reference entries. This can occur in a
        # substrate "hole" region where the reflected reference has been
        # strongly attenuated. In that case, the local phase reference is
        # undefined, so we clamp to zero rather than emitting a huge value
        # from an arbitrary-phase 0/0 division.
        ref_power = np.abs(E_ref) ** 2
        safe = ref_power > 1e-24
        phi = np.zeros(E_sum.shape, dtype=float)
        # (E_sum · conj(E_ref)) has the same phase as E_sum/E_ref but is
        # numerically stable where |E_ref| is small.
        product = E_sum * np.conj(E_ref)
        phi[safe] = np.angle(product[safe])
        return phi

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Full-frame phase: arg(E_ref + E_sca_total) − arg(E_ref), radians.

        This method implements the shared imaging-model entry point.
        ``output_type = "phase"`` declares that the return value is a phase map.
        """
        return self._phase(background_field + E_sca_total, background_field)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Per-particle phase shift: arg(E_ref + E_sca_i) − arg(E_ref).
        """
        return self._phase(background_field + E_sca_particle, background_field)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Convert a demodulated phase map into a count-like video image.

        QPI's physical output is phase in radians, not optical intensity, so
        the base interferometric |E|^2-to-count scaling is invalid here. For
        rendered videos we store phase contrast around the detector background
        with a configurable radians-to-counts display scale. The CRLB and mask
        paths continue to use the actual phase contrast image.
        """
        del E_ref_intensity_final
        phase_to_count = float(
            params.get(
                "qpi_phase_to_count_scale",
                params.get("background_intensity", 100.0),
            )
        )
        return np.asarray(background_final, dtype=float) + phase_to_count * np.asarray(intensity, dtype=float)


# ---------------------------------------------------------------------------
# Reflection Interference Contrast Microscopy (RICM)
# ---------------------------------------------------------------------------

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
        elif self._interface_reflection_model == "param":
            self._r_s = complex(float(params.get("ricm_interface_reflection_coefficient", 0.20)))
        else:
            raise ValueError(
                "PARAMS['ricm_interface_reflection_model'] must be 'param' or 'fresnel'; "
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
        return np.abs(self._r_s * background_field + pref * E_sca_total) ** 2

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
        baseline_intensity = np.abs(self._r_s * background_field) ** 2
        with_particle = np.abs(
            self._r_s * background_field + pref * E_sca_particle
        ) ** 2
        return with_particle - baseline_intensity


# ---------------------------------------------------------------------------
# Off-axis digital holographic microscopy (DHM)
# ---------------------------------------------------------------------------

class OffAxisHolographyImagingModel(ImagingModel):
    """
    Off-axis digital holographic microscopy (DHM).

    A tilted reference beam is introduced at angle θ relative to the sample
    beam, producing a detector fringe pattern:

        I(r, t) = | E_ref(r) · e^{i K · r} + E_sca(r, t) |²
                = |E_ref|² + |E_sca|²
                  + 2 · Re( E_ref*(r) · e^{-i K·r} · E_sca(r, t) )

    where K = (2π / T) · (cos θ, sin θ) is the tilt wavevector, T is the
    fringe period in pixels, and θ is the fringe orientation in radians.

    Off-axis holography's distinguishing feature: a single raw frame
    encodes the full complex field of the sample, recoverable by Fourier-
    domain bandpass-demodulation around K.  This is what makes it appealing
    for dynamic / single-shot QPI of live samples.  We render the raw
    fringe frame; demodulation is a post-processing step outside the
    simulator's scope.

    Parameters (all taken from PARAMS with nominal defaults):
        off_axis_fringe_period_px   (default 10.0)
        off_axis_fringe_angle_rad   (default 0.0, fringes run along y)

    Validation:
        reference_field_amplitude must be > 0.
        off_axis_fringe_period_px must be >= 2 (so K is within Nyquist).
    """

    output_type = "fringe"
    uses_sample_environment_pattern = True  # Off-axis DHM has a coherent reference arm that can carry substrate structure.

    def __init__(self, params: dict) -> None:
        E_ref_amplitude = float(params.get("reference_field_amplitude", 0.0))
        if E_ref_amplitude <= 0.0:
            raise ValueError(
                "PARAMS['reference_field_amplitude'] must be positive for "
                "OffAxisHolographyImagingModel (imaging_model='off_axis_holography')."
            )
        self._period_px = float(params.get("off_axis_fringe_period_px", 10.0))
        if not np.isfinite(self._period_px) or self._period_px < 2.0:
            raise ValueError(
                "off_axis_fringe_period_px must be >= 2.0 (Nyquist); got "
                f"{self._period_px}."
            )
        self._angle_rad = float(params.get("off_axis_fringe_angle_rad", 0.0))
        # Lazily-initialised fringe phase cache keyed by shape, since the
        # array is identical across frames.
        self._tilt_phase_cache: dict = {}

    def _tilt_field(self, shape: tuple) -> np.ndarray:
        """Return a unit-amplitude carrier e^{i K·r} of shape ``shape``."""
        cached = self._tilt_phase_cache.get(shape)
        if cached is not None:
            return cached
        H, W = shape[-2], shape[-1]
        # Pixel grid: row 0 at y=0, col 0 at x=0.
        yy, xx = np.meshgrid(
            np.arange(H, dtype=float),
            np.arange(W, dtype=float),
            indexing="ij",
        )
        K = 2.0 * np.pi / self._period_px
        phase = K * (xx * np.cos(self._angle_rad) + yy * np.sin(self._angle_rad))
        carrier = np.exp(1j * phase)
        if shape != (H, W):
            carrier = carrier.reshape(shape)
        self._tilt_phase_cache[shape] = carrier
        return carrier

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Off-axis fringe frame: |E_ref · e^{iK·r} + E_sca_total|².
        """
        carrier = self._tilt_field(E_sca_total.shape)
        return np.abs(background_field * carrier + E_sca_total) ** 2

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Per-particle fringe contrast: |E_ref · e^{iK·r} + E_sca_i|² − |E_ref|².
        """
        carrier = self._tilt_field(E_sca_particle.shape)
        ref_intensity = np.abs(background_field) ** 2
        return np.abs(background_field * carrier + E_sca_particle) ** 2 - ref_intensity


# ---------------------------------------------------------------------------
# Electron microscopy: transmission electron phase-contrast (TEM)
# ---------------------------------------------------------------------------

# Physical constants in SI, used only by electron-optics formulas.
_PLANCK_H_J_S            = 6.62607015e-34
_REDUCED_PLANCK_HBAR_J_S = _PLANCK_H_J_S / (2.0 * np.pi)
_ELECTRON_MASS_KG        = 9.1093837015e-31
_ELEMENTARY_CHARGE_C     = 1.602176634e-19
_SPEED_OF_LIGHT_M_S      = 2.99792458e8


def electron_wavelength_m(acceleration_kV: float) -> float:
    """
    Relativistic de Broglie wavelength of an electron accelerated through
    ``acceleration_kV`` kilovolts. Returns wavelength in metres.

    The non-relativistic form would be lambda = h / sqrt(2 m_e e V), but
    at 100-300 kV the relativistic correction factor (1 + eV/(2 m c^2))
    under the square root reduces the apparent wavelength by ~5-20%.
    We use the relativistic expression to remain accurate across the
    full range of modern TEM operating voltages (80-300 kV).
    """
    V = float(acceleration_kV) * 1.0e3  # volts
    if V <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    eV = _ELEMENTARY_CHARGE_C * V
    m = _ELECTRON_MASS_KG
    c = _SPEED_OF_LIGHT_M_S
    h = _PLANCK_H_J_S
    # Relativistic momentum from E_kin = eV: p = sqrt( (eV)^2 / c^2 + 2 m eV ).
    p = np.sqrt((eV / c) ** 2 + 2.0 * m * eV)
    return h / p


def electron_interaction_parameter_rad_per_V_nm(acceleration_kV: float) -> float:
    """
    Relativistic weak-phase interaction parameter in rad/(V nm).

    The TEM source map uses ``phi = sigma * V_mip * thickness`` where
    ``V_mip`` is the material mean inner potential in volts and thickness is
    in nanometres.
    """
    V = float(acceleration_kV) * 1.0e3
    if V <= 0.0:
        raise ValueError(
            f"electron accelerating voltage must be positive; got {acceleration_kV} kV."
        )
    kinetic_J = _ELEMENTARY_CHARGE_C * V
    m = _ELECTRON_MASS_KG
    c = _SPEED_OF_LIGHT_M_S
    p = np.sqrt((kinetic_J / c) ** 2 + 2.0 * m * kinetic_J)
    sigma_per_V_m = (
        _ELEMENTARY_CHARGE_C
        * (kinetic_J + m * c ** 2)
        / (_REDUCED_PLANCK_HBAR_J_S * c ** 2 * p)
    )
    return float(sigma_per_V_m * 1.0e-9)


def scherzer_defocus_m(acceleration_kV: float, Cs_mm: float) -> float:
    """
    Scherzer defocus in metres given the electron wavelength (from the
    accelerating voltage) and the objective spherical aberration Cs (in mm).
    Convention: positive output means underfocus (Scherzer condition).
    The Scherzer defocus is the value that produces the broadest pass-band
    in the phase-contrast transfer function:

        Delta f_Sch = sqrt( 1.5 · lambda · C_s ).
    """
    Cs_m = 1.0e-3 * float(Cs_mm)
    if Cs_m < 0.0:
        raise ValueError(f"C_s must be non-negative; got {Cs_mm} mm.")
    wavelength_m = electron_wavelength_m(acceleration_kV)
    return np.sqrt(1.5 * wavelength_m * Cs_m)


class TransmissionElectronMicroscopyImagingModel(ImagingModel):
    """
    Transmission electron microscopy (TEM) phase-contrast imaging model.

    This model converts a projected material-potential source map into a TEM
    phase-contrast image by applying the standard electron contrast transfer
    function (CTF) in Fourier space.
    The underlying physics is the weak-phase-object approximation used
    throughout electron microscopy: a thin specimen imparts a small phase
    shift proportional to its projected electrostatic potential, and the
    detector records a filtered image of that phase shift.

    Under this approximation the complex exit wave is
        psi_exit(r) ~= 1 + i sigma V_proj(r),
    and after the objective lens and detector sampling the recorded
    intensity becomes
        I(r) ~= 1 - 2 sigma V_proj * PSF_TEM(r),
    with the TEM point-spread function given in Fourier space by
        CTF(k)    = -2 sin(chi(k)) * E(k),
        chi(k)    = pi C_s lambda^3 k^4 / 2  -  pi lambda Delta_f k^2,
        E(k)      = exp(-pi^2 alpha^2 (C_s lambda^2 k^3 - Delta_f k)^2),
    where lambda is the relativistic de Broglie wavelength of the electron,
    C_s is the spherical aberration coefficient, Delta_f is the defocus,
    and alpha is the illumination-angle half-width controlling partial
    coherence.  These formulas follow Kirkland (2010), Chap. 5.

    Syniscopy source representation
    -------------------------------
    During rendering, particle material properties are accumulated into a
    projected phase-shift source map ``sigma * V_proj(r)``. The imaging model
    applies the CTF in Fourier space and returns the linearized weak-phase
    intensity ``1 + CTF * source``, clamped at zero for count-domain noise
    sampling. This is a simplified weak-phase TEM path, not a multislice
    electron-scattering simulator.

    Parameters (PARAMS keys, all optional with nominal defaults)
    ------------------------------------------------------------
    The defaults define a stable moderate-contrast synthetic TEM regime; use
    calibrated values for instrument-specific studies.

    - ``tem_acceleration_kV``         (default 300.0) accelerating voltage
    - ``tem_Cs_mm``                   (default 0.5)   spherical aberration
    - ``tem_defocus_nm``              (default: Scherzer) defocus Delta_f
    - ``tem_partial_coherence_alpha_mrad`` (default 0.1) illumination half-angle
    - ``tem_phase_shift_per_volt_nm`` (default: relativistic electron
      interaction parameter for ``tem_acceleration_kV``) projected phase scale
      multiplying material mean inner potential and projected thickness.
    - ``tem_pixel_size_pm``           optional compatibility assertion for
      the CTF Fourier-grid pitch. When supplied, it must match the actual
      rendered model-canvas pitch ``pixel_size_nm / psf_oversampling_factor``.
    - ``tem_dose_per_pixel``          (default 100)    mean electron
      count per pixel for the unscattered beam.  Used by
      scale_intensity_to_counts to convert the dimensionless weak-phase
      image into detector counts.

    Output
    ------
    Returns an ``intensity`` output in the same dimensionless (reference=1)
    scale as the other intensity-output imaging models, so the standard
    noise and quantization layers apply directly.

    Validation
    ----------
    Accelerating voltage must be > 0.  Cs, alpha, dose must be
    non-negative.  Defocus may be positive (underfocus) or negative
    (overfocus) per the Scherzer convention.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_pre_crop_optical_filtering = True

    def __init__(self, params: dict) -> None:
        self._V_kV = float(params.get("tem_acceleration_kV", 300.0))
        if self._V_kV <= 0.0:
            raise ValueError(
                f"PARAMS['tem_acceleration_kV'] must be positive; got {self._V_kV}."
            )
        self._Cs_mm = float(params.get("tem_Cs_mm", 0.5))
        if self._Cs_mm < 0.0:
            raise ValueError(
                f"PARAMS['tem_Cs_mm'] must be non-negative; got {self._Cs_mm}."
            )
        self._alpha_mrad = float(params.get("tem_partial_coherence_alpha_mrad", 0.1))
        if self._alpha_mrad < 0.0:
            raise ValueError(
                f"PARAMS['tem_partial_coherence_alpha_mrad'] must be non-negative; "
                f"got {self._alpha_mrad}."
            )

        # Resolve wavelength and defocus (defocus defaults to Scherzer).
        self._lambda_m = electron_wavelength_m(self._V_kV)
        if "tem_defocus_nm" in params and params["tem_defocus_nm"] is not None:
            self._defocus_m = 1.0e-9 * float(params["tem_defocus_nm"])
        else:
            self._defocus_m = scherzer_defocus_m(self._V_kV, self._Cs_mm)

        self._phase_shift_per_volt_nm = float(params.get(
            "tem_phase_shift_per_volt_nm",
            electron_interaction_parameter_rad_per_V_nm(self._V_kV),
        ))
        if (
            not np.isfinite(self._phase_shift_per_volt_nm)
            or self._phase_shift_per_volt_nm < 0.0
        ):
            raise ValueError(
                "PARAMS['tem_phase_shift_per_volt_nm'] must be finite and "
                f"non-negative; got {self._phase_shift_per_volt_nm}."
            )

        os_factor = int(params.get("psf_oversampling_factor", 1))
        if os_factor <= 0:
            raise ValueError(
                f"PARAMS['psf_oversampling_factor'] must be positive; got {os_factor}."
            )
        canvas_pitch_nm = float(params["pixel_size_nm"]) / float(os_factor)
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "PARAMS['pixel_size_nm'] / PARAMS['psf_oversampling_factor'] must "
                f"resolve to a positive pitch; got {canvas_pitch_nm} nm."
            )

        # Fourier-grid pitch is the physical pitch of the rendered model canvas.
        # tem_pixel_size_pm is retained only as a compatibility assertion so it
        # cannot silently make the CTF grid disagree with detector/Fisher units.
        self._pixel_size_m = 1.0e-9 * canvas_pitch_nm
        if "tem_pixel_size_pm" in params and params["tem_pixel_size_pm"] is not None:
            requested_m = 1.0e-12 * float(params["tem_pixel_size_pm"])
            if (
                not np.isfinite(requested_m)
                or requested_m <= 0.0
                or not np.isclose(requested_m, self._pixel_size_m, rtol=1e-6, atol=1e-15)
            ):
                raise ValueError(
                    "PARAMS['tem_pixel_size_pm'] must match the rendered model-canvas "
                    "pitch pixel_size_nm / psf_oversampling_factor. "
                    f"Got tem_pixel_size_pm={params['tem_pixel_size_pm']} pm and "
                    f"canvas pitch={canvas_pitch_nm * 1000.0:.6g} pm."
                )
        if not np.isfinite(self._pixel_size_m) or self._pixel_size_m <= 0.0:
            raise ValueError(
                "PARAMS['tem_pixel_size_pm'] or PARAMS['pixel_size_nm'] must resolve "
                f"to a positive pixel pitch; got {self._pixel_size_m} m."
            )

        self._dose_per_pixel = float(params.get(
            "tem_dose_per_pixel",
            100.0,
        ))
        if not np.isfinite(self._dose_per_pixel) or self._dose_per_pixel < 0.0:
            raise ValueError(
                "PARAMS['tem_dose_per_pixel'] must be finite and non-negative; "
                f"got {self._dose_per_pixel}."
            )

        # Cache the CTF array per frame shape. The CTF depends only on
        # the shape, pixel pitch, lambda, Cs, defocus, alpha, so once
        # computed it is reused across all frames of a run.
        self._ctf_cache: dict = {}

    # -- CTF construction -------------------------------------------------

    def _chi(self, k: np.ndarray) -> np.ndarray:
        """Aberration phase chi(k) = (pi/2) C_s lambda^3 k^4 - pi df lambda k^2.

        Standard TEM phase-contrast CTF (Reimer; Williams & Carter; Kirkland).
        With k as spatial frequency (cycles per metre, 1/m) and C_s, df,
        lambda all in metres, both terms are dimensionless: C_s * lambda^3 *
        k^4 = m * m^3 * m^-4 = 1 and df * lambda * k^2 = m * m * m^-2 = 1.
        """
        lam = self._lambda_m
        Cs_m = 1.0e-3 * self._Cs_mm
        df_m = self._defocus_m
        return (np.pi * (lam ** 3) * Cs_m * 0.5) * k ** 4 - (np.pi * lam * df_m) * k ** 2

    def _envelope(self, k: np.ndarray) -> np.ndarray:
        """Partial-coherence envelope from illumination half-angle alpha."""
        alpha_rad = 1.0e-3 * self._alpha_mrad
        if alpha_rad == 0.0:
            return np.ones_like(k)
        lam = self._lambda_m
        Cs_m = 1.0e-3 * self._Cs_mm
        df_m = self._defocus_m
        # Kirkland Eq. 5.77: exp(-pi^2 alpha^2 (Cs lambda^2 k^3 - df k)^2).
        arg = (Cs_m * lam ** 2) * k ** 3 - df_m * k
        return np.exp(-(np.pi * alpha_rad) ** 2 * arg ** 2)

    def _ctf(self, shape: tuple) -> np.ndarray:
        """Build (or retrieve from cache) the TEM CTF at the given shape."""
        cached = self._ctf_cache.get(shape)
        if cached is not None:
            return cached
        H, W = shape[-2], shape[-1]
        dx = self._pixel_size_m
        # fftfreq returns cycles per metre, matching the CTF helper formulas.
        fx = np.fft.fftfreq(W, d=dx)
        fy = np.fft.fftfreq(H, d=dx)
        kx = fx
        ky = fy
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        k = np.sqrt(KX ** 2 + KY ** 2)
        chi = self._chi(k)
        env = self._envelope(k)
        # Standard TEM phase-contrast CTF: -2 sin(chi) * envelope.
        ctf = -2.0 * np.sin(chi) * env
        if shape != (H, W):
            ctf = ctf.reshape(shape)
        self._ctf_cache[shape] = ctf
        return ctf

    def _apply_ctf(self, E_sca: np.ndarray) -> np.ndarray:
        """Return the real-valued phase-contrast image of E_sca as projected potential."""
        ctf = self._ctf(E_sca.shape)
        return np.real(np.fft.ifft2(ctf * np.fft.fft2(E_sca)))

    def _projected_phase_source(
        self,
        *,
        shape: tuple[int, int],
        center_x_canvas: float,
        center_y_canvas: float,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
    ) -> np.ndarray:
        source = np.zeros(shape, dtype=float)
        self.accumulate_particle_source(
            source,
            center_x_canvas=center_x_canvas,
            center_y_canvas=center_y_canvas,
            diameter_nm=diameter_nm,
            pixel_size_nm=pixel_size_nm,
            os_factor=os_factor,
            material_properties=material_properties,
            params=params,
        )
        return source

    def probe_wavelength_nm(self, params: dict) -> float:
        del params
        return float(self._lambda_m * 1.0e9)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        return {
            "kind": "tem_ctf",
            "probe_wavelength_nm": self.probe_wavelength_nm(params),
            "acceleration_kV": self._V_kV,
            "Cs_mm": self._Cs_mm,
            "defocus_m": self._defocus_m,
            "shape": tuple(shape),
        }

    # -- Contract methods -------------------------------------------------

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        TEM phase-contrast intensity.  With |E_direct|=1 and weak-phase
        approximation, the recorded intensity equals
            I(r) = 1 - 2 sigma V_proj * PSF_TEM(r)
        with E_sca representing sigma*V_proj becomes
            I(r) = 1 + CTF(k) * E_sca  (applied in Fourier space, real part).

        We return 1 + CTF-filtered E_sca (clamped at 0 from below to keep
        the downstream shot-noise layer physically valid).
        """
        filtered = self._apply_ctf(E_sca_total)
        I = 1.0 + filtered
        return np.maximum(I, 0.0)

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        return np.zeros(shape, dtype=float)

    def accumulate_particle_source(
        self,
        source_canvas,
        *,
        center_x_canvas: float,
        center_y_canvas: float,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
    ) -> None:
        del particle_z_nm
        if source_canvas is None:
            return
        mip = float(getattr(material_properties, "mean_inner_potential_V", 0.0))
        if mip <= 0.0 or self._phase_shift_per_volt_nm <= 0.0:
            return
        scale = float(params.get("tem_projected_potential_scale", 1.0))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "PARAMS['tem_projected_potential_scale'] must be finite and "
                f"non-negative; got {scale}."
            )
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        h, w = source_canvas.shape
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 2)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 3)))
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 2)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 3)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r <= radius_px
        thickness_px = np.zeros_like(r, dtype=float)
        thickness_px[inside] = 2.0 * np.sqrt(np.maximum(radius_px ** 2 - r[inside] ** 2, 0.0))
        thickness_nm = thickness_px * float(pixel_size_nm) / float(os_factor)
        phase = scale * self._phase_shift_per_volt_nm * mip * thickness_nm
        edge_width = max(0.75, 0.5 * float(os_factor))
        taper = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        source_canvas[y0:y1, x0:x1] += phase * taper

    def compute_scene_intensity(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, background_field, frame_index
        if particle_source_maps is None or len(particle_source_maps) == 0:
            source = np.zeros_like(E_sca_total, dtype=float)
        else:
            source = np.sum(
                [np.asarray(source_map, dtype=float) for source_map in particle_source_maps],
                axis=0,
            )
        return np.maximum(1.0 + self._apply_ctf(source), 0.0)

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
        scale = float(params.get("tem_sample_environment_potential_scale", 1.0))
        substrate_phase = scale * self._phase_shift_per_volt_nm * sample_environment.substrate.projected_potential_V_nm()
        substrate_contrast = self._apply_ctf(substrate_phase)
        return np.maximum(intensity + substrate_contrast, 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Legacy direct-call contrast: CTF-filtered projected phase source."""
        del background_field, params
        return self._apply_ctf(E_sca_particle)

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        background = background_field
        if particle_instance is None:
            return self.compute_per_particle_contrast(E_sca_particle, background, params)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct TEM particle contrast for composite particles requires a "
                "rendered source map; use compute_particle_contrast_from_source_map()."
            )
        shape = E_sca_particle.shape
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(np.clip(int(frame_index), 0, traj.shape[0] - 1))
        os_factor = int(params.get("psf_oversampling_factor", 1))
        px = float(traj[frame_idx, 0]) / float(params["pixel_size_nm"]) * float(os_factor)
        py = float(traj[frame_idx, 1]) / float(params["pixel_size_nm"]) * float(os_factor)
        source = self._projected_phase_source(
            shape=shape,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=float(params["pixel_size_nm"]),
            os_factor=os_factor,
            material_properties=getattr(particle_instance, "material_properties", None),
            params=params,
        )
        return self._apply_ctf(source)

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params, frame_index
        return self._apply_ctf(np.asarray(particle_source_map, dtype=float))

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        TEM intensity-to-counts conversion.

        The TEM model's compute_intensity returns a dimensionless image
        normalized such that the unscattered direct beam has intensity
        one.  We convert to detector counts by multiplying by the mean
        electron dose per pixel, falling back to the generic
        background_intensity if ``tem_dose_per_pixel`` is not set.
        """
        dose = float(params.get(
            "tem_dose_per_pixel",
            self._dose_per_pixel,
        ))
        if not np.isfinite(dose) or dose < 0.0:
            raise ValueError(
                "PARAMS['tem_dose_per_pixel'] must be finite and non-negative; "
                f"got {dose}."
            )
        return dose * intensity


# ---------------------------------------------------------------------------
# Scanning electron microscopy (SEM) — secondary-electron topography contrast
# ---------------------------------------------------------------------------

class ScanningElectronMicroscopyImagingModel(ImagingModel):
    """
    Scanning electron microscopy (SEM) imaging model — secondary-electron
    topography contrast.

    SEM differs from TEM in two important ways that together define its
    characteristic image appearance:

    * The signal is not a transmitted electron wave but a per-pixel
      *secondary-electron yield* proxy.  The implemented source is weighted by
      the particle material's nominal secondary-electron yield coefficient and
      by the gradient of that projected source, which gives the expected
      edge-brightening behavior without claiming a full surface-transport
      calculation.

    * The beam is raster-scanned with a finite-size probe and the
      resulting signal is the convolution of the material's per-point
      emission with the probe intensity profile.  The probe size, not
      the illumination wavelength, is what ultimately limits SEM
      resolution.

    Syniscopy field representation
    ------------------------------
    Syniscopy's hybrid-assembly pipeline provides a projected particle source
    canvas weighted by the particle material's nominal secondary-electron yield.
    The SEM proxy model composes a secondary-electron response image as
          SE(r) = delta_baseline
                + edge_gain  * |nabla S(r)|
                + bulk_gain  * S(r),
    convolved by a Gaussian probe of standard deviation
    ``sem_probe_sigma_pixels``.  The edge term is the source-gradient
    magnitude and produces the characteristic SEM edge-brightening at particle
    boundaries.  This remains a qualitative secondary-electron proxy, not a
    Monte Carlo electron-transport model.

    Parameters (PARAMS keys, optional, nominal defaults)
    ----------------------------------------------------
    The defaults set a stable moderate-contrast synthetic SEM regime; use
    calibrated values for instrument-specific studies.

    - ``sem_probe_sigma_pixels``    (default 1.0) Gaussian probe spot size
    - ``sem_edge_contrast_gain``    (default 10.0) weight on the gradient-
                                    magnitude term (secondary-emission edge
                                    enhancement).
    - ``sem_bulk_contrast_gain``    (default 1.0) weight on the material
                                    source term (bulk/Z-like contribution).
    - ``sem_baseline_yield``        (default 0.05) yield from the substrate
                                    with no particle present.
    - ``sem_electrons_per_pixel``   (default 1000.0) dose scale used by
                                    scale_intensity_to_counts.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_pre_crop_optical_filtering = True

    def __init__(self, params: dict) -> None:
        self._probe_sigma_px = float(params.get("sem_probe_sigma_pixels", 1.0))
        if self._probe_sigma_px < 0.0:
            raise ValueError(
                f"PARAMS['sem_probe_sigma_pixels'] must be non-negative; "
                f"got {self._probe_sigma_px}."
            )
        self._edge_gain = float(params.get("sem_edge_contrast_gain", 10.0))
        self._bulk_gain = float(params.get("sem_bulk_contrast_gain", 1.0))
        self._baseline = float(params.get("sem_baseline_yield", 0.05))
        if self._baseline < 0.0:
            raise ValueError(
                f"PARAMS['sem_baseline_yield'] must be non-negative; "
                f"got {self._baseline}."
            )

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _gradient_magnitude(rho: np.ndarray) -> np.ndarray:
        """2D gradient magnitude via central differences (vectorized)."""
        if min(rho.shape) < 2:
            return np.zeros_like(rho, dtype=float)
        gy = np.empty_like(rho)
        gx = np.empty_like(rho)
        gy[1:-1, :] = 0.5 * (rho[2:, :] - rho[:-2, :])
        gy[0, :] = rho[1, :] - rho[0, :]
        gy[-1, :] = rho[-1, :] - rho[-2, :]
        gx[:, 1:-1] = 0.5 * (rho[:, 2:] - rho[:, :-2])
        gx[:, 0] = rho[:, 1] - rho[:, 0]
        gx[:, -1] = rho[:, -1] - rho[:, -2]
        return np.sqrt(gx ** 2 + gy ** 2)

    def _probe_blur(self, arr: np.ndarray) -> np.ndarray:
        """Gaussian blur approximating the SEM probe profile.

        Uses scipy's gaussian_filter if available, otherwise falls back to
        a separable direct 1D Gaussian convolution. We keep the fallback so
        the model has no hard scipy dependency.
        """
        if self._probe_sigma_px == 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=self._probe_sigma_px)
        except ImportError:
            # Separable direct Gaussian convolution along each axis.
            # Build a 1D Gaussian kernel of radius 4*sigma (truncate).
            sigma = self._probe_sigma_px
            radius = max(int(4 * sigma), 1)
            x = np.arange(-radius, radius + 1, dtype=float)
            k1d = np.exp(-0.5 * (x / sigma) ** 2)
            k1d /= k1d.sum()
            out = arr.astype(float, copy=True)
            # Convolve along rows and then columns.
            for axis in (0, 1):
                out = np.apply_along_axis(
                    lambda v: np.convolve(v, k1d, mode="same"),
                    axis, out,
                )
            return out

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        return {
            "kind": "sem_gaussian_probe",
            "probe_wavelength_nm": self.probe_wavelength_nm(params),
            "acceleration_kV": float(params.get("sem_acceleration_kV", 5.0)),
            "probe_sigma_pixels": self._probe_sigma_px,
            "shape": tuple(shape),
        }

    def probe_wavelength_nm(self, params: dict) -> float:
        acceleration_kV = float(params.get("sem_acceleration_kV", 5.0))
        return float(electron_wavelength_m(acceleration_kV) * 1.0e9)

    # -- Contract methods -------------------------------------------------

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Return the SEM secondary-electron yield image (dimensionless)."""
        rho = np.abs(E_sca_total) ** 2
        return self._yield_from_source(rho)

    def _yield_from_source(self, source: np.ndarray) -> np.ndarray:
        return np.maximum(self._baseline + self._contrast_from_source(source), 0.0)

    def _contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        source = np.maximum(source, 0.0)
        edge = self._gradient_magnitude(source)
        source_blur = self._probe_blur(source)
        edge_blur = self._probe_blur(edge)
        se = self._edge_gain * edge_blur + self._bulk_gain * source_blur
        return np.maximum(se, 0.0)

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        return np.zeros(shape, dtype=float)

    def accumulate_particle_source(
        self,
        source_canvas,
        *,
        center_x_canvas: float,
        center_y_canvas: float,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
    ) -> None:
        del params, particle_z_nm
        if source_canvas is None:
            return
        yield_coeff = float(getattr(material_properties, "se_yield_coefficient", 0.0))
        if yield_coeff <= 0.0:
            return
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        h, w = source_canvas.shape
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 2)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 3)))
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 2)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 3)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r <= radius_px
        thickness_px = np.zeros_like(r, dtype=float)
        thickness_px[inside] = 2.0 * np.sqrt(np.maximum(radius_px ** 2 - r[inside] ** 2, 0.0))
        edge_width = max(0.75, 0.5 * float(os_factor))
        taper = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        # Normalize by diameter so ``se_yield_coefficient`` remains the main
        # material-scale control rather than growing quadratically with size.
        diameter_px = max(2.0 * radius_px, 1.0)
        source_canvas[y0:y1, x0:x1] += yield_coeff * (thickness_px / diameter_px) * taper

    def compute_scene_intensity(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, background_field, params, frame_index
        if particle_source_maps is None or len(particle_source_maps) == 0:
            source = np.abs(E_sca_total) ** 2
        else:
            source = np.sum(
                [np.asarray(source_map, dtype=float) for source_map in particle_source_maps],
                axis=0,
            )
        return self._yield_from_source(source)

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
        topo = sample_environment.substrate.topography_gradient()
        yield_map = sample_environment.substrate.secondary_electron_yield_map()
        edge_gain = float(params.get("sem_sample_environment_edge_gain", self._edge_gain))
        substrate = yield_map + edge_gain * topo
        substrate = self._probe_blur(substrate)
        return np.maximum(intensity + substrate, 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Per-particle SEM contrast: baseline-subtracted SE image."""
        rho = np.abs(E_sca_particle) ** 2
        edge = self._gradient_magnitude(rho)
        rho_blur = self._probe_blur(rho)
        edge_blur = self._probe_blur(edge)
        return self._edge_gain * edge_blur + self._bulk_gain * rho_blur

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        if particle_instance is None:
            return self.compute_per_particle_contrast(E_sca_particle, background_field, params)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct SEM particle contrast for composite particles requires a "
                "rendered source map; use compute_particle_contrast_from_source_map()."
            )
        del background_field
        source = np.zeros_like(E_sca_particle, dtype=float)
        material = getattr(particle_instance, "material_properties", None)
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(np.clip(int(frame_index), 0, traj.shape[0] - 1))
        px = float(traj[frame_idx, 0]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        py = float(traj[frame_idx, 1]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        self.accumulate_particle_source(
            source,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=float(params["pixel_size_nm"]),
            os_factor=int(params.get("psf_oversampling_factor", 1)),
            material_properties=material,
            params=params,
            particle_z_nm=pz,
        )
        return self._contrast_from_source(source)

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params, frame_index
        return self._contrast_from_source(np.asarray(particle_source_map, dtype=float))

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Convert dimensionless SE yield to detector electron counts.

        SEM detectors report integrated electron counts per pixel; we
        multiply the dimensionless yield by the per-pixel dose.  No
        reference-beam division is involved (no E_ref in SEM).
        """
        dose = float(params.get(
            "sem_electrons_per_pixel",
            float(params.get("background_intensity", 1000.0)),
        ))
        return dose * intensity


# ---------------------------------------------------------------------------
# Fluorescence widefield (incoherent photon emission)
# ---------------------------------------------------------------------------

class FluorescenceWidefieldImagingModel(ImagingModel):
    """
    Widefield epi-fluorescence imaging model.

    Fluorescence is rendered from material-property source maps, not from
    coherent scattering intensity. During rendering each particle/sub-particle
    contributes a projected emitter-density profile weighted by its chord
    length through the sphere, MaterialProperties.fluorophore_density, and
    excitation/emission spectral overlap. The scene source map is then blurred
    by the emission PSF and scaled to detector counts by scale_intensity_to_counts.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_pre_crop_optical_filtering = True

    def __init__(self, params: dict) -> None:
        self._Qf = float(params.get("fluorescence_quantum_yield", 0.5))
        if not (0.0 <= self._Qf <= 1.0):
            raise ValueError(
                f"PARAMS['fluorescence_quantum_yield'] must be in [0, 1]; "
                f"got {self._Qf}."
            )
        self._excitation = float(params.get("fluorescence_excitation_scale", 1.0))
        if self._excitation < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_excitation_scale'] must be non-negative; "
                f"got {self._excitation}."
            )
        self._emission_sigma_px = float(
            params.get("fluorescence_emission_psf_sigma_px", 1.0)
        )
        if self._emission_sigma_px < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_emission_psf_sigma_px'] must be "
                f"non-negative; got {self._emission_sigma_px}."
            )
        self._uniform_background = float(params.get("fluorescence_background", 0.0))
        if self._uniform_background < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_background'] must be non-negative; "
                f"got {self._uniform_background}."
            )
        self._spectral_bandwidth_nm = float(params.get("fluorescence_spectral_bandwidth_nm", 40.0))
        if self._spectral_bandwidth_nm <= 0.0:
            raise ValueError("PARAMS['fluorescence_spectral_bandwidth_nm'] must be positive.")
        self._tau_frames = params.get("fluorescence_photobleach_tau_frames", None)
        if self._tau_frames is not None:
            self._tau_frames = float(self._tau_frames)
            if self._tau_frames <= 0.0:
                raise ValueError(
                    f"PARAMS['fluorescence_photobleach_tau_frames'] must be "
                    f"positive when set; got {self._tau_frames}."
                )

    def _emission_blur(self, arr: np.ndarray) -> np.ndarray:
        if self._emission_sigma_px == 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=self._emission_sigma_px)
        except ImportError:
            sigma = self._emission_sigma_px
            radius = max(int(4 * sigma), 1)
            x = np.arange(-radius, radius + 1, dtype=float)
            k1d = np.exp(-0.5 * (x / sigma) ** 2)
            k1d /= k1d.sum()
            out = arr.astype(float, copy=True)
            for axis in (0, 1):
                out = np.apply_along_axis(
                    lambda v: np.convolve(v, k1d, mode="same"),
                    axis, out,
                )
            return out

    def _bleach_factor(self, frame_index: int = 0) -> float:
        if self._tau_frames is None:
            return 1.0
        t = float(frame_index)
        return float(np.exp(-t / self._tau_frames))

    def _spectral_factor(self, peak_nm: float | None, wavelength_nm: float) -> float:
        if peak_nm is None:
            return 1.0
        peak = float(peak_nm)
        if peak <= 0.0:
            raise ValueError("Material excitation/emission peak wavelengths must be positive when set.")
        delta = (float(wavelength_nm) - peak) / self._spectral_bandwidth_nm
        return float(np.exp(-0.5 * delta * delta))

    def _material_source_scale(self, material, params: dict) -> float:
        if material is None:
            return 0.0
        density = float(getattr(material, "fluorophore_density", 0.0))
        if density <= 0.0:
            return 0.0
        excitation_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
        emission_nm = self.probe_wavelength_nm(params)
        return (
            density
            * self._spectral_factor(getattr(material, "excitation_peak_nm", None), excitation_nm)
            * self._spectral_factor(getattr(material, "emission_peak_nm", None), emission_nm)
        )

    def _material_source_scale_for_particle(
        self,
        material,
        params: dict,
        *,
        particle_z_nm: float | None = None,
    ) -> float:
        del particle_z_nm
        return self._material_source_scale(material, params)

    def probe_wavelength_nm(self, params: dict) -> float:
        return float(params.get("fluorescence_emission_wavelength_nm", 520.0))

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        del params
        return np.ones(shape, dtype=float)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        return {
            "kind": "fluorescence_emission_psf",
            "excitation_wavelength_nm": float(params.get("fluorescence_excitation_wavelength_nm", 488.0)),
            "emission_wavelength_nm": self.probe_wavelength_nm(params),
            "probe_wavelength_nm": self.probe_wavelength_nm(params),
            "emission_sigma_px": self._emission_sigma_px,
            "spectral_bandwidth_nm": self._spectral_bandwidth_nm,
            "shape": tuple(shape),
        }

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        return np.zeros(shape, dtype=float)

    def accumulate_particle_source(
        self,
        source_canvas,
        *,
        center_x_canvas: int,
        center_y_canvas: int,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
    ) -> None:
        if source_canvas is None:
            return
        scale = self._material_source_scale_for_particle(
            material_properties,
            params,
            particle_z_nm=particle_z_nm,
        )
        if scale <= 0.0:
            return
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        h, w = source_canvas.shape
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 1)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 2)))
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 1)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 2)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r <= radius_px
        thickness_px = np.zeros_like(r, dtype=float)
        thickness_px[inside] = 2.0 * np.sqrt(np.maximum(radius_px ** 2 - r[inside] ** 2, 0.0))
        thickness_nm = thickness_px * float(pixel_size_nm) / float(os_factor)
        edge_width = max(0.75, 0.5 * float(os_factor))
        disk = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        source_canvas[y0:y1, x0:x1] += scale * thickness_nm * disk

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_field
        # Direct calls without render-supplied material source maps have no
        # particle fluorescence source, so they return a background-only image.
        return np.full(E_sca_total.shape, self._uniform_background, dtype=float)

    def compute_scene_intensity(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, background_field
        if particle_source_maps is None or len(particle_source_maps) == 0:
            source = np.zeros_like(E_sca_total, dtype=float)
        else:
            source = np.sum(np.asarray(particle_source_maps, dtype=float), axis=0)
        emission = self._emission_blur(source)
        bleach = self._bleach_factor(frame_index=frame_index)
        intensity = self._Qf * self._excitation * bleach * emission + self._uniform_background
        return np.maximum(intensity, 0.0)

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field
        if particle_instance is None:
            return np.zeros_like(E_sca_particle, dtype=float)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct fluorescence particle contrast for composite particles "
                "requires a rendered source map; use compute_particle_contrast_from_source_map()."
            )
        shape = E_sca_particle.shape
        source = np.zeros(shape, dtype=float)
        material = getattr(particle_instance, "material_properties", None)
        scale = self._material_source_scale(material, params)
        if scale <= 0.0:
            return source
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(frame_index)
        frame_idx = int(np.clip(frame_idx, 0, traj.shape[0] - 1))
        px = float(traj[frame_idx, 0]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        py = float(traj[frame_idx, 1]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        self.accumulate_particle_source(
            source,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=float(params["pixel_size_nm"]),
            os_factor=int(params.get("psf_oversampling_factor", 1)),
            material_properties=material,
            params=params,
            particle_z_nm=pz,
        )
        return (
            self._Qf
            * self._excitation
            * self._bleach_factor(frame_index=frame_idx)
            * self._emission_blur(source)
        )

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params
        source = np.asarray(particle_source_map, dtype=float)
        return (
            self._Qf
            * self._excitation
            * self._bleach_factor(frame_index=frame_index)
            * self._emission_blur(source)
        )

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_field, params
        return np.zeros_like(E_sca_particle, dtype=float)

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
        excitation_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
        reflection = sample_environment.substrate.reflection_amplitude(excitation_nm)
        modulation = _mean_normalized_map(np.abs(1.0 + reflection) ** 2)
        mod_gain = float(params.get("fluorescence_sample_environment_excitation_modulation_gain", 0.25))
        autofl_gain = float(params.get("fluorescence_sample_environment_autofluorescence_gain", 1.0))
        autofl = autofl_gain * self._emission_blur(
            sample_environment.substrate.autofluorescence_density()
        )
        return np.maximum(intensity * (1.0 + mod_gain * (modulation - 1.0)) + autofl, 0.0)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_final, E_ref_intensity_final
        scale = float(params.get(
            "fluorescence_photon_count_scale",
            float(params.get("background_intensity", 500.0)),
        ))
        return scale * intensity


class TIRFFluorescenceImagingModel(FluorescenceWidefieldImagingModel):
    """TIRF fluorescence with evanescent excitation applied to material source maps."""

    @staticmethod
    def penetration_depth_nm(params: dict) -> float:
        if params.get("tirf_use_angle_derived_penetration_depth", False):
            wavelength_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
            n_prism = float(params.get("tirf_prism_refractive_index", 1.518))
            n_sample = float(params.get("tirf_sample_refractive_index", params.get("refractive_index_medium", 1.333)))
            angle_rad = np.deg2rad(float(params.get("tirf_incident_angle_deg", 66.0)))
            sin_term = n_prism * np.sin(angle_rad)
            under_root = sin_term * sin_term - n_sample * n_sample
            if under_root <= 0.0:
                raise ValueError(
                    "TIRF incident angle must exceed the critical angle when "
                    "'tirf_use_angle_derived_penetration_depth' is enabled."
                )
            return float(wavelength_nm / (4.0 * np.pi * np.sqrt(under_root)))
        penetration_nm = float(params.get("tirf_penetration_depth_nm", 120.0))
        if penetration_nm <= 0.0:
            raise ValueError("PARAMS['tirf_penetration_depth_nm'] must be positive.")
        return penetration_nm

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        effective_na = params.get("tirf_effective_numerical_aperture", None)
        if effective_na is None:
            self._tirf_emission_sigma_multiplier = 1.0
        else:
            effective_na = float(effective_na)
            if effective_na <= 0.0:
                raise ValueError("PARAMS['tirf_effective_numerical_aperture'] must be positive when set.")
            detection_na = float(params.get("numerical_aperture", effective_na))
            if detection_na <= 0.0:
                raise ValueError(
                    "PARAMS['numerical_aperture'] must be positive when TIRF "
                    "effective NA is set."
                )
            self._tirf_emission_sigma_multiplier = max(detection_na / effective_na, 1e-6)

    def _emission_blur(self, arr: np.ndarray) -> np.ndarray:
        sigma = self._emission_sigma_px * self._tirf_emission_sigma_multiplier
        if sigma == 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=sigma)
        except ImportError:
            radius = max(int(4 * sigma), 1)
            x = np.arange(-radius, radius + 1, dtype=float)
            k1d = np.exp(-0.5 * (x / sigma) ** 2)
            k1d /= k1d.sum()
            out = arr.astype(float, copy=True)
            for axis in (0, 1):
                out = np.apply_along_axis(
                    lambda v: np.convolve(v, k1d, mode="same"),
                    axis, out,
                )
            return out

    def _material_source_scale_for_particle(
        self,
        material,
        params: dict,
        *,
        particle_z_nm: float | None = None,
    ) -> float:
        base = super()._material_source_scale_for_particle(
            material,
            params,
            particle_z_nm=particle_z_nm,
        )
        penetration_nm = self.penetration_depth_nm(params)
        if particle_z_nm is None:
            particle_height_nm = float(params.get("tirf_particle_height_nm", 0.0))
        else:
            particle_height_nm = float(particle_z_nm) + float(params.get("tirf_height_offset_nm", 0.0))
        excitation_factor = np.exp(-max(particle_height_nm, 0.0) / penetration_nm)
        return float(base * excitation_factor)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update(
            kind="tirf_evanescent_fluorescence",
            penetration_depth_nm=self.penetration_depth_nm(params),
            emission_sigma_multiplier=self._tirf_emission_sigma_multiplier,
        )
        return response


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "bright_field": _PartiallyCoherentBrightfieldImagingModel,
    "fluorescence_widefield": FluorescenceWidefieldImagingModel,
    "tirf_fluorescence": TIRFFluorescenceImagingModel,
    "dark_field": _AnnularDarkFieldImagingModel,
    "zernike_phase_contrast": ZernikePhaseContrastImagingModel,
    "differential_phase_contrast": DifferentialPhaseContrastImagingModel,
    "quantitative_phase": QuantitativePhaseImagingModel,
    "off_axis_holography": OffAxisHolographyImagingModel,
    "ricm": ReflectionInterferenceContrastImagingModel,
    "interferometric": InterferometricImagingModel,
    "tem_phase_contrast": TransmissionElectronMicroscopyImagingModel,
    "sem_secondary_electron": ScanningElectronMicroscopyImagingModel,
    "partially_coherent_bright_field": _PartiallyCoherentBrightfieldImagingModel,
    "coherent_bright_field": CoherentBrightfieldImagingModel,
    "coherent_dark_field": CoherentDarkFieldImagingModel,
}


def get_imaging_model_class(model_name: str) -> type[ImagingModel]:
    """Return the registered imaging-model class for ``model_name``."""
    key = canonical_modality_name(model_name)
    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown imaging_model '{model_name}'. "
            f"Supported values are: {list(_MODEL_REGISTRY.keys())}."
        )
    return cls


def modality_uses_sample_environment_pattern(model_name: str) -> bool:
    """Whether ``model_name`` physically uses the optical substrate pattern."""
    return bool(
        getattr(get_imaging_model_class(model_name), "uses_sample_environment_pattern", False)
    )


def modality_uses_relative_reference_contrast(model_name: str) -> bool:
    """Whether reference-frame subtraction should use ``(signal - reference) / reference``."""
    return canonical_modality_name(model_name) in RELATIVE_REFERENCE_CONTRAST_MODALITIES


def modality_display_name(model_name: str) -> str:
    """Human-readable modality name for reports and generated tables."""
    names = {
        "bright_field": "partially coherent Köhler bright-field",
        "fluorescence_widefield": "widefield fluorescence",
        "tirf_fluorescence": "TIRF fluorescence",
        "dark_field": "annular Köhler dark-field",
        "partially_coherent_bright_field": "partially coherent Köhler bright-field",
        "coherent_bright_field": "coherent bright-field (COBRI)",
        "coherent_dark_field": "coherent dark-field (zero-order blocked)",
        "zernike_phase_contrast": "Zernike phase contrast",
        "differential_phase_contrast": "differential phase contrast (DPC)",
        "quantitative_phase": "quantitative phase imaging (QPI)",
        "off_axis_holography": "off-axis digital holography (DHM)",
        "ricm": "reflection interference contrast (RICM)",
        "interferometric": "interferometric scattering (iSCAT)",
        "tem_phase_contrast": "TEM phase contrast",
        "sem_secondary_electron": "SEM secondary-electron",
    }
    key = canonical_modality_name(model_name)
    label = names.get(key, str(model_name).replace("_", " "))
    return label


def get_imaging_model(params: dict) -> ImagingModel:
    """
    Instantiate and return the imaging model specified by PARAMS['imaging_model'].

    If no value is supplied, the project configuration selects ``"bright_field"``.

    Args:
        params: Global simulation parameter dictionary (PARAMS).

    Returns:
        An instance of the appropriate imaging model class.

    Raises:
        ValueError: If the model name is unknown.
    """
    model_name = canonical_modality_name(params.get("imaging_model", "bright_field"))

    return get_imaging_model_class(model_name)(params)


__all__ = [
    "LABEL_FREE_OPTICAL_MODALITIES",
    "CANONICAL_COHERENT_MODALITIES",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "MODALITY_ALIASES",
    "canonical_modality_name",
    "InterferometricImagingModel",
    "CoherentDarkFieldImagingModel",
    "CoherentBrightfieldImagingModel",
    "ZernikePhaseContrastImagingModel",
    "DifferentialPhaseContrastImagingModel",
    "QuantitativePhaseImagingModel",
    "ReflectionInterferenceContrastImagingModel",
    "OffAxisHolographyImagingModel",
    "electron_wavelength_m",
    "electron_interaction_parameter_rad_per_V_nm",
    "scherzer_defocus_m",
    "TransmissionElectronMicroscopyImagingModel",
    "ScanningElectronMicroscopyImagingModel",
    "FluorescenceWidefieldImagingModel",
    "TIRFFluorescenceImagingModel",
    "get_imaging_model_class",
    "modality_uses_sample_environment_pattern",
    "modality_uses_relative_reference_contrast",
    "modality_display_name",
    "get_imaging_model",
]
