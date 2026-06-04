"""Typed runtime views over validated simulation parameter dictionaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from .defaults import PARAMS, RUNTIME_INTERNAL_DEFAULTS
from modality_registry import canonical_modality_name


def _copy_if_mutable(value: Any) -> Any:
    if isinstance(value, (dict, list, set)):
        return deepcopy(value)
    return value


def param_value(params: Mapping[str, Any], key: str) -> Any:
    """Return a simulation parameter, deriving missing values from PARAMS.

    Leaf modules should use this for PARAMS-owned fields instead of spelling a
    local fallback literal. Normalized parameter dictionaries already contain
    every public key; this only protects direct helper use with partial dicts.
    """
    if key in params:
        return params[key]
    if key in PARAMS:
        return _copy_if_mutable(PARAMS[key])
    raise KeyError(f"Unknown simulation parameter key: {key!r}")


def internal_param_value(params: Mapping[str, Any], key: str) -> Any:
    if key in params:
        return params[key]
    if key in RUNTIME_INTERNAL_DEFAULTS:
        return _copy_if_mutable(RUNTIME_INTERNAL_DEFAULTS[key])
    raise KeyError(f"Unknown internal simulation parameter key: {key!r}")


def _finite_float(value: Any, *, key: str, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"PARAMS['{key}'] must be finite; got {value!r}.")
    if positive and out <= 0.0:
        raise ValueError(f"PARAMS['{key}'] must be positive; got {out!r}.")
    if nonnegative and out < 0.0:
        raise ValueError(f"PARAMS['{key}'] must be non-negative; got {out!r}.")
    return out


def resolved_modality(params: Mapping[str, Any]) -> str:
    """Return the canonical runtime imaging modality."""
    return canonical_modality_name(str(param_value(params, "imaging_model")).strip())


def resolved_pixel_size_nm(params: Mapping[str, Any]) -> float:
    return _finite_float(param_value(params, "pixel_size_nm"), key="pixel_size_nm", positive=True)


def resolved_psf_oversampling_factor(params: Mapping[str, Any]) -> int:
    value = int(param_value(params, "psf_oversampling_factor"))
    if value <= 0:
        raise ValueError(f"PARAMS['psf_oversampling_factor'] must be positive; got {value!r}.")
    return value


def resolved_image_size_pixels(params: Mapping[str, Any]) -> int:
    value = int(param_value(params, "image_size_pixels"))
    if value <= 0:
        raise ValueError(f"PARAMS['image_size_pixels'] must be positive; got {value!r}.")
    return value


def resolved_model_canvas_shape(params: Mapping[str, Any]) -> tuple[int, int]:
    side = resolved_image_size_pixels(params) * resolved_psf_oversampling_factor(params)
    return side, side


def resolved_model_canvas_pixel_size_nm(params: Mapping[str, Any]) -> float:
    return resolved_pixel_size_nm(params) / float(resolved_psf_oversampling_factor(params))


def resolved_vectorial_pupil_samples(params: Mapping[str, Any]) -> int:
    raw = param_value(params, "vectorial_pupil_samples")
    if raw is None:
        raw = param_value(params, "pupil_samples")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"vectorial_pupil_samples/pupil_samples must be positive; got {value!r}.")
    return value


def resolved_background_intensity(params: Mapping[str, Any]) -> float:
    return _finite_float(param_value(params, "background_intensity"), key="background_intensity", nonnegative=True)


def resolved_dark_field_background_count(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "dark_field_background_count"),
        key="dark_field_background_count",
        nonnegative=True,
    )


def resolved_dark_field_illumination_count(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "dark_field_illumination_count"),
        key="dark_field_illumination_count",
        nonnegative=True,
    )


def resolved_qpi_phase_to_count_scale(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "qpi_phase_to_count_scale"),
        key="qpi_phase_to_count_scale",
        positive=True,
    )


def resolved_fluorescence_photon_count_scale(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "fluorescence_photon_count_scale"),
        key="fluorescence_photon_count_scale",
        nonnegative=True,
    )


def resolved_sem_electrons_per_pixel(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "sem_electrons_per_pixel"),
        key="sem_electrons_per_pixel",
        nonnegative=True,
    )


def resolved_tem_dose_per_pixel(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "tem_dose_per_pixel"),
        key="tem_dose_per_pixel",
        nonnegative=True,
    )


def resolved_detector_qe(params: Mapping[str, Any], *, fluorescence: bool = False) -> float:
    if fluorescence and "fluorescence_detector_qe" in params:
        value = params["fluorescence_detector_qe"]
    elif "detector_qe" in params:
        value = params["detector_qe"]
    else:
        value = param_value(params, "fluorescence_detector_qe" if fluorescence else "detector_qe")
    qe = _finite_float(value, key="detector_qe", nonnegative=True)
    if qe > 1.0:
        raise ValueError(f"detector_qe must be <= 1; got {qe!r}.")
    return qe


def resolved_reference_field_amplitude(params: Mapping[str, Any]) -> float:
    return _finite_float(
        param_value(params, "reference_field_amplitude"),
        key="reference_field_amplitude",
        nonnegative=True,
    )


def resolved_vectorial_detection_mode(params: Mapping[str, Any]) -> str:
    return str(param_value(params, "vectorial_detection_mode")).strip().lower()


def resolved_random_seed(params: Mapping[str, Any]) -> int | None:
    raw = param_value(params, "random_seed")
    return None if raw is None else int(raw)


def resolved_sem_monte_carlo_seed(params: Mapping[str, Any]) -> int:
    raw = param_value(params, "sem_monte_carlo_seed")
    if raw is None:
        raw = resolved_random_seed(params)
    return 0 if raw is None else int(raw)


def resolved_fps(params: Mapping[str, Any]) -> float:
    return _finite_float(param_value(params, "fps"), key="fps", positive=True)


def resolved_temperature_K(params: Mapping[str, Any]) -> float:
    return _finite_float(param_value(params, "temperature_K"), key="temperature_K", positive=True)


def resolved_viscosity_Pa_s(params: Mapping[str, Any]) -> float:
    return _finite_float(param_value(params, "viscosity_Pa_s"), key="viscosity_Pa_s", positive=True)


def resolved_rotational_diffusion_enabled(params: Mapping[str, Any]) -> bool:
    return bool(param_value(params, "rotational_diffusion_enabled"))


def resolved_particles(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    particles = param_value(params, "particles")
    if particles is None:
        return []
    if not isinstance(particles, list):
        raise TypeError("PARAMS['particles'] must be a list of particle specs.")
    return deepcopy(particles)


def resolved_sample_environment_pattern_dimensions(params: Mapping[str, Any]) -> dict[str, Any]:
    raw = param_value(params, "sample_environment_pattern_dimensions")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("PARAMS['sample_environment_pattern_dimensions'] must be a dictionary.")
    dimensions = deepcopy(PARAMS["sample_environment_pattern_dimensions"])
    dimensions.update(dict(raw))
    return dimensions


def resolved_sample_environment_pattern_dimension(params: Mapping[str, Any], key: str) -> Any:
    dimensions = resolved_sample_environment_pattern_dimensions(params)
    if key not in dimensions:
        raise KeyError(f"Unknown sample_environment_pattern_dimensions key: {key!r}.")
    return dimensions[key]


@dataclass(frozen=True)
class SamplingGeometry:
    """Resolved detector/model-canvas sampling parameters."""

    image_size_pixels: int
    detector_pixel_size_nm: float
    psf_oversampling_factor: int
    vectorial_pupil_samples: int

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SamplingGeometry":
        return cls(
            image_size_pixels=resolved_image_size_pixels(params),
            detector_pixel_size_nm=resolved_pixel_size_nm(params),
            psf_oversampling_factor=resolved_psf_oversampling_factor(params),
            vectorial_pupil_samples=resolved_vectorial_pupil_samples(params),
        )

    @property
    def model_canvas_shape(self) -> tuple[int, int]:
        side = self.image_size_pixels * self.psf_oversampling_factor
        return side, side

    @property
    def model_canvas_pixel_size_nm(self) -> float:
        return self.detector_pixel_size_nm / float(self.psf_oversampling_factor)


@dataclass(frozen=True)
class CountBudgetSettings:
    """Resolved count/dose scaling parameters shared by detector-domain models."""

    background_intensity: float
    dark_field_background_count: float
    dark_field_illumination_count: float
    qpi_phase_to_count_scale: float
    fluorescence_photon_count_scale: float
    sem_electrons_per_pixel: float
    tem_dose_per_pixel: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "CountBudgetSettings":
        return cls(
            background_intensity=resolved_background_intensity(params),
            dark_field_background_count=resolved_dark_field_background_count(params),
            dark_field_illumination_count=resolved_dark_field_illumination_count(params),
            qpi_phase_to_count_scale=resolved_qpi_phase_to_count_scale(params),
            fluorescence_photon_count_scale=resolved_fluorescence_photon_count_scale(params),
            sem_electrons_per_pixel=resolved_sem_electrons_per_pixel(params),
            tem_dose_per_pixel=resolved_tem_dose_per_pixel(params),
        )


@dataclass(frozen=True)
class DetectorSettings:
    """Resolved detector efficiency settings."""

    detector_qe: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, fluorescence: bool = False) -> "DetectorSettings":
        return cls(detector_qe=resolved_detector_qe(params, fluorescence=fluorescence))


@dataclass(frozen=True)
class OpticalModeSettings:
    """Resolved optical backend and coherent-reference settings."""

    reference_field_amplitude: float
    optical_field_backend: str
    vectorial_detection_mode: str
    polarization_model: str
    vectorial_polarization_rotation_deg: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OpticalModeSettings":
        return cls(
            reference_field_amplitude=resolved_reference_field_amplitude(params),
            optical_field_backend=str(param_value(params, "optical_field_backend")).strip().lower(),
            vectorial_detection_mode=resolved_vectorial_detection_mode(params),
            polarization_model=str(param_value(params, "polarization_model")).strip().lower(),
            vectorial_polarization_rotation_deg=_finite_float(
                param_value(params, "vectorial_polarization_rotation_deg"),
                key="vectorial_polarization_rotation_deg",
            ),
        )

    @property
    def uses_full_vector_field(self) -> bool:
        return (
            self.optical_field_backend == "vectorial_debye"
            and self.vectorial_detection_mode == "full_vector"
        )


@dataclass(frozen=True)
class DpcSettings:
    """Resolved DPC-specific model settings."""

    channel_model: str
    transfer_model: str
    output_channel: str
    intensity_gain_x: float
    intensity_gain_y: float
    phase_gradient_gain_x: float
    phase_gradient_gain_y: float
    optical: OpticalModeSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "DpcSettings":
        def axis_gain(axis_key: str, generic_key: str) -> float:
            if axis_key in params:
                value = params[axis_key]
            elif generic_key in params:
                value = params[generic_key]
            else:
                value = param_value(params, axis_key)
            return _finite_float(value, key=axis_key)

        return cls(
            channel_model=str(param_value(params, "dpc_channel_model")).strip().lower(),
            transfer_model=str(param_value(params, "dpc_transfer_model")).strip().lower(),
            output_channel=str(param_value(params, "dpc_output_channel")).strip().lower(),
            intensity_gain_x=axis_gain("dpc_intensity_gain_x", "dpc_intensity_gain"),
            intensity_gain_y=axis_gain("dpc_intensity_gain_y", "dpc_intensity_gain"),
            phase_gradient_gain_x=axis_gain("dpc_phase_gradient_gain_x", "dpc_phase_gradient_gain"),
            phase_gradient_gain_y=axis_gain("dpc_phase_gradient_gain_y", "dpc_phase_gradient_gain"),
            optical=OpticalModeSettings.from_params(params),
        )


@dataclass(frozen=True)
class DarkFieldSettings:
    """Resolved coherent/annular dark-field scaling settings."""

    field_gain: float
    illumination_count: float
    background_count: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "DarkFieldSettings":
        field_gain = _finite_float(param_value(params, "dark_field_field_gain"), key="dark_field_field_gain", positive=True)
        counts = CountBudgetSettings.from_params(params)
        return cls(
            field_gain=field_gain,
            illumination_count=counts.dark_field_illumination_count,
            background_count=counts.dark_field_background_count,
        )


@dataclass(frozen=True)
class KohlerBrightFieldSettings:
    """Resolved partially-coherent bright-field Köhler settings."""

    coherence_factor: float
    source_samples: int
    optical: OpticalModeSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "KohlerBrightFieldSettings":
        source_samples = int(param_value(params, "kohler_source_samples"))
        if source_samples <= 0:
            raise ValueError(f"PARAMS['kohler_source_samples'] must be positive; got {source_samples!r}.")
        return cls(
            coherence_factor=_finite_float(param_value(params, "kohler_coherence_factor"), key="kohler_coherence_factor", nonnegative=True),
            source_samples=source_samples,
            optical=OpticalModeSettings.from_params(params),
        )


@dataclass(frozen=True)
class AnnularDarkFieldSettings:
    """Resolved annular Köhler dark-field settings."""

    inner_sigma: float
    outer_sigma: float
    source_samples: int
    dark_field: DarkFieldSettings
    optical: OpticalModeSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "AnnularDarkFieldSettings":
        source_samples = int(param_value(params, "annular_dark_field_source_samples"))
        if source_samples <= 0:
            raise ValueError(
                "PARAMS['annular_dark_field_source_samples'] must be positive; "
                f"got {source_samples!r}."
            )
        return cls(
            inner_sigma=_finite_float(
                param_value(params, "annular_dark_field_inner_sigma"),
                key="annular_dark_field_inner_sigma",
                positive=True,
            ),
            outer_sigma=_finite_float(
                param_value(params, "annular_dark_field_outer_sigma"),
                key="annular_dark_field_outer_sigma",
                positive=True,
            ),
            source_samples=source_samples,
            dark_field=DarkFieldSettings.from_params(params),
            optical=OpticalModeSettings.from_params(params),
        )


@dataclass(frozen=True)
class IscatSettings:
    """Resolved interferometric-scattering reference and collection settings."""

    reference_model: str
    reference_phase_rad: float
    reference_amplitude_scale: Any
    reference_normalize_fresnel_phase_only: bool
    reference_coefficient: Any
    reference_medium_material: str
    reference_substrate_material: str
    collection_model: str
    collection_reference_fraction: float
    reference_field_amplitude: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "IscatSettings":
        if "iscat_reference_medium_material" in params:
            reference_medium_material = params["iscat_reference_medium_material"]
        elif "medium_material" in params:
            reference_medium_material = params["medium_material"]
        else:
            reference_medium_material = param_value(params, "iscat_reference_medium_material")

        if "iscat_reference_substrate_material" in params:
            reference_substrate_material = params["iscat_reference_substrate_material"]
        elif "bulk_substrate_material" in params:
            reference_substrate_material = params["bulk_substrate_material"]
        else:
            reference_substrate_material = param_value(params, "iscat_reference_substrate_material")

        return cls(
            reference_model=str(param_value(params, "iscat_reference_model")).strip().lower(),
            reference_phase_rad=_finite_float(param_value(params, "iscat_reference_phase_rad"), key="iscat_reference_phase_rad"),
            reference_amplitude_scale=param_value(params, "iscat_reference_amplitude_scale"),
            reference_normalize_fresnel_phase_only=bool(param_value(params, "iscat_reference_normalize_fresnel_phase_only")),
            reference_coefficient=param_value(params, "iscat_reference_coefficient"),
            reference_medium_material=str(reference_medium_material),
            reference_substrate_material=str(reference_substrate_material),
            collection_model=str(param_value(params, "iscat_collection_model")).strip().lower(),
            collection_reference_fraction=_finite_float(
                param_value(params, "iscat_collection_reference_fraction"),
                key="iscat_collection_reference_fraction",
                positive=True,
            ),
            reference_field_amplitude=resolved_reference_field_amplitude(params),
        )


@dataclass(frozen=True)
class RicmSettings:
    """Resolved RICM reflection settings."""

    interface_reflection_model: str
    particle_reflection_model: str
    interface_medium_material: str
    interface_substrate_material: str
    thinfilm_layers: Any
    interface_reflection_coefficient: float
    particle_medium_material: str
    particle_reflection_coefficient: float
    interface_phase_shift_rad: float
    reference_field_amplitude: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "RicmSettings":
        return cls(
            interface_reflection_model=str(param_value(params, "ricm_interface_reflection_model")).strip().lower(),
            particle_reflection_model=str(param_value(params, "ricm_particle_reflection_model")).strip().lower(),
            interface_medium_material=str(param_value(params, "ricm_interface_medium_material")),
            interface_substrate_material=str(param_value(params, "ricm_interface_substrate_material")),
            thinfilm_layers=param_value(params, "ricm_thinfilm_layers"),
            interface_reflection_coefficient=_finite_float(
                param_value(params, "ricm_interface_reflection_coefficient"),
                key="ricm_interface_reflection_coefficient",
            ),
            particle_medium_material=str(param_value(params, "ricm_particle_medium_material")),
            particle_reflection_coefficient=_finite_float(
                param_value(params, "ricm_particle_reflection_coefficient"),
                key="ricm_particle_reflection_coefficient",
            ),
            interface_phase_shift_rad=_finite_float(
                param_value(params, "ricm_interface_phase_shift_rad"),
                key="ricm_interface_phase_shift_rad",
            ),
            reference_field_amplitude=resolved_reference_field_amplitude(params),
        )


@dataclass(frozen=True)
class VectorialOpticsSettings:
    """Resolved vectorial-Debye sampling and aberration settings."""

    sampling: SamplingGeometry
    optical: OpticalModeSettings
    obliquity_apodization: bool
    apodization_factor: float
    spherical_aberration_strength: float
    random_aberration_strength: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "VectorialOpticsSettings":
        return cls(
            sampling=SamplingGeometry.from_params(params),
            optical=OpticalModeSettings.from_params(params),
            obliquity_apodization=bool(param_value(params, "vectorial_obliquity_apodization")),
            apodization_factor=_finite_float(param_value(params, "apodization_factor"), key="apodization_factor", nonnegative=True),
            spherical_aberration_strength=_finite_float(
                param_value(params, "spherical_aberration_strength"),
                key="spherical_aberration_strength",
                nonnegative=True,
            ),
            random_aberration_strength=_finite_float(
                param_value(params, "random_aberration_strength"),
                key="random_aberration_strength",
                nonnegative=True,
            ),
        )


@dataclass(frozen=True)
class FluorescenceSettings:
    """Resolved fluorescence-model settings."""

    backend: str
    quantum_yield: float
    excitation_scale: float
    photons_per_fluorophore_per_frame: float | None
    collection_efficiency: float
    detector_qe: float
    photon_count_scale: float
    emission_psf_sigma_nm: float | None
    emission_psf_sigma_px: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "FluorescenceSettings":
        quantum_yield = _finite_float(
            param_value(params, "fluorescence_quantum_yield"),
            key="fluorescence_quantum_yield",
            nonnegative=True,
        )
        if quantum_yield > 1.0:
            raise ValueError(f"fluorescence_quantum_yield must be <= 1; got {quantum_yield!r}.")
        collection_efficiency = _finite_float(
            param_value(params, "fluorescence_collection_efficiency"),
            key="fluorescence_collection_efficiency",
            nonnegative=True,
        )
        if collection_efficiency > 1.0:
            raise ValueError(f"fluorescence_collection_efficiency must be <= 1; got {collection_efficiency!r}.")
        photons_raw = param_value(params, "fluorescence_photons_per_fluorophore_per_frame")
        photons = None if photons_raw is None else _finite_float(
            photons_raw,
            key="fluorescence_photons_per_fluorophore_per_frame",
            nonnegative=True,
        )
        sigma_nm_raw = param_value(params, "fluorescence_emission_psf_sigma_nm")
        sigma_nm = None if sigma_nm_raw is None else _finite_float(
            sigma_nm_raw,
            key="fluorescence_emission_psf_sigma_nm",
            nonnegative=True,
        )
        return cls(
            backend=str(param_value(params, "fluorescence_backend")).strip().lower(),
            quantum_yield=quantum_yield,
            excitation_scale=_finite_float(
                param_value(params, "fluorescence_excitation_scale"),
                key="fluorescence_excitation_scale",
                nonnegative=True,
            ),
            photons_per_fluorophore_per_frame=photons,
            collection_efficiency=collection_efficiency,
            detector_qe=DetectorSettings.from_params(params, fluorescence=True).detector_qe,
            photon_count_scale=resolved_fluorescence_photon_count_scale(params),
            emission_psf_sigma_nm=sigma_nm,
            emission_psf_sigma_px=_finite_float(
                param_value(params, "fluorescence_emission_psf_sigma_px"),
                key="fluorescence_emission_psf_sigma_px",
                nonnegative=True,
            ),
        )


@dataclass(frozen=True)
class SemSettings:
    """Resolved SEM source/probe/count settings."""

    source_z_origin: str
    source_z_offset_nm: float
    volume_slices: int
    probe_sigma_nm: float | None
    probe_sigma_px: float
    electrons_per_pixel: float
    sampling: SamplingGeometry

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SemSettings":
        volume_slices = int(param_value(params, "sem_volume_slices"))
        if volume_slices <= 0:
            raise ValueError(f"PARAMS['sem_volume_slices'] must be positive; got {volume_slices!r}.")
        sigma_nm_raw = param_value(params, "sem_probe_sigma_nm")
        sigma_nm = None if sigma_nm_raw is None else _finite_float(
            sigma_nm_raw,
            key="sem_probe_sigma_nm",
            nonnegative=True,
        )
        return cls(
            source_z_origin=str(param_value(params, "sem_source_z_origin")).strip().lower(),
            source_z_offset_nm=_finite_float(param_value(params, "sem_source_z_offset_nm"), key="sem_source_z_offset_nm"),
            volume_slices=volume_slices,
            probe_sigma_nm=sigma_nm,
            probe_sigma_px=_finite_float(param_value(params, "sem_probe_sigma_pixels"), key="sem_probe_sigma_pixels", nonnegative=True),
            electrons_per_pixel=resolved_sem_electrons_per_pixel(params),
            sampling=SamplingGeometry.from_params(params),
        )


@dataclass(frozen=True)
class TemSettings:
    """Resolved TEM count/sampling settings."""

    dose_per_pixel: float
    multislice_slices: int
    sampling: SamplingGeometry

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "TemSettings":
        multislice_slices = int(param_value(params, "tem_multislice_slices"))
        if multislice_slices <= 0:
            raise ValueError(
                "PARAMS['tem_multislice_slices'] must be positive; "
                f"got {multislice_slices!r}."
            )
        return cls(
            dose_per_pixel=resolved_tem_dose_per_pixel(params),
            multislice_slices=multislice_slices,
            sampling=SamplingGeometry.from_params(params),
        )


@dataclass(frozen=True)
class RenderRuntimeConfig:
    """Core per-render settings consumed by the frame renderer."""

    fps: float
    random_seed: int | None
    image_size_pixels: int
    pixel_size_nm: float
    psf_oversampling_factor: int
    exposure_time_ms: float | None
    mask_generation_enabled: bool
    mask_output_directory: str
    bit_depth: int
    motion_blur_enabled: bool
    motion_blur_subsamples: int
    return_ideal_float_frames: bool
    reference_field_amplitude: float
    background_intensity: float

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "RenderRuntimeConfig":
        fps = resolved_fps(params)
        random_seed = resolved_random_seed(params)
        image_size_pixels = resolved_image_size_pixels(params)
        pixel_size_nm = resolved_pixel_size_nm(params)
        psf_oversampling_factor = resolved_psf_oversampling_factor(params)
        exposure_raw = param_value(params, "exposure_time_ms")
        exposure_time_ms = None if exposure_raw is None else float(exposure_raw)
        bit_depth = int(param_value(params, "bit_depth"))
        if bit_depth <= 0:
            raise ValueError("PARAMS['bit_depth'] must be positive.")
        motion_blur_subsamples = int(param_value(params, "motion_blur_subsamples"))
        if motion_blur_subsamples <= 0:
            raise ValueError("PARAMS['motion_blur_subsamples'] must be positive.")
        reference_field_amplitude = resolved_reference_field_amplitude(params)
        background_intensity = resolved_background_intensity(params)

        return cls(
            fps=fps,
            random_seed=random_seed,
            image_size_pixels=image_size_pixels,
            pixel_size_nm=pixel_size_nm,
            psf_oversampling_factor=psf_oversampling_factor,
            exposure_time_ms=exposure_time_ms,
            mask_generation_enabled=bool(param_value(params, "mask_generation_enabled")),
            mask_output_directory=str(param_value(params, "mask_output_directory")),
            bit_depth=bit_depth,
            motion_blur_enabled=bool(param_value(params, "motion_blur_enabled")),
            motion_blur_subsamples=motion_blur_subsamples,
            return_ideal_float_frames=bool(param_value(params, "return_ideal_float_frames")),
            reference_field_amplitude=reference_field_amplitude,
            background_intensity=background_intensity,
        )


__all__ = [
    "AnnularDarkFieldSettings",
    "CountBudgetSettings",
    "DarkFieldSettings",
    "DetectorSettings",
    "DpcSettings",
    "FluorescenceSettings",
    "IscatSettings",
    "KohlerBrightFieldSettings",
    "OpticalModeSettings",
    "RicmSettings",
    "RenderRuntimeConfig",
    "SamplingGeometry",
    "SemSettings",
    "TemSettings",
    "VectorialOpticsSettings",
    "internal_param_value",
    "param_value",
    "resolved_background_intensity",
    "resolved_dark_field_background_count",
    "resolved_dark_field_illumination_count",
    "resolved_detector_qe",
    "resolved_fluorescence_photon_count_scale",
    "resolved_fps",
    "resolved_image_size_pixels",
    "resolved_model_canvas_pixel_size_nm",
    "resolved_model_canvas_shape",
    "resolved_modality",
    "resolved_particles",
    "resolved_pixel_size_nm",
    "resolved_psf_oversampling_factor",
    "resolved_qpi_phase_to_count_scale",
    "resolved_random_seed",
    "resolved_reference_field_amplitude",
    "resolved_rotational_diffusion_enabled",
    "resolved_sample_environment_pattern_dimension",
    "resolved_sample_environment_pattern_dimensions",
    "resolved_sem_electrons_per_pixel",
    "resolved_sem_monte_carlo_seed",
    "resolved_tem_dose_per_pixel",
    "resolved_temperature_K",
    "resolved_vectorial_detection_mode",
    "resolved_vectorial_pupil_samples",
    "resolved_viscosity_Pa_s",
]
