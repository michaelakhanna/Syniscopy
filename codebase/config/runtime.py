"""Typed runtime views over validated simulation parameter dictionaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import numbers
from typing import Any, Mapping

from configured_parameters import configured_optional as _configured_optional, configured_value as _configured_value
from electron_optics import (
    electron_interaction_parameter_rad_per_V_nm,
    electron_wavelength_m,
    scherzer_defocus_m,
)
from modality_registry import is_fluorescence_modality, require_modality_name
from param_schema import default_param_value
from sem_source_contracts import (
    SEMSourceRepresentationResolution,
    resolve_sem_source_representation,
)
from shared_constants import NUM_FRAME_DURATION_SEARCH_STEPS
from simulation_runtime_state import runtime_state_or_default


@dataclass(frozen=True)
class FocusPlaneState:
    """Run-local render focus plane in sample-world nanometres."""

    z_nm: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "FocusPlaneState":
        raw = runtime_state_or_default(params if isinstance(params, dict) else None).focus_plane_z_nm
        if raw is None:
            return cls(z_nm=0.0)
        out = float(raw)
        if not math.isfinite(out):
            raise ValueError(f"runtime focus_plane_z_nm must be finite; got {raw!r}.")
        return cls(z_nm=out)


def _finite_float(value: Any, *, key: str, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"parameters['{key}'] must be finite; got {value!r}.")
    if positive and out <= 0.0:
        raise ValueError(f"parameters['{key}'] must be positive; got {out!r}.")
    if nonnegative and out < 0.0:
        raise ValueError(f"parameters['{key}'] must be non-negative; got {out!r}.")
    return out


@dataclass(frozen=True)
class ModalitySettings:
    """Canonical imaging modality identity for one configured microscope."""

    modality: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "ModalitySettings":
        return cls(
            modality=require_modality_name(
                str(_configured_value(params, "imaging_model")).strip(),
                item_label="parameters['imaging_model']",
            )
        )

    def is_modality(self, name: str) -> bool:
        return self.modality == require_modality_name(
            str(name).strip(),
            item_label="modality",
        )

    @property
    def is_ricm(self) -> bool:
        return self.modality == "ricm"

    @property
    def is_tem_phase_contrast(self) -> bool:
        return self.modality == "tem_phase_contrast"

    @property
    def is_sem_secondary_electron(self) -> bool:
        return self.modality == "sem_secondary_electron"


@dataclass(frozen=True)
class AcquisitionProfile:
    """Resolved frame timing and exposure profile for one acquisition."""

    fps: float
    duration_seconds: float
    num_frames: int
    frame_interval_s: float
    exposure_time_ms: float | None
    exposure_time_s: float
    exposure_signal_scale: float
    random_seed: int | None

    @staticmethod
    def validate_requested_num_frames(raw_num_frames: Any) -> int:
        if isinstance(raw_num_frames, bool):
            raise ValueError("parameters['num_frames'] must be a positive integer, not bool.")
        if (
            isinstance(raw_num_frames, numbers.Real)
            and not float(raw_num_frames).is_integer()
        ):
            raise ValueError("parameters['num_frames'] must be an integer frame count.")
        try:
            num_frames = int(raw_num_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters['num_frames'] must be a positive integer.") from exc
        if num_frames <= 0:
            raise ValueError("parameters['num_frames'] must be positive.")
        return num_frames

    @staticmethod
    def duration_seconds_for_frame_count(fps: float, num_frames: int) -> float:
        fps_value = _finite_float(fps, key="fps", positive=True)
        frame_count = AcquisitionProfile.validate_requested_num_frames(num_frames)
        duration_seconds = float(frame_count) / fps_value
        for _ in range(NUM_FRAME_DURATION_SEARCH_STEPS):
            if int(fps_value * duration_seconds) == frame_count:
                return float(duration_seconds)
            duration_seconds = math.nextafter(duration_seconds, math.inf)
        raise RuntimeError(
            "Could not choose duration_seconds that reproduces "
            f"num_frames={frame_count} at fps={fps_value}."
        )

    @classmethod
    def requested_num_frames_from_params(cls, params: Mapping[str, Any]) -> int | None:
        raw_num_frames = _configured_optional(params, "num_frames")
        if raw_num_frames is None:
            return None
        return cls.validate_requested_num_frames(raw_num_frames)

    @staticmethod
    def declared_duration_seconds_from_params(params: Mapping[str, Any]) -> float | None:
        raw_duration = _configured_optional(params, "duration_seconds")
        if raw_duration is None:
            return None
        return _finite_float(raw_duration, key="duration_seconds", positive=True)

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "AcquisitionProfile":
        fps = _finite_float(_configured_optional(params, "fps"), key="fps", positive=True)
        raw_num_frames = _configured_optional(params, "num_frames")
        if raw_num_frames is None:
            duration_seconds = _finite_float(_configured_optional(params, "duration_seconds"), key="duration_seconds", positive=True)
            num_frames = int(fps * duration_seconds)
            if num_frames <= 0:
                raise ValueError(
                    "The product parameters['fps'] * parameters['duration_seconds'] must be "
                    "positive to generate at least one frame."
                )
        else:
            num_frames = cls.validate_requested_num_frames(raw_num_frames)
            duration_seconds = cls.duration_seconds_for_frame_count(fps, num_frames)

        frame_interval_s = 1.0 / fps
        exposure_raw = _configured_optional(params, "exposure_time_ms")
        if exposure_raw is None:
            exposure_time_ms = None
            exposure_time_s = frame_interval_s
        else:
            exposure_time_ms = _finite_float(exposure_raw, key="exposure_time_ms", positive=True)
            exposure_time_s = exposure_time_ms / 1000.0
            if exposure_time_s > frame_interval_s + 1.0e-12:
                raise ValueError(
                    "parameters['exposure_time_ms'] must satisfy exposure_time_ms <= 1000 / fps."
                )
        exposure_signal_scale = exposure_time_s * fps
        if not math.isfinite(exposure_signal_scale) or exposure_signal_scale <= 0.0:
            raise ValueError(
                "Acquisition exposure signal scale must be positive and finite; "
                f"got {exposure_signal_scale!r}."
            )
        random_seed = _configured_optional(params, "random_seed")
        random_seed_out = None if random_seed is None else int(random_seed)

        return cls(
            fps=float(fps),
            duration_seconds=float(duration_seconds),
            num_frames=int(num_frames),
            frame_interval_s=float(frame_interval_s),
            exposure_time_ms=exposure_time_ms,
            exposure_time_s=float(exposure_time_s),
            exposure_signal_scale=float(exposure_signal_scale),
            random_seed=random_seed_out,
        )


@dataclass(frozen=True)
class MotionDynamicsSettings:
    """Resolved Brownian motion and dynamic-estimator settings."""

    temperature_K: float
    viscosity_Pa_s: float
    drift_velocity_nm_per_s: tuple[float, float, float]
    vibration_jitter_std_nm: float
    vibration_include_axial: bool
    initial_z_span_nm: float
    z_motion_constraint_model: str
    rotational_diffusion_enabled: bool
    rotational_diffusion_mode: str
    rotational_step_std_deg: float
    dynamic_bayesian_enabled: bool
    sequence_fisher_enabled: bool
    dynamic_process_noise_scale: float
    dynamic_initial_variance_nm2: float
    dynamic_include_smoothing: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "MotionDynamicsSettings":
        drift_velocity = cls._resolve_drift_velocity(_configured_value(params, "drift_velocity_nm_per_s"))
        vibration_jitter_std_nm = _finite_float(
            _configured_value(params, "vibration_jitter_std_nm"),
            key="vibration_jitter_std_nm",
            nonnegative=True,
        )
        initial_z_span_nm = _finite_float(
            _configured_value(params, "initial_z_span_nm"),
            key="initial_z_span_nm",
            positive=True,
        )
        z_motion_constraint_model = str(
            _configured_value(params, "z_motion_constraint_model")
        ).strip().lower()
        valid_z_models = {"unconstrained", "reflecting_floor_z0", "reflecting_ceiling_z0"}
        if z_motion_constraint_model not in valid_z_models:
            raise ValueError(
                f"Unsupported z_motion_constraint_model {z_motion_constraint_model!r}. "
                "Supported values are 'unconstrained', "
                "'reflecting_floor_z0', and 'reflecting_ceiling_z0'."
            )
        rotational_step_std_deg = _finite_float(
            _configured_value(params, "rotational_step_std_deg"),
            key="rotational_step_std_deg",
            nonnegative=True,
        )
        dynamic_process_noise_scale = _finite_float(
            _configured_value(params, "dynamic_process_noise_scale"),
            key="dynamic_process_noise_scale",
            nonnegative=True,
        )
        dynamic_initial_variance_nm2 = _finite_float(
            _configured_value(params, "dynamic_initial_variance_nm2"),
            key="dynamic_initial_variance_nm2",
            positive=True,
        )
        return cls(
            temperature_K=_finite_float(
                _configured_value(params, "temperature_K"),
                key="temperature_K",
                positive=True,
            ),
            viscosity_Pa_s=_finite_float(
                _configured_value(params, "viscosity_Pa_s"),
                key="viscosity_Pa_s",
                positive=True,
            ),
            drift_velocity_nm_per_s=drift_velocity,
            vibration_jitter_std_nm=vibration_jitter_std_nm,
            vibration_include_axial=bool(_configured_value(params, "vibration_include_axial")),
            initial_z_span_nm=initial_z_span_nm,
            z_motion_constraint_model=z_motion_constraint_model,
            rotational_diffusion_enabled=bool(_configured_value(params, "rotational_diffusion_enabled")),
            rotational_diffusion_mode=str(_configured_value(params, "rotational_diffusion_mode")).strip().lower(),
            rotational_step_std_deg=rotational_step_std_deg,
            dynamic_bayesian_enabled=bool(_configured_value(params, "dynamic_bayesian_enabled")),
            sequence_fisher_enabled=bool(_configured_value(params, "sequence_fisher_enabled")),
            dynamic_process_noise_scale=dynamic_process_noise_scale,
            dynamic_initial_variance_nm2=dynamic_initial_variance_nm2,
            dynamic_include_smoothing=bool(_configured_value(params, "dynamic_include_smoothing")),
        )

    @staticmethod
    def _resolve_drift_velocity(raw: Any) -> tuple[float, float, float]:
        if isinstance(raw, numbers.Real) and not isinstance(raw, bool):
            values = (float(raw), 0.0, 0.0)
        else:
            try:
                values_tuple = tuple(float(value) for value in raw)
            except TypeError as exc:
                raise ValueError(
                    "parameters['drift_velocity_nm_per_s'] must be a scalar or length-3 sequence."
                ) from exc
            if len(values_tuple) == 1:
                values = (float(values_tuple[0]), 0.0, 0.0)
            elif len(values_tuple) == 3:
                values = values_tuple
            else:
                raise ValueError(
                    "parameters['drift_velocity_nm_per_s'] must be a scalar or length-3 sequence."
                )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "parameters['drift_velocity_nm_per_s'] must contain finite values; "
                f"got {raw!r}."
            )
        return values

    @property
    def axial_vibration_margin_nm(self) -> float:
        if not self.vibration_include_axial or self.vibration_jitter_std_nm <= 0.0:
            return 0.0
        return 4.0 * self.vibration_jitter_std_nm

    def axial_drift_extent_nm(self, duration_seconds: float) -> tuple[float, float]:
        drift_end_nm = self.drift_velocity_nm_per_s[2] * max(float(duration_seconds), 0.0)
        return min(0.0, drift_end_nm), max(0.0, drift_end_nm)


@dataclass(frozen=True)
class EmpiricalBackgroundSettings:
    """Resolved empirical low-frequency background field settings."""

    enabled: bool
    model: str
    relative_std: float
    scales_px: tuple[float, ...]
    scale_weights: tuple[float, ...]
    gradient_relative_strength: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "EmpiricalBackgroundSettings":
        scales_raw = _configured_value(params, "empirical_background_scales_px")
        weights_raw = _configured_value(params, "empirical_background_scale_weights")
        scales = tuple(float(v) for v in scales_raw)
        weights = tuple(float(v) for v in weights_raw)
        if len(scales) != len(weights):
            raise ValueError(
                "empirical_background_scales_px and empirical_background_scale_weights "
                "must have the same length."
            )
        return cls(
            enabled=bool(_configured_value(params, "empirical_background_enabled")),
            model=str(_configured_value(params, "empirical_background_model")).strip().lower(),
            relative_std=_finite_float(
                _configured_value(params, "empirical_background_relative_std"),
                key="empirical_background_relative_std",
                nonnegative=True,
            ),
            scales_px=scales,
            scale_weights=weights,
            gradient_relative_strength=_finite_float(
                _configured_value(params, "empirical_background_gradient_relative_strength"),
                key="empirical_background_gradient_relative_strength",
                nonnegative=True,
            ),
        )

    @property
    def active(self) -> bool:
        return self.enabled and self.model != "none"


@dataclass(frozen=True)
class SampleEnvironmentRoughnessSettings:
    """Resolved sample-environment roughness/speckle settings."""

    model: str
    source: Any
    source_basis: str
    source_coupling: str
    amplitude: float
    correlation_pixels: float
    phase_std: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SampleEnvironmentRoughnessSettings":
        return cls(
            model=str(_configured_value(params, "sample_environment_pattern_roughness_model")).strip().lower(),
            source=deepcopy(_configured_value(params, "sample_environment_pattern_roughness_source")),
            source_basis=str(
                _configured_value(params, "sample_environment_pattern_roughness_source_basis")
            ).strip().lower(),
            source_coupling=str(
                _configured_value(params, "sample_environment_pattern_roughness_source_coupling")
            ).strip().lower(),
            amplitude=_finite_float(
                _configured_value(params, "sample_environment_pattern_roughness_amplitude"),
                key="sample_environment_pattern_roughness_amplitude",
                nonnegative=True,
            ),
            correlation_pixels=_finite_float(
                _configured_value(params, "sample_environment_pattern_roughness_correlation_pixels"),
                key="sample_environment_pattern_roughness_correlation_pixels",
                nonnegative=True,
            ),
            phase_std=_finite_float(
                _configured_value(params, "sample_environment_pattern_roughness_phase_std"),
                key="sample_environment_pattern_roughness_phase_std",
                nonnegative=True,
            ),
        )

    @property
    def active(self) -> bool:
        return self.model != "none" and (
            self.model == "source_matched"
            or self.amplitude > 0.0
            or self.phase_std > 0.0
        )


@dataclass(frozen=True)
class SampleEnvironmentSettings:
    """Resolved sample medium, substrate-pattern, roughness, and shading settings."""

    enabled: bool
    pattern_enabled: bool
    pattern: str
    pattern_preset: str
    pattern_material: Any
    pattern_dimensions: dict[str, Any]
    pattern_randomization_enabled: bool
    pattern_position_jitter_std_nm: float
    pattern_shape_regularity: float
    pattern_edge_perturbation_max_rel_radius: float
    pattern_edge_perturbation_mode_count: int
    medium_material: str
    mounting_interface_material: str
    bulk_substrate_material: str
    mounting_interface_thickness_nm: float
    exclusion_method: str
    pattern_contrast_model: str
    pattern_contrast_amplitude: float
    bright_field_gain: float
    bright_field_phase_gain: float
    dark_field_edge_gain: float
    dark_field_scatter_pedestal: float
    sem_edge_gain: float
    tem_potential_scale: float
    fluorescence_excitation_modulation_gain: float
    fluorescence_autofluorescence_gain: float
    roughness: SampleEnvironmentRoughnessSettings
    empirical_background: EmpiricalBackgroundSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SampleEnvironmentSettings":
        raw_dimensions = _configured_value(params, "sample_environment_pattern_dimensions")
        if raw_dimensions is None:
            raw_dimensions = {}
        if not isinstance(raw_dimensions, Mapping):
            raise TypeError("parameters['sample_environment_pattern_dimensions'] must be a dictionary.")
        dimensions = default_param_value("sample_environment_pattern_dimensions")
        dimensions.update(dict(raw_dimensions))
        pattern_shape_regularity = _finite_float(
            _configured_value(params, "sample_environment_pattern_shape_regularity"),
            key="sample_environment_pattern_shape_regularity",
            nonnegative=True,
        )
        if pattern_shape_regularity > 1.0:
            raise ValueError(
                "parameters['sample_environment_pattern_shape_regularity'] must be <= 1.0; "
                f"got {pattern_shape_regularity!r}."
            )
        pattern_edge_perturbation_mode_count = int(
            _configured_value(params, "sample_environment_pattern_edge_perturbation_mode_count")
        )
        if pattern_edge_perturbation_mode_count < 0:
            raise ValueError(
                "parameters['sample_environment_pattern_edge_perturbation_mode_count'] "
                f"must be non-negative; got {pattern_edge_perturbation_mode_count!r}."
            )
        return cls(
            enabled=bool(_configured_value(params, "sample_environment_enabled")),
            pattern_enabled=bool(_configured_value(params, "sample_environment_pattern_enabled")),
            pattern=str(_configured_value(params, "sample_environment_pattern")).strip().lower(),
            pattern_preset=str(_configured_value(params, "sample_environment_pattern_preset")).strip().lower(),
            pattern_material=deepcopy(_configured_value(params, "sample_environment_pattern_material")),
            pattern_dimensions=dimensions,
            pattern_randomization_enabled=bool(
                _configured_value(params, "sample_environment_pattern_randomization_enabled")
            ),
            pattern_position_jitter_std_nm=_finite_float(
                _configured_value(params, "sample_environment_pattern_position_jitter_std_nm"),
                key="sample_environment_pattern_position_jitter_std_nm",
                nonnegative=True,
            ),
            pattern_shape_regularity=pattern_shape_regularity,
            pattern_edge_perturbation_max_rel_radius=_finite_float(
                _configured_value(params, "sample_environment_pattern_edge_perturbation_max_rel_radius"),
                key="sample_environment_pattern_edge_perturbation_max_rel_radius",
                nonnegative=True,
            ),
            pattern_edge_perturbation_mode_count=pattern_edge_perturbation_mode_count,
            medium_material=str(_configured_value(params, "medium_material")),
            mounting_interface_material=str(_configured_value(params, "mounting_interface_material")),
            bulk_substrate_material=str(_configured_value(params, "bulk_substrate_material")),
            mounting_interface_thickness_nm=_finite_float(
                _configured_value(params, "mounting_interface_thickness_nm"),
                key="mounting_interface_thickness_nm",
                nonnegative=True,
            ),
            exclusion_method=str(_configured_value(params, "sample_environment_exclusion_method")).strip().lower(),
            pattern_contrast_model=str(
                _configured_value(params, "sample_environment_pattern_contrast_model")
            ).strip().lower(),
            pattern_contrast_amplitude=_finite_float(
                _configured_value(params, "sample_environment_pattern_contrast_amplitude"),
                key="sample_environment_pattern_contrast_amplitude",
                nonnegative=True,
            ),
            bright_field_gain=_finite_float(
                _configured_value(params, "bright_field_sample_environment_gain"),
                key="bright_field_sample_environment_gain",
                nonnegative=True,
            ),
            bright_field_phase_gain=_finite_float(
                _configured_value(params, "bright_field_sample_environment_phase_gain"),
                key="bright_field_sample_environment_phase_gain",
                nonnegative=True,
            ),
            dark_field_edge_gain=_finite_float(
                _configured_value(params, "dark_field_sample_environment_edge_gain"),
                key="dark_field_sample_environment_edge_gain",
                nonnegative=True,
            ),
            dark_field_scatter_pedestal=_finite_float(
                _configured_value(params, "dark_field_sample_environment_scatter_pedestal"),
                key="dark_field_sample_environment_scatter_pedestal",
                nonnegative=True,
            ),
            sem_edge_gain=_finite_float(
                _configured_value(params, "sem_sample_environment_edge_gain"),
                key="sem_sample_environment_edge_gain",
                nonnegative=True,
            ),
            tem_potential_scale=_finite_float(
                _configured_value(params, "tem_sample_environment_potential_scale"),
                key="tem_sample_environment_potential_scale",
                nonnegative=True,
            ),
            fluorescence_excitation_modulation_gain=_finite_float(
                _configured_value(params, "fluorescence_sample_environment_excitation_modulation_gain"),
                key="fluorescence_sample_environment_excitation_modulation_gain",
                nonnegative=True,
            ),
            fluorescence_autofluorescence_gain=_finite_float(
                _configured_value(params, "fluorescence_sample_environment_autofluorescence_gain"),
                key="fluorescence_sample_environment_autofluorescence_gain",
                nonnegative=True,
            ),
            roughness=SampleEnvironmentRoughnessSettings.from_params(params),
            empirical_background=EmpiricalBackgroundSettings.from_params(params),
        )

    @property
    def pattern_active(self) -> bool:
        return (
            self.enabled
            and self.pattern_enabled
            and self.pattern_preset != "empty_background"
            and self.pattern != "none"
        )

    def pattern_active_for_model(self, model: object | None = None) -> bool:
        active = self.pattern_active
        if model is not None:
            active = active and bool(getattr(model, "uses_sample_environment_pattern", False))
        return bool(active)

    def dimension(self, key: str) -> Any:
        if key not in self.pattern_dimensions:
            raise KeyError(f"Unknown sample_environment_pattern_dimensions key: {key!r}.")
        return self.pattern_dimensions[key]

    def packet_metadata(self, *, optical_instrument: Any | None = None) -> dict[str, Any]:
        """Return schema-backed sample-environment metadata for matched packets."""

        out: dict[str, Any] = {
            "sample_environment_enabled": self.enabled,
            "sample_environment_pattern_enabled": self.pattern_enabled,
            "sample_environment_pattern": self.pattern,
            "sample_environment_pattern_preset": self.pattern_preset,
            "sample_environment_pattern_dimensions": deepcopy(self.pattern_dimensions),
            "sample_environment_pattern_material": deepcopy(self.pattern_material),
            "sample_environment_pattern_roughness_model": self.roughness.model,
            "sample_environment_pattern_roughness_source_coupling": self.roughness.source_coupling,
            "sample_environment_pattern_roughness_amplitude": self.roughness.amplitude,
            "sample_environment_pattern_roughness_phase_std": self.roughness.phase_std,
            "medium_material": self.medium_material,
            "mounting_interface_material": self.mounting_interface_material,
            "bulk_substrate_material": self.bulk_substrate_material,
            "mounting_interface_thickness_nm": self.mounting_interface_thickness_nm,
        }
        if optical_instrument is not None:
            out.update(
                {
                    "refractive_index_medium": optical_instrument.refractive_index_medium,
                    "refractive_index_immersion": optical_instrument.refractive_index_immersion,
                }
            )
        return out


@dataclass(frozen=True)
class SamplingGeometry:
    """Resolved detector/model-canvas sampling parameters."""

    image_size_pixels: int
    detector_pixel_size_nm: float
    psf_oversampling_factor: int

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SamplingGeometry":
        image_size_pixels = int(_configured_optional(params, "image_size_pixels"))
        if image_size_pixels <= 0:
            raise ValueError(f"parameters['image_size_pixels'] must be positive; got {image_size_pixels!r}.")
        detector_pixel_size_nm = _finite_float(_configured_optional(params, "pixel_size_nm"), key="pixel_size_nm", positive=True)
        psf_oversampling_factor = int(_configured_optional(params, "psf_oversampling_factor"))
        if psf_oversampling_factor <= 0:
            raise ValueError(
                f"parameters['psf_oversampling_factor'] must be positive; got {psf_oversampling_factor!r}."
            )
        return cls(
            image_size_pixels=image_size_pixels,
            detector_pixel_size_nm=detector_pixel_size_nm,
            psf_oversampling_factor=psf_oversampling_factor,
        )

    @property
    def model_canvas_shape(self) -> tuple[int, int]:
        side = self.image_size_pixels * self.psf_oversampling_factor
        return side, side

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self.image_size_pixels, self.image_size_pixels

    @property
    def detector_fov_size_nm(self) -> float:
        return float(self.image_size_pixels) * self.detector_pixel_size_nm

    @property
    def model_canvas_pixel_size_nm(self) -> float:
        return self.detector_pixel_size_nm / float(self.psf_oversampling_factor)


SUPERVISION_TARGETS = ("mask_supported", "mask_geometry")
SUPERVISION_SUPPORT_FACTORS = ("temporal", "signal", "information", "ambiguity")


@dataclass(frozen=True)
class SupervisionGeometryThresholds:
    """Pixel-geometry-derived supervision thresholds."""

    crlb_xy_max_nm: float
    ambiguity_distance_scale_nm: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SupervisionGeometryThresholds":
        sampling = SamplingGeometry.from_params(params)
        crlb_xy_max = _configured_value(params, "supervision_crlb_xy_max_nm")
        ambiguity_scale = _configured_value(params, "supervision_ambiguity_distance_scale_nm")
        crlb_xy_max_nm = (
            sampling.detector_pixel_size_nm
            if crlb_xy_max is None
            else _finite_float(crlb_xy_max, key="supervision_crlb_xy_max_nm", nonnegative=True)
        )
        ambiguity_distance_scale_nm = (
            2.0 * sampling.detector_pixel_size_nm
            if ambiguity_scale is None
            else _finite_float(
                ambiguity_scale,
                key="supervision_ambiguity_distance_scale_nm",
                nonnegative=True,
            )
        )
        return cls(
            crlb_xy_max_nm=float(crlb_xy_max_nm),
            ambiguity_distance_scale_nm=float(ambiguity_distance_scale_nm),
        )


@dataclass(frozen=True)
class SupervisionSettings:
    """Resolved supervision-policy configuration."""

    target: str
    support_factors: tuple[str, ...]
    supported_threshold: float
    temporal_support_enabled: bool
    signal_support_enabled: bool
    information_support_enabled: bool
    ambiguity_support_enabled: bool
    stop_when_all_temporally_unsupported: bool
    prior_log_odds: float
    decision_rule: str
    log_odds_threshold: float
    log_odds_clip_epsilon: float
    score_calibration_mode: str
    score_calibration_parameters: dict[str, Any]
    geometry_thresholds: SupervisionGeometryThresholds

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SupervisionSettings":
        target = str(_configured_value(params, "supervision_target")).strip()
        if target not in SUPERVISION_TARGETS:
            raise ValueError(
                "parameters['supervision_target'] must be one of "
                f"{SUPERVISION_TARGETS}; got {target!r}."
            )
        enabled_by_factor = {
            "temporal": bool(_configured_value(params, "supervision_temporal_support_enabled")),
            "signal": bool(_configured_value(params, "supervision_signal_support_enabled")),
            "information": bool(_configured_value(params, "supervision_information_support_enabled")),
            "ambiguity": bool(_configured_value(params, "supervision_ambiguity_support_enabled")),
        }
        explicit_factors = _configured_value(params, "supervision_support_factors")
        if explicit_factors is None:
            if target == "mask_geometry":
                factors: tuple[str, ...] = ()
            else:
                factors = tuple(
                    factor
                    for factor in SUPERVISION_SUPPORT_FACTORS
                    if enabled_by_factor[factor]
                )
        elif isinstance(explicit_factors, str):
            factors = tuple(
                factor.strip()
                for factor in explicit_factors.split(",")
                if factor.strip()
            )
        else:
            factors = tuple(str(factor).strip() for factor in explicit_factors)

        invalid = [factor for factor in factors if factor not in SUPERVISION_SUPPORT_FACTORS]
        if invalid:
            raise ValueError(
                "parameters['supervision_support_factors'] contains unsupported "
                f"factor(s) {invalid}; supported factors are {SUPERVISION_SUPPORT_FACTORS}."
            )
        duplicates = []
        seen = set()
        for factor in factors:
            if factor in seen and factor not in duplicates:
                duplicates.append(factor)
            seen.add(factor)
        if duplicates:
            raise ValueError(
                "parameters['supervision_support_factors'] contains duplicate factor(s) "
                f"{duplicates}. Each support factor may be listed at most once."
            )
        if explicit_factors is not None:
            disabled = [
                factor for factor in factors
                if not enabled_by_factor[factor]
            ]
            if disabled:
                raise ValueError(
                    "parameters['supervision_support_factors'] explicitly includes disabled "
                    f"factor(s) {disabled}. Remove the factor(s) or enable their "
                    "corresponding supervision_*_support_enabled flag."
                )

        supported_threshold = _finite_float(
            _configured_value(params, "supervision_supported_threshold"),
            key="supervision_supported_threshold",
            nonnegative=True,
        )
        if supported_threshold > 1.0:
            raise ValueError(
                "parameters['supervision_supported_threshold'] must be in [0, 1]; "
                f"got {supported_threshold!r}."
            )
        decision_rule = str(_configured_value(params, "supervision_decision_rule")).strip().lower()
        if decision_rule not in {"log_odds", "product"}:
            raise ValueError("parameters['supervision_decision_rule'] must be 'log_odds' or 'product'.")
        log_odds_clip_epsilon = _finite_float(
            _configured_value(params, "supervision_log_odds_clip_epsilon"),
            key="supervision_log_odds_clip_epsilon",
            positive=True,
        )
        if log_odds_clip_epsilon >= 0.5:
            raise ValueError(
                "parameters['supervision_log_odds_clip_epsilon'] must lie in (0, 0.5); "
                f"got {log_odds_clip_epsilon!r}."
            )
        calibration_mode = str(
            _configured_value(params, "supervision_score_calibration_mode")
        ).strip().lower()
        if calibration_mode not in {"uncalibrated_support", "platt_logistic", "isotonic"}:
            raise ValueError(
                "parameters['supervision_score_calibration_mode'] must be "
                "'uncalibrated_support', 'platt_logistic', or 'isotonic'."
            )
        calibration_parameters = _configured_value(params, "supervision_score_calibration_parameters")
        if calibration_parameters is None:
            calibration_parameters_out: dict[str, Any] = {}
        elif isinstance(calibration_parameters, Mapping):
            calibration_parameters_out = deepcopy(dict(calibration_parameters))
        else:
            raise ValueError("parameters['supervision_score_calibration_parameters'] must be None or a dict.")

        return cls(
            target=target,
            support_factors=factors,
            supported_threshold=supported_threshold,
            temporal_support_enabled=enabled_by_factor["temporal"],
            signal_support_enabled=enabled_by_factor["signal"],
            information_support_enabled=enabled_by_factor["information"],
            ambiguity_support_enabled=enabled_by_factor["ambiguity"],
            stop_when_all_temporally_unsupported=bool(
                _configured_value(params, "supervision_stop_when_all_temporally_unsupported")
            ),
            prior_log_odds=_finite_float(
                _configured_value(params, "supervision_prior_log_odds"),
                key="supervision_prior_log_odds",
            ),
            decision_rule=decision_rule,
            log_odds_threshold=_finite_float(
                _configured_value(params, "supervision_log_odds_threshold"),
                key="supervision_log_odds_threshold",
            ),
            log_odds_clip_epsilon=log_odds_clip_epsilon,
            score_calibration_mode=calibration_mode,
            score_calibration_parameters=calibration_parameters_out,
            geometry_thresholds=SupervisionGeometryThresholds.from_params(params),
        )


@dataclass(frozen=True)
class MaskGenerationSettings:
    """Resolved mask-generation and mask-output settings."""

    enabled: bool
    output_directory: str
    outer_ring_count: int
    max_area_fraction: float
    exact_leave_one_out_max_work_units: int
    exact_leave_one_out_allow_expensive: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "MaskGenerationSettings":
        outer_ring_count = int(_configured_value(params, "mask_outer_ring_count"))
        if outer_ring_count < 0:
            raise ValueError("parameters['mask_outer_ring_count'] must be non-negative.")
        max_area_fraction = _finite_float(
            _configured_value(params, "mask_max_area_fraction"),
            key="mask_max_area_fraction",
            nonnegative=True,
        )
        if max_area_fraction > 1.0:
            raise ValueError(
                "parameters['mask_max_area_fraction'] must be <= 1.0; "
                f"got {max_area_fraction!r}."
            )
        max_work_units_raw = _configured_value(params, "mask_exact_leave_one_out_max_work_units")
        if isinstance(max_work_units_raw, bool):
            raise ValueError(
                "parameters['mask_exact_leave_one_out_max_work_units'] must be a positive integer."
            )
        max_work_units = int(max_work_units_raw)
        if max_work_units <= 0:
            raise ValueError(
                "parameters['mask_exact_leave_one_out_max_work_units'] must be positive."
            )
        return cls(
            enabled=bool(_configured_value(params, "mask_generation_enabled")),
            output_directory=str(_configured_value(params, "mask_output_directory")),
            outer_ring_count=outer_ring_count,
            max_area_fraction=max_area_fraction,
            exact_leave_one_out_max_work_units=max_work_units,
            exact_leave_one_out_allow_expensive=bool(
                _configured_value(params, "mask_exact_leave_one_out_allow_expensive")
            ),
        )


@dataclass(frozen=True)
class SimulationOutputSettings:
    """Resolved simulation artifact-writing policy."""

    output_filename: str
    save_frame_sequence: bool
    save_raw_camera_video: bool
    save_raw_camera_frame_sequence: bool
    save_raw_frame_views: bool
    return_ideal_float_frames: bool
    multichannel_output_mode: str
    multichannel_sidecar_directory: str | None
    mask_generation: MaskGenerationSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SimulationOutputSettings":
        output_filename = str(_configured_value(params, "output_filename"))
        if not output_filename:
            raise ValueError("parameters['output_filename'] must be a non-empty path.")

        raw_mode = str(_configured_value(params, "multichannel_output_mode")).strip().lower()
        mode = raw_mode.replace("-", "_").replace(" ", "_")
        allowed_modes = {"rgb", "channels", "both", "none"}
        if mode not in allowed_modes:
            raise ValueError(
                "parameters['multichannel_output_mode'] must be one of "
                "{'rgb', 'channels', 'both', 'none'}; got "
                f"{raw_mode!r}."
            )

        sidecar_raw = _configured_value(params, "multichannel_sidecar_directory")
        sidecar_directory = None if sidecar_raw in (None, "") else str(sidecar_raw)

        return cls(
            output_filename=output_filename,
            save_frame_sequence=bool(_configured_value(params, "save_frame_sequence")),
            save_raw_camera_video=bool(_configured_value(params, "save_raw_camera_video")),
            save_raw_camera_frame_sequence=bool(
                _configured_value(params, "save_raw_camera_frame_sequence")
            ),
            save_raw_frame_views=bool(_configured_value(params, "save_raw_frame_views")),
            return_ideal_float_frames=bool(_configured_value(params, "return_ideal_float_frames")),
            multichannel_output_mode=mode,
            multichannel_sidecar_directory=sidecar_directory,
            mask_generation=MaskGenerationSettings.from_params(params),
        )

    @property
    def mask_generation_enabled(self) -> bool:
        return self.mask_generation.enabled

    @property
    def mask_output_directory(self) -> str:
        return self.mask_generation.output_directory

    @property
    def raw_signal_video_filename(self) -> str:
        return self.sibling_output_filename("_raw_signal")

    def sibling_output_filename(self, suffix: str) -> str:
        slash_index = max(self.output_filename.rfind("/"), self.output_filename.rfind("\\"))
        dot_index = self.output_filename.rfind(".")
        if dot_index <= slash_index:
            raise ValueError(
                "parameters['output_filename'] must include a filename extension; "
                f"got {self.output_filename!r}."
            )
        return f"{self.output_filename[:dot_index]}{suffix}{self.output_filename[dot_index:]}"

    @property
    def multichannel_sidecar_output_directory(self) -> str:
        if self.multichannel_sidecar_directory:
            return self.multichannel_sidecar_directory
        slash_index = max(self.output_filename.rfind("/"), self.output_filename.rfind("\\"))
        dot_index = self.output_filename.rfind(".")
        if dot_index <= slash_index:
            stem = self.output_filename
        else:
            stem = self.output_filename[:dot_index]
        return f"{stem}_channels"

    def require_dataset_frame_artifacts(self) -> None:
        if not self.save_frame_sequence:
            raise ValueError(
                "Dataset generation requires save_frame_sequence=True. "
                "PNG frame sequences are the canonical 8-bit contrast-analysis "
                "training/inference artifact. Enable save_raw_camera_video for a "
                "raw-camera preview and save_raw_frame_views for quantitative "
                "raw/ideal arrays."
            )

    def require_dataset_primary_multichannel_video(self, *, channels_enabled: bool) -> None:
        if channels_enabled and self.multichannel_output_mode not in {"rgb", "both"}:
            raise ValueError(
                "Dataset generation requires multichannel_output_mode='rgb' "
                "or 'both' when channels are enabled, because the dataset "
                "manifest needs a primary training video. Use the low-level "
                "run_simulation path for sidecar-only or no-video spectral renders."
            )


@dataclass(frozen=True)
class MatchedMicroscopeSettings:
    """Resolved matched-microscope packet request."""

    microscopes: Any

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "MatchedMicroscopeSettings":
        return cls(microscopes=deepcopy(_configured_value(params, "matched_microscopes")))

    @property
    def enabled(self) -> bool:
        return self.microscopes is not None


@dataclass(frozen=True)
class UnitContractSettings:
    """Resolved unit-contract validation toggle."""

    enabled: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "UnitContractSettings":
        return cls(enabled=bool(_configured_value(params, "unit_contracts_enabled")))


@dataclass(frozen=True)
class CountBudgetSettings:
    """Resolved count/dose scaling parameters shared by detector-domain models."""

    background_intensity: float
    dark_field_background_count: float
    dark_field_illumination_count: float
    sem_electrons_per_pixel: float
    tem_dose_per_pixel: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "CountBudgetSettings":
        return cls(
            background_intensity=_finite_float(
                _configured_value(params, "background_intensity"),
                key="background_intensity",
                nonnegative=True,
            ),
            dark_field_background_count=_finite_float(
                _configured_value(params, "dark_field_background_count"),
                key="dark_field_background_count",
                nonnegative=True,
            ),
            dark_field_illumination_count=_finite_float(
                _configured_value(params, "dark_field_illumination_count"),
                key="dark_field_illumination_count",
                nonnegative=True,
            ),
            sem_electrons_per_pixel=_finite_float(
                _configured_value(params, "sem_electrons_per_pixel"),
                key="sem_electrons_per_pixel",
                nonnegative=True,
            ),
            tem_dose_per_pixel=_finite_float(
                _configured_value(params, "tem_dose_per_pixel"),
                key="tem_dose_per_pixel",
                nonnegative=True,
            ),
        )


@dataclass(frozen=True)
class QpiReadoutSettings:
    """Resolved QPI phase-readout budget and display scaling settings."""

    visibility: float
    background_intensity: float
    configured_detected_quanta_per_pixel: float
    detected_quanta_uses_background_reference: bool
    phase_to_count_scale: float
    phase_noise_std_rad: float | None
    exposure_signal_scale: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "QpiReadoutSettings":
        visibility = _finite_float(
            _configured_value(params, "qpi_visibility"),
            key="qpi_visibility",
            positive=True,
        )
        if visibility > 1.0:
            raise ValueError(f"parameters['qpi_visibility'] must be <= 1.0; got {visibility!r}.")

        background_intensity = CountBudgetSettings.from_params(params).background_intensity
        detected_quanta_raw = _configured_value(params, "qpi_detected_quanta_per_pixel")
        uses_background = detected_quanta_raw is None
        detected_quanta_source = background_intensity if uses_background else detected_quanta_raw
        configured_detected_quanta = _finite_float(
            detected_quanta_source,
            key="qpi_detected_quanta_per_pixel/background_intensity",
            positive=True,
        )

        phase_noise_raw = _configured_value(params, "qpi_phase_noise_std_rad")
        phase_noise_std_rad = (
            None
            if phase_noise_raw is None
            else _finite_float(
                phase_noise_raw,
                key="qpi_phase_noise_std_rad",
                nonnegative=True,
            )
        )

        return cls(
            visibility=visibility,
            background_intensity=background_intensity,
            configured_detected_quanta_per_pixel=configured_detected_quanta,
            detected_quanta_uses_background_reference=uses_background,
            phase_to_count_scale=_finite_float(
                _configured_value(params, "qpi_phase_to_count_scale"),
                key="qpi_phase_to_count_scale",
                positive=True,
            ),
            phase_noise_std_rad=phase_noise_std_rad,
            exposure_signal_scale=(
                AcquisitionProfile.from_params(params).exposure_signal_scale
            ),
        )

    @property
    def detected_quanta_per_pixel(self) -> float:
        return self.configured_detected_quanta_per_pixel * self.exposure_signal_scale

    @property
    def reference_background_quanta_scale(self) -> float:
        if self.detected_quanta_uses_background_reference:
            return 1.0
        if self.background_intensity <= 0.0:
            raise ValueError(
                "background_intensity must be positive when scaling a per-frame "
                "QPI detected-quanta map from qpi_detected_quanta_per_pixel."
            )
        return self.configured_detected_quanta_per_pixel / self.background_intensity

    @property
    def reference_background_quanta_provenance(self) -> str:
        if self.detected_quanta_uses_background_reference:
            return "reference_background_map"
        return "reference_background_map_scaled_from_qpi_detected_quanta_per_pixel"

    def shot_variance_rad2(self, *, shot_noise_enabled: bool) -> float:
        if not bool(shot_noise_enabled):
            return 0.0
        return 1.0 / (
            self.visibility
            * self.visibility
            * self.detected_quanta_per_pixel
        )

    def readout_variance_rad2(self, *, gaussian_noise_enabled: bool) -> float:
        if self.phase_noise_std_rad is None or not bool(gaussian_noise_enabled):
            return 0.0
        return self.phase_noise_std_rad * self.phase_noise_std_rad


@dataclass(frozen=True)
class DetectorSettings:
    """Resolved detector efficiency settings."""

    detector_qe: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, fluorescence: bool = False) -> "DetectorSettings":
        if fluorescence:
            fluorescence_value = _configured_value(params, "fluorescence_detector_qe")
            value = (
                fluorescence_value
                if fluorescence_value is not None
                else _configured_value(params, "detector_qe")
            )
        else:
            value = _configured_value(params, "detector_qe")
        qe = _finite_float(value, key="detector_qe", nonnegative=True)
        if qe > 1.0:
            raise ValueError(f"detector_qe must be <= 1; got {qe!r}.")
        return cls(detector_qe=qe)


@dataclass(frozen=True)
class DetectorReadoutSettings:
    """Resolved detector output quantization and preview range."""

    bit_depth: int
    saturation_level: float | None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "DetectorReadoutSettings":
        bit_depth = int(_configured_value(params, "bit_depth"))
        if bit_depth < 1 or bit_depth > 16:
            raise ValueError(f"parameters['bit_depth'] must be in [1, 16]; got {bit_depth!r}.")
        saturation_raw = _configured_value(params, "saturation_level")
        saturation_level = None
        if saturation_raw is not None:
            saturation_level = _finite_float(
                saturation_raw,
                key="saturation_level",
                positive=True,
            )
        return cls(bit_depth=bit_depth, saturation_level=saturation_level)

    @property
    def max_camera_count(self) -> float:
        if self.saturation_level is not None:
            return float(self.saturation_level)
        return float((1 << self.bit_depth) - 1)


@dataclass(frozen=True)
class BackgroundSubtractionSettings:
    """Resolved contrast background-subtraction policy."""

    method: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "BackgroundSubtractionSettings":
        method = str(_configured_value(params, "background_subtraction_method")).strip().lower()
        if not method:
            raise ValueError("parameters['background_subtraction_method'] must be non-empty.")
        return cls(method=method)


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
            reference_field_amplitude=_finite_float(
                _configured_value(params, "reference_field_amplitude"),
                key="reference_field_amplitude",
                nonnegative=True,
            ),
            optical_field_backend=str(_configured_value(params, "optical_field_backend")).strip().lower(),
            vectorial_detection_mode=str(_configured_value(params, "vectorial_detection_mode")).strip().lower(),
            polarization_model=str(_configured_value(params, "polarization_model")).strip().lower(),
            vectorial_polarization_rotation_deg=_finite_float(
                _configured_value(params, "vectorial_polarization_rotation_deg"),
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
class OpticalScatteringSettings:
    """Resolved geometry-to-scattered-field model selector."""

    model: str
    cluster_model: str
    dda_voxel_size_nm: float | None
    dda_max_dipoles: int

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OpticalScatteringSettings":
        model = str(_configured_value(params, "optical_scattering_model")).strip().lower()
        valid = {"auto", "mie", "analytic_polarizability", "born_rayleigh_gans"}
        if model not in valid:
            raise ValueError(
                "optical_scattering_model must be one of "
                f"{sorted(valid)}; got {model!r}."
            )
        cluster_model = str(_configured_value(params, "optical_cluster_scattering_model")).strip().lower()
        valid_cluster = {
            "auto",
            "independent",
            "coupled_dipole",
            "discrete_dipole_dda",
            "multi_sphere_t_matrix",
        }
        if cluster_model not in valid_cluster:
            raise ValueError(
                "optical_cluster_scattering_model must be one of "
                f"{sorted(valid_cluster)}; got {cluster_model!r}."
            )
        dda_voxel = _configured_value(params, "optical_cluster_dda_voxel_size_nm")
        dda_voxel_size_nm = (
            None
            if dda_voxel is None
            else _finite_float(dda_voxel, key="optical_cluster_dda_voxel_size_nm", positive=True)
        )
        dda_max_dipoles = int(_configured_value(params, "optical_cluster_dda_max_dipoles"))
        if dda_max_dipoles <= 0:
            raise ValueError("optical_cluster_dda_max_dipoles must be positive.")
        return cls(
            model=model,
            cluster_model=cluster_model,
            dda_voxel_size_nm=dda_voxel_size_nm,
            dda_max_dipoles=dda_max_dipoles,
        )


@dataclass(frozen=True)
class OpticalPsfGridSettings:
    """Resolved optical PSF z-stack planning policy."""

    z_stack_range_nm: float
    z_stack_step_nm: float
    min_half_span_steps: float
    max_z_slices: int | None
    shared_z_grid_enabled: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OpticalPsfGridSettings":
        max_slices_raw = _configured_value(params, "max_psf_z_slices")
        max_slices = None if max_slices_raw is None else int(max_slices_raw)
        if max_slices is not None and max_slices <= 0:
            raise ValueError("parameters['max_psf_z_slices'] must be positive or None.")
        return cls(
            z_stack_range_nm=_finite_float(
                _configured_value(params, "z_stack_range_nm"),
                key="z_stack_range_nm",
                positive=True,
            ),
            z_stack_step_nm=_finite_float(
                _configured_value(params, "z_stack_step_nm"),
                key="z_stack_step_nm",
                positive=True,
            ),
            min_half_span_steps=_finite_float(
                _configured_value(params, "psf_z_grid_min_half_span_steps"),
                key="psf_z_grid_min_half_span_steps",
                nonnegative=True,
            ),
            max_z_slices=max_slices,
            shared_z_grid_enabled=bool(_configured_value(params, "shared_psf_z_grid_enabled")),
        )

    @property
    def default_half_span_nm(self) -> float:
        return 0.5 * self.z_stack_range_nm


@dataclass(frozen=True)
class OpticalPsfSupportSettings:
    """Resolved optical PSF spatial-support guard policy."""

    intensity_fraction_threshold: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OpticalPsfSupportSettings":
        threshold = _finite_float(
            _configured_value(params, "psf_intensity_fraction_threshold"),
            key="psf_intensity_fraction_threshold",
            positive=True,
        )
        if threshold >= 1.0:
            raise ValueError(
                "parameters['psf_intensity_fraction_threshold'] must be in the open interval (0, 1); "
                f"got {threshold!r}."
            )
        return cls(intensity_fraction_threshold=threshold)


@dataclass(frozen=True)
class FisherAnalysisSettings:
    """Resolved Fisher diagnostic controls for renderer/report boundaries."""

    detected_quanta_derivative_target: str
    likelihood_model: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "FisherAnalysisSettings":
        detected_target = str(
            _configured_value(params, "detected_quanta_derivative_target")
        ).strip().lower()
        valid_detected_targets = {"signed_contrast_scaled", "count_mean_derivative"}
        if detected_target not in valid_detected_targets:
            raise ValueError(
                "parameters['detected_quanta_derivative_target'] must be one of "
                f"{sorted(valid_detected_targets)}; got {detected_target!r}."
            )
        likelihood_model = str(_configured_value(params, "fisher_likelihood_model")).strip().lower()
        valid_likelihood_models = {
            "gaussian_fixed_variance",
            "poisson_exact",
            "gaussian_parameter_dependent_variance",
            "poisson_gaussian_approx",
        }
        if likelihood_model not in valid_likelihood_models:
            raise ValueError(
                "parameters['fisher_likelihood_model'] must be one of "
                f"{sorted(valid_likelihood_models)}; got {likelihood_model!r}."
            )
        return cls(
            detected_quanta_derivative_target=detected_target,
            likelihood_model=likelihood_model,
        )

    def require_signed_contrast_detected_quanta_target(self, *, context: str) -> None:
        if self.detected_quanta_derivative_target != "signed_contrast_scaled":
            raise ValueError(
                f"{context} currently computes Fisher on signed analysis contrast frames. "
                "Set parameters['detected_quanta_derivative_target']='signed_contrast_scaled' "
                "for this path, or route count-mean derivative comparisons through the "
                "detected-quanta comparator."
            )


@dataclass(frozen=True)
class BackendProfileSettings:
    """Resolved backend profile labeling and optional guard overrides."""

    profile_fidelity_label: str
    optical_filter_guard_pixels: float | None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "BackendProfileSettings":
        guard_raw = _configured_value(params, "optical_filter_guard_pixels")
        guard = None if guard_raw is None else _finite_float(
            guard_raw,
            key="optical_filter_guard_pixels",
            nonnegative=True,
        )
        return cls(
            profile_fidelity_label=str(_configured_value(params, "profile_fidelity_label")),
            optical_filter_guard_pixels=guard,
        )

    def filter_guard_radius_pixels(self) -> int | None:
        if self.optical_filter_guard_pixels is None:
            return None
        return int(math.ceil(self.optical_filter_guard_pixels))


@dataclass(frozen=True)
class OpticalInstrumentSettings:
    """Resolved optical instrument geometry and pupil sampling settings."""

    wavelength_nm: float
    probe_wavelength_nm: float
    probe_wavelength_nm_is_explicit: bool
    numerical_aperture: float
    refractive_index_medium: float
    refractive_index_immersion: float
    magnification: float
    objective_model: str | None
    objective_focal_length_mm: float
    pupil_samples: int
    vectorial_pupil_samples: int
    vectorial_pupil_samples_is_explicit: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OpticalInstrumentSettings":
        wavelength_nm = _finite_float(_configured_value(params, "wavelength_nm"), key="wavelength_nm", positive=True)
        probe_raw = _configured_value(params, "probe_wavelength_nm")
        probe_explicit = probe_raw is not None
        if probe_raw is None:
            probe_raw = (
                _configured_value(params, "ricm_wavelength_nm")
                if ModalitySettings.from_params(params).is_ricm
                else wavelength_nm
            )
        pupil_samples = int(_configured_value(params, "pupil_samples"))
        if pupil_samples <= 0:
            raise ValueError(f"parameters['pupil_samples'] must be positive; got {pupil_samples!r}.")
        vectorial_raw = _configured_value(params, "vectorial_pupil_samples")
        vectorial_explicit = vectorial_raw is not None
        vectorial_pupil_samples = pupil_samples if vectorial_raw is None else int(vectorial_raw)
        if vectorial_pupil_samples <= 0:
            raise ValueError(
                "parameters['vectorial_pupil_samples'] must be None or positive; "
                f"got {vectorial_raw!r}."
            )
        return cls(
            wavelength_nm=wavelength_nm,
            probe_wavelength_nm=_finite_float(probe_raw, key="probe_wavelength_nm", positive=True),
            probe_wavelength_nm_is_explicit=probe_explicit,
            numerical_aperture=_finite_float(
                _configured_value(params, "numerical_aperture"),
                key="numerical_aperture",
                positive=True,
            ),
            refractive_index_medium=_finite_float(
                _configured_value(params, "refractive_index_medium"),
                key="refractive_index_medium",
                positive=True,
            ),
            refractive_index_immersion=_finite_float(
                _configured_value(params, "refractive_index_immersion"),
                key="refractive_index_immersion",
                positive=True,
            ),
            magnification=_finite_float(_configured_value(params, "magnification"), key="magnification", positive=True),
            objective_model=(
                None
                if _configured_value(params, "objective_model") is None
                else str(_configured_value(params, "objective_model"))
            ),
            objective_focal_length_mm=_finite_float(
                _configured_value(params, "objective_focal_length_mm"),
                key="objective_focal_length_mm",
                positive=True,
            ),
            pupil_samples=pupil_samples,
            vectorial_pupil_samples=vectorial_pupil_samples,
            vectorial_pupil_samples_is_explicit=vectorial_explicit,
        )

    @property
    def collection_half_angle_rad(self) -> float:
        return math.asin(
            max(
                -1.0,
                min(1.0, self.numerical_aperture / self.refractive_index_medium),
            )
        )

    @property
    def cutoff_cycles_per_nm(self) -> float:
        return max(self.numerical_aperture / self.probe_wavelength_nm, 1e-30)


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
    source_samples: int
    illumination_sigma: float
    optical: OpticalModeSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "DpcSettings":
        def _matches_default(value: Any, default: Any) -> bool:
            try:
                return float(value) == float(default)
            except (TypeError, ValueError):
                return value == default

        def axis_gain(axis_key: str, generic_key: str) -> float:
            generic_present = generic_key in params
            axis_present = axis_key in params
            generic_default = default_param_value(generic_key)
            axis_default = default_param_value(axis_key)
            generic_is_override = (
                generic_present
                and not _matches_default(_configured_value(params, generic_key), generic_default)
            )
            axis_is_unset_default = (
                (not axis_present)
                or _matches_default(_configured_value(params, axis_key), axis_default)
            )
            if generic_is_override and axis_is_unset_default:
                value = _configured_value(params, generic_key)
            elif axis_present:
                value = _configured_value(params, axis_key)
            elif generic_present:
                value = _configured_value(params, generic_key)
            else:
                value = _configured_value(params, axis_key)
            return _finite_float(value, key=axis_key, nonnegative=True)

        source_samples = int(_configured_value(params, "dpc_source_samples"))
        if source_samples <= 0:
            raise ValueError("parameters['dpc_source_samples'] must be positive.")
        illumination_sigma = _finite_float(
            _configured_value(params, "dpc_illumination_sigma"),
            key="dpc_illumination_sigma",
            positive=True,
        )
        if illumination_sigma > 1.0:
            raise ValueError("parameters['dpc_illumination_sigma'] must satisfy 0 < sigma <= 1.")

        return cls(
            channel_model=str(_configured_value(params, "dpc_channel_model")).strip().lower(),
            transfer_model=str(_configured_value(params, "dpc_transfer_model")).strip().lower(),
            output_channel=str(_configured_value(params, "dpc_output_channel")).strip().lower(),
            intensity_gain_x=axis_gain("dpc_intensity_gain_x", "dpc_intensity_gain"),
            intensity_gain_y=axis_gain("dpc_intensity_gain_y", "dpc_intensity_gain"),
            phase_gradient_gain_x=axis_gain("dpc_phase_gradient_gain_x", "dpc_phase_gradient_gain"),
            phase_gradient_gain_y=axis_gain("dpc_phase_gradient_gain_y", "dpc_phase_gradient_gain"),
            source_samples=source_samples,
            illumination_sigma=illumination_sigma,
            optical=OpticalModeSettings.from_params(params),
        )


@dataclass(frozen=True)
class OffAxisHolographySettings:
    """Resolved off-axis holography carrier/reference settings."""

    fringe_period_detector_px: float
    fringe_angle_rad: float
    reference_amplitude_scale: float
    optical: OpticalModeSettings

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "OffAxisHolographySettings":
        period_detector_px = _finite_float(
            _configured_value(params, "off_axis_fringe_period_px"),
            key="off_axis_fringe_period_px",
            positive=True,
        )
        if period_detector_px < 2.0:
            raise ValueError(
                "parameters['off_axis_fringe_period_px'] must be >= 2.0 (Nyquist); "
                f"got {period_detector_px!r}."
            )
        return cls(
            fringe_period_detector_px=period_detector_px,
            fringe_angle_rad=_finite_float(
                _configured_value(params, "off_axis_fringe_angle_rad"),
                key="off_axis_fringe_angle_rad",
            ),
            reference_amplitude_scale=_finite_float(
                _configured_value(params, "off_axis_reference_amplitude_scale"),
                key="off_axis_reference_amplitude_scale",
                nonnegative=True,
            ),
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
        field_gain = _finite_float(_configured_value(params, "dark_field_field_gain"), key="dark_field_field_gain", positive=True)
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
        source_samples = int(_configured_value(params, "kohler_source_samples"))
        if source_samples <= 0:
            raise ValueError(f"parameters['kohler_source_samples'] must be positive; got {source_samples!r}.")
        coherence_factor = _finite_float(
            _configured_value(params, "kohler_coherence_factor"),
            key="kohler_coherence_factor",
            nonnegative=True,
        )
        if coherence_factor > 1.0:
            raise ValueError(
                f"parameters['kohler_coherence_factor'] must be <= 1.0; got {coherence_factor!r}."
            )
        return cls(
            coherence_factor=coherence_factor,
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
        instrument = OpticalInstrumentSettings.from_params(params)
        source_samples = int(_configured_value(params, "annular_dark_field_source_samples"))
        if source_samples <= 0:
            raise ValueError(
                "parameters['annular_dark_field_source_samples'] must be positive; "
                f"got {source_samples!r}."
            )
        inner_sigma = _finite_float(
            _configured_value(params, "annular_dark_field_inner_sigma"),
            key="annular_dark_field_inner_sigma",
            positive=True,
        )
        outer_sigma = _finite_float(
            _configured_value(params, "annular_dark_field_outer_sigma"),
            key="annular_dark_field_outer_sigma",
            positive=True,
        )
        optical = OpticalModeSettings.from_params(params)
        if inner_sigma <= 1.0:
            raise ValueError(
                "parameters['annular_dark_field_inner_sigma'] must exceed 1.0 "
                f"for dark-field illumination; got {inner_sigma!r}."
            )
        if inner_sigma >= outer_sigma:
            raise ValueError(
                "Annular dark-field source sigmas must satisfy inner < outer; "
                f"got inner={inner_sigma!r}, outer={outer_sigma!r}."
            )
        source_na = outer_sigma * instrument.numerical_aperture
        if source_na > instrument.refractive_index_medium + 1e-12:
            raise ValueError(
                "Annular dark-field source exceeds the immersion-medium NA: "
                "outer_sigma * numerical_aperture must be <= refractive_index_medium; "
                f"got {outer_sigma!r} * {instrument.numerical_aperture!r} = {source_na!r} "
                f"> {instrument.refractive_index_medium!r}."
            )
        return cls(
            inner_sigma=inner_sigma,
            outer_sigma=outer_sigma,
            source_samples=source_samples,
            dark_field=DarkFieldSettings.from_params(params),
            optical=optical,
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
    collection_reference_fraction: float | None
    reference_field_amplitude: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "IscatSettings":
        sample_environment = SampleEnvironmentSettings.from_params(params)
        if "iscat_reference_medium_material" in params:
            reference_medium_material = _configured_value(params, "iscat_reference_medium_material")
        else:
            reference_medium_material = sample_environment.medium_material

        if "iscat_reference_substrate_material" in params:
            reference_substrate_material = _configured_value(params, "iscat_reference_substrate_material")
        else:
            reference_substrate_material = sample_environment.bulk_substrate_material

        return cls(
            reference_model=str(_configured_value(params, "iscat_reference_model")).strip().lower(),
            reference_phase_rad=_finite_float(_configured_value(params, "iscat_reference_phase_rad"), key="iscat_reference_phase_rad"),
            reference_amplitude_scale=_configured_value(params, "iscat_reference_amplitude_scale"),
            reference_normalize_fresnel_phase_only=bool(_configured_value(params, "iscat_reference_normalize_fresnel_phase_only")),
            reference_coefficient=_configured_value(params, "iscat_reference_coefficient"),
            reference_medium_material=str(reference_medium_material),
            reference_substrate_material=str(reference_substrate_material),
            collection_model=str(_configured_value(params, "iscat_collection_model")).strip().lower(),
            collection_reference_fraction=(
                None
                if _configured_value(params, "iscat_collection_reference_fraction") is None
                else _finite_float(
                    _configured_value(params, "iscat_collection_reference_fraction"),
                    key="iscat_collection_reference_fraction",
                    positive=True,
                )
            ),
            reference_field_amplitude=OpticalModeSettings.from_params(params).reference_field_amplitude,
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
    particle_material: Any
    particle_reflection_coefficient: float
    interface_phase_shift_rad: float
    reference_field_amplitude: float
    gap_nm: float
    use_particle_z_as_gap: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "RicmSettings":
        return cls(
            interface_reflection_model=str(_configured_value(params, "ricm_interface_reflection_model")).strip().lower(),
            particle_reflection_model=str(_configured_value(params, "ricm_particle_reflection_model")).strip().lower(),
            interface_medium_material=str(_configured_value(params, "ricm_interface_medium_material")),
            interface_substrate_material=str(_configured_value(params, "ricm_interface_substrate_material")),
            thinfilm_layers=_configured_value(params, "ricm_thinfilm_layers"),
            interface_reflection_coefficient=_finite_float(
                _configured_value(params, "ricm_interface_reflection_coefficient"),
                key="ricm_interface_reflection_coefficient",
            ),
            particle_medium_material=str(_configured_value(params, "ricm_particle_medium_material")),
            particle_material=deepcopy(_configured_value(params, "ricm_particle_material")),
            particle_reflection_coefficient=_finite_float(
                _configured_value(params, "ricm_particle_reflection_coefficient"),
                key="ricm_particle_reflection_coefficient",
            ),
            interface_phase_shift_rad=_finite_float(
                _configured_value(params, "ricm_interface_phase_shift_rad"),
                key="ricm_interface_phase_shift_rad",
            ),
            reference_field_amplitude=OpticalModeSettings.from_params(params).reference_field_amplitude,
            gap_nm=_finite_float(
                _configured_value(params, "ricm_gap_nm"),
                key="ricm_gap_nm",
                nonnegative=True,
            ),
            use_particle_z_as_gap=bool(_configured_value(params, "ricm_use_particle_z_as_gap")),
        )


# ---------------------------------------------------------------------------
# Optical aberration resolution -- SINGLE source of truth for every PSF path.
#
# The simulator models three independent optical-imperfection effects:
#   * spherical_aberration_strength : primary spherical Zernike, in waves RMS
#   * apodization_factor            : Gaussian pupil amplitude exp(-a * rho^2)
#   * random_aberration_strength    : residual random pupil-phase realization
#
# Previously these were three separate knobs, each defaulting to a STRONG value
# (0.25 / 1.8 / 1.5 -> Strehl < 0.1), so obtaining a clean diffraction-limited
# PSF meant zeroing all three by hand -- and the scalar and vectorial backends
# read them independently, which let the two drift (and hid a NameError). They
# are now resolved in ONE place, chosen by a single preset:
#
#   "ideal"     (default) -> all zero: the diffraction-limited PSF. The correct
#                            default for physics validation, reproducible CRLB,
#                            and analytic-reference comparison.
#   "realistic"           -> a documented, near-diffraction-limited residual set
#                            (Strehl ~0.9) for training-data realism.
#   "custom"              -> use the individual *_strength params verbatim.
#
# Tune the realism level in ONE place: REALISTIC_ABERRATION below.
# ---------------------------------------------------------------------------

OPTICAL_ABERRATION_PRESETS = ("ideal", "realistic", "custom")

IDEAL_ABERRATION = {
    "spherical_aberration_strength": 0.0,
    "apodization_factor": 0.0,
    "random_aberration_strength": 0.0,
}

# Near-diffraction-limited "good real microscope" (Strehl ~0.9). Tunable here.
REALISTIC_ABERRATION = {
    "spherical_aberration_strength": 0.05,  # ~0.05 waves RMS primary spherical
    "apodization_factor": 0.5,              # mild Gaussian pupil (edge amp e^-0.5)
    "random_aberration_strength": 0.05,     # small residual random pupil phase
}


@dataclass(frozen=True)
class AberrationSettings:
    """Resolved optical-aberration amplitudes shared by every PSF backend.

    This is the only place the three aberration effects are turned into numbers,
    so the scalar and vectorial backends (and the pupil-sampling helper) cannot
    disagree about the optical system they are modeling.
    """

    preset: str
    apodization_factor: float
    spherical_aberration_strength: float
    random_aberration_strength: float
    optical_aberration_seed: int | None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "AberrationSettings":
        raw_preset = _configured_optional(params, "optical_aberration_preset")
        preset = "ideal" if raw_preset is None else str(raw_preset).strip().lower()
        if preset not in OPTICAL_ABERRATION_PRESETS:
            raise ValueError(
                "parameters['optical_aberration_preset'] must be one of "
                f"{OPTICAL_ABERRATION_PRESETS}; got {raw_preset!r}."
            )
        if preset == "ideal":
            values = IDEAL_ABERRATION
        elif preset == "realistic":
            values = REALISTIC_ABERRATION
        else:  # "custom": honor the individual knobs verbatim
            values = {
                "spherical_aberration_strength": _configured_value(params, "spherical_aberration_strength"),
                "apodization_factor": _configured_value(params, "apodization_factor"),
                "random_aberration_strength": _configured_value(params, "random_aberration_strength"),
            }
        return cls(
            preset=preset,
            apodization_factor=_finite_float(
                values["apodization_factor"], key="apodization_factor", nonnegative=True
            ),
            spherical_aberration_strength=_finite_float(
                values["spherical_aberration_strength"],
                key="spherical_aberration_strength",
                nonnegative=True,
            ),
            random_aberration_strength=_finite_float(
                values["random_aberration_strength"],
                key="random_aberration_strength",
                nonnegative=True,
            ),
            optical_aberration_seed=(
                None if (seed_raw := _configured_optional(params, "optical_aberration_seed")) is None else int(seed_raw)
            ),
        )


@dataclass(frozen=True)
class VectorialOpticsSettings:
    """Resolved vectorial-Debye sampling and aberration settings."""

    sampling: SamplingGeometry
    instrument: OpticalInstrumentSettings
    optical: OpticalModeSettings
    obliquity_apodization: bool
    apodization_factor: float
    spherical_aberration_strength: float
    random_aberration_strength: float
    optical_aberration_seed: int | None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "VectorialOpticsSettings":
        aberration = AberrationSettings.from_params(params)
        return cls(
            sampling=SamplingGeometry.from_params(params),
            instrument=OpticalInstrumentSettings.from_params(params),
            optical=OpticalModeSettings.from_params(params),
            obliquity_apodization=bool(_configured_value(params, "vectorial_obliquity_apodization")),
            apodization_factor=aberration.apodization_factor,
            spherical_aberration_strength=aberration.spherical_aberration_strength,
            random_aberration_strength=aberration.random_aberration_strength,
            optical_aberration_seed=aberration.optical_aberration_seed,
        )


@dataclass(frozen=True)
class CoverslipAberrationSettings:
    """Resolved coverslip-mismatch aberration settings."""

    model: str
    thickness_um: float
    design_thickness_um: float
    refractive_index: float
    design_refractive_index: float
    subtract_piston: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "CoverslipAberrationSettings":
        model = str(_configured_value(params, "coverslip_aberration_model")).strip().lower()
        if model in {"", "disabled", "off"}:
            model = "none"
        if model not in {"none", "gibson_lanni", "coverslip_mismatch"}:
            raise ValueError(
                "coverslip_aberration_model must be 'none', 'gibson_lanni', or "
                f"'coverslip_mismatch'; got {model!r}."
            )
        return cls(
            model=model,
            thickness_um=_finite_float(
                _configured_value(params, "coverslip_thickness_um"),
                key="coverslip_thickness_um",
                nonnegative=True,
            ),
            design_thickness_um=_finite_float(
                _configured_value(params, "coverslip_design_thickness_um"),
                key="coverslip_design_thickness_um",
                nonnegative=True,
            ),
            refractive_index=_finite_float(
                _configured_value(params, "coverslip_refractive_index"),
                key="coverslip_refractive_index",
                positive=True,
            ),
            design_refractive_index=_finite_float(
                _configured_value(params, "coverslip_design_refractive_index"),
                key="coverslip_design_refractive_index",
                positive=True,
            ),
            subtract_piston=bool(_configured_value(params, "coverslip_aberration_subtract_piston")),
        )

    @property
    def metadata_model(self) -> str:
        return "gibson_lanni" if self.model == "coverslip_mismatch" else self.model


@dataclass(frozen=True)
class SpectralIntegrationSettings:
    """Resolved spectral-channel expansion settings."""

    model: str
    illumination_center_nm: float
    illumination_fwhm_nm: float
    illumination_num_samples: int
    broadband_wavelengths_nm: Any | None
    broadband_weights: Any | None
    detector_spectral_response_model: str
    allow_broadband_overwrite_channels: bool
    channels: Any | None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SpectralIntegrationSettings":
        model = str(_configured_value(params, "spectral_integration_model")).strip().lower()
        if model in {"", "single_wavelength"}:
            model = "single_wavelength"
        if model not in {"single_wavelength", "configured_channels", "broadband_quadrature"}:
            raise ValueError(
                "spectral_integration_model must be 'single_wavelength', "
                f"'configured_channels', or 'broadband_quadrature'; got {model!r}."
            )
        sample_count = int(_configured_value(params, "illumination_spectrum_num_samples"))
        if sample_count <= 0:
            raise ValueError("illumination_spectrum_num_samples must be positive.")
        detector_model = str(
            _configured_value(params, "detector_spectral_response_model")
        ).strip().lower()
        if detector_model not in {"rgb_heuristic", "flat", "table"}:
            raise ValueError(
                "detector_spectral_response_model must be 'rgb_heuristic', "
                f"'flat', or 'table'; got {detector_model!r}."
            )
        return cls(
            model=model,
            illumination_center_nm=_finite_float(
                _configured_value(params, "illumination_spectrum_center_nm"),
                key="illumination_spectrum_center_nm",
                positive=True,
            ),
            illumination_fwhm_nm=_finite_float(
                _configured_value(params, "illumination_spectrum_fwhm_nm"),
                key="illumination_spectrum_fwhm_nm",
                nonnegative=True,
            ),
            illumination_num_samples=sample_count,
            broadband_wavelengths_nm=_configured_value(params, "broadband_wavelengths_nm"),
            broadband_weights=_configured_value(params, "broadband_weights"),
            detector_spectral_response_model=detector_model,
            allow_broadband_overwrite_channels=bool(
                _configured_value(params, "allow_broadband_overwrite_channels")
            ),
            channels=_configured_value(params, "channels"),
        )


@dataclass(frozen=True)
class ZernikePhaseSettings:
    """Resolved Zernike phase-ring contrast settings."""

    model: str
    inner_fraction: float
    outer_fraction: float
    shift_rad: float
    amplitude: float
    gain: float
    bias: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "ZernikePhaseSettings":
        model = str(_configured_value(params, "zernike_model")).strip().lower()
        if model not in {"pupil_phase_ring", "fourier_phase_ring_proxy"}:
            raise ValueError(
                "parameters['zernike_model'] must be 'pupil_phase_ring', "
                f"or 'fourier_phase_ring_proxy'; got {model!r}."
            )
        inner = _finite_float(
            _configured_value(params, "zernike_phase_ring_inner_fraction"),
            key="zernike_phase_ring_inner_fraction",
            nonnegative=True,
        )
        outer = _finite_float(
            _configured_value(params, "zernike_phase_ring_outer_fraction"),
            key="zernike_phase_ring_outer_fraction",
            positive=True,
        )
        if inner < 0.0 or outer <= inner or outer > 1.0:
            raise ValueError(
                "Zernike phase ring fractions must satisfy 0 <= inner < outer <= 1."
            )
        return cls(
            model=model,
            inner_fraction=inner,
            outer_fraction=outer,
            shift_rad=_finite_float(
                _configured_value(params, "zernike_phase_ring_shift_rad"),
                key="zernike_phase_ring_shift_rad",
            ),
            amplitude=_finite_float(
                _configured_value(params, "zernike_phase_ring_amplitude"),
                key="zernike_phase_ring_amplitude",
            ),
            gain=_finite_float(
                _configured_value(params, "zernike_phase_ring_gain"),
                key="zernike_phase_ring_gain",
                nonnegative=True,
            ),
            bias=_finite_float(
                _configured_value(params, "zernike_phase_bias"),
                key="zernike_phase_bias",
                nonnegative=True,
            ),
        )

    @property
    def coordinate_system(self) -> str:
        if self.model == "pupil_phase_ring":
            return "objective_pupil_na_over_wavelength"
        return "fft_nyquist_normalized"


@dataclass(frozen=True)
class VolumeRenderingSettings:
    """Resolved volumetric z-plane and projection settings."""

    z_planes_nm: Any | None
    z_count: int
    z_range_nm: float
    z_step_nm: float
    imaging_mode: str
    confocal_pinhole_sigma_nm: float
    light_sheet_center_z_nm: float
    light_sheet_sigma_nm: float
    output_mode: str
    holotomography_projection_angles_deg: Any
    holotomography_output_mode: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "VolumeRenderingSettings":
        z_count = int(_configured_value(params, "volumetric_z_count"))
        if z_count <= 0:
            raise ValueError("volumetric_z_count must be positive.")
        z_range_nm = _finite_float(
            _configured_value(params, "volumetric_z_range_nm"),
            key="volumetric_z_range_nm",
            nonnegative=True,
        )
        z_step_nm = _finite_float(
            _configured_value(params, "volumetric_z_step_nm"),
            key="volumetric_z_step_nm",
            positive=True,
        )
        imaging_mode = str(_configured_value(params, "volumetric_imaging_mode")).strip().lower()
        if imaging_mode not in {
            "single_plane",
            "z_stack",
            "holotomography_projection",
            "confocal",
            "light_sheet",
        }:
            raise ValueError(f"Unsupported volumetric_imaging_mode {imaging_mode!r}.")
        return cls(
            z_planes_nm=_configured_value(params, "volumetric_z_planes_nm"),
            z_count=z_count,
            z_range_nm=z_range_nm,
            z_step_nm=z_step_nm,
            imaging_mode=imaging_mode,
            confocal_pinhole_sigma_nm=_finite_float(
                _configured_value(params, "confocal_pinhole_sigma_nm"),
                key="confocal_pinhole_sigma_nm",
                positive=True,
            ),
            light_sheet_center_z_nm=_finite_float(
                _configured_value(params, "light_sheet_center_z_nm"),
                key="light_sheet_center_z_nm",
            ),
            light_sheet_sigma_nm=_finite_float(
                _configured_value(params, "light_sheet_sigma_nm"),
                key="light_sheet_sigma_nm",
                positive=True,
            ),
            output_mode=str(_configured_value(params, "volume_output_mode")).strip().lower(),
            holotomography_projection_angles_deg=_configured_value(
                params,
                "holotomography_projection_angles_deg",
            ),
            holotomography_output_mode=str(
                _configured_value(params, "holotomography_output_mode")
            ).strip().lower(),
        )


@dataclass(frozen=True)
class FluorescenceSettings:
    """Resolved fluorescence-model settings."""

    backend: str
    source_representation: str
    volume_slices: int
    volume_slice_thickness_nm: float | None
    excitation_wavelength_nm: float
    emission_wavelength_nm: float
    quantum_yield: float
    excitation_scale: float
    absorbed_excitation_photons_per_fluorophore_per_frame: float
    collection_efficiency: float
    detector_qe: float
    background_count: float
    spectral_bandwidth_nm: float
    bleaching_rate_per_frame: float
    emission_psf_sigma_nm: float | None
    emission_psf_sigma_px: float
    reference_status: str
    reference_validation_hash: str | None
    allow_psf_fallback: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "FluorescenceSettings":
        sampling = SamplingGeometry.from_params(params)
        backend = cls._normalize_backend(
            _configured_value(params, "fluorescence_backend"),
            key="fluorescence_backend",
        )
        source_representation = cls._normalize_source_representation(
            _configured_value(params, "fluorescence_source_representation"),
            key="fluorescence_source_representation",
        )
        volume_slices = cls._positive_int(
            _configured_value(params, "fluorescence_volume_slices"),
            key="fluorescence_volume_slices",
        )
        volume_slice_thickness_nm = cls._optional_positive_float(
            _configured_value(params, "fluorescence_volume_slice_thickness_nm"),
            key="fluorescence_volume_slice_thickness_nm",
        )
        reference_status = str(_configured_value(params, "fluorescence_reference_status")).strip().lower()
        if reference_status not in {"physics_based_unvalidated", "reference_validated"}:
            raise ValueError(
                "parameters['fluorescence_reference_status'] must be 'physics_based_unvalidated' "
                f"or 'reference_validated'; got {reference_status!r}."
            )
        reference_hash_raw = _configured_value(params, "fluorescence_reference_validation_hash")
        reference_hash = None if reference_hash_raw is None else str(reference_hash_raw)
        if reference_status == "reference_validated" and not reference_hash:
            raise ValueError(
                "parameters['fluorescence_reference_status']='reference_validated' requires "
                "parameters['fluorescence_reference_validation_hash']."
            )
        quantum_yield = _finite_float(
            _configured_value(params, "fluorescence_quantum_yield"),
            key="fluorescence_quantum_yield",
            nonnegative=True,
        )
        if quantum_yield > 1.0:
            raise ValueError(f"fluorescence_quantum_yield must be <= 1; got {quantum_yield!r}.")
        collection_efficiency = _finite_float(
            _configured_value(params, "fluorescence_collection_efficiency"),
            key="fluorescence_collection_efficiency",
            nonnegative=True,
        )
        if collection_efficiency > 1.0:
            raise ValueError(f"fluorescence_collection_efficiency must be <= 1; got {collection_efficiency!r}.")
        sigma_nm_raw = _configured_value(params, "fluorescence_emission_psf_sigma_nm")
        sigma_nm = None if sigma_nm_raw is None else _finite_float(
            sigma_nm_raw,
            key="fluorescence_emission_psf_sigma_nm",
            nonnegative=True,
        )
        if sigma_nm is not None:
            emission_psf_sigma_px = sigma_nm / float(sampling.detector_pixel_size_nm)
        else:
            emission_psf_sigma_px = 1.0
        return cls(
            backend=backend,
            source_representation=source_representation,
            volume_slices=volume_slices,
            volume_slice_thickness_nm=volume_slice_thickness_nm,
            excitation_wavelength_nm=_finite_float(
                _configured_value(params, "fluorescence_excitation_wavelength_nm"),
                key="fluorescence_excitation_wavelength_nm",
                positive=True,
            ),
            emission_wavelength_nm=_finite_float(
                _configured_value(params, "fluorescence_emission_wavelength_nm"),
                key="fluorescence_emission_wavelength_nm",
                positive=True,
            ),
            quantum_yield=quantum_yield,
            excitation_scale=_finite_float(
                _configured_value(params, "fluorescence_excitation_scale"),
                key="fluorescence_excitation_scale",
                nonnegative=True,
            ),
            absorbed_excitation_photons_per_fluorophore_per_frame=(
                _finite_float(
                    _configured_value(params, "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"),
                    key="fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame",
                    nonnegative=True,
                )
            ),
            collection_efficiency=collection_efficiency,
            detector_qe=DetectorSettings.from_params(params, fluorescence=True).detector_qe,
            background_count=_finite_float(
                _configured_value(params, "fluorescence_background"),
                key="fluorescence_background",
                nonnegative=True,
            ),
            spectral_bandwidth_nm=_finite_float(
                _configured_value(params, "fluorescence_spectral_bandwidth_nm"),
                key="fluorescence_spectral_bandwidth_nm",
                positive=True,
            ),
            bleaching_rate_per_frame=_finite_float(
                _configured_value(params, "fluorescence_bleaching_rate_per_frame"),
                key="fluorescence_bleaching_rate_per_frame",
                nonnegative=True,
            ),
            emission_psf_sigma_nm=sigma_nm,
            emission_psf_sigma_px=_finite_float(
                emission_psf_sigma_px,
                key="fluorescence_emission_psf_sigma_px",
                nonnegative=True,
            ),
            reference_status=reference_status,
            reference_validation_hash=reference_hash,
            allow_psf_fallback=bool(_configured_value(params, "fluorescence_allow_psf_fallback")),
        )

    @staticmethod
    def _normalize_backend(value: Any, *, key: str) -> str:
        backend = str(value).strip().lower()
        if backend not in {"parametric_psf", "vectorial_photophysics"}:
            raise ValueError(
                f"parameters['{key}'] must be 'parametric_psf' or "
                f"'vectorial_photophysics'; got {backend!r}."
            )
        return backend

    @staticmethod
    def _normalize_source_representation(value: Any, *, key: str) -> str:
        source_representation = str(value).strip().lower()
        if source_representation not in {"projected_2d", "volume"}:
            raise ValueError(
                f"parameters['{key}'] must be 'projected_2d' or 'volume'; "
                f"got {source_representation!r}."
            )
        return source_representation

    def with_source_model(self, *, backend: str, source_representation: str) -> "FluorescenceSettings":
        return replace(
            self,
            backend=self._normalize_backend(backend, key="tirf_fluorescence_backend"),
            source_representation=self._normalize_source_representation(
                source_representation,
                key="tirf_source_representation",
            ),
        )

    @staticmethod
    def _positive_int(value: Any, *, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"parameters['{key}'] must be a positive integer; got {value!r}.")
        out = int(value)
        if out <= 0:
            raise ValueError(f"parameters['{key}'] must be a positive integer; got {value!r}.")
        return out

    @staticmethod
    def _optional_positive_float(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        return _finite_float(value, key=key, positive=True)


@dataclass(frozen=True)
class TirfSettings:
    """Owned TIRF excitation/interface settings composed with fluorescence detection."""

    fluorescence: FluorescenceSettings
    fluorescence_backend: str
    source_representation: str
    penetration_depth_nm: float
    use_angle_derived_penetration_depth: bool
    prism_refractive_index: float
    sample_refractive_index: float
    incident_angle_deg: float
    height_offset_nm: float
    effective_numerical_aperture: float | None
    vectorial_effective_na_applied: bool

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "TirfSettings":
        fluorescence = FluorescenceSettings.from_params(params)
        fluorescence_backend = FluorescenceSettings._normalize_backend(
            _configured_value(params, "tirf_fluorescence_backend"),
            key="tirf_fluorescence_backend",
        )
        source_representation = FluorescenceSettings._normalize_source_representation(
            _configured_value(params, "tirf_source_representation"),
            key="tirf_source_representation",
        )
        prism_ri = _finite_float(
            _configured_value(params, "tirf_prism_refractive_index"),
            key="tirf_prism_refractive_index",
            positive=True,
        )
        sample_ri = _finite_float(
            _configured_value(params, "tirf_sample_refractive_index"),
            key="tirf_sample_refractive_index",
            positive=True,
        )
        incident_angle_deg = _finite_float(
            _configured_value(params, "tirf_incident_angle_deg"),
            key="tirf_incident_angle_deg",
        )
        use_angle_depth = bool(_configured_value(params, "tirf_use_angle_derived_penetration_depth"))
        if use_angle_depth:
            wavelength_nm = fluorescence.excitation_wavelength_nm
            angle_rad = math.radians(incident_angle_deg)
            sin_term = prism_ri * math.sin(angle_rad)
            under_root = sin_term * sin_term - sample_ri * sample_ri
            if not math.isfinite(under_root) or under_root <= 0.0:
                raise ValueError(
                    "TIRF incident angle must exceed the critical angle when "
                    "parameters['tirf_use_angle_derived_penetration_depth'] is enabled."
                )
            penetration_depth_nm = float(wavelength_nm / (4.0 * math.pi * math.sqrt(under_root)))
        else:
            penetration_depth_nm = _finite_float(
                _configured_value(params, "tirf_penetration_depth_nm"),
                key="tirf_penetration_depth_nm",
                positive=True,
            )
        effective_na = cls._optional_positive_float(
            _configured_value(params, "tirf_effective_numerical_aperture"),
            key="tirf_effective_numerical_aperture",
        )
        return cls(
            fluorescence=fluorescence.with_source_model(
                backend=fluorescence_backend,
                source_representation=source_representation,
            ),
            fluorescence_backend=fluorescence_backend,
            source_representation=source_representation,
            penetration_depth_nm=penetration_depth_nm,
            use_angle_derived_penetration_depth=use_angle_depth,
            prism_refractive_index=prism_ri,
            sample_refractive_index=sample_ri,
            incident_angle_deg=incident_angle_deg,
            height_offset_nm=_finite_float(
                _configured_value(params, "tirf_height_offset_nm"),
                key="tirf_height_offset_nm",
            ),
            effective_numerical_aperture=effective_na,
            vectorial_effective_na_applied=(
                effective_na is not None and fluorescence_backend == "vectorial_photophysics"
            ),
        )

    @staticmethod
    def _optional_positive_float(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        return _finite_float(value, key=key, positive=True)

    @property
    def critical_angle_deg(self) -> float:
        ratio = min(self.sample_refractive_index / max(self.prism_refractive_index, 1e-12), 1.0)
        return float(math.degrees(math.asin(ratio)))


@dataclass(frozen=True)
class SemSettings:
    """Resolved SEM source/probe/count settings."""

    model: str
    backend: str
    source_representation: str
    source_resolution: SEMSourceRepresentationResolution
    source_z_origin: str
    source_z_offset_nm: float
    volume_slices: int
    volume_slice_thickness_nm: float | None
    probe_sigma_nm: float | None
    probe_sigma_px: float
    filter_guard_pixels: float | None
    acceleration_kV: float
    interaction_volume_nm: float
    baseline_yield: float
    edge_contrast_gain: float
    bulk_contrast_gain: float
    topography_contrast_gain: float
    detector_acceptance: float
    detector_takeoff_angle_deg: float
    detector_direction_xy: tuple[float, float]
    escape_depth_nm: float
    backscatter_fraction: float
    transport_material_scale: float
    transport_source_exponent: float
    transport_topography_exponent: float
    beam_current_nA: float
    dwell_time_us: float
    electrons_per_pixel: float
    sem_monte_carlo_seed: int
    monte_carlo_trajectories: int
    monte_carlo_steps: int
    monte_carlo_step_nm: float | None
    monte_carlo_range_nm: float | None
    monte_carlo_scatter_std_deg: float
    monte_carlo_kernel_size_px: int | None
    physical_max_steps: int
    physical_energy_cutoff_keV: float
    physical_elastic_model: str
    reference_kernel_path: str | None
    reference_kernel_sha256: str | None
    reference_material: str
    reference_geometry: str
    reference_source_depth_nm: float
    reference_incident_angle_deg: float
    sampling: SamplingGeometry

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "SemSettings":
        sampling = SamplingGeometry.from_params(params)
        model = str(_configured_value(params, "sem_model")).strip().lower()
        backend = str(_configured_value(params, "sem_backend")).strip().lower()
        model_choices = {
            "gaussian_probe_secondary_yield",
            "interaction_volume_proxy",
            "physical_electron_transport",
        }
        backend_choices = {
            "gaussian_probe_proxy",
            "interaction_volume_proxy",
            "monte_carlo_transport",
            "monte_carlo_physical",
            "syniscopy_transport_lite",
            "reference_kernel_table",
        }
        if model not in model_choices:
            raise ValueError(
                "parameters['sem_model'] must be one of "
                f"{sorted(model_choices)!r}; got {model!r}."
            )
        if backend not in backend_choices:
            raise ValueError(
                "parameters['sem_backend'] must be one of "
                f"{sorted(backend_choices)!r}; got {backend!r}."
            )
        if model == "interaction_volume_proxy" and backend != "interaction_volume_proxy":
            raise ValueError(
                "parameters['sem_model']='interaction_volume_proxy' requires "
                "parameters['sem_backend']='interaction_volume_proxy'."
            )
        if backend == "interaction_volume_proxy" and model != "interaction_volume_proxy":
            raise ValueError(
                "parameters['sem_backend']='interaction_volume_proxy' requires "
                "parameters['sem_model']='interaction_volume_proxy'."
            )
        physical_backends = {
            "monte_carlo_physical",
            "monte_carlo_transport",
            "syniscopy_transport_lite",
            "reference_kernel_table",
        }
        if model == "physical_electron_transport" and backend not in physical_backends:
            raise ValueError(
                "parameters['sem_model']='physical_electron_transport' requires "
                "a physical-family SEM backend."
            )
        if backend in physical_backends and model != "physical_electron_transport":
            raise ValueError(
                f"parameters['sem_backend']={backend!r} requires "
                "parameters['sem_model']='physical_electron_transport'."
            )

        source_resolution = resolve_sem_source_representation(
            _configured_value(params, "sem_source_representation"),
            backend_name=backend,
        )
        source_z_origin = str(_configured_value(params, "sem_source_z_origin")).strip().lower()
        if source_z_origin not in {"entry_surface_depth", "focus_plane_relative"}:
            raise ValueError(
                "parameters['sem_source_z_origin'] must be 'entry_surface_depth' "
                f"or 'focus_plane_relative'; got {source_z_origin!r}."
            )
        if source_resolution.effective == "volume" and source_z_origin == "focus_plane_relative":
            raise ValueError(
                "SEM volume source maps represent physical material depth, not "
                "focus-relative imaging-response defocus. Use "
                "sem_source_z_origin='entry_surface_depth' for volume SEM transport."
            )

        volume_slices = int(_configured_value(params, "sem_volume_slices"))
        if volume_slices <= 0:
            raise ValueError(f"parameters['sem_volume_slices'] must be positive; got {volume_slices!r}.")
        volume_slice_thickness_raw = _configured_value(params, "sem_volume_slice_thickness_nm")
        volume_slice_thickness_nm = None if volume_slice_thickness_raw is None else _finite_float(
            volume_slice_thickness_raw,
            key="sem_volume_slice_thickness_nm",
            positive=True,
        )
        sigma_nm_raw = _configured_value(params, "sem_probe_sigma_nm")
        sigma_nm = None if sigma_nm_raw is None else _finite_float(
            sigma_nm_raw,
            key="sem_probe_sigma_nm",
            nonnegative=True,
        )
        if sigma_nm is not None:
            probe_sigma_px = sigma_nm / float(sampling.model_canvas_pixel_size_nm)
        else:
            probe_sigma_px = 1.0
        filter_guard_raw = _configured_value(params, "sem_filter_guard_pixels")
        filter_guard_pixels = None if filter_guard_raw is None else _finite_float(
            filter_guard_raw,
            key="sem_filter_guard_pixels",
            nonnegative=True,
        )
        detector_direction = cls._unit_detector_direction(_configured_value(params, "sem_detector_direction_xy"))
        detector_acceptance = _finite_float(
            _configured_value(params, "sem_detector_acceptance"),
            key="sem_detector_acceptance",
            nonnegative=True,
        )
        if detector_acceptance > 1.0:
            raise ValueError(
                f"parameters['sem_detector_acceptance'] must be <= 1; got {detector_acceptance!r}."
            )
        detector_takeoff_angle_deg = _finite_float(
            _configured_value(params, "sem_detector_takeoff_angle_deg"),
            key="sem_detector_takeoff_angle_deg",
            nonnegative=True,
        )
        if detector_takeoff_angle_deg > 90.0:
            raise ValueError(
                "parameters['sem_detector_takeoff_angle_deg'] is measured above the specimen "
                f"surface and must be <= 90 degrees; got {detector_takeoff_angle_deg!r}."
            )
        backscatter_fraction = _finite_float(
            _configured_value(params, "sem_backscatter_fraction"),
            key="sem_backscatter_fraction",
            nonnegative=True,
        )
        if backscatter_fraction > 1.0:
            raise ValueError(
                f"parameters['sem_backscatter_fraction'] must be <= 1; got {backscatter_fraction!r}."
            )
        kernel_raw = _configured_value(params, "sem_monte_carlo_kernel_size_px")
        kernel_size = None if kernel_raw is None else int(kernel_raw)
        if kernel_size is not None and kernel_size <= 0:
            raise ValueError(
                "parameters['sem_monte_carlo_kernel_size_px'] must be positive when supplied."
            )
        if kernel_size is not None and kernel_size % 2 == 0:
            kernel_size += 1
        physical_elastic_model = str(_configured_value(params, "sem_physical_elastic_model")).strip().lower()
        physical_elastic_models = {"screened_rutherford", "mott_browning"}
        if physical_elastic_model not in physical_elastic_models:
            raise ValueError(
                "parameters['sem_physical_elastic_model'] must be one of "
                f"{sorted(physical_elastic_models)!r}; got {physical_elastic_model!r}."
            )
        transport_source_exponent = _finite_float(
            _configured_value(params, "sem_transport_source_exponent"),
            key="sem_transport_source_exponent",
            positive=True,
        )
        if transport_source_exponent < 0.05:
            raise ValueError("parameters['sem_transport_source_exponent'] must be >= 0.05.")
        transport_topography_exponent = _finite_float(
            _configured_value(params, "sem_transport_topography_exponent"),
            key="sem_transport_topography_exponent",
            positive=True,
        )
        if transport_topography_exponent < 0.05:
            raise ValueError("parameters['sem_transport_topography_exponent'] must be >= 0.05.")
        reference_incident_angle_deg = _finite_float(
            _configured_value(params, "sem_reference_incident_angle_deg"),
            key="sem_reference_incident_angle_deg",
            nonnegative=True,
        )
        if reference_incident_angle_deg > 180.0:
            raise ValueError(
                "parameters['sem_reference_incident_angle_deg'] must be <= 180 degrees; "
                f"got {reference_incident_angle_deg!r}."
            )
        return cls(
            model=model,
            backend=backend,
            source_representation=source_resolution.requested,
            source_resolution=source_resolution,
            source_z_origin=source_z_origin,
            source_z_offset_nm=_finite_float(_configured_value(params, "sem_source_z_offset_nm"), key="sem_source_z_offset_nm"),
            volume_slices=volume_slices,
            volume_slice_thickness_nm=volume_slice_thickness_nm,
            probe_sigma_nm=sigma_nm,
            probe_sigma_px=_finite_float(
                probe_sigma_px,
                key="sem_probe_sigma_px",
                nonnegative=True,
            ),
            filter_guard_pixels=filter_guard_pixels,
            acceleration_kV=_finite_float(_configured_value(params, "sem_acceleration_kV"), key="sem_acceleration_kV", positive=True),
            interaction_volume_nm=_finite_float(
                _configured_value(params, "sem_interaction_volume_nm"),
                key="sem_interaction_volume_nm",
                nonnegative=True,
            ),
            baseline_yield=_finite_float(_configured_value(params, "sem_baseline_yield"), key="sem_baseline_yield", nonnegative=True),
            edge_contrast_gain=_finite_float(
                _configured_value(params, "sem_edge_contrast_gain"),
                key="sem_edge_contrast_gain",
                nonnegative=True,
            ),
            bulk_contrast_gain=_finite_float(
                _configured_value(params, "sem_bulk_contrast_gain"),
                key="sem_bulk_contrast_gain",
                nonnegative=True,
            ),
            topography_contrast_gain=_finite_float(
                _configured_value(params, "sem_topography_contrast_gain"),
                key="sem_topography_contrast_gain",
                nonnegative=True,
            ),
            detector_acceptance=detector_acceptance,
            detector_takeoff_angle_deg=detector_takeoff_angle_deg,
            detector_direction_xy=detector_direction,
            escape_depth_nm=_finite_float(_configured_value(params, "sem_escape_depth_nm"), key="sem_escape_depth_nm", nonnegative=True),
            backscatter_fraction=backscatter_fraction,
            transport_material_scale=_finite_float(
                _configured_value(params, "sem_transport_material_scale"),
                key="sem_transport_material_scale",
                nonnegative=True,
            ),
            transport_source_exponent=transport_source_exponent,
            transport_topography_exponent=transport_topography_exponent,
            beam_current_nA=_finite_float(_configured_value(params, "sem_beam_current_nA"), key="sem_beam_current_nA", nonnegative=True),
            dwell_time_us=_finite_float(_configured_value(params, "sem_dwell_time_us"), key="sem_dwell_time_us", nonnegative=True),
            electrons_per_pixel=CountBudgetSettings.from_params(params).sem_electrons_per_pixel,
            sem_monte_carlo_seed=(
                0
                if (seed_raw := _configured_optional(params, "sem_monte_carlo_seed")) is None
                and (seed_raw := AcquisitionProfile.from_params(params).random_seed) is None
                else int(seed_raw)
            ),
            monte_carlo_trajectories=cls._positive_int(_configured_value(params, "sem_monte_carlo_trajectories"), key="sem_monte_carlo_trajectories"),
            monte_carlo_steps=cls._positive_int(_configured_value(params, "sem_monte_carlo_steps"), key="sem_monte_carlo_steps"),
            monte_carlo_step_nm=cls._optional_positive_float(_configured_value(params, "sem_monte_carlo_step_nm"), key="sem_monte_carlo_step_nm"),
            monte_carlo_range_nm=cls._optional_positive_float(_configured_value(params, "sem_monte_carlo_range_nm"), key="sem_monte_carlo_range_nm"),
            monte_carlo_scatter_std_deg=_finite_float(
                _configured_value(params, "sem_monte_carlo_scatter_std_deg"),
                key="sem_monte_carlo_scatter_std_deg",
                nonnegative=True,
            ),
            monte_carlo_kernel_size_px=kernel_size,
            physical_max_steps=cls._positive_int(_configured_value(params, "sem_physical_max_steps"), key="sem_physical_max_steps"),
            physical_energy_cutoff_keV=_finite_float(
                _configured_value(params, "sem_physical_energy_cutoff_keV"),
                key="sem_physical_energy_cutoff_keV",
                positive=True,
            ),
            physical_elastic_model=physical_elastic_model,
            reference_kernel_path=(
                None
                if _configured_value(params, "sem_reference_kernel_path") is None
                else str(_configured_value(params, "sem_reference_kernel_path"))
            ),
            reference_kernel_sha256=(
                None
                if _configured_value(params, "sem_reference_kernel_sha256") is None
                else str(_configured_value(params, "sem_reference_kernel_sha256"))
            ),
            reference_material=str(_configured_value(params, "sem_reference_material")).strip().lower() or "default",
            reference_geometry=str(_configured_value(params, "sem_reference_geometry")).strip().lower() or "normal",
            reference_source_depth_nm=_finite_float(
                _configured_value(params, "sem_reference_source_depth_nm"),
                key="sem_reference_source_depth_nm",
                nonnegative=True,
            ),
            reference_incident_angle_deg=reference_incident_angle_deg,
            sampling=sampling,
        )

    @staticmethod
    def _positive_int(value: Any, *, key: str) -> int:
        out = int(value)
        if out <= 0:
            raise ValueError(f"parameters['{key}'] must be positive; got {value!r}.")
        return out

    @staticmethod
    def _optional_positive_float(value: Any, *, key: str) -> float | None:
        return None if value is None else _finite_float(value, key=key, positive=True)

    @staticmethod
    def _optional_nonnegative_float(value: Any, *, key: str) -> float | None:
        return None if value is None else _finite_float(value, key=key, nonnegative=True)

    @staticmethod
    def _unit_detector_direction(value: Any) -> tuple[float, float]:
        try:
            x_raw, y_raw = value
        except Exception as exc:
            raise ValueError("parameters['sem_detector_direction_xy'] must be a finite length-2 vector.") from exc
        x = float(x_raw)
        y = float(y_raw)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("parameters['sem_detector_direction_xy'] must be finite.")
        norm = math.sqrt(x * x + y * y)
        if norm <= 0.0:
            raise ValueError("parameters['sem_detector_direction_xy'] must have nonzero norm.")
        return (x / norm, y / norm)

    @property
    def effective_source_representation(self) -> str:
        return self.source_resolution.effective

    @property
    def beam_energy_gain(self) -> float:
        return math.sqrt(max(self.acceleration_kV, 1.0e-9) / 5.0)

    def configured_incident_electrons_per_pixel(self) -> tuple[float, str]:
        electrons = self.electrons_from_beam_current()
        if electrons is not None:
            return electrons, "beam_current_and_dwell_time"
        return self.electrons_per_pixel, "sem_electrons_per_pixel"

    def electrons_from_beam_current(self) -> float | None:
        if self.beam_current_nA <= 0.0 or self.dwell_time_us <= 0.0:
            return None
        charge_c = self.beam_current_nA * 1.0e-9 * self.dwell_time_us * 1.0e-6
        return float(charge_c / 1.602176634e-19)

    def volume_slice_thickness_for_backend(self, backend_name: str | None = None) -> float:
        if self.volume_slice_thickness_nm is not None:
            return self.volume_slice_thickness_nm
        backend = self.backend if backend_name is None else str(backend_name).strip().lower()
        if backend == "monte_carlo_transport":
            base_depth = self.monte_carlo_range_nm if self.monte_carlo_range_nm is not None else self.interaction_volume_nm
        else:
            base_depth = self.interaction_volume_nm
        return max(float(base_depth) / float(self.volume_slices), 1.0e-9)

    def monte_carlo_transport_range_nm(self, canvas_pitch_nm: float) -> float:
        base_range = self.monte_carlo_range_nm if self.monte_carlo_range_nm is not None else self.interaction_volume_nm
        return max(float(base_range) * self.beam_energy_gain, float(canvas_pitch_nm))

    def monte_carlo_step_nm_for_range(self, range_nm: float) -> float:
        if self.monte_carlo_step_nm is not None:
            return self.monte_carlo_step_nm
        return max(float(range_nm) / float(self.monte_carlo_steps), 0.05)

    def monte_carlo_kernel_size_px_for(self, *, canvas_pitch_nm: float, probe_sigma_px: float, range_nm: float) -> int:
        if self.monte_carlo_kernel_size_px is not None:
            return self.monte_carlo_kernel_size_px
        radius_px = int(math.ceil((3.0 * float(probe_sigma_px)) + (1.5 * float(range_nm) / float(canvas_pitch_nm)) + 4.0))
        return max(9, 2 * radius_px + 1)

    def filter_guard_radius_pixels(self, *, probe_sigma_px: float, backend_guard_radius: float | None = None) -> int:
        raw = self.filter_guard_pixels
        if raw is None:
            raw = max(4.0 * float(probe_sigma_px), 2.0)
            if backend_guard_radius is not None:
                raw = max(raw, float(backend_guard_radius))
        if not math.isfinite(float(raw)) or float(raw) < 0.0:
            raise ValueError(f"SEM filter guard radius must be finite and non-negative; got {raw!r}.")
        return int(math.ceil(float(raw)))


@dataclass(frozen=True)
class TemSettings:
    """Owned TEM instrument, source, and multislice settings."""

    model: str
    backend: str
    potential_source: str
    acceleration_kV: float
    spherical_aberration_mm: float
    partial_coherence_alpha_mrad: float
    defocus_nm: float | None
    pixel_size_pm_assertion: float | None
    phase_shift_per_volt_nm_override: float | None
    objective_aperture_mrad: float | None
    slice_thickness_nm: float | None
    projected_potential_scale: float
    filter_guard_pixels: float | None
    reference_status: str
    reference_validation_hash: str | None
    dose_per_pixel: float
    multislice_slices: int
    sampling: SamplingGeometry

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "TemSettings":
        model = str(_configured_value(params, "tem_model")).strip().lower()
        backend = str(_configured_value(params, "tem_backend")).strip().lower()
        allowed_models = {
            "weak_phase_ctf",
            "multislice_lite",
            "syniscopy_multislice",
            "multislice_physical",
        }
        allowed_backends = {
            "ctf_proxy",
            "multislice_lite",
            "syniscopy_multislice",
            "multislice_physical",
        }
        if model not in allowed_models:
            raise ValueError(
                "parameters['tem_model'] must be one of "
                f"{sorted(allowed_models)!r}; got {model!r}."
            )
        if backend not in allowed_backends:
            raise ValueError(
                "parameters['tem_backend'] must be one of "
                f"{sorted(allowed_backends)!r}; got {backend!r}."
            )
        required_backend_by_model = {
            "weak_phase_ctf": "ctf_proxy",
            "multislice_lite": "multislice_lite",
            "syniscopy_multislice": "syniscopy_multislice",
            "multislice_physical": "multislice_physical",
        }
        required_backend = required_backend_by_model[model]
        if backend != required_backend:
            raise ValueError(
                f"parameters['tem_model']={model!r} requires "
                f"parameters['tem_backend']={required_backend!r}; got {backend!r}."
            )
        potential_source = str(_configured_value(params, "tem_potential_source")).strip().lower()
        allowed_sources = {
            "material_projected_inner_potential",
            "sample_environment_projected_potential",
            "material_plus_sample_environment",
        }
        if potential_source not in allowed_sources:
            raise ValueError(
                "parameters['tem_potential_source'] must be one of "
                f"{sorted(allowed_sources)!r}; got {potential_source!r}."
            )
        multislice_slices = cls._positive_int(
            _configured_value(params, "tem_multislice_slices"),
            key="tem_multislice_slices",
        )
        reference_status = str(_configured_value(params, "tem_reference_status")).strip().lower()
        if reference_status not in {"physics_based_unvalidated", "reference_validated"}:
            raise ValueError(
                "parameters['tem_reference_status'] must be 'physics_based_unvalidated' "
                f"or 'reference_validated'; got {reference_status!r}."
            )
        reference_hash_raw = _configured_value(params, "tem_reference_validation_hash")
        reference_hash = None if reference_hash_raw is None else str(reference_hash_raw)
        if reference_status == "reference_validated" and not reference_hash:
            raise ValueError(
                "parameters['tem_reference_status']='reference_validated' requires "
                "parameters['tem_reference_validation_hash']."
            )
        return cls(
            model=model,
            backend=backend,
            potential_source=potential_source,
            acceleration_kV=_finite_float(
                _configured_value(params, "tem_acceleration_kV"),
                key="tem_acceleration_kV",
                positive=True,
            ),
            spherical_aberration_mm=_finite_float(
                _configured_value(params, "tem_Cs_mm"),
                key="tem_Cs_mm",
                nonnegative=True,
            ),
            partial_coherence_alpha_mrad=_finite_float(
                _configured_value(params, "tem_partial_coherence_alpha_mrad"),
                key="tem_partial_coherence_alpha_mrad",
                nonnegative=True,
            ),
            defocus_nm=cls._optional_float(_configured_value(params, "tem_defocus_nm"), key="tem_defocus_nm"),
            pixel_size_pm_assertion=cls._optional_positive_float(
                _configured_value(params, "tem_pixel_size_pm"),
                key="tem_pixel_size_pm",
            ),
            phase_shift_per_volt_nm_override=cls._optional_nonnegative_float(
                _configured_value(params, "tem_phase_shift_per_volt_nm"),
                key="tem_phase_shift_per_volt_nm",
            ),
            objective_aperture_mrad=cls._optional_positive_float(
                _configured_value(params, "tem_objective_aperture_mrad"),
                key="tem_objective_aperture_mrad",
            ),
            slice_thickness_nm=cls._optional_positive_float(
                _configured_value(params, "tem_slice_thickness_nm"),
                key="tem_slice_thickness_nm",
            ),
            projected_potential_scale=_finite_float(
                _configured_value(params, "tem_projected_potential_scale"),
                key="tem_projected_potential_scale",
                nonnegative=True,
            ),
            filter_guard_pixels=cls._optional_nonnegative_float(
                _configured_value(params, "tem_filter_guard_pixels"),
                key="tem_filter_guard_pixels",
            ),
            reference_status=reference_status,
            reference_validation_hash=reference_hash,
            dose_per_pixel=CountBudgetSettings.from_params(params).tem_dose_per_pixel,
            multislice_slices=multislice_slices,
            sampling=SamplingGeometry.from_params(params),
        )

    @staticmethod
    def _positive_int(value: Any, *, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"parameters['{key}'] must be a positive integer; got {value!r}.")
        out = int(value)
        if out <= 0:
            raise ValueError(f"parameters['{key}'] must be a positive integer; got {value!r}.")
        return out

    @staticmethod
    def _optional_float(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        return _finite_float(value, key=key)

    @staticmethod
    def _optional_positive_float(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        return _finite_float(value, key=key, positive=True)

    @staticmethod
    def _optional_nonnegative_float(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        return _finite_float(value, key=key, nonnegative=True)

    @property
    def electron_wavelength_m(self) -> float:
        return electron_wavelength_m(self.acceleration_kV)

    @property
    def defocus_m(self) -> float:
        if self.defocus_nm is not None:
            return 1.0e-9 * float(self.defocus_nm)
        return scherzer_defocus_m(self.acceleration_kV, self.spherical_aberration_mm)

    @property
    def phase_shift_per_volt_nm(self) -> float:
        if self.phase_shift_per_volt_nm_override is not None:
            return float(self.phase_shift_per_volt_nm_override)
        return electron_interaction_parameter_rad_per_V_nm(self.acceleration_kV)

    def assert_canvas_pixel_pitch(self) -> None:
        if self.pixel_size_pm_assertion is None:
            return
        requested_m = 1.0e-12 * float(self.pixel_size_pm_assertion)
        actual_m = 1.0e-9 * float(self.sampling.model_canvas_pixel_size_nm)
        if (
            not math.isfinite(requested_m)
            or requested_m <= 0.0
            or not math.isclose(requested_m, actual_m, rel_tol=1e-6, abs_tol=1e-15)
        ):
            raise ValueError(
                "parameters['tem_pixel_size_pm'] must match the rendered model-canvas "
                "pitch pixel_size_nm / psf_oversampling_factor. "
                f"Got tem_pixel_size_pm={self.pixel_size_pm_assertion} pm and "
                f"canvas pitch={self.sampling.model_canvas_pixel_size_nm * 1000.0:.6g} pm."
            )

    def filter_guard_radius_pixels(self, *, automatic_guard_radius: int) -> int:
        if self.filter_guard_pixels is None:
            return int(automatic_guard_radius)
        guard = float(self.filter_guard_pixels)
        if not math.isfinite(guard) or guard < 0.0:
            raise ValueError(
                "parameters['tem_filter_guard_pixels'] must be non-negative and finite; "
                f"got {self.filter_guard_pixels!r}."
            )
        return int(math.ceil(guard))


@dataclass(frozen=True)
class MicroscopeRuntimeSettings:
    """Resolved runtime settings for one concrete microscope candidate."""

    modality: ModalitySettings
    acquisition: AcquisitionProfile
    sampling: SamplingGeometry
    count_budget: CountBudgetSettings
    detector: DetectorSettings
    detector_readout: DetectorReadoutSettings
    optical_instrument: OpticalInstrumentSettings | None = None
    optical_mode: OpticalModeSettings | None = None
    vectorial_optics: VectorialOpticsSettings | None = None
    modality_settings: Any | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "MicroscopeRuntimeSettings":
        modality = ModalitySettings.from_params(params)
        modality_name = modality.modality
        acquisition = AcquisitionProfile.from_params(params)
        sampling = SamplingGeometry.from_params(params)
        count_budget = CountBudgetSettings.from_params(params)
        detector = DetectorSettings.from_params(
            params,
            fluorescence=is_fluorescence_modality(modality_name),
        )
        detector_readout = DetectorReadoutSettings.from_params(params)

        if modality_name == "sem_secondary_electron":
            return cls(
                modality=modality,
                acquisition=acquisition,
                sampling=sampling,
                count_budget=count_budget,
                detector=detector,
                detector_readout=detector_readout,
                modality_settings=SemSettings.from_params(params),
            )
        if modality_name == "tem_phase_contrast":
            return cls(
                modality=modality,
                acquisition=acquisition,
                sampling=sampling,
                count_budget=count_budget,
                detector=detector,
                detector_readout=detector_readout,
                modality_settings=TemSettings.from_params(params),
            )

        optical_instrument = OpticalInstrumentSettings.from_params(params)
        optical_mode = OpticalModeSettings.from_params(params)
        vectorial_optics = VectorialOpticsSettings.from_params(params)
        modality_settings: Any | None = None

        if modality_name in {"bright_field", "partially_coherent_bright_field"}:
            modality_settings = KohlerBrightFieldSettings.from_params(params)
        elif modality_name == "dark_field":
            DarkFieldSettings.from_params(params)
            modality_settings = AnnularDarkFieldSettings.from_params(params)
        elif modality_name == "differential_phase_contrast":
            modality_settings = DpcSettings.from_params(params)
        elif modality_name == "quantitative_phase":
            modality_settings = QpiReadoutSettings.from_params(params)
        elif modality_name == "interferometric":
            modality_settings = IscatSettings.from_params(params)
        elif modality_name == "ricm":
            modality_settings = RicmSettings.from_params(params)
        elif modality_name == "fluorescence_widefield":
            modality_settings = FluorescenceSettings.from_params(params)
        elif modality_name == "tirf_fluorescence":
            modality_settings = TirfSettings.from_params(params)
        elif modality_name in {
            "coherent_bright_field",
            "coherent_dark_field",
            "off_axis_holography",
            "zernike_phase_contrast",
        }:
            modality_settings = None
        else:
            raise ValueError(f"Unsupported microscope runtime modality {modality_name!r}.")

        return cls(
            modality=modality,
            acquisition=acquisition,
            sampling=sampling,
            count_budget=count_budget,
            detector=detector,
            detector_readout=detector_readout,
            optical_instrument=optical_instrument,
            optical_mode=optical_mode,
            vectorial_optics=vectorial_optics,
            modality_settings=modality_settings,
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
        acquisition = AcquisitionProfile.from_params(params)
        sampling = SamplingGeometry.from_params(params)
        detector_readout = DetectorReadoutSettings.from_params(params)
        mask_settings = MaskGenerationSettings.from_params(params)
        motion_blur_subsamples = int(_configured_value(params, "motion_blur_subsamples"))
        if motion_blur_subsamples <= 0:
            raise ValueError("parameters['motion_blur_subsamples'] must be positive.")
        reference_field_amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        background_intensity = CountBudgetSettings.from_params(params).background_intensity

        return cls(
            fps=acquisition.fps,
            random_seed=acquisition.random_seed,
            image_size_pixels=sampling.image_size_pixels,
            pixel_size_nm=sampling.detector_pixel_size_nm,
            psf_oversampling_factor=sampling.psf_oversampling_factor,
            exposure_time_ms=acquisition.exposure_time_ms,
            mask_generation_enabled=mask_settings.enabled,
            mask_output_directory=mask_settings.output_directory,
            bit_depth=detector_readout.bit_depth,
            motion_blur_enabled=bool(_configured_value(params, "motion_blur_enabled")),
            motion_blur_subsamples=motion_blur_subsamples,
            return_ideal_float_frames=bool(_configured_value(params, "return_ideal_float_frames")),
            reference_field_amplitude=reference_field_amplitude,
            background_intensity=background_intensity,
        )


__all__ = [
    "AcquisitionProfile",
    "AnnularDarkFieldSettings",
    "BackgroundSubtractionSettings",
    "BackendProfileSettings",
    "CountBudgetSettings",
    "CoverslipAberrationSettings",
    "DarkFieldSettings",
    "DetectorSettings",
    "DetectorReadoutSettings",
    "DpcSettings",
    "EmpiricalBackgroundSettings",
    "FisherAnalysisSettings",
    "FluorescenceSettings",
    "FocusPlaneState",
    "IscatSettings",
    "KohlerBrightFieldSettings",
    "MatchedMicroscopeSettings",
    "MaskGenerationSettings",
    "MicroscopeRuntimeSettings",
    "ModalitySettings",
    "MotionDynamicsSettings",
    "OffAxisHolographySettings",
    "OpticalInstrumentSettings",
    "OpticalModeSettings",
    "OpticalPsfGridSettings",
    "OpticalPsfSupportSettings",
    "OpticalScatteringSettings",
    "QpiReadoutSettings",
    "RicmSettings",
    "RenderRuntimeConfig",
    "SamplingGeometry",
    "SampleEnvironmentRoughnessSettings",
    "SampleEnvironmentSettings",
    "SemSettings",
    "SimulationOutputSettings",
    "SpectralIntegrationSettings",
    "SupervisionGeometryThresholds",
    "SupervisionSettings",
    "TemSettings",
    "TirfSettings",
    "UnitContractSettings",
    "VectorialOpticsSettings",
    "VolumeRenderingSettings",
    "ZernikePhaseSettings",
    "AberrationSettings",
]
