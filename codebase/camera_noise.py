"""
camera_noise.py - Counts-domain camera-noise model for Syniscopy.

The user-facing image level is camera counts/ADU. The model keeps shot noise
in physical count units without making users tune an unobserved photon/electron
budget:

    signal_counts -> signal_electrons = signal_counts * camera_gain_e_per_count
    Poisson sample in electrons
    convert back to counts

camera_gain_e_per_count is a camera-conversion/calibration parameter. For a
calibrated reproduction workflow it should be estimated from the real video or
left at the documented default; it is not treated as a free signal-fitting knob.
"""

from __future__ import annotations
from configured_parameters import configured_optional, configured_value

from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
from threading import RLock
from typing import Any, Mapping

import numpy as np

from array_representation import (
    ArrayRepresentation,
    COORD_DETECTOR_XY,
    DOMAIN_CAMERA_COUNT,
    DOMAIN_ELECTRON_COUNT,
    DOMAIN_INCIDENT_QUANTA,
    STAGE_DETECTOR_INPUT,
    VALUE_ABSOLUTE,
)
from config.runtime import (
    AcquisitionProfile,
    BackgroundSubtractionSettings,
    CountBudgetSettings,
    DetectorSettings,
    ModalitySettings,
    QpiReadoutSettings,
    SampleEnvironmentSettings,
)
from measurement_units import normalize_detector_noise_input_domain
from modality_registry import (
    is_electron_modality,
    is_fluorescence_modality,
    modality_uses_relative_reference_contrast,
    require_modality_name,
)
from noise_contracts import AnalysisNoiseModel, PhaseLikelihoodBasis, phase_variance_from_likelihood_basis
from shared_constants import (
    RAW_BACKGROUND_SUBTRACTION_METHODS,
    REFERENCE_BACKGROUND_SUBTRACTION_METHODS,
    VIDEO_BACKGROUND_SUBTRACTION_METHODS,
)
from simulation_runtime_state import runtime_state, runtime_state_or_default
from stochastic_runtime import derive_seed, rng_from_seed, rng_from_seed_words
from unit_contracts import assert_compatible


def _default_detector_noise_rng() -> np.random.Generator:
    return rng_from_seed(None, stream="detector_noise_runtime")


@dataclass
class DetectorNoiseRuntime:
    """Run-scoped detector-noise state: random source plus resolved map caches."""

    rng: Any = field(default_factory=_default_detector_noise_rng)
    static_detector_map_cache: dict[tuple, dict[str, np.ndarray]] = field(default_factory=dict)
    camera_noise_map_cache: dict[tuple[str, tuple[int, ...], Any], np.ndarray] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)

    def clear_static_detector_maps(self) -> None:
        with self.lock:
            self.static_detector_map_cache.clear()

    def clear_noise_maps(self) -> None:
        with self.lock:
            self.camera_noise_map_cache.clear()

    def clear(self) -> None:
        with self.lock:
            self.static_detector_map_cache.clear()
            self.camera_noise_map_cache.clear()


_DEFAULT_DETECTOR_NOISE_RUNTIME = DetectorNoiseRuntime()


def _noise_runtime(runtime: DetectorNoiseRuntime | None = None) -> DetectorNoiseRuntime:
    return runtime if runtime is not None else _DEFAULT_DETECTOR_NOISE_RUNTIME


def detector_noise_runtime_from_seed(seed: int | None = None, *, rng: Any | None = None) -> DetectorNoiseRuntime:
    """Return an isolated detector-noise runtime for deterministic direct calls."""
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng to detector_noise_runtime_from_seed(), not both.")
    resolved_rng = rng if rng is not None else rng_from_seed(seed, stream="detector_noise_runtime")
    return DetectorNoiseRuntime(rng=resolved_rng)


def clear_detector_static_noise_cache(runtime: DetectorNoiseRuntime | None = None) -> None:
    """Clear cached fixed-pattern/hot-pixel maps for long dataset runs."""
    _noise_runtime(runtime).clear_static_detector_maps()


@dataclass(frozen=True)
class CameraNoiseConfig:
    shot_noise_enabled: bool
    gaussian_noise_enabled: bool
    camera_gain_e_per_count: float
    detector_qe: float
    detector_input_is_incident_quanta: bool
    detector_noise_input_domain: str
    emccd_enabled: bool
    emccd_gain: float
    emccd_excess_noise_factor: float
    emccd_excess_noise_factor_basis: str
    emccd_shot_variance_multiplier: float
    read_noise_e: float | None
    dark_current_e_per_pixel_per_s: float
    exposure_time_s: float
    saturation_level: float | None
    saturation_e: float | None
    adc_quantization: bool
    adc_quantization_counts: float
    background_offset_counts: float
    background_offset_stage: str
    read_noise_counts: float
    dark_offset_counts: float
    dark_offset_stage: str
    fixed_pattern_gain_std: float
    fixed_pattern_offset_counts: float
    hot_pixel_fraction: float
    hot_pixel_value_counts: float | None
    fixed_pattern_gain_map: str | None
    fixed_pattern_offset_map: str | None
    scmos_gain_map: str | None
    scmos_variance_map: str | None
    scmos_read_noise_map: str | None
    read_noise_map_mode: str
    hot_pixel_mask: str | None
    nonlinearity_calibration: str | None
    flat_field_map: str | None
    dark_frame_map: str | None
    dark_frame_map_stage: str
    scan_line_noise_counts: float
    clip_output_to_nonnegative: bool
    noise_parameterization: str
    nonlinear_detector_effects_active: bool
    deterministic_detector_transfer_active: bool
    safe_for_linear_fisher_variance: bool

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> "CameraNoiseConfig":
        return _camera_noise_config_from_params(params)

    @property
    def generated_static_detector_maps_requested(self) -> bool:
        """Return whether stochastic fixed-pattern/hot-pixel maps are needed."""

        return bool(
            self.fixed_pattern_gain_std > 0.0
            or self.fixed_pattern_offset_counts > 0.0
            or self.hot_pixel_fraction > 0.0
        )


DETECTOR_FRAME_REPRESENTATION_CONTRACT_ID = "syniscopy-detector-frame-representation-v1"

DETECTOR_QE_OWNER_DETECTOR_NOISE_MODEL = "detector_noise_model"
DETECTOR_QE_OWNER_RENDERER_FLUORESCENCE_COUNT_SCALE = "renderer_fluorescence_count_scale"
DETECTOR_QE_OWNER_ELECTRON_BACKEND = "electron_backend"
DETECTOR_QE_OWNER_NOT_APPLICABLE = "not_applicable"
DETECTOR_QE_OWNER_ALREADY_COUNT_DOMAIN = "already_count_domain"


@dataclass(frozen=True)
class DetectorFrameRepresentationContract:
    """Resolved detector-frame representation consumed by stochastic and Fisher noise."""

    representation: ArrayRepresentation
    detector_qe_owner: str
    detector_input_is_incident_quanta: bool
    detector_noise_input_domain: str
    modality: str
    contract_id: str = DETECTOR_FRAME_REPRESENTATION_CONTRACT_ID
    resolution: str = ""

    @property
    def frame_basis(self) -> str:
        return self.representation.semantic_label or self.representation.domain

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "representation": asdict(self.representation),
            "frame_basis": self.frame_basis,
            "detector_qe_owner": self.detector_qe_owner,
            "detector_input_is_incident_quanta": self.detector_input_is_incident_quanta,
            "detector_noise_input_domain": self.detector_noise_input_domain,
            "modality": self.modality,
            "contract_id": self.contract_id,
            "resolution": self.resolution,
        }
        payload.update(self.representation.metadata(prefix="detector_frame_array"))
        return payload


def _detector_frame_representation(domain: str, *, units: str, label: str) -> ArrayRepresentation:
    return ArrayRepresentation(
        domain=domain,
        value_form=VALUE_ABSOLUTE,
        units=units,
        coordinate_frame=COORD_DETECTOR_XY,
        pipeline_stage=STAGE_DETECTOR_INPUT,
        semantic_label=label,
    )


def _response_function_from_render_metadata(render_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(render_metadata or {})
    response = metadata.get("response_function")
    return dict(response) if isinstance(response, Mapping) else {}


def _fluorescence_renderer_owns_detector_qe(
    params: dict[str, Any],
    render_metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when the producer already converted fluorescence photons to counts."""

    modality = ModalitySettings.from_params(params).modality
    if is_fluorescence_modality(modality):
        return True
    response = _response_function_from_render_metadata(render_metadata)
    return bool(
        response.get("fluorescence_absolute_scale") == "physical_absorbed_excitation_photon_budget"
        or response.get("fluorescence_background_units") == "detected_counts_per_pixel"
    )


def _detector_input_is_incident_quanta_from_config(params: dict[str, Any], *, context: str) -> bool:
    value = _cfg_value(params, "detector_input_is_incident_quanta")
    if not isinstance(value, bool):
        raise ValueError(f"{context}: detector_input_is_incident_quanta must be boolean; got {value!r}.")
    return bool(value)


def resolve_detector_frame_representation_contract(
    params: dict[str, Any] | None = None,
    *,
    render_metadata: Mapping[str, Any] | None = None,
    context: str = "detector frame basis",
) -> DetectorFrameRepresentationContract:
    """Resolve the unique detector-frame representation before any QE/noise transfer.

    This is the cross-seam contract for renderer output, detector noise, and
    Fisher/CRLB likelihoods.  Fluorescence/TIRF renderers already multiply the
    absorbed-excitation photon budget by quantum yield, collection efficiency,
    detector QE, and detector pixel area.  Their detector-input frames are
    therefore detected camera counts.  Letting a nested noise override mark the
    same frames as incident quanta would apply QE again and corrupt the Fisher
    variance, so that illegal state is rejected here before any consumer can
    build stochastic frames, deterministic detector means, or analysis noise.
    """

    local = dict(params or {})
    modality = ModalitySettings.from_params(local).modality
    detector_noise_input_domain = _detector_noise_input_domain_from_config(local)
    incident = _detector_input_is_incident_quanta_from_config(local, context=context)

    if detector_noise_input_domain == "electron_count":
        return DetectorFrameRepresentationContract(
            representation=_detector_frame_representation(
                DOMAIN_ELECTRON_COUNT,
                units="electron_count",
                label="electron_count",
            ),
            detector_qe_owner=DETECTOR_QE_OWNER_ELECTRON_BACKEND,
            detector_input_is_incident_quanta=False,
            detector_noise_input_domain=detector_noise_input_domain,
            modality=modality,
            resolution="electron_count_input_domain_disables_incident_photon_qe_transfer",
        )

    if _fluorescence_renderer_owns_detector_qe(local, render_metadata):
        if incident:
            raise ValueError(
                f"{context}: detector-frame basis contract violation for modality {modality!r}. "
                "Fluorescence/TIRF renderers output detected camera counts because the "
                "fluorescence photon budget has already been multiplied by detector QE. "
                "Do not set detector_input_is_incident_quanta=True in the effective "
                "top-level, noise_model, or modality_noise configuration for these frames."
            )
        return DetectorFrameRepresentationContract(
            representation=_detector_frame_representation(
                DOMAIN_CAMERA_COUNT,
                units="detector_count",
                label="camera_counts",
            ),
            detector_qe_owner=DETECTOR_QE_OWNER_RENDERER_FLUORESCENCE_COUNT_SCALE,
            detector_input_is_incident_quanta=False,
            detector_noise_input_domain=detector_noise_input_domain,
            modality=modality,
            resolution="fluorescence_renderer_outputs_detected_counts_after_qe",
        )

    if incident:
        return DetectorFrameRepresentationContract(
            representation=_detector_frame_representation(
                DOMAIN_INCIDENT_QUANTA,
                units="quanta",
                label="incident_quanta",
            ),
            detector_qe_owner=DETECTOR_QE_OWNER_DETECTOR_NOISE_MODEL,
            detector_input_is_incident_quanta=True,
            detector_noise_input_domain=detector_noise_input_domain,
            modality=modality,
            resolution="incident_quanta_input_requires_detector_noise_qe_transfer",
        )

    return DetectorFrameRepresentationContract(
        representation=_detector_frame_representation(
            DOMAIN_CAMERA_COUNT,
            units="detector_count",
            label="camera_counts",
        ),
        detector_qe_owner=DETECTOR_QE_OWNER_ALREADY_COUNT_DOMAIN,
        detector_input_is_incident_quanta=False,
        detector_noise_input_domain=detector_noise_input_domain,
        modality=modality,
        resolution="count_domain_input_no_detector_qe_transfer",
    )


def canonicalize_detector_frame_noise_params(
    params: dict[str, Any] | None = None,
    *,
    render_metadata: Mapping[str, Any] | None = None,
    context: str = "detector frame basis",
) -> dict[str, Any]:
    """Return params whose nested noise overrides obey the detector-frame contract."""

    out = dict(params or {})
    contract = resolve_detector_frame_representation_contract(
        out,
        render_metadata=render_metadata,
        context=context,
    )
    if not contract.detector_input_is_incident_quanta:
        # Store the resolved compatibility flag in every active noise container.
        # This prevents old call sites that still pass whole parameters dictionaries
        # from bypassing the shared basis contract through nested modality_noise.
        out["detector_input_is_incident_quanta"] = False
        noise_model = dict(out.get("noise_model", {}) or {})
        noise_model["detector_input_is_incident_quanta"] = False
        out["noise_model"] = noise_model
        modality_noise = dict(out.get("modality_noise", {}) or {})
        modality = contract.modality
        for raw_key, value in list(modality_noise.items()):
            if not isinstance(value, dict):
                continue
            canonical = require_modality_name(
                raw_key,
                item_label="parameters['modality_noise'] modality key",
            )
            if canonical == modality:
                override = dict(value)
                override["detector_input_is_incident_quanta"] = False
                modality_noise[raw_key] = override
        out["modality_noise"] = modality_noise
    return out


DETECTOR_TRANSFER_STAGE_CONTRACT_ID = "detector-transfer-stages-v1"
DETECTOR_ADDITIVE_STAGE_PRE_POISSON = "pre_poisson"
DETECTOR_ADDITIVE_STAGE_POST_POISSON_PRE_GAIN = "post_poisson_pre_gain"
DETECTOR_ADDITIVE_STAGE_POST_GAIN = "post_gain"
DETECTOR_ADDITIVE_STAGES = {
    DETECTOR_ADDITIVE_STAGE_PRE_POISSON,
    DETECTOR_ADDITIVE_STAGE_POST_POISSON_PRE_GAIN,
    DETECTOR_ADDITIVE_STAGE_POST_GAIN,
}
EMCCD_EXCESS_NOISE_BASIS_CONTRACT_ID = "emccd-excess-noise-basis-v1"
EMCCD_EXCESS_NOISE_FACTOR_BASIS_VARIANCE_MULTIPLIER = "variance_multiplier"
EMCCD_EXCESS_NOISE_FACTOR_BASIS_NOISE_FACTOR_STD = "noise_factor_std"
EMCCD_EXCESS_NOISE_FACTOR_BASES = {
    EMCCD_EXCESS_NOISE_FACTOR_BASIS_VARIANCE_MULTIPLIER,
    EMCCD_EXCESS_NOISE_FACTOR_BASIS_NOISE_FACTOR_STD,
}
QPI_PHASE_LIKELIHOOD_CONTRACT_ID = "qpi-phase-likelihood-v1"
QPI_DETECTED_QUANTA_MAP_PARAM = "_qpi_detected_quanta_per_pixel_map"
QPI_PHASE_LIKELIHOOD_PROVENANCE_PARAM = "_qpi_phase_likelihood_provenance"
QPI_PHASE_LIKELIHOOD_CONTRACT_PARAM = "_qpi_phase_likelihood_contract_id"
ANALYSIS_NOISE_PARAMETER_FRAMES_KEY = "analysis_noise_parameter_frames"


@dataclass(frozen=True)
class DetectorTransferStages:
    """Detector-transfer stages that must not be collapsed across noise seams."""

    poisson_driver_mean_counts: np.ndarray
    post_poisson_multiplicative_gain: np.ndarray | float
    pre_poisson_additive_counts: np.ndarray
    post_poisson_pre_gain_additive_counts: np.ndarray
    post_gain_additive_counts: np.ndarray
    additive_stage_by_source: dict[str, str]
    contract_id: str = DETECTOR_TRANSFER_STAGE_CONTRACT_ID


def _normalise_noise_key(name: Any) -> str:
    return str(name).strip().lower()


def normalize_detector_additive_stage(
    value: Any,
    *,
    context: str = "detector additive offset stage",
) -> str:
    """Return the stage where an additive count-domain detector term belongs."""

    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if raw not in DETECTOR_ADDITIVE_STAGES:
        raise ValueError(
            f"{context} must be one of {sorted(DETECTOR_ADDITIVE_STAGES)}; "
            f"got {value!r}."
        )
    return raw


def normalize_emccd_excess_noise_factor_basis(
    value: Any,
    *,
    context: str = "emccd_excess_noise_factor_basis",
) -> str:
    """Return the declared basis for the public EMCCD excess-noise factor."""

    basis = str(value).strip().lower()
    if basis not in EMCCD_EXCESS_NOISE_FACTOR_BASES:
        raise ValueError(
            f"{context} must be one of {sorted(EMCCD_EXCESS_NOISE_FACTOR_BASES)}; "
            f"got {value!r}."
        )
    return basis


def emccd_shot_variance_multiplier_from_factor(
    factor: float,
    basis: Any = EMCCD_EXCESS_NOISE_FACTOR_BASIS_VARIANCE_MULTIPLIER,
) -> float:
    """Convert a public EMCCD excess-noise input to the internal variance multiplier."""

    raw = float(factor)
    if not np.isfinite(raw) or raw < 1.0:
        raise ValueError(f"emccd_excess_noise_factor must be finite and >= 1; got {factor}.")
    canonical_basis = normalize_emccd_excess_noise_factor_basis(basis)
    # The renderer/Fisher seam consumes only a shot-variance multiplier. Vendor
    # EMCCD noise factors are standard-deviation factors, so they enter variance
    # as F_n**2; the canonical variance-multiplier basis is already in the seam's
    # units.
    multiplier = raw if canonical_basis == EMCCD_EXCESS_NOISE_FACTOR_BASIS_VARIANCE_MULTIPLIER else raw * raw
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError(
            "Resolved EMCCD shot-variance multiplier must be finite and >= 1; "
            f"got {multiplier!r} from factor={factor!r}, basis={canonical_basis!r}."
        )
    return float(multiplier)


def _explicit_noise_input_domain(noise_model: dict[str, Any]) -> Any | None:
    if "detector_noise_input_domain" in noise_model:
        return noise_model["detector_noise_input_domain"]
    return None


def _detector_noise_input_domain_from_config(params: dict[str, Any]) -> str:
    noise_model = _effective_noise_overrides(params)
    explicit = _explicit_noise_input_domain(noise_model)
    source = "canonical modality default"
    if explicit is not None:
        domain = explicit
        source = "effective noise_model/modality_noise override"
    else:
        configured = configured_value(params, "detector_noise_input_domain")
        if configured is not None:
            domain = configured
            source = "top-level detector_noise_input_domain"
        else:
            modality = ModalitySettings.from_params(params).modality
            domain = "electron_count" if is_electron_modality(modality) else "camera_counts"
    normalized = normalize_detector_noise_input_domain(domain)
    modality = ModalitySettings.from_params(params).modality
    if is_electron_modality(modality) and normalized != "electron_count":
        # Electron renderers expose detector-input means directly in electron-count
        # units. Reinterpreting those arrays as camera counts would apply the
        # camera gain in the Fisher variance and can change candidate rankings.
        raise ValueError(
            "detector-frame basis contract violation for electron modality "
            f"{modality!r}: {source} resolved detector_noise_input_domain="
            f"{normalized!r}, but TEM/SEM renderer outputs are electron-count "
            "frames. Use detector_noise_input_domain='electron_count' or omit "
            "the override; a future camera-count electron backend must declare "
            "and perform an explicit frame-basis conversion instead of "
            "reinterpreting electron-count arrays."
        )
    return normalized


def _map_cache_key(kind: str, raw: Any) -> tuple[str, tuple[int, ...], Any]:
    if isinstance(raw, (str, Path)):
        path = Path(raw).expanduser()
        try:
            stat = path.stat()
            mtime = int(stat.st_mtime)
            size = int(stat.st_size)
        except OSError:
            mtime = 0
            size = 0
        return (kind, (0,), (str(path), size, mtime,))
    if isinstance(raw, np.ndarray):
        fingerprint = hashlib.blake2b(raw.tobytes(), digest_size=8).hexdigest()
        return (kind, tuple(raw.shape), (raw.dtype.str, fingerprint,))
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw)
        fingerprint = hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()
        return (kind, tuple(arr.shape), (arr.dtype.str, fingerprint,))
    return (kind, tuple(), (type(raw).__name__, raw,))


def _load_map_array(path: str) -> np.ndarray:
    path_obj = Path(path).expanduser()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Noise map file not found: {path_obj}")
    ext = path_obj.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path_obj))
        return np.asarray(arr)
    if ext == ".npz":
        with np.load(str(path_obj)) as data:
            if len(data.files) == 0:
                raise ValueError(f"Noise map .npz archive is empty: {path_obj}")
            arr = data[data.files[0]]
        return np.asarray(arr)
    if ext in {".json", ".jsn"}:
        raw = json.loads(path_obj.read_text(encoding="utf-8"))
        return np.asarray(raw)
    try:
        return np.loadtxt(str(path_obj))
    except ValueError as exc:
        raise ValueError(f"Noise map file format not supported: {path_obj}") from exc


def _coerce_map_value(
    raw: Any,
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | float | int | bool | None, str]:
    if raw is None:
        return None, ""
    if isinstance(raw, (str, bytes, Path)):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        text = str(raw).strip()
    else:
        text = None
    if text is not None:
        if not text:
            raise TypeError(f"Unsupported detector noise map value type: {type(raw)!r}.")
        path = Path(text).expanduser()
        if path.exists():
            cache_key = _map_cache_key("file", path)
            active_runtime = _noise_runtime(runtime)
            with active_runtime.lock:
                if cache_key not in active_runtime.camera_noise_map_cache:
                    active_runtime.camera_noise_map_cache[cache_key] = np.asarray(_load_map_array(str(path)), dtype=dtype)
                return active_runtime.camera_noise_map_cache[cache_key], str(path)
        try:
            return np.array(float(text), dtype=dtype), "inline_scalar"
        except ValueError as exc:
            raise FileNotFoundError(f"Noise map file not found and not parseable as a scalar: {path}") from exc
    if isinstance(raw, bool):
        return np.array(float(raw), dtype=float), "inline"
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return np.array(float(raw), dtype=dtype), "inline_scalar"
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=dtype), "inline_array"
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=dtype)
        key = _map_cache_key("inline", arr)
        active_runtime = _noise_runtime(runtime)
        with active_runtime.lock:
            if key not in active_runtime.camera_noise_map_cache:
                active_runtime.camera_noise_map_cache[key] = arr
            return active_runtime.camera_noise_map_cache[key], f"inline_array{arr.shape}"
    raise TypeError(f"Unsupported detector noise map value type: {type(raw)!r}.")


def _resolve_map(
    raw: Any,
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | None, str]:
    value, summary = _coerce_map_value(raw, dtype=dtype, runtime=runtime)
    if value is None:
        return None, ""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr), summary
    return np.asarray(arr, dtype=dtype), summary


def _resolve_boolean_mask(
    name: str,
    raw: Any,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | None, str]:
    if raw is None:
        return None, ""
    if isinstance(raw, (str, bytes, Path)):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        arr = _coerce_map_value(raw, dtype=float, runtime=runtime)[0]
        if np.asarray(arr).ndim == 0:
            return np.asarray(bool(np.asarray(arr) > 0.5)), str(raw)
        return np.asarray(arr, dtype=float) > 0.5, str(raw)
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.asarray(bool(arr > 0.5)), f"inline_array{arr.shape}"
        return (arr > 0.5), f"inline_array{arr.shape}"
    if isinstance(raw, (int, float, bool, np.integer, np.floating)):
        value = float(raw)
        if not np.isfinite(value):
            raise TypeError(f"Unsupported detector noise boolean map value type: {type(raw)!r}.")
        return np.asarray(value > 0.5), "inline_scalar"
    if np.asarray(raw).ndim == 0:
        return np.asarray(float(np.asarray(raw))) > 0.5, "inline_scalar"
    raise TypeError(f"Unsupported detector noise boolean map value type: {type(raw)!r}.")


def _resolve_map_shape(
    name: str,
    raw: Any,
    shape: tuple[int, ...],
    *,
    dtype: Any = float,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray | float | None, str]:
    value, summary = _resolve_map(raw, dtype=dtype, runtime=runtime)
    if value is None:
        return None, summary
    arr = np.asarray(value)
    if not shape:
        if arr.ndim == 0:
            return float(arr), summary
        return float(np.mean(arr)), summary
    if arr.ndim > 0 and tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"{name} map shape mismatch: expected {tuple(shape)} but got {tuple(arr.shape)}. "
            "Respecify the map to match the frame shape or supply a scalar override."
        )
    if np.any(~np.isfinite(np.asarray(arr, dtype=float))):
        raise ValueError(f"{name} map must contain finite values; got non-finite entries.")
    return value, summary


def _map_to_array(value: np.ndarray | float | int, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    return arr


def _mask_to_array(value: np.ndarray | bool, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(shape, bool(arr), dtype=bool)
    return arr.astype(bool)


def _effective_noise_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Return the effective camera-noise configuration for this modality.

    Precedence is: base params < configured parameters["noise_model"] <
    configured parameters["modality_noise"][imaging_model]. Only noise_model and
    modality_noise are public configuration containers.
    """
    cfg: dict[str, Any] = {}
    cfg.update(noise_model_overrides_from_params(params))

    modality_key = _normalise_noise_key(ModalitySettings.from_params(params).modality)
    per_modality = modality_noise_overrides_from_params(params)
    if isinstance(per_modality, dict) and modality_key:
        seen_modality_keys: set[str] = set()
        for raw_key, override in per_modality.items():
            raw_norm = _normalise_noise_key(
                require_modality_name(
                    raw_key,
                    item_label="parameters['modality_noise'] modality key",
                )
            )
            if raw_norm in seen_modality_keys:
                raise ValueError(
                    "parameters['modality_noise'] contains duplicate modality keys after "
                    f"canonicalization: {raw_key!r} resolves to {raw_norm!r}."
                )
            seen_modality_keys.add(raw_norm)
            if raw_norm == modality_key:
                if override is None:
                    continue
                if not isinstance(override, dict):
                    raise TypeError(
                        "parameters['modality_noise'][imaging_model] must be a dictionary."
                    )
                cfg.update(override)
    return cfg


def noise_model_overrides_from_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public base detector-noise override container."""

    noise_model = configured_value(dict(params), "noise_model") or {}
    return dict(noise_model) if isinstance(noise_model, Mapping) else {}


def modality_noise_overrides_from_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public per-modality detector-noise override container."""

    modality_noise = configured_value(dict(params), "modality_noise") or {}
    return dict(modality_noise) if isinstance(modality_noise, Mapping) else {}


def _cfg_value(params: dict[str, Any], key: str) -> Any:
    noise_model = _effective_noise_overrides(params)
    if key in noise_model:
        return noise_model[key]
    return configured_value(params, key)


def _camera_noise_config_from_params(params: dict[str, Any] | None = None) -> CameraNoiseConfig:
    """
    Return the effective counts-domain camera-noise configuration.

    Values are resolved from parameters plus the optional ``noise_model`` and
    ``modality_noise`` override containers. Numeric noise amplitudes are in
    camera counts unless the field name states otherwise.
    """
    params = dict(params or {})
    detector_noise_input_domain = _detector_noise_input_domain_from_config(params)

    def _cfg_bool(key: str) -> bool:
        value = _cfg_value(params, key)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean true/false value; got {value!r}.")
        return bool(value)

    gain = float(_cfg_value(params, "camera_gain_e_per_count"))
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"camera_gain_e_per_count must be finite and positive; got {gain}.")
    if detector_noise_input_domain == "electron_count":
        gain = 1.0

    noise_model = _effective_noise_overrides(params)
    if "detector_qe" in noise_model:
        detector_qe = float(noise_model["detector_qe"])
    elif (
        "fluorescence_detector_qe" in noise_model
        and noise_model["fluorescence_detector_qe"] is not None
    ):
        detector_qe = float(noise_model["fluorescence_detector_qe"])
    else:
        detector_qe = DetectorSettings.from_params(
            params,
            fluorescence=is_fluorescence_modality(ModalitySettings.from_params(params).modality),
        ).detector_qe
    if not np.isfinite(detector_qe) or detector_qe < 0.0 or detector_qe > 1.0:
        raise ValueError(f"detector_qe must be finite and in [0, 1]; got {detector_qe}.")
    emccd_enabled = _cfg_bool("emccd_enabled")
    emccd_gain = float(_cfg_value(params, "emccd_gain"))
    if not np.isfinite(emccd_gain) or emccd_gain <= 0.0:
        raise ValueError(f"emccd_gain must be finite and positive; got {emccd_gain}.")
    emccd_excess = float(_cfg_value(params, "emccd_excess_noise_factor"))
    emccd_excess_basis = normalize_emccd_excess_noise_factor_basis(
        _cfg_value(params, "emccd_excess_noise_factor_basis")
    )
    emccd_shot_variance_multiplier = emccd_shot_variance_multiplier_from_factor(
        emccd_excess,
        emccd_excess_basis,
    )
    read_noise_e = _cfg_value(params, "read_noise_e")
    read_noise = float(_cfg_value(params, "read_noise_counts"))
    if read_noise_e is not None:
        read_noise = float(read_noise_e) / gain
    if not np.isfinite(read_noise) or read_noise < 0.0:
        raise ValueError(f"read_noise_counts must be finite and non-negative; got {read_noise}.")

    dark_offset = float(_cfg_value(params, "dark_offset_counts"))
    background_offset = float(_cfg_value(params, "background_offset_counts"))
    background_offset_stage = normalize_detector_additive_stage(
        _cfg_value(params, "background_offset_stage"),
        context="background_offset_stage",
    )
    dark_offset_stage = normalize_detector_additive_stage(
        _cfg_value(params, "dark_offset_stage"),
        context="dark_offset_stage",
    )
    dark_frame_map_stage = normalize_detector_additive_stage(
        _cfg_value(params, "dark_frame_map_stage"),
        context="dark_frame_map_stage",
    )
    dark_current_e = float(_cfg_value(params, "dark_current_e_per_pixel_per_s"))
    exposure_time_s = float(_cfg_value(params, "exposure_time_s"))
    saturation_level = _cfg_value(params, "saturation_level")
    saturation_level = None if saturation_level is None else float(saturation_level)
    saturation_e = _cfg_value(params, "saturation_e")
    saturation_e = None if saturation_e is None else float(saturation_e)
    adc_quantization = _cfg_bool("adc_quantization")
    adc_quantization_counts = float(_cfg_value(params, "adc_quantization_counts"))
    fpn_gain = float(_cfg_value(params, "fixed_pattern_gain_std"))
    fpn_offset = float(_cfg_value(params, "fixed_pattern_offset_counts"))
    hot_fraction = float(_cfg_value(params, "hot_pixel_fraction"))
    hot_value = _cfg_value(params, "hot_pixel_value_counts")
    hot_value = None if hot_value is None else float(hot_value)
    line_noise = float(_cfg_value(params, "scan_line_noise_counts"))
    fixed_pattern_gain_map = _cfg_value(params, "fixed_pattern_gain_map")
    fixed_pattern_offset_map = _cfg_value(params, "fixed_pattern_offset_map")
    scmos_variance_map = _cfg_value(params, "scmos_variance_map")
    scmos_gain_map = _cfg_value(params, "scmos_gain_map")
    scmos_read_noise_map = _cfg_value(params, "scmos_read_noise_map")
    read_noise_map_mode = str(_cfg_value(params, "read_noise_map_mode")).strip().lower()
    if read_noise_map_mode not in {"replace", "add"}:
        raise ValueError(
            "read_noise_map_mode must be 'replace' or 'add'; "
            f"got {read_noise_map_mode!r}."
        )
    hot_pixel_mask = _cfg_value(params, "hot_pixel_mask")
    flat_field_map = _cfg_value(params, "flat_field_map")
    dark_frame_map = _cfg_value(params, "dark_frame_map")
    if fixed_pattern_gain_map is not None:
        _coerce_map_value(fixed_pattern_gain_map, dtype=float)
    if fixed_pattern_offset_map is not None:
        _coerce_map_value(fixed_pattern_offset_map, dtype=float)
    if scmos_variance_map is not None:
        _coerce_map_value(scmos_variance_map, dtype=float)
    if scmos_gain_map is not None:
        _coerce_map_value(scmos_gain_map, dtype=float)
    if scmos_read_noise_map is not None:
        _coerce_map_value(scmos_read_noise_map, dtype=float)
    if flat_field_map is not None:
        _coerce_map_value(flat_field_map, dtype=float)
    if dark_frame_map is not None:
        _coerce_map_value(dark_frame_map, dtype=float)
    if hot_pixel_mask is not None:
        _resolve_boolean_mask("hot_pixel_mask", hot_pixel_mask)

    # Base validation: required non-negative scalar noise terms.
    for key, value in {
        "fixed_pattern_gain_std": fpn_gain,
        "fixed_pattern_offset_counts": fpn_offset,
        "hot_pixel_fraction": hot_fraction,
        "scan_line_noise_counts": line_noise,
        "background_offset_counts": background_offset,
        "dark_offset_counts": dark_offset,
        "dark_current_e_per_pixel_per_s": dark_current_e,
        "exposure_time_s": exposure_time_s,
        "adc_quantization_counts": adc_quantization_counts,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and non-negative; got {value}.")
    if hot_value is not None and (not np.isfinite(hot_value) or hot_value < 0.0):
        raise ValueError(
            "hot_pixel_value_counts must be finite and non-negative when supplied; "
            f"got {hot_value}."
        )

    # Hot pixel fraction is additionally bounded above by one.
    if hot_fraction > 1.0:
        raise ValueError(f"hot_pixel_fraction must be <= 1.0; got {hot_fraction}.")

    # ADC quantization must be strictly positive when enabled.
    if adc_quantization and adc_quantization_counts <= 0.0:
        raise ValueError("adc_quantization_counts must be positive when adc_quantization is enabled.")
    if saturation_level is not None and (not np.isfinite(saturation_level) or saturation_level <= 0.0):
        raise ValueError(f"saturation_level must be finite and positive when supplied; got {saturation_level}.")
    if saturation_e is not None and (not np.isfinite(saturation_e) or saturation_e <= 0.0):
        raise ValueError(f"saturation_e must be finite and positive when supplied; got {saturation_e}.")

    noise_parameterization = str(_cfg_value(params, "noise_parameterization")).strip().lower()
    if noise_parameterization != "camera_counts":
        raise ValueError(
            "noise_parameterization must be 'camera_counts'; "
            f"got {_cfg_value(params, 'noise_parameterization')!r}."
        )
    detector_frame_representation_contract = resolve_detector_frame_representation_contract(
        params,
        context="CameraNoiseConfig.from_params",
    )
    detector_input_is_incident_quanta = (
        detector_frame_representation_contract.detector_input_is_incident_quanta
    )

    nonlinearity_calibration = _cfg_value(params, "nonlinearity_calibration")
    if nonlinearity_calibration is not None:
        nonlinearity_calibration = str(nonlinearity_calibration)

    nonlinear_active = bool(
        saturation_level is not None
        or saturation_e is not None
        or adc_quantization
        or nonlinearity_calibration is not None
    )
    deterministic_transfer_active = bool(
        fixed_pattern_gain_map is not None
        or fixed_pattern_offset_map is not None
        or scmos_gain_map is not None
        or flat_field_map is not None
        or dark_frame_map is not None
        or hot_pixel_mask is not None
        or fpn_gain > 0.0
        or fpn_offset > 0.0
        or hot_fraction > 0.0
    )
    correlated_noise_active = bool(line_noise > 0.0)
    safe_for_linear_fisher_variance = not (
        nonlinear_active or deterministic_transfer_active or correlated_noise_active
    )

    return CameraNoiseConfig(
        shot_noise_enabled=_cfg_bool("shot_noise_enabled"),
        gaussian_noise_enabled=_cfg_bool("gaussian_noise_enabled"),
        camera_gain_e_per_count=gain,
        detector_qe=detector_qe,
        detector_input_is_incident_quanta=detector_input_is_incident_quanta,
        detector_noise_input_domain=detector_noise_input_domain,
        emccd_enabled=emccd_enabled,
        emccd_gain=emccd_gain,
        emccd_excess_noise_factor=emccd_excess,
        emccd_excess_noise_factor_basis=emccd_excess_basis,
        emccd_shot_variance_multiplier=emccd_shot_variance_multiplier,
        read_noise_e=None if read_noise_e is None else float(read_noise_e),
        read_noise_counts=read_noise,
        dark_current_e_per_pixel_per_s=dark_current_e,
        exposure_time_s=exposure_time_s,
        saturation_level=saturation_level,
        saturation_e=saturation_e,
        adc_quantization=adc_quantization,
        adc_quantization_counts=adc_quantization_counts,
        background_offset_counts=background_offset,
        background_offset_stage=background_offset_stage,
        dark_offset_counts=dark_offset,
        dark_offset_stage=dark_offset_stage,
        fixed_pattern_gain_std=fpn_gain,
        fixed_pattern_offset_counts=fpn_offset,
        fixed_pattern_gain_map=fixed_pattern_gain_map,
        fixed_pattern_offset_map=fixed_pattern_offset_map,
        scmos_variance_map=scmos_variance_map,
        scmos_gain_map=scmos_gain_map,
        scmos_read_noise_map=scmos_read_noise_map,
        read_noise_map_mode=read_noise_map_mode,
        hot_pixel_mask=hot_pixel_mask,
        nonlinearity_calibration=nonlinearity_calibration,
        flat_field_map=flat_field_map,
        dark_frame_map=dark_frame_map,
        dark_frame_map_stage=dark_frame_map_stage,
        hot_pixel_fraction=hot_fraction,
        hot_pixel_value_counts=hot_value,
        scan_line_noise_counts=line_noise,
        clip_output_to_nonnegative=_cfg_bool("clip_output_to_nonnegative"),
        noise_parameterization=noise_parameterization,
        nonlinear_detector_effects_active=nonlinear_active,
        deterministic_detector_transfer_active=deterministic_transfer_active,
        safe_for_linear_fisher_variance=safe_for_linear_fisher_variance,
    )


def camera_noise_metadata(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved camera-noise configuration as JSON-safe metadata."""
    cfg = CameraNoiseConfig.from_params(params)
    reference_noise = _analysis_reference_frame_noise_contract(dict(params or {}))
    detector_representation = resolve_detector_frame_representation_contract(
        dict(params or {}),
        context="camera_noise_metadata",
    )
    meta = asdict(cfg)
    meta["detector_frame_representation_contract_id"] = detector_representation.contract_id
    meta["detector_frame_basis"] = detector_representation.frame_basis
    meta["detector_qe_owner"] = detector_representation.detector_qe_owner
    meta["detector_frame_basis_resolution"] = detector_representation.resolution
    meta["scan_line_noise_in_fisher_variance"] = bool(cfg.scan_line_noise_counts > 0.0)
    meta["noise_covariance_kind"] = (
        "row_correlated_scan_lines"
        if cfg.scan_line_noise_counts > 0.0
        else "independent_pixels"
    )
    meta["noise_independence_assumption"] = (
        "diagonal_variance includes the row-noise covariance diagonal; "
        "AnalysisNoiseModel carries row-correlated couplings for covariance Fisher"
        if cfg.scan_line_noise_counts > 0.0
        else "independent pixel variance"
    )
    meta["scan_line_noise_stage"] = (
        "post_gain_output_domain_row_offset"
        if cfg.scan_line_noise_counts > 0.0
        else "not_applied"
    )
    meta["scan_line_noise_gated_by_gaussian_noise_enabled"] = False
    meta["scan_line_noise_input_referred_by_emccd_gain"] = False
    meta.update(reference_noise)
    meta["emccd_gain_applied_stage"] = (
        "input_referred_read_noise_reduction"
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0
        else "not_applied"
    )
    meta["emccd_excess_noise_basis_contract_id"] = EMCCD_EXCESS_NOISE_BASIS_CONTRACT_ID
    meta["emccd_excess_noise_factor_raw"] = float(cfg.emccd_excess_noise_factor)
    meta["emccd_excess_noise_factor_basis_resolved"] = cfg.emccd_excess_noise_factor_basis
    meta["emccd_shot_variance_multiplier"] = float(cfg.emccd_shot_variance_multiplier)
    meta["detector_transfer_stage_contract_id"] = DETECTOR_TRANSFER_STAGE_CONTRACT_ID
    meta["detector_additive_stage_contract"] = {
        "background_offset_counts": cfg.background_offset_stage,
        "dark_offset_counts": cfg.dark_offset_stage,
        "dark_frame_map": cfg.dark_frame_map_stage,
    }
    meta["shot_noise_variance_stage"] = (
        "poisson_driver_mean_counts_then_post_poisson_gain_squared"
    )
    meta["fisher_variance_model_scope"] = (
        "linear_poisson_gaussian_only"
        if cfg.safe_for_linear_fisher_variance
        else (
            "row_correlated_noise_requires_covariance_fisher"
            if cfg.scan_line_noise_counts > 0.0
            else "diagnostic_only_linearized_detector_variance"
        )
    )
    meta["detector_likelihood_status"] = (
        "linear_poisson_gaussian_compatible"
        if cfg.safe_for_linear_fisher_variance
        else (
            "row_correlated_scan_line_noise_requires_analysis_noise_model"
            if cfg.scan_line_noise_counts > 0.0
            else "nonlinear_or_static_transfer_not_in_linear_fisher"
        )
    )
    meta["safe_for_covariance_fisher_variance"] = bool(
        not (cfg.nonlinear_detector_effects_active or cfg.deterministic_detector_transfer_active)
    )
    meta["covariance_fisher_variance_model_scope"] = (
        "linear_poisson_gaussian_with_row_covariance"
        if meta["safe_for_covariance_fisher_variance"]
        else "diagnostic_only_nonlinear_or_static_transfer_not_in_covariance_fisher"
    )
    return meta


def _analysis_reference_frame_noise_contract(params: dict[str, Any]) -> dict[str, Any]:
    """Declare how reference-frame noise is propagated into analysis contrast."""

    method = BackgroundSubtractionSettings.from_params(params).method
    if method not in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        return {
            "analysis_reference_frame_noise_contract": "not_applicable",
            "analysis_reference_frame_stochastic_model": "no_reference_frame_in_analysis_statistic",
        }
    modality = ModalitySettings.from_params(params).modality
    output_type = "intensity"
    try:
        from imaging_models import get_imaging_model_class

        output_type = getattr(get_imaging_model_class(modality), "output_type", "intensity")
    except Exception:  # noqa: BLE001 - metadata must remain available during partial imports
        output_type = "unknown"
    if output_type == "phase":
        return {
            "analysis_reference_frame_noise_contract": "deterministic_reference_centering",
            "analysis_reference_frame_stochastic_model": (
                "phase-output QPI analysis frames are already reference-normalized "
                "phase maps; the empty reference frame centers the displayed "
                "contrast but does not add an independent phase likelihood "
                "variance to Fisher"
            ),
            "analysis_reference_frame_variance_basis": (
                "single_reference_normalized_phase_map_variance"
            ),
        }
    elif modality_uses_relative_reference_contrast(modality):
        variance_basis = "relative_reference_error_propagation_signal_and_independent_reference"
    else:
        variance_basis = "additive_signal_plus_independent_reference_variance"
    return {
        "analysis_reference_frame_noise_contract": "independent_noisy_reference_frame",
        "analysis_reference_frame_stochastic_model": (
            "reference frames are treated as independently measured noisy "
            "observations; reference variance is propagated into Fisher "
            "likelihood variance rather than treated as deterministic calibration"
        ),
        "analysis_reference_frame_variance_basis": variance_basis,
    }


def _input_referred_signal_counts_for_shot_noise(
    signal_counts: np.ndarray | float,
    cfg: CameraNoiseConfig,
) -> np.ndarray:
    """Return the signal contribution in the Poisson-sampling count domain."""
    signal = np.asarray(signal_counts, dtype=float)
    input_referred = signal.astype(float, copy=True)
    if cfg.detector_input_is_incident_quanta:
        input_referred = input_referred * cfg.detector_qe / cfg.camera_gain_e_per_count
    return np.asarray(input_referred, dtype=float)


def _resolve_additive_offset_stage_maps(
    shape: tuple[int, ...],
    cfg: CameraNoiseConfig,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Split additive detector offsets into explicit detector-transfer stage maps."""

    pre = np.zeros(shape, dtype=float)
    post_pre_gain = np.zeros(shape, dtype=float)
    post_gain = np.zeros(shape, dtype=float)
    stage_by_source = {
        "background_offset_counts": cfg.background_offset_stage,
        "dark_offset_counts": cfg.dark_offset_stage,
        "dark_frame_map": cfg.dark_frame_map_stage,
    }

    def add_offset(source: str, values: np.ndarray | float | int | None, stage: str) -> None:
        nonlocal pre, post_pre_gain, post_gain
        if values is None:
            return
        arr = _map_to_array(values, shape)
        if stage == DETECTOR_ADDITIVE_STAGE_PRE_POISSON:
            pre = pre + arr
        elif stage == DETECTOR_ADDITIVE_STAGE_POST_POISSON_PRE_GAIN:
            post_pre_gain = post_pre_gain + arr
        elif stage == DETECTOR_ADDITIVE_STAGE_POST_GAIN:
            post_gain = post_gain + arr
        else:  # Defensive guard; public validation should already catch this.
            raise ValueError(f"Unsupported additive detector stage for {source}: {stage!r}.")

    dark_frame_map, _ = _resolve_map_shape(
        "dark_frame_map",
        cfg.dark_frame_map,
        shape,
        dtype=float,
        runtime=runtime,
    )
    # Additive scalar offsets and dark-frame maps are detector-transfer terms,
    # not optical contrast terms.  Their declared stage must be consumed by the
    # stochastic Poisson sampler and Fisher shot-variance code through the same
    # stage object, so future patches cannot make raw frames and CRLB disagree.
    add_offset("background_offset_counts", cfg.background_offset_counts, cfg.background_offset_stage)
    add_offset("dark_offset_counts", cfg.dark_offset_counts, cfg.dark_offset_stage)
    add_offset("dark_frame_map", dark_frame_map, cfg.dark_frame_map_stage)
    return pre, post_pre_gain, post_gain, stage_by_source


def _detector_transfer_stages(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> DetectorTransferStages:
    """
    Resolve detector-transfer stages used by both rendering metadata and Fisher noise.

    The stochastic renderer samples Poisson noise before flat-field, fixed-pattern
    gain, and sCMOS gain maps are applied.  The shot-noise driver must therefore
    stay input-referred; only the variance is subsequently scaled by the square
    of the post-Poisson multiplicative gain.  Collapsing these stages back into
    the full deterministic detector mean reintroduces the historical G**3 shot-
    variance bug for any deterministic gain map G.
    """
    cfg = CameraNoiseConfig.from_params(params)
    signal = np.asarray(signal_counts, dtype=float)
    active_runtime = _noise_runtime(runtime)

    pre_additive, post_pre_gain_additive, post_gain_additive, additive_stage_by_source = _resolve_additive_offset_stage_maps(
        signal.shape,
        cfg,
        runtime=active_runtime,
    )
    poisson_driver_mean = _input_referred_signal_counts_for_shot_noise(signal, cfg)
    if cfg.dark_current_e_per_pixel_per_s and cfg.exposure_time_s:
        poisson_driver_mean = poisson_driver_mean + (
            cfg.dark_current_e_per_pixel_per_s
            * cfg.exposure_time_s
            / cfg.camera_gain_e_per_count
        )
    poisson_driver_mean = poisson_driver_mean + pre_additive

    post_poisson_gain = _post_poisson_multiplicative_gain(
        signal.shape,
        cfg,
        params,
        runtime=active_runtime,
    )
    # NOTE: previously this also precomputed deterministic_output_mean /
    # deterministic_baseline by calling deterministic_detector_transfer_counts(),
    # which is apply_camera_noise_counts(stochastic=False) -> _detector_transfer_stages
    # -> back here: an INFINITE mutual recursion. Those two fields were never read
    # by any caller, so they are removed. The noise-free transfer remains available
    # via deterministic_detector_transfer_counts(), which no longer recurses.
    return DetectorTransferStages(
        poisson_driver_mean_counts=np.asarray(poisson_driver_mean, dtype=float),
        post_poisson_multiplicative_gain=post_poisson_gain,
        pre_poisson_additive_counts=np.asarray(pre_additive, dtype=float),
        post_poisson_pre_gain_additive_counts=np.asarray(post_pre_gain_additive, dtype=float),
        post_gain_additive_counts=np.asarray(post_gain_additive, dtype=float),
        additive_stage_by_source=dict(additive_stage_by_source),
    )


def _shot_noise_variance_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return shot-noise variance after post-Poisson detector gain, in counts^2."""
    cfg = CameraNoiseConfig.from_params(params)
    signal = np.asarray(signal_counts, dtype=float)
    if not cfg.shot_noise_enabled:
        return np.zeros_like(signal, dtype=float)
    stages = _detector_transfer_stages(signal, params, runtime=runtime)
    counts_pos = np.where(
        np.isfinite(stages.poisson_driver_mean_counts)
        & (stages.poisson_driver_mean_counts > 0.0),
        stages.poisson_driver_mean_counts,
        0.0,
    )
    shot_variance = counts_pos / cfg.camera_gain_e_per_count
    if cfg.emccd_enabled:
        # Only the canonical variance multiplier may cross into Fisher/CRLB.
        # Consuming the raw public factor here would reintroduce the std-vs-variance seam.
        shot_variance = shot_variance * cfg.emccd_shot_variance_multiplier
    # The stage resolver deliberately returns a pre-gain Poisson driver.  This
    # is the only place the post-Poisson flat-field/FPN/sCMOS gain may enter the
    # shot variance; applying it before this point would add an erroneous extra
    # factor of G to the variance.
    shot_variance = shot_variance * np.square(
        np.asarray(stages.post_poisson_multiplicative_gain, dtype=float)
    )
    return np.maximum(shot_variance, 0.0)


def _stochastic_detected_mean_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Return the Poisson-driving mean count map.

    This compatibility helper exposes the staged detector contract used by the
    variance model.  It must not be rebuilt from deterministic detector-output
    frames, because those frames already include post-Poisson gain maps.
    """
    return _detector_transfer_stages(
        signal_counts,
        params,
        runtime=runtime,
    ).poisson_driver_mean_counts


def shot_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return shot-noise standard deviation in camera counts."""
    return np.sqrt(
        np.maximum(
            _shot_noise_variance_counts(signal_counts, params, runtime=runtime),
            0.0,
        )
    )


def _post_poisson_multiplicative_gain(
    shape: tuple[int, ...],
    cfg: CameraNoiseConfig,
    params: dict[str, Any] | None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray | float:
    """
    Return the deterministic gain applied after Poisson/EMCCD shot sampling.

    apply_camera_noise_counts() samples shot noise in the input-referred detector
    count domain, then multiplies by flat-field, fixed-pattern gain, and sCMOS
    gain maps before adding read/scan-line noise. The shot-noise variance used by
    Fisher/supervision must therefore be multiplied by this gain squared.
    """
    active_runtime = _noise_runtime(runtime)
    gain: np.ndarray | float = 1.0
    fixed_pattern_gain_map, _ = _resolve_map_shape(
        "fixed_pattern_gain_map",
        cfg.fixed_pattern_gain_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    flat_field_map, _ = _resolve_map_shape(
        "flat_field_map",
        cfg.flat_field_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    scmos_gain_map, _ = _resolve_map_shape(
        "scmos_gain_map",
        cfg.scmos_gain_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    static_maps: dict[str, np.ndarray] = {}
    if fixed_pattern_gain_map is None and cfg.fixed_pattern_gain_std > 0.0:
        static_maps = _static_detector_maps(shape, cfg, params, runtime=active_runtime)
        fixed_pattern_gain_map = static_maps.get("gain")
    if flat_field_map is not None:
        gain = np.asarray(gain, dtype=float) * _map_to_array(flat_field_map, shape)
    if fixed_pattern_gain_map is not None:
        gain = np.asarray(gain, dtype=float) * _map_to_array(fixed_pattern_gain_map, shape)
    if scmos_gain_map is not None:
        gain = np.asarray(gain, dtype=float) * _map_to_array(scmos_gain_map, shape)
    return gain


def _readout_noise_variance_counts(
    signal_counts: np.ndarray | float,
    cfg: CameraNoiseConfig,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    shape = np.asarray(signal_counts, dtype=float).shape
    read_var = np.zeros(shape, dtype=float)
    if cfg.gaussian_noise_enabled:
        scmos_variance_map, _ = _resolve_map_shape(
            "scmos_variance_map",
            cfg.scmos_variance_map,
            shape,
            dtype=float,
            runtime=runtime,
        )
        scmos_read_noise_map, _ = _resolve_map_shape(
            "scmos_read_noise_map",
            cfg.scmos_read_noise_map,
            shape,
            dtype=float,
            runtime=runtime,
        )
        has_calibrated_read_map = scmos_variance_map is not None or scmos_read_noise_map is not None
        if has_calibrated_read_map and cfg.read_noise_map_mode == "replace":
            read_var = np.zeros(shape, dtype=float)
        else:
            read_var = np.full(shape, cfg.read_noise_counts, dtype=float) ** 2
        if scmos_variance_map is not None:
            read_var = read_var + _map_to_array(scmos_variance_map, shape)
        if scmos_read_noise_map is not None:
            read_var = read_var + np.square(_map_to_array(scmos_read_noise_map, shape))
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0:
            read_var = read_var / (cfg.emccd_gain ** 2)
    return np.maximum(read_var, 0.0)


def _scan_line_noise_diagonal_variance_counts(
    signal_counts: np.ndarray | float,
    cfg: CameraNoiseConfig,
) -> np.ndarray:
    """Return the diagonal contribution of output-domain row-offset noise."""

    shape = np.asarray(signal_counts, dtype=float).shape
    if cfg.scan_line_noise_counts > 0.0:
        return np.full(shape, float(cfg.scan_line_noise_counts) ** 2, dtype=float)
    return np.zeros(shape, dtype=float)


def total_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return combined shot and read-noise standard deviation in camera counts."""
    variance = total_noise_variance_counts(signal_counts, params, runtime=runtime)
    return np.sqrt(np.maximum(variance, 0.0))


def total_noise_variance_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """Return combined shot and read-noise variance in camera-count units."""
    cfg = CameraNoiseConfig.from_params(params)
    signal = np.asarray(signal_counts, dtype=float)
    active_runtime = _noise_runtime(runtime)
    shot_var = _shot_noise_variance_counts(signal, params, runtime=active_runtime)
    readout_var = _readout_noise_variance_counts(signal, cfg, runtime=active_runtime)
    scan_line_var = _scan_line_noise_diagonal_variance_counts(signal, cfg)
    return np.maximum(shot_var + readout_var + scan_line_var, 0.0)


def detected_mean_counts_before_noise(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Return the renderer's pre-Poisson detector mean including deterministic pedestals.

    This is not the Fisher shot-noise driver.  Variance code must use
    _detector_transfer_stages().poisson_driver_mean_counts so deterministic
    post-Poisson gains cannot be folded into the Poisson mean.
    """
    cfg = CameraNoiseConfig.from_params(params)
    counts = np.asarray(signal_counts, dtype=float)
    noisy_mean = counts.astype(float, copy=True)
    shape = noisy_mean.shape
    active_runtime = runtime or DetectorNoiseRuntime()
    dark_frame_map, _ = _resolve_map_shape(
        "dark_frame_map",
        cfg.dark_frame_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    if dark_frame_map is not None:
        dark_frame_map = _map_to_array(dark_frame_map, shape)
    stages = _detector_transfer_stages(noisy_mean, params, runtime=active_runtime)
    # The compatibility helper returns the full deterministic pre-random mean.
    # It therefore combines the shared Poisson driver with declared post-Poisson
    # additive bias terms, rather than reimplementing additive stage placement.
    return np.asarray(
        stages.poisson_driver_mean_counts
        + stages.post_poisson_pre_gain_additive_counts
        + stages.post_gain_additive_counts,
        dtype=float,
    )


def detector_mean_frames_for_analysis(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return signal/reference frames in the detector-mean domain."""
    signal_mean = deterministic_detector_transfer_counts(signal_counts, params, runtime=runtime)
    reference_mean = (
        None
        if reference_counts is None
        else deterministic_detector_transfer_counts(reference_counts, params, runtime=runtime)
    )
    return signal_mean, reference_mean


def detector_photoresponse_frames_for_analysis(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return deterministic detector photoresponse frames for physical contrast.

    Relative-reference contrast is a sample/reference photoresponse observable:
    ``(S - R) / R``.  Additive detector pedestals such as bias offsets, dark
    offsets, dark-frame maps, and generated fixed-pattern offsets belong in raw
    detector means and shot-noise variance, but they must not redefine the
    physical reference denominator.  Subtract the zero-input deterministic
    detector baseline while preserving multiplicative detector transfer.
    """
    signal = np.asarray(signal_counts, dtype=float)
    active_runtime = runtime or DetectorNoiseRuntime()
    signal_mean = deterministic_detector_transfer_counts(
        signal,
        params,
        runtime=active_runtime,
    )
    signal_baseline = deterministic_detector_transfer_counts(
        np.zeros_like(signal, dtype=float),
        params,
        runtime=active_runtime,
    )
    signal_photoresponse = signal_mean - signal_baseline
    if reference_counts is None:
        return signal_photoresponse, None
    reference = np.asarray(reference_counts, dtype=float)
    reference_mean = deterministic_detector_transfer_counts(
        reference,
        params,
        runtime=active_runtime,
    )
    reference_baseline = deterministic_detector_transfer_counts(
        np.zeros_like(reference, dtype=float),
        params,
        runtime=active_runtime,
    )
    return signal_photoresponse, reference_mean - reference_baseline


def detector_contrast_frames_for_analysis(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    runtime: DetectorNoiseRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return count-domain signal/reference frames in the correct contrast basis."""
    if relative_reference is None:
        relative_reference = modality_uses_relative_reference_contrast(
            ModalitySettings.from_params(dict(params or {})).modality
        )
    if bool(relative_reference):
        return detector_photoresponse_frames_for_analysis(
            signal_counts,
            reference_counts,
            params,
            runtime=runtime,
        )
    return detector_mean_frames_for_analysis(
        signal_counts,
        reference_counts,
        params,
        runtime=runtime,
    )



def qpi_phase_noise_components_rad2(
    params: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    """Return scalar QPI phase-noise components for uniform likelihood metadata."""
    params = dict(params or {})
    cfg = CameraNoiseConfig.from_params(params)
    settings = QpiReadoutSettings.from_params(params)
    shot_variance = settings.shot_variance_rad2(
        shot_noise_enabled=bool(cfg.shot_noise_enabled),
    )
    readout_variance = settings.readout_variance_rad2(
        gaussian_noise_enabled=bool(cfg.gaussian_noise_enabled),
    )
    total_variance = float(shot_variance + readout_variance)
    return {
        "shot_variance_rad2": float(shot_variance),
        "readout_variance_rad2": float(readout_variance),
        "total_variance_rad2": total_variance,
        "visibility": float(settings.visibility),
        "configured_detected_quanta_per_pixel": float(
            settings.configured_detected_quanta_per_pixel
        ),
        "detected_quanta_per_pixel": float(settings.detected_quanta_per_pixel),
        "detected_quanta_exposure_scale": float(settings.exposure_signal_scale),
        "shot_noise_enabled": bool(cfg.shot_noise_enabled),
        "gaussian_noise_enabled": bool(cfg.gaussian_noise_enabled),
    }


def _qpi_phase_readout_variance_rad2(params: dict[str, Any], cfg: CameraNoiseConfig) -> float:
    return QpiReadoutSettings.from_params(params).readout_variance_rad2(
        gaussian_noise_enabled=bool(cfg.gaussian_noise_enabled),
    )


def _qpi_phase_likelihood_requires_explicit_map(params: dict[str, Any]) -> bool:
    """Return whether scalar QPI quanta would erase reference-field structure."""

    settings = SampleEnvironmentSettings.from_params(params)
    return bool(
        settings.pattern_active
        or (settings.enabled and settings.roughness.active)
        or settings.empirical_background.active
    )


def qpi_phase_likelihood_noise_params(
    params: dict[str, Any] | None,
    reference_detected_quanta_per_pixel: np.ndarray | float,
    *,
    provenance: str = "reference_background_map",
) -> dict[str, Any]:
    """Attach an explicit QPI detected-quanta map to a per-frame noise payload.

    QPI analysis frames are phase maps in radians and display frames are scaled
    counts, so neither is a reliable photon basis.  Renderers that know the
    coherent-reference/background quanta must cross the noise seam through this
    helper rather than asking camera_noise.py to infer photons from phase or
    display data.
    """

    out = dict(params or {})
    quanta = np.asarray(reference_detected_quanta_per_pixel, dtype=float)
    if quanta.shape == ():
        value = float(quanta)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "reference_detected_quanta_per_pixel must be positive and finite; "
                f"got {reference_detected_quanta_per_pixel!r}."
            )
    elif np.any(~np.isfinite(quanta)) or np.any(quanta <= 0.0):
        raise ValueError("reference_detected_quanta_per_pixel map must contain only positive finite values.")
    out[QPI_DETECTED_QUANTA_MAP_PARAM] = quanta
    out[QPI_PHASE_LIKELIHOOD_PROVENANCE_PARAM] = str(provenance)
    out[QPI_PHASE_LIKELIHOOD_CONTRACT_PARAM] = QPI_PHASE_LIKELIHOOD_CONTRACT_ID
    return out



def qpi_phase_likelihood_parameter_frame(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return the QPI per-frame likelihood overlay that must accompany render frames.

    This sidecar is intentionally only the likelihood-owned overlay, not a full
    parameters copy.  The renderer is the only owner of post-pattern/post-roughness
    coherent-reference quanta; downstream Fisher/report callers must merge this
    overlay before constructing phase-domain noise.  Reconstructing it from QPI
    display counts would silently mix visualization units with photon support.
    """

    payload = dict(params or {})
    if QPI_DETECTED_QUANTA_MAP_PARAM not in payload:
        return {}
    return {
        QPI_DETECTED_QUANTA_MAP_PARAM: np.asarray(
            payload[QPI_DETECTED_QUANTA_MAP_PARAM], dtype=float
        ),
        QPI_PHASE_LIKELIHOOD_PROVENANCE_PARAM: str(
            payload.get(QPI_PHASE_LIKELIHOOD_PROVENANCE_PARAM, "detected_quanta_map")
        ),
        QPI_PHASE_LIKELIHOOD_CONTRACT_PARAM: str(
            payload.get(QPI_PHASE_LIKELIHOOD_CONTRACT_PARAM, QPI_PHASE_LIKELIHOOD_CONTRACT_ID)
        ),
    }


def analysis_noise_params_for_frame(
    params: dict[str, Any] | None,
    render_metadata: dict[str, Any] | None,
    *,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Merge render-time likelihood sidecars into params for one analysis frame.

    Exposure time is metadata for every modality.  Phase-domain QPI additionally
    needs a per-frame detected-quanta map produced by the renderer after sample
    patterns, roughness, and empirical backgrounds have modified the coherent
    reference.  Keeping this merge in one helper prevents report, packet,
    calibration, and sequence paths from each inventing a different fallback.
    """

    out = dict(params or {})
    metadata = dict(render_metadata or {})
    effective_exposure_time_s = metadata.get("effective_exposure_time_s")
    if effective_exposure_time_s is not None:
        out["exposure_time_s"] = float(effective_exposure_time_s)

    def finalize(merged: dict[str, Any]) -> dict[str, Any]:
        # Analysis/Fisher callers consume the same detector-input frames that the
        # renderer produced.  Canonicalize the frame-basis contract here so QPI
        # sidecars, fluorescence detected-count frames, reports, calibration,
        # and matched packets cannot each invent a different QE ownership rule.
        return canonicalize_detector_frame_noise_params(
            merged,
            render_metadata=metadata,
            context="analysis_noise_params_for_frame",
        )

    frame_overlays = metadata.get(ANALYSIS_NOISE_PARAMETER_FRAMES_KEY)
    if frame_overlays is None:
        return finalize(out)
    if not isinstance(frame_overlays, (list, tuple)):
        raise TypeError(
            f"{ANALYSIS_NOISE_PARAMETER_FRAMES_KEY} must be a list of per-frame mappings."
        )
    index = int(frame_index)
    if index < 0 or index >= len(frame_overlays):
        raise ValueError(
            f"frame_index={index} is outside {ANALYSIS_NOISE_PARAMETER_FRAMES_KEY} "
            f"range 0..{len(frame_overlays) - 1}."
        )
    overlay = frame_overlays[index] or {}
    if not isinstance(overlay, Mapping):
        raise TypeError(
            f"{ANALYSIS_NOISE_PARAMETER_FRAMES_KEY}[{index}] must be a mapping; "
            f"got {type(overlay).__name__}."
        )
    if not overlay:
        return finalize(out)
    for key, value in overlay.items():
        if key == QPI_DETECTED_QUANTA_MAP_PARAM:
            out[key] = np.asarray(value, dtype=float)
        else:
            out[str(key)] = value
    return finalize(out)

def _qpi_phase_likelihood_basis_for_frame(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None,
) -> PhaseLikelihoodBasis:
    params = dict(params or {})
    signal = np.asarray(signal_counts, dtype=float)
    cfg = CameraNoiseConfig.from_params(params)
    settings = QpiReadoutSettings.from_params(params)
    map_raw = configured_optional(params, QPI_DETECTED_QUANTA_MAP_PARAM)
    if map_raw is None:
        if _qpi_phase_likelihood_requires_explicit_map(params):
            raise ValueError(
                "QPI phase Fisher/noise requires an explicit detected-quanta map "
                "when sample-environment patterns, roughness, or empirical "
                "backgrounds make the coherent-reference quanta spatially varying. "
                "Use qpi_phase_likelihood_noise_params() at the renderer/noise seam "
                "instead of falling back to a scalar qpi_detected_quanta_per_pixel."
            )
        components = qpi_phase_noise_components_rad2(params)
        detected_quanta = float(components["detected_quanta_per_pixel"])
        provenance = "uniform_config_scalar"
    else:
        # The map is stored in the renderer's pre-exposure quanta/count basis.
        # Apply the same exposure scaling as scalar qpi_detected_quanta_per_pixel
        # so scalar and map paths remain physically comparable.
        detected_quanta = np.asarray(map_raw, dtype=float) * settings.exposure_signal_scale
        provenance = str(
            configured_optional(params, QPI_PHASE_LIKELIHOOD_PROVENANCE_PARAM, "detected_quanta_map")
        )
    return PhaseLikelihoodBasis(
        detected_quanta_per_pixel=detected_quanta,
        visibility=settings.visibility,
        readout_variance_rad2=_qpi_phase_readout_variance_rad2(params, cfg),
        provenance=provenance,
        contract_id=QPI_PHASE_LIKELIHOOD_CONTRACT_ID,
    )


def qpi_phase_noise_variance_rad2(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    variance_floor: float = 1e-30,
) -> np.ndarray:
    """Return QPI phase-domain variance from an explicit phase-likelihood basis."""
    signal = np.asarray(signal_counts, dtype=float)
    basis = _qpi_phase_likelihood_basis_for_frame(signal, params)
    cfg = CameraNoiseConfig.from_params(dict(params or {}))
    return phase_variance_from_likelihood_basis(
        basis,
        signal.shape,
        shot_noise_enabled=bool(cfg.shot_noise_enabled),
        gaussian_noise_enabled=bool(cfg.gaussian_noise_enabled),
        variance_floor=variance_floor,
    )




def contrast_noise_variance_counts(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Propagate detector noise into the contrast image used by Fisher diagnostics.

    For raw/no-subtraction views this returns the signal-frame count variance.
    For additive reference subtraction, ``C = S - R`` and
    ``Var(C) = Var(S) + Var(R)``. For relative reference contrast,
    ``C = (S - R) / R = S/R - 1`` and first-order propagation gives
    ``Var(C) = Var(S)/R^2 + S^2 Var(R)/R^4``.
    """
    params = dict(params or {})
    method = BackgroundSubtractionSettings.from_params(params).method
    if method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        raise ValueError(
            "video_median contrast-noise variance requires the signal-frame stack; "
            "a reference frame is not part of the video_median contrast statistic. "
            "Use a stack-aware variance helper or set background_subtraction_method='raw' "
            "for detector-count support calculations."
        )
    signal = np.asarray(signal_counts, dtype=float)
    var_signal = total_noise_variance_counts(signal, params, runtime=runtime)
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS or reference_counts is None:
        return np.maximum(var_signal, float(variance_floor))

    reference = np.asarray(reference_counts, dtype=float)
    if signal.shape != reference.shape:
        raise ValueError(
            "signal_counts and reference_counts must have the same shape for "
            f"contrast-noise propagation; got {signal.shape} and {reference.shape}."
        )
    var_reference = total_noise_variance_counts(reference, params, runtime=runtime)

    if relative_reference is None:
        imaging_model_name = ModalitySettings.from_params(params).modality
        from imaging_models import modality_uses_relative_reference_contrast

        relative_reference = modality_uses_relative_reference_contrast(imaging_model_name)

    if bool(relative_reference):
        signal_mean, reference_mean = detector_photoresponse_frames_for_analysis(
            signal,
            reference,
            params,
            runtime=runtime,
        )
        ref_safe = np.maximum(np.abs(reference_mean), 1e-12)
        variance = var_signal / (ref_safe ** 2) + (signal_mean ** 2) * var_reference / (ref_safe ** 4)
    else:
        variance = var_signal + var_reference
    return np.maximum(variance, float(variance_floor))


def analysis_contrast_noise_variance(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Propagate detector noise into the analysis contrast returned by public views.

    Most modalities use count-domain contrast conventions and can delegate
    directly to :func:`contrast_noise_variance_counts`. Phase-output modalities
    such as QPI expose phase contrast in radians, while their rendered signal
    and reference frames are count-like display images. For those modes, phase
    variance is inferred from visibility and detected quanta, plus any explicit
    phase-readout calibration noise.
    """
    params = dict(params or {})
    imaging_model_name = ModalitySettings.from_params(params).modality
    from imaging_models import get_imaging_model_class

    output_type = getattr(get_imaging_model_class(imaging_model_name), "output_type", "intensity")
    if output_type == "phase":
        del relative_reference, runtime
        method = BackgroundSubtractionSettings.from_params(params).method
        signal_variance = qpi_phase_noise_variance_rad2(
            signal_counts,
            params,
            variance_floor=variance_floor,
        )
        if method in RAW_BACKGROUND_SUBTRACTION_METHODS or method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
            # QPI raw/video-median contrast for phase modalities remains count-domain
            # because no reference normalization by phase-to-count scale is applied.
            phase_scale = QpiReadoutSettings.from_params(params).phase_to_count_scale
            count_variance = signal_variance * (phase_scale * phase_scale)
            assert_compatible(
                context="analysis_contrast_noise_variance",
                measurement_domain="count",
                signal_units="detector_count",
                noise_variance_units="detector_count_squared",
                params=params,
            )
            return np.maximum(count_variance, float(variance_floor))
        if method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS and reference_counts is not None:
            assert_compatible(
                context="analysis_contrast_noise_variance",
                measurement_domain="phase",
                signal_units="radian",
                noise_variance_units="radian_squared",
                params=params,
            )
            return signal_variance
        assert_compatible(
            context="analysis_contrast_noise_variance",
            measurement_domain="phase",
            signal_units="radian",
            noise_variance_units="radian_squared",
            params=params,
        )
        return signal_variance

    if relative_reference is None:
        relative_reference = modality_uses_relative_reference_contrast(imaging_model_name)

    variance = contrast_noise_variance_counts(
        signal_counts,
        reference_counts,
        params,
        relative_reference=relative_reference,
        variance_floor=variance_floor,
        runtime=runtime,
    )
    method = BackgroundSubtractionSettings.from_params(params).method
    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        assert_compatible(
            context="analysis_contrast_noise_variance",
            measurement_domain="count",
            signal_units="detector_count",
            noise_variance_units="detector_count_squared",
            params=params,
        )
    elif bool(relative_reference):
        assert_compatible(
            context="analysis_contrast_noise_variance",
            measurement_domain="contrast",
            signal_units="relative_reference",
            noise_variance_units="relative_reference_squared",
            params=params,
        )
    else:
        if is_electron_modality(imaging_model_name):
            # TEM/SEM contrast frames are electron-count differences; keep the
            # variance contract in that same domain for Fisher consumers.
            assert_compatible(
                context="analysis_contrast_noise_variance",
                measurement_domain="electron_count",
                signal_units="electron_count",
                noise_variance_units="electron_count_squared",
                params=params,
            )
        else:
            assert_compatible(
                context="analysis_contrast_noise_variance",
                measurement_domain="count",
                signal_units="detector_count",
                noise_variance_units="detector_count_squared",
                params=params,
            )
    return variance


def analysis_contrast_noise_model(
    signal_counts: np.ndarray | float,
    reference_counts: np.ndarray | float | None,
    params: dict[str, Any] | None = None,
    *,
    relative_reference: bool | None = None,
    variance_floor: float = 1e-30,
    runtime: DetectorNoiseRuntime | None = None,
) -> AnalysisNoiseModel:
    """Return a typed analysis-noise model including structured covariance."""
    params = dict(params or {})
    variance = analysis_contrast_noise_variance(
        signal_counts,
        reference_counts,
        params,
        relative_reference=relative_reference,
        variance_floor=variance_floor,
        runtime=runtime,
    )
    cfg = CameraNoiseConfig.from_params(params)
    method = BackgroundSubtractionSettings.from_params(params).method
    imaging_model_name = ModalitySettings.from_params(params).modality
    reference_noise_contract = _analysis_reference_frame_noise_contract(params)
    from imaging_models import get_imaging_model_class

    output_type = getattr(get_imaging_model_class(imaging_model_name), "output_type", "intensity")
    if relative_reference is None:
        relative_reference = modality_uses_relative_reference_contrast(imaging_model_name)

    if output_type == "phase" and method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        measurement_domain = "phase"
        signal_units = "radian"
        noise_units = "radian_squared"
    elif method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        measurement_domain = "count"
        signal_units = "detector_count"
        noise_units = "detector_count_squared"
    elif bool(relative_reference):
        measurement_domain = "contrast"
        signal_units = "relative_reference"
        noise_units = "relative_reference_squared"
    else:
        if is_electron_modality(imaging_model_name):
            # Analysis contrast and likelihood metadata must match the TEM/SEM
            # renderer basis.  The numeric variance is unchanged here; the
            # detector-frame resolver above prevents the camera-gain path from
            # being applied to electron-count arrays.
            measurement_domain = "electron_count"
            signal_units = "electron_count"
            noise_units = "electron_count_squared"
        else:
            measurement_domain = "count"
            signal_units = "detector_count"
            noise_units = "detector_count_squared"

    line_variance = float(cfg.scan_line_noise_counts) ** 2
    row_variance = 0.0
    row_component_variances: tuple[float, ...] = ()
    row_couplings = None
    safe = True
    reason = ""
    covariance_kind = "independent_pixels"
    if line_variance > 0.0:
        covariance_kind = "row_correlated_scan_lines"
        signal_arr = np.asarray(signal_counts, dtype=float)
        if output_type == "phase":
            safe = False
            reason = (
                "scan-line covariance after phase contrast requires an explicit "
                "phase-domain detector covariance calibration."
            )
        elif method in RAW_BACKGROUND_SUBTRACTION_METHODS:
            row_variance = line_variance
            row_component_variances = (line_variance,)
            row_couplings = np.ones((1, *signal_arr.shape), dtype=float)
        elif method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS and reference_counts is not None:
            reference_arr = np.asarray(reference_counts, dtype=float)
            if bool(relative_reference):
                signal_mean, reference_mean = detector_photoresponse_frames_for_analysis(
                    signal_arr,
                    reference_arr,
                    params,
                    runtime=runtime,
                )
                ref_safe = np.maximum(np.abs(reference_mean), 1e-12)
                signal_coupling = 1.0 / ref_safe
                reference_coupling = -signal_mean / (ref_safe * ref_safe)
            else:
                signal_coupling = np.ones_like(signal_arr, dtype=float)
                reference_coupling = -np.ones_like(signal_arr, dtype=float)
                row_variance = 2.0 * line_variance
            row_component_variances = (line_variance, line_variance)
            row_couplings = np.stack(
                [
                    np.asarray(signal_coupling, dtype=float),
                    np.asarray(reference_coupling, dtype=float),
                ],
                axis=0,
            )
        else:
            safe = False
            reason = "video/unsupported background subtraction lacks stack-aware covariance."

    if not reason and reference_noise_contract["analysis_reference_frame_noise_contract"] != "not_applicable":
        reason = str(reference_noise_contract["analysis_reference_frame_stochastic_model"])

    if row_couplings is not None:
        couplings = np.asarray(row_couplings, dtype=float)
        component_vars = np.asarray(row_component_variances, dtype=float)
        variance_arr = np.asarray(variance, dtype=float)
        if couplings.ndim != 3:
            raise RuntimeError(
                "AnalysisNoiseModel row_correlated_couplings must have shape "
                "(components, rows, columns)."
            )
        if component_vars.shape != (couplings.shape[0],):
            raise RuntimeError(
                "AnalysisNoiseModel row_correlated_component_variances must have "
                "one variance per row-correlated coupling component."
            )
        row_cov_diag = np.sum(
            component_vars[:, None, None] * np.square(couplings),
            axis=0,
        )
        if variance_arr.shape != row_cov_diag.shape:
            raise RuntimeError(
                "AnalysisNoiseModel diagonal_variance shape does not match the "
                "row-correlated covariance diagonal shape."
            )
        finite_scale = float(np.nanmax(np.abs(variance_arr))) if variance_arr.size else 0.0
        tol = 1e-12 * max(1.0, finite_scale)
        if np.any(variance_arr + tol < row_cov_diag):
            raise RuntimeError(
                "AnalysisNoiseModel diagonal_variance does not include the "
                "row-correlated scan-line covariance diagonal. This would make "
                "Fisher precision subtract more row covariance than exists in the "
                "declared diagonal variance."
            )

    return AnalysisNoiseModel(
        diagonal_variance=variance,
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_units,
        covariance_kind=covariance_kind,
        row_correlated_variance=row_variance,
        row_correlated_component_variances=row_component_variances,
        row_correlated_couplings=row_couplings,
        safe_for_ordering=safe,
        safe_for_fusion=safe,
        status_reason=reason,
    )


def estimate_detector_noise_std_counts(
    signal_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Return detector-noise standard deviation in camera counts/ADU.

    This is the counts-domain helper expected by supervision and audit code.
    It delegates to the canonical total_noise_std_counts implementation so
    rendering, supervision, and metadata use the same camera-noise model.
    """
    return total_noise_std_counts(signal_counts, params, runtime=runtime)


def _seed_component(value: float) -> int:
    return int(round(abs(float(value)) * 1_000_000_000.0)) % (2**32)


def _detector_static_noise_seed(
    params: dict[str, Any] | None,
    runtime: DetectorNoiseRuntime | None = None,
) -> int:
    """
    Return a simulation-stable seed for static detector maps.

    Fixed-pattern gain/offset maps and hot-pixel locations are detector
    properties: they must persist across frames and across signal/reference
    renders within one simulated video. The public random_seed gives
    reproducible per-video detector maps; if no seed is supplied, a run-scoped
    internal seed is assigned once to the active parameter dictionary.
    """
    if not isinstance(params, dict):
        return 0
    state = runtime_state(params)
    if state.detector_static_seed is not None:
        return int(state.detector_static_seed) % (2**32)
    acquisition_seed = AcquisitionProfile.from_params(params).random_seed
    if acquisition_seed is not None:
        seed = derive_seed(int(acquisition_seed), stream="detector_static_maps", bits=32)
    else:
        seed = int(_noise_runtime(runtime).rng.integers(0, 2**32, dtype=np.uint32))
    state.detector_static_seed = seed
    return seed


def _static_detector_maps(
    shape: tuple[int, ...],
    cfg: CameraNoiseConfig,
    params: dict[str, Any] | None,
    runtime: DetectorNoiseRuntime | None = None,
) -> dict[str, np.ndarray]:
    active_runtime = _noise_runtime(runtime)
    seed = _detector_static_noise_seed(params, active_runtime)
    key = (
        seed,
        tuple(int(x) for x in shape),
        _seed_component(cfg.fixed_pattern_gain_std),
        _seed_component(cfg.fixed_pattern_offset_counts),
        _seed_component(cfg.hot_pixel_fraction),
    )
    with active_runtime.lock:
        cached = active_runtime.static_detector_map_cache.get(key)
        if cached is not None:
            return cached

    static_rng = rng_from_seed_words(
        [
            seed,
            len(shape),
            *[int(x) % (2**32) for x in shape],
            key[2],
            key[3],
            key[4],
        ],
        stream="detector_static_maps",
    )
    maps: dict[str, np.ndarray] = {}
    if cfg.fixed_pattern_gain_std > 0.0:
        maps["gain"] = static_rng.normal(
            loc=1.0,
            scale=cfg.fixed_pattern_gain_std,
            size=shape,
        )
    if cfg.fixed_pattern_offset_counts > 0.0:
        maps["offset"] = static_rng.normal(
            loc=0.0,
            scale=cfg.fixed_pattern_offset_counts,
            size=shape,
        )
    if cfg.hot_pixel_fraction > 0.0:
        maps["hot_mask"] = static_rng.random(size=shape) < cfg.hot_pixel_fraction

    with active_runtime.lock:
        active_runtime.static_detector_map_cache[key] = maps
    return maps



def apply_camera_noise_counts(
    frame_counts: np.ndarray,
    params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
    random_seed: int | None = None,
    stochastic: bool = True,
) -> np.ndarray:
    """Apply camera noise to a frame already expressed in camera counts/ADU."""
    if rng is not None and random_seed is not None:
        raise ValueError("Pass either rng or random_seed to apply_camera_noise_counts(), not both.")
    if runtime is not None and random_seed is not None:
        raise ValueError("Pass either runtime or random_seed to apply_camera_noise_counts(), not both.")
    active_runtime = (
        detector_noise_runtime_from_seed(random_seed)
        if random_seed is not None
        else _noise_runtime(runtime)
    )
    cfg = CameraNoiseConfig.from_params(params)
    rng = rng if rng is not None else active_runtime.rng

    clean = np.asarray(frame_counts, dtype=float)
    shape = clean.shape
    noisy = clean.copy()
    fixed_pattern_gain_map, _ = _resolve_map_shape(
        "fixed_pattern_gain_map",
        cfg.fixed_pattern_gain_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    fixed_pattern_offset_map, _ = _resolve_map_shape(
        "fixed_pattern_offset_map",
        cfg.fixed_pattern_offset_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    scmos_gain_map, _ = _resolve_map_shape("scmos_gain_map", cfg.scmos_gain_map, shape, dtype=float, runtime=active_runtime)
    scmos_variance_map, _ = _resolve_map_shape(
        "scmos_variance_map",
        cfg.scmos_variance_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    scmos_read_noise_map, _ = _resolve_map_shape(
        "scmos_read_noise_map",
        cfg.scmos_read_noise_map,
        shape,
        dtype=float,
        runtime=active_runtime,
    )
    flat_field_map, _ = _resolve_map_shape("flat_field_map", cfg.flat_field_map, shape, dtype=float, runtime=active_runtime)
    hot_pixel_mask, _ = _resolve_boolean_mask("hot_pixel_mask", cfg.hot_pixel_mask, runtime=active_runtime)

    static_maps: dict[str, np.ndarray] = {}
    if (
        (fixed_pattern_gain_map is None and cfg.fixed_pattern_gain_std > 0.0)
        or (fixed_pattern_offset_map is None and cfg.fixed_pattern_offset_counts > 0.0)
        or (hot_pixel_mask is None and cfg.hot_pixel_fraction > 0.0)
    ):
        static_maps = _static_detector_maps(clean.shape, cfg, params, runtime=active_runtime)

    if fixed_pattern_gain_map is None and "gain" in static_maps:
        fixed_pattern_gain_map = static_maps["gain"]
    if fixed_pattern_offset_map is None and "offset" in static_maps:
        fixed_pattern_offset_map = static_maps["offset"]
    if hot_pixel_mask is None and "hot_mask" in static_maps:
        hot_pixel_mask = static_maps["hot_mask"]

    if fixed_pattern_gain_map is not None:
        fixed_pattern_gain_map = _map_to_array(fixed_pattern_gain_map, shape)
    if fixed_pattern_offset_map is not None:
        fixed_pattern_offset_map = _map_to_array(fixed_pattern_offset_map, shape)
    if scmos_variance_map is not None:
        scmos_variance_map = _map_to_array(scmos_variance_map, shape)
    if scmos_gain_map is not None:
        scmos_gain_map = _map_to_array(scmos_gain_map, shape)
    if scmos_read_noise_map is not None:
        scmos_read_noise_map = _map_to_array(scmos_read_noise_map, shape)
    if flat_field_map is not None:
        flat_field_map = _map_to_array(flat_field_map, shape)
    if hot_pixel_mask is not None:
        hot_pixel_mask = _mask_to_array(hot_pixel_mask, shape)

    stages = _detector_transfer_stages(clean, params, runtime=active_runtime)
    # The Poisson driver is resolved once by the detector-transfer stage
    # contract.  Rebuilding it here would let raw stochastic frames and
    # Fisher/CRLB variance diverge for additive offset terms.
    noisy = np.asarray(stages.poisson_driver_mean_counts, dtype=float)

    if stochastic and cfg.shot_noise_enabled:
        counts_pos = np.where(np.isfinite(noisy) & (noisy > 0.0), noisy, 0.0)
        detected_counts_mean = counts_pos
        electron_mean = detected_counts_mean * cfg.camera_gain_e_per_count
        electron_sample = rng.poisson(electron_mean).astype(float)
        poisson_counts = electron_sample / cfg.camera_gain_e_per_count
        shot_residual = poisson_counts - detected_counts_mean
        noisy = detected_counts_mean + shot_residual
        if cfg.emccd_enabled and cfg.emccd_shot_variance_multiplier > 1.0:
            # The stochastic branch must match the Fisher branch exactly: Poisson
            # contributes one unit of variance, and EMCCD multiplication adds the
            # remaining canonical variance-multiplier excess.
            excess_var = np.maximum(detected_counts_mean / cfg.camera_gain_e_per_count, 0.0) * (cfg.emccd_shot_variance_multiplier - 1.0)
            noisy = noisy + rng.normal(scale=np.sqrt(excess_var), size=clean.shape)

    if np.any(stages.post_poisson_pre_gain_additive_counts != 0.0):
        noisy = noisy + stages.post_poisson_pre_gain_additive_counts

    if flat_field_map is not None:
        noisy = noisy * flat_field_map

    if fixed_pattern_gain_map is not None:
        noisy = noisy * fixed_pattern_gain_map
    elif cfg.fixed_pattern_gain_std > 0.0:
        noisy = noisy * static_maps["gain"]

    if scmos_gain_map is not None:
        noisy = noisy * scmos_gain_map

    if np.any(stages.post_gain_additive_counts != 0.0):
        noisy = noisy + stages.post_gain_additive_counts

    if fixed_pattern_offset_map is not None:
        noisy = noisy + fixed_pattern_offset_map
    elif cfg.fixed_pattern_offset_counts > 0.0:
        noisy = noisy + static_maps["offset"]

    if hot_pixel_mask is not None:
        hot_mask = hot_pixel_mask
        if cfg.hot_pixel_value_counts is None:
            finite = noisy[np.isfinite(noisy)]
            hot_value = float(finite.max()) if finite.size else 0.0
        else:
            hot_value = cfg.hot_pixel_value_counts
        noisy[hot_mask] = hot_value

    if stochastic and cfg.scan_line_noise_counts > 0.0:
        noisy = noisy + rng.normal(scale=cfg.scan_line_noise_counts, size=(clean.shape[0], 1))

    if stochastic and cfg.gaussian_noise_enabled:
        has_calibrated_read_map = scmos_variance_map is not None or scmos_read_noise_map is not None
        if has_calibrated_read_map and cfg.read_noise_map_mode == "replace":
            read_variance = np.zeros(shape, dtype=float)
        else:
            read_variance = np.full(shape, cfg.read_noise_counts, dtype=float) ** 2
        if scmos_variance_map is not None:
            read_variance = read_variance + scmos_variance_map
        if scmos_read_noise_map is not None:
            read_variance = read_variance + np.square(scmos_read_noise_map)
        if cfg.emccd_enabled and cfg.emccd_gain > 1.0:
            read_variance = read_variance / (cfg.emccd_gain ** 2)
        read_sigma = np.sqrt(np.maximum(read_variance, 0.0))
        if np.any(read_sigma > 0.0):
            noisy = noisy + rng.normal(scale=read_sigma, size=clean.shape)

    if cfg.saturation_e is not None:
        noisy = np.minimum(noisy, cfg.saturation_e / cfg.camera_gain_e_per_count)
    if cfg.saturation_level is not None:
        noisy = np.minimum(noisy, cfg.saturation_level)
    if cfg.adc_quantization:
        noisy = np.round(noisy / cfg.adc_quantization_counts) * cfg.adc_quantization_counts

    if cfg.clip_output_to_nonnegative:
        noisy = np.where(np.isfinite(noisy), np.maximum(noisy, 0.0), 0.0)

    return noisy.astype(float, copy=False)


def deterministic_detector_transfer_counts(
    frame_counts: np.ndarray | float,
    params: dict[str, Any] | None = None,
    *,
    runtime: DetectorNoiseRuntime | None = None,
) -> np.ndarray:
    """
    Apply the deterministic detector transfer without stochastic shot/read noise.

    This is the canonical owner for public pre-noise detector-mean frames. It
    uses the same transfer order as apply_camera_noise_counts(...), with
    Poisson/read/scan-line sampling disabled.
    """
    return apply_camera_noise_counts(
        frame_counts,
        params,
        runtime=runtime,
        stochastic=False,
    )



def estimate_contrast_noise_std_from_params(params: dict[str, Any]) -> float:
    """
    Estimate noise in contrast units for supervision support gates.

    For count images, contrast noise is noise_counts / reference_counts. This is
    deliberately tied to the same camera-count model used by rendering.
    """
    imaging_model = ModalitySettings.from_params(params).modality
    counts = CountBudgetSettings.from_params(params)

    if imaging_model in {"dark_field", "coherent_dark_field"}:
        background_counts = max(counts.dark_field_background_count, 1e-9)
        normalization_counts = max(counts.dark_field_illumination_count, 1e-9)
    else:
        background_counts = max(counts.background_intensity, 1e-9)
        normalization_counts = background_counts

    noise_counts = float(total_noise_std_counts(background_counts, params))
    return float(noise_counts / normalization_counts)


def calibrate_camera_gain_e_per_count_from_video(
    video_path: str,
    *,
    max_frames: int = 80,
    sample_stride: int = 1,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    min_gain: float = 0.25,
    max_gain: float = 256.0,
) -> float:
    """
    Estimate an effective camera_gain_e_per_count from a real video.

    This is a practical count-domain calibration for dataset matching. It uses
    temporal variance over relatively flat pixels and estimates

        variance_counts ~= mean_counts / camera_gain_e_per_count.

    The result is an effective camera-conversion number for simulation matching,
    not a claim about the physical camera manual.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video calibration.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for noise calibration: {video_path}")

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % sample_stride == 0:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray.astype(np.float32))
        idx += 1
    cap.release()

    if len(frames) < 8:
        return 1.0

    stack = np.stack(frames, axis=0)
    temporal_mean = np.mean(stack, axis=0)
    temporal_var = np.var(stack, axis=0, ddof=1)

    # Reject saturated/extreme pixels and high-structure pixels so sample
    # features and background gradients do not dominate the noise estimate.
    lo = np.quantile(temporal_mean, lower_quantile)
    hi = np.quantile(temporal_mean, upper_quantile)
    gx = np.zeros_like(temporal_mean)
    gy = np.zeros_like(temporal_mean)
    gx[:, 1:-1] = np.abs(temporal_mean[:, 2:] - temporal_mean[:, :-2])
    gy[1:-1, :] = np.abs(temporal_mean[2:, :] - temporal_mean[:-2, :])
    grad = gx + gy
    grad_cut = np.quantile(grad, 0.50)
    mask = (
        np.isfinite(temporal_mean)
        & np.isfinite(temporal_var)
        & (temporal_mean > lo)
        & (temporal_mean < hi)
        & (temporal_var > 1e-6)
        & (grad <= grad_cut)
    )

    if np.count_nonzero(mask) < 100:
        mask = np.isfinite(temporal_mean) & np.isfinite(temporal_var) & (temporal_var > 1e-6)

    mean_counts = float(np.median(temporal_mean[mask]))
    var_counts = float(np.median(temporal_var[mask]))
    if not np.isfinite(mean_counts) or not np.isfinite(var_counts) or mean_counts <= 0.0 or var_counts <= 0.0:
        return 1.0

    gain = mean_counts / var_counts
    gain = max(float(min_gain), min(float(max_gain), float(gain)))
    return float(gain)
