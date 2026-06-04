"""Executable comparison, validity, backend, and artifact contracts for Syniscopy."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib, json
from typing import Any, Mapping, Sequence
import numpy as np
from json_utils import json_safe_with_nonfinite_tags

CONTRACT_SCHEMA_VERSION = "syniscopy-comparison-contract-v1"
FISHER_RESULT_SCHEMA_VERSION = "syniscopy-fisher-result-v1"
BACKEND_CONTRACT_SCHEMA_VERSION = "syniscopy-backend-contract-v1"
MODEL_CARD_SCHEMA_VERSION = "syniscopy-model-card-v1"
ARTIFACT_GRAPH_SCHEMA_VERSION = "syniscopy-artifact-graph-v1"

class ConvergenceStatus(str, Enum):
    """Canonical scalar convergence-status vocabulary.

    Paper-facing gates should store this enum's string values in scalar
    ``convergence_status`` fields. Structured per-row diagnostics should carry
    one of these scalar values inside a Fisher convergence metadata envelope.
    """

    FINITE_CONVERGED = "finite_converged"
    STABLE_SINGULAR = "stable_singular"
    FAILED_CONVERGENCE = "failed_convergence"
    ILL_CONDITIONED = "ill_conditioned"
    NONFINITE = "nonfinite"
    NOT_APPLICABLE = "not_applicable"
    UNCHECKED = "unchecked"
    PRODUCTION_GRID_ONLY = "production_grid_only"
    EXTERNAL_ARTIFACT_REQUIRED = "external_artifact_required"

class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    INVALID = "invalid"
    UNCHECKED = "unchecked"
    EXTERNAL_ARTIFACT_REQUIRED = "external_artifact_required"

class FisherMode(str, Enum):
    GAUSSIAN_FIXED_VARIANCE = "gaussian_fixed_variance"
    POISSON_EXACT = "poisson_exact"
    GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE = "gaussian_parameter_dependent_variance"
    POISSON_GAUSSIAN_APPROX = "poisson_gaussian_approx"
    POISSON_GAUSSIAN_NUMERICAL = "poisson_gaussian_numerical"
    MEAN_FISHER_DIAGNOSTIC = "mean_fisher_diagnostic"

def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe_with_nonfinite_tags(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

@dataclass(frozen=True)
class ComparisonContract:
    contract_id: str
    contract_name: str
    shared_scene_id: str = "shared-scene-configured-profile"
    particle_state_id: str = "configured-particle-state"
    sample_environment_id: str = "configured-sample-environment"
    allowed_modality_profiles: tuple[str, ...] = ()
    detector_noise_policy: str = "resolved-camera-noise-metadata-required"
    quanta_budget_policy: str = "profile-native-detected-counts"
    dose_cost_policy: str = "not-normalized-unless-cost-contract-present"
    native_regime_policy: str = "configured-profile-not-native-benchmark"
    physical_compatibility_policy: str = "explicit-status-required-for-fusion"
    fisher_mode: str = FisherMode.GAUSSIAN_FIXED_VARIANCE.value
    derivative_method: str = "stationary_shift_or_rerendered_with_metadata"
    convergence_policy: str = "status-required"
    validity_policy: str = "downstream-results-inherit-parent-status"
    output_units: Mapping[str, str] = field(default_factory=lambda: {"lateral_crlb": "nm", "axial_crlb": "nm"})
    ranking_objective: str = "sigma_xy_nm"
    constraints: Mapping[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    paper_label: str = ""
    schema_version: str = CONTRACT_SCHEMA_VERSION
    def to_dict(self) -> dict[str, Any]:
        d = json_safe_with_nonfinite_tags(asdict(self));
        if not d.get("provenance_hash"):
            payload = dict(d); payload.pop("provenance_hash", None); d["provenance_hash"] = stable_hash(payload)
        return d

@dataclass(frozen=True)
class FisherResult:
    result_id: str
    parent_result_ids: tuple[str, ...]
    source_contract: str
    modality: str
    backend_id: str
    profile_id: str
    fisher_matrix: Any
    covariance_or_pseudoinverse: Any = None
    crlb_summary: Mapping[str, Any] = field(default_factory=dict)
    derivative_method: str = "unchecked"
    derivative_step: Any = None
    derivative_units: Mapping[str, str] = field(default_factory=dict)
    candidate_steps: Mapping[str, Sequence[float]] = field(default_factory=dict)
    convergence_status: str = ConvergenceStatus.UNCHECKED.value
    singular_axes: tuple[str, ...] = ()
    rank: int | None = None
    rank_tolerance: float = 1e-12
    condition_number: float | None = None
    model_fidelity: str = "model_conditional"
    validation_status: str = ValidationStatus.UNCHECKED.value
    production_grid_diagnostic: bool = False
    safe_for_ordering: bool = False
    safe_for_fusion: bool = False
    safe_for_time_allocation: bool = False
    safe_for_registration: bool = False
    safe_for_detected_quanta_ranking: bool = False
    notes: tuple[str, ...] = ()
    schema_version: str = FISHER_RESULT_SCHEMA_VERSION
    def __post_init__(self) -> None:
        object.__setattr__(self, "convergence_status", normalize_convergence_status(self.convergence_status))
    def to_dict(self) -> dict[str, Any]: return json_safe_with_nonfinite_tags(asdict(self))

@dataclass(frozen=True)
class BackendContract:
    modality_id: str
    canonical_name: str
    paper_label: str
    backend_family: str
    uses_scalar_scattered_field: bool = False
    uses_vectorial_field: bool = False
    uses_incoherent_source_map: bool = False
    uses_electron_projected_potential: bool = False
    uses_probe_scan: bool = False
    uses_reference_interference: bool = False
    uses_emission_psf: bool = False
    uses_sample_environment: bool = True
    uses_thinfilm_stack: bool = False
    uses_phase_object: bool = False
    uses_detector_counts: bool = True
    uses_phase_units: bool = False
    native_units: str = "detector_count"
    measurement_domain: str = "count"
    signal_units: str = "detector_count"
    contrast_frame_units: str = "detector_count_or_contrast"
    axial_sensitivity_mechanism: str = "not_declared"
    axial_sensitive: str = "conditional"
    source_input_kind: str = "not_declared"
    source_map_ndim: int | None = None
    source_axis_order: str | None = None
    source_projection_policy: str | None = None
    backend_consumes_volume_source: bool = False
    volume_transport_model: str | None = None
    required_material_fields: tuple[str, ...] = ()
    required_param_fields: tuple[str, ...] = ()
    required_detector_fields: tuple[str, ...] = ("camera_gain_e_per_count", "read_noise_counts", "detector_qe")
    known_omissions: tuple[str, ...] = ()
    fidelity_class: str = "model_conditional"
    backend_fidelity_level: str = "proxy"
    reference_backend_metadata: Mapping[str, Any] | None = None
    validation_status: str = ValidationStatus.DIAGNOSTIC_ONLY.value
    schema_version: str = BACKEND_CONTRACT_SCHEMA_VERSION
    @property
    def backend_id(self) -> str: return f"{self.modality_id}:{self.backend_family}"
    def to_dict(self) -> dict[str, Any]:
        d = json_safe_with_nonfinite_tags(asdict(self)); d["backend_id"] = self.backend_id; return d

@dataclass(frozen=True)
class DetectorModel:
    detector_qe: float
    detector_input_is_incident_quanta: bool
    emccd_enabled: bool
    emccd_gain: float
    emccd_excess_noise_factor: float
    camera_gain_e_per_count: float
    dark_offset_counts: float
    read_noise_e: float | None
    read_noise_counts: float
    saturation_level: float | None
    saturation_e: float | None
    dark_current_e_per_pixel_per_s: float
    exposure_time_s: float
    fixed_pattern_gain_std: float
    fixed_pattern_offset_counts: float
    hot_pixel_fraction: float
    hot_pixel_value_counts: float | None
    fixed_pattern_gain_map: str | None
    fixed_pattern_offset_map: str | None
    hot_pixel_mask: str | None
    scmos_variance_map: str | None
    scmos_gain_map: str | None
    scmos_read_noise_map: str | None
    read_noise_map_mode: str
    scan_line_noise_counts: float
    detector_noise_input_domain: str
    nonlinear_detector_effects_active: bool
    deterministic_detector_transfer_active: bool
    safe_for_linear_fisher_variance: bool
    adc_quantization: bool
    adc_quantization_counts: float
    clip_output_to_nonnegative: bool
    noise_parameterization: str
    nonlinearity_calibration: str | None
    background_offset_counts: float
    flat_field_map: str | None
    dark_frame_map: str | None
    def to_dict(self) -> dict[str, Any]: return json_safe_with_nonfinite_tags(asdict(self))

@dataclass(frozen=True)
class AcquisitionCostModel:
    photons_detected: float | None = None
    photons_incident: float | None = None
    electron_dose: float | None = None
    incident_primary_electrons_per_pixel: float | None = None
    detected_electron_count_kind: str | None = None
    configured_count_budget: float | None = None
    configured_count_budget_units: str | None = None
    count_budget_source: str | None = None
    count_budget_semantics: str | None = None
    dwell_time_s: float | None = None
    exposure_time_s: float | None = None
    photobleaching_cost: float | None = None
    radiation_damage_cost: float | None = None
    heating_cost: float | None = None
    sample_preparation_class: str = "not_declared"
    destructive: bool = False
    live_compatible: bool = True
    vacuum_compatible: bool = False
    field_of_view_um: float | None = None
    throughput_hz: float | None = None
    frame_time_s: float | None = None
    switching_setup_cost_s: float | None = None
    dose_budget: float | None = None
    bleaching_budget: float | None = None
    def to_dict(self) -> dict[str, Any]: return json_safe_with_nonfinite_tags(asdict(self))

@dataclass(frozen=True)
class ArtifactNode:
    artifact_id: str
    artifact_path: str
    artifact_type: str
    source_notebook_or_script: str
    input_config_hash: str = ""
    source_code_hash: str = ""
    output_hash: str = ""
    parent_artifacts: tuple[str, ...] = ()
    child_artifacts: tuple[str, ...] = ()
    model_version: str = ""
    convergence_status: str = ConvergenceStatus.UNCHECKED.value
    validation_status: str = ValidationStatus.UNCHECKED.value
    paper_consumers: tuple[str, ...] = ()
    generation_timestamp: str = ""
    external_asset_dependencies: tuple[str, ...] = ()
    heavy_execution: bool = False
    schema_version: str = ARTIFACT_GRAPH_SCHEMA_VERSION
    def __post_init__(self) -> None:
        object.__setattr__(self, "convergence_status", normalize_convergence_status(self.convergence_status))
    def to_dict(self) -> dict[str, Any]:
        d = json_safe_with_nonfinite_tags(asdict(self));
        if not d.get("input_config_hash"): d["input_config_hash"] = stable_hash({"artifact_id": self.artifact_id, "artifact_path": self.artifact_path})
        return d

def default_comparison_contracts(allowed_modality_profiles: Sequence[str] = ()) -> dict[str, ComparisonContract]:
    profiles = tuple(str(x) for x in allowed_modality_profiles); common = {"allowed_modality_profiles": profiles}
    return {
        "Contract-LP": ComparisonContract("Contract-LP", "Configured-profile lateral CRLB", derivative_method="adaptive_rerendered_xy_or_stationary_shift_with_status", ranking_objective="sigma_xy_nm", paper_label="configured-profile lateral CRLB", **common),
        "Contract-LZ": ComparisonContract("Contract-LZ", "Configured-profile axial/SE(3) CRLB", derivative_method="adaptive_axial_or_se3_with_singularity_status", ranking_objective="sigma_xyz_nm_or_rank", paper_label="configured-profile axial/SE(3) diagnostic", **common),
        "Contract-Q": ComparisonContract("Contract-Q", "Detected-quanta-normalized CRLB", quanta_budget_policy="fixed-total-detected-quanta-with-explicit-distribution", dose_cost_policy="not-dose; cost model required for dose claims", fisher_mode=FisherMode.POISSON_GAUSSIAN_APPROX.value, ranking_objective="sigma_xy_nm_at_detected_quanta_budget", paper_label="detected-quanta-normalized diagnostic", **common),
        "Contract-NR": ComparisonContract("Contract-NR", "Native-regime/source-use reference context", native_regime_policy="source-reported-or-calibrated-native-profile-required", ranking_objective="source-reference-check", paper_label="native-regime reference context", **common),
        "Contract-COST": ComparisonContract("Contract-COST", "Physical cost/dose-aware acquisition contract", dose_cost_policy="explicit-acquisition-cost-model-required", ranking_objective="constrained-cost-objective", constraints={"requires_cost_model": True}, paper_label="cost/dose-aware contract", **common),
        "Contract-FUSION-PHYSICAL": ComparisonContract("Contract-FUSION-PHYSICAL", "Physically feasible fusion contract", physical_compatibility_policy="compatibility-graph-must-allow-subset", ranking_objective="compatible-fused-sigma_xy_nm", constraints={"requires_physical_compatibility": True}, paper_label="physically feasible fusion", **common),
        "Contract-FUSION-ALGEBRAIC": ComparisonContract("Contract-FUSION-ALGEBRAIC", "Algebraic independent-channel Fisher-sum diagnostic", physical_compatibility_policy="diagnostic-only-unless-compatibility-graph-passes", ranking_objective="algebraic-fused-sigma_xy_nm", paper_label="algebraic Fisher-sum diagnostic", **common),
    }

def validate_comparison_contract(contract: Mapping[str, Any]) -> None:
    required = {"contract_id","contract_name","shared_scene_id","particle_state_id","sample_environment_id","allowed_modality_profiles","detector_noise_policy","quanta_budget_policy","dose_cost_policy","native_regime_policy","physical_compatibility_policy","fisher_mode","derivative_method","convergence_policy","validity_policy","output_units","ranking_objective","constraints","provenance_hash","paper_label"}
    missing = sorted(required - set(contract))
    if missing: raise ValueError("comparison contract missing field(s): " + ", ".join(missing))

def contracts_manifest(allowed_modality_profiles: Sequence[str] = ()) -> dict[str, Any]:
    contracts = {k: v.to_dict() for k, v in default_comparison_contracts(allowed_modality_profiles).items()}
    for c in contracts.values(): validate_comparison_contract(c)
    return {"schema_version": CONTRACT_SCHEMA_VERSION, "contracts": contracts}

def _canonical_modality_for_contract(modality: str) -> str:
    """Return the canonical imaging-model key used by public contracts."""
    raw = str(modality).strip()
    from modality_registry import canonical_modality_name

    return canonical_modality_name(raw)


def backend_contract_for_modality(modality: str, response: Mapping[str, Any] | None = None) -> BackendContract:
    m = _canonical_modality_for_contract(modality)
    label = str((response or {}).get("display_name") or m.replace("_", " ").title())
    resp = dict(response or {})
    optical_backend = str(resp.get("optical_field_backend", "vectorial_debye")).strip().lower()
    vectorial_detection_mode = str(resp.get("vectorial_detection_mode", "full_vector")).strip().lower()
    optical_is_full_vector = (
        optical_backend == "vectorial_debye"
        and vectorial_detection_mode == "full_vector"
    )

    if m in {"bright_field", "partially_coherent_bright_field"}:
        return BackendContract(
            m,
            m,
            label,
            "vectorial_partially_coherent_kohler" if optical_is_full_vector else "partially_coherent_kohler",
            uses_scalar_scattered_field=not optical_is_full_vector,
            uses_vectorial_field=optical_is_full_vector,
            native_units="relative_reference",
            measurement_domain="contrast",
            signal_units="relative_reference",
            contrast_frame_units="relative_reference",
            known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",),
            fidelity_class="configured_kohler_vectorial" if optical_is_full_vector else "configured_kohler_scalar",
        )
    if m == "dark_field":
        return BackendContract(
            m,
            m,
            label,
            "vectorial_partially_coherent_kohler" if optical_is_full_vector else "partially_coherent_kohler",
            uses_scalar_scattered_field=not optical_is_full_vector,
            uses_vectorial_field=optical_is_full_vector,
            measurement_domain="count",
            signal_units="detector_count",
            contrast_frame_units="detector_count_difference",
            known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",),
            fidelity_class="configured_kohler_vectorial" if optical_is_full_vector else "configured_kohler_scalar",
        )
    if m in {"coherent_bright_field", "interferometric"}:
        return BackendContract(
            m,
            m,
            label,
            "vectorial_label_free_optical" if optical_is_full_vector else "scalar_label_free_optical",
            uses_scalar_scattered_field=not optical_is_full_vector,
            uses_vectorial_field=optical_is_full_vector,
            uses_reference_interference=(m == "interferometric"),
            native_units="relative_reference",
            measurement_domain="contrast",
            signal_units="relative_reference",
            contrast_frame_units="relative_reference",
            known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",),
            fidelity_class="vectorial_optical" if optical_is_full_vector else "scalar_optical",
        )
    if m == "coherent_dark_field":
        return BackendContract(
            m,
            m,
            label,
            "vectorial_label_free_optical" if optical_is_full_vector else "scalar_label_free_optical",
            uses_scalar_scattered_field=not optical_is_full_vector,
            uses_vectorial_field=optical_is_full_vector,
            measurement_domain="count",
            signal_units="detector_count",
            contrast_frame_units="detector_count_difference",
            known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",),
            fidelity_class="vectorial_optical" if optical_is_full_vector else "scalar_optical",
        )
    if m == "zernike_phase_contrast":
        return BackendContract(m, m, label, "vectorial_phase_imaging" if optical_is_full_vector else "phase_imaging", uses_scalar_scattered_field=not optical_is_full_vector, uses_vectorial_field=optical_is_full_vector, uses_phase_object=True, measurement_domain="count", signal_units="detector_count", contrast_frame_units="detector_count_difference", known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",), fidelity_class="vectorial_phase_ring" if optical_is_full_vector else "scalar_phase_proxy")
    if m == "differential_phase_contrast":
        vectorial_dpc = bool(resp.get("dpc_vectorial_backend_enabled"))
        return BackendContract(
            m,
            m,
            label,
            "phase_imaging",
            uses_scalar_scattered_field=not vectorial_dpc,
            uses_vectorial_field=vectorial_dpc,
            uses_phase_object=True,
            measurement_domain="count",
            signal_units="detector_count",
            contrast_frame_units="detector_count_difference",
            known_omissions=() if vectorial_dpc else ("scalar phase-object approximation",),
            fidelity_class="vectorial_phase_proxy" if vectorial_dpc else "scalar_phase_proxy",
        )
    if m == "quantitative_phase":
        return BackendContract(m, m, label, "vectorial_phase_transfer" if optical_is_full_vector else "phase_transfer", uses_scalar_scattered_field=not optical_is_full_vector, uses_vectorial_field=optical_is_full_vector, uses_phase_object=True, uses_phase_units=True, native_units="radian", measurement_domain="phase", signal_units="radian", contrast_frame_units="radian", known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",), fidelity_class="vectorial_phase_transfer" if optical_is_full_vector else "scalar_phase_proxy")
    if m == "off_axis_holography":
        return BackendContract(m, m, label, "vectorial_holographic_interference" if optical_is_full_vector else "holographic_interference", uses_scalar_scattered_field=not optical_is_full_vector, uses_vectorial_field=optical_is_full_vector, uses_phase_object=True, native_units="detector_count", measurement_domain="fringe_count", signal_units="detector_count", contrast_frame_units="detector_count_difference", known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",), fidelity_class="vectorial_holographic_interference" if optical_is_full_vector else "scalar_phase_proxy")
    if m == "ricm":
        return BackendContract(m, m, label, "vectorial_thinfilm_reflection_interference" if optical_is_full_vector else "thinfilm_reflection_interference", uses_scalar_scattered_field=not optical_is_full_vector, uses_vectorial_field=optical_is_full_vector, uses_reference_interference=True, uses_thinfilm_stack=True, native_units="relative_reference", measurement_domain="contrast", signal_units="relative_reference", contrast_frame_units="relative_reference", known_omissions=() if optical_is_full_vector else ("scalar/analyzer optical field projection",), fidelity_class="thinfilm_vectorial" if optical_is_full_vector else "thinfilm_scalar")
    if m in {"fluorescence_widefield", "tirf_fluorescence"}:
        level = str(resp.get("backend_fidelity_level") or "proxy")
        ref = resp.get("reference_backend_metadata")
        axial = "evanescent excitation weighting of projected source map" if m == "tirf_fluorescence" else "projected emitter source map with optional explicit z-dependent emission PSF"
        return BackendContract(m, m, label, resp.get("fluorescence_backend", "incoherent_fluorescence_source_map" if m == "fluorescence_widefield" else "photophysics_fluorescence"), uses_scalar_scattered_field=False, uses_vectorial_field=(resp.get("fluorescence_backend") == "vectorial_photophysics"), uses_incoherent_source_map=True, uses_emission_psf=True, native_units="detector_count", measurement_domain="count", signal_units="detector_count", contrast_frame_units="detector_count_difference", axial_sensitivity_mechanism=axial, axial_sensitive="conditional" if m == "tirf_fluorescence" else "no", source_input_kind=str(resp.get("source_input_kind") or "projected_2d_fluorophore_emitter_density"), source_map_ndim=resp.get("source_map_ndim"), source_axis_order=resp.get("source_axis_order"), source_projection_policy=resp.get("source_projection_policy"), backend_consumes_volume_source=bool(resp.get("backend_consumes_volume_source", False)), volume_transport_model=resp.get("volume_transport_model"), required_material_fields=("fluorophore_density", "excitation_peak_nm", "emission_peak_nm"), required_param_fields=("fluorescence_quantum_yield",), known_omissions=() if level in {"high_fidelity", "reference_validated"} else ("physical per-fluorophore photon budget/reference calibration not supplied",), fidelity_class=str(resp.get("fidelity_label") or "fluorescence_source_map"), backend_fidelity_level=level, reference_backend_metadata=ref)
    if m == "tem_phase_contrast":
        level = str(resp.get("backend_fidelity_level") or "proxy")
        ref = resp.get("reference_backend_metadata")
        return BackendContract(m, m, label, str(resp.get("tem_backend") or "electron_projected_potential_ctf"), uses_scalar_scattered_field=False, uses_electron_projected_potential=True, native_units="electron_count", measurement_domain="electron_count", signal_units="electron_count", contrast_frame_units="electron_count_difference", axial_sensitivity_mechanism="through-focus/tilt-series required for finite axial sensitivity", axial_sensitive="conditional", source_input_kind=str(resp.get("source_input_kind") or resp.get("tem_source_dimensionality") or "tem_source_backend_gated"), source_map_ndim=resp.get("source_map_ndim"), source_axis_order=resp.get("source_axis_order"), source_projection_policy=resp.get("source_projection_policy"), backend_consumes_volume_source=bool(resp.get("backend_consumes_volume_source", False)), volume_transport_model=resp.get("volume_transport_model"), required_material_fields=("projected_electrostatic_potential", "thickness_nm"), known_omissions=() if level != "proxy" else ("full multislice/reference validation not active unless selected",), fidelity_class=str(resp.get("fidelity_label") or "electron_ctf_proxy"), backend_fidelity_level=level, reference_backend_metadata=ref)
    if m == "sem_secondary_electron":
        level = str(resp.get("backend_fidelity_level") or "proxy")
        ref = resp.get("reference_backend_metadata")
        return BackendContract(m, m, label, str(resp.get("sem_backend") or "sem_probe_secondary_yield"), uses_scalar_scattered_field=False, uses_probe_scan=True, native_units="secondary_electron_yield", measurement_domain="electron_count", signal_units="electron_count", contrast_frame_units="electron_count_difference", axial_sensitivity_mechanism="topography/focus/tilt/interaction-volume conditional", axial_sensitive="conditional", source_input_kind=str(resp.get("source_input_kind") or "projected_2d_source_map"), source_map_ndim=resp.get("source_map_ndim"), source_axis_order=resp.get("source_axis_order"), source_projection_policy=resp.get("source_projection_policy"), backend_consumes_volume_source=bool(resp.get("backend_consumes_volume_source", False)), volume_transport_model=resp.get("volume_transport_model"), required_material_fields=("secondary_electron_yield", "topography"), known_omissions=() if level in {"high_fidelity", "reference_validated"} else ("reference-kernel validation required for native benchmark labeling",), fidelity_class=str(resp.get("fidelity_label") or "sem_yield_proxy"), backend_fidelity_level=level, reference_backend_metadata=ref)
    return BackendContract(m, m, label, "undeclared_backend", validation_status=ValidationStatus.UNCHECKED.value, known_omissions=("backend declaration missing",))

def detector_model_from_params(params: Mapping[str, Any]) -> DetectorModel:
    from camera_noise import resolve_camera_noise_config

    cfg = resolve_camera_noise_config(dict(params))
    return DetectorModel(
        detector_qe=float(cfg.detector_qe),
        detector_input_is_incident_quanta=bool(cfg.detector_input_is_incident_quanta),
        emccd_enabled=bool(cfg.emccd_enabled),
        emccd_gain=float(cfg.emccd_gain),
        emccd_excess_noise_factor=float(cfg.emccd_excess_noise_factor),
        camera_gain_e_per_count=float(cfg.camera_gain_e_per_count),
        dark_offset_counts=float(cfg.dark_offset_counts),
        read_noise_e=cfg.read_noise_e,
        read_noise_counts=float(cfg.read_noise_counts),
        saturation_level=cfg.saturation_level,
        saturation_e=cfg.saturation_e,
        dark_current_e_per_pixel_per_s=float(cfg.dark_current_e_per_pixel_per_s),
        exposure_time_s=float(cfg.exposure_time_s),
        fixed_pattern_gain_std=float(cfg.fixed_pattern_gain_std),
        fixed_pattern_offset_counts=float(cfg.fixed_pattern_offset_counts),
        hot_pixel_fraction=float(cfg.hot_pixel_fraction),
        hot_pixel_value_counts=cfg.hot_pixel_value_counts,
        fixed_pattern_gain_map=cfg.fixed_pattern_gain_map,
        fixed_pattern_offset_map=cfg.fixed_pattern_offset_map,
        hot_pixel_mask=cfg.hot_pixel_mask,
        scmos_variance_map=cfg.scmos_variance_map,
        scmos_gain_map=cfg.scmos_gain_map,
        scmos_read_noise_map=cfg.scmos_read_noise_map,
        read_noise_map_mode=str(cfg.read_noise_map_mode),
        scan_line_noise_counts=float(cfg.scan_line_noise_counts),
        detector_noise_input_domain=str(cfg.detector_noise_input_domain),
        nonlinear_detector_effects_active=bool(cfg.nonlinear_detector_effects_active),
        deterministic_detector_transfer_active=bool(cfg.deterministic_detector_transfer_active),
        safe_for_linear_fisher_variance=bool(cfg.safe_for_linear_fisher_variance),
        adc_quantization=bool(cfg.adc_quantization),
        adc_quantization_counts=float(cfg.adc_quantization_counts),
        clip_output_to_nonnegative=bool(cfg.clip_output_to_nonnegative),
        noise_parameterization=str(cfg.noise_parameterization),
        nonlinearity_calibration=cfg.nonlinearity_calibration,
        background_offset_counts=float(cfg.background_offset_counts),
        flat_field_map=cfg.flat_field_map,
        dark_frame_map=cfg.dark_frame_map,
    )

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def acquisition_cost_from_profile(modality: str, params: Mapping[str, Any] | None = None) -> AcquisitionCostModel:
    p = {} if params is None else dict(params)
    m = str(modality).strip().lower().replace("-", "_").replace(" ", "_")
    electron = m.startswith("tem") or m.startswith("sem")
    fluorescence = "fluorescence" in m or "tirf" in m

    photons_detected = _optional_float(p.get("photons_detected"))
    photons_incident = _optional_float(p.get("photons_incident"))
    electron_dose = _optional_float(p.get("electron_dose"))
    incident_primary_electrons = None
    detected_electron_count_kind = None
    configured_count_budget = None
    configured_count_budget_units = None
    count_budget_source = None
    count_budget_semantics = None

    if m.startswith("tem"):
        incident_primary_electrons = _optional_float(
            p.get("tem_dose_per_pixel", p.get("electron_dose"))
        )
        electron_dose = electron_dose if electron_dose is not None else incident_primary_electrons
        configured_count_budget = incident_primary_electrons
        configured_count_budget_units = "incident_primary_electrons_per_pixel"
        count_budget_source = "tem_dose_per_pixel" if "tem_dose_per_pixel" in p else "electron_dose"
        detected_electron_count_kind = "transmitted_primary_electron_count"
        count_budget_semantics = (
            "TEM signal formation multiplies normalized direct-beam intensity by "
            "incident primary-electron dose per pixel; output counts are expected "
            "transmitted/detected primary-electron counts under unit collection."
        )
    elif m.startswith("sem"):
        incident_primary_electrons = _optional_float(
            p.get("sem_electrons_per_pixel", p.get("electron_dose"))
        )
        electron_dose = electron_dose if electron_dose is not None else incident_primary_electrons
        configured_count_budget = incident_primary_electrons
        configured_count_budget_units = "incident_primary_electrons_per_pixel"
        count_budget_source = "sem_electrons_per_pixel" if "sem_electrons_per_pixel" in p else "electron_dose"
        detected_electron_count_kind = "yield_weighted_secondary_electron_count"
        count_budget_semantics = (
            "SEM signal formation multiplies secondary-electron yield by incident "
            "primary-electron dose per pixel; detected count images are "
            "yield-weighted expected secondary-electron counts, not equal-dose "
            "TEM transmitted-electron counts."
        )
    elif fluorescence:
        collection = _optional_float(p.get("fluorescence_collection_efficiency")) or 1.0
        qe = _optional_float(p.get("fluorescence_detector_qe", p.get("detector_qe"))) or 1.0
        if p.get("fluorescence_photons_per_fluorophore_per_frame") is not None:
            emitted = _optional_float(p.get("fluorescence_photons_per_fluorophore_per_frame"))
            configured_count_budget = emitted * collection * qe if emitted is not None else None
            configured_count_budget_units = "detected_counts_per_fluorophore_per_frame"
            count_budget_source = "fluorescence_photons_per_fluorophore_per_frame"
            photons_incident = photons_incident if photons_incident is not None else emitted
            photons_detected = photons_detected if photons_detected is not None else configured_count_budget
            count_budget_semantics = (
                "Fluorescence per-fluorophore photon budget is emitted photons per "
                "frame before collection and detector QE; configured_count_budget "
                "records the detected-count equivalent per fluorophore."
            )
        elif p.get("fluorescence_photon_count_scale") is not None:
            emitted_scale = _optional_float(p.get("fluorescence_photon_count_scale"))
            configured_count_budget = emitted_scale * collection * qe if emitted_scale is not None else None
            configured_count_budget_units = "detected_counts_per_source_unit"
            count_budget_source = "fluorescence_photon_count_scale"
            photons_incident = photons_incident if photons_incident is not None else emitted_scale
            photons_detected = photons_detected if photons_detected is not None else configured_count_budget
            count_budget_semantics = (
                "Legacy fluorescence photon_count_scale is an emitted/source scale "
                "before collection and detector QE; configured_count_budget records "
                "the detected-count equivalent per source unit."
            )
    else:
        configured_count_budget = _optional_float(
            p.get("background_intensity", p.get("dark_field_illumination_count"))
        )
        if configured_count_budget is not None:
            configured_count_budget_units = "detector_counts_per_pixel_per_frame"
            count_budget_source = (
                "background_intensity"
                if "background_intensity" in p
                else "dark_field_illumination_count"
            )
            photons_detected = photons_detected if photons_detected is not None else configured_count_budget
            count_budget_semantics = (
                "Optical label-free profiles use configured detector-count scale "
                "per pixel per frame unless a separate calibrated photon budget is declared."
            )

    dwell_time_s = _optional_float(p.get("dwell_time_s"))
    if dwell_time_s is None and p.get("sem_dwell_time_us") is not None:
        dwell_time_s = float(p.get("sem_dwell_time_us")) * 1.0e-6

    return AcquisitionCostModel(
        photons_detected=photons_detected,
        photons_incident=photons_incident,
        electron_dose=electron_dose,
        incident_primary_electrons_per_pixel=incident_primary_electrons,
        detected_electron_count_kind=detected_electron_count_kind,
        configured_count_budget=configured_count_budget,
        configured_count_budget_units=configured_count_budget_units,
        count_budget_source=count_budget_source,
        count_budget_semantics=count_budget_semantics,
        dwell_time_s=dwell_time_s,
        exposure_time_s=_optional_float(p.get("exposure_time_s")),
        photobleaching_cost=p.get("photobleaching_cost", None if not fluorescence else 0.0),
        radiation_damage_cost=p.get("radiation_damage_cost", None if not electron else 0.0),
        heating_cost=p.get("heating_cost"),
        sample_preparation_class=p.get("sample_preparation_class", "vacuum_electron" if electron else "ambient_optical"),
        destructive=bool(p.get("destructive", electron)),
        live_compatible=bool(p.get("live_compatible", not electron)),
        vacuum_compatible=bool(p.get("vacuum_compatible", electron)),
        field_of_view_um=p.get("field_of_view_um"),
        throughput_hz=p.get("throughput_hz"),
        frame_time_s=p.get("frame_time_s"),
        switching_setup_cost_s=p.get("switching_setup_cost_s"),
        dose_budget=p.get("dose_budget"),
        bleaching_budget=p.get("bleaching_budget"),
    )

def model_card_from_profile_card(profile_card: Mapping[str, Any]) -> dict[str, Any]:
    modality=str(profile_card.get("canonical_modality_name", profile_card.get("modality_id", "unknown")))
    backend=profile_card.get("backend_contract") or backend_contract_for_modality(modality, profile_card.get("response_function", {})).to_dict(); active=profile_card.get("active_parameters", {})
    fidelity=profile_card.get("backend_fidelity_metadata") or {}
    convergence_status = normalize_convergence_status(
        fidelity.get(
            "convergence_status",
            profile_card.get("convergence_status", ConvergenceStatus.UNCHECKED.value),
        )
    )
    return {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "model_card_id": f"model-card:{modality}",
        "human_label": profile_card.get("display_name", modality),
        "modality_family": modality.split("_")[0],
        "backend_family": backend.get("backend_family"),
        "backend_name": str(fidelity.get("backend_name", backend.get("backend_family", "imaging-model"))),
        "equations_or_model_family": str(
            fidelity.get(
                "equations_or_model_family",
                profile_card.get("forward_observable", backend.get("backend_family", "imaging-model")),
            )
        ),
        "implemented_approximation_level": str(fidelity.get("implemented_approximation_level", backend.get("fidelity_class", "proxy_model"))),
        "backend_fidelity_level": str(fidelity.get("backend_fidelity_level", backend.get("backend_fidelity_level", "proxy"))),
        "reference_backend_metadata": json_safe_with_nonfinite_tags(
            fidelity.get("reference_backend_metadata", backend.get("reference_backend_metadata"))
        ),
        "validation_status": str(fidelity.get("validation_status", backend.get("validation_status", ValidationStatus.UNCHECKED.value))),
        "convergence_status": convergence_status,
        "comparison_contract_id": str(fidelity.get("comparison_contract_id", "Contract-NR")),
        "artifact_provenance_id": fidelity.get("artifact_provenance_id"),
        "parameters_and_units": active,
        "detector_noise_model": profile_card.get("noise_model"),
        "detector_model": profile_card.get("detector_model"),
        "sample_environment_response": profile_card.get("sample_environment_usage", {}),
        "source_map_or_field_map_path": profile_card.get("source_map_path"),
        "axial_sensitivity_mechanism": backend.get("axial_sensitivity_mechanism"),
        "lateral_sensitivity_mechanism": profile_card.get("derivative_validity_scope"),
        "count_phase_electron_yield_units": {
            "measurement_domain": profile_card.get("measurement_domain"),
            "signal_units": profile_card.get("signal_units"),
            "contrast_frame_units": backend.get("contrast_frame_units"),
        },
        "native_operating_assumptions": str(
            fidelity.get("native_operating_assumptions", backend.get("native_units", "not_declared"))
        ),
        "configurable_keys": sorted(str(k) for k in active.keys()) if isinstance(active, Mapping) else [],
        "fixed_constants": profile_card.get("fixed_constants", {}),
        "default_values": profile_card.get("default_values", {}),
        "known_omissions": backend.get("known_omissions", ()),
        "validation_tests_passed": profile_card.get("validation_tests_passed", ()),
        "source_references": profile_card.get("source_references", ()),
        "paper_table_label": profile_card.get("display_name", modality),
        "table_safe_short_label": profile_card.get("display_name", modality),
        "provenance_hash": stable_hash({"modality": modality, "backend": backend, "active": active}),
    }

def validation_status_from_convergence(status: str) -> str:
    s=str(status)
    if s == ConvergenceStatus.FINITE_CONVERGED.value: return ValidationStatus.VALIDATED.value
    if s == ConvergenceStatus.STABLE_SINGULAR.value: return ValidationStatus.DIAGNOSTIC_ONLY.value
    if s in {ConvergenceStatus.FAILED_CONVERGENCE.value,ConvergenceStatus.ILL_CONDITIONED.value,ConvergenceStatus.NONFINITE.value}: return ValidationStatus.INVALID.value
    if s == ConvergenceStatus.EXTERNAL_ARTIFACT_REQUIRED.value: return ValidationStatus.EXTERNAL_ARTIFACT_REQUIRED.value
    return ValidationStatus.UNCHECKED.value

def combine_parent_statuses(parent_metadata: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    statuses={} if parent_metadata is None else {str(k): normalize_convergence_status(dict(v).get("convergence_status", ConvergenceStatus.UNCHECKED.value)) for k,v in parent_metadata.items()}
    if not statuses:
        return {"parent_convergence_statuses": {}, "validation_status": ValidationStatus.UNCHECKED.value, "production_grid_diagnostic": True, "safe_for_ordering": False, "safe_for_fusion": False, "safe_for_time_allocation": False, "safe_for_registration": False, "safe_for_detected_quanta_ranking": False, "status_reason": "no parent convergence metadata supplied"}
    values=set(statuses.values())
    if values == {ConvergenceStatus.FINITE_CONVERGED.value}:
        validation=ValidationStatus.VALIDATED.value; safe=True; reason="all supplied parent statuses are finite-converged"
    elif ConvergenceStatus.STABLE_SINGULAR.value in values:
        validation=ValidationStatus.DIAGNOSTIC_ONLY.value; safe=False; reason="one or more parent Fisher results are stable-singular; rank-aware diagnostics are allowed but ordering/fusion safety is false"
    elif values & {ConvergenceStatus.FAILED_CONVERGENCE.value, ConvergenceStatus.ILL_CONDITIONED.value, ConvergenceStatus.NONFINITE.value}:
        validation=ValidationStatus.INVALID.value; safe=False; reason="one or more parent Fisher results failed convergence, were ill-conditioned, or were nonfinite"
    else:
        validation=ValidationStatus.DIAGNOSTIC_ONLY.value; safe=False; reason="one or more parent Fisher results are unchecked, production-grid-only, or external-artifact-required"
    return {"parent_convergence_statuses": statuses, "validation_status": validation, "production_grid_diagnostic": validation != ValidationStatus.VALIDATED.value, "safe_for_ordering": safe, "safe_for_fusion": safe, "safe_for_time_allocation": safe, "safe_for_registration": safe, "safe_for_detected_quanta_ranking": safe, "status_reason": reason}

def fisher_result_from_crlb_result(result: Mapping[str, Any], *, result_id: str, source_contract: str, modality: str, convergence_status: str, backend_id: str = "", profile_id: str = "", parent_result_ids: Sequence[str] = ()) -> FisherResult:
    normalized_convergence = normalize_convergence_status(convergence_status)
    validation=validation_status_from_convergence(normalized_convergence); safe=validation==ValidationStatus.VALIDATED.value; rank=result.get("rank", result.get("fisher_rank"))
    covariance = result.get("covariance", None)
    if covariance is None:
        covariance = result.get("pseudoinverse", None)
    derivative_units = result.get("derivative_units", None)
    if derivative_units is None:
        derivative_units = result.get("derivative_units_by_axis", None)
    if derivative_units is None and isinstance(result.get("derivative_metadata"), Mapping):
        derivative_units = dict(result["derivative_metadata"]).get("derivative_units", None)
    if derivative_units is None:
        derivative_units = {}
    singular_axes = result.get("singular_axes", result.get("axes_singular", ()))
    crlb_summary = {
        k:v for k,v in result.items()
        if str(k).startswith("sigma_") or str(k).endswith("_nm")
    }
    for key in ("state_axes", "sigma_units_by_axis", "fisher_units", "fisher_units_by_entry"):
        if key in result:
            crlb_summary[key] = result[key]
    if "derivative_method" not in result:
        raise KeyError("CRLB result metadata must include 'derivative_method'.")
    return FisherResult(result_id=result_id,parent_result_ids=tuple(str(x) for x in parent_result_ids),source_contract=source_contract,modality=modality,backend_id=backend_id,profile_id=profile_id,fisher_matrix=result.get("fisher_matrix"),covariance_or_pseudoinverse=covariance,crlb_summary=crlb_summary,derivative_method=str(result["derivative_method"]),derivative_step=result.get("derivative_step", result.get("lateral_step_nm")),derivative_units=derivative_units,candidate_steps=result.get("candidate_steps", {}),convergence_status=normalized_convergence,singular_axes=tuple(singular_axes or ()),rank=None if rank is None else int(rank),rank_tolerance=float(result.get("rank_tolerance", 1e-12)),condition_number=result.get("condition_number"),validation_status=validation,production_grid_diagnostic=not safe,safe_for_ordering=safe,safe_for_fusion=safe,safe_for_time_allocation=safe,safe_for_registration=safe,safe_for_detected_quanta_ranking=safe,notes=("structured CRLB dictionary",))

def artifact_graph_manifest(nodes: Sequence[ArtifactNode | Mapping[str, Any]]) -> dict[str, Any]:
    out=[]
    for node in nodes:
        out.append(node.to_dict() if isinstance(node, ArtifactNode) else ArtifactNode(**dict(node)).to_dict())
    ids=[n["artifact_id"] for n in out]
    if len(ids)!=len(set(ids)): raise ValueError("artifact graph contains duplicate artifact_id values")
    return {"schema_version": ARTIFACT_GRAPH_SCHEMA_VERSION, "nodes": out}


def normalize_convergence_status(status: Any) -> str:
    """Validate and return the canonical scalar ``ConvergenceStatus`` value.

    This is the single scalar-status normalizer for paper-facing Fisher gates.
    Callers that need diagnostics should wrap the returned value in a structured
    Fisher convergence metadata record rather than inventing new status names.
    """
    if isinstance(status, ConvergenceStatus):
        return status.value
    raw = ConvergenceStatus.UNCHECKED.value if status is None else str(status).strip()
    allowed = {item.value for item in ConvergenceStatus}
    if raw in allowed:
        return raw
    raise ValueError(
        f"Unknown convergence_status {status!r}. Supported values are: {sorted(allowed)}."
    )


def normalize_validation_status(status: Any) -> str:
    if isinstance(status, ValidationStatus):
        return status.value
    raw = ValidationStatus.UNCHECKED.value if status is None else str(status).strip()
    allowed = {item.value for item in ValidationStatus}
    if raw in allowed:
        return raw
    raise ValueError(
        f"Unknown validation_status {status!r}. Supported values are: {sorted(allowed)}."
    )


def required_contract_ids() -> tuple[str, ...]:
    return (
        "Contract-LP",
        "Contract-LZ",
        "Contract-Q",
        "Contract-NR",
        "Contract-COST",
        "Contract-FUSION-PHYSICAL",
        "Contract-FUSION-ALGEBRAIC",
    )


def validate_contract_manifest(manifest: Mapping[str, Any], *, require_all: bool = True) -> dict[str, Any]:
    """Validate comparison-contract manifests before paper-facing output."""
    contracts = manifest.get("contracts", manifest) if isinstance(manifest, Mapping) else {}
    if isinstance(contracts, Sequence) and not isinstance(contracts, (str, bytes, bytearray)):
        contracts = {str(item.get("contract_id")): item for item in contracts if isinstance(item, Mapping)}
    missing = [cid for cid in required_contract_ids() if cid not in contracts]
    invalid = []
    for cid, contract in dict(contracts).items():
        try:
            validate_comparison_contract(dict(contract))
        except Exception as exc:
            invalid.append({"contract_id": str(cid), "reason": str(exc)})
    ok = (not invalid) and ((not missing) or not require_all)
    if require_all and missing:
        ok = False
    return {"ok": ok, "missing_contract_ids": missing, "invalid_contracts": invalid}


def detected_quanta_contract_metadata(
    *,
    total_detected_quanta_budget: float,
    distribution_rule: str = "profile_specific_detected_count_image",
    normalization_domain: str = "central_plane_or_2d_image",
    support_mask_used: bool = False,
    readout_variance_fraction_by_modality: Mapping[str, Any] | None = None,
    phase_mapping: str = "phase variance var(phi)=1/(visibility^2*n_Q)+readout",
    count_mapping: str = "count mean scaled so central-plane sum equals budget",
) -> dict[str, Any]:
    return {
        "contract_id": "Contract-Q",
        "total_detected_quanta_budget": float(total_detected_quanta_budget),
        "distribution_rule": distribution_rule,
        "normalization_domain": normalization_domain,
        "support_mask_used": bool(support_mask_used),
        "phase_domain_mapping": phase_mapping,
        "count_domain_mapping": count_mapping,
        "readout_variance_fraction_by_modality": dict(readout_variance_fraction_by_modality or {}),
        "budget_sum_check": "per modality central-plane detected-quanta sum equals total budget after scaling",
        "dose_semantics": "detected quanta are not dose unless a dose_cost_contract is supplied",
    }


def validate_fisher_result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validation report for status-rich Fisher/CRLB dictionaries."""
    required = (
        "result_id", "source_contract", "modality", "convergence_status",
        "validation_status", "derivative_method", "derivative_units",
    )
    missing = [key for key in required if key not in result]
    convergence = normalize_convergence_status(result.get("convergence_status"))
    validation = normalize_validation_status(result.get("validation_status", validation_status_from_convergence(convergence)))
    safe = (
        validation == ValidationStatus.VALIDATED.value
        and convergence == ConvergenceStatus.FINITE_CONVERGED.value
    )
    return {
        "ok": not missing and convergence != ConvergenceStatus.UNCHECKED.value,
        "missing_fields": missing,
        "convergence_status": convergence,
        "validation_status": validation,
        "safe_for_ordering": bool(result.get("safe_for_ordering", safe)) and safe,
        "safe_for_fusion": bool(result.get("safe_for_fusion", safe)) and safe,
        "safe_for_time_allocation": bool(result.get("safe_for_time_allocation", safe)) and safe,
        "safe_for_registration": bool(result.get("safe_for_registration", safe)) and safe,
    }
