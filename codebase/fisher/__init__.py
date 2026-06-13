"""Supported package-level Fisher diagnostics API.

The names exported here are the stable cross-module entry points used by
Syniscopy reports, simulations, and supplemental notebooks. Lower-level Fisher
helpers remain available from their owning submodules and should be imported
directly from those submodules when needed.
"""

from __future__ import annotations

from .axial import (
    compute_fisher_information_3d,
    compute_localization_crlb_3d,
)
from .candidates import (
    FisherCandidate,
    FisherMatrixCandidate,
)
from .comparison import (
    COMPARISON_TARGET_AXIAL_Z,
    COMPARISON_TARGET_LATERAL_XY,
    COMPARISON_TARGET_LOCALIZATION_XYZ,
    COMPARISON_TARGET_ORIENTATION,
    compare_fisher_candidates,
)
from .density import (
    compute_information_density_maps,
    compute_nuisance_adjusted_fisher,
)
from .detected_quanta import (
    DetectedQuantaCandidate,
    compare_detected_quanta_normalized_fisher_candidates,
    write_detected_quanta_derivative_convergence_csv,
)
from .dhm_demodulated import (
    OFF_AXIS_DEMODULATED_COVARIANCE_KIND,
    build_off_axis_demodulated_observation,
    compute_off_axis_demodulated_fisher_information,
    compute_off_axis_demodulated_localization_crlb,
    compute_off_axis_demodulated_localization_crlb_from_field,
    is_off_axis_demodulated_fisher_payload,
    is_off_axis_holography_modality,
)
from .dynamic_bayesian import (
    build_brownian_process_covariance,
    compute_dynamic_bayesian_crlb_from_fisher_sequence,
    sequence_sum_fisher_to_crlb,
    summarize_fisher_sequence,
)
from .fusion import (
    compute_candidate_registration_degradation_curve,
    compute_fisher_candidate_fusion_crlb,
    compute_candidate_fusion_crlb_from_fisher_matrices,
    sigma_xy_from_fisher,
)
from .lateral import (
    compute_fisher_information,
    compute_likelihood_fisher_information,
    compute_localization_crlb,
)
from .lateral_derivative_contracts import (
    ArrayOnlyFisherDerivativeContext,
    LATERAL_DERIVATIVE_BASIS,
    LateralDerivativePlan,
    array_only_derivative_context_metadata,
    lateral_derivative_plan_metadata,
    normalize_array_only_fisher_derivative_context,
    require_array_only_3d_fisher_derivative_basis_safe,
    require_array_only_spectral_lateral_derivative_ready,
    require_lateral_derivative_plan_supported,
    spectral_lateral_derivative_plan,
)
from .scaling_laws import (
    compute_rayleigh_amplitude_scaling_control,
    summarize_closed_form_scaling_checks,
)
from .se3 import (
    compute_localization_orientation_crlb,
    predict_se3_rank_from_contrast_stabilizer,
)
from .time_allocation import compute_optimal_time_allocation_crlb

__all__ = [
    "COMPARISON_TARGET_AXIAL_Z",
    "COMPARISON_TARGET_LATERAL_XY",
    "COMPARISON_TARGET_LOCALIZATION_XYZ",
    "COMPARISON_TARGET_ORIENTATION",
    "FisherCandidate",
    "FisherMatrixCandidate",
    "DetectedQuantaCandidate",
    "OFF_AXIS_DEMODULATED_COVARIANCE_KIND",
    "build_brownian_process_covariance",
    "build_off_axis_demodulated_observation",
    "compare_detected_quanta_normalized_fisher_candidates",
    "compare_fisher_candidates",
    "compute_dynamic_bayesian_crlb_from_fisher_sequence",
    "compute_fisher_information",
    "compute_fisher_information_3d",
    "compute_off_axis_demodulated_fisher_information",
    "compute_information_density_maps",
    "compute_likelihood_fisher_information",
    "ArrayOnlyFisherDerivativeContext",
    "LATERAL_DERIVATIVE_BASIS",
    "LateralDerivativePlan",
    "array_only_derivative_context_metadata",
    "lateral_derivative_plan_metadata",
    "normalize_array_only_fisher_derivative_context",
    "require_array_only_3d_fisher_derivative_basis_safe",
    "require_array_only_spectral_lateral_derivative_ready",
    "require_lateral_derivative_plan_supported",
    "spectral_lateral_derivative_plan",
    "compute_localization_crlb",
    "compute_localization_crlb_3d",
    "compute_off_axis_demodulated_localization_crlb",
    "compute_off_axis_demodulated_localization_crlb_from_field",
    "compute_localization_orientation_crlb",
    "compute_fisher_candidate_fusion_crlb",
    "compute_candidate_fusion_crlb_from_fisher_matrices",
    "compute_nuisance_adjusted_fisher",
    "compute_optimal_time_allocation_crlb",
    "compute_rayleigh_amplitude_scaling_control",
    "compute_candidate_registration_degradation_curve",
    "predict_se3_rank_from_contrast_stabilizer",
    "is_off_axis_demodulated_fisher_payload",
    "is_off_axis_holography_modality",
    "sequence_sum_fisher_to_crlb",
    "sigma_xy_from_fisher",
    "summarize_closed_form_scaling_checks",
    "summarize_fisher_sequence",
    "write_detected_quanta_derivative_convergence_csv",
]
