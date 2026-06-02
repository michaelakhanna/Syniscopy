"""Supported package-level Fisher diagnostics API.

The names exported here are the stable cross-module entry points used by
Syniscopy reports, simulations, and supplemental notebooks. Lower-level Fisher
helpers remain available from their owning submodules and should be imported
directly from those submodules when needed.
"""

from __future__ import annotations

from .axial import (
    compare_modality_axial_information_content,
    compute_localization_crlb_3d,
)
from .density import (
    compute_information_density_maps,
    compute_nuisance_adjusted_fisher,
)
from .detected_quanta import (
    compare_modality_information_content_detected_quanta_normalized,
    write_detected_quanta_derivative_convergence_csv,
)
from .dynamic_bayesian import (
    build_brownian_process_covariance,
    compute_dynamic_bayesian_crlb_from_fisher_sequence,
    sequence_sum_fisher_to_crlb,
    summarize_fisher_sequence,
)
from .fusion import (
    compute_registration_degradation_curve,
    compute_modality_fusion_crlb,
    compute_modality_fusion_crlb_from_fisher_matrices,
    sigma_xy_from_fisher,
)
from .lateral import (
    adaptive_lateral_crlb_from_rerender_pairs,
    compare_modality_information_content,
    compare_modality_information_content_from_crlb_results,
    compute_fisher_information,
    compute_localization_crlb,
    compute_localization_crlb_from_lateral_rerenders,
)
from .scaling_laws import (
    compute_rayleigh_amplitude_scaling_control,
    summarize_closed_form_scaling_checks,
)
from .se3 import (
    compute_localization_orientation_crlb,
    predict_se3_rank_from_symmetry,
)
from .time_allocation import compute_optimal_time_allocation_crlb

__all__ = [
    "adaptive_lateral_crlb_from_rerender_pairs",
    "build_brownian_process_covariance",
    "compare_modality_axial_information_content",
    "compare_modality_information_content",
    "compare_modality_information_content_detected_quanta_normalized",
    "compare_modality_information_content_from_crlb_results",
    "compute_dynamic_bayesian_crlb_from_fisher_sequence",
    "compute_fisher_information",
    "compute_information_density_maps",
    "compute_localization_crlb",
    "compute_localization_crlb_3d",
    "compute_localization_crlb_from_lateral_rerenders",
    "compute_localization_orientation_crlb",
    "compute_modality_fusion_crlb",
    "compute_modality_fusion_crlb_from_fisher_matrices",
    "compute_nuisance_adjusted_fisher",
    "compute_optimal_time_allocation_crlb",
    "compute_rayleigh_amplitude_scaling_control",
    "compute_registration_degradation_curve",
    "predict_se3_rank_from_symmetry",
    "sequence_sum_fisher_to_crlb",
    "sigma_xy_from_fisher",
    "summarize_closed_form_scaling_checks",
    "summarize_fisher_sequence",
    "write_detected_quanta_derivative_convergence_csv",
]
