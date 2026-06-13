"""Fisher Contracts parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


FISHER_CONTRACT_SCHEMA: dict[str, ParamSpec] = {
"fisher_likelihood_model": ParamSpec(
    key="fisher_likelihood_model",
    default='gaussian_fixed_variance',
    type="enum",
    choices=[
        "gaussian_fixed_variance",
        "poisson_exact",
        "gaussian_parameter_dependent_variance",
        "poisson_gaussian_approx",
    ],
    ui_label="Fisher likelihood model",
    group="Advanced Fisher",
    description="Statistical model for Fisher diagnostics.",
),
"detected_quanta_derivative_target": ParamSpec(
    key="detected_quanta_derivative_target",
    default='signed_contrast_scaled',
    type="enum",
    choices=["signed_contrast_scaled", "count_mean_derivative"],
    ui_label="Detected quanta derivative target",
    group="Advanced Fisher",
    description="Finite-difference derivative target used by Fisher diagnostics.",
),
"profile_fidelity_label": ParamSpec(
    key="profile_fidelity_label",
    default='model_conditional_profile',
    type="string",
    ui_label="Profile fidelity label",
    group="Advanced Fisher",
    description="Fallback fidelity label used only when a backend response does not report one.",
),
}

__all__ = ["FISHER_CONTRACT_SCHEMA"]
