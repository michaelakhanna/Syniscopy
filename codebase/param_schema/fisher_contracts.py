"""Fisher Contracts parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


FISHER_CONTRACT_SCHEMA: dict[str, ParamSpec] = {
"fisher_lateral_step_nm": ParamSpec(
    key="fisher_lateral_step_nm",
    type="float",
    default=5.0,
    min=0.001,
    max=1000.0,
    ui_label="Fisher lateral step (nm)",
    group="Advanced Fisher",
    description=(
        "Symmetric x/y perturbation step for rerendered lateral Fisher "
        "derivatives. Stationary-shift derivatives use the detector grid."
    ),
),
"fisher_lateral_derivative_mode": ParamSpec(
    key="fisher_lateral_derivative_mode",
    type="enum",
    default="stationary_shift",
    choices=["stationary_shift", "rerendered_xy"],
    ui_label="Fisher lateral derivative mode",
    group="Advanced Fisher",
    description=(
        "Use stationary detector-grid shifts for uniform scenes or explicit "
        "rerendered x/y scene perturbations for structured sample environments."
    ),
),
"fisher_likelihood_model": ParamSpec(
    key="fisher_likelihood_model",
    type="enum",
    default="mean_fisher_diagnostic",
    choices=[
        "gaussian_fixed_variance",
        "poisson_exact",
        "gaussian_parameter_dependent_variance",
        "poisson_gaussian_approx",
        "poisson_gaussian_plugin",
        "mean_fisher_diagnostic",
    ],
    ui_label="Fisher likelihood model",
    group="Advanced Fisher",
    description="Statistical model for Fisher diagnostics.",
),
"detected_quanta_derivative_target": ParamSpec(
    key="detected_quanta_derivative_target",
    type="enum",
    default="signed_contrast_scaled",
    choices=["signed_contrast_scaled", "count_mean_derivative"],
    ui_label="Detected quanta derivative target",
    group="Advanced Fisher",
    description="Finite-difference derivative target used by Fisher diagnostics.",
),
}

__all__ = ["FISHER_CONTRACT_SCHEMA"]
