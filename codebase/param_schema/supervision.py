"""Supervision parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


SUPERVISION_SCHEMA: dict[str, ParamSpec] = {
"mask_generation_enabled": ParamSpec(
    key="mask_generation_enabled",
    type="bool",
    default=True,
    ui_label="Enable mask generation",
    group="Workflow",
    description="Enable/disable mask generation and mask writing.",
),
"mask_output_directory": ParamSpec(
    key="mask_output_directory",
    type="string",
    default=None,
    ui_label="Mask output directory",
    group="Workflow",
    description="Directory for generated mask outputs when mask generation is enabled.",
),
"mask_max_area_fraction": ParamSpec(
    key="mask_max_area_fraction",
    type="float",
    default=0.25,
    min=0.0,
    max=1.0,
    ui_label="Mask max area fraction",
    group="Workflow",
    description="Upper limit on the generated mask area as a fraction of frame area.",
),
"mask_outer_ring_count": ParamSpec(
    key="mask_outer_ring_count",
    type="int",
    default=0,
    min=0,
    max=6,
    ui_label="Mask outer rings",
    group="Mask",
    description="Number of PSF rings outside the central lobe to include in masks.",
),
"supervision_log_odds_clip_epsilon": ParamSpec(
    key="supervision_log_odds_clip_epsilon",
    type="float",
    default=1e-12,
    min=1e-15,
    max=0.499999,
    ui_label="Log-odds clip epsilon",
    group="Advanced supervision",
    description=(
        "Numerical clipping floor used when converting support factors to "
        "log-odds. Must lie in (0, 0.5)."
    ),
),
"supervision_target": ParamSpec(
    key="supervision_target",
    type="enum",
    default="mask_supported",
    choices=["mask_supported", "mask_geometry"],
    ui_label="Supervision target",
    group="Advanced supervision",
    description=(
        "Annotation target prior to optional ignore-mask and "
        "loss-weight derivation."
    ),
),
"supervision_support_factors": ParamSpec(
    key="supervision_support_factors",
    type="string",
    default=None,
    ui_label="Supervision support factors",
    group="Advanced supervision",
    description=(
        "Comma-separated factor list, or 'None' to use enabled flags "
        "('temporal,signal,information,ambiguity')."
    ),
),
"supervision_supported_threshold": ParamSpec(
    key="supervision_supported_threshold",
    type="float",
    default=0.2,
    min=0.0,
    max=1.0,
    ui_label="Supervision supported threshold",
    group="Advanced supervision",
    description="Decision threshold for product-score and factor-based gating.",
),
"supervision_temporal_support_enabled": ParamSpec(
    key="supervision_temporal_support_enabled",
    type="bool",
    default=True,
    ui_label="Enable temporal support",
    group="Advanced supervision",
    description="Include Brownian-trajectory plausibility in supervision.",
),
"supervision_signal_support_enabled": ParamSpec(
    key="supervision_signal_support_enabled",
    type="bool",
    default=True,
    ui_label="Enable signal support",
    group="Advanced supervision",
    description="Include per-frame signal evidence in supervision.",
),
"supervision_information_support_enabled": ParamSpec(
    key="supervision_information_support_enabled",
    type="bool",
    default=True,
    ui_label="Enable information support",
    group="Advanced supervision",
    description="Include CRLB-based information support in supervision.",
),
"supervision_ambiguity_support_enabled": ParamSpec(
    key="supervision_ambiguity_support_enabled",
    type="bool",
    default=True,
    ui_label="Enable ambiguity support",
    group="Advanced supervision",
    description="Include competitor-overlap ambiguity support in supervision.",
),
"supervision_crlb_xy_max_nm": ParamSpec(
    key="supervision_crlb_xy_max_nm",
    type="float",
    default=None,
    min=0.0,
    max=1e12,
    ui_label="Supervision CRLB XY max (nm)",
    group="Advanced supervision",
    description="Maximum localization uncertainty used for information support; None uses pixel size.",
),
"supervision_stop_when_all_temporally_unsupported": ParamSpec(
    key="supervision_stop_when_all_temporally_unsupported",
    type="bool",
    default=False,
    ui_label="Stop track on temporal failure",
    group="Advanced supervision",
    description="Drop frames when temporal support stays unsupported for all particles.",
),
"supervision_ambiguity_distance_scale_nm": ParamSpec(
    key="supervision_ambiguity_distance_scale_nm",
    type="float",
    default=None,
    min=0.0,
    max=1e12,
    ui_label="Ambiguity distance scale (nm)",
    group="Advanced supervision",
    description="Distance scale used by ambiguity support; None uses 2*pixel_size_nm.",
),
"supervision_prior_log_odds": ParamSpec(
    key="supervision_prior_log_odds",
    type="float",
    default=0.0,
    min=-100.0,
    max=100.0,
    ui_label="Supervision prior log-odds",
    group="Advanced supervision",
    description="Additive prior term in the supervision log-odds score.",
),
"supervision_decision_rule": ParamSpec(
    key="supervision_decision_rule",
    type="enum",
    default="log_odds",
    choices=["log_odds", "product"],
    ui_label="Supervision decision rule",
    group="Advanced supervision",
    description="Choose additive log-odds or multiplicative factor gating.",
),
"supervision_log_odds_threshold": ParamSpec(
    key="supervision_log_odds_threshold",
    type="float",
    default=0.0,
    min=-100.0,
    max=100.0,
    ui_label="Supervision log-odds threshold",
    group="Advanced supervision",
    description="Threshold on supervision log-odds map for support acceptance.",
),
"supervision_score_calibration_mode": ParamSpec(
    key="supervision_score_calibration_mode",
    type="enum",
    default="uncalibrated_support",
    choices=["uncalibrated_support", "platt_logistic", "isotonic"],
    ui_label="Support calibration",
    group="Advanced supervision",
    description="Optional calibration mapping from support scores to empirical probabilities.",
),
"supervision_score_calibration_parameters": ParamSpec(
    key="supervision_score_calibration_parameters",
    type="json",
    default=None,
    ui_label="Calibration parameters",
    group="Advanced supervision",
    description="Platt or isotonic calibration parameters used only when calibration mode is enabled.",
),
}

__all__ = ["SUPERVISION_SCHEMA"]
