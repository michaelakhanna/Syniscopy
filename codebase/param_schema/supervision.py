"""Supervision parameter-schema fragment."""

from __future__ import annotations

from ._spec import ParamSpec


SUPERVISION_SCHEMA: dict[str, ParamSpec] = {
"mask_generation_enabled": ParamSpec(
    key="mask_generation_enabled",
    default=True,
    type="bool",
    ui_label="Enable mask generation",
    group="Workflow",
    description="Enable/disable mask generation and mask writing.",
),
"mask_exact_leave_one_out_max_work_units": ParamSpec(
    key="mask_exact_leave_one_out_max_work_units",
    default=20000,
    type="int",
    min=1,
    max=10**12,
    ui_label="Mask exact render cap",
    group="Workflow",
    description=(
        "Maximum exact leave-one-out mask render work units before requiring "
        "mask_exact_leave_one_out_allow_expensive=True. Work units are "
        "frames × particles × motion-blur subsamples, multiplied by TEM "
        "multislice slice count for physical multislice TEM."
    ),
),
"mask_exact_leave_one_out_allow_expensive": ParamSpec(
    key="mask_exact_leave_one_out_allow_expensive",
    default=False,
    type="bool",
    ui_label="Allow expensive masks",
    group="Workflow",
    description=(
        "Allow exact leave-one-out mask generation even when the estimated "
        "full-render work exceeds mask_exact_leave_one_out_max_work_units."
    ),
),
"mask_output_directory": ParamSpec(
    key="mask_output_directory",
    default='outputs/syniscopy_masks',
    type="string",
    ui_label="Mask output directory",
    group="Workflow",
    description="Directory for generated mask outputs when mask generation is enabled.",
),
"mask_max_area_fraction": ParamSpec(
    key="mask_max_area_fraction",
    default=0.25,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Mask max area fraction",
    group="Workflow",
    description="Upper limit on the generated mask area as a fraction of frame area.",
),
"mask_outer_ring_count": ParamSpec(
    key="mask_outer_ring_count",
    default=0,
    type="int",
    min=0,
    max=6,
    ui_label="Mask outer rings",
    group="Mask",
    description="Number of PSF rings outside the central lobe to include in masks.",
),
"supervision_log_odds_clip_epsilon": ParamSpec(
    key="supervision_log_odds_clip_epsilon",
    default=1e-12,
    type="float",
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
    default='mask_supported',
    type="enum",
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
    default=None,
    type="string",
    ui_label="Supervision support factors",
    group="Advanced supervision",
    description=(
        "Comma-separated factor list, or 'None' to use enabled flags "
        "('temporal,signal,information,ambiguity')."
    ),
),
"supervision_supported_threshold": ParamSpec(
    key="supervision_supported_threshold",
    default=0.2,
    type="float",
    min=0.0,
    max=1.0,
    ui_label="Supervision supported threshold",
    group="Advanced supervision",
    description="Decision threshold for product-score and factor-based gating.",
),
"supervision_temporal_support_enabled": ParamSpec(
    key="supervision_temporal_support_enabled",
    default=True,
    type="bool",
    ui_label="Enable temporal support",
    group="Advanced supervision",
    description="Include Brownian-trajectory plausibility in supervision.",
),
"supervision_signal_support_enabled": ParamSpec(
    key="supervision_signal_support_enabled",
    default=True,
    type="bool",
    ui_label="Enable signal support",
    group="Advanced supervision",
    description="Include per-frame signal evidence in supervision.",
),
"supervision_information_support_enabled": ParamSpec(
    key="supervision_information_support_enabled",
    default=True,
    type="bool",
    ui_label="Enable information support",
    group="Advanced supervision",
    description="Include CRLB-based information support in supervision.",
),
"supervision_ambiguity_support_enabled": ParamSpec(
    key="supervision_ambiguity_support_enabled",
    default=True,
    type="bool",
    ui_label="Enable ambiguity support",
    group="Advanced supervision",
    description="Include competitor-overlap ambiguity support in supervision.",
),
"supervision_crlb_xy_max_nm": ParamSpec(
    key="supervision_crlb_xy_max_nm",
    default=None,
    type="float",
    min=0.0,
    max=1e12,
    ui_label="Supervision CRLB XY max (nm)",
    group="Advanced supervision",
    description="Maximum localization uncertainty used for information support; None uses pixel size.",
),
"supervision_stop_when_all_temporally_unsupported": ParamSpec(
    key="supervision_stop_when_all_temporally_unsupported",
    default=False,
    type="bool",
    ui_label="Stop track on temporal failure",
    group="Advanced supervision",
    description="Drop frames when temporal support stays unsupported for all particles.",
),
"supervision_ambiguity_distance_scale_nm": ParamSpec(
    key="supervision_ambiguity_distance_scale_nm",
    default=None,
    type="float",
    min=0.0,
    max=1e12,
    ui_label="Ambiguity distance scale (nm)",
    group="Advanced supervision",
    description="Distance scale used by ambiguity support; None uses 2*pixel_size_nm.",
),
"supervision_prior_log_odds": ParamSpec(
    key="supervision_prior_log_odds",
    default=0.0,
    type="float",
    min=-100.0,
    max=100.0,
    ui_label="Supervision prior log-odds",
    group="Advanced supervision",
    description="Additive prior term in the supervision log-odds score.",
),
"supervision_decision_rule": ParamSpec(
    key="supervision_decision_rule",
    default='log_odds',
    type="enum",
    choices=["log_odds", "product"],
    ui_label="Supervision decision rule",
    group="Advanced supervision",
    description="Choose additive log-odds or multiplicative factor gating.",
),
"supervision_log_odds_threshold": ParamSpec(
    key="supervision_log_odds_threshold",
    default=0.0,
    type="float",
    min=-100.0,
    max=100.0,
    ui_label="Supervision log-odds threshold",
    group="Advanced supervision",
    description="Threshold on supervision log-odds map for support acceptance.",
),
"supervision_score_calibration_mode": ParamSpec(
    key="supervision_score_calibration_mode",
    default='uncalibrated_support',
    type="enum",
    choices=["uncalibrated_support", "platt_logistic", "isotonic"],
    ui_label="Support calibration",
    group="Advanced supervision",
    description="Optional calibration mapping from support scores to empirical probabilities.",
),
"supervision_score_calibration_parameters": ParamSpec(
    key="supervision_score_calibration_parameters",
    default=None,
    type="json",
    ui_label="Calibration parameters",
    group="Advanced supervision",
    description="Platt or isotonic calibration parameters used only when calibration mode is enabled.",
),
}

__all__ = ["SUPERVISION_SCHEMA"]
