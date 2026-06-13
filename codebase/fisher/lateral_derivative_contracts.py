"""Shared lateral Fisher spectral-derivative contracts.

Lateral localization Fisher uses one derivative basis: the exact FFT gradient
of the sampled band-limited contrast image. The former runtime choice between
stationary image shifts and explicit x/y rerenders is intentionally gone; x/y
rerenders remain a valid idea for other state perturbations only when those
owners explicitly need them, not as a lateral Fisher mode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

LATERAL_DERIVATIVE_BASIS = "spectral_band_limited"
ARRAY_ONLY_3D_STATE_AXES = ("x", "y", "z")
ARRAY_ONLY_3D_DERIVATIVE_BASIS_BY_AXIS = {
    "x": LATERAL_DERIVATIVE_BASIS,
    "y": LATERAL_DERIVATIVE_BASIS,
    "z": "symmetric_rerendered_z_pair",
}
ARRAY_ONLY_3D_DERIVATIVE_CONTRACT_VERSION = "axiswise-array-only-3d-fisher-v2"


@dataclass(frozen=True)
class LateralDerivativePlan:
    """Declared derivative basis for a lateral Fisher consumer."""

    basis: str = LATERAL_DERIVATIVE_BASIS
    supported: bool = True
    resolution: str = "single_center_render_fft_spectral_gradient"
    reasons: tuple[str, ...] = ()
    contract_version: str = "syniscopy-lateral-spectral-derivative-v2"

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ArrayOnlyFisherDerivativeContext:
    """Provenance carried by array-only Fisher comparisons.

    The lateral derivative basis no longer depends on a scene-mode resolver, but
    array-only comparison still needs provenance so ranking artifacts can state
    what rendered observation the supplied arrays came from.
    """

    params: Mapping[str, Any] | None = None
    model: Any = None
    response_function: Mapping[str, Any] | None = None
    num_particles: int | None = None
    structured_environment_active: bool | None = None
    stationary_template_provenance: str | None = None
    contract_version: str = "syniscopy-array-only-fisher-derivative-context-v2"

    @classmethod
    def single_rigid_template(
        cls,
        provenance: str = "caller_declared_single_rigid_template_analysis_likelihood_basis",
    ) -> "ArrayOnlyFisherDerivativeContext":
        return cls(stationary_template_provenance=str(provenance))

    def has_scene_or_template_provenance(self) -> bool:
        return bool(
            self.params is not None
            or self.model is not None
            or self.response_function is not None
            or self.num_particles is not None
            or self.structured_environment_active is not None
            or self.stationary_template_provenance
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "array_only_derivative_context_contract_version": self.contract_version,
            "array_only_derivative_context_present": True,
            "array_only_derivative_context_has_params": self.params is not None,
            "array_only_derivative_context_has_model": self.model is not None,
            "array_only_derivative_context_has_response_function": self.response_function is not None,
            "array_only_derivative_context_num_particles": self.num_particles,
            "array_only_derivative_context_structured_environment_active": self.structured_environment_active,
            "stationary_template_provenance": self.stationary_template_provenance or "",
        }


def spectral_lateral_derivative_plan() -> LateralDerivativePlan:
    """Return the single supported lateral derivative declaration."""

    return LateralDerivativePlan()


def normalize_array_only_fisher_derivative_context(
    value: Any,
    *,
    context: str,
) -> ArrayOnlyFisherDerivativeContext:
    """Normalize one per-candidate array-only derivative provenance payload."""

    if isinstance(value, ArrayOnlyFisherDerivativeContext):
        derivative_context = value
    elif isinstance(value, Mapping):
        derivative_context = ArrayOnlyFisherDerivativeContext(
            params=value.get("params"),
            model=value.get("model"),
            response_function=value.get("response_function"),
            num_particles=value.get("num_particles"),
            structured_environment_active=value.get("structured_environment_active"),
            stationary_template_provenance=(
                value.get("stationary_template_provenance")
                or value.get("stationary_lateral_template_provenance")
                or value.get("template_provenance")
            ),
        )
    else:
        raise TypeError(
            f"{context} requires each array-only derivative context to be a "
            "mapping or ArrayOnlyFisherDerivativeContext; got "
            f"{type(value).__name__}."
        )
    if not derivative_context.has_scene_or_template_provenance():
        raise ValueError(
            f"{context} was given an empty array-only derivative context. "
            "Provide scene/model provenance or an explicit template provenance "
            "before ranking array-only Fisher candidates."
        )
    return derivative_context


def lateral_derivative_plan_metadata(plan: LateralDerivativePlan | None = None) -> dict[str, Any]:
    """Return flat output metadata for the lateral spectral derivative basis."""

    resolved = plan or spectral_lateral_derivative_plan()
    return {
        "fisher_lateral_derivative_basis": resolved.basis,
        "fisher_lateral_derivative_basis_resolution": resolved.resolution,
        "fisher_lateral_derivative_step_size_free": True,
        "fisher_lateral_derivative_contract_version": resolved.contract_version,
        "fisher_lateral_derivative_basis_reasons": list(resolved.reasons),
    }


def array_only_derivative_context_metadata(
    derivative_context: ArrayOnlyFisherDerivativeContext,
    plan: LateralDerivativePlan | None = None,
) -> dict[str, Any]:
    """Merge provenance and derivative-basis metadata for protected outputs."""

    payload = lateral_derivative_plan_metadata(plan)
    payload.update(derivative_context.to_metadata())
    return payload


def require_lateral_derivative_plan_supported(
    plan: LateralDerivativePlan | None = None,
    *,
    context: str = "lateral Fisher",
) -> None:
    """Raise if a hand-built lateral derivative declaration is unsupported."""

    resolved = plan or spectral_lateral_derivative_plan()
    if not resolved.supported or resolved.basis != LATERAL_DERIVATIVE_BASIS:
        reason_text = ", ".join(resolved.reasons) or resolved.resolution
        raise ValueError(
            f"{context} requires lateral derivative basis "
            f"{LATERAL_DERIVATIVE_BASIS!r}; got {resolved.basis!r}: {reason_text}."
        )


def require_array_only_spectral_lateral_derivative_ready(
    *,
    modality: Any,
    params: Mapping[str, Any] | None = None,
    model: Any = None,
    response_function: Mapping[str, Any] | None = None,
    num_particles: int | None = None,
    structured_environment_active: bool | None = None,
    context: str,
) -> LateralDerivativePlan:
    """Declare the spectral lateral basis for array-only Fisher consumers."""

    del params, num_particles, structured_environment_active
    blockers: list[str] = []
    modality_name = str(modality or "").strip().lower()
    if modality_name == "off_axis_holography":
        blockers.append(
            "off-axis holography array-only inputs must be demodulated complex sideband fields with propagated covariance"
        )
    if model is not None:
        if bool(getattr(model, "requires_rerendered_lateral_fisher", False)):
            blockers.append("imaging model declares requires_rerendered_lateral_fisher=True")
        if bool(getattr(model, "has_detector_fixed_lateral_carrier", False)):
            blockers.append("imaging model declares has_detector_fixed_lateral_carrier=True")
        if not bool(getattr(model, "stationary_lateral_fisher_safe_for_single_uniform_scene", True)):
            blockers.append(
                "imaging model declares stationary_lateral_fisher_safe_for_single_uniform_scene=False"
            )
    response = dict(response_function or {})
    if bool(response.get("requires_rerendered_lateral_fisher", False)):
        blockers.append("response metadata declares requires_rerendered_lateral_fisher=True")
    if bool(response.get("has_detector_fixed_lateral_carrier", False)):
        blockers.append("response metadata declares has_detector_fixed_lateral_carrier=True")
    if response.get("stationary_lateral_fisher_safe_for_single_uniform_scene") is False:
        blockers.append(
            "response metadata declares stationary_lateral_fisher_safe_for_single_uniform_scene=False"
        )
    if blockers:
        raise ValueError(
            f"{context} cannot use array-only spectral lateral Fisher on this "
            "rendered observable. The stationary spectral derivative would "
            "differentiate detector/world-fixed structure rather than only the "
            "particle shift. Use the off-axis demodulated-field Fisher owner or "
            "supply a demodulated particle-shift observable with its propagated "
            "sideband covariance. Blockers: "
            + "; ".join(dict.fromkeys(blockers))
            + "."
        )
    plan = spectral_lateral_derivative_plan()
    require_lateral_derivative_plan_supported(plan, context=context)
    return plan


def array_only_3d_derivative_basis_metadata(
    lateral_plan: LateralDerivativePlan | None = None,
    *,
    z_step_nm: float,
) -> dict[str, Any]:
    """Describe the complete derivative basis used by array-only 3D Fisher."""

    plan = lateral_plan or spectral_lateral_derivative_plan()
    return {
        "fisher_derivative_contract_version": ARRAY_ONLY_3D_DERIVATIVE_CONTRACT_VERSION,
        "state_axes": list(ARRAY_ONLY_3D_STATE_AXES),
        "axis_derivative_basis_by_axis": dict(ARRAY_ONLY_3D_DERIVATIVE_BASIS_BY_AXIS),
        "x_y_lateral_derivative_basis": plan.basis,
        "x_y_lateral_plan": lateral_derivative_plan_metadata(plan),
        "z_step_nm": float(z_step_nm),
        "z_derivative_source": "three_plane_z_stack_outer_planes",
        "requires_explicit_xy_rerender_bundle": False,
    }


def require_array_only_3d_fisher_derivative_basis_safe(
    *,
    modality: Any,
    z_step_nm: float,
    params: Mapping[str, Any] | None = None,
    model: Any = None,
    response_function: Mapping[str, Any] | None = None,
    num_particles: int | None = None,
    structured_environment_active: bool | None = None,
    context: str,
) -> dict[str, Any]:
    """Guard array-only 3D Fisher consumers before any numeric CRLB is built.

    The z derivative remains a three-plane rerendered stack. The x/y entries use
    the single spectral lateral basis.
    """

    require_array_only_spectral_lateral_derivative_ready(
        modality=modality,
        params=params,
        model=model,
        response_function=response_function,
        num_particles=num_particles,
        structured_environment_active=structured_environment_active,
        context=context,
    )
    if not np.isfinite(float(z_step_nm)) or float(z_step_nm) <= 0.0:
        raise ValueError(f"{context} requires positive z_step_nm; got {z_step_nm!r}.")
    return array_only_3d_derivative_basis_metadata(
        spectral_lateral_derivative_plan(),
        z_step_nm=float(z_step_nm),
    )


__all__ = [
    "ARRAY_ONLY_3D_DERIVATIVE_BASIS_BY_AXIS",
    "ArrayOnlyFisherDerivativeContext",
    "ARRAY_ONLY_3D_DERIVATIVE_CONTRACT_VERSION",
    "ARRAY_ONLY_3D_STATE_AXES",
    "LATERAL_DERIVATIVE_BASIS",
    "LateralDerivativePlan",
    "array_only_3d_derivative_basis_metadata",
    "array_only_derivative_context_metadata",
    "lateral_derivative_plan_metadata",
    "normalize_array_only_fisher_derivative_context",
    "require_array_only_3d_fisher_derivative_basis_safe",
    "require_array_only_spectral_lateral_derivative_ready",
    "require_lateral_derivative_plan_supported",
    "spectral_lateral_derivative_plan",
]
