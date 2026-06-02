"""Physical compatibility metadata for modality fusion diagnostics.

This module distinguishes algebraic Fisher addition from physically feasible
multi-modality acquisition.  It is deliberately conservative: a subset can be
reported as algebraic even when the Fisher sum is computable.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

from experiment_contracts import acquisition_cost_from_profile
from modality_registry import (
    ELECTRON_MODALITIES,
    LABEL_FREE_OPTICAL_MODALITIES,
    canonical_modality_name,
)


_ELECTRON_MODALITIES = set(ELECTRON_MODALITIES)
_FLUORESCENCE_MODALITIES = {"fluorescence_widefield", "tirf_fluorescence"}
_LIVE_OPTICAL_MODALITIES = set(LABEL_FREE_OPTICAL_MODALITIES) | _FLUORESCENCE_MODALITIES


def _canonical(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return canonical_modality_name(normalized)


def _has_declared_key(params: dict[str, Any], key: str) -> bool:
    return key in params and params[key] is not None


def _same_declared_value(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    key: str,
) -> bool:
    return (
        _has_declared_key(params_a, key)
        and _has_declared_key(params_b, key)
        and params_a[key] == params_b[key]
    )


def modality_physical_profile(modality: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the physical acquisition profile used by compatibility checks."""
    m = _canonical(modality)
    p = {} if params is None else dict(params)
    electron = m in _ELECTRON_MODALITIES
    fluorescence = m in _FLUORESCENCE_MODALITIES
    optical = m in _LIVE_OPTICAL_MODALITIES
    destructive = bool(p.get("destructive", electron))
    live_compatible = bool(p.get("live_compatible", optical and not electron))
    vacuum_compatible = bool(p.get("vacuum_compatible", electron))
    profile = {
        "modality": m,
        "sample_preparation_class": p.get("sample_preparation_class", "vacuum_electron" if electron else ("fluorescence_labelled" if fluorescence else "ambient_optical")),
        "live_compatible": live_compatible,
        "vacuum_compatible": vacuum_compatible,
        "destructive": destructive,
        "electron_beam_modality": bool(electron),
        "optical_modality": bool(optical and not electron),
        "fluorescence_labeling_required": bool(fluorescence),
        "same_particle_state_preserving": bool(p.get("same_particle_state_preserving", not destructive)),
        "simultaneous_acquisition_possible": bool(p.get("simultaneous_acquisition_possible", optical and not electron)),
        "sequential_acquisition_possible": bool(p.get("sequential_acquisition_possible", True)),
        "registration_required": bool(p.get("registration_required", True)),
        "dose_cost_compatible": bool(p.get("dose_cost_compatible", not destructive)),
        "field_of_view_compatible": bool(p.get("field_of_view_compatible", True)),
        "substrate_compatible": bool(p.get("substrate_compatible", True)),
        "environment_compatible": bool(p.get("environment_compatible", not (electron and live_compatible))),
    }
    profile["acquisition_cost_model"] = acquisition_cost_from_profile(m, p).to_dict()
    return profile


def compatibility_status(
    modality_a: str,
    modality_b: str,
    params_a: dict[str, Any] | None = None,
    params_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return pairwise compatibility metadata for a proposed fusion pair."""
    a = _canonical(modality_a)
    b = _canonical(modality_b)
    pa = modality_physical_profile(a, params_a)
    pb = modality_physical_profile(b, params_b)
    params_a = {} if params_a is None else dict(params_a)
    params_b = {} if params_b is None else dict(params_b)
    same_quanta = _same_declared_value(params_a, params_b, "same_source_quanta_id")
    same_detector = _same_declared_value(params_a, params_b, "detector_channel_id")
    electron_live_conflict = (pa["electron_beam_modality"] and pb["live_compatible"]) or (pb["electron_beam_modality"] and pa["live_compatible"])
    destructive_conflict = pa["destructive"] or pb["destructive"]
    separate_sample = bool(params_a.get("separate_sample_diagnostic") or params_b.get("separate_sample_diagnostic"))
    wf_tirf_pair = a in _FLUORESCENCE_MODALITIES and b in _FLUORESCENCE_MODALITIES and a != b
    independent_budgets = bool(
        params_a.get("independent_excitation_budget")
        and params_b.get("independent_excitation_budget")
        and _has_declared_key(params_a, "detector_channel_id")
        and _has_declared_key(params_b, "detector_channel_id")
        and params_a["detector_channel_id"] != params_b["detector_channel_id"]
    )
    hard_reasons: list[str] = []
    review_reasons: list[str] = []
    if electron_live_conflict and not separate_sample:
        hard_reasons.append("electron/vacuum destructive or live-sample conflict")
    if destructive_conflict and not separate_sample:
        review_reasons.append("destructive modality requires sequential/separate-sample interpretation")
    if same_quanta or same_detector:
        review_reasons.append("same quanta/detector channel risks double counting")
    if wf_tirf_pair and not independent_budgets:
        review_reasons.append("fluorescence pair needs independent excitation/detection budget declaration")
    if not (pa["field_of_view_compatible"] and pb["field_of_view_compatible"]):
        review_reasons.append("field-of-view compatibility not established")
    if hard_reasons:
        status = "incompatible_hard_stop"
        mode = "algebraic_diagnostic_only"
    elif review_reasons:
        status = "requires_physical_design_review"
        mode = "requires_physical_design_review"
    else:
        status = "physically_feasible_sequential_or_simultaneous"
        mode = "physically_feasible_fusion"
    return {
        "modalities": [a, b],
        "profiles": {a: pa, b: pb},
        "compatible": status == "physically_feasible_sequential_or_simultaneous",
        "reason": "; ".join(hard_reasons + review_reasons) or "no pairwise incompatibility detected by declared profile metadata",
        "required_review": bool(review_reasons and not hard_reasons),
        "incompatible_hard_stop": bool(hard_reasons),
        "compatible_only_as_sequential": bool(destructive_conflict and not hard_reasons),
        "compatible_only_as_algebraic_diagnostic": bool(hard_reasons or same_quanta or same_detector),
        "physical_feasibility_status": status,
        "physical_compatibility_status": status,
        "fusion_mode": mode,
        "independent_noise_assumption": not (same_quanta or same_detector),
        "same_sample_state_required": True,
        "double_count_risk": bool(same_quanta or same_detector),
        "same_quanta_reconstruction_risk": bool(same_quanta),
        "destructive_measurement_conflict": bool(destructive_conflict and not separate_sample),
        "live_sample_conflict": bool(electron_live_conflict and not separate_sample),
        "preparation_conflict": bool(electron_live_conflict and not separate_sample),
        "fusion_interpretation": mode,
    }


def fusion_subset_metadata(
    modalities: list[str] | tuple[str, ...],
    profile_cards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize compatibility for a fused subset."""
    names = [_canonical(m) for m in modalities]
    cards = {} if profile_cards is None else dict(profile_cards)
    profiles = {name: modality_physical_profile(name, cards.get(name, {})) for name in names}
    pairwise = [compatibility_status(a, b, cards.get(a, {}), cards.get(b, {})) for a, b in combinations(names, 2)]
    statuses = {item["physical_feasibility_status"] for item in pairwise}
    if not pairwise:
        overall = "physically_feasible_sequential_or_simultaneous"
        mode = "physically_feasible_fusion"
    elif "incompatible_hard_stop" in statuses:
        overall = "incompatible_hard_stop"
        mode = "algebraic_diagnostic_only"
    elif len(names) > 4:
        overall = "requires_physical_design_review"
        mode = "algebraic_diagnostic_only"
    elif "requires_physical_design_review" in statuses:
        overall = "requires_physical_design_review"
        mode = "requires_physical_design_review"
    else:
        overall = "physically_feasible_sequential_or_simultaneous"
        mode = "physically_feasible_fusion"
    cost_models = {name: acquisition_cost_from_profile(name, cards.get(name, {})).to_dict() for name in names}
    return {
        "modalities": names,
        "modality_physical_profiles": profiles,
        "pairwise_compatibility": pairwise,
        "compatible": overall == "physically_feasible_sequential_or_simultaneous",
        "reason": (
            "; ".join(item["reason"] for item in pairwise if item.get("reason"))
            + ("; " if pairwise and len(names) > 4 else "")
            + ("full-library or broad subset defaults to algebraic diagnostic until a physical acquisition design is declared" if len(names) > 4 else "")
        ) if pairwise else "single-modality subset",
        "required_review": overall == "requires_physical_design_review",
        "incompatible_hard_stop": overall == "incompatible_hard_stop",
        "compatible_only_as_sequential": any(item["compatible_only_as_sequential"] for item in pairwise),
        "compatible_only_as_algebraic_diagnostic": mode == "algebraic_diagnostic_only",
        "physical_feasibility_status": overall,
        "physical_compatibility_status": overall,
        "fusion_mode": mode,
        "algebraic_fusion_allowed": True,
        "physically_feasible_fusion_allowed": mode == "physically_feasible_fusion",
        "independent_noise_assumption": all(item["independent_noise_assumption"] for item in pairwise) if pairwise else True,
        "same_sample_state_required": True,
        "double_count_risk": any(item["double_count_risk"] for item in pairwise),
        "same_quanta_reconstruction_risk": any(item["same_quanta_reconstruction_risk"] for item in pairwise),
        "destructive_measurement_conflict": any(item["destructive_measurement_conflict"] for item in pairwise),
        "live_sample_conflict": any(item["live_sample_conflict"] for item in pairwise),
        "preparation_conflict": any(item["preparation_conflict"] for item in pairwise),
        "fusion_interpretation": mode,
        "acquisition_cost_models": cost_models,
        "requires_physical_design_review": overall != "physically_feasible_sequential_or_simultaneous",
    }


def filter_physically_feasible_subsets(subsets: list[tuple[str, ...]], profile_cards: dict[str, dict[str, Any]] | None = None) -> list[tuple[str, ...]]:
    """Return only subsets whose declared compatibility metadata supports physical fusion."""
    return [subset for subset in subsets if fusion_subset_metadata(subset, profile_cards).get("physically_feasible_fusion_allowed")]


def check_subset_compatibility(
    modalities: Sequence[str],
    profile_cards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return physical feasibility metadata for a proposed fused subset."""
    meta = fusion_subset_metadata(tuple(modalities), profile_cards)
    reasons = []
    reason = str(meta.get("reason", "")).strip()
    if reason:
        reasons.append(reason)
    return {
        "modalities": list(meta.get("modalities", list(modalities))),
        "physically_feasible": bool(meta.get("physically_feasible_fusion_allowed", False)),
        "algebraic_fisher_sum_allowed": bool(meta.get("algebraic_fusion_allowed", True)),
        "fusion_mode": meta.get("fusion_mode"),
        "physical_feasibility_status": meta.get("physical_feasibility_status"),
        "requires_physical_design_review": bool(meta.get("requires_physical_design_review", False)),
        "reasons": reasons,
        "metadata": meta,
    }
