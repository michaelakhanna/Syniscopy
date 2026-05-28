"""
supervision_policy.py — Syniscopy supervision-target policy.

The simulator has access to the full latent particle geometry, but a training
mask should not always mean "all pixels occupied by the simulated object are
positive supervision." Some frames contain real particle signal that is too
weak or too temporally implausible to use as ordinary
foreground/background supervision. This module centralizes that decision.

The policy composes four support factors:

    temporal_support     Brownian-step plausibility from trackability.py.
    signal_support       Frame-local signal evidence against detector noise.
    information_support  CRLB-based localizability from fisher_diagnostic.py.
    ambiguity_support    Assignment confidence; defaults to the uniform
                         no-competition special case when competitor state is
                         not supplied.

All factors are heuristic support/confidence factors in [0, 1], not calibrated
probabilities. Unsupported object pixels are routed to ignore_mask, not to
background, so downstream training does not learn false negatives from frames
where the simulator knows an object exists but supervision is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dataset_schema import SUPPORTED_MASK_TARGETS, build_annotation_schema
from fisher_diagnostic import compute_localization_crlb
from trackability import TrackabilityModel
SUPPORTED_TARGETS = SUPPORTED_MASK_TARGETS
SUPPORTED_FACTORS = ("temporal", "signal", "information", "ambiguity")


def _bounded_support(value: float) -> float:
    """Return a finite support factor in [0, 1], mapping invalid values to 0."""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _normalise_target_and_factors(
    params: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    raw_target = str(params.get("supervision_target", "mask_supported")).strip()

    if raw_target not in SUPPORTED_TARGETS:
        raise ValueError(
            "PARAMS['supervision_target'] must be one of "
            f"{SUPPORTED_TARGETS}; got {raw_target!r}."
        )

    explicit_factors = params.get("supervision_support_factors")
    if explicit_factors is None:
        if raw_target == "mask_geometry":
            factors: tuple[str, ...] = ()
        else:
            factors = tuple(
                factor for factor in SUPPORTED_FACTORS
                if bool(params.get(f"supervision_{factor}_support_enabled", True))
            )
    elif isinstance(explicit_factors, str):
        factors = tuple(
            factor.strip()
            for factor in explicit_factors.split(",")
            if factor.strip()
        )
    else:
        factors = tuple(str(factor).strip() for factor in explicit_factors)

    invalid = [factor for factor in factors if factor not in SUPPORTED_FACTORS]
    if invalid:
        raise ValueError(
            "PARAMS['supervision_support_factors'] contains unsupported "
            f"factor(s) {invalid}; supported factors are {SUPPORTED_FACTORS}."
        )
    seen = set()
    duplicates = []
    for factor in factors:
        if factor in seen and factor not in duplicates:
            duplicates.append(factor)
        seen.add(factor)
    if duplicates:
        raise ValueError(
            "PARAMS['supervision_support_factors'] contains duplicate factor(s) "
            f"{duplicates}. Each support factor may be listed at most once."
        )
    if explicit_factors is not None:
        disabled = [
            factor for factor in factors
            if not bool(params.get(f"supervision_{factor}_support_enabled", True))
        ]
        if disabled:
            raise ValueError(
                "PARAMS['supervision_support_factors'] explicitly includes disabled "
                f"factor(s) {disabled}. Remove the factor(s) or enable their "
                "corresponding supervision_*_support_enabled flag."
            )
    return raw_target, factors


def build_policy_annotation_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Build the annotation schema using the supervision policy contract."""
    target, factors = _normalise_target_and_factors(params)
    return build_annotation_schema(selected_target=target, support_factors=factors)


def resolve_policy_contract(params: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized supervision policy fields used in manifests."""
    target, factors = _normalise_target_and_factors(params)
    return {
        "target": target,
        "support_factors": list(factors),
    }




def estimate_contrast_noise_std(params: dict[str, Any]) -> float:
    """
    Estimate detector noise in per-particle contrast-image units.

    The estimate is tied to the same counts-domain camera-noise model used by
    rendering instead of maintaining a separate shot-noise formula here.
    """
    from camera_noise import estimate_contrast_noise_std_from_params

    return estimate_contrast_noise_std_from_params(params)






def compute_signal_support(
    contrast_image: np.ndarray,
    noise_std: float,
) -> tuple[float, float]:
    """
    Return (signal_support, snr) from a per-particle contrast image.
    """
    contrast_abs = np.abs(np.asarray(contrast_image, dtype=float))
    if contrast_abs.size == 0:
        return 0.0, 0.0
    if not np.all(np.isfinite(contrast_abs)):
        return 0.0, 0.0
    max_contrast = float(contrast_abs.max())
    if not np.isfinite(noise_std) or noise_std < 0.0:
        return 0.0, 0.0
    if noise_std <= 0.0:
        support = 1.0 if max_contrast > 0.0 else 0.0
        snr = float("inf") if max_contrast > 0.0 else 0.0
    else:
        snr = max_contrast / float(noise_std)
        support = float(1.0 - np.exp(-0.5 * snr**2))
    return _bounded_support(support), float(snr)


def _crlb_support_from_sigma(
    sigma_xy_nm: float,
    pixel_size_nm: float,
    max_sigma_nm: float | None,
) -> float:
    if max_sigma_nm is None or max_sigma_nm <= 0.0:
        max_sigma_nm = pixel_size_nm
    if not np.isfinite(sigma_xy_nm):
        return 0.0
    if sigma_xy_nm <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + (sigma_xy_nm / max_sigma_nm) ** 2))


def compute_information_support(
    contrast_image: np.ndarray,
    params: dict[str, Any],
    noise_std: float,
    noise_variance_map: np.ndarray | float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute CRLB metadata and a bounded information-support factor.
    """
    if not bool(params.get("supervision_information_support_enabled", True)):
        return 1.0, {
            "sigma_xy_nm": None,
            "singular": False,
            "support_disabled": True,
            "support_evaluated": False,
        }

    pixel_size_nm = float(params["pixel_size_nm"])
    variance = (
        noise_variance_map
        if noise_variance_map is not None
        else max(float(noise_std) ** 2, 1e-30)
    )
    try:
        crlb = compute_localization_crlb(
            np.asarray(contrast_image, dtype=float),
            variance,
            pixel_size_nm=pixel_size_nm,
        )
    except Exception as exc:  # noqa: BLE001 - downgrade to unsupported metadata
        return 0.0, {
            "sigma_xy_nm": float("inf"),
            "singular": True,
            "error": repr(exc),
            "support_evaluated": True,
        }

    sigma_xy_nm = float(crlb.get("sigma_xy_nm", float("inf")))
    support = _crlb_support_from_sigma(
        sigma_xy_nm=sigma_xy_nm,
        pixel_size_nm=pixel_size_nm,
        max_sigma_nm=params.get("supervision_crlb_xy_max_nm", pixel_size_nm),
    )
    return support, {
        "sigma_xy_nm": sigma_xy_nm,
        "sigma_x_nm": float(crlb.get("sigma_x_nm", float("inf"))),
        "sigma_y_nm": float(crlb.get("sigma_y_nm", float("inf"))),
        "singular": bool(crlb.get("singular", False)),
        "rank": crlb.get("rank"),
        "axes_singular": list(crlb.get("axes_singular", [])),
        "fisher_det": float(crlb.get("fisher_det", 0.0)),
        "support_evaluated": True,
    }


def _neutral_information_support() -> tuple[float, dict[str, Any]]:
    return 1.0, {
        "sigma_xy_nm": None,
        "singular": False,
        "support_evaluated": False,
    }


def compute_assignment_ambiguity_support(
    particle_index: int,
    position_nm: np.ndarray,
    geometry_mask: np.ndarray,
    all_positions_nm: np.ndarray | list[np.ndarray] | None = None,
    all_geometry_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Estimate assignment support for a particle-frame.

    When no frame-level competitor state is supplied, this returns the
    uniform-assignment special case: no competitor is preferred, so ambiguity
    contributes a multiplicative support factor of one.
    """
    params = params or {}
    particle_index = int(particle_index)
    geom = np.asarray(geometry_mask) > 0
    geometry_pixels = int(np.count_nonzero(geom))

    nearest_distance = float("inf")
    mask_overlap_pixels = 0
    competition_terms: list[float] = []

    if all_positions_nm is not None:
        positions = np.asarray(all_positions_nm, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if 0 <= particle_index < len(positions):
            own = np.asarray(position_nm, dtype=float).reshape(-1)
            if own.size == 0:
                own = positions[particle_index].reshape(-1)
            dim = min(3, own.size, positions.shape[1])
            scale_value = params.get("supervision_ambiguity_distance_scale_nm", None)
            if scale_value is None:
                # Two detector pixels is the default assignment-ambiguity scale;
                # tighten via supervision_ambiguity_distance_scale_nm for dense scenes.
                scale_value = 2.0 * float(params.get("pixel_size_nm", 1.0))
            scale = float(scale_value)
            if scale > 0.0 and np.isfinite(scale):
                for idx, candidate in enumerate(positions):
                    if idx == particle_index:
                        continue
                    delta = candidate[:dim] - own[:dim]
                    if not np.all(np.isfinite(delta)):
                        continue
                    distance = float(np.linalg.norm(delta))
                    nearest_distance = min(nearest_distance, distance)
                    competition_terms.append(float(np.exp(-0.5 * (distance / scale) ** 2)))

    if all_geometry_masks is not None:
        for idx, candidate_mask in enumerate(all_geometry_masks):
            if idx == particle_index:
                continue
            candidate = np.asarray(candidate_mask) > 0
            if candidate.shape != geom.shape:
                raise ValueError(
                    "all_geometry_masks entries must have the same shape as geometry_mask; "
                    f"got {candidate.shape} and {geom.shape}."
                )
            overlap = int(np.count_nonzero(geom & candidate))
            mask_overlap_pixels += overlap
            if geometry_pixels > 0:
                competition_terms.append(float(overlap / geometry_pixels))

    assignment_competition = float(sum(competition_terms))
    if assignment_competition <= 0.0:
        support = 1.0
        odds = float("inf")
        uniform_special_case = True
    else:
        support = float(1.0 / (1.0 + assignment_competition))
        odds = float(1.0 / assignment_competition)
        uniform_special_case = False

    meta = {
        "nearest_competitor_distance_nm": nearest_distance,
        "mask_overlap_pixels": int(mask_overlap_pixels),
        "assignment_odds_ratio": odds,
        "ambiguity_support": support,
        "uniform_ambiguity_special_case": uniform_special_case,
        "assignment_competition": assignment_competition,
        "support_evaluated": True,
    }
    return support, meta


def _neutral_ambiguity_support() -> tuple[float, dict[str, Any]]:
    return 1.0, {
        "nearest_competitor_distance_nm": float("inf"),
        "mask_overlap_pixels": 0,
        "assignment_odds_ratio": float("inf"),
        "ambiguity_support": 1.0,
        "uniform_ambiguity_special_case": True,
        "assignment_competition": 0.0,
        "support_evaluated": False,
    }


def compute_supervision_log_odds_components(
    *,
    signal_support: float,
    information_support: float,
    temporal_support: float,
    ambiguity_support: float = 1.0,
    included_factors: tuple[str, ...] | list[str] | None = None,
    prior_log_odds: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Convert bounded support factors into additive clipped log-odds terms.

    The support factors are bounded approximations, not calibrated
    probabilities; this helper exposes the additive decomposition used by the
    supervision theorem while keeping numerical edge cases finite.
    """
    eps = float(eps)
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must lie in (0, 0.5).")

    def _logit(value: float) -> float:
        clipped = min(1.0 - eps, max(eps, _bounded_support(value)))
        return float(np.log(clipped / (1.0 - clipped)))

    selected = None if included_factors is None else set(included_factors)

    def _component(factor: str, value: float) -> float:
        if selected is not None and factor not in selected:
            return 0.0
        return _logit(value)

    components = {
        "L_signal": _component("signal", signal_support),
        "L_information": _component("information", information_support),
        "L_temporal": _component("temporal", temporal_support),
        "L_ambiguity": _component("ambiguity", ambiguity_support),
        "L_prior": float(prior_log_odds),
    }
    total = float(
        components["L_signal"]
        + components["L_information"]
        + components["L_temporal"]
        + components["L_ambiguity"]
        + components["L_prior"]
    )
    if total >= 0.0:
        logistic_score = 1.0 / (1.0 + float(np.exp(-total)))
    else:
        exp_total = float(np.exp(total))
        logistic_score = exp_total / (1.0 + exp_total)
    components["L_total"] = total
    components["logistic_support_score"] = float(logistic_score)
    return components


@dataclass
class SupervisionAudit:
    """Accumulates per-video supervision-policy summary statistics."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        total = len(self.records)
        if total == 0:
            return {
                "num_particle_frames": 0,
                "drop_reason_counts": {},
            }

        def _values(key: str) -> list[float]:
            vals: list[float] = []
            for rec in self.records:
                evaluated_key = f"{key}_evaluated"
                if evaluated_key in rec and not bool(rec[evaluated_key]):
                    continue
                value = rec.get(key)
                if value is None:
                    continue
                if isinstance(value, (int, float)) and np.isfinite(value):
                    vals.append(float(value))
            return vals

        def _hist(key: str, bins: int = 10, lo: float = 0.0, hi: float = 1.0):
            vals = _values(key)
            if not vals:
                return {"bins": [], "counts": []}
            counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
            return {
                "bins": [float(x) for x in edges.tolist()],
                "counts": [int(x) for x in counts.tolist()],
            }

        crlb_vals = [
            float(rec["crlb_xy_nm"])
            for rec in self.records
            if isinstance(rec.get("crlb_xy_nm"), (int, float))
            and np.isfinite(float(rec["crlb_xy_nm"]))
        ]
        crlb_hist = {"bins": [], "counts": []}
        if crlb_vals:
            upper = max(max(crlb_vals), 1.0)
            counts, edges = np.histogram(crlb_vals, bins=10, range=(0.0, upper))
            crlb_hist = {
                "bins": [float(x) for x in edges.tolist()],
                "counts": [int(x) for x in counts.tolist()],
            }

        drop_counts: dict[str, int] = {}
        for rec in self.records:
            reason = str(rec.get("drop_reason", "supported"))
            drop_counts[reason] = drop_counts.get(reason, 0) + 1

        geom_pixels = sum(int(rec.get("geometry_pixels", 0)) for rec in self.records)
        ignore_pixels = sum(int(rec.get("ignore_pixels", 0)) for rec in self.records)
        positive_pixels = sum(int(rec.get("target_pixels", 0)) for rec in self.records)

        return {
            "num_particle_frames": total,
            "signal_support_histogram": _hist("signal_support"),
            "temporal_support_histogram": _hist("temporal_support"),
            "information_support_histogram": _hist("information_support"),
            "ambiguity_support_histogram": _hist("ambiguity_support"),
            "crlb_xy_histogram": crlb_hist,
            "fisher_singular_fraction": float(
                sum(bool(rec.get("fisher_singular", False)) for rec in self.records)
                / total
            ),
            "low_signal_particle_fraction": float(
                sum("low_signal" in str(rec.get("drop_reason", "")) for rec in self.records)
                / total
            ),
            "high_crlb_particle_fraction": float(
                sum("high_crlb" in str(rec.get("drop_reason", "")) for rec in self.records)
                / total
            ),
            "ignored_particle_fraction": float(
                sum(int(rec.get("ignore_pixels", 0)) > 0 for rec in self.records)
                / total
            ),
            "ignored_pixel_fraction": float(ignore_pixels / geom_pixels)
            if geom_pixels > 0 else 0.0,
            "positive_supervision_pixel_fraction": float(positive_pixels / geom_pixels)
            if geom_pixels > 0 else 0.0,
            "drop_reason_counts": drop_counts,
        }


class SupervisionPolicy:
    """
    Per-video supervision-policy evaluator.

    The default target is ``mask_supported``: the geometry/support mask filtered
    by the configured decision rule. ``mask_geometry`` remains available when a
    caller wants the pre-gating simulator target.
    """

    def __init__(self, params: dict[str, Any], num_particles: int):
        self.params = params
        self.num_particles = int(num_particles)
        self.target, self.support_factors = _normalise_target_and_factors(params)

        self.temporal_enabled = bool(
            params.get("supervision_temporal_support_enabled", True)
        )
        self.signal_enabled = bool(
            params.get("supervision_signal_support_enabled", True)
        )
        self.support_threshold = float(
            params.get("supervision_supported_threshold", 0.2)
        )
        if not (0.0 <= self.support_threshold <= 1.0):
            raise ValueError("supervision_supported_threshold must be in [0, 1].")
        self.decision_rule = str(
            params.get("supervision_decision_rule", "log_odds")
        ).strip().lower()
        if self.decision_rule not in {"log_odds", "product"}:
            raise ValueError(
                "supervision_decision_rule must be 'log_odds' or 'product'."
            )
        self.log_odds_threshold = float(params.get("supervision_log_odds_threshold", 0.0))

        self.temporal_model = (
            TrackabilityModel(params, self.num_particles)
            if self.temporal_enabled else None
        )
        self.noise_std = estimate_contrast_noise_std(params)

    def evaluate(
        self,
        *,
        particle_index: int,
        frame_index: int,
        position_nm: np.ndarray,
        contrast_image: np.ndarray,
        geometry_mask: np.ndarray,
        all_positions_nm: np.ndarray | list[np.ndarray] | None = None,
        all_geometry_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        noise_std: float | None = None,
        noise_variance_map: np.ndarray | float | None = None,
    ) -> dict[str, Any]:
        geom = (np.asarray(geometry_mask) > 0).astype(np.uint8) * 255
        geom_bool = geom > 0
        geometry_pixels = int(np.count_nonzero(geom))

        if self.temporal_model is None:
            temporal_support = 1.0
        else:
            temporal_support = self.temporal_model.update_and_compute(
                particle_index=particle_index,
                frame_index=frame_index,
                position_nm=position_nm,
            )

        if not self.signal_enabled:
            signal_support, snr = 1.0, float("inf")
        else:
            local_noise_std = self.noise_std if noise_std is None else float(noise_std)
            signal_support, snr = compute_signal_support(
                contrast_image=contrast_image,
                noise_std=local_noise_std,
            )

        if "information" in self.support_factors:
            local_noise_std = self.noise_std if noise_std is None else float(noise_std)
            information_support, crlb_meta = compute_information_support(
                contrast_image=contrast_image,
                params=self.params,
                noise_std=local_noise_std,
                noise_variance_map=noise_variance_map,
            )
        else:
            information_support, crlb_meta = _neutral_information_support()
        if "ambiguity" in self.support_factors:
            ambiguity_support, ambiguity_meta = compute_assignment_ambiguity_support(
                particle_index=particle_index,
                position_nm=position_nm,
                geometry_mask=geom,
                all_positions_nm=all_positions_nm,
                all_geometry_masks=all_geometry_masks,
                params=self.params,
            )
        else:
            ambiguity_support, ambiguity_meta = _neutral_ambiguity_support()

        factor_values = {
            "temporal": _bounded_support(temporal_support),
            "signal": _bounded_support(signal_support),
            "information": _bounded_support(information_support),
            "ambiguity": _bounded_support(ambiguity_support),
        }
        temporal_support = factor_values["temporal"]
        signal_support = factor_values["signal"]
        information_support = factor_values["information"]
        ambiguity_support = factor_values["ambiguity"]
        support_score = 1.0
        for factor in self.support_factors:
            support_score *= factor_values[factor]
        log_odds_components = compute_supervision_log_odds_components(
            signal_support=signal_support,
            information_support=information_support,
            temporal_support=temporal_support,
            ambiguity_support=ambiguity_support,
            included_factors=self.support_factors,
            prior_log_odds=float(self.params.get("supervision_prior_log_odds", 0.0)),
        )

        factor_maps = {
            factor: np.full(geom.shape, factor_values[factor], dtype=float)
            for factor in SUPPORTED_FACTORS
        }
        if "ambiguity" in self.support_factors and all_geometry_masks is not None:
            overlap_count = np.zeros(geom.shape, dtype=float)
            for idx, candidate_mask in enumerate(all_geometry_masks):
                if idx == int(particle_index):
                    continue
                candidate = np.asarray(candidate_mask) > 0
                if candidate.shape != geom.shape:
                    raise ValueError(
                        "all_geometry_masks entries must have the same shape as geometry_mask; "
                        f"got {candidate.shape} and {geom.shape}."
                    )
                overlap_count += candidate.astype(float)
            overlap_support_map = 1.0 / (1.0 + overlap_count)
            factor_maps["ambiguity"] = np.minimum(
                factor_maps["ambiguity"],
                overlap_support_map,
            )

        support_score_map = np.ones(geom.shape, dtype=float)
        for factor in self.support_factors:
            support_score_map *= factor_maps[factor]

        eps = 1e-12
        log_odds_map = np.full(
            geom.shape,
            float(self.params.get("supervision_prior_log_odds", 0.0)),
            dtype=float,
        )
        for factor in self.support_factors:
            clipped = np.clip(factor_maps[factor], eps, 1.0 - eps)
            log_odds_map += np.log(clipped / (1.0 - clipped))
        logistic_support_map = np.where(
            log_odds_map >= 0.0,
            1.0 / (1.0 + np.exp(-log_odds_map)),
            np.exp(log_odds_map) / (1.0 + np.exp(log_odds_map)),
        )

        if self.decision_rule == "product":
            supported_pixels = geom_bool & (support_score_map >= self.support_threshold)
            decision_value = support_score
            decision_threshold = self.support_threshold
        else:
            supported_pixels = geom_bool & (log_odds_map >= self.log_odds_threshold)
            decision_value = float(log_odds_components["L_total"])
            decision_threshold = self.log_odds_threshold

        mask_geometry = geom
        mask_supported = supported_pixels.astype(np.uint8) * 255

        selected = mask_geometry if self.target == "mask_geometry" else mask_supported
        selected_bool = selected > 0
        ignore_mask = np.where((geom > 0) & (selected == 0), 255, 0).astype(np.uint8)

        reasons: list[str] = []
        if geometry_pixels == 0:
            reasons.append("out_of_frame")
        if "signal" in self.support_factors and signal_support < self.support_threshold:
            reasons.append("low_signal")
        if (
            "information" in self.support_factors
            and information_support < self.support_threshold
        ):
            reasons.append("high_crlb")
        if "information" in self.support_factors and crlb_meta.get("singular", False):
            reasons.append("fisher_singular")
        if "temporal" in self.support_factors and temporal_support < self.support_threshold:
            reasons.append("implausible_brownian_step")
        if "ambiguity" in self.support_factors and ambiguity_support < self.support_threshold:
            reasons.append("ambiguous_assignment")
        if (
            "ambiguity" in self.support_factors
            and 0 < int(np.count_nonzero(mask_supported)) < geometry_pixels
        ):
            reasons.append("partial_overlap_ambiguity")
        if (
            self.target == "mask_geometry"
            and "information" in self.support_factors
            and crlb_meta.get("singular", False)
        ):
            reasons.append("geometry_target_fisher_singular")

        has_ignored_pixels = np.count_nonzero(ignore_mask) > 0
        if geometry_pixels == 0:
            drop_reason = "out_of_frame"
        elif int(np.count_nonzero(selected)) == 0:
            drop_reason = "+".join(reasons or ["unsupported"])
        elif has_ignored_pixels:
            drop_reason = "+".join(reasons or ["partially_supported"])
        else:
            drop_reason = "supported"

        if self.target == "mask_geometry":
            loss_weight_map = geom_bool.astype(float)
        elif self.decision_rule == "product":
            loss_weight_map = support_score_map
        else:
            loss_weight_map = logistic_support_map
        loss_weight_map = np.clip(loss_weight_map, 0.0, 1.0)
        loss_weight = np.where(
            selected_bool,
            np.rint(255.0 * loss_weight_map),
            0,
        ).astype(np.uint8)
        if geometry_pixels > 0:
            loss_weight_value = float(np.mean(loss_weight_map[geom_bool]))
            supported_pixel_fraction = float(np.count_nonzero(mask_supported) / geometry_pixels)
        else:
            loss_weight_value = 0.0
            supported_pixel_fraction = 0.0

        if self.temporal_model is not None and (
            "temporal" in self.support_factors
            and temporal_support < self.support_threshold
        ):
            self.temporal_model.mark_lost(particle_index)

        return {
            "masks": {
                "mask_geometry": mask_geometry,
                "mask_supported": mask_supported,
                "ignore_mask": ignore_mask,
                "loss_weight": loss_weight,
            },
            "record": {
                "frame_index": int(frame_index),
                "particle_index": int(particle_index),
                "supervision_target": self.target,
                "support_factors": list(self.support_factors),
                "temporal_support": float(temporal_support),
                "signal_support": float(signal_support),
                "information_support": float(information_support),
                "ambiguity_support": float(ambiguity_support),
                "information_support_evaluated": bool(
                    crlb_meta.get("support_evaluated", True)
                ),
                "ambiguity_support_evaluated": bool(
                    ambiguity_meta.get("support_evaluated", True)
                ),
                "support_score": float(support_score),
                "supervision_decision_rule": self.decision_rule,
                "supervision_decision_value": float(decision_value),
                "supervision_decision_threshold": float(decision_threshold),
                "supervision_log_odds_components": log_odds_components,
                "supported_pixel_fraction": supported_pixel_fraction,
                "snr": float(snr),
                "crlb_xy_nm": crlb_meta.get("sigma_xy_nm"),
                "fisher_rank": crlb_meta.get("rank"),
                "fisher_axes_singular": crlb_meta.get("axes_singular", []),
                "assignment_odds_ratio": ambiguity_meta["assignment_odds_ratio"],
                "nearest_competitor_distance_nm": ambiguity_meta[
                    "nearest_competitor_distance_nm"
                ],
                "mask_overlap_pixels": ambiguity_meta["mask_overlap_pixels"],
                "fisher_singular": bool(crlb_meta.get("singular", False)),
                "drop_reason": drop_reason,
                "loss_weight": loss_weight_value,
                "geometry_pixels": geometry_pixels,
                "target_pixels": int(np.count_nonzero(selected)),
                "ignore_pixels": int(np.count_nonzero(ignore_mask)),
            },
        }
