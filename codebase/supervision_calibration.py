"""Calibration helpers for supervision support scores."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_score_calibration(scores: np.ndarray | float, calibration: dict[str, Any] | None) -> np.ndarray:
    """Map bounded support scores to calibrated probabilities when configured."""
    arr = np.asarray(scores, dtype=float)
    clipped = np.clip(arr, 0.0, 1.0)
    if not calibration:
        return clipped
    mode = str(calibration.get("mode", "uncalibrated_support")).strip().lower()
    if mode in {"", "uncalibrated_support", "none"}:
        return clipped
    if mode == "platt_logistic":
        slope = float(calibration.get("slope", 1.0))
        intercept = float(calibration.get("intercept", 0.0))
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise ValueError("Platt calibration slope/intercept must be finite.")
        logits = slope * clipped + intercept
        return np.where(logits >= 0.0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))
    if mode == "isotonic":
        x = np.asarray(calibration.get("score_breakpoints", []), dtype=float)
        y = np.asarray(calibration.get("probabilities", []), dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
            raise ValueError("Isotonic calibration requires matching score_breakpoints/probabilities with length >= 2.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Isotonic calibration values must be finite.")
        order = np.argsort(x)
        return np.clip(np.interp(clipped, x[order], y[order]), 0.0, 1.0)
    raise ValueError(f"Unsupported supervision calibration mode {mode!r}.")


def calibration_contract(calibration: dict[str, Any] | None) -> dict[str, Any]:
    mode = str((calibration or {}).get("mode", "uncalibrated_support")).strip().lower()
    if mode in {"", "none"}:
        mode = "uncalibrated_support"
    calibrated = mode in {"platt_logistic", "isotonic"}
    return {
        "calibration_mode": mode,
        "calibration_status": "calibrated_probability" if calibrated else "uncalibrated_support",
        "probability_semantics": (
            "empirically_calibrated_probability"
            if calibrated
            else "bounded_support_score_not_probability"
        ),
        "calibration_parameters_supplied": bool(calibration),
    }


def reliability_summary(scores: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    """Return Brier score and expected calibration error for calibrated outputs."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty arrays with matching shape.")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
        raise ValueError("scores and labels must be finite.")
    s = np.clip(s, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    rows = []
    lower_edges = edges[:-1]
    upper_edges = edges[1:]
    if lower_edges.shape != upper_edges.shape:
        raise RuntimeError("Calibration bin edge construction produced mismatched intervals.")
    for lo, hi in zip(lower_edges, upper_edges):
        mask = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            rows.append({"lo": float(lo), "hi": float(hi), "count": 0})
            continue
        conf = float(np.mean(s[mask]))
        acc = float(np.mean(y[mask]))
        ece += (count / s.size) * abs(conf - acc)
        rows.append({"lo": float(lo), "hi": float(hi), "count": count, "mean_score": conf, "mean_label": acc})
    return {
        "brier_score": float(np.mean((s - y) ** 2)),
        "expected_calibration_error": float(ece),
        "bins": rows,
    }
