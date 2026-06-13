"""Typed Fisher comparison candidates.

The compared object at the Fisher boundary is a candidate: one rendered
analysis signal together with its likelihood/noise, units, pixel geometry,
modality metadata, parent status, and derivative provenance.  Comparison code
must consume candidate objects rather than parallel ``dict[str, X]`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FisherCandidate:
    """One candidate's complete Fisher-comparison payload."""

    key: str
    signal: Any
    noise_variance: Any
    modality: str
    pixel_size_nm: float
    measurement_domain: str = "contrast"
    signal_units: str = "contrast"
    noise_variance_units: str | None = None
    analysis_noise_model: Any | None = None
    noise_covariance_kind: str | None = None
    parent_result_metadata: Mapping[str, Any] = field(default_factory=dict)
    derivative_context: Any | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("FisherCandidate.key must be non-empty.")
        modality = str(self.modality).strip()
        if not modality:
            raise ValueError(f"FisherCandidate[{key!r}] must declare physical modality metadata.")
        pixel_size = float(self.pixel_size_nm)
        if not np.isfinite(pixel_size) or pixel_size <= 0.0:
            raise ValueError(
                f"FisherCandidate[{key!r}].pixel_size_nm must be positive and finite; "
                f"got {self.pixel_size_nm!r}."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "pixel_size_nm", pixel_size)
        object.__setattr__(self, "measurement_domain", str(self.measurement_domain or "contrast"))
        object.__setattr__(self, "signal_units", str(self.signal_units or "contrast"))
        object.__setattr__(
            self,
            "noise_variance_units",
            None if self.noise_variance_units in {None, ""} else str(self.noise_variance_units),
        )
        object.__setattr__(self, "parent_result_metadata", dict(self.parent_result_metadata or {}))


@dataclass(frozen=True)
class FisherMatrixCandidate:
    """One candidate after Fisher assembly, with scheduling/fusion metadata."""

    key: str
    fisher_matrix: Any
    dt_seconds: float = 1.0
    parent_result_metadata: Mapping[str, Any] = field(default_factory=dict)
    acquisition_cost: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("FisherMatrixCandidate.key must be non-empty.")
        matrix = np.asarray(self.fisher_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError(
                f"FisherMatrixCandidate[{key!r}].fisher_matrix must be a square "
                f"2-D matrix with at least two axes; got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"FisherMatrixCandidate[{key!r}].fisher_matrix must contain only finite values."
            )
        dt = float(self.dt_seconds)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"FisherMatrixCandidate[{key!r}].dt_seconds must be finite and positive; "
                f"got {self.dt_seconds!r}."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "fisher_matrix", matrix)
        object.__setattr__(self, "dt_seconds", dt)
        object.__setattr__(self, "parent_result_metadata", dict(self.parent_result_metadata or {}))
        object.__setattr__(self, "acquisition_cost", dict(self.acquisition_cost or {}))
        object.__setattr__(self, "constraints", dict(self.constraints or {}))


def _candidate_keys(candidates: Sequence[Any], *, context: str = "Fisher candidate") -> list[str]:
    keys = [candidate.key for candidate in candidates]
    if not keys:
        raise ValueError(f"{context} list is empty; nothing to compare.")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"{context} keys must be unique; duplicates: {duplicates!r}.")
    return keys


def candidate_metadata_records(candidates: Sequence[FisherCandidate]) -> list[dict[str, Any]]:
    """Return per-candidate metadata records derived from typed candidates."""

    _candidate_keys(candidates)
    records = []
    for candidate in candidates:
        records.append(
            {
                "candidate_key": candidate.key,
                "modality": candidate.modality,
                "pixel_size_nm": candidate.pixel_size_nm,
                "measurement_domain": candidate.measurement_domain,
                "signal_units": candidate.signal_units,
                "noise_variance_units": candidate.noise_variance_units,
            }
        )
    return records


def matrix_candidate_metadata_records(
    candidates: Sequence[FisherMatrixCandidate],
) -> list[dict[str, Any]]:
    """Return per-candidate metadata records derived from matrix candidates."""

    _candidate_keys(candidates, context="Fisher matrix candidate")
    records = []
    for candidate in candidates:
        record: dict[str, Any] = {
            "candidate_key": candidate.key,
            "dt_seconds": candidate.dt_seconds,
        }
        if candidate.parent_result_metadata:
            record["parent_result_metadata"] = dict(candidate.parent_result_metadata)
        if candidate.acquisition_cost:
            record["acquisition_cost"] = dict(candidate.acquisition_cost)
        if candidate.constraints:
            record["constraints"] = dict(candidate.constraints)
        records.append(record)
    return records


__all__ = [
    "FisherCandidate",
    "FisherMatrixCandidate",
    "candidate_metadata_records",
    "matrix_candidate_metadata_records",
]
