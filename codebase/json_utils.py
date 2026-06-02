"""Shared JSON normalization helpers for Syniscopy metadata writers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Literal
from pathlib import Path
import json

import numpy as np

NonfinitePolicy = Literal["null", "tag", "string", "passthrough"]

# Manifest writers must use this module for JSON normalization so provenance
# artifacts stay byte-stable across emitters. By default, non-finite floats are
# converted to null for strict ``allow_nan=False`` JSON; diagnostics that must
# preserve NaN/+inf/-inf provenance use ``json_safe_with_nonfinite_tags``.
# Complex values are always represented as {"real": ..., "imag": ...}.


def _nonfinite_value(value: float, policy: NonfinitePolicy) -> Any:
    if policy == "passthrough":
        return value
    if policy == "null":
        return None
    if policy == "string":
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if math.isnan(value):
        return {"nonfinite": "nan"}
    return {"nonfinite": "posinf" if value > 0.0 else "neginf"}


def json_safe(
    value: Any,
    *,
    nonfinite: NonfinitePolicy = "null",
    complex_values: bool = True,
    flexible_numpy: bool = False,
) -> Any:
    """
    Convert common Python/NumPy values into JSON-compatible metadata values.

    ``nonfinite="null"`` preserves the strict JSON-writer behavior used by
    manifests that pass ``allow_nan=False``. ``nonfinite="tag"`` keeps explicit
    provenance for NaN/+inf/-inf values in diagnostic metadata.
    """
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(
            value.to_dict(),
            nonfinite=nonfinite,
            complex_values=complex_values,
            flexible_numpy=flexible_numpy,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(
            asdict(value),
            nonfinite=nonfinite,
            complex_values=complex_values,
            flexible_numpy=flexible_numpy,
        )
    if isinstance(value, Mapping):
        return {
            str(k): json_safe(
                v,
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray))
    ):
        return [
            json_safe(
                v,
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            )
            for v in value
        ]
    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist(),
            nonfinite=nonfinite,
            complex_values=complex_values,
            flexible_numpy=flexible_numpy,
        )
    if isinstance(value, np.generic):
        return json_safe(
            value.item(),
            nonfinite=nonfinite,
            complex_values=complex_values,
            flexible_numpy=flexible_numpy,
        )
    if flexible_numpy and hasattr(value, "tolist"):
        try:
            return json_safe(
                value.tolist(),
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"Could not convert {type(value).__name__}.tolist() result to JSON-safe data."
            ) from exc
    if isinstance(value, complex):
        if not complex_values:
            return value
        return {
            "real": json_safe(
                float(value.real),
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            ),
            "imag": json_safe(
                float(value.imag),
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            ),
        }
    if flexible_numpy and hasattr(value, "item"):
        try:
            return json_safe(
                value.item(),
                nonfinite=nonfinite,
                complex_values=complex_values,
                flexible_numpy=flexible_numpy,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"Could not convert {type(value).__name__}.item() result to JSON-safe data."
            ) from exc
    if isinstance(value, float):
        return value if math.isfinite(value) else _nonfinite_value(value, nonfinite)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-safe.")


def json_safe_with_nonfinite_tags(value: Any, *, flexible_numpy: bool = False) -> Any:
    """JSON-safe normalization that preserves NaN/+inf/-inf as tagged dicts."""
    return json_safe(value, nonfinite="tag", flexible_numpy=flexible_numpy)


def load_typed_json(path: str | Path, *, expected: type[Any] | tuple[type[Any], ...], context: str) -> Any:
    """Load JSON and validate its top-level Python type.

    Parameters
    ----------
    path:
        JSON file path. Relative paths are interpreted as given.
    expected:
        One type or tuple of types that the loaded JSON must be.
    context:
        Human-readable prefix for validation errors (for example the CLI flag
        name used to supply the file path).

    Raises
    ------
    ValueError
        When the loaded value is not one of the expected Python types.
    """

    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        value = json.load(fh)

    if not isinstance(value, expected):
        if isinstance(expected, tuple):
            expected_names = [t.__name__ for t in expected]
            expected_name = " or ".join(expected_names)
        else:
            expected_name = expected.__name__
        raise ValueError(f"{context} must contain a JSON {expected_name}.")
    return value
