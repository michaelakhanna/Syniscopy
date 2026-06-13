"""
Runtime unit-contract checks for cross-module seams.

The checks are value-neutral: they never convert, coerce, or rescale arrays.
They only validate declared measurement-domain and unit metadata.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from config import UnitContractSettings
from measurement_units import normalize_measurement_domain, normalize_signal_units


class UnitContractError(RuntimeError):
    """Raised when declared unit metadata is incompatible at a seam."""


_DISABLED_ENV_VALUES = {"0", "false", "off", "no", "disabled"}

_SIGNAL_DOMAIN = {
    "detector_count": "count",
    "radian": "phase",
    "electron_count": "electron_count",
    "relative_reference": "contrast",
    "detected_quanta": "detected_quanta",
    "model_contrast": "model_signal",
    "model_signal": "model_signal",
    "contrast": "contrast",
}


def unit_contracts_enabled(params: Mapping[str, Any] | None = None) -> bool:
    """Return whether unit contracts are enabled."""

    if params is not None:
        try:
            return UnitContractSettings.from_params(params).enabled
        except KeyError:
            pass
    env_value = os.environ.get("SYNISCOPY_UNIT_CONTRACTS", "1").strip().lower()
    return env_value not in _DISABLED_ENV_VALUES


def _normalize_domain(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return normalize_measurement_domain(text)
    except ValueError:
        return text


def _normalize_signal(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return normalize_signal_units(text)
    except ValueError:
        return text


def _normalize_variance(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def variance_units_for_signal(signal_units: str | None) -> str | None:
    signal = _normalize_signal(signal_units)
    if signal is None:
        return None
    return f"{signal}_squared"


def _domain_from_signal(signal_units: str | None) -> str | None:
    signal = _normalize_signal(signal_units)
    if signal is None:
        return None
    return _SIGNAL_DOMAIN.get(signal)


def _domain_from_variance(variance_units: str | None) -> str | None:
    variance = _normalize_variance(variance_units)
    if variance is None or not variance.endswith("_squared"):
        return None
    return _domain_from_signal(variance[: -len("_squared")])


def _require_equal(context: str, label: str, actual: str | None, expected: str | None) -> None:
    if actual is None or expected is None or actual == expected:
        return
    raise UnitContractError(
        f"{context}: unit contract mismatch for {label}: "
        f"declared {actual!r}, expected {expected!r}"
    )


def assert_compatible(
    *,
    context: str,
    measurement_domain: str | None = None,
    signal_units: str | None = None,
    noise_variance_units: str | None = None,
    expected_measurement_domain: str | None = None,
    expected_signal_units: str | None = None,
    expected_noise_variance_units: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> None:
    """Assert declared domain/signal/variance units are mutually compatible."""

    if not unit_contracts_enabled(params):
        return

    declared_domain = _normalize_domain(measurement_domain)
    declared_signal = _normalize_signal(signal_units)
    declared_variance = _normalize_variance(noise_variance_units)
    expected_domain = _normalize_domain(expected_measurement_domain)
    expected_signal = _normalize_signal(expected_signal_units)
    expected_variance = _normalize_variance(expected_noise_variance_units)

    _require_equal(context, "measurement_domain", declared_domain, expected_domain)
    _require_equal(context, "signal_units", declared_signal, expected_signal)
    _require_equal(context, "noise_variance_units", declared_variance, expected_variance)

    signal_domain = _domain_from_signal(declared_signal)
    variance_domain = _domain_from_variance(declared_variance)
    _require_equal(context, "signal measurement domain", signal_domain, declared_domain)
    _require_equal(context, "variance measurement domain", variance_domain, declared_domain)
    _require_equal(context, "variance/signal domain", variance_domain, signal_domain)

    inferred_variance = variance_units_for_signal(declared_signal)
    _require_equal(context, "variance units", declared_variance, inferred_variance)
