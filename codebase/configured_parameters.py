"""Central access boundary for public simulation parameter payloads.

Public parameter payloads are still mappings at API edges, but direct mapping
syntax must not leak through runtime, renderer, Fisher, or report code.  This
module is the only owner of that low-level access pattern; concept owners
consume configured values through these functions and expose typed fields.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping


def configured_value(configured_parameters: Mapping[str, Any], key: str) -> Any:
    """Return a required configured public parameter."""

    if key in configured_parameters:
        return configured_parameters[key]
    raise KeyError(f"Missing configured simulation parameter key: {key!r}")


def configured_optional(
    configured_parameters: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Return an optional configured public parameter."""

    return configured_parameters.get(key, default)


def configured_present(configured_parameters: Mapping[str, Any], key: str) -> bool:
    """Return whether the configured payload explicitly contains a key."""

    return key in configured_parameters


def configured_assign(
    configured_parameters: MutableMapping[str, Any],
    key: str,
    value: Any,
) -> None:
    """Assign a public parameter during explicit payload assembly."""

    configured_parameters[key] = value


__all__ = [
    "configured_assign",
    "configured_optional",
    "configured_present",
    "configured_value",
]
