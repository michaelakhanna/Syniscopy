"""Shared parameter-schema type declarations."""

from __future__ import annotations

from typing import Any, List, Literal, TypedDict


ParamType = Literal["float", "int", "bool", "enum", "string", "json"]


class ParamSpec(TypedDict, total=False):
    """Container for user-facing parameter metadata.

    Runtime defaults are owned by the same concept-schema fragment that owns
    the public parameter key. There is no global default-parameter dictionary.
    """

    key: str
    container_key: str
    target_path: str
    type: ParamType
    min: float
    max: float
    choices: List[Any]
    default: Any
    ui_label: str
    group: str
    description: str
