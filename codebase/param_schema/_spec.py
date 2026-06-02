"""Shared parameter-schema type declarations."""

from __future__ import annotations

from typing import Any, List, Literal, TypedDict


ParamType = Literal["float", "int", "bool", "enum", "string", "json"]


class ParamSpec(TypedDict, total=False):
    """Container for user-facing parameter metadata."""

    key: str
    container_key: str
    target_path: str
    type: ParamType
    default: Any
    min: float
    max: float
    choices: List[Any]
    ui_label: str
    group: str
    description: str
