"""Dataset composition-plan and request-signature helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from json_utils import json_safe

from .overrides import (
    _coerce_optional_int,
    _coerce_positive_int,
    _reject_dataset_managed_overrides,
)
from .runtime import _normalize_dataset_preset_name

def _normalize_composition_entry(
    entry: Mapping[str, Any],
    *,
    default_preset_name: str,
    default_instrument_preset: Optional[str],
    default_random_seed: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise TypeError("composition entries must be mapping objects.")

    num_videos = _coerce_positive_int(entry.get("num_videos"), "composition entry 'num_videos'")
    preset_name = _normalize_dataset_preset_name(
        str(entry.get("preset_name", default_preset_name))
    )
    instrument_preset = entry.get("instrument_preset", default_instrument_preset)
    if instrument_preset is not None:
        instrument_preset = str(instrument_preset)

    recipe_overrides = entry.get("recipe_overrides")
    if recipe_overrides is not None and not isinstance(recipe_overrides, Mapping):
        raise TypeError("composition entry 'recipe_overrides' must be a mapping.")
    param_overrides = entry.get("param_overrides")
    if param_overrides is not None and not isinstance(param_overrides, Mapping):
        raise TypeError("composition entry 'param_overrides' must be a mapping.")

    random_seed = _coerce_optional_int(
        entry.get("random_seed", default_random_seed),
        "composition entry 'random_seed'",
    )

    normalized = {
        "num_videos": num_videos,
        "preset_name": preset_name,
        "instrument_preset": instrument_preset,
        "random_seed": None if random_seed is None else int(random_seed),
        "recipe_overrides": json_safe(recipe_overrides or {}),
        "param_overrides": json_safe(param_overrides or {}),
    }
    if "name" in entry and entry.get("name") is not None:
        normalized["name"] = str(entry.get("name"))
    _reject_dataset_managed_overrides(recipe_overrides, "composition entry recipe_overrides")
    _reject_dataset_managed_overrides(param_overrides, "composition entry param_overrides")
    return normalized

def _build_composition_plan(
    *,
    num_videos: int,
    preset_name: str,
    instrument_preset: Optional[str],
    random_seed: Optional[int],
    recipe_overrides: Optional[Mapping[str, Any]],
    param_overrides: Optional[Mapping[str, Any]],
    composition: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    if composition is None:
        base_entry = {
            "num_videos": int(num_videos),
            "preset_name": preset_name,
            "instrument_preset": instrument_preset,
            "random_seed": random_seed,
            "recipe_overrides": json_safe(recipe_overrides or {}),
            "param_overrides": json_safe(param_overrides or {}),
            "name": "dataset",
        }
        _reject_dataset_managed_overrides(recipe_overrides, "recipe_overrides")
        _reject_dataset_managed_overrides(param_overrides, "param_overrides")
        leaf_signature = _request_signature(_dataset_request_payload(
            num_videos=num_videos,
            preset_name=preset_name,
            instrument_preset=instrument_preset,
            random_seed=random_seed,
            recipe_overrides=recipe_overrides,
            param_overrides=param_overrides,
            param_builder_name=None,
            composition=None,
        ))
        assignment: list[Dict[str, Any]] = [
            {
                "video_index": idx,
                "leaf_index": 0,
                "leaf_name": "dataset",
                "leaf_local_index": idx,
                "leaf_spec": base_entry,
                "leaf_signature": leaf_signature,
            }
            for idx in range(num_videos)
        ]
        return [base_entry], assignment, [
            {
                "name": "dataset",
                "num_videos": int(num_videos),
                "preset_name": preset_name,
                "instrument_preset": instrument_preset,
                "random_seed": random_seed,
                "recipe_overrides": json_safe(recipe_overrides or {}),
                "param_overrides": json_safe(param_overrides or {}),
                "leaf_signature": leaf_signature,
            }
        ]

    normalized_children: list[Dict[str, Any]] = []
    for entry in composition:
        normalized_children.append(
            _normalize_composition_entry(
                entry,
                default_preset_name=preset_name,
                default_instrument_preset=instrument_preset,
                default_random_seed=random_seed,
            )
        )

    requested_total = sum(child["num_videos"] for child in normalized_children)
    if requested_total != num_videos:
        raise ValueError(
            "composition plan total must equal --num_videos; "
            f"composition sum is {requested_total} but num_videos is {num_videos}."
        )

    leaf_assignments: list[Dict[str, Any]] = []
    normalized_with_signatures: list[Dict[str, Any]] = []
    next_index = 0
    for leaf_index, child in enumerate(normalized_children):
        child_signature = _request_signature(
            _dataset_request_payload(
                num_videos=child["num_videos"],
                preset_name=child["preset_name"],
                instrument_preset=child["instrument_preset"],
                random_seed=child["random_seed"],
                recipe_overrides=child["recipe_overrides"],
                param_overrides=child["param_overrides"],
                composition=None,
                param_builder_name=None,
            )
        )
        child_name = child.get("name", f"dataset_{leaf_index:02d}")
        child_descriptor = {
            "name": child_name,
            "num_videos": int(child["num_videos"]),
            "preset_name": child["preset_name"],
            "instrument_preset": child["instrument_preset"],
            "random_seed": child["random_seed"],
            "recipe_overrides": json_safe(child["recipe_overrides"]),
            "param_overrides": json_safe(child["param_overrides"]),
            "leaf_signature": child_signature,
        }
        normalized_with_signatures.append(child_descriptor)
        for local_index in range(child["num_videos"]):
            leaf_assignments.append(
                {
                    "video_index": next_index,
                    "leaf_index": leaf_index,
                    "leaf_name": child_name,
                    "leaf_local_index": local_index,
                    "leaf_spec": child,
                    "leaf_signature": child_signature,
                }
            )
            next_index += 1

    return (
        normalized_children,
        leaf_assignments,
        [json_safe(child_descriptor) for child_descriptor in normalized_with_signatures],
    )

def _dataset_request_payload(
    *,
    num_videos: int,
    preset_name: str,
    instrument_preset: Optional[str],
    random_seed: Optional[int],
    recipe_overrides: Optional[Mapping[str, Any]],
    param_overrides: Optional[Mapping[str, Any]],
    param_builder_name: Optional[str] = None,
    composition: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "num_videos": int(num_videos),
        "preset_name": preset_name,
        "instrument_preset": instrument_preset,
        "random_seed": None if random_seed is None else int(random_seed),
        "recipe_overrides": json_safe(recipe_overrides or {}),
        "param_overrides": json_safe(param_overrides or {}),
        "param_builder_name": param_builder_name,
    }
    if composition is not None:
        payload["composition"] = json_safe(composition)
    return payload

def _dataset_request_payload_from_leaf(
    *,
    num_videos: int,
    leaf_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "num_videos": int(num_videos),
        "preset_name": str(leaf_spec.get("preset_name")),
        "instrument_preset": leaf_spec.get("instrument_preset"),
        "random_seed": None if leaf_spec.get("random_seed") is None else int(leaf_spec.get("random_seed")),
        "recipe_overrides": json_safe(leaf_spec.get("recipe_overrides") or {}),
        "param_overrides": json_safe(leaf_spec.get("param_overrides") or {}),
        "param_builder_name": None,
    }

def _request_signature(payload: Mapping[str, Any]) -> str:
    signature_payload = dict(payload)
    # Video count is a target size, not a physics/config identity. Increasing
    # it should extend the same dataset rather than look like a new condition.
    signature_payload.pop("num_videos", None)
    encoded = json.dumps(
        json_safe(signature_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _requests_identical(lhs: Optional[Mapping[str, Any]], rhs: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping):
        return False
    return json_safe(lhs) == json_safe(rhs)
