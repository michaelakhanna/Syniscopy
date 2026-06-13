"""Frame-rendering disk output helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from config import MaskGenerationSettings
import numpy as np

from json_utils import json_safe
from mask_generation import save_mask
from supervision_policy import build_policy_annotation_schema


def save_supervision_masks(
    masks: Mapping[str, np.ndarray],
    mask_root_dir: str,
    *,
    particle_index: int,
    frame_index: int,
) -> None:
    """Write one particle/frame supervision-mask bundle."""
    for schema_name, mask_arr in masks.items():
        save_mask(
            mask_arr,
            os.path.join(mask_root_dir, str(schema_name)),
            particle_index=particle_index,
            frame_index=frame_index,
        )


def write_supervision_sidecars(
    *,
    params: Mapping[str, Any],
    supervision_records: Sequence[Mapping[str, Any]],
    supervision_audit_summary: Mapping[str, Any],
) -> None:
    """Write supervision records, audit summary, and annotation schema sidecars."""
    mask_output_directory = MaskGenerationSettings.from_params(params).output_directory
    audit_path = os.path.join(mask_output_directory, "supervision_audit.json")
    records_path = os.path.join(mask_output_directory, "supervision_records.jsonl")
    schema_path = os.path.join(mask_output_directory, "annotation_schema.json")
    with open(records_path, "w", encoding="utf-8") as fh:
        for record in supervision_records:
            fh.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(
            json_safe(dict(supervision_audit_summary)),
            fh,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(
            build_policy_annotation_schema(dict(params)),
            fh,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


__all__ = ["save_supervision_masks", "write_supervision_sidecars"]
