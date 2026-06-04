"""Dataset JSON IO helpers shared by state and completeness checks."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

from json_utils import json_safe


logger = logging.getLogger(__name__)


def _load_json_file(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not load JSON file %s: %s", path, exc)
        return None


def _write_json_file(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(json_safe(payload), fh, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp_path, path)


__all__ = ["_load_json_file", "_write_json_file"]
