"""Dataset state-file and resume bookkeeping."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping

from .json_io import _load_json_file, _write_json_file
from metadata import build_dataset_index_entry

logger = logging.getLogger(__name__)

_DATASET_STATE_FILENAME = "dataset_generation_state.json"

def _load_completed_dataset_entries(base_output_dir: str) -> Dict[int, Dict[str, Any]]:
    from .completeness import _video_assets_complete

    entries: Dict[int, Dict[str, Any]] = {}
    manifest_path = os.path.join(base_output_dir, "dataset_manifest.json")
    manifest = _load_json_file(manifest_path)
    if isinstance(manifest, dict):
        for entry in manifest.get("videos", []):
            if not isinstance(entry, dict) or "video_index" not in entry:
                continue
            try:
                index = int(entry["video_index"])
            except (TypeError, ValueError):
                logger.debug(
                    "Skipping dataset manifest entry with invalid video_index: %r",
                    entry.get("video_index"),
                )
                continue
            if _video_assets_complete(base_output_dir, index):
                entries[index] = entry

    metadata_dir = os.path.join(base_output_dir, "metadata")
    if os.path.isdir(metadata_dir):
        for filename in sorted(os.listdir(metadata_dir)):
            if not filename.startswith("video_") or not filename.endswith(".json"):
                continue
            try:
                index = int(filename[len("video_") : -len(".json")])
            except ValueError:
                continue
            if index in entries or not _video_assets_complete(base_output_dir, index):
                continue
            video_manifest = _load_json_file(os.path.join(metadata_dir, filename))
            if isinstance(video_manifest, dict):
                entries[index] = build_dataset_index_entry(video_manifest)
    return entries
