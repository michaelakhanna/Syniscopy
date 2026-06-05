"""Dataset generation coordinator."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from metadata import (
    build_simulation_manifest,
    build_source_provenance,
    save_dataset_manifest,
    save_simulation_manifest,
)

from .composition import (
    _build_composition_plan,
    _dataset_request_payload,
    _request_signature,
)
from .completeness import _video_assets_complete
from .overrides import _reject_dataset_managed_overrides
from .runtime import (
    _clear_dataset_output_directory,
    _remove_video_artifacts,
    _resolve_base_output_dir,
    _normalize_dataset_preset_name,
)
from .seeds import _derive_video_seed
from .video import process_dataset_videos
from .state import (
    _DATASET_STATE_FILENAME,
    _load_completed_dataset_entries,
    _load_json_file,
    _write_json_file,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    logger.setLevel(level)


def _state_target_indices(state: object) -> list[int]:
    if not isinstance(state, dict):
        return []
    values = state.get("target_indices")
    if not isinstance(values, list):
        return []
    indices: list[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return []
        if index < 0:
            return []
        indices.append(index)
    return indices


def _remap_leaf_assignments_to_target_indices(
    assignment_by_index: dict[int, dict[str, Any]],
    leaf_assignments: Sequence[Mapping[str, Any]],
    target_indices: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    remapped_assignment_by_index: dict[int, dict[str, Any]] = {}
    remapped_leaf_assignments: list[dict[str, Any]] = []
    for request_index, target_index in enumerate(target_indices):
        assignment = dict(assignment_by_index.get(request_index, {}))
        if not assignment:
            continue
        assignment["video_index"] = int(target_index)
        remapped_assignment_by_index[int(target_index)] = assignment
        remapped_leaf_assignments.append(assignment)
    if remapped_assignment_by_index:
        return remapped_assignment_by_index, remapped_leaf_assignments
    return assignment_by_index, [dict(item) for item in leaf_assignments]


def generate_dataset(
    num_videos: int,
    preset_name: Optional[str] = "default",
    instrument_preset: Optional[str] = None,
    base_output_dir: Optional[str] = None,
    random_seed: Optional[int] = None,
    recipe_overrides: Optional[Mapping[str, Any]] = None,
    param_overrides: Optional[Mapping[str, Any]] = None,
    resume_existing: bool = True,
    reset_existing: bool = False,
    append_on_config_change: bool = False,
    video_param_builder: Optional[Callable[[int, np.random.Generator], Dict[str, Any]]] = None,
    param_builder_name: Optional[str] = None,
    verbose: bool = False,
    composition: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Coordinate dataset generation, resume state, and final manifests."""
    if verbose:
        _configure_logging(verbose=True)
    if num_videos <= 0:
        raise ValueError("num_videos must be a positive integer.")
    effective_preset_name = _normalize_dataset_preset_name(preset_name)
    builder_label = (
        param_builder_name
        or getattr(video_param_builder, "__name__", None)
        or None
    )
    if video_param_builder is not None and instrument_preset is not None:
        raise ValueError(
            "instrument_preset cannot be combined with video_param_builder. "
            "Apply instrument settings inside the builder or use the default "
            "dataset preset path."
        )
    if video_param_builder is not None and composition is not None:
        raise ValueError(
            "composition cannot be combined with video_param_builder. "
            "Set composition at the recipe/override level or keep one of them."
        )
    _reject_dataset_managed_overrides(recipe_overrides, "recipe_overrides")
    _reject_dataset_managed_overrides(param_overrides, "param_overrides")
    dataset_source_name = builder_label or effective_preset_name
    leaf_specs, leaf_assignments, composition_summary = _build_composition_plan(
        num_videos=num_videos,
        preset_name=effective_preset_name,
        instrument_preset=instrument_preset,
        random_seed=random_seed,
        recipe_overrides=recipe_overrides,
        param_overrides=param_overrides,
        composition=composition,
    )
    composition_plan = composition_summary if composition is not None else None
    assignment_by_index = {
        int(item["video_index"]): item
        for item in leaf_assignments
    }

    base_output_dir = _resolve_base_output_dir(base_output_dir)
    source_provenance = build_source_provenance()

    request_payload = _dataset_request_payload(
        num_videos=num_videos,
        preset_name=effective_preset_name,
        instrument_preset=instrument_preset,
        random_seed=random_seed,
        recipe_overrides=recipe_overrides,
        param_overrides=param_overrides,
        param_builder_name=builder_label,
        composition=composition_plan,
    )
    request_payload["source_provenance_fingerprint"] = source_provenance["fingerprint"]
    request_signature = _request_signature(request_payload)
    state_path = os.path.join(base_output_dir, _DATASET_STATE_FILENAME)
    prior_state = _load_json_file(state_path)
    prior_request = prior_state.get("request") if isinstance(prior_state, dict) else None
    prior_signature = (
        prior_state.get("request_signature")
        if isinstance(prior_state, dict)
        else None
    )
    same_signature = bool(prior_signature == request_signature)
    prior_mode = str(prior_state.get("mode", "")) if isinstance(prior_state, dict) else ""
    prior_target_indices = _state_target_indices(prior_state)
    existing_entries_by_index = _load_completed_dataset_entries(base_output_dir)
    existing_indices = sorted(existing_entries_by_index)
    append_requested = bool(
        append_on_config_change
        and existing_indices
        and prior_signature is not None
        and not same_signature
    )

    rewrite_requested = bool(
        reset_existing
        or (existing_indices and prior_signature is None)
        or (
            prior_signature is not None
            and not same_signature
            and not append_requested
        )
    )
    if rewrite_requested:
        if reset_existing:
            logger.info("Reset requested; removing existing dataset directory: %s", base_output_dir)
        elif prior_signature is None:
            logger.info(
                "Existing dataset has no request signature metadata; rewriting dataset "
                "to avoid config-ambiguity."
            )
        else:
            logger.info(
                "Existing dataset uses a different generation request; rewriting dataset "
                "instead of appending."
            )
        _clear_dataset_output_directory(base_output_dir)
        existing_entries_by_index = {}
        existing_indices = []

    # Subdirectories for videos and masks.
    video_dir = os.path.join(base_output_dir, "videos")
    frames_root_dir = os.path.join(base_output_dir, "frames")
    raw_camera_frames_root_dir = os.path.join(base_output_dir, "raw_camera_frames")
    masks_root_dir = os.path.join(base_output_dir, "masks")
    raw_views_dir = os.path.join(base_output_dir, "raw_frame_views")
    packets_dir = os.path.join(base_output_dir, "counterfactual_packets")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_root_dir, exist_ok=True)
    os.makedirs(raw_camera_frames_root_dir, exist_ok=True)
    os.makedirs(masks_root_dir, exist_ok=True)
    os.makedirs(raw_views_dir, exist_ok=True)
    os.makedirs(packets_dir, exist_ok=True)

    if append_requested:
        append_start_index = max(existing_indices) + 1
        request_indices = list(range(num_videos))
        target_indices = [
            append_start_index + request_index
            for request_index in request_indices
        ]
        mode = "append"
        start_index = target_indices[0]
        assignment_by_index, leaf_assignments = _remap_leaf_assignments_to_target_indices(
            assignment_by_index,
            leaf_assignments,
            target_indices,
        )
        logger.info(
            "Existing dataset uses a different generation request; appending "
            "current request as video indices %s.",
            target_indices,
        )
    elif existing_indices and same_signature:
        requested_indices = list(range(0, num_videos))
        if prior_mode == "append" and prior_target_indices:
            if len(prior_target_indices) == num_videos:
                target_indices = prior_target_indices
            else:
                append_start_index = min(prior_target_indices)
                target_indices = [
                    append_start_index + request_index
                    for request_index in requested_indices
                ]
            obsolete_indices = []
            mode = "append"
            assignment_by_index, leaf_assignments = _remap_leaf_assignments_to_target_indices(
                assignment_by_index,
                leaf_assignments,
                target_indices,
            )
        else:
            target_indices = requested_indices
            obsolete_indices = [idx for idx in existing_indices if idx >= num_videos]
            mode = "resume"
        for obsolete_index in obsolete_indices:
            _remove_video_artifacts(base_output_dir, obsolete_index)
            existing_entries_by_index.pop(obsolete_index, None)
        if obsolete_indices:
            existing_indices = sorted(existing_entries_by_index)
        start_index = target_indices[0] if target_indices else 0
    else:
        start_index = 0
        target_indices = list(range(num_videos))
        mode = "rewrite" if rewrite_requested else "resume"

    _write_json_file(
        state_path,
        {
            "schema_version": "syniscopy-dataset-generation-state-v1",
            "request": request_payload,
            "request_signature": request_signature,
            "mode": mode,
            "target_indices": target_indices,
            "completed_indices_at_start": existing_indices,
        },
    )

    seed_by_index = {
        int(video_index): _derive_video_seed(random_seed, int(video_index))
        for video_index in target_indices
    }
    if composition is not None:
        seed_by_index = {}
        for item in leaf_assignments:
            idx = int(item["video_index"])
            leaf_seed = item["leaf_spec"].get("random_seed")
            if leaf_seed is None:
                leaf_seed = random_seed
            seed_by_index[idx] = _derive_video_seed(leaf_seed, idx)

    dataset_entries_by_index, representative_params, generated_count, skipped_count = process_dataset_videos(
        existing_entries_by_index=existing_entries_by_index,
        target_indices=target_indices,
        base_output_dir=base_output_dir,
        source_provenance=source_provenance,
        composition=composition,
        assignment_by_index=assignment_by_index,
        leaf_specs=leaf_specs,
        seed_by_index=seed_by_index,
        num_videos=num_videos,
        dataset_source_name=dataset_source_name,
        instrument_preset=instrument_preset,
        mode=mode,
        resume_existing=resume_existing,
        video_dir=video_dir,
        frames_root_dir=frames_root_dir,
        raw_camera_frames_root_dir=raw_camera_frames_root_dir,
        masks_root_dir=masks_root_dir,
        raw_views_dir=raw_views_dir,
        packets_dir=packets_dir,
        video_param_builder=video_param_builder,
        param_overrides=param_overrides,
        random_seed=random_seed,
        start_index=start_index,
        leaf_assignments=leaf_assignments,
    )

    dataset_entries = [
        dataset_entries_by_index[index]
        for index in sorted(dataset_entries_by_index)
        if _video_assets_complete(base_output_dir, index)
    ]

    logger.info(
        "Dataset resume summary: generated=%s, skipped=%s, complete_total=%s",
        generated_count,
        skipped_count,
        len(dataset_entries),
    )

    # After all videos are generated, write the dataset-level manifest.
    dataset_manifest_path = save_dataset_manifest(
        base_output_dir=base_output_dir,
        dataset_entries=dataset_entries,
        source_provenance=source_provenance,
    )
    logger.info("Dataset-level manifest written to %s", dataset_manifest_path)
    simulation_manifest = build_simulation_manifest(
        base_output_dir=base_output_dir,
        dataset_entries=dataset_entries,
        params_template=representative_params,
        random_seed=random_seed,
        dataset_preset=dataset_source_name,
    )
    simulation_manifest["params_template_scope"] = (
        "current_request_only" if mode == "append" else "dataset"
    )
    simulation_manifest["heterogeneous_dataset"] = bool(composition is not None)
    simulation_manifest["current_request"] = request_payload
    simulation_manifest["current_request_signature"] = request_signature
    simulation_manifest["target_indices"] = target_indices
    if composition is not None:
        simulation_manifest["composition"] = composition_plan
    simulation_manifest["completed_indices_at_start"] = existing_indices
    simulation_manifest_path = save_simulation_manifest(
        manifest=simulation_manifest,
        base_output_dir=base_output_dir,
    )
    logger.info("Simulation manifest written to %s", simulation_manifest_path)
    _write_json_file(
        state_path,
        {
            "schema_version": "syniscopy-dataset-generation-state-v1",
            "request": request_payload,
            "request_signature": request_signature,
            "source_provenance": source_provenance,
            "mode": mode,
            "target_indices": target_indices,
            "completed_indices": [entry["video_index"] for entry in dataset_entries],
            "complete_total": len(dataset_entries),
        },
    )
    logger.info("Dataset generation complete.")
    return base_output_dir
