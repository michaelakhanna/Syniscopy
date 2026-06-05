"""Per-video dataset generation runtime."""

from __future__ import annotations
from config import param_value

import logging
import os
import shutil
from copy import deepcopy
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from camera_noise import clear_detector_static_noise_cache
from common_utils import relative_path
from counterfactual_packets import save_counterfactual_modality_packet
from json_utils import json_safe
from metadata import build_dataset_index_entry, build_video_manifest, save_video_manifest
from simulation import render_matched_modality_observations, run_simulation
from substrate.patterns import clear_sample_environment_pattern_layout_cache

from .completeness import _validate_dataset_output_contract, _video_assets_complete
from .overrides import apply_parameter_overrides
from .runtime import (
    _final_frames_from_simulation_result,
    _raw_signal_frames_from_result_metadata,
    _raw_frame_view_payload,
    _remove_video_artifacts,
    _save_raw_camera_frame_sequence,
    _save_lossless_frame_sequence,
    _video_manifest_path,
    build_dataset_video_params,
)
from .seeds import _derive_video_seed
from .state import _load_json_file

logger = logging.getLogger(__name__)


def process_dataset_videos(
    *,
    existing_entries_by_index: Dict[int, Dict[str, Any]],
    target_indices: list[int],
    base_output_dir: str,
    source_provenance: Mapping[str, Any],
    composition: Optional[Sequence[Mapping[str, Any]]],
    assignment_by_index: Mapping[int, Mapping[str, Any]],
    leaf_specs: list[Dict[str, Any]],
    seed_by_index: Mapping[int, int],
    num_videos: int,
    dataset_source_name: str,
    instrument_preset: Optional[str],
    mode: str,
    resume_existing: bool,
    video_dir: str,
    frames_root_dir: str,
    raw_camera_frames_root_dir: str,
    masks_root_dir: str,
    raw_views_dir: str,
    packets_dir: str,
    video_param_builder: Optional[Callable[[int, np.random.Generator], Dict[str, Any]]],
    param_overrides: Optional[Mapping[str, Any]],
    random_seed: Optional[int],
    start_index: int,
    leaf_assignments: list[Dict[str, Any]],
) -> tuple[Dict[int, Dict[str, Any]], Dict[str, Any], int, int]:
    # Accumulate dataset-level manifest entries here.
    dataset_entries_by_index: Dict[int, Dict[str, Any]] = dict(existing_entries_by_index)
    representative_params: Dict[str, Any] | None = None

    logger.info(
        "Generating/resuming %s requested video(s) using dataset_source=%r, "
        "instrument_preset=%r, mode=%r...",
        num_videos,
        dataset_source_name,
        instrument_preset,
        mode,
    )

    generated_count = 0
    skipped_count = 0
    for batch_offset, video_index in enumerate(target_indices):
        clear_detector_static_noise_cache()
        clear_sample_environment_pattern_layout_cache()
        video_id = f"video_{video_index:04d}"
        logger.info("=== %s (%s / %s) ===", video_id, batch_offset + 1, len(target_indices))

        if resume_existing and _video_assets_complete(base_output_dir, video_index):
            logger.info("Skipping completed %s", video_id)
            manifest = _load_json_file(_video_manifest_path(base_output_dir, video_index))
            if isinstance(manifest, dict):
                recorded_source = manifest.get("source_provenance")
                recorded_fingerprint = (
                    recorded_source.get("fingerprint")
                    if isinstance(recorded_source, dict)
                    else manifest.get("source_provenance_fingerprint")
                )
                if recorded_fingerprint != source_provenance["fingerprint"]:
                    logger.info(
                        "Skipping %s avoided because existing video source fingerprint is stale or missing.",
                        video_id,
                    )
                    _remove_video_artifacts(base_output_dir, video_index)
                    existing_entries_by_index.pop(video_index, None)
                    dataset_entries_by_index.pop(video_index, None)
                    continue
            else:
                logger.info(
                    "Skipping %s avoided because existing video manifest is missing or invalid.",
                    video_id,
                )
                _remove_video_artifacts(base_output_dir, video_index)
                existing_entries_by_index.pop(video_index, None)
                dataset_entries_by_index.pop(video_index, None)
                continue
            if composition is not None:
                expected_signature = assignment_by_index.get(video_index, {}).get(
                    "leaf_signature"
                )
                actual_signature = manifest.get("composition_leaf_signature")
                if expected_signature is not None and actual_signature != expected_signature:
                    logger.info(
                        "Skipping %s avoided because existing video is from a different composition leaf.",
                        video_id,
                    )
                    _remove_video_artifacts(base_output_dir, video_index)
                    existing_entries_by_index.pop(video_index, None)
                    dataset_entries_by_index.pop(video_index, None)
                    continue
            entry = existing_entries_by_index.get(video_index)
            if entry is not None:
                dataset_entries_by_index[video_index] = entry
            skipped_count += 1
            continue

        video_seed = seed_by_index[int(video_index)]

        video_rng = np.random.default_rng(video_seed)

        assignment = assignment_by_index.get(int(video_index))
        active_leaf_spec = (
            assignment["leaf_spec"] if assignment is not None else leaf_specs[0]
        )

        if video_param_builder is not None:
            params = video_param_builder(video_index, video_rng)
            if not isinstance(params, dict):
                raise TypeError("video_param_builder must return a PARAMS dictionary.")
            params = apply_parameter_overrides(params, param_overrides)
        else:
            params = build_dataset_video_params(
                video_index=video_index,
                rng=video_rng,
                preset_name=active_leaf_spec["preset_name"],
                instrument_preset=active_leaf_spec.get("instrument_preset"),
                recipe_overrides=active_leaf_spec.get("recipe_overrides"),
                param_overrides=active_leaf_spec.get("param_overrides"),
            )
        # This is internal dataset state, not a public recipe override. Keep it
        # out of apply_parameter_overrides() so the override validator remains
        # strict while optics and other deterministic physics paths still see a
        # per-video seed.
        params["random_seed"] = int(video_seed)
        params["_substrate_pattern_layout_cache_token"] = f"{video_id}:{int(video_seed)}"
        _validate_dataset_output_contract(params)

        video_filename = os.path.join(video_dir, f"{video_id}.avi")
        frame_sequence_dir = os.path.join(frames_root_dir, video_id)
        raw_camera_frame_sequence_dir = os.path.join(raw_camera_frames_root_dir, video_id)
        masks_dir = os.path.join(masks_root_dir, video_id)
        raw_views_path = os.path.join(raw_views_dir, f"{video_id}.npz")
        channel_sidecar_dir = os.path.join(video_dir, "channels", video_id)
        matched_packet_path = os.path.join(packets_dir, f"{video_id}.npz")

        # If an interrupted attempt stopped mid-video, clear that one incomplete
        # video's owned outputs only. Completed videos are never touched by resume.
        if os.path.exists(video_filename):
            os.remove(video_filename)
        raw_signal_video_filename = os.path.splitext(video_filename)[0] + "_raw_signal.avi"
        if os.path.exists(raw_signal_video_filename):
            os.remove(raw_signal_video_filename)
        if os.path.exists(raw_views_path):
            os.remove(raw_views_path)
        if os.path.exists(matched_packet_path):
            os.remove(matched_packet_path)
        raw_views_tmp_path = raw_views_path + ".tmp"
        if os.path.exists(raw_views_tmp_path):
            os.remove(raw_views_tmp_path)
        if os.path.isdir(frame_sequence_dir):
            shutil.rmtree(frame_sequence_dir)
        if os.path.isdir(raw_camera_frame_sequence_dir):
            shutil.rmtree(raw_camera_frame_sequence_dir)
        if os.path.isdir(masks_dir):
            shutil.rmtree(masks_dir)
        if os.path.isdir(channel_sidecar_dir):
            shutil.rmtree(channel_sidecar_dir)
        os.makedirs(masks_dir, exist_ok=True)

        params["output_filename"] = video_filename
        params["mask_output_directory"] = masks_dir
        params["multichannel_sidecar_directory"] = channel_sidecar_dir
        if representative_params is None:
            representative_params = deepcopy(params)

        save_frame_sequence = bool(param_value(params, 'save_frame_sequence'))
        save_raw_frame_views = bool(param_value(params, 'save_raw_frame_views'))
        save_raw_camera_frame_sequence = bool(param_value(params, 'save_raw_camera_frame_sequence'))
        simulation_result = run_simulation(
            params,
            return_frames=bool(
                save_frame_sequence
                or save_raw_frame_views
                or save_raw_camera_frame_sequence
            ),
        )
        result_metadata = (
            dict(simulation_result.get("metadata", {}) or {})
            if isinstance(simulation_result, Mapping)
            else {}
        )

        frame_sequence_rel = None
        if save_frame_sequence:
            if simulation_result is None:
                raise RuntimeError("Frame sequence saving requires returned final frames, but simulation returned None.")
            final_frames_for_sequence = _final_frames_from_simulation_result(simulation_result)
            _save_lossless_frame_sequence(final_frames_for_sequence, frame_sequence_dir)
            frame_sequence_rel = os.path.join("frames", video_id)

        raw_camera_frame_sequence_rel = None
        if save_raw_camera_frame_sequence:
            if simulation_result is None:
                raise RuntimeError(
                    "Raw camera frame sequence saving requires returned raw frames, "
                    "but simulation returned None."
                )
            raw_signal_frames_for_sequence = _raw_signal_frames_from_result_metadata(result_metadata)
            _save_raw_camera_frame_sequence(
                raw_signal_frames_for_sequence,
                raw_camera_frame_sequence_dir,
                bit_depth=int(param_value(params, "bit_depth")),
            )
            raw_camera_frame_sequence_rel = os.path.join("raw_camera_frames", video_id)

        raw_views_rel = None
        if save_raw_frame_views and simulation_result is not None:
            raw_views_rel = os.path.join("raw_frame_views", f"{video_id}.npz")
            final_frames_for_raw_view = _final_frames_from_simulation_result(simulation_result)
            with open(raw_views_tmp_path, "wb") as fh:
                np.savez_compressed(
                    fh,
                    **_raw_frame_view_payload(result_metadata, final_frames_for_raw_view),
                )
            os.replace(raw_views_tmp_path, raw_views_path)

        leaf_name = (
            assignment.get("leaf_name")
            if composition is not None and assignment is not None
            else dataset_source_name
        )
        if composition is not None and assignment is not None:
            leaf_index = int(assignment.get("leaf_index"))
            leaf_local_index = int(assignment.get("leaf_local_index"))
            leaf_signature = assignment.get("leaf_signature")
            manifest = build_video_manifest(
                params=params,
                base_output_dir=base_output_dir,
                video_index=video_index,
                dataset_preset=str(leaf_name),
                instrument_preset=active_leaf_spec.get("instrument_preset"),
                video_seed=video_seed,
                result_metadata=result_metadata,
                composition_leaf_index=leaf_index,
                composition_leaf_name=str(leaf_name),
                composition_leaf_signature=leaf_signature,
                composition_local_index=leaf_local_index,
            )
        else:
            manifest = build_video_manifest(
                params=params,
                base_output_dir=base_output_dir,
                video_index=video_index,
                dataset_preset=leaf_name,
                instrument_preset=active_leaf_spec.get("instrument_preset"),
                video_seed=video_seed,
                result_metadata=result_metadata,
            )
        sidecars = result_metadata.get("channel_sidecar_videos", [])
        if sidecars:
            manifest["channel_sidecar_videos"] = [
                json_safe(relative_path(base_output_dir, str(path)))
                for path in sidecars
            ]
        if raw_views_rel is not None:
            manifest["raw_frame_views_npz"] = raw_views_rel
            manifest["background_subtracted_video_path"] = manifest.get("output_video_path")
        if raw_camera_frame_sequence_rel is not None:
            manifest["raw_camera_frame_sequence_dir"] = raw_camera_frame_sequence_rel
        matched_modalities = param_value(params, "matched_modalities")
        if matched_modalities is not None:
            packet_payload = render_matched_modality_observations(
                params,
                matched_modalities,
                frame_index=0,
            )
            saved_packet_path = save_counterfactual_modality_packet(
                matched_packet_path,
                latent_state=packet_payload["latent_state"],
                images_by_modality=packet_payload["images_by_modality"],
                rendered_signal_frame_by_modality=packet_payload.get("rendered_signal_frame_by_modality"),
                reference_frame_by_modality=packet_payload.get("reference_frame_by_modality"),
                noise_variance_by_modality=packet_payload.get("noise_variance_by_modality"),
                masks=packet_payload.get("masks"),
                fisher_by_modality=packet_payload.get("fisher_by_modality"),
                crlb_by_modality=packet_payload.get("crlb_by_modality"),
                metadata=packet_payload["metadata"],
                require_information_fields=True,
            )
            manifest["matched_modality_packet_npz"] = json_safe(
                relative_path(base_output_dir, saved_packet_path)
            )
            manifest["matched_modalities_requested"] = [str(name) for name in matched_modalities]
            manifest["matched_modalities"] = [
                str(name) for name in packet_payload.get("metadata", {}).get("modalities", [])
            ]
        if frame_sequence_rel is not None:
            manifest["frame_sequence_dir"] = frame_sequence_rel
            manifest["training_frames_dir"] = frame_sequence_rel
            manifest["preview_video_path"] = manifest.get("output_video_path")
        manifest_path = save_video_manifest(
            manifest=manifest,
            base_output_dir=base_output_dir,
            video_index=video_index,
        )
        logger.info("Saved per-video manifest to %s", manifest_path)

        dataset_entries_by_index[video_index] = build_dataset_index_entry(manifest)
        generated_count += 1
        clear_detector_static_noise_cache()
        clear_sample_environment_pattern_layout_cache()

    if representative_params is None:
        # No new videos were needed. Reconstruct a representative template so
        # the simulation manifest still reflects the active request.
        template_seed = seed_by_index.get(start_index, _derive_video_seed(random_seed, start_index))
        template_rng = np.random.default_rng(template_seed)
        template_leaf = None
        if composition is not None and leaf_assignments:
            template_leaf = assignment_by_index.get(start_index, {}).get("leaf_spec")
        if video_param_builder is not None:
            representative_params = apply_parameter_overrides(
                video_param_builder(start_index, template_rng),
                param_overrides,
            )
            representative_params["random_seed"] = int(template_seed)
            _validate_dataset_output_contract(representative_params)
        else:
            template_spec = template_leaf if template_leaf is not None else leaf_specs[0]
            representative_params = build_dataset_video_params(
                video_index=start_index,
                rng=template_rng,
                preset_name=template_spec["preset_name"],
                instrument_preset=template_spec.get("instrument_preset"),
                recipe_overrides=template_spec.get("recipe_overrides"),
                param_overrides=template_spec.get("param_overrides"),
            )
            representative_params["random_seed"] = int(template_seed)
            _validate_dataset_output_contract(representative_params)

    return dataset_entries_by_index, representative_params, generated_count, skipped_count
