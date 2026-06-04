"""Dataset output completeness checks."""

from __future__ import annotations
from config import param_value

import os
from typing import Any, Mapping

import numpy as np

from shared_constants import MATCHED_INFORMATION_MASK_ROLES

def _existing_nonempty_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def _raw_frame_views_complete(path: str, expected_num_frames: int) -> bool:
    if expected_num_frames <= 0 or not _existing_nonempty_file(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "background_subtracted_frames" not in data.files:
                return False
            frames = np.asarray(data["background_subtracted_frames"])
            if frames.ndim < 3 or int(frames.shape[0]) != int(expected_num_frames):
                return False
            if "trajectories_nm" in data.files:
                np.asarray(data["trajectories_nm"])
    except Exception:
        return False
    return True

def _avi_video_complete(path: str, expected_num_frames: int) -> bool:
    if expected_num_frames <= 0 or not _existing_nonempty_file(path):
        return False
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return False
        readable_count = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            readable_count += 1
            if readable_count > expected_num_frames:
                return False
        return readable_count == expected_num_frames
    finally:
        capture.release()

def _counterfactual_packet_complete(path: str) -> bool:
    if not _existing_nonempty_file(path):
        return False
    try:
        from counterfactual_packets import load_counterfactual_modality_packet

        packet = load_counterfactual_modality_packet(path)
        metadata = packet.get("metadata")
        if not isinstance(metadata, dict):
            return False
        if metadata.get("schema_version") != "syniscopy-matched-modality-packet-v1":
            return False
        if metadata.get("packet_kind") != "matched_modality_information_packet":
            return False
        modalities = metadata.get("modalities", [])
        if not isinstance(modalities, list) or len(modalities) < 2:
            return False
        if set(metadata.get("crlb_by_modality", {}).keys()) != set(modalities):
            return False
        if set(packet.get("images_by_modality", {}).keys()) != set(modalities):
            return False
        if set(packet.get("fisher_by_modality", {}).keys()) != set(modalities):
            return False
        shared_frame = (metadata.get("metadata") or {}).get("shared_coordinate_frame")
        if not isinstance(shared_frame, dict):
            return False
        if not bool(metadata.get("has_fisher_by_modality", False)):
            return False
    except Exception:
        return False
    return True

def _frame_sequence_complete(frames_path: str, expected_num_frames: int) -> bool:
    if expected_num_frames <= 0 or not os.path.isdir(frames_path):
        return False
    expected = [f"{idx:06d}.png" for idx in range(expected_num_frames)]
    actual = sorted(name for name in os.listdir(frames_path) if name.lower().endswith(".png"))
    if actual != expected:
        return False
    return all(_existing_nonempty_file(os.path.join(frames_path, name)) for name in expected)

def _mask_outputs_complete(mask_path: str, manifest: Mapping[str, Any], expected_num_frames: int) -> bool:
    if not os.path.isdir(mask_path):
        return False
    for filename in ("annotation_schema.json", "supervision_audit.json", "supervision_records.jsonl"):
        if not _existing_nonempty_file(os.path.join(mask_path, filename)):
            return False

    particles = manifest.get("particles", [])
    if not isinstance(particles, list) or not particles:
        return False

    records_path = os.path.join(mask_path, "supervision_records.jsonl")
    with open(records_path, "r", encoding="utf-8") as fh:
        record_count = sum(1 for line in fh if line.strip())
    if record_count != expected_num_frames * len(particles):
        return False

    schema = manifest.get("annotation_schema", {})
    target_names = tuple((schema.get("targets") or {}).keys()) if isinstance(schema, dict) else ()
    if not target_names:
        target_names = MATCHED_INFORMATION_MASK_ROLES
    for target_name in target_names:
        for particle_index in range(len(particles)):
            particle_dir = os.path.join(mask_path, str(target_name), f"particle_{particle_index + 1}")
            if not os.path.isdir(particle_dir):
                return False
            for frame_index in range(expected_num_frames):
                filename = f"frame_{frame_index:04d}.png"
                if not _existing_nonempty_file(os.path.join(particle_dir, filename)):
                    return False
    return True

def _video_assets_complete(base_output_dir: str, video_index: int) -> bool:
    from .json_io import _load_json_file
    from .runtime import _video_manifest_path

    manifest = _load_json_file(_video_manifest_path(base_output_dir, video_index))
    if not isinstance(manifest, dict):
        return False
    video_rel = manifest.get("output_video_path")
    frames_rel = manifest.get("frame_sequence_dir")
    mask_rel = manifest.get("mask_root_dir")
    if not video_rel or not frames_rel or not mask_rel:
        return False
    video_path = os.path.join(base_output_dir, str(video_rel))
    frames_path = os.path.join(base_output_dir, str(frames_rel))
    mask_path = os.path.join(base_output_dir, str(mask_rel))
    try:
        expected_num_frames = int(manifest.get("num_frames", 0))
    except (TypeError, ValueError):
        return False
    if not _avi_video_complete(video_path, expected_num_frames):
        return False
    if not _frame_sequence_complete(frames_path, expected_num_frames):
        return False
    if bool(manifest.get("mask_generation_enabled", True)) and not _mask_outputs_complete(
        mask_path,
        manifest,
        expected_num_frames,
    ):
        return False
    raw_views_rel = manifest.get("raw_frame_views_npz")
    if raw_views_rel and not _raw_frame_views_complete(
        os.path.join(base_output_dir, str(raw_views_rel)),
        expected_num_frames,
    ):
        return False
    for sidecar_rel in manifest.get("channel_sidecar_videos") or []:
        if not _avi_video_complete(
            os.path.join(base_output_dir, str(sidecar_rel)),
            expected_num_frames,
        ):
            return False
    packet_rel = manifest.get("matched_modality_packet_npz")
    matched_modalities = manifest.get("matched_modalities")
    if matched_modalities:
        if not packet_rel:
            return False
    if packet_rel:
        if not _counterfactual_packet_complete(os.path.join(base_output_dir, str(packet_rel))):
            return False
    return True

def _validate_dataset_output_contract(params: Mapping[str, Any]) -> None:
    """
    Dataset generation must produce a primary video file referenced by the
    manifest. Multichannel direct-render modes that skip the primary video are
    valid for low-level simulation calls but not for this dataset entry point.
    """
    channels = param_value(params, 'channels')
    if channels is not None and param_value(params, "matched_modalities") is not None:
        raise ValueError(
            "Dataset generation cannot combine PARAMS['channels'] with "
            "PARAMS['matched_modalities']; matched packets render their own "
            "modality set and reject spectral channels."
        )
    if not bool(param_value(params, 'save_frame_sequence')):
        raise ValueError(
            "Dataset generation requires save_frame_sequence=True. "
            "PNG frame sequences are the canonical 8-bit training/inference "
            "artifact; AVI is only a preview. Enable save_raw_frame_views for "
            "quantitative raw/ideal arrays."
        )
    volumetric_mode = str(param_value(params, 'volumetric_imaging_mode')).strip().lower()
    if volumetric_mode != "single_plane":
        raise ValueError(
            "Dataset generation requires volumetric_imaging_mode='single_plane'. "
            "Volumetric simulation outputs are analysis-volume dictionaries, "
            "not the public (T, C, H, W) video-frame result required by the "
            "dataset frame-sequence writer. Use generate_volumetric_views() "
            "for volumetric analysis outputs."
        )
    if channels:
        output_mode = str(param_value(params, 'multichannel_output_mode')).strip().lower()
        if output_mode not in {"rgb", "both"}:
            raise ValueError(
                "Dataset generation requires multichannel_output_mode='rgb' "
                "or 'both' when channels are enabled, because the dataset "
                "manifest needs a primary training video. Use the low-level "
                "run_simulation path for sidecar-only or no-video spectral renders."
            )
