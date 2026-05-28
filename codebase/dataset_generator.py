"""
High-level dataset generation wrapper for Syniscopy.

This module sits on top of the core simulation pipeline and uses the preset
system plus the run_simulation entry point to generate many videos in a single
run.

Responsibilities:
    - Select the low-level dataset-generation parameter base for each video.
    - Optionally accept a caller-supplied parameter builder for custom
      per-video sampling.
    - Ensure that each video and its corresponding masks are written to
      unique, organized output locations.
    - Invoke run_simulation(params) once per video.
    - Construct and save per-video and dataset-level metadata manifests
      describing the generated samples in a machine-readable format.

Randomness & reproducibility:
    - A dataset-level seed (random_seed) and the absolute video index determine
      a per-video seed.
    - That per-video seed is used to:
        * Seed the canonical global np.random RNG used by simulation paths that
          draw from module-level NumPy randomness.
        * Construct a per-video Generator for caller-supplied parameter
          builders.
        * Populate params["random_seed"] after public override validation, so
          deterministic physics paths that derive their own Generator from the
          parameter dictionary also vary by video.

    As a result, providing the same random_seed and the same dataset
    configuration (num_videos, presets, etc.) makes the entire dataset
    generation process fully reproducible.

This module orchestrates multiple simulator runs and writes dataset-level
metadata/manifest files.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Mapping

import cv2
import numpy as np

from config import PARAMS, validate_params
from main import render_matched_modality_observations, run_simulation
from counterfactual_packets import save_counterfactual_modality_packet
from presets import apply_instrument_preset
from particle_specs import normalize_particle_specs
from metadata import (
    build_video_manifest,
    save_video_manifest,
    build_dataset_index_entry,
    save_dataset_manifest,
    build_simulation_manifest,
    save_simulation_manifest,
)


logger = logging.getLogger(__name__)

_PARTICLE_OVERRIDE_KEYS = {"particles"}
_DATASET_MANAGED_OVERRIDE_KEYS = {
    "mask_output_directory",
    "multichannel_sidecar_directory",
    "output_filename",
    "random_seed",
}

_DATASET_STATE_FILENAME = "dataset_generation_state.json"
_NUM_FRAME_DURATION_SEARCH_STEPS = 32

PUBLIC_DATASET_PRESET_DESCRIPTIONS: Dict[str, str] = {
    "default": (
        "Core config.PARAMS surface for programmatic dataset generation."
    ),
}


def configure_logging(verbose: bool = False) -> None:
    """Configure dataset-generation logging for command-line entry points."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    logger.setLevel(level)


def _normalize_num_frames_override(params, override_keys):
    """
    Make dataset-level num_frames overrides honor the renderer timebase.

    The renderer computes actual frame count as int(fps * duration_seconds).
    Explicit num_frames overrides therefore update duration_seconds so the
    renderer produces the requested count exactly.
    """
    import numpy as _np

    if "num_frames" not in override_keys:
        return params

    raw_num_frames = params.get("num_frames")
    if isinstance(raw_num_frames, bool):
        raise ValueError("param_overrides['num_frames'] must be a positive integer, not bool.")

    if isinstance(raw_num_frames, (float, _np.floating)) and not float(raw_num_frames).is_integer():
        raise ValueError("param_overrides['num_frames'] must be an integer frame count.")

    try:
        requested_num_frames = int(raw_num_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("param_overrides['num_frames'] must be a positive integer.") from exc

    if requested_num_frames <= 0:
        raise ValueError("param_overrides['num_frames'] must be positive.")

    fps = float(params.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError("PARAMS['fps'] must be positive when num_frames is overridden.")

    if "duration_seconds" in override_keys:
        supplied_duration = float(params["duration_seconds"])
        implied_num_frames = int(fps * supplied_duration)
        if implied_num_frames != requested_num_frames:
            raise ValueError(
                "Conflicting dataset timing overrides: "
                f"num_frames={requested_num_frames} but "
                f"duration_seconds={supplied_duration} and fps={fps} imply "
                f"{implied_num_frames} frame(s) under the renderer's "
                "int(fps * duration_seconds) rule."
            )
        params.pop("num_frames", None)
        return params

    duration_seconds = requested_num_frames / fps
    for _ in range(_NUM_FRAME_DURATION_SEARCH_STEPS):
        if int(fps * duration_seconds) == requested_num_frames:
            params["duration_seconds"] = float(duration_seconds)
            params.pop("num_frames", None)
            return params
        duration_seconds = float(_np.nextafter(duration_seconds, _np.inf))

    raise RuntimeError(
        "Could not choose duration_seconds that reproduces "
        f"num_frames={requested_num_frames} at fps={fps}."
    )


def _reject_dataset_managed_overrides(
    overrides: Optional[Mapping[str, Any]],
    override_name: str,
) -> None:
    if not overrides:
        return
    managed = sorted(
        str(key)
        for key in overrides
        if str(key) in _DATASET_MANAGED_OVERRIDE_KEYS
    )
    if managed:
        raise ValueError(
            f"{override_name} contains dataset-managed key(s) {managed}. "
            "Pass dataset output and seed settings to generate_dataset() instead."
        )


def apply_parameter_overrides(
    params: Dict[str, Any],
    param_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return a copy of ``params`` with explicit user overrides applied.

    This is the supported programmatic entry point for notebooks and other UIs that
    need to expose dataset-generation knobs without editing ``config.py`` or
    module globals. Values are copied into the returned
    dictionary; the input dictionary is not modified.

    Particle identity, material, geometry, and motion are supplied through the
    canonical ``particles`` object. Other particle-list formats are not
    accepted here.
    """
    out = deepcopy(params)
    if not param_overrides:
        if out.get("num_frames") is not None:
            _normalize_num_frames_override(out, {"num_frames"})
        validate_params(out, allowed_internal_keys=set(out))
        normalize_particle_specs(out, mutate=True)
        return out

    normalized_overrides: Dict[str, Any] = {}
    for raw_key, value in param_overrides.items():
        canonical_key = str(raw_key)
        allowed_extra_keys = {"num_frames", *_PARTICLE_OVERRIDE_KEYS}
        if canonical_key not in PARAMS and canonical_key not in allowed_extra_keys:
            raise ValueError(
                f"Unknown parameter override {canonical_key!r}. Use only keys from "
                "config.PARAMS plus the canonical particles object."
            )
        if canonical_key in _DATASET_MANAGED_OVERRIDE_KEYS:
            raise ValueError(
                f"Parameter override {canonical_key!r} is managed by dataset generation. "
                "Pass dataset output and seed settings to generate_dataset() instead."
            )
        if canonical_key in normalized_overrides:
            raise ValueError(
                f"Duplicate parameter override {canonical_key!r}; each "
                "override object key may be supplied only once."
            )
        normalized_overrides[canonical_key] = value

    override_keys = set(normalized_overrides)
    if out.get("num_frames") is not None:
        override_keys.add("num_frames")
    for key, value in normalized_overrides.items():
        out[key] = deepcopy(value)

    _normalize_num_frames_override(out, override_keys)
    validate_params(
        out,
        allowed_extra_keys=_PARTICLE_OVERRIDE_KEYS,
        allowed_internal_keys=set(out),
    )
    normalize_particle_specs(out, mutate=True)
    return out


def _resolve_base_output_dir(base_output_dir: Optional[str]) -> str:
    """
    Resolve the base output directory for the dataset.

    If base_output_dir is None, a project-relative default path is used:
        outputs/syniscopy_dataset

    The directory is created if it does not exist.

    Args:
        base_output_dir (Optional[str]): User-specified base directory or None.

    Returns:
        str: Absolute path to the base output directory.
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(
            "outputs",
            "syniscopy_dataset",
        )

    base_output_dir = os.path.abspath(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def _relative_path(base_dir: str, path: str) -> str:
    base_dir_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    try:
        return os.path.relpath(path_abs, base_dir_abs)
    except ValueError:
        return path_abs


def get_dataset_preset_names() -> tuple[str, ...]:
    """
    Return low-level dataset-generation parameter bases.

    User-facing microscope configurations live as recipe files outside the core
    package. This function exists for programmatic callers that intentionally
    want the complete renderer parameter dictionary.
    """
    return tuple(PUBLIC_DATASET_PRESET_DESCRIPTIONS.keys())


def get_default_dataset_params() -> Dict[str, Any]:
    """
    Return the complete default simulation parameter dictionary.

    This is intentionally a copy of ``config.PARAMS`` rather than a second
    hand-maintained dictionary. Users can inspect or dump this structure to see
    every configurable parameter that the core generator accepts.
    """
    return deepcopy(PARAMS)


def _normalize_dataset_preset_name(preset_name: Optional[str]) -> str:
    if preset_name is None:
        return "default"
    normalized = str(preset_name).strip().lower()
    if normalized == "":
        return "default"
    return normalized


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
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp_path, path)


def _dataset_request_payload(
    *,
    num_videos: int,
    preset_name: str,
    instrument_preset: Optional[str],
    random_seed: Optional[int],
    recipe_overrides: Optional[Mapping[str, Any]],
    param_overrides: Optional[Mapping[str, Any]],
    param_builder_name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "num_videos": int(num_videos),
        "preset_name": preset_name,
        "instrument_preset": instrument_preset,
        "random_seed": None if random_seed is None else int(random_seed),
        "recipe_overrides": _json_safe(recipe_overrides or {}),
        "param_overrides": _json_safe(param_overrides or {}),
        "param_builder_name": param_builder_name,
    }


def _request_signature(payload: Mapping[str, Any]) -> str:
    signature_payload = dict(payload)
    # Video count is a target size, not a physics/config identity. Increasing
    # it should extend the same dataset rather than look like a new condition.
    signature_payload.pop("num_videos", None)
    encoded = json.dumps(
        _json_safe(signature_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive_video_seed(random_seed: Optional[int], video_index: int) -> int:
    """
    Derive a stable NumPy-compatible seed from the dataset seed and video index.

    The seed must not depend on resume batch size or iteration offset. Otherwise
    re-running an interrupted dataset with a smaller ``num_videos`` can reuse a
    seed for multiple target indices.
    """
    if random_seed is None:
        entropy = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        payload = f"unseeded:{entropy}:{int(video_index)}"
    else:
        payload = f"seeded:{int(random_seed)}:{int(video_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def _video_manifest_path(base_output_dir: str, video_index: int) -> str:
    return os.path.join(base_output_dir, "metadata", f"video_{video_index:04d}.json")


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
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                return False
            metadata = json.loads(str(data["metadata_json"].item()))
            modalities = metadata.get("modalities", [])
            if not isinstance(modalities, list) or len(modalities) < 2:
                return False
            image_keys = [key for key in data.files if key.startswith("image__")]
            if len(image_keys) != len(modalities):
                return False
            shared = (metadata.get("metadata") or {}).get("shared_coordinate_frame")
            if not isinstance(shared, dict):
                return False
            fisher_keys = [key for key in data.files if key.startswith("fisher__")]
            if metadata.get("has_fisher_by_modality") is not True:
                return False
            if len(fisher_keys) != len(modalities):
                return False
            crlb = metadata.get("crlb_by_modality")
            if not isinstance(crlb, dict) or set(crlb) != set(modalities):
                return False
            mask_keys = [key for key in data.files if key.startswith("mask__")]
            if not mask_keys:
                return False
            mask_names = metadata.get("masks")
            if not isinstance(mask_names, list):
                return False
            for modality in modalities:
                prefix = f"{modality}__"
                if not any(str(name) == str(modality) or str(name).startswith(prefix) for name in mask_names):
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
        target_names = ("mask_geometry", "mask_supported", "ignore_mask", "loss_weight")
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
    channels = params.get("channels", None)
    if not bool(params.get("save_frame_sequence", True)):
        raise ValueError(
            "Dataset generation requires save_frame_sequence=True. "
            "Lossless PNG frame sequences are the canonical training/inference "
            "artifact; AVI is only a preview."
        )
    if channels:
        output_mode = str(params.get("multichannel_output_mode", "rgb")).strip().lower()
        if output_mode not in {"rgb", "both"}:
            raise ValueError(
                "Dataset generation requires multichannel_output_mode='rgb' "
                "or 'both' when channels are enabled, because the dataset "
                "manifest needs a primary training video. Use the low-level "
                "run_simulation path for sidecar-only or no-video spectral renders."
            )


def _final_frames_from_simulation_result(simulation_result: Mapping[str, Any]) -> np.ndarray:
    if "frames" not in simulation_result:
        raise ValueError("run_simulation(return_frames=True) must return a 'frames' array.")
    frames = np.asarray(simulation_result["frames"])
    if frames.ndim != 4:
        raise ValueError(
            "run_simulation returned frames with an invalid shape; expected "
            f"(T, C, H, W), got {frames.shape}."
        )
    if frames.shape[1] == 1:
        return frames[:, 0, :, :]
    if frames.shape[1] == 3:
        return np.moveaxis(frames, 1, -1)
    raise ValueError(
        "Lossless PNG frame sequences support one-channel grayscale or RGB "
        f"frames; got C={frames.shape[1]}."
    )


def _save_lossless_frame_sequence(frames: np.ndarray, frame_dir: str) -> int:
    """
    Write background-subtracted final frames as a lossless PNG sequence.

    These PNGs are the canonical data artifact for training/inference. The AVI
    video is only a compact preview, because inter-frame codecs can smear noisy
    microscopy frames after the first keyframe.
    """
    import cv2

    frames = np.asarray(frames)
    if frames.ndim < 3 or frames.shape[0] == 0:
        raise ValueError("Cannot save a frame sequence from an empty final-frame array.")

    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(float, copy=False)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Cannot save non-finite frame sequence data; "
                    f"frame {frame_index} contains NaN or Inf."
                )
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.ascontiguousarray(arr)

        if arr.ndim == 2:
            to_write = arr
        elif arr.ndim == 3 and arr.shape[2] == 3:
            to_write = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                "Frame sequence frames must be grayscale or RGB uint8 arrays; "
                f"got {arr.shape}."
            )

        out_path = os.path.join(frame_dir, f"{frame_index:06d}.png")
        ok = cv2.imwrite(out_path, to_write)
        if not ok:
            raise RuntimeError(f"Failed to write lossless frame {out_path!r}.")

    return int(frames.shape[0])


def _load_completed_dataset_entries(base_output_dir: str) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    manifest_path = os.path.join(base_output_dir, "dataset_manifest.json")
    manifest = _load_json_file(manifest_path)
    if isinstance(manifest, dict):
        for entry in manifest.get("videos", []):
            if not isinstance(entry, dict) or "video_index" not in entry:
                continue
            index = int(entry["video_index"])
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


def _build_params_for_video(preset_name: Optional[str]) -> Dict[str, Any]:
    """
    Construct a parameter dictionary for a single video.
    """
    normalized_preset = _normalize_dataset_preset_name(preset_name)

    if normalized_preset == "default":
        params = deepcopy(PARAMS)
    else:
        raise ValueError(
            f"Unknown public dataset preset {normalized_preset!r}. "
            "Use preset_name='default' plus param_overrides, or pass a "
            "caller-local video_param_builder."
        )

    return params


def build_dataset_video_params(
    video_index: int,
    rng: np.random.Generator,
    preset_name: Optional[str] = "default",
    instrument_preset: Optional[str] = None,
    recipe_overrides: Optional[Mapping[str, Any]] = None,
    param_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the full PARAMS dictionary for one dataset video.

    This is the public counterpart to the generator's per-video construction
    step. Notebooks can use it to preview the same preset and override
    parameter set that full dataset generation will use. The default builder is
    deterministic; ``video_index`` and ``rng`` are accepted so callers can swap
    in a custom ``video_param_builder`` with the same signature.
    """
    del video_index, rng
    params = _build_params_for_video(
        preset_name=preset_name,
    )
    params = apply_parameter_overrides(params, recipe_overrides)
    if instrument_preset is not None:
        # Instrument presets are explicit CLI/API choices. Apply them after the
        # default recipe so microscope optics override recipe-level optical
        # defaults.
        params = apply_instrument_preset(params, instrument_preset)
    params = apply_parameter_overrides(params, param_overrides)
    return params


def _raw_frame_view_payload(
    result_metadata: Mapping[str, Any],
    final_frames_for_raw_view,
) -> Dict[str, np.ndarray]:
    """Build the compressed NPZ payload for raw-frame audit views."""
    payload: Dict[str, np.ndarray] = {
        "background_subtracted_frames": np.asarray(final_frames_for_raw_view),
        "trajectories_nm": np.asarray(result_metadata.get("trajectories_nm", [])),
    }
    for key in (
        "raw_signal_frames",
        "raw_reference_frames",
        "raw_signal_frames_rgb",
        "raw_reference_frames_rgb",
        "raw_signal_frames_by_spectral_sample",
        "raw_reference_frames_by_spectral_sample",
        "background_subtracted_frames_rgb",
    ):
        if key in result_metadata:
            payload[key] = np.asarray(result_metadata[key])
    if "source_map_provenance" in result_metadata:
        payload["source_map_provenance_json"] = np.array(
            json.dumps(
                _json_safe(result_metadata["source_map_provenance"]),
                sort_keys=True,
                allow_nan=False,
            )
        )
    return payload


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
) -> str:
    """
    Generate a dataset of simulated microscopy videos and corresponding masks.

    This function is the main programmatic entry point for dataset generation.
    It repeatedly constructs a fresh parameter dictionary for each video, sets
    unique output paths, and calls run_simulation(params). For each video it
    also constructs and saves a per-video manifest JSON file and accumulates
    a minimal dataset-level index entry.

    Preset selection:
        - ``preset_name="default"`` uses a deepcopy of config.PARAMS so the
          core generator exposes its full parameter set.
        - Custom workflows can provide a ``video_param_builder`` from their own
          script instead of adding case names to the preset list.
        - ``instrument_preset`` applies a named microscope setup to the default
          recipe; per-video variation should use ``video_param_builder``.

    Output layout:
        base_output_dir/
            videos/
                video_0000.avi
                video_0001.avi
                ...
            frames/
                video_0000/
                    000000.png
                    000001.png
                    ...
            masks/
                video_0000/
                    mask_supported/
                    mask_geometry/
                    ignore_mask/
                    loss_weight/
                    supervision_records.jsonl
                    supervision_audit.json
                    annotation_schema.json
                video_0001/
                    ...
            metadata/
                video_0000.json
                video_0001.json
                ...
            dataset_manifest.json

    This layout ensures that:
        - Each video file has a unique filename.
        - Each video's masks are isolated in their own subtree.
        - Each video has a corresponding JSON manifest describing its
          parameters and particle attributes.
        - A single dataset_manifest.json summarizes all videos for easy
          iteration by downstream ML code.

    Randomness and reproducibility:
        - For each video, a per-video integer seed is derived from
          ``random_seed`` and that video's absolute index.
        - That per-video seed is used to:
              * Seed the canonical global np.random RNG, which is used by the
                core simulation (Brownian motion, random aberrations, detector
                noise, etc.).
              * Construct a per-video Generator passed to caller-local
                parameter builders.

        Providing the same random_seed, presets, and other arguments therefore
        makes each video index deterministic and reproducible even across
        interrupted/resumed runs with different batch sizes.

    Args:
        num_videos (int): Number of videos to generate. Must be >= 1.
        preset_name (Optional[str]): Public dataset preset name. See
            get_dataset_preset_names().
        instrument_preset (Optional[str]): Name of the instrument preset to
            use for instrument-specific workflows. Only applied to the default preset.
        base_output_dir (Optional[str]): Base directory under which all videos
            and masks will be written. If None, defaults to a project-relative
            ``outputs/syniscopy_dataset`` directory.
        random_seed (Optional[int]): Optional seed for the dataset-level NumPy
            random number generator. When provided, all random choices in both
            dataset-level parameter sampling and the core simulation
            (Brownian motion, optical aberration randomness, detector noise,
            etc.) become reproducible across runs with the same configuration.
        param_overrides (Optional[Mapping[str, Any]]): PARAMS key/value
            overrides applied to every generated video.
        resume_existing (bool): If True, skip already completed videos when a
            prior run with the same request is found. This is the default so
            interrupted long-running jobs continue instead of starting over.
        reset_existing (bool): If True, delete ``base_output_dir`` before
            generation. Use only when intentionally replacing a dataset.
        append_on_config_change (bool): If True, a new request in an existing
            folder appends a new batch of videos instead of deleting the existing
            dataset. The default is False to protect existing datasets from
            mismatched append operations.
        video_param_builder (Optional[Callable]): Optional callable that returns
            a complete PARAMS dictionary for each video.
        param_builder_name (Optional[str]): Stable label recorded in manifests
            when using ``video_param_builder``.
        verbose (bool): If True, emit dataset-generation progress logs at INFO.

    Raises:
        ValueError: If num_videos < 1.
    """
    if verbose:
        configure_logging(verbose=True)
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
    _reject_dataset_managed_overrides(recipe_overrides, "recipe_overrides")
    _reject_dataset_managed_overrides(param_overrides, "param_overrides")
    dataset_source_name = builder_label or effective_preset_name

    base_output_dir = _resolve_base_output_dir(base_output_dir)

    if reset_existing and os.path.exists(base_output_dir):
        logger.info("Reset requested; removing existing dataset directory: %s", base_output_dir)
        shutil.rmtree(base_output_dir)
        os.makedirs(base_output_dir, exist_ok=True)

    # Subdirectories for videos and masks.
    video_dir = os.path.join(base_output_dir, "videos")
    frames_root_dir = os.path.join(base_output_dir, "frames")
    masks_root_dir = os.path.join(base_output_dir, "masks")
    raw_views_dir = os.path.join(base_output_dir, "raw_frame_views")
    packets_dir = os.path.join(base_output_dir, "counterfactual_packets")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_root_dir, exist_ok=True)
    os.makedirs(masks_root_dir, exist_ok=True)
    os.makedirs(raw_views_dir, exist_ok=True)
    os.makedirs(packets_dir, exist_ok=True)

    request_payload = _dataset_request_payload(
        num_videos=num_videos,
        preset_name=effective_preset_name,
        instrument_preset=instrument_preset,
        random_seed=random_seed,
        recipe_overrides=recipe_overrides,
        param_overrides=param_overrides,
        param_builder_name=builder_label,
    )
    request_signature = _request_signature(request_payload)
    state_path = os.path.join(base_output_dir, _DATASET_STATE_FILENAME)
    prior_state = _load_json_file(state_path)
    prior_signature = (
        prior_state.get("request_signature")
        if isinstance(prior_state, dict)
        else None
    )
    prior_target_indices = (
        prior_state.get("target_indices")
        if isinstance(prior_state, dict)
        and isinstance(prior_state.get("target_indices"), list)
        else None
    )
    existing_entries_by_index = _load_completed_dataset_entries(base_output_dir)
    existing_indices = sorted(existing_entries_by_index)

    same_request = bool(prior_signature == request_signature)
    if existing_indices and not same_request:
        if reset_existing:
            same_request = True
        elif append_on_config_change:
            logger.info(
                "Existing dataset uses a different generation request; "
                "appending a new batch instead of deleting it."
            )
        else:
            raise ValueError(
                "Existing dataset was generated with different parameters. "
                "Use reset_existing=True to replace it or "
                "append_on_config_change=True to append a new batch."
            )

    if same_request and prior_target_indices:
        prior_indices = sorted({int(idx) for idx in prior_target_indices})
        start_index = min(prior_indices) if prior_indices else 0
        requested_indices = list(range(start_index, start_index + num_videos))
        target_indices = sorted(set(prior_indices).union(requested_indices))
        start_index = min(target_indices) if target_indices else 0
        mode = str(prior_state.get("mode", "resume")) if isinstance(prior_state, dict) else "resume"
    elif existing_indices and not same_request:
        start_index = max(existing_indices) + 1
        target_indices = list(range(start_index, start_index + num_videos))
        mode = "append"
    else:
        start_index = 0
        target_indices = list(range(num_videos))
        mode = "resume"

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
        video_id = f"video_{video_index:04d}"
        logger.info("=== %s (%s / %s) ===", video_id, batch_offset + 1, len(target_indices))

        if resume_existing and _video_assets_complete(base_output_dir, video_index):
            logger.info("Skipping completed %s", video_id)
            entry = existing_entries_by_index.get(video_index)
            if entry is not None:
                dataset_entries_by_index[video_index] = entry
            skipped_count += 1
            continue

        video_seed = seed_by_index[int(video_index)]

        # Seed the module-level NumPy RNG so internal randomness in the core
        # simulation is reproducible for this video.
        np.random.seed(video_seed)

        video_rng = np.random.default_rng(video_seed)

        if video_param_builder is not None:
            params = video_param_builder(video_index, video_rng)
            if not isinstance(params, dict):
                raise TypeError("video_param_builder must return a PARAMS dictionary.")
            params = apply_parameter_overrides(params, param_overrides)
        else:
            params = build_dataset_video_params(
                video_index=video_index,
                rng=video_rng,
                preset_name=effective_preset_name,
                instrument_preset=instrument_preset,
                recipe_overrides=recipe_overrides,
                param_overrides=param_overrides,
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
        masks_dir = os.path.join(masks_root_dir, video_id)
        raw_views_path = os.path.join(raw_views_dir, f"{video_id}.npz")
        channel_sidecar_dir = os.path.join(video_dir, "channels", video_id)
        matched_packet_path = os.path.join(packets_dir, f"{video_id}.npz")

        # If an interrupted attempt stopped mid-video, clear that one incomplete
        # video's owned outputs only. Completed videos are never touched by resume.
        if os.path.exists(video_filename):
            os.remove(video_filename)
        if os.path.exists(raw_views_path):
            os.remove(raw_views_path)
        if os.path.exists(matched_packet_path):
            os.remove(matched_packet_path)
        raw_views_tmp_path = raw_views_path + ".tmp"
        if os.path.exists(raw_views_tmp_path):
            os.remove(raw_views_tmp_path)
        if os.path.isdir(frame_sequence_dir):
            shutil.rmtree(frame_sequence_dir)
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

        save_frame_sequence = bool(params.get("save_frame_sequence", True))
        save_raw_frame_views = bool(params.get("save_raw_frame_views", False))
        simulation_result = run_simulation(
            params,
            return_frames=bool(save_frame_sequence or save_raw_frame_views),
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

        manifest = build_video_manifest(
            params=params,
            base_output_dir=base_output_dir,
            video_index=video_index,
            dataset_preset=dataset_source_name,
            instrument_preset=instrument_preset,
            video_seed=video_seed,
        )
        sidecars = result_metadata.get("channel_sidecar_videos", [])
        if sidecars:
            manifest["channel_sidecar_videos"] = [
                _json_safe(_relative_path(base_output_dir, str(path)))
                for path in sidecars
            ]
        if raw_views_rel is not None:
            manifest["raw_frame_views_npz"] = raw_views_rel
            manifest["background_subtracted_video_path"] = manifest.get("output_video_path")
        matched_modalities = params.get("matched_modalities")
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
                masks=packet_payload.get("masks"),
                fisher_by_modality=packet_payload.get("fisher_by_modality"),
                crlb_by_modality=packet_payload.get("crlb_by_modality"),
                metadata=packet_payload["metadata"],
                require_information_fields=True,
            )
            manifest["matched_modality_packet_npz"] = _json_safe(
                _relative_path(base_output_dir, saved_packet_path)
            )
            manifest["matched_modalities"] = [str(name) for name in matched_modalities]
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

    if representative_params is None:
        # No new videos were needed. Reconstruct a representative template so
        # the simulation manifest still reflects the active request.
        template_seed = seed_by_index.get(start_index, _derive_video_seed(random_seed, start_index))
        template_rng = np.random.default_rng(template_seed)
        if video_param_builder is not None:
            representative_params = apply_parameter_overrides(
                video_param_builder(start_index, template_rng),
                param_overrides,
            )
            representative_params["random_seed"] = int(template_seed)
            _validate_dataset_output_contract(representative_params)
        else:
            representative_params = build_dataset_video_params(
                video_index=start_index,
                rng=template_rng,
                preset_name=effective_preset_name,
                instrument_preset=instrument_preset,
                recipe_overrides=recipe_overrides,
                param_overrides=param_overrides,
            )
            representative_params["random_seed"] = int(template_seed)
            _validate_dataset_output_contract(representative_params)

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
    simulation_manifest["heterogeneous_dataset"] = bool(mode == "append")
    simulation_manifest["current_request"] = request_payload
    simulation_manifest["current_request_signature"] = request_signature
    simulation_manifest["target_indices"] = target_indices
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
            "mode": mode,
            "target_indices": target_indices,
            "completed_indices": [entry["video_index"] for entry in dataset_entries],
            "complete_total": len(dataset_entries),
        },
    )
    logger.info("Dataset generation complete.")
    return base_output_dir


def _json_safe(value: Any) -> Any:
    """
    Convert PARAMS values into JSON-friendly structures for template export.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {
            "real": _json_safe(float(value.real)),
            "imag": _json_safe(float(value.imag)),
        }
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_default_params_template(output_path: str) -> str:
    """
    Write the full default PARAMS surface as a JSON template and return the path.
    """
    output_path = os.path.abspath(os.path.expanduser(output_path))
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            _json_safe(get_default_dataset_params()),
            fh,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return output_path


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset generation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate datasets from Syniscopy simulation parameters."
        )
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate (default: 1).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help=(
            "Dataset preset to use. Public options include: "
            + ", ".join(get_dataset_preset_names())
            + ". Default: default."
        ),
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help=(
            "Name of the instrument preset to use (e.g., 'widefield_60x_high_na'). "
            "Applied only when --preset is default."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Base output directory for videos and masks. "
            "Defaults to 'outputs/syniscopy_dataset' if not provided."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional dataset-level random seed. When provided, all random "
            "choices in dataset-level parameter sampling and in the core "
            "simulation (Brownian motion, aberrations, detector noise, etc.) "
            "become reproducible across runs with the same configuration."
        ),
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing PARAMS key/value overrides to apply "
            "to every generated video."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output dataset directory before generation.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip completed videos in an existing dataset directory.",
    )
    parser.add_argument(
        "--append_on_config_change",
        action="store_true",
        help=(
            "Append when the output directory already contains videos from a "
            "different generation request. By default this raises instead of "
            "mixing datasets."
        ),
    )
    parser.add_argument(
        "--write_params_template",
        type=str,
        default=None,
        help=(
            "Write the complete default PARAMS surface to this JSON file and exit."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit dataset-generation progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point for dataset generation.

    Examples:
        # Generate a single video using the base config.PARAMS:
        python dataset_generator.py

        # Write a full editable parameter template:
        python dataset_generator.py --write_params_template params_template.json

        # Generate 5 videos using a named instrument preset, writing
        # outputs under a custom directory:
        python dataset_generator.py --num_videos 5 --instrument widefield_60x_high_na --output_dir /path/to/dataset

        # Generate a reproducible dataset with a fixed random seed:
        python dataset_generator.py --num_videos 10 --seed 12345
    """
    args = _parse_args()
    configure_logging(verbose=args.verbose)
    if args.write_params_template:
        path = write_default_params_template(args.write_params_template)
        logger.info("Wrote PARAMS template to %s", path)
        return
    param_overrides = None
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as fh:
            param_overrides = json.load(fh)
        if not isinstance(param_overrides, dict):
            raise ValueError("--params_json must contain a JSON object.")
    generate_dataset(
        num_videos=args.num_videos,
        preset_name=args.preset,
        instrument_preset=args.instrument,
        base_output_dir=args.output_dir,
        random_seed=args.seed,
        param_overrides=param_overrides,
        resume_existing=not args.no_resume,
        reset_existing=args.reset,
        append_on_config_change=args.append_on_config_change,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
