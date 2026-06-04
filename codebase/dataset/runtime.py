"""Dataset runtime path, parameter, and frame-asset helpers."""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config import PARAMS
from json_utils import json_safe
from presets import apply_instrument_preset

from .overrides import apply_parameter_overrides

PUBLIC_DATASET_PRESET_DESCRIPTIONS: Dict[str, str] = {
    "default": "Core config.PARAMS surface for programmatic dataset generation.",
}

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

def _remove_video_artifacts(base_output_dir: str, video_index: int) -> None:
    video_id = f"video_{video_index:04d}"
    removals = [
        os.path.join(base_output_dir, "videos", f"{video_id}.avi"),
        os.path.join(base_output_dir, "frames", video_id),
        os.path.join(base_output_dir, "masks", video_id),
        os.path.join(base_output_dir, "raw_frame_views", f"{video_id}.npz"),
        os.path.join(base_output_dir, "raw_frame_views", f"{video_id}.npz.tmp"),
        os.path.join(base_output_dir, "videos", "channels", video_id),
        os.path.join(base_output_dir, "counterfactual_packets", f"{video_id}.npz"),
        os.path.join(base_output_dir, "metadata", f"{video_id}.json"),
    ]
    for path in removals:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

def _clear_dataset_output_directory(base_output_dir: str) -> None:
    if os.path.isdir(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

def _video_manifest_path(base_output_dir: str, video_index: int) -> str:
    return os.path.join(base_output_dir, "metadata", f"video_{video_index:04d}.json")

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
    Write background-subtracted final frames as a PNG sequence.

    These PNGs losslessly encode the 8-bit display/training frames. They are not
    the quantitative raw/ideal simulation arrays; enable save_raw_frame_views for
    those audit artifacts. The AVI video is only a compact preview.
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
    parameter set that full dataset generation will use. The default builder
    is recipe-deterministic, but it records per-video seed/cache metadata from
    ``video_index`` and ``rng`` so previews and generated videos share the same
    run-scoped deterministic context.
    """
    video_index = int(video_index)
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
    seed_value = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    params["random_seed"] = seed_value
    params["_substrate_pattern_layout_cache_token"] = f"video_{video_index:04d}:{seed_value}"
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
        "ideal_signal_frames",
        "ideal_reference_frames",
        "raw_signal_frames_rgb",
        "raw_reference_frames_rgb",
        "ideal_signal_frames_by_spectral_sample",
        "ideal_reference_frames_by_spectral_sample",
        "background_subtracted_frames_rgb",
    ):
        if key in result_metadata:
            payload[key] = np.asarray(result_metadata[key])
    if "source_map_provenance" in result_metadata:
        payload["source_map_provenance_json"] = np.array(
            json.dumps(
                json_safe(result_metadata["source_map_provenance"]),
                sort_keys=True,
                allow_nan=False,
            )
        )
    if "render_metadata" in result_metadata:
        payload["render_metadata_json"] = np.array(
            json.dumps(
                json_safe(result_metadata["render_metadata"]),
                sort_keys=True,
                allow_nan=False,
            )
        )
    return payload

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
            json_safe(get_default_dataset_params()),
            fh,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return output_path
