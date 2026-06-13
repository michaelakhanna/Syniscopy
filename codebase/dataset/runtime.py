"""Dataset runtime path, parameter, and frame-asset helpers."""

from __future__ import annotations
from configured_parameters import configured_assign

import json
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config import default_params
from json_utils import json_safe
from presets import apply_instrument_preset
from simulation_runtime_state import runtime_state

from .overrides import apply_parameter_overrides

PUBLIC_DATASET_PRESET_DESCRIPTIONS: Dict[str, str] = {
    "default": "Core concept-owned default parameter surface for programmatic dataset generation.",
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

    Users can inspect or dump this structure to see every configurable
    parameter that the core generator accepts.
    """
    return default_params()

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
        os.path.join(base_output_dir, "videos", f"{video_id}_raw_signal.avi"),
        os.path.join(base_output_dir, "frames", video_id),
        os.path.join(base_output_dir, "raw_camera_frames", video_id),
        os.path.join(base_output_dir, "masks", video_id),
        os.path.join(base_output_dir, "raw_frame_views", f"{video_id}.npz"),
        os.path.join(base_output_dir, "raw_frame_views", f"{video_id}.npz.tmp"),
        os.path.join(base_output_dir, "videos", "channels", video_id),
        os.path.join(base_output_dir, "matched_microscope_packets", f"{video_id}.npz"),
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
    Write background-subtracted contrast-analysis frames as a PNG sequence.

    These PNGs losslessly encode the 8-bit display/training frames. They are not
    the quantitative raw/ideal simulation arrays; enable save_raw_frame_views for
    those audit artifacts or save_raw_camera_frame_sequence for uint16 raw camera
    frames. The main AVI video is a compact contrast-analysis preview.
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


def _raw_signal_frames_from_result_metadata(result_metadata: Mapping[str, Any]) -> np.ndarray:
    if "raw_signal_frames_rgb" in result_metadata:
        return np.asarray(result_metadata["raw_signal_frames_rgb"])
    if "raw_signal_frames" in result_metadata:
        return np.asarray(result_metadata["raw_signal_frames"])
    raise ValueError(
        "Raw camera frame sequence requested, but simulation metadata does not "
        "contain raw_signal_frames or raw_signal_frames_rgb."
    )


def _save_raw_camera_frame_sequence(
    frames: np.ndarray,
    frame_dir: str,
    *,
    bit_depth: int,
) -> int:
    """Write raw detector signal frames as uint16 PNG files."""
    import cv2

    frames = np.asarray(frames)
    if frames.ndim < 3 or frames.shape[0] == 0:
        raise ValueError("Cannot save a raw camera frame sequence from an empty frame array.")
    bit_depth = int(bit_depth)
    if bit_depth < 1 or bit_depth > 16:
        raise ValueError(f"bit_depth must be in [1, 16] for raw camera PNG output; got {bit_depth}.")
    max_count = float((1 << bit_depth) - 1)

    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        arr = np.asarray(frame, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "Cannot save non-finite raw camera frame data; "
                f"frame {frame_index} contains NaN or Inf."
            )
        arr_u16 = np.rint(np.clip(arr, 0.0, max_count)).astype(np.uint16)
        if arr_u16.ndim == 2:
            to_write = arr_u16
        elif arr_u16.ndim == 3 and arr_u16.shape[2] == 3:
            to_write = cv2.cvtColor(arr_u16, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                "Raw camera frames must be grayscale or RGB arrays; "
                f"got {arr_u16.shape}."
            )

        out_path = os.path.join(frame_dir, f"{frame_index:06d}.png")
        ok = cv2.imwrite(out_path, to_write)
        if not ok:
            raise RuntimeError(f"Failed to write raw camera frame {out_path!r}.")

    return int(frames.shape[0])


def _build_params_for_video(preset_name: Optional[str]) -> Dict[str, Any]:
    """
    Construct a parameter dictionary for a single video.
    """
    normalized_preset = _normalize_dataset_preset_name(preset_name)

    if normalized_preset == "default":
        params = default_params()
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
    Build the full parameters dictionary for one dataset video.

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
    configured_assign(params, 'random_seed', seed_value)
    runtime_state(params).substrate_pattern_layout_cache_token = f"video_{video_index:04d}:{seed_value}"
    return params

def _raw_frame_view_payload(
    result_metadata: Mapping[str, Any],
    final_frames_for_raw_view,
) -> Dict[str, np.ndarray]:
    """Build the compressed NPZ payload for raw-frame audit views."""
    payload: Dict[str, np.ndarray] = {
        "background_subtracted_frames": np.asarray(final_frames_for_raw_view),
        "trajectories_nm": np.asarray(result_metadata.get("trajectories_nm", [])),
        "rendered_trajectories_nm": np.asarray(
            result_metadata.get("rendered_trajectories_nm", [])
        ),
    }
    for key in (
        "raw_signal_frames",
        "raw_reference_frames",
        "ideal_signal_frames",
        "ideal_reference_frames",
        "detector_input_signal_frames",
        "detector_input_reference_frames",
        "detector_mean_signal_frames",
        "detector_mean_reference_frames",
        "detector_object_field_frames",
        "raw_signal_frames_rgb",
        "raw_reference_frames_rgb",
        "ideal_signal_frames_by_spectral_sample",
        "ideal_reference_frames_by_spectral_sample",
        "detector_input_signal_frames_by_spectral_sample",
        "detector_input_reference_frames_by_spectral_sample",
        "detector_mean_signal_frames_by_spectral_sample",
        "detector_mean_reference_frames_by_spectral_sample",
        "background_subtracted_frames_rgb",
        "contrast_frames_float",
        "raw_observation_contrast_frames",
        "analysis_contrast_frames",
        "quantitative_contrast_frames",
    ):
        if key in result_metadata:
            payload[key] = np.asarray(result_metadata[key])
    for key in (
        "analysis_contrast_frame_units",
        "analysis_contrast_frame_semantics",
        "analysis_contrast_frame_source",
        "analysis_contrast_frame_basis",
        "analysis_contrast_frame_contrast_basis",
        "analysis_contrast_frame_quantitative",
        "analysis_contrast_frame_safe_for_fisher",
        "analysis_contrast_frame_provenance_warning",
        "analysis_contrast_frame_contract_id",
        "raw_observation_contrast_frame_source",
        "raw_observation_contrast_frame_basis",
        "raw_observation_contrast_frame_contrast_basis",
        "raw_observation_contrast_frame_units",
        "raw_observation_contrast_frame_quantitative",
        "quantitative_contrast_frame_key",
        "quantitative_contrast_frame_source",
        "quantitative_contrast_frame_quantitative",
        "quantitative_contrast_frame_basis",
        "quantitative_contrast_frame_contrast_basis",
        "quantitative_contrast_frame_units",
        "quantitative_contrast_frame_semantics",
        "quantitative_contrast_frame_safe_for_fisher",
        "quantitative_contrast_frame_provenance_warning",
        "quantitative_contrast_frame_contract_id",
        "quantitative_contrast_background_subtraction_method",
        "quantitative_contrast_display_background_subtraction_applied",
    ):
        if key in result_metadata:
            payload[key] = np.array(str(result_metadata[key]))
    if "source_map_provenance" in result_metadata:
        payload["source_map_provenance_json"] = np.array(
            json.dumps(
                json_safe(result_metadata["source_map_provenance"]),
                sort_keys=True,
                allow_nan=False,
            )
        )
    if "trajectory_semantics" in result_metadata:
        payload["trajectory_semantics_json"] = np.array(
            json.dumps(
                json_safe(result_metadata["trajectory_semantics"]),
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
    Write the full default parameters surface as a JSON template and return the path.
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
