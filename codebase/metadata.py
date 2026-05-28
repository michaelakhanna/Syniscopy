"""
Metadata and manifest utilities for Syniscopy dataset generation.

This module provides a small, self-contained API for constructing and
saving JSON metadata for each generated video and for the dataset as a
whole. It is used by dataset_generator.generate_dataset and does not
change the physics or rendering behavior of the simulation.

Concepts
--------
- Per-video manifest:
    A JSON file that describes:
        * The video index and IDs.
        * Paths to the AVI preview file, mask directory, and, when generated, the
          lossless PNG frame sequence (relative to the dataset root).
        * Key simulation parameters relevant for ML training (fps,
          duration, image size, pixel size).
        * Substrate-pattern and background subtraction configuration.
        * Canonical particle objects, including motion diameter, component
          diameters, material labels, refractive indices, and signal multipliers.

- Dataset-level manifest:
    A JSON file that lists all videos in the dataset with minimal
    information needed to iterate over them (paths, presets, seeds).

Manifests are written after ``run_simulation(params)`` returns for each video.
They use information already present in the parameter dictionary plus the
per-video seed and preset names. Paths are stored relative to the dataset root
(``base_output_dir``) whenever possible.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List

from supervision_policy import build_policy_annotation_schema, resolve_policy_contract

SIMULATOR_VERSION = "1.0.5"
from materials import material_properties_to_dict, resolve_particle_material_properties
from particle_specs import get_particle_specs, particle_specs_to_public_dicts


def _relative_path(base_dir: str, path: str) -> str:
    """
    Return 'path' expressed relative to 'base_dir', if possible.

    On systems where base_dir and path are on different drives or when
    relpath fails, this falls back to returning the absolute path. This
    keeps manifests robust without imposing strict requirements on how
    users specify output directories.
    """
    base_dir_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    try:
        return os.path.relpath(path_abs, base_dir_abs)
    except ValueError:
        # Windows drive mismatches use the absolute path.
        return path_abs


def _safe_float(value: Any) -> float:
    """
    Convert a numeric-like value to a plain Python float.

    This is primarily used to convert numpy scalar types (e.g., np.float64,
    np.int64) into JSON-serializable primitives.
    """
    return float(value)


def _safe_complex_to_dict(z: complex | Any) -> Dict[str, float]:
    """
    Convert a complex (or complex-like) value into a dict with 'real' and
    'imag' fields suitable for JSON serialization.

    If 'z' is not already a complex instance but supports .real and .imag,
    those attributes are used.
    """
    if isinstance(z, complex):
        real = z.real
        imag = z.imag
    else:
        # Accept numpy complex scalars, etc.
        real = getattr(z, "real", 0.0)
        imag = getattr(z, "imag", 0.0)
    return {
        "real": _json_safe(_safe_float(real)),
        "imag": _json_safe(_safe_float(imag)),
    }


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/complex containers into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, complex):
        return _safe_complex_to_dict(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError, OverflowError):
            return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return value


def _resolved_noise_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical resolved camera-noise metadata for manifests."""
    try:
        from camera_noise import camera_noise_metadata

        return _json_safe(camera_noise_metadata(params))
    except Exception as exc:
        return {"metadata_error": repr(exc)}


def _resolved_num_frames_or_none(params: Dict[str, Any]) -> int | None:
    """Return the effective frame count when timing parameters are complete."""
    try:
        from trajectory import resolve_num_frames

        return int(resolve_num_frames(params))
    except Exception:
        raw_num_frames = params.get("num_frames")
        if raw_num_frames is None:
            return None
        try:
            return int(raw_num_frames)
        except (TypeError, ValueError):
            return None


def _git_commit_or_none(repo_root: str) -> str | None:
    """Best-effort git commit lookup; returns None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = out.strip()
    return commit or None


def build_video_manifest(
    params: Dict[str, Any],
    base_output_dir: str,
    video_index: int,
    dataset_preset: str | None,
    instrument_preset: str | None,
    video_seed: int,
) -> Dict[str, Any]:
    """
    Construct a per-video manifest dictionary from the simulation parameters
    and metadata known at the dataset orchestration level.

    This function assumes:
        - run_simulation(params) has already been called for this video.
        - params["particles"] describes the canonical particle objects.
        - params["output_filename"] and params["mask_output_directory"] are
          set to the paths used by the simulation.

    The returned dictionary is fully JSON-serializable and is intended to
    be written by save_video_manifest().
    """
    # Basic video-level properties
    fps = _safe_float(params["fps"])
    duration_seconds = _safe_float(params["duration_seconds"])
    raw_num_frames = params.get("num_frames", None)
    num_frames = (
        int(raw_num_frames)
        if raw_num_frames is not None
        else int(fps * duration_seconds)
    )
    image_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = _safe_float(params["pixel_size_nm"])

    output_filename = params["output_filename"]
    mask_output_directory = params["mask_output_directory"]

    manifest: Dict[str, Any] = {
        "video_index": int(video_index),
        "dataset_preset": dataset_preset,
        "instrument_preset": instrument_preset,
        "random_seed": int(video_seed),
        "output_video_path": _relative_path(base_output_dir, output_filename),
        "mask_root_dir": _relative_path(base_output_dir, mask_output_directory),
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "image_size_pixels": image_size_pixels,
        "pixel_size_nm": pixel_size_nm,
    }

    # Substrate-pattern and background-related configuration.
    _sub_enabled = bool(params.get("sample_environment_pattern_enabled", False))
    _sub_model = params.get("sample_environment_pattern", None)
    _sub_preset = params.get("sample_environment_pattern_preset", None)

    manifest["sample_environment_pattern_enabled"] = _sub_enabled
    manifest["sample_environment_pattern"] = str(_sub_model) if _sub_model is not None else None
    manifest["sample_environment_pattern_preset"] = str(_sub_preset) if _sub_preset is not None else None

    # Canonical counts-domain noise metadata resolved through camera_noise.py.
    noise_metadata = _resolved_noise_metadata(params)
    manifest["camera_noise"] = noise_metadata

    manifest["background_subtraction_method"] = str(
        params.get("background_subtraction_method", "video_median")
    )
    manifest["mask_generation_enabled"] = bool(params.get("mask_generation_enabled", False))
    manifest["mask_outer_ring_count"] = int(params.get("mask_outer_ring_count", 0))
    manifest["annotation_schema"] = build_policy_annotation_schema(params)
    policy_contract = resolve_policy_contract(params)
    manifest["supervision_policy"] = {
        "target": policy_contract["target"],
        "support_factors": policy_contract["support_factors"],
        "supported_threshold": _safe_float(
            params.get("supervision_supported_threshold", 0.2)
        ),
        "temporal_support_enabled": bool(
            params.get("supervision_temporal_support_enabled", True)
        ),
        "signal_support_enabled": bool(
            params.get("supervision_signal_support_enabled", True)
        ),
        "information_support_enabled": bool(
            params.get("supervision_information_support_enabled", True)
        ),
        "ambiguity_support_enabled": bool(
            params.get("supervision_ambiguity_support_enabled", True)
        ),
        "crlb_xy_max_nm": params.get("supervision_crlb_xy_max_nm", None),
        "ambiguity_distance_scale_nm": params.get(
            "supervision_ambiguity_distance_scale_nm",
            None,
        ),
        "prior_log_odds": _safe_float(
            params.get("supervision_prior_log_odds", 0.0)
        ),
    }
    manifest["crlb_policy"] = {
        "lateral_crlb_metadata": True,
        "axial_crlb_metadata": False,
        "orientation_crlb_metadata": False,
    }
    manifest["empirical_background_enabled"] = bool(
        params.get("empirical_background_enabled", False)
    )
    manifest["empirical_background_model"] = str(
        params.get("empirical_background_model", "multiscale_gaussian_field")
    )

    particle_specs = get_particle_specs(params)
    particle_material_properties = resolve_particle_material_properties(params)
    particles_meta = particle_specs_to_public_dicts(particle_specs)
    for i, entry in enumerate(particles_meta):
        entry["particle_index"] = int(i)
        entry["primary_material_properties"] = material_properties_to_dict(
            particle_material_properties[i],
            wavelength_nm=float(params.get("wavelength_nm", 532.0)),
        )
    manifest["particles"] = particles_meta


    return manifest


def save_video_manifest(
    manifest: Dict[str, Any],
    base_output_dir: str,
    video_index: int,
) -> str:
    """
    Save a per-video manifest to the dataset's metadata directory.

    The file is written as:
        <base_output_dir>/metadata/video_XXXX.json

    where XXXX is the zero-padded video index (4 digits). The parent
    'metadata' directory is created if it does not exist.

    Returns:
        str: Absolute path to the saved manifest file.
    """
    metadata_dir = os.path.join(base_output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    filename = os.path.join(metadata_dir, f"video_{video_index:04d}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True, allow_nan=False)

    return os.path.abspath(filename)


def build_dataset_index_entry(
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct a minimal dataset-level index entry from a per-video manifest.

    This extracts the fields needed to iterate over a dataset and locate each
    video's assets without duplicating all per-particle metadata.
    """
    entry: Dict[str, Any] = {
        "video_index": int(manifest["video_index"]),
        "video_id": f"video_{manifest['video_index']:04d}",
        "output_video_path": manifest["output_video_path"],
        "frame_sequence_dir": manifest.get("frame_sequence_dir"),
        "training_frames_dir": manifest.get("training_frames_dir", manifest.get("frame_sequence_dir")),
        "preview_video_path": manifest.get("preview_video_path", manifest.get("output_video_path")),
        "mask_root_dir": manifest["mask_root_dir"],
        "random_seed": int(manifest["random_seed"]),
        "dataset_preset": manifest.get("dataset_preset"),
        "instrument_preset": manifest.get("instrument_preset"),
    }
    if manifest.get("channel_sidecar_videos"):
        entry["channel_sidecar_videos"] = list(manifest["channel_sidecar_videos"])
    if manifest.get("matched_modality_packet_npz"):
        entry["matched_modality_packet_npz"] = manifest["matched_modality_packet_npz"]
        entry["matched_modalities"] = list(manifest.get("matched_modalities", []))
    return entry


def save_dataset_manifest(
    base_output_dir: str,
    dataset_entries: List[Dict[str, Any]],
) -> str:
    """
    Save the dataset-level manifest file listing all videos.

    The file is written as:
        <base_output_dir>/dataset_manifest.json

    The JSON structure is:

        {
          "base_output_dir": "<absolute path>",
          "num_videos": <int>,
          "videos": [ ... entries ... ]
        }

    Returns:
        str: Absolute path to the saved dataset manifest file.
    """
    if not isinstance(dataset_entries, list):
        raise TypeError("dataset_entries must be a list of per-video index entries.")

    base_output_dir_abs = os.path.abspath(base_output_dir)

    payload: Dict[str, Any] = {
        "schema_version": "syniscopy-dataset-manifest-v1",
        "base_output_dir": base_output_dir_abs,
        "num_videos": len(dataset_entries),
        "videos": dataset_entries,
    }

    filename = os.path.join(base_output_dir_abs, "dataset_manifest.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)

    return filename


def build_simulation_manifest(
    *,
    base_output_dir: str,
    dataset_entries: List[Dict[str, Any]],
    params_template: Dict[str, Any] | None,
    random_seed: int | None,
    dataset_preset: str | None,
    simulator_version: str = SIMULATOR_VERSION,
) -> Dict[str, Any]:
    """
    Build a first-class machine-readable manifest for a generated dataset.

    This supplements the lightweight ``dataset_manifest.json`` with the full
    simulator/version/configuration contract expected by reproducibility tools
    and downstream training notebooks.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    noise_metadata = (
        None if params_template is None else _resolved_noise_metadata(params_template)
    )
    return {
        "schema_version": "syniscopy-simulation-manifest-v1",
        "simulator": "Syniscopy",
        "simulator_version": simulator_version,
        "git_commit": _git_commit_or_none(repo_root),
        "random_seed": None if random_seed is None else int(random_seed),
        "dataset_preset": dataset_preset,
        "base_output_dir": os.path.abspath(base_output_dir),
        "num_videos": len(dataset_entries),
        "videos": dataset_entries,
        "modality": (
            None if params_template is None
            else str(params_template.get("imaging_model", "bright_field"))
        ),
        "particle_geometry": (
            None if params_template is None
            else _json_safe({
                "particles": params_template.get(
                    "_resolved_particles",
                    params_template.get("particles"),
                ),
            })
        ),
        "trajectory_parameters": (
            None if params_template is None
            else _json_safe({
                "fps": params_template.get("fps"),
                "duration_seconds": params_template.get("duration_seconds"),
                "num_frames": _resolved_num_frames_or_none(params_template),
                "temperature_K": params_template.get("temperature_K"),
                "viscosity_Pa_s": params_template.get("viscosity_Pa_s"),
                "initial_z_span_nm": params_template.get("initial_z_span_nm"),
                "z_motion_constraint_model": params_template.get(
                    "z_motion_constraint_model"
                ),
                "rotational_diffusion_enabled": params_template.get(
                    "rotational_diffusion_enabled"
                ),
                "rotational_diffusion_mode": params_template.get(
                    "rotational_diffusion_mode"
                ),
                "rotational_step_std_deg": params_template.get(
                    "rotational_step_std_deg"
                ),
                "sample_environment_exclusion_method": params_template.get(
                    "sample_environment_exclusion_method"
                ),
            })
        ),
        "sample_environment_parameters": (
            None if params_template is None
            else _json_safe({
                key: params_template.get(key)
                for key in params_template
                if str(key).startswith("sample_environment_pattern")
                or str(key).startswith("empirical_background")
            })
        ),
        "camera_noise": noise_metadata,
        "supervision_policy": (
            None if params_template is None
            else _json_safe({
                **resolve_policy_contract(params_template),
                "supported_threshold": params_template.get(
                    "supervision_supported_threshold"
                ),
                "temporal_support_enabled": params_template.get(
                    "supervision_temporal_support_enabled"
                ),
                "signal_support_enabled": params_template.get(
                    "supervision_signal_support_enabled"
                ),
                "information_support_enabled": params_template.get(
                    "supervision_information_support_enabled"
                ),
                "ambiguity_support_enabled": params_template.get(
                    "supervision_ambiguity_support_enabled"
                ),
                "ambiguity_distance_scale_nm": params_template.get(
                    "supervision_ambiguity_distance_scale_nm"
                ),
                "prior_log_odds": params_template.get(
                    "supervision_prior_log_odds"
                ),
            })
        ),
        "crlb_policy": {
            "lateral_crlb_metadata": True,
            "axial_crlb_metadata": False,
            "orientation_crlb_metadata": False,
        },
        "annotation_schema": (
            None if params_template is None
            else build_policy_annotation_schema(params_template)
        ),
    }


def save_simulation_manifest(manifest: Dict[str, Any], base_output_dir: str) -> str:
    filename = os.path.join(os.path.abspath(base_output_dir), "simulation_manifest.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True, allow_nan=False)
    return filename
