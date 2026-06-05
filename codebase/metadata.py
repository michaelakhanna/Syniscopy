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

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List

from common_utils import relative_path
from config.runtime import param_value, resolved_modality
from supervision_policy import build_policy_annotation_schema, resolve_policy_contract

SIMULATOR_VERSION = "1.0.5"
from imaging_models import get_imaging_model
from json_utils import json_safe
from material_serialization import material_properties_to_dict
from particle_material_resolution import (
    resolve_component_material_properties,
    resolve_particle_material_properties,
)
from optical_params import resolve_probe_wavelength_nm
from particle_specs import get_particle_specs, particle_specs_to_public_dicts
from experiment_contracts import (
    ArtifactNode,
    artifact_graph_manifest,
    backend_contract_for_modality,
    contracts_manifest,
    detector_model_from_params,
)


def _safe_float(value: Any) -> float:
    """
    Convert a numeric-like value to a plain Python float.

    This is primarily used to convert numpy scalar types (e.g., np.float64,
    np.int64) into JSON-serializable primitives.
    """
    return float(value)


def _resolved_noise_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical resolved camera-noise metadata for manifests."""
    try:
        from camera_noise import camera_noise_metadata

        return json_safe(camera_noise_metadata(params), flexible_numpy=True)
    except Exception as exc:
        return {"metadata_error": repr(exc)}


def _resolved_num_frames_or_none(params: Dict[str, Any]) -> int | None:
    """Return the effective frame count when timing parameters are complete."""
    try:
        from trajectory import resolve_num_frames

        return int(resolve_num_frames(params))
    except Exception:
        try:
            raw_num_frames = param_value(params, "num_frames")
        except KeyError:
            return None
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
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.strip()
    return commit or None


def _git_dirty_or_none(repo_root: str) -> bool | None:
    """Best-effort dirty-worktree flag; returns None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.strip())


def _git_root_or_none(source_root: str) -> str | None:
    """Resolve the git root for source provenance without assuming repo layout."""
    candidates = [
        source_root,
        os.path.join(source_root, "codebase"),
        os.path.dirname(__file__),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        root = out.strip()
        if root:
            return os.path.abspath(root)
    return None


def _sha256_file_or_none(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_source_provenance(repo_root: str | None = None) -> Dict[str, Any]:
    """Build the source-code fingerprint that governs generated outputs.

    The dataset generator may resume partially complete datasets. This
    fingerprint is intentionally independent of numeric run parameters: if the
    code/notebook/source package changes, regenerated outputs must not silently
    claim the current source provenance while reusing stale frames or masks.
    """
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    repo_root = os.path.abspath(repo_root)
    git_root = _git_root_or_none(repo_root)
    tracked_roots = [
        "codebase",
        "recipes",
        "scripts",
        "supplemental",
        "sam2_starter",
    ]
    suffixes = {".py", ".ipynb", ".md"}
    excluded_parts = {
        "outputs",
        "__pycache__",
        ".ipynb_checkpoints",
    }
    file_records: list[Dict[str, Any]] = []
    for root_name in tracked_roots:
        root = os.path.join(repo_root, root_name)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in excluded_parts and not d.startswith(".")
            ]
            rel_dir = os.path.relpath(dirpath, repo_root)
            rel_parts = set(rel_dir.split(os.sep))
            if rel_parts & excluded_parts:
                continue
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, repo_root)
                if rel == os.path.join("supplemental", "syniscopy_source.zip"):
                    continue
                if os.path.splitext(filename)[1] not in suffixes:
                    continue
                digest = _sha256_file_or_none(path)
                if digest is not None:
                    file_records.append({"path": rel.replace(os.sep, "/"), "sha256": digest})
    file_records.sort(key=lambda item: item["path"])
    aggregate = hashlib.sha256()
    for record in file_records:
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    source_zip = os.path.join(repo_root, "supplemental", "syniscopy_source.zip")
    return {
        "schema_version": "syniscopy-source-provenance-v1",
        "repo_root": repo_root,
        "git_root": git_root,
        "git_commit": None if git_root is None else _git_commit_or_none(git_root),
        "git_dirty": None if git_root is None else _git_dirty_or_none(git_root),
        "fingerprint": aggregate.hexdigest(),
        "file_count": len(file_records),
        "source_zip_path": (
            os.path.relpath(source_zip, repo_root).replace(os.sep, "/")
            if os.path.exists(source_zip)
            else None
        ),
        "source_zip_sha256": _sha256_file_or_none(source_zip),
    }


def build_video_manifest(
    params: Dict[str, Any],
    base_output_dir: str,
    video_index: int,
    dataset_preset: str | None,
    instrument_preset: str | None,
    video_seed: int,
    result_metadata: Dict[str, Any] | None = None,
    composition_leaf_index: int | None = None,
    composition_leaf_name: str | None = None,
    composition_leaf_signature: str | None = None,
    composition_local_index: int | None = None,
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
    raw_num_frames = param_value(params, 'num_frames')
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
        "source_provenance": build_source_provenance(),
        "output_video_path": relative_path(base_output_dir, output_filename),
        "analysis_video_path": relative_path(base_output_dir, output_filename),
        "mask_root_dir": relative_path(base_output_dir, mask_output_directory),
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "image_size_pixels": image_size_pixels,
        "pixel_size_nm": pixel_size_nm,
        "frame_products": {
            "output_video_path": "8-bit background-subtracted contrast-analysis AVI",
            "analysis_video_path": "same artifact as output_video_path; contrast-analysis preview",
            "raw_signal_video_path": "8-bit windowed preview of raw detector signal frames",
            "raw_camera_frame_sequence_dir": "uint16 PNG sequence of raw detector signal frames when requested",
            "training_frames_dir": "lossless 8-bit background-subtracted frame sequence for downstream training",
            "mask_root_dir": "per-particle binary supervision sidecars on the final rendered frame grid",
            "metadata_json": "machine-readable provenance and per-particle/frame records",
        },
    }

    # Substrate-pattern and background-related configuration.
    _sub_enabled = bool(param_value(params, 'sample_environment_pattern_enabled'))
    _sub_model = param_value(params, "sample_environment_pattern")
    _sub_preset = param_value(params, "sample_environment_pattern_preset")

    manifest["sample_environment_pattern_enabled"] = _sub_enabled
    manifest["sample_environment_pattern"] = str(_sub_model) if _sub_model is not None else None
    manifest["sample_environment_pattern_preset"] = str(_sub_preset) if _sub_preset is not None else None

    render_metadata = dict((result_metadata or {}).get("render_metadata") or {})
    response_function = dict(render_metadata.get("response_function") or {})

    # Canonical counts-domain noise metadata resolved through camera_noise.py.
    noise_metadata = _resolved_noise_metadata(params)
    manifest["camera_noise"] = noise_metadata
    modality_name = resolved_modality(params)
    backend_contract = backend_contract_for_modality(modality_name, response_function).to_dict()
    manifest["backend_contract"] = backend_contract
    manifest["detector_model"] = detector_model_from_params(params).to_dict()
    manifest["comparison_contracts"] = contracts_manifest([modality_name])
    manifest["artifact_graph"] = artifact_graph_manifest([ArtifactNode(artifact_id=f"video:{int(video_index):04d}", artifact_path=manifest["output_video_path"], artifact_type="simulation_video_manifest", source_notebook_or_script="codebase/dataset_generator.py", model_version=backend_contract.get("backend_id", ""), paper_consumers=("dataset_manifest",), heavy_execution=False)])
    if result_metadata:
        raw_signal_video_path = result_metadata.get("raw_signal_video_path")
        if raw_signal_video_path:
            manifest["raw_signal_video_path"] = relative_path(base_output_dir, str(raw_signal_video_path))
            manifest["raw_signal_video_semantics"] = result_metadata.get(
                "raw_signal_video_semantics",
                "windowed_raw_detector_count_preview_uint8",
            )
        if result_metadata.get("analysis_video_semantics"):
            manifest["analysis_video_semantics"] = result_metadata.get("analysis_video_semantics")
        if render_metadata is not None:
            manifest["render_metadata"] = json_safe(render_metadata, flexible_numpy=True)
        source_map_provenance = result_metadata.get("source_map_provenance")
        if source_map_provenance is not None:
            manifest["source_map_provenance"] = json_safe(source_map_provenance, flexible_numpy=True)
        if result_metadata.get("clip_diagnostics") is not None:
            manifest["clip_diagnostics"] = json_safe(result_metadata.get("clip_diagnostics"), flexible_numpy=True)

    manifest["background_subtraction_method"] = str(
        param_value(params, 'background_subtraction_method')
    )
    manifest["mask_generation_enabled"] = bool(param_value(params, "mask_generation_enabled"))
    manifest["mask_outer_ring_count"] = int(param_value(params, 'mask_outer_ring_count'))
    if composition_leaf_index is not None:
        manifest["composition_leaf_index"] = int(composition_leaf_index)
    if composition_leaf_name is not None:
        manifest["composition_leaf_name"] = str(composition_leaf_name)
    if composition_leaf_signature is not None:
        manifest["composition_leaf_signature"] = str(composition_leaf_signature)
    if composition_local_index is not None:
        manifest["composition_local_index"] = int(composition_local_index)
    manifest["annotation_schema"] = build_policy_annotation_schema(params)
    policy_contract = resolve_policy_contract(params)
    manifest["supervision_policy"] = {
        "target": policy_contract["target"],
        "support_factors": policy_contract["support_factors"],
        "supported_threshold": _safe_float(
            param_value(params, 'supervision_supported_threshold')
        ),
        "temporal_support_enabled": bool(
            param_value(params, 'supervision_temporal_support_enabled')
        ),
        "signal_support_enabled": bool(
            param_value(params, 'supervision_signal_support_enabled')
        ),
        "information_support_enabled": bool(
            param_value(params, 'supervision_information_support_enabled')
        ),
        "ambiguity_support_enabled": bool(
            param_value(params, 'supervision_ambiguity_support_enabled')
        ),
        "crlb_xy_max_nm": param_value(params, 'supervision_crlb_xy_max_nm'),
        "ambiguity_distance_scale_nm": param_value(params, 'supervision_ambiguity_distance_scale_nm'),
        "prior_log_odds": _safe_float(
            param_value(params, 'supervision_prior_log_odds')
        ),
    }
    manifest["crlb_policy"] = {
        "lateral_crlb_metadata": True,
        "axial_crlb_metadata": False,
        "orientation_crlb_metadata": False,
    }
    manifest["empirical_background_enabled"] = bool(
        param_value(params, 'empirical_background_enabled')
    )
    manifest["empirical_background_model"] = str(
        param_value(params, 'empirical_background_model')
    )

    particle_specs = get_particle_specs(params)
    try:
        require_optical = bool(getattr(get_imaging_model(params), "requires_complex_optical_psf", True))
    except Exception:
        require_optical = True
    particle_material_properties = resolve_particle_material_properties(
        params,
        require_optical_refractive_index=require_optical,
    )
    particles_meta = particle_specs_to_public_dicts(particle_specs)
    material_wavelength_nm = resolve_probe_wavelength_nm(params)
    for i, entry in enumerate(particles_meta):
        entry["particle_index"] = int(i)
        entry["primary_material_properties"] = material_properties_to_dict(
            particle_material_properties[i],
            wavelength_nm=material_wavelength_nm,
        )
        components_meta = entry.get("components", [])
        for component_index, component in enumerate(getattr(particle_specs[i], "components", []) or []):
            if component_index >= len(components_meta):
                continue
            try:
                component_material = resolve_component_material_properties(
                    params,
                    component,
                    require_optical_refractive_index=require_optical,
                )
                components_meta[component_index]["material_properties"] = material_properties_to_dict(
                    component_material,
                    wavelength_nm=material_wavelength_nm,
                )
            except Exception as exc:
                components_meta[component_index]["material_properties_error"] = repr(exc)
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
        json.dump(json_safe(manifest, flexible_numpy=True), f, indent=2, sort_keys=True, allow_nan=False)

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
        "analysis_video_path": manifest.get("analysis_video_path", manifest["output_video_path"]),
        "raw_signal_video_path": manifest.get("raw_signal_video_path"),
        "frame_sequence_dir": manifest.get("frame_sequence_dir"),
        "raw_camera_frame_sequence_dir": manifest.get("raw_camera_frame_sequence_dir"),
        "training_frames_dir": manifest.get("training_frames_dir", manifest.get("frame_sequence_dir")),
        "preview_video_path": manifest.get("preview_video_path", manifest.get("output_video_path")),
        "mask_root_dir": manifest["mask_root_dir"],
        "random_seed": int(manifest["random_seed"]),
        "dataset_preset": manifest.get("dataset_preset"),
        "instrument_preset": manifest.get("instrument_preset"),
    }
    if manifest.get("composition_leaf_index") is not None:
        entry["composition_leaf_index"] = manifest.get("composition_leaf_index")
    if manifest.get("composition_leaf_name") is not None:
        entry["composition_leaf_name"] = manifest.get("composition_leaf_name")
    if manifest.get("composition_leaf_signature") is not None:
        entry["composition_leaf_signature"] = manifest.get("composition_leaf_signature")
    if manifest.get("composition_local_index") is not None:
        entry["composition_local_index"] = manifest.get("composition_local_index")
    if manifest.get("channel_sidecar_videos"):
        entry["channel_sidecar_videos"] = list(manifest["channel_sidecar_videos"])
    if manifest.get("matched_modality_packet_npz"):
        entry["matched_modality_packet_npz"] = manifest["matched_modality_packet_npz"]
        entry["matched_modalities"] = list(manifest.get("matched_modalities", []))
    if manifest.get("source_provenance"):
        entry["source_provenance_fingerprint"] = manifest["source_provenance"].get("fingerprint")
        entry["source_git_commit"] = manifest["source_provenance"].get("git_commit")
        entry["source_git_dirty"] = manifest["source_provenance"].get("git_dirty")
    if manifest.get("frame_products"):
        entry["frame_products"] = dict(manifest["frame_products"])
    return entry


def save_dataset_manifest(
    base_output_dir: str,
    dataset_entries: List[Dict[str, Any]],
    source_provenance: Dict[str, Any] | None = None,
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
        "source_provenance": (
            source_provenance
            if source_provenance is not None
            else build_source_provenance()
        ),
        "videos": dataset_entries,
    }

    filename = os.path.join(base_output_dir_abs, "dataset_manifest.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload, flexible_numpy=True), f, indent=2, sort_keys=True, allow_nan=False)

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
        "source_provenance": build_source_provenance(repo_root),
        "random_seed": None if random_seed is None else int(random_seed),
        "dataset_preset": dataset_preset,
        "base_output_dir": os.path.abspath(base_output_dir),
        "num_videos": len(dataset_entries),
        "videos": dataset_entries,
        "modality": (
            None if params_template is None
            else str(param_value(params_template, "imaging_model"))
        ),
        "particle_geometry": (
            None if params_template is None
            else json_safe({
                "particles": (
                    params_template["_resolved_particles"]
                    if "_resolved_particles" in params_template
                    else param_value(params_template, "particles")
                ),
            }, flexible_numpy=True)
        ),
        "trajectory_parameters": (
            None if params_template is None
            else json_safe({
                "fps": param_value(params_template, "fps"),
                "duration_seconds": param_value(params_template, "duration_seconds"),
                "num_frames": _resolved_num_frames_or_none(params_template),
                "temperature_K": param_value(params_template, "temperature_K"),
                "viscosity_Pa_s": param_value(params_template, "viscosity_Pa_s"),
                "initial_z_span_nm": param_value(params_template, "initial_z_span_nm"),
                "z_motion_constraint_model": param_value(params_template, "z_motion_constraint_model"),
                "rotational_diffusion_enabled": param_value(params_template, "rotational_diffusion_enabled"),
                "rotational_diffusion_mode": param_value(params_template, "rotational_diffusion_mode"),
                "rotational_step_std_deg": param_value(params_template, "rotational_step_std_deg"),
                "sample_environment_exclusion_method": param_value(params_template, "sample_environment_exclusion_method"),
            }, flexible_numpy=True)
        ),
        "sample_environment_parameters": (
            None if params_template is None
            else json_safe({
                key: params_template[key]
                for key in params_template
                if str(key).startswith("sample_environment_pattern")
                or str(key).startswith("empirical_background")
            }, flexible_numpy=True)
        ),
        "camera_noise": noise_metadata,
        "supervision_policy": (
            None if params_template is None
            else json_safe({
                **resolve_policy_contract(params_template),
                "supported_threshold": param_value(params_template, "supervision_supported_threshold"),
                "temporal_support_enabled": param_value(params_template, "supervision_temporal_support_enabled"),
                "signal_support_enabled": param_value(params_template, "supervision_signal_support_enabled"),
                "information_support_enabled": param_value(params_template, "supervision_information_support_enabled"),
                "ambiguity_support_enabled": param_value(params_template, "supervision_ambiguity_support_enabled"),
                "ambiguity_distance_scale_nm": param_value(params_template, "supervision_ambiguity_distance_scale_nm"),
                "prior_log_odds": param_value(params_template, "supervision_prior_log_odds"),
            }, flexible_numpy=True)
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
        json.dump(json_safe(manifest, flexible_numpy=True), f, indent=2, sort_keys=True, allow_nan=False)
    return filename
