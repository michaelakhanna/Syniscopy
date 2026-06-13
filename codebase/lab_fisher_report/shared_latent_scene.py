"""Shared latent-scene ownership for lab Fisher microscope comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from config import AcquisitionProfile
from json_utils import json_safe
from simulation.latent_scene import _simulate_latent_scene_at_times

from .scene_view import scene_provenance_from_params


def _rounded_time_key(value: float) -> str:
    return f"{float(value):.12g}"


def _frame_center_times_s(params: Mapping[str, Any]) -> np.ndarray:
    acquisition = AcquisitionProfile.from_params(params)
    return (
        (np.arange(acquisition.num_frames, dtype=float) + 0.5)
        * acquisition.frame_interval_s
    )


@dataclass(frozen=True)
class SharedLatentSchedule:
    """Physical report timebase and microscope sampling indices."""

    times_s: np.ndarray
    microscope_frame_indices: dict[str, np.ndarray]
    microscope_frame_times_s: dict[str, np.ndarray]
    policy: str = "union_frame_center_grid"

    @property
    def schedule_id(self) -> str:
        payload = {
            "policy": self.policy,
            "times_s": [float(v) for v in self.times_s],
            "microscope_frame_indices": {
                key: [int(v) for v in value]
                for key, value in sorted(self.microscope_frame_indices.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "latent-schedule-" + hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def fusion_time_alignment(self) -> str:
        frame_times = list(self.microscope_frame_times_s.values())
        if len(frame_times) <= 1:
            return "coincident"
        reference = np.asarray(frame_times[0], dtype=float)
        for candidate in frame_times[1:]:
            arr = np.asarray(candidate, dtype=float)
            if arr.shape != reference.shape or not np.allclose(arr, reference, rtol=0.0, atol=1e-12):
                return "asynchronous"
        return "coincident"

    def metadata_for_microscope(self, microscope_name: str) -> dict[str, Any]:
        frame_times = self.microscope_frame_times_s[microscope_name]
        return {
            "latent_schedule_id": self.schedule_id,
            "state_time_policy": self.policy,
            "observation_times_s": [float(v) for v in frame_times],
            "fusion_time_alignment": self.fusion_time_alignment,
        }


@dataclass(frozen=True)
class SharedLatentScene:
    """One report-level latent physical scene and microscope-specific views."""

    scene_params: dict[str, Any]
    schedule: SharedLatentSchedule
    latent_scene: dict[str, Any]
    provenance_id: str

    def view_for_microscope(self, microscope_name: str) -> dict[str, Any]:
        indices = np.asarray(
            self.schedule.microscope_frame_indices[microscope_name],
            dtype=int,
        )
        trajectories = np.asarray(self.latent_scene["trajectories_nm"], dtype=float)
        view: dict[str, Any] = dict(self.latent_scene)
        view["trajectories_nm"] = trajectories[:, indices, :].copy()
        orientations = self.latent_scene.get("orientations")
        if orientations is not None:
            view["orientations"] = np.asarray(orientations, dtype=float)[:, indices, :, :].copy()
        view["num_frames"] = int(indices.size)
        view["latent_times_s"] = np.asarray(self.schedule.times_s, dtype=float)[indices].copy()
        view["latent_scene_id"] = self.provenance_id
        view["latent_schedule_id"] = self.schedule.schedule_id
        view["shared_coordinate_frame"] = "lab_report_shared_scene_xy_nm"
        return view

    def metadata_for_microscope(self, microscope_name: str) -> dict[str, Any]:
        metadata = self.schedule.metadata_for_microscope(microscope_name)
        metadata.update(
            {
                "latent_scene_id": self.provenance_id,
                "same_latent_scene": True,
                "shared_coordinate_frame": "lab_report_shared_scene_xy_nm",
            }
        )
        return metadata


def build_shared_latent_schedule(
    resolved_params_by_microscope: Mapping[str, Mapping[str, Any]],
) -> SharedLatentSchedule:
    """Build the report-level union of microscope observation times."""

    if not resolved_params_by_microscope:
        raise ValueError("Cannot build a shared latent schedule without resolved microscopes.")

    time_by_key: dict[str, float] = {}
    microscope_keys: dict[str, list[str]] = {}
    microscope_frame_times: dict[str, np.ndarray] = {}
    for microscope_name, params in resolved_params_by_microscope.items():
        frame_times = _frame_center_times_s(params)
        microscope_frame_times[str(microscope_name)] = frame_times
        keys: list[str] = []
        for time_s in frame_times:
            key = _rounded_time_key(float(time_s))
            time_by_key.setdefault(key, float(time_s))
            keys.append(key)
        microscope_keys[str(microscope_name)] = keys

    ordered_keys = sorted(time_by_key, key=lambda key: time_by_key[key])
    times = np.asarray([time_by_key[key] for key in ordered_keys], dtype=float)
    index_by_key = {key: idx for idx, key in enumerate(ordered_keys)}
    microscope_frame_indices = {
        microscope_name: np.asarray([index_by_key[key] for key in keys], dtype=int)
        for microscope_name, keys in microscope_keys.items()
    }
    return SharedLatentSchedule(
        times_s=times,
        microscope_frame_indices=microscope_frame_indices,
        microscope_frame_times_s=microscope_frame_times,
    )


def build_shared_latent_scene(
    scene_params: Mapping[str, Any],
    resolved_params_by_microscope: Mapping[str, Mapping[str, Any]],
) -> SharedLatentScene:
    """Simulate one latent physical scene for a lab Fisher report."""

    schedule = build_shared_latent_schedule(resolved_params_by_microscope)
    params = deepcopy(dict(scene_params))
    latent_scene = _simulate_latent_scene_at_times(params, schedule.times_s)
    provenance = scene_provenance_from_params(params)
    payload = {
        "scene_provenance": provenance,
        "schedule_id": schedule.schedule_id,
        "latent_times_s": [float(v) for v in schedule.times_s],
        "particle_count": int(np.asarray(latent_scene["trajectories_nm"]).shape[0]),
    }
    encoded = json.dumps(
        json_safe(payload, nonfinite="string"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    provenance_id = "latent-scene-" + hashlib.sha256(encoded).hexdigest()[:16]
    return SharedLatentScene(
        scene_params=params,
        schedule=schedule,
        latent_scene=latent_scene,
        provenance_id=provenance_id,
    )


__all__ = [
    "SharedLatentSchedule",
    "SharedLatentScene",
    "build_shared_latent_schedule",
    "build_shared_latent_scene",
]
