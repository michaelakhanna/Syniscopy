"""
dataset_schema.py — helpers for Syniscopy generated-dataset annotations.

These helpers are intentionally generic: they describe the simulator output
contract and do not depend on any one trainer. Downstream notebooks, training
starters, and local tools can use them to select one of the exported
supervision targets without reimplementing the dataset layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_MASK_TARGETS = ("mask_supported", "mask_geometry")

ANNOTATION_SCHEMA_VERSION = "syniscopy-supervision-v1"
ANNOTATION_TARGET_DESCRIPTIONS = {
    "mask_geometry": "projected object and contrast-support mask before support-factor gating",
    "mask_supported": "mask after configured support-factor gating",
    "ignore_mask": "object pixels unsupported for selected supervision",
    "loss_weight": "uint8 0..255 continuous per-pixel loss weight",
}


def build_annotation_schema(
    *,
    selected_target: str | None = None,
    support_factors: Iterable[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "targets": dict(ANNOTATION_TARGET_DESCRIPTIONS),
    }
    if selected_target is not None:
        schema["selected_target"] = validate_supervision_target(selected_target)
    if support_factors is not None:
        schema["support_factors"] = [str(f) for f in support_factors]
    return schema


@dataclass(frozen=True)
class FrameAnnotationPaths:
    video_id: str
    frame_index: int
    particle_index: int
    target_mask: Path | None
    ignore_mask: Path | None
    loss_weight: Path | None


def validate_supervision_target(target: str) -> str:
    target = str(target)
    if target not in SUPPORTED_MASK_TARGETS:
        raise ValueError(
            f"supervision target must be one of {SUPPORTED_MASK_TARGETS}; got {target!r}."
        )
    return target


def _particle_folder(particle_index: int) -> str:
    return f"particle_{int(particle_index) + 1}"


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def mask_path_for(
    mask_root: str | Path,
    particle_index: int,
    frame_index: int,
    *,
    target: str = "mask_supported"
) -> Path | None:
    """Return the positive-mask path for a particle/frame and target."""
    target = validate_supervision_target(target)
    root = Path(mask_root)
    filename = f"frame_{int(frame_index):04d}.png"
    return _first_existing([
        root / target / _particle_folder(particle_index) / filename,
    ])


def ignore_mask_path_for(
    mask_root: str | Path,
    particle_index: int,
    frame_index: int,
) -> Path | None:
    root = Path(mask_root)
    filename = f"frame_{int(frame_index):04d}.png"
    return _first_existing([
        root / "ignore_mask" / _particle_folder(particle_index) / filename,
    ])


def loss_weight_path_for(
    mask_root: str | Path,
    particle_index: int,
    frame_index: int,
) -> Path | None:
    root = Path(mask_root)
    filename = f"frame_{int(frame_index):04d}.png"
    return _first_existing([
        root / "loss_weight" / _particle_folder(particle_index) / filename,
    ])


def annotation_paths_for_frame(
    *,
    video_id: str,
    mask_root: str | Path,
    particle_index: int,
    frame_index: int,
    target: str = "mask_supported"
) -> FrameAnnotationPaths:
    """Return target/ignore/loss-weight paths for one particle/frame."""
    return FrameAnnotationPaths(
        video_id=str(video_id),
        frame_index=int(frame_index),
        particle_index=int(particle_index),
        target_mask=mask_path_for(
            mask_root,
            particle_index,
            frame_index,
            target=target,
        ),
        ignore_mask=ignore_mask_path_for(mask_root, particle_index, frame_index),
        loss_weight=loss_weight_path_for(mask_root, particle_index, frame_index),
    )
