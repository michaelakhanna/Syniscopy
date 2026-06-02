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
MASK_LABEL_ENCODING = "per_particle_binary_png_sidecars"
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
        "mask_label_encoding": MASK_LABEL_ENCODING,
        "sidecar_layout": "target_name/particle_N/frame_XXXX.png",
        "coordinate_frame": "final rendered frame grid, row-major yx pixel coordinates",
        "binary_mask_value_encoding": "uint8 PNG with 0=absent and 255=present",
        "loss_weight_value_encoding": "uint8 PNG with 0=ignored/no loss and 1..255=relative positive-pixel weight",
        "selected_target_relationship": (
            "mask_supported must be a subset of mask_geometry; ignore_mask is "
            "mask_geometry AND NOT mask_supported; loss_weight must be positive "
            "only inside the selected positive target and zero inside ignore_mask."
        ),
        "sam2_gt_label_encoding": (
            "SAM2 conversion stores per-frame GT masks as uint8 instance-label "
            "maps with 0=background and particle-specific object IDs 1..255; "
            "overlaps are removed from GT and marked in Ignore."
        ),
        "targets": dict(ANNOTATION_TARGET_DESCRIPTIONS),
        "ignore_mask_semantics": (
            "ignore_mask marks simulated object/support pixels that are not "
            "valid foreground supervision for the selected target; downstream "
            "losses must not treat these pixels as background negatives."
        ),
        "loss_weight_semantics": (
            "loss_weight is a uint8 0..255 positive-target weight map. It is "
            "zero outside selected positive target pixels and must be zeroed "
            "where ignore_mask is positive."
        ),
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
