#!/usr/bin/env python3
"""Rebuild reviewed Liverpool caustic clips from the DataCat 50nm download.

The public Syniscopy source tree ships the reviewed clip manifest and audit
metadata, but not the redistributed AVI clip payload. Download the DataCat
``50nm/`` folder, then run this script to regenerate the local reviewed clips
needed by the prompt-manifest and transfer-inference notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2


DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
DEFAULT_MANIFEST_REL = Path("data/liverpool_caustic_50nm_review/selected_clip_manifest.json")


def find_supplemental_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    if DRIVE_MYDRIVE.exists():
        candidates.extend([DRIVE_MYDRIVE / "supplemental", DRIVE_MYDRIVE / "SyniscopySupplemental"])
    candidates.extend([Path("/content/supplemental"), Path("/content/SyniscopySupplemental")])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "supplemental" / "E01.ipynb").exists():
            resolved = resolved / "supplemental"
        if (resolved / "E01.ipynb").exists():
            return resolved
    raise RuntimeError("Syniscopy supplemental folder not found.")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips", data) if isinstance(data, dict) else data
    if not isinstance(clips, list):
        raise TypeError(f"review manifest must contain a list of clips: {path}")
    return [dict(item) for item in clips]


def raw_root_candidates(supplemental_root: Path, explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_root = os.environ.get("SYNISCOPY_DATACAT_50NM_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            supplemental_root / "50nm",
            supplemental_root / "data" / "50nm",
            supplemental_root.parent / "50nm",
            supplemental_root.parent / "data" / "50nm",
            Path.cwd() / "50nm",
            Path.cwd() / "data" / "50nm",
        ]
    )
    return candidates


def resolve_source_video(source_video: str, supplemental_root: Path, raw_root: Path | None) -> Path:
    source_rel = Path(source_video)
    candidates: list[Path] = []
    for root in raw_root_candidates(supplemental_root, raw_root):
        if source_rel.parts[:1] == ("50nm",):
            candidates.append(root / Path(*source_rel.parts[1:]))
        candidates.append(root / source_rel)
    candidates.extend([supplemental_root / source_rel, supplemental_root.parent / source_rel])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates[:8])
    raise FileNotFoundError(
        f"Could not resolve source video {source_video!r}. "
        "Download the DataCat 50nm folder and pass --raw-root /path/to/50nm.\n"
        f"Searched:\n  {searched}"
    )


def supplemental_path(supplemental_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:1] == ("supplemental",):
        return supplemental_root.parent / path
    return supplemental_root / path


def write_clip_with_cv2(src: Path, dst: Path, start: int, stop: int, fps: float) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open source video: {src}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 0 else (decoded_fps if decoded_fps > 0 else 5.0)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open output clip: {dst}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    source_idx = start
    while source_idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        source_idx += 1
    writer.release()
    cap.release()
    return written


def rebuild_clips(
    *,
    supplemental_root: Path,
    manifest_path: Path,
    raw_root: Path | None,
    force: bool,
    dry_run: bool,
) -> int:
    rows = load_manifest(manifest_path)
    rebuilt = 0
    skipped = 0
    for idx, row in enumerate(rows):
        source_video = resolve_source_video(str(row["source_video"]), supplemental_root, raw_root)
        clip_path = supplemental_path(supplemental_root, str(row["clip_video"]))
        start = int(row["source_frame_start"])
        stop = int(row["source_frame_end_exclusive"])
        expected = int(row.get("clip_frame_count") or max(0, stop - start))
        fps = float(row.get("fps_metadata") or 5.0)
        if clip_path.exists() and not force:
            print(f"[skip] {idx:02d} exists: {clip_path}")
            skipped += 1
            continue
        print(f"[clip] {idx:02d} {source_video} frames {start}:{stop} -> {clip_path}")
        if dry_run:
            rebuilt += 1
            continue
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        written = write_clip_with_cv2(source_video, clip_path, start, stop, fps)
        if written != expected:
            raise RuntimeError(
                f"Clip {idx:02d} wrote {written} frames, expected {expected}: {clip_path}"
            )
        rebuilt += 1
    print(f"review clip rebuild complete: rebuilt={rebuilt}, skipped={skipped}, total={len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplemental-root", type=Path, help="Path to the supplemental folder.")
    parser.add_argument("--manifest", type=Path, help="Reviewed clip manifest path.")
    parser.add_argument("--raw-root", type=Path, help="Path to the downloaded DataCat 50nm folder.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing reviewed clip AVIs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print planned clips without writing.")
    args = parser.parse_args()

    supplemental_root = find_supplemental_root(args.supplemental_root)
    manifest_path = args.manifest.expanduser() if args.manifest else supplemental_root / DEFAULT_MANIFEST_REL
    if not manifest_path.exists():
        raise FileNotFoundError(f"review manifest not found: {manifest_path}")
    return rebuild_clips(
        supplemental_root=supplemental_root,
        manifest_path=manifest_path,
        raw_root=args.raw_root,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
