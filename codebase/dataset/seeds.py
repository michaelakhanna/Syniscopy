"""Dataset seed derivation."""

from __future__ import annotations

from typing import Optional

from stochastic_runtime import derive_seed

def _derive_video_seed(
    random_seed: Optional[int],
    video_index: int,
    *,
    entropy: Optional[int] = None,
) -> int:
    """
    Derive a stable NumPy-compatible seed from the dataset seed and video index.

    The seed must not depend on resume batch size or iteration offset. Otherwise
    re-running an interrupted dataset with a smaller ``num_videos`` can reuse a
    seed for multiple target indices.
    """
    return derive_seed(
        None if random_seed is None else int(random_seed),
        stream="dataset_video",
        index=int(video_index),
        entropy=entropy,
        bits=31,
    )
