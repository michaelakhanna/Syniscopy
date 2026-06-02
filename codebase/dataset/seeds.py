"""Dataset seed derivation."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

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
    if random_seed is None:
        if entropy is None:
            entropy = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        payload = f"unseeded:{int(entropy)}:{int(video_index)}"
    else:
        payload = f"seeded:{int(random_seed)}:{int(video_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)
