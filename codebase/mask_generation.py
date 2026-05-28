import os
import cv2
import numpy as np


def _compute_lobe_boundary_radius_pixels(
    contrast_image: np.ndarray,
    center_yx: tuple[int, int],
    outer_ring_count: int = 0,
    tiny_abs: float = 1e-9,
    zero_level_fraction: float = 1e-3,
) -> float:
    """
    Compute the mask boundary radius (in pixels) for a single particle in a
    contrast image, using radial sign changes:

        - The central lobe is the region around the particle center where the
          sign of the contrast matches the sign at the center.
        - Ring boundaries are the radii where the *ring-averaged* contrast
          changes sign after ignoring near-zero rings.
        - ``outer_ring_count=0`` returns the central-lobe boundary, preserving
          the central-lobe mask definition.
        - ``outer_ring_count=N`` includes N additional surrounding lobes/rings
          and returns the next sign-change boundary.
        - If enough sign changes are not detected, the fallback boundary is the
          largest radius whose ring-averaged magnitude remains above a small
          fraction of the central magnitude.

    The algorithm operates entirely in the final image grid so that the mask
    aligns exactly with the rendered contrast image. Because signs are compared
    relative to the center sign, the same definition works for bright-center
    and dark-center contrast conventions.

    Args:
        contrast_image (np.ndarray):
            2D float array of the particle-specific contrast at final
            resolution. Can be in arbitrary units; only relative sign and
            magnitude are used.
        center_yx (tuple[int, int]):
            (cy, cx) integer indices of the particle center in the contrast
            image. These must be consistent with the coordinate mapping used
            in rendering (trajectory_nm / pixel_size_nm).
        outer_ring_count (int):
            Number of complete rings outside the central lobe to include.
            Must be >= 0.
        tiny_abs (float):
            Absolute threshold below which values are treated as zero to avoid
            numerical noise.
        zero_level_fraction (float):
            Fraction of the central ring-averaged magnitude used as a fallback
            cutoff when no sign flip occurs. Must be >0.

    Returns:
        float: Mask boundary radius in pixels (>= 0). If the contrast
        image carries essentially no signal, returns 0.0.
    """
    outer_ring_count = int(outer_ring_count)
    if outer_ring_count < 0:
        raise ValueError("outer_ring_count must be >= 0.")

    img = np.asarray(contrast_image, dtype=float)
    if img.ndim != 2:
        raise ValueError("contrast_image must be a 2D array.")

    H, W = img.shape
    cy, cx = center_yx

    # Clamp center to valid range to avoid index errors at the edges.
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    # Center value and sign.
    center_val = float(img[cy, cx])
    if abs(center_val) < tiny_abs:
        # If the center is near zero, search a small neighborhood for a
        # stronger signal to define the central sign.
        y0 = max(cy - 2, 0)
        y1 = min(cy + 3, H)
        x0 = max(cx - 2, 0)
        x1 = min(cx + 3, W)
        neighborhood = img[y0:y1, x0:x1]
        if neighborhood.size == 0:
            return 0.0
        # Find the pixel with the largest absolute contrast.
        idx_flat = np.argmax(np.abs(neighborhood))
        max_val = float(neighborhood.flat[idx_flat])
        if abs(max_val) < tiny_abs:
            # No meaningful signal in the neighborhood.
            return 0.0
        center_val = max_val

    center_sign = 1.0 if center_val >= 0.0 else -1.0

    # Build a radial map around the center.
    yy, xx = np.indices((H, W))
    dy = yy - cy
    dx = xx - cx
    r_float = np.sqrt(dx * dx + dy * dy)
    r_index = r_float.astype(np.int64)

    # If the image is extremely small, just return zero radius or one pixel.
    if r_index.size == 0:
        return 0.0

    max_ring = int(r_index.max())
    if max_ring == 0:
        # Single-pixel image or everything at center; treat as trivial lobe.
        return 0.0

    flat_vals = img.ravel()
    flat_rings = r_index.ravel()

    # Compute ring-averaged contrast as a function of integer radius.
    counts = np.bincount(flat_rings, minlength=max_ring + 1)
    sum_vals = np.bincount(flat_rings, weights=flat_vals, minlength=max_ring + 1)

    # Avoid division by zero.
    ring_mean = np.zeros(max_ring + 1, dtype=float)
    nonzero = counts > 0
    ring_mean[nonzero] = sum_vals[nonzero] / counts[nonzero]

    # Center ring magnitude.
    ring0_mag = abs(ring_mean[0])
    if ring0_mag < tiny_abs:
        # If the ring-averaged center magnitude is essentially zero but we had
        # a non-zero pixel center, we still treat the central magnitude as the
        # reference.
        ring0_mag = abs(center_val)

    if ring0_mag < tiny_abs:
        # No meaningful signal for defining a lobe.
        return 0.0

    # Threshold for considering a ring to carry significant contrast.
    mag_threshold = zero_level_fraction * ring0_mag

    # Find the sign-change boundary after the requested number of additional
    # rings. A ring is counted by crossing from one significant sign zone into
    # the next; the center lobe boundary is the first crossing.
    r_boundary_index = None
    previous_sign = center_sign
    transitions_seen = 0
    target_transition = outer_ring_count + 1
    for k in range(1, max_ring + 1):
        v = ring_mean[k]
        if abs(v) < mag_threshold:
            # Very small average; treat as effectively zero but not as a sign
            # flip. This avoids numerical ringing from defining mask semantics.
            continue
        current_sign = 1.0 if v >= 0.0 else -1.0
        if current_sign != previous_sign:
            transitions_seen += 1
            if transitions_seen == target_transition:
                r_boundary_index = k
                break
            previous_sign = current_sign

    if r_boundary_index is not None:
        # Place the boundary slightly inside the ring where the sign flips.
        r_boundary = float(r_boundary_index) - 0.5
        if r_boundary < 0.0:
            r_boundary = 0.0
        return r_boundary

    # Fallback: no sign flip detected. Use the largest radius where the
    # ring-averaged magnitude is above threshold.
    significant_indices = np.where(np.abs(ring_mean) >= mag_threshold)[0]
    if significant_indices.size == 0:
        return 0.0

    k_max = int(significant_indices[-1])
    r_boundary = float(k_max) + 0.5
    return max(r_boundary, 0.0)


def _compute_central_lobe_mask_floodfill(
    contrast_image: np.ndarray,
    center_yx: tuple[int, int],
    tiny_abs: float = 1e-9,
    zero_level_fraction: float = 1e-3,
    max_search_radius_px: int | None = None,
) -> np.ndarray:
    """
    Compute a central-lobe binary mask for a particle in a contrast image via
    a 4-connected flood fill from the particle center.

    This is the appropriate algorithm for non-radially-symmetric particles
    (dimers, rod stacks, or other rigid composites) whose central-lobe region is
    elongated or multi-lobed. The ring-averaging boundary helper with
    ``outer_ring_count=0`` assumes radial symmetry and would collapse the
    asymmetry of a composite into a circular mask that under-covers the
    elongated direction and over-covers the short axis.

    Algorithm:
        1. Seed the flood fill at the center pixel. If the center pixel has
           essentially zero contrast, search a small neighborhood for the
           nearby pixel with the largest absolute contrast and use that as the
           seed instead.
        2. The sign of the seed defines the central sign.
        3. BFS outward from the seed: a neighboring pixel is added to the mask
           iff (a) it has the same sign as the seed, (b) its absolute contrast
           is >= zero_level_fraction * |seed_val|, and (c) it is within
           max_search_radius_px of the seed (if specified).
        4. The returned mask covers every pixel connected (4-neighbor) to the
           seed through the same-sign, above-threshold region.

    Args:
        contrast_image (np.ndarray): 2D float contrast image at final resolution.
        center_yx (tuple[int, int]): (cy, cx) integer indices of the particle
            center. If the exact center pixel is zero, a small neighborhood is
            searched for a seed.
        tiny_abs (float): Absolute threshold below which values are treated as
            numerically zero.
        zero_level_fraction (float): Fraction of |seed| below which a pixel is
            considered below the contrast threshold.
        max_search_radius_px (int | None): Hard cap on distance from the seed
            (in pixels). If None, the flood fill extends to the entire image
            (subject to the sign and magnitude conditions).

    Returns:
        np.ndarray: uint8 binary mask (0 or 255) with the same shape as
        contrast_image.
    """
    img = np.asarray(contrast_image, dtype=float)
    if img.ndim != 2:
        raise ValueError("contrast_image must be a 2D array.")

    H, W = img.shape
    cy, cx = center_yx
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    seed_val = float(img[cy, cx])
    if abs(seed_val) < tiny_abs:
        # Relocate the seed to the strongest pixel in a small neighborhood.
        y0 = max(cy - 2, 0)
        y1 = min(cy + 3, H)
        x0 = max(cx - 2, 0)
        x1 = min(cx + 3, W)
        neighborhood = img[y0:y1, x0:x1]
        if neighborhood.size == 0:
            return np.zeros_like(img, dtype=np.uint8)
        idx_flat = int(np.argmax(np.abs(neighborhood)))
        local_y, local_x = np.unravel_index(idx_flat, neighborhood.shape)
        seed_val = float(neighborhood[local_y, local_x])
        if abs(seed_val) < tiny_abs:
            return np.zeros_like(img, dtype=np.uint8)
        cy = y0 + int(local_y)
        cx = x0 + int(local_x)

    seed_sign = 1.0 if seed_val >= 0.0 else -1.0
    mag_threshold = zero_level_fraction * abs(seed_val)
    if mag_threshold < tiny_abs:
        mag_threshold = tiny_abs

    visited = np.zeros((H, W), dtype=bool)
    mask_bool = np.zeros((H, W), dtype=bool)

    # Iterative BFS with a list-as-queue. A pure Python loop is fine here
    # because the flood-fill region is bounded by the central lobe itself.
    queue = [(cy, cx)]
    visited[cy, cx] = True
    mask_bool[cy, cx] = True

    r_cap_sq = None if max_search_radius_px is None else float(max_search_radius_px) ** 2

    while queue:
        y, x = queue.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if r_cap_sq is not None:
                dy = ny - cy
                dx = nx - cx
                if (dy * dy + dx * dx) > r_cap_sq:
                    continue
            v = img[ny, nx]
            if abs(v) < mag_threshold:
                continue
            sign = 1.0 if v >= 0.0 else -1.0
            if sign != seed_sign:
                continue
            mask_bool[ny, nx] = True
            queue.append((ny, nx))

    return (mask_bool.astype(np.uint8) * 255)


def generate_central_lobe_mask(
    contrast_image: np.ndarray,
    center_yx: tuple[int, int],
    outer_ring_count: int = 0,
    use_floodfill: bool = False,
    max_search_radius_px: int | None = None,
    max_area_fraction: float = 0.25,
) -> np.ndarray:
    """
    Generate a binary lobe/ring mask for a particle.

    Two algorithms are available:

    * Ring-averaged sign-flip (default, ``use_floodfill=False``):
      ``mask(y, x) = 1`` iff ``r(y, x) <= r_boundary``, where ``r_boundary`` is
      determined by radial sign-change boundaries. ``outer_ring_count=0``
      means central lobe only; larger values include that many surrounding
      rings before closing the mask boundary. This algorithm assumes radial
      symmetry and is the correct choice for single-sphere particles.

    * 4-connected flood fill (``use_floodfill=True``):
      starts at the center (or the strongest pixel in a small neighborhood),
      and expands to all connected pixels that share the seed's sign and lie
      above a small fraction of the seed's magnitude. This is the correct
      choice for composite particles (dimers, rod stacks, or other rigid shapes)
      whose central-lobe region is elongated or multi-lobed. Using the ring
      average for such a particle produces a roughly circular mask that under-
      covers the elongated direction and over-covers the short axis.

    The default (``use_floodfill=False``) is the canonical radial mask path
    for single-sphere particles when ``outer_ring_count=0``.

    Edge cases:
        - If the contrast image contains no meaningful signal near the center,
          the returned mask is all zeros.
        - If the particle is near the frame boundary, the resulting mask is
          clipped by the image boundaries, which is consistent with the
          finite camera FOV.

    Args:
        contrast_image (np.ndarray):
            2D float array of the per-particle contrast at final resolution.
        center_yx (tuple[int, int]):
            (cy, cx) integer pixel indices of the particle center.
        outer_ring_count (int):
            Number of complete rings outside the central lobe to include.
            Must be >= 0. Ignored by the flood-fill path, which only computes
            the central same-sign connected lobe.
        use_floodfill (bool):
            If True, use the 4-connected flood-fill algorithm. If False
            (default), use the ring-averaged sign-flip algorithm.
        max_search_radius_px (int | None):
            Only used when ``use_floodfill=True``. Hard cap on the flood-fill
            distance from the seed, in pixels. None disables the cap.
        max_area_fraction (float):
            Safety cap for single-particle masks. If the inferred lobe covers
            more than this fraction of the frame, the contrast image is treated
            as unsupported/ambiguous and an empty mask is returned. This prevents
            flat or zero-particle frames from becoming full-frame masks.

    Returns:
        np.ndarray: uint8 binary mask with values 0 or 255 and the same shape
        as ``contrast_image``.
    """
    img = np.asarray(contrast_image, dtype=float)
    if img.ndim != 2:
        raise ValueError("contrast_image must be a 2D array.")

    H, W = img.shape
    cy, cx = center_yx
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    if use_floodfill:
        if int(outer_ring_count) != 0:
            raise ValueError(
                "outer_ring_count is only supported by the radial sign-change mask path."
            )
        mask = _compute_central_lobe_mask_floodfill(
            img, (cy, cx), max_search_radius_px=max_search_radius_px
        )
        if max_area_fraction is not None:
            area_fraction = float(np.count_nonzero(mask)) / float(mask.size or 1)
            if area_fraction > float(max_area_fraction):
                return np.zeros_like(img, dtype=np.uint8)
        return mask

    # Default radially-symmetric path for single-sphere particles.
    r_boundary = _compute_lobe_boundary_radius_pixels(
        img,
        (cy, cx),
        outer_ring_count=int(outer_ring_count),
    )
    if r_boundary <= 0.0:
        return np.zeros_like(img, dtype=np.uint8)

    yy, xx = np.indices((H, W))
    dy = yy - cy
    dx = xx - cx
    r_float = np.sqrt(dx * dx + dy * dy)

    mask_bool = r_float <= r_boundary
    mask = mask_bool.astype(np.uint8) * 255
    if max_area_fraction is not None:
        area_fraction = float(np.count_nonzero(mask)) / float(mask.size or 1)
        if area_fraction > float(max_area_fraction):
            return np.zeros_like(img, dtype=np.uint8)
    return mask


def save_mask(
    mask: np.ndarray,
    base_mask_directory: str,
    particle_index: int,
    frame_index: int,
) -> None:
    """
    Save a single-particle mask image to disk using the established directory
    and filename conventions.

    Directory structure:
        base_mask_directory/
            particle_1/
                frame_0000.png
                frame_0001.png
                ...
            particle_2/
                ...

    Args:
        mask (np.ndarray): The binary mask image to save (uint8, 0 or 255).
        base_mask_directory (str): Root directory for all particle masks.
        particle_index (int): Zero-based particle index.
        frame_index (int): Zero-based frame index.
    """
    particle_dir = os.path.join(base_mask_directory, f"particle_{particle_index + 1}")
    os.makedirs(particle_dir, exist_ok=True)

    filename = os.path.join(particle_dir, f"frame_{frame_index:04d}.png")
    if not cv2.imwrite(filename, mask):
        # cv2.imwrite returns False (does NOT raise) on failure, so a silent
        # failure here would let the rest of the pipeline declare the
        # dataset complete with missing mask files. Surface the failure
        # loudly instead.
        raise IOError(
            f"cv2.imwrite returned False writing mask {filename!r} "
            f"(particle_index={particle_index}, frame_index={frame_index}, "
            f"mask shape={mask.shape}, dtype={mask.dtype}). Check disk "
            f"space, permissions on {particle_dir!r}, and that OpenCV was "
            f"built with PNG codec support."
        )
