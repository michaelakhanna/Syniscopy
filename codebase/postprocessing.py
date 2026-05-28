import os
import logging
from contextlib import contextmanager

import numpy as np
import cv2
from tqdm import tqdm

RAW_BACKGROUND_SUBTRACTION_METHODS = {
    "none",
    "raw",
    "raw_signal",
    "off",
    "disabled",
    "no_subtraction",
}
VIDEO_BACKGROUND_SUBTRACTION_METHODS = {"video_median"}
REFERENCE_BACKGROUND_SUBTRACTION_METHODS = {"reference_frame"}

logger = logging.getLogger(__name__)
_RELATIVE_REFERENCE_FLOOR = 1e-9


@contextmanager
def _suppress_opencv_videoio_warnings():
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging is None or not hasattr(logging, "getLogLevel") or not hasattr(logging, "setLogLevel"):
        yield
        return
    previous_level = logging.getLogLevel()
    try:
        logging.setLogLevel(logging.LOG_LEVEL_SILENT)
        yield
    finally:
        logging.setLogLevel(previous_level)


def _background_subtraction_method(params) -> str:
    if params is None:
        params = {}
    return str(params.get("background_subtraction_method", "video_median")).strip().lower()


def _uses_relative_reference_contrast(params) -> bool:
    if params is None:
        params = {}
    imaging_model_name = str(params.get("imaging_model", "bright_field")).strip().lower()
    from imaging_model import modality_uses_relative_reference_contrast

    return modality_uses_relative_reference_contrast(imaging_model_name)


def _uses_phase_contrast_units(params) -> bool:
    if params is None:
        params = {}
    imaging_model_name = str(params.get("imaging_model", "bright_field")).strip().lower()
    from imaging_model import get_imaging_model_class

    return getattr(get_imaging_model_class(imaging_model_name), "output_type", "intensity") == "phase"


def _phase_display_count_scale(params) -> float:
    if params is None:
        params = {}
    scale = float(
        params.get(
            "qpi_phase_to_count_scale",
            params.get("background_intensity", 100.0),
        )
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "qpi_phase_to_count_scale must be positive and finite when converting "
            "QPI detector-count frames back to phase contrast."
        )
    return scale


def _frame_as_float(frame, *, name: str, index: int | None = None) -> np.ndarray:
    label = name if index is None else f"{name}[{index}]"
    source = np.asarray(frame)
    dtype = np.float64 if np.issubdtype(source.dtype, np.floating) else np.float32
    arr = np.asarray(frame, dtype=dtype)
    if arr.ndim < 2:
        raise ValueError(f"{label} must be at least 2D; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return arr


def _frame_as_float32(frame, *, name: str, index: int | None = None) -> np.ndarray:
    """Compatibility wrapper for display paths that explicitly need float32."""
    arr = _frame_as_float(frame, name=name, index=index)
    return np.asarray(arr, dtype=np.float32)


def _require_same_shape(
    signal_f: np.ndarray,
    reference_f: np.ndarray,
    *,
    index: int | None = None,
) -> None:
    if signal_f.shape != reference_f.shape:
        suffix = "" if index is None else f" at frame {index}"
        raise ValueError(
            "signal and reference frame shapes must match for reference_frame "
            f"background subtraction{suffix}; got {signal_f.shape} and "
            f"{reference_f.shape}."
        )


def _contrast_stack(contrast_frames) -> np.ndarray:
    arrays = [
        _frame_as_float(frame, name="contrast_frames", index=idx)
        for idx, frame in enumerate(contrast_frames)
    ]
    if not arrays:
        raise ValueError(
            "contrast_frames must be non-empty when computing a normalization range."
        )
    first_shape = arrays[0].shape
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape != first_shape:
            raise ValueError(
                "All contrast frames must have the same shape for normalization; "
                f"frame 0 has {first_shape}, frame {idx} has {arr.shape}."
            )
    return np.stack(arrays, axis=0)


def _relative_reference_denominator(reference_frame: np.ndarray) -> np.ndarray:
    return np.maximum(reference_frame, _RELATIVE_REFERENCE_FLOOR)




def compute_contrast_frames(signal_frames, reference_frames, params):
    """
    Compute floating-point contrast frames before 8-bit display normalization.

    Raw/no-subtraction methods preserve detector-domain signal in a floating
    dtype. Floating-point simulation frames keep their precision so weak
    ideal-frame signals are not rounded away before Fisher analysis.
    """
    if signal_frames is None or len(signal_frames) == 0:
        return []

    method = _background_subtraction_method(params)

    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        return [
            _frame_as_float(frame, name="signal_frames", index=idx).copy()
            for idx, frame in enumerate(signal_frames)
        ]

    contrast_frames = []

    if method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        if reference_frames is None or len(reference_frames) == 0:
            raise ValueError(
                "reference_frames must be provided when background_subtraction_method "
                "is 'reference_frame'."
            )
        if len(reference_frames) != len(signal_frames):
            raise ValueError(
                "reference_frame background subtraction requires the same number "
                f"of signal and reference frames; got {len(signal_frames)} and "
                f"{len(reference_frames)}."
            )

        use_relative = _uses_relative_reference_contrast(params)
        use_phase_units = _uses_phase_contrast_units(params)
        if use_relative:
            logger.info(
                "Applying background subtraction using per-frame reference images "
                "(relative contrast: (S-R)/R)..."
            )
        elif use_phase_units:
            logger.info(
                "Applying background subtraction using per-frame reference images "
                "(phase contrast: (S-R)/phase_display_count_scale)..."
            )
        else:
            logger.info(
                "Applying background subtraction using per-frame reference images "
                "(additive contrast: S-R)..."
            )

        for idx, (signal_frame, ref_frame) in enumerate(tqdm(
            zip(signal_frames, reference_frames),
            total=len(signal_frames),
            disable=not logger.isEnabledFor(logging.INFO),
        )):
            signal_f = _frame_as_float(signal_frame, name="signal_frames", index=idx)
            ref_f = _frame_as_float(ref_frame, name="reference_frames", index=idx)
            _require_same_shape(signal_f, ref_f, index=idx)
            if use_relative:
                subtracted = (signal_f - ref_f) / _relative_reference_denominator(ref_f)
            elif use_phase_units:
                subtracted = (signal_f - ref_f) / _phase_display_count_scale(params)
            else:
                subtracted = signal_f - ref_f
            contrast_frames.append(subtracted)

    elif method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        logger.info(
            "Applying background subtraction using temporal median of signal "
            "frames (video_median method)..."
        )

        num_frames = len(signal_frames)
        frame_shape = np.asarray(signal_frames[0]).shape
        stack_dtype = np.float64 if np.issubdtype(np.asarray(signal_frames[0]).dtype, np.floating) else np.float32
        signal_stack = np.empty((num_frames, *frame_shape), dtype=stack_dtype)
        for idx, frame in enumerate(signal_frames):
            frame_arr = _frame_as_float(frame, name="signal_frames", index=idx)
            if frame_arr.shape != frame_shape:
                raise ValueError(
                    "All signal frames must have the same shape for video_median "
                    f"background subtraction; frame 0 has {frame_shape}, frame "
                    f"{idx} has {frame_arr.shape}."
                )
            signal_stack[idx] = frame_arr

        background_frame = np.median(signal_stack, axis=0).astype(stack_dtype, copy=False)

        for frame in tqdm(
            signal_frames,
            total=num_frames,
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            subtracted = np.asarray(frame, dtype=stack_dtype) - background_frame
            contrast_frames.append(subtracted)

        for idx, frame in enumerate(contrast_frames):
            median_val = float(np.median(frame))
            if median_val != 0.0:
                contrast_frames[idx] = frame - median_val

    else:
        raise ValueError(
            f"Unknown background_subtraction_method: {method}. "
            "Supported values are raw/no-subtraction methods, 'reference_frame', "
            "and 'video_median'."
        )

    return contrast_frames




def compute_single_frame_contrast(signal_frame, reference_frame, params):
    """
    Compute one floating-point contrast frame.

    Raw/no-subtraction methods do not require a reference frame and preserve
    floating simulation precision.
    """
    if signal_frame is None:
        raise ValueError("signal_frame must be provided for single-frame contrast.")

    signal_f = _frame_as_float(signal_frame, name="signal_frame")
    method = _background_subtraction_method(params)

    if method in RAW_BACKGROUND_SUBTRACTION_METHODS:
        return signal_f.copy()

    if method in VIDEO_BACKGROUND_SUBTRACTION_METHODS:
        median_val = float(np.median(signal_f))
        return signal_f - median_val

    if reference_frame is None:
        raise ValueError(
            "reference_frame must be provided unless background_subtraction_method "
            "is one of the raw/no-subtraction methods."
        )

    reference_f = _frame_as_float(reference_frame, name="reference_frame")
    _require_same_shape(signal_f, reference_f)

    if method in REFERENCE_BACKGROUND_SUBTRACTION_METHODS:
        if _uses_relative_reference_contrast(params):
            contrast = (signal_f - reference_f) / _relative_reference_denominator(reference_f)
        elif _uses_phase_contrast_units(params):
            contrast = (signal_f - reference_f) / _phase_display_count_scale(params)
        else:
            contrast = signal_f - reference_f
        return contrast

    raise ValueError(
        f"Unknown background_subtraction_method for single-frame contrast: {method}."
    )




def compute_single_frame_views(signal_frame, reference_frame, params):
    """
    Compute the public raw, contrast, and display views for a single frame.

    This routes through compute_single_frame_contrast so every public
    single-frame helper uses the same result key names as multi-frame
    simulation output. For ``video_median`` background subtraction, a
    single-frame call uses that frame's scalar median because no temporal
    stack is available.
    """
    if signal_frame is None:
        raise ValueError("signal_frame must be provided for single-frame views.")

    signal_f = _frame_as_float(signal_frame, name="signal_frame")
    reference_f = None if reference_frame is None else _frame_as_float(
        reference_frame,
        name="reference_frame",
    )
    contrast = compute_single_frame_contrast(signal_f, reference_f, params)

    final_frame_8bit = normalize_contrast_frames([contrast], signal_f.shape)[0]

    return {
        "raw_signal_frame": signal_f,
        "raw_reference_frame": reference_f,
        "contrast_frame": contrast,
        "final_frame_8bit": final_frame_8bit,
    }



def compute_normalization_range(contrast_frames):
    """
    Compute the 0.5 and 99.5 percentile display window over all contrast frames.
    """
    stack = _contrast_stack(contrast_frames)
    min_val, max_val = np.percentile(stack, [0.5, 99.5])
    return float(min_val), float(max_val)


def normalize_contrast_frames(contrast_frames, original_frame_shape):
    """
    Normalize contrast frames to an 8-bit [0, 255] range using global
    percentile-based windowing.

    This helper implements intensity windowing and normalization:

        1. Determine the 0.5 and 99.5 percentile values across the entire set
           of contrast frames. This robustly defines the minimum and maximum
           interesting signal.
        2. Normalize each contrast frame to [0, 255] using these values,
           clipping out-of-range values.
        3. In the degenerate case where max_val <= min_val, return a stack of
           constant mid-gray frames (value 128) with the same shape as the
           original images.

    Args:
        contrast_frames (list of np.ndarray): List of floating-point contrast
            frames.
        original_frame_shape (tuple[int, int]): Shape (height, width) of the
            original frames, used to construct replacement frames in the
            degenerate case.

    Returns:
        list of np.ndarray: List of uint8 frames normalized to [0, 255].
    """
    if contrast_frames is None or len(contrast_frames) == 0:
        return []

    stack = _contrast_stack(contrast_frames)
    min_val, max_val = compute_normalization_range(stack)

    display_frames = []
    if max_val > min_val:
        for frame in stack:
            norm_frame = 255 * (frame - min_val) / (max_val - min_val)
            display_frames.append(
                np.clip(norm_frame, 0, 255).astype(np.uint8)
            )
    else:
        display_frames = [
            np.full(original_frame_shape, 128, dtype=np.uint8)
            for _ in stack
        ]

    return display_frames


def apply_background_subtraction(signal_frames, reference_frames, params):
    """
    Compute the selected contrast view and normalize it to an 8-bit range for
    video encoding.

    This function is the public entry point for post-processing and is factored
    into two steps:

        1. compute_contrast_frames(...):
               Computes floating-point contrast frames according to the selected
               background subtraction method. Supported groups are
               reference-frame contrast, video-median contrast, and raw
               detector-domain output without subtraction.
        2. normalize_contrast_frames(...):
               Applies percentile-based intensity windowing (0.5 and 99.5
               percentiles) and maps the contrast frames into 8-bit [0, 255].

    Behavior:

        - If signal_frames is empty, returns an empty list.
        - For method "reference_frame" with missing reference_frames, raises
          ValueError.
        - For raw/no-subtraction methods, reference_frames is ignored.
        - For unknown methods, raises ValueError.
        - Otherwise, returns a list of 8-bit frames ready for video encoding.

    Args:
        signal_frames (list of np.ndarray): List of raw signal frames
            (typically uint16).
        reference_frames (list of np.ndarray): List of raw reference frames
            (typically uint16). Required for 'reference_frame' method.
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain "background_subtraction_method".

    Returns:
        list of np.ndarray: List of 8-bit, normalized frames ready for video
            encoding. Returns an empty list only when signal_frames is empty.
    """
    if signal_frames is None or len(signal_frames) == 0:
        return []

    # Step 1: compute floating-point contrast frames via the selected method.
    contrast_frames = compute_contrast_frames(signal_frames, reference_frames, params)
    if not contrast_frames:
        return []

    # Step 2: normalize contrast frames to 8-bit for video encoding.
    original_shape = signal_frames[0].shape
    display_frames = normalize_contrast_frames(contrast_frames, original_shape)

    return display_frames


def save_video(filename, frames, fps, size=None, *, color_order="rgb"):
    """
    Encode frames to an AVI preview file and fail loudly if unavailable.

    Inter-frame codecs visibly smear noisy microscopy frames. Syniscopy
    therefore writes AVI only, using intraframe MJPG for broadly readable
    preview movies. The dataset generator also writes lossless PNG frame
    sequences, which are the canonical training/inference artifact.

    Grayscale frames are converted to BGR before writing, and RGB frames are
    converted to BGR unless color_order="bgr" is explicitly set.

    Args:
        filename (str): Output path ending in .avi.
        frames (sequence[np.ndarray]): 2D uint8 grayscale or 3D uint8 RGB/BGR frames.
        fps (int | float): Positive frame rate.
        size (tuple[int, int] | None): Optional (width, height). If omitted,
            the size is inferred from the first frame.
        color_order (str): "rgb" or "bgr" for 3-channel frames. Grayscale frames
            ignore this value.

    Raises:
        ValueError: malformed frames, frame-size mismatch, or invalid fps.
        RuntimeError: video backend could not open or produced no nonempty file.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("save_video requires at least one frame.")

    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive; got {fps}.")

    filename = os.path.abspath(os.path.expanduser(str(filename)))
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".avi":
        raise ValueError(
            "Syniscopy writes AVI preview videos only. Use an output filename "
            "ending in '.avi'; lossless PNG frame sequences are the canonical "
            "training/inference artifact."
        )

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    order = str(color_order).strip().lower()
    if order not in {"rgb", "bgr"}:
        raise ValueError("color_order must be 'rgb' or 'bgr'.")

    def _prepare_frame(frame, expected_size=None):
        arr = np.asarray(frame)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Video frames must be 2D grayscale or 3D color arrays; got shape {arr.shape}.")
        if arr.ndim == 3 and arr.shape[2] != 3:
            raise ValueError(f"Color video frames must have exactly 3 channels; got shape {arr.shape}.")

        if arr.dtype != np.uint8:
            arr = arr.astype(float, copy=False)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Cannot save non-finite video data; "
                    "frames must not contain NaN or Inf."
                )
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.ascontiguousarray(arr)

        height, width = arr.shape[:2]
        actual_size = (int(width), int(height))
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                f"Frame size mismatch: expected {expected_size}, got {actual_size}."
            )

        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif order == "rgb":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = np.ascontiguousarray(arr)
        return bgr, actual_size

    if size is not None:
        if len(size) != 2:
            raise ValueError("size must be a (width, height) tuple when provided.")
        resolved_size = (int(size[0]), int(size[1]))
        if resolved_size[0] <= 0 or resolved_size[1] <= 0:
            raise ValueError(f"size values must be positive; got {resolved_size}.")
        first_frame, _ = _prepare_frame(frames[0], expected_size=resolved_size)
    else:
        first_frame, resolved_size = _prepare_frame(frames[0], expected_size=None)

    prepared_frames = [first_frame]
    for frame in frames[1:]:
        prepared, _ = _prepare_frame(frame, expected_size=resolved_size)
        prepared_frames.append(prepared)

    codec_candidates = ("MJPG",)

    logger.info("Saving final video to %s...", filename)
    failures: list[str] = []
    for candidate in codec_candidates:
        with _suppress_opencv_videoio_warnings():
            writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*candidate),
                fps,
                resolved_size,
                True,
            )
        if not writer.isOpened():
            writer.release()
            failures.append(f"{candidate}: writer did not open")
            continue

        try:
            for prepared in prepared_frames:
                writer.write(prepared)
        finally:
            writer.release()

        if not os.path.exists(filename) or os.path.getsize(filename) <= 0:
            failures.append(f"{candidate}: writer produced no nonempty file")
            continue

        logger.info(
            "Using intraframe MJPG AVI preview; PNG frame sequences remain the canonical lossless artifact."
        )
        logger.info("Simulation finished successfully!")
        return

    raise RuntimeError(
        f"Could not write a nonempty AVI preview for {filename!r}; "
        f"fps={fps}, size={resolved_size}, failures={failures!r}."
    )
