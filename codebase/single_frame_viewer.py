import argparse
import copy
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import cv2
import numpy as np

from config import PARAMS
from main import generate_single_frame_views
from param_utils import build_params_from_controls
from postprocessing import compute_single_frame_contrast


def _resolve_output_dir(output_dir_arg: str | None) -> str:
    """
    Resolve and create the output directory for the single-frame viewer.

    If output_dir_arg is provided, it is used directly. Otherwise a
    project-relative default directory is used.
    """
    if output_dir_arg:
        out_dir = os.path.abspath(os.path.expanduser(output_dir_arg))
    else:
        out_dir = os.path.join(
            "outputs",
            "syniscopy_single_frame",
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _compute_center_position_nm(params_viewer: dict) -> tuple[float, float]:
    """
    Compute the (x, y) center position in nanometers for the given parameter
    dictionary based on image_size_pixels and pixel_size_nm.
    """
    image_size_pixels = float(params_viewer["image_size_pixels"])
    pixel_size_nm = float(params_viewer["pixel_size_nm"])
    L_nm = image_size_pixels * pixel_size_nm
    x_center_nm = 0.5 * L_nm
    y_center_nm = 0.5 * L_nm
    return x_center_nm, y_center_nm


def _tailor_params_for_single_centered_particle(params_base: dict, output_dir: str) -> dict:
    """
    Create a tailored parameter dictionary for a single, centered, static
    particle observed in exactly one frame, leaving the rest of the pipeline
    unchanged.

    This function:
        - Starts from params_base overlaid on the central parameter schema to
          obtain a consistent base.
        - Forces exactly one frame by adjusting duration_seconds.
        - Ensures a single canonical particle object.
        - Places the particle near the center in x and y.
        - Places the particle at a fixed z below focus with unconstrained
          z-motion.
        - Disables masks, trackability, and motion blur.
        - Redirects all outputs into the provided output_dir.
        - Uses a background subtraction method that is meaningful for a
          single noisy frame.
    """
    schema_base = build_params_from_controls(control_values={})
    params = copy.deepcopy(schema_base)
    params.update(copy.deepcopy(params_base))

    fps = float(params.get("fps", 24.0))
    if fps <= 0.0:
        fps = 24.0
        params["fps"] = fps

    duration_seconds = 1.0 / fps
    params["duration_seconds"] = duration_seconds

    base_particles = params.get("particles", [])
    first_particle = base_particles[0] if base_particles else {}
    first_component = (first_particle.get("components") or [{}])[0]
    default_diameter = float(first_component.get("diameter_nm", 100.0))
    default_material = first_component.get("material", "Gold")
    default_multiplier = float(first_particle.get("signal_multiplier", 1.0))

    # The default viewer renders an empty-background single particle, so disable
    # substrate-pattern rendering and lateral substrate exclusion.
    params["sample_environment_pattern_enabled"] = False
    params["sample_environment_pattern"] = "none"
    params["sample_environment_pattern_preset"] = "empty_background"

    x_init_nm, y_init_nm = _compute_center_position_nm(params)

    # Use the configured z_stack_range_nm to choose a z that is safely within
    # the PSF stack but below focus. For the viewer, we adopt the convention
    # that negative z is "below focus" and use one quarter of the range.
    z_stack_range_nm = float(params.get("z_stack_range_nm", 30500.0))
    z_initial_nm = -0.25 * z_stack_range_nm

    # Use unconstrained z-motion so the preview can place the particle below
    # focus without reflecting it at z = 0.
    params["z_motion_constraint_model"] = "unconstrained"

    params["particles"] = [
        {
            "name": "preview_particle",
            "motion": {
                "hydrodynamic_diameter_nm": default_diameter,
                "initial_position_nm": [x_init_nm, y_init_nm, z_initial_nm],
            },
            "signal_multiplier": default_multiplier,
            "components": [
                {
                    "shape": "sphere",
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": default_diameter,
                    "material": default_material,
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                    "material_properties": None,
                }
            ],
        }
    ]

    params["mask_generation_enabled"] = False
    params["supervision_temporal_support_enabled"] = False
    params["motion_blur_enabled"] = False
    params["motion_blur_subsamples"] = 1

    params["background_subtraction_method"] = "reference_frame"

    # Leave detector-noise controls as configured in the schema-controlled base.

    video_path = os.path.join(output_dir, "single_frame.avi")
    params["output_filename"] = video_path

    # Masks are disabled, but the main pipeline still expects this path.
    params["mask_output_directory"] = os.path.join(output_dir, "masks")

    os.makedirs(os.path.dirname(params["output_filename"]), exist_ok=True)
    os.makedirs(params["mask_output_directory"], exist_ok=True)

    return params


def _tailor_existing_params_for_preview(params_base: dict, output_dir: str) -> dict:
    """
    Adapt an already-resolved PARAMS dictionary for a one-frame preview.

    Unlike ``_tailor_params_for_single_centered_particle``, this preserves the
    caller's optical, noise, substrate, and particle settings. It only forces a
    one-frame render and redirects outputs into ``output_dir``.
    """
    params = copy.deepcopy(params_base)

    fps = float(params.get("fps", 24.0))
    if fps <= 0.0:
        fps = 24.0
        params["fps"] = fps
    params["duration_seconds"] = 1.0 / fps
    params["num_frames"] = 1

    params["mask_generation_enabled"] = False
    params["supervision_temporal_support_enabled"] = False
    params["motion_blur_enabled"] = False
    params["motion_blur_subsamples"] = 1
    params["background_subtraction_method"] = "reference_frame"

    params["output_filename"] = os.path.join(output_dir, "single_frame.avi")
    params["mask_output_directory"] = os.path.join(output_dir, "masks")

    os.makedirs(os.path.dirname(params["output_filename"]), exist_ok=True)
    os.makedirs(params["mask_output_directory"], exist_ok=True)

    return params


def _complex_to_json(obj: complex) -> dict[str, float]:
    """
    Convert a complex number into a JSON-friendly dict representation.
    """
    return {"real": float(obj.real), "imag": float(obj.imag)}


def _numpy_to_native(obj: Any) -> Any:
    """
    Convert NumPy scalar/array types to plain Python types and lists so that
    the structure becomes JSON-serializable.

    - np.ndarray -> list
    - np.generic scalars -> Python scalars
    - complex numbers -> {"real": float, "imag": float}
    """
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return np.vectorize(
                lambda v: _complex_to_json(complex(v)),
                otypes=[object],
            )(obj).tolist()
        return obj.tolist()

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, complex):
        return _complex_to_json(obj)

    if is_dataclass(obj):
        return _numpy_to_native(asdict(obj))

    return obj


def _make_params_json_serializable(params: dict) -> dict:
    """
    Recursively convert a parameter dictionary containing NumPy arrays,
    NumPy scalars, and complex numbers into a JSON-serializable structure.

    Complex numbers are represented as {"real": ..., "imag": ...}.
    """
    def convert(value: Any) -> Any:
        # Handle dicts
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}

        # Handle sequences (lists/tuples)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]

        # First normalize NumPy / complex types to native.
        native = _numpy_to_native(value)

        # After normalization, complex numbers will have been converted already.
        if isinstance(native, dict):
            # Could be a pre-existing dict that should itself be converted.
            return {str(k): convert(v) for k, v in native.items()}

        return native

    return convert(params)


def _dump_params_to_json(params: dict, output_path: str) -> None:
    """
    Serialize the given parameter dictionary (after JSON-ification) to the
    specified JSON file path.
    """
    serializable_params = _make_params_json_serializable(params)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_params, f, indent=2, sort_keys=True)


def _save_frame_as_png(frame: np.ndarray, path: str) -> None:
    """
    Save a single 2D numeric frame as an 8-bit PNG, applying per-frame
    min/max scaling when necessary.

    If frame is already uint8, it is written as-is. For floating-point or
    higher-bit-depth integer arrays, the data are linearly mapped to [0, 255]
    based on their own min/max to ensure a visible dynamic range.
    """
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        img = arr
    else:
        arr_float = arr.astype(float)
        vmin = np.min(arr_float)
        vmax = np.max(arr_float)
        if vmax > vmin:
            norm = (arr_float - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr_float, dtype=float)
        img = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    if not cv2.imwrite(path, img):
        raise IOError(f"Failed to write image to {path}")


def save_single_frame_preview(
    params: dict,
    output_dir: str,
    seed: int | None = None,
) -> dict[str, str]:
    """
    Render and save a one-frame preview from an existing PARAMS dictionary.

    The supplied parameter dictionary is preserved except for preview-only
    single-frame settings. The function saves PNG views and returns their
    output paths.
    """
    output_dir = _resolve_output_dir(output_dir)
    if seed is not None:
        np.random.seed(seed)

    params_preview = _tailor_existing_params_for_preview(params, output_dir)
    views = generate_single_frame_views(params_preview)

    outputs: dict[str, str] = {}
    view_to_name = {
        "raw_signal_frame": "single_frame_raw_signal.png",
        "raw_reference_frame": "single_frame_reference.png",
        "contrast_frame": "single_frame_contrast.png",
        "final_frame_8bit": "single_frame_final.png",
    }

    for key, filename in view_to_name.items():
        frame = views.get(key)
        if frame is None:
            continue
        path = os.path.join(output_dir, filename)
        _save_frame_as_png(frame, path)
        outputs[key] = path

    params_json_path = os.path.join(output_dir, "params_used.json")
    _dump_params_to_json(views.get("params_resolved", params_preview), params_json_path)
    outputs["params_json"] = params_json_path

    return outputs


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the single-frame viewer script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single Syniscopy preview frame using the "
            "existing simulation pipeline, and save multiple 2D view images "
            "along with the resolved parameter dictionary."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory in which to store the single-frame video and PNG views "
            "plus parameter JSON. Defaults to outputs/syniscopy_single_frame."
        ),
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a PARAMS dictionary to preview. "
            "This is the same PARAMS dictionary accepted by dataset_generator.py."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for NumPy to make the viewer run "
            "reproducible."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Render and save the single-frame preview outputs.

    The output directory receives raw signal, reference, contrast, final 8-bit,
    and resolved-parameter files for the preview frame.
    """
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as fh:
            params_loaded = json.load(fh)
        if not isinstance(params_loaded, dict):
            raise ValueError("--params_json must contain a JSON object.")
        params_viewer = _tailor_existing_params_for_preview(params_loaded, output_dir)
    else:
        # Tailor params for a single near-centered particle and a single frame,
        # starting from the schema-controlled PARAMS base.
        params_viewer = _tailor_params_for_single_centered_particle(PARAMS, output_dir)

    # Generate all in-memory views using the viewer-core function.
    views = generate_single_frame_views(params_viewer)

    raw_signal = views.get("raw_signal_frame", None)
    raw_reference = views.get("raw_reference_frame", None)
    contrast = views.get("contrast_frame", None)
    final_8bit = views.get("final_frame_8bit", None)
    params_resolved = views.get("params_resolved", params_viewer)

    # --- Save raw signal and reference views ---
    if raw_signal is not None:
        raw_signal_path = os.path.join(output_dir, "single_frame_raw_signal.png")
        _save_frame_as_png(raw_signal, raw_signal_path)
        print(f"Raw signal image: {raw_signal_path}")
    else:
        print("Raw signal frame was not produced; skipping raw signal export.")

    if raw_reference is not None:
        raw_reference_path = os.path.join(output_dir, "single_frame_reference.png")
        _save_frame_as_png(raw_reference, raw_reference_path)
        print(f"Raw reference image: {raw_reference_path}")
    else:
        print("Raw reference frame was not produced; skipping raw reference export.")

    # --- Save contrast view ---
    if contrast is None and raw_signal is not None and raw_reference is not None:
        try:
            contrast = compute_single_frame_contrast(raw_signal, raw_reference, params_resolved)
        except ValueError as exc:
            print(f"Could not compute single-frame contrast: {exc}")

    if contrast is not None:
        contrast_path = os.path.join(output_dir, "single_frame_contrast.png")
        _save_frame_as_png(contrast, contrast_path)
        print(f"Contrast image: {contrast_path}")
    else:
        print("Skipping contrast export due to missing contrast frame.")

    # --- Save the final postprocessed 8-bit frame ---
    if final_8bit is not None:
        final_image_path = os.path.join(output_dir, "single_frame_final.png")
        _save_frame_as_png(final_8bit, final_image_path)
        print(f"Final processed image: {final_image_path}")
    else:
        print("No final 8-bit frame was produced; skipping final PNG export.")

    # Dump the fully resolved parameter dictionary after the run.
    params_json_path = os.path.join(output_dir, "params_used.json")
    _dump_params_to_json(params_resolved, params_json_path)
    print(f"Resolved parameters: {params_json_path}")

    print("Single-frame viewer run complete.")


if __name__ == "__main__":
    main()
