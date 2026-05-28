#!/usr/bin/env python3
"""Simple local entry point for generating a Syniscopy training dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_RECIPE = "recipes/default.py:DEFAULT"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a local Syniscopy dataset from an explicit microscope "
            "recipe. Recipes expose bench and sample parameters for dataset "
            "generation."
        )
    )
    parser.add_argument(
        "--output",
        default="datasets/syniscopy_dataset",
        help="Output dataset directory. Default: datasets/syniscopy_dataset.",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of videos to generate. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Dataset random seed. Default: 12345.",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Optional instrument preset name from codebase/presets.py.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON object with PARAMS overrides.",
    )
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        help=(
            "Python recipe in the form path.py:NAME. "
            f"Default: {DEFAULT_RECIPE}."
        ),
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="List editable recipes from the default public recipe and exit.",
    )
    parser.add_argument(
        "--write-params-template",
        default=None,
        help=(
            "Write the public editable recipe parameters to JSON and exit. "
            "This writes recipe-facing dataset keys rather than every "
            "simulator runtime parameter."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output directory before generation.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate requested videos even if matching outputs exist.",
    )
    parser.add_argument(
        "--append-on-config-change",
        action="store_true",
        help=(
            "Append a new batch if the output directory already contains a "
            "dataset made with different parameters. By default this is an "
            "error; use --reset to replace the directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print dataset-generation progress while frames and masks are written.",
    )
    return parser.parse_args()


def _load_recipe(recipe_spec: str) -> dict:
    if ":" in recipe_spec:
        file_part, name = recipe_spec.split(":", 1)
    else:
        file_part, name = recipe_spec, "DEFAULT"
    path = Path(file_part).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {path}")
    spec = importlib.util.spec_from_file_location("syniscopy_dataset_recipe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import recipe file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, name):
        available = [
            k for k, v in vars(module).items()
            if k.isupper() and isinstance(v, dict) and k != "OPTIONS"
        ]
        raise KeyError(f"Recipe {name!r} not found in {path}. Available: {available}")
    value = getattr(module, name)
    if not isinstance(value, dict):
        raise TypeError(f"Recipe {name!r} in {path} must be a dictionary.")
    return dict(value)


def _json_safe(value: Any) -> Any:
    """Convert recipe values into JSON-serializable primitives."""
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _list_recipes() -> None:
    file_part, _ = DEFAULT_RECIPE.split(":", 1)
    path = REPO_ROOT / file_part
    spec = importlib.util.spec_from_file_location("syniscopy_dataset_recipe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import recipe file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in vars(module).items():
        if key.isupper() and isinstance(value, dict) and key != "OPTIONS":
            print(key)


def main() -> int:
    args = _parse_args()
    if args.list_recipes:
        _list_recipes()
        return 0
    if args.write_params_template:
        template = _load_recipe(args.recipe or DEFAULT_RECIPE)
        out = Path(args.write_params_template).expanduser()
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(template), fh, indent=2, sort_keys=True)
        print(f"Wrote public dataset parameter template: {out}")
        return 0

    recipe_overrides = None
    user_overrides = None
    if args.recipe:
        recipe_overrides = _load_recipe(args.recipe)
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as fh:
            user_overrides = json.load(fh)
        if not isinstance(user_overrides, dict):
            raise ValueError("--params-json must contain one JSON object.")

    from dataset_generator import generate_dataset

    dataset_dir = generate_dataset(
        num_videos=args.num_videos,
        preset_name="default",
        instrument_preset=args.instrument,
        base_output_dir=args.output,
        random_seed=args.seed,
        recipe_overrides=recipe_overrides,
        param_overrides=user_overrides,
        resume_existing=not args.no_resume,
        reset_existing=args.reset,
        append_on_config_change=args.append_on_config_change,
        verbose=args.verbose,
    )
    print(f"Dataset ready: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
