"""CLI and compatibility wrapper for Syniscopy dataset generation."""

from __future__ import annotations

import argparse
import logging

from dataset import (
    apply_parameter_overrides,
    build_dataset_video_params,
    generate_dataset,
    get_dataset_preset_names,
    get_default_dataset_params,
    write_default_params_template,
)
from json_utils import load_typed_json


logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure dataset-generation logging for command-line entry points."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    logger.setLevel(level)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate datasets from Syniscopy simulation parameters."
    )
    parser.add_argument("--num-videos", "--num_videos", dest="num_videos", type=int, default=1)
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="Dataset preset. Public options: " + ", ".join(get_dataset_preset_names()),
    )
    parser.add_argument("--instrument", type=str, default=None)
    parser.add_argument("--output", "--output_dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--params-json", "--params_json", dest="params_json", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--composition-json", "--composition_json", type=str, default=None)
    parser.add_argument("--no-resume", "--no_resume", dest="no_resume", action="store_true")
    parser.add_argument("--append-on-config-change", "--append_on_config_change", dest="append_on_config_change", action="store_true")
    parser.add_argument("--write-params-template", "--write_params_template", dest="write_params_template", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(verbose=args.verbose)
    if args.write_params_template:
        path = write_default_params_template(args.write_params_template)
        logger.info("Wrote PARAMS template to %s", path)
        return

    composition = None
    if args.composition_json:
        composition = load_typed_json(
            args.composition_json,
            expected=list,
            context="--composition_json",
        )

    param_overrides = None
    if args.params_json:
        param_overrides = load_typed_json(
            args.params_json,
            expected=dict,
            context="--params_json",
        )

    generate_dataset(
        num_videos=args.num_videos,
        preset_name=args.preset,
        instrument_preset=args.instrument,
        base_output_dir=args.output_dir,
        random_seed=args.seed,
        param_overrides=param_overrides,
        composition=composition,
        resume_existing=not args.no_resume,
        reset_existing=args.reset,
        append_on_config_change=args.append_on_config_change,
        verbose=args.verbose,
    )


__all__ = [
    "apply_parameter_overrides",
    "build_dataset_video_params",
    "configure_logging",
    "generate_dataset",
    "get_dataset_preset_names",
    "get_default_dataset_params",
    "main",
    "write_default_params_template",
]


if __name__ == "__main__":
    main()
