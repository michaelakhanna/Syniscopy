#!/usr/bin/env python3
"""CLI wrapper for the lab Fisher report package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_package_entrypoints():
    package_dir = Path(__file__).with_name("lab_fisher_report")
    spec = importlib.util.spec_from_file_location(
        "_syniscopy_lab_fisher_report",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load lab_fisher_report package from {package_dir!s}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main, module.run_report


def _load_cli_module():
    package_dir = Path(__file__).with_name("lab_fisher_report")
    spec = importlib.util.spec_from_file_location(
        "_syniscopy_lab_fisher_report_cli",
        package_dir / "cli.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load lab_fisher_report CLI parser from {package_dir!s}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cli_parser():
    return _load_cli_module()._parse_args


def _validated_microscope_set_for_cli(args, path):
    from lab_fisher_report.microscopes import load_microscope_set, resolve_microscope_params
    from lab_fisher_report.params_assembly import _make_microscope_base_and_shared_params

    microscope_set = load_microscope_set(path)
    base_params, shared_params = _make_microscope_base_and_shared_params(
        args,
        microscope_shared_params=microscope_set.shared_params,
    )
    resolved = {
        scope.name: resolve_microscope_params(
            scope,
            base_params,
            shared_params=shared_params,
        )
        for scope in microscope_set.microscopes
    }
    return microscope_set, resolved


def _handle_lightweight_cli() -> int | None:
    cli = _load_cli_module()
    args = cli._parse_args()

    if args.list_modalities:
        from modality_registry import SUPPORTED_MODALITIES

        for modality in SUPPORTED_MODALITIES:
            print(modality)
        return 0
    if args.list_instruments:
        from presets import get_instrument_preset_names

        for name in sorted(get_instrument_preset_names()):
            print(name)
        return 0
    if args.write_template:
        path = cli._write_microscope_set_template(
            args.write_template,
            modality=args.template_modality,
        )
        print(f"Wrote lab Fisher microscope template: {path}")
        return 0
    if args.validate_microscopes:
        microscope_set, resolved = _validated_microscope_set_for_cli(
            args,
            args.validate_microscopes,
        )
        print(
            "Validated resolved microscope set: "
            f"{len(microscope_set.microscopes)} microscope(s), "
            f"{len(microscope_set.shared_params)} shared param(s), "
            f"{len(resolved)} resolved parameters record(s)."
        )
        return 0
    if args.list_microscopes:
        microscope_set, _validated = _validated_microscope_set_for_cli(
            args,
            args.list_microscopes,
        )
        for scope in microscope_set.microscopes:
            instrument = scope.instrument or "-"
            print(f"{scope.name}\t{scope.modality}\t{instrument}")
        return 0
    return None


def main() -> int:
    lightweight_result = _handle_lightweight_cli()
    if lightweight_result is not None:
        return lightweight_result
    # Keep every public CLI path in the package coordinator. The coordinator owns
    # report generation and generated modality sweeps. CLI-only paths above call
    # the same parser/template/microscope modules without importing the render
    # stack, so source inspection and template generation stay lightweight.
    package_main, _run_report = _load_package_entrypoints()
    return package_main()


def run_report(*args, **kwargs):
    _package_main, package_run_report = _load_package_entrypoints()
    return package_run_report(*args, **kwargs)

__all__ = ["main", "run_report"]


if __name__ == "__main__":
    raise SystemExit(main())
