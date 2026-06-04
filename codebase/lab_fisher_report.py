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


def main() -> int:
    cli_module = _load_cli_module()
    args = cli_module._parse_args()
    if args.write_template:
        path = cli_module._write_template(args.write_template)
        print(f"Wrote lab Fisher template: {path}")
        return 0
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
    package_main, _run_report = _load_package_entrypoints()
    return package_main()


def run_report(*args, **kwargs):
    _package_main, package_run_report = _load_package_entrypoints()
    return package_run_report(*args, **kwargs)

__all__ = ["main", "run_report"]


if __name__ == "__main__":
    raise SystemExit(main())
