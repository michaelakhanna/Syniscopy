"""Pytest collection wrapper for generated large-section validation scripts."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
LARGE_SECTION_SCRIPTS = tuple(
    sorted(
        path
        for pattern in ("L*.py", "M*.py", "V*.py")
        for path in SCRIPT_DIR.glob(pattern)
    )
)


@pytest.mark.parametrize(
    "script_path",
    LARGE_SECTION_SCRIPTS,
    ids=[path.stem for path in LARGE_SECTION_SCRIPTS],
)
def test_large_section_script(script_path: Path) -> None:
    """Run each standalone validation script as a pytest test case."""

    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)]
    try:
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code in (None, 0):
                return
            raise AssertionError(f"{script_path.name} exited with status {code!r}") from exc
    finally:
        sys.argv = old_argv
