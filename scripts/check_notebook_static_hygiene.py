"""Static hygiene checks for release-bound notebooks.

This gate intentionally does not execute notebooks. It checks the properties
that should be true of source notebooks before packaging or CI: valid JSON,
cleared outputs/execution counts, no local-user absolute paths, and no stale
unit-control branches that conflict with canonical core metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    *(ROOT / "supplemental").glob("E*.ipynb"),
    *(ROOT / "sam2_starter").glob("*.ipynb"),
]

_POSIX_USER_ROOT = "/" + "Users" + "/"
_POSIX_HOME_RE = "/" + "home" + r"/[^/\s]+/"
_WINDOWS_USER_RE = "C:" + r"\\Users\\"
LOCAL_PATH_RE = re.compile(
    f"({_POSIX_USER_ROOT}|{_POSIX_HOME_RE}|{_WINDOWS_USER_RE})"
)
STALE_UNIT_BRANCH_RE = re.compile(r"units\s*==\s*['\"]radians['\"]")


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def main() -> None:
    failures: list[str] = []
    expected = [ROOT / "supplemental" / f"E{i:02d}.ipynb" for i in range(1, 10)]
    expected.extend(sorted((ROOT / "sam2_starter").glob("*.ipynb")))
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    if missing:
        failures.append(f"missing notebooks: {missing}")

    for path in sorted(set(NOTEBOOKS)):
        rel = path.relative_to(ROOT)
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{rel}: invalid notebook JSON: {type(exc).__name__}: {exc}")
            continue
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = _cell_source(cell)
            if cell.get("execution_count") is not None:
                failures.append(f"{rel}: code cell {idx} has saved execution_count")
            if cell.get("outputs"):
                failures.append(f"{rel}: code cell {idx} has saved outputs")
            if LOCAL_PATH_RE.search(src):
                failures.append(f"{rel}: code cell {idx} contains a local-user absolute path")
            if STALE_UNIT_BRANCH_RE.search(src):
                failures.append(f"{rel}: code cell {idx} branches on stale units == 'radians'")

    for notebook_name in ("E03.ipynb", "E04.ipynb"):
        notebook_path = ROOT / "supplemental" / notebook_name
        if not notebook_path.exists():
            continue
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        joined = "\n".join(_cell_source(cell) for cell in nb.get("cells", []) if cell.get("cell_type") == "code")
        call_count = joined.count("compare_modality_information_content_detected_quanta_normalized(")
        strict_count = joined.count("require_detected_count_images=True")
        if call_count and strict_count < call_count:
            failures.append(
                f"supplemental/{notebook_name}: detected-quanta normalized calls must pass "
                "require_detected_count_images=True"
            )
        if notebook_name == "E03.ipynb":
            for required in (
                "fusion_subset_metadata",
                "profile_card_paper_use_category",
                "detected_count_distribution_rule",
                "count_mean_source",
            ):
                if required not in joined:
                    failures.append(f"supplemental/E03.ipynb: missing paper-gate metadata hook {required!r}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Notebook static hygiene OK for {len(set(NOTEBOOKS))} notebooks.")


if __name__ == "__main__":
    main()
