#!/usr/bin/env python3
"""Generate a parameter-exposure matrix for physics/workflow/hidden audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

import config
from param_schema import PARAM_SCHEMA


WORKFLOW_CONTROL_KEYS = {
    "channels",
    "duration_seconds",
    "fps",
    "num_frames",
    "random_seed",
    "output_filename",
    "mask_output_directory",
    "mask_generation_enabled",
    "mask_max_area_fraction",
    "multichannel_output_mode",
    "multichannel_sidecar_directory",
    "return_ideal_float_frames",
    "save_frame_sequence",
    "save_raw_camera_frame_sequence",
    "save_raw_camera_video",
    "save_raw_frame_views",
}


def _summarize_payload(payload: dict[str, Any]) -> str:
    """Return a compact markdown summary of the current matrix."""
    counts = payload["counts"]
    by_tier: dict[str, list[str]] = defaultdict(list)
    for row in payload["rows"]:
        by_tier[row["exposure_tier"]].append(row["param_key"])

    sections = [
        "# Parameter Exposure Summary",
        "",
        f"Generated (UTC): `{payload['generated_utc']}`",
        "",
        "## Totals",
        "",
        "- Params: " + str(counts["params"]),
        "- Schema controls: " + str(counts["schema_controls"]),
        "- Internal/runtime keys: " + str(counts["internal"]),
        "",
        "- Decision (core+advanced): " + str(
            counts["exposure_tier"].get("core", 0) + counts["exposure_tier"].get("advanced", 0)
        ),
        "- Workflow: " + str(counts["exposure_tier"].get("workflow", 0)),
        "- Hidden: " + str(counts["exposure_tier"].get("hidden", 0)),
        "",
        "## Tiered parameter lists",
        "",
    ]

    for tier in ("hidden", "workflow", "core", "advanced"):
        keys = sorted(by_tier.get(tier, []))
        sections.append(f"### {tier.title()} ({len(keys)})")
        if keys:
            sections.append("")
            for key in keys:
                row = next(
                    (item for item in payload["rows"] if item["param_key"] == key),
                    None,
                )
                if row is None:
                    continue
                surface = row.get("surface", "unknown")
                notes = row.get("notes", "")
                sections.append(f"- `{key}` ({surface}) — {notes}")
        else:
            sections.append("")
            sections.append("- _none_")
        sections.append("")

    sections.extend(
        [
            "## Policy-aligned interpretation",
            "",
            "- **Hidden** keys are strictly runtime-only and should never be user-set.",
            "- **Workflow** keys control output and framing, not optical/numeric physics.",
            "- **Decision** keys are physics + bench-facing controls used for comparative setup selection.",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def _schema_key_by_base() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, spec in PARAM_SCHEMA.items():
        out[spec["key"]].append({"schema_key": key, **spec})
    return out


def _is_advanced_group(group: str) -> bool:
    return "Advanced" in str(group)


def _is_workflow_group(group: str) -> bool:
    return str(group).strip().lower() == "workflow"


def _is_workflow_param(param_key: str, specs: list[dict[str, Any]] | None = None) -> bool:
    if param_key in WORKFLOW_CONTROL_KEYS:
        return True
    if specs is None:
        return False
    return any(
        _is_workflow_group(str(spec.get("group", ""))) or str(spec.get("key", "")) in WORKFLOW_CONTROL_KEYS
        for spec in specs
    )


def _record_for_param(
    param_key: str,
    *,
    in_params: bool,
    schema_specs: dict[str, list[dict[str, Any]]],
    internal_keys: set[str],
) -> dict[str, Any]:
    if param_key in internal_keys:
        return {
            "param_key": param_key,
            "in_params": in_params,
            "in_schema": bool(schema_specs.get(param_key)),
            "schema_keys": [spec["schema_key"] for spec in schema_specs.get(param_key, [])],
            "schema_groups": [spec["group"] for spec in schema_specs.get(param_key, [])],
            "exposure_tier": "hidden",
            "surface": "internal",
            "notes": "Derived/runtime key intentionally excluded from user surfaces.",
        }

    specs = schema_specs.get(param_key)
    if specs:
        groups = [str(spec.get("group", "")) for spec in specs]
        if _is_workflow_param(param_key, specs):
            exposure = "workflow"
        elif any(_is_advanced_group(g) for g in groups):
            exposure = "advanced"
        else:
            exposure = "core"
        return {
            "param_key": param_key,
            "in_params": in_params,
            "in_schema": True,
            "schema_keys": [spec["schema_key"] for spec in specs],
            "schema_groups": groups,
            "exposure_tier": exposure,
            "surface": "schema-control",
            "notes": "User-settable through schema-defined control surface.",
        }

    if param_key in WORKFLOW_CONTROL_KEYS:
        return {
            "param_key": param_key,
            "in_params": in_params,
            "in_schema": False,
            "schema_keys": [],
            "schema_groups": [],
            "exposure_tier": "workflow",
            "surface": "params-json / recipe override",
            "notes": "Workflow/output control not part of schema surface.",
        }

    return {
        "param_key": param_key,
        "in_params": in_params,
        "in_schema": False,
        "schema_keys": [],
        "schema_groups": [],
        "exposure_tier": "advanced",
        "surface": "params-json / recipe override",
        "notes": "Tunable by PARAMS override; not currently in schema controls.",
    }


def build_matrix() -> dict[str, Any]:
    internal_keys = set(config.KNOWN_INTERNAL_PARAM_KEYS)
    params_keys = set(config.PARAMS.keys())
    schema_by_base = _schema_key_by_base()

    rows = []

    for key in sorted(params_keys | internal_keys):
        rows.append(_record_for_param(key, in_params=key in params_keys, schema_specs=schema_by_base, internal_keys=internal_keys))

    # add schema-only controls that are intentionally mapped onto PARAMS keys
    for schema_key, specs in PARAM_SCHEMA.items():
        base = specs["key"]
        if schema_key != base and all(r["param_key"] != schema_key for r in rows):
            rows.append(
                {
                    "param_key": schema_key,
                    "in_params": False,
                    "in_schema": True,
                    "schema_keys": [schema_key],
                    "schema_groups": [specs["group"]],
                    "exposure_tier": "workflow"
                    if _is_workflow_param(base)
                    or _is_workflow_group(specs["group"])
                    else ("advanced" if _is_advanced_group(specs["group"]) else "core"),
                    "surface": "schema-control",
                    "notes": "Schema alias for an object field inside PARAMS['particles'].",
                }
            )

    rows.sort(key=lambda item: (item["exposure_tier"], item["param_key"]))
    tiers = [r["exposure_tier"] for r in rows]

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "params": len(config.PARAMS),
            "schema_controls": len(PARAM_SCHEMA),
            "internal": len(internal_keys),
            "rows": len(rows),
            "exposure_tier": {k: tiers.count(k) for k in sorted(set(tiers))},
        },
        "rows": rows,
    }


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate parameter exposure matrix JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "param_exposure_matrix.json",
        help="Output JSON path for matrix artifact.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Optional output markdown summary path. When omitted, no summary file is "
            "written."
        ),
    )
    args = parser.parse_args()

    payload = build_matrix()
    _write_json(payload, args.output)
    if args.summary is not None:
        summary_path = args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_summarize_payload(payload), encoding="utf-8")
    print(str(args.output))
    if args.summary is not None:
        print(str(args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
