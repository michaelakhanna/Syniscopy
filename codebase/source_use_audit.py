"""Regenerable source-use audit cards for Syniscopy native-regime references."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SOURCE_USE_AUDIT_SCHEMA_VERSION = "syniscopy-source-use-audit-v1"


@dataclass(frozen=True)
class SourceUseAuditCard:
    row_id: str
    modality: str
    citation_key: str
    row_role: str = "citation_provenance"
    source_category: str = ""
    source_doi_or_url: str = ""
    source_reported_quantity: str = ""
    source_reported_value: str = ""
    source_reported_scale: str = ""
    computed_quantity: str = "n/a"
    computed_value: str = "n/a"
    units: str = ""
    unit_conversion: str = "none"
    uncertainty: str = "not_reported"
    extracted_parameters: dict[str, Any] | None = None
    missing_or_defaulted_parameters: tuple[str, ...] = ()
    match_category: str = "modality_principle_citation_only"
    match_score: float | None = None
    parameter_match_status: str = "not_a_computed_comparison"
    match_status: str = "not_a_computed_comparison"
    mismatch_reason: str = "source does not provide all fields needed for quote-matched localization validation"
    computation_method: str = "metadata_card_only"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SOURCE_USE_AUDIT_SCHEMA_VERSION
        d["extracted_parameters"] = dict(self.extracted_parameters or {})
        d["provenance_hash"] = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return d


def manifest_from_cards(cards: Iterable[SourceUseAuditCard | dict[str, Any]]) -> dict[str, Any]:
    rows = [card.to_dict() if isinstance(card, SourceUseAuditCard) else dict(card) for card in cards]
    ids = [str(row.get("row_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source-use audit row_id values must be unique")
    return {"schema_version": SOURCE_USE_AUDIT_SCHEMA_VERSION, "row_count": len(rows), "rows": rows}


def write_source_use_audit_outputs(cards: Iterable[SourceUseAuditCard | dict[str, Any]], json_path: str | Path, csv_path: str | Path) -> dict[str, Any]:
    manifest = manifest_from_cards(cards)
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "row_id", "modality", "citation_key", "row_role", "source_category", "source_doi_or_url",
        "source_reported_quantity", "source_reported_value", "source_reported_scale",
        "computed_quantity", "computed_value", "units", "unit_conversion",
        "uncertainty", "extracted_parameters", "missing_or_defaulted_parameters",
        "match_category", "match_score", "parameter_match_status", "match_status",
        "mismatch_reason", "computation_method",
        "provenance_hash",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in manifest["rows"]:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key, "")
                for key in fields
            })
    return manifest
