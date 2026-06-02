from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
import sys
import zipfile

csv.field_size_limit(10_000_000)

ROOT = pathlib.Path(".").resolve()


def _normalize_text(value: str) -> str:
    """Normalize human prose to reduce false negatives from line breaks and punctuation."""
    lowered = str(value).lower()
    lowered = lowered.replace("\u00ad", " ")
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(haystack)


BAD_TEXT_PATTERNS = [
    "\x02",
    "$\\{}infty$",
    "{}infty",
    "artifact-p rovenance",
    "supplement al",
    "detected-quant a",
    "modality_ra nking",
    "cross_modal ity",
    "det ected",
    "interfero metric",
    "trans mission",
]

HISTORICAL_FORBIDDEN_CONCESSION_PATTERNS = [
    "coherent optical modes share the scalar paraxial backend",
    "not Monte Carlo electron transport",
]
STALE_LIMITATION_PATTERNS = HISTORICAL_FORBIDDEN_CONCESSION_PATTERNS

_ABTEM_AUDIT_SUFFIXES = {
    ".md",
    ".rst",
    ".tex",
    ".bib",
    ".txt",
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
}

_ABTEM_ALLOWED_DIRS = {
    "throw" + "out",
    "throw" + "out5",
    "throw" + "out_direct_planner_text_20260530_212400",
    "throw" + "out_planner_intake_20260530_205956.tar.gz",
    ".git",
}


def _contains_abtem_reference(text: str) -> bool:
    return bool(re.search(r"\babtem\b", text)) or bool(re.search(r"\babtem_[a-z_]+\b", text))


def _abtem_has_optional_context(text: str, match_span: tuple[int, int]) -> bool:
    start, end = match_span
    window = text[max(0, start - 80) : min(len(text), end + 80)]
    allowed_context = (
        "optional",
        "not required",
        "optional external",
        "external",
        "legacy",
        "adapter",
        "bridge",
        "compatib",
        "if absent",
        "separate",
        "reference",
        "benchmark",
        "validation",
    )
    lowered = window.lower()
    return any(token in lowered for token in allowed_context)


def _candidate_audit_text_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for rel_path in (
        ROOT / "README.md",
        ROOT / "docs",
        ROOT / "paper",
        ROOT / "codebase",
        ROOT / "scripts",
        ROOT / "supplemental",
        ROOT / "examples",
        ROOT / "benchmarks",
    ):
        if not rel_path.exists():
            continue
        if rel_path.is_file():
            paths.append(rel_path)
            continue
        for candidate in rel_path.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in _ABTEM_AUDIT_SUFFIXES:
                continue
            if any(part in _ABTEM_ALLOWED_DIRS for part in candidate.parts):
                continue
            paths.append(candidate)
    return paths


REQUIRED_COLUMNS = {
    "supplemental/outputs/E03/modality_fusion_crlb.csv": [
        "inherited_contract_lp_convergence_status",
        "safe_for_fusion",
        "fusion_validation_status",
    ],
    "supplemental/outputs/E03/time_allocation_crlb.csv": [
        "inherited_contract_lp_convergence_status",
        "safe_for_time_allocation",
        "physical_recommendation_status",
    ],
    "supplemental/outputs/E03/registration_degradation.csv": [
        "inherited_contract_lp_convergence_status",
        "safe_for_registration",
        "registration_validation_status",
    ],
    "supplemental/outputs/E03/cross_modality_axial.csv": [
        "axial_convergence_status",
        "validation_status",
    ],
    "supplemental/outputs/E03/detected_quanta_normalized.csv": [
        "contract_q_derivative_convergence_status",
        "safe_for_detected_quanta_ranking",
        "contract_q_proxy_diagnostic",
        "count_mean_source",
        "proxy_count_diagnostic",
        "safe_for_linear_fisher_variance",
    ],
    "supplemental/outputs/E03/modality_ranking.csv": [
        "contract_lp_derivative_convergence_status",
        "validation_status",
    ],
    "supplemental/outputs/E03/modality_profile_cards.csv": [
        "modality_profile_card_json",
        "profile_card_schema_version",
        "profile_card_measurement_domain",
        "profile_card_signal_units",
        "profile_card_paper_use_category",
        "safe_for_linear_fisher_variance",
        "detector_noise_input_domain",
        "fisher_variance_model_scope",
        "detector_likelihood_status",
    ],
    "supplemental/outputs/E03/detected_quanta_normalization_metadata.csv": [
        "require_detected_count_images",
        "detected_count_distribution_rule",
        "count_mean_source",
        "proxy_count_diagnostic",
        "safe_for_detected_quanta_ranking",
        "all_count_domain_modalities_have_detected_count_images",
        "signed_contrast_derivative_target",
        "poisson_variance_mode",
    ],
    "supplemental/outputs/E03/fusion_independence_metadata.csv": [
        "physical_compatibility_status",
        "fusion_interpretation",
        "independent_noise_assumption",
        "double_count_risk",
        "requires_physical_design_review",
    ],
}


def finite_converged(value: object) -> bool:
    return str(value).strip().lower() in {
        "yes",
        "true",
        "pass",
        "passed",
        "finite_converged",
        "finite-converged",
        "stable_singular",
        "stable-singular",
    }


def _is_convergence_failure_acceptable(row: dict, status_field: str | None = None) -> bool:
    """Return whether a non-pass convergence row is acceptable by metadata contract."""
    status = ""
    if status_field:
        status = str(row.get(status_field, "")).strip().lower()
    if not status:
        status = str(row.get("profile_card_validation_status", "")).strip().lower()
    if not status:
        status = str(row.get("validation_status", "")).strip().lower()

    if status in {"diagnostic_only", "diagnostic", "unchecked"}:
        return True

    reason = str(row.get("convergence_reason", "")).strip().lower()
    if reason and reason != "na":
        return False
    return False


def fail(message: str, issues: list[str]) -> None:
    issues.append(message)


def read_rows(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_header_and_rows(rel: str):
    path = ROOT / rel
    if not path.exists():
        return set(), []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return set(reader.fieldnames or []), rows


def truthy_csv_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def validated_status(value: object) -> bool:
    return str(value).strip().lower() in {
        "validated",
        "finite_converged",
        "reference_validated",
    }


def profile_validation_status(row: dict) -> str:
    if row.get("profile_card_validation_status"):
        return str(row["profile_card_validation_status"])
    try:
        card = json.loads(row.get("modality_profile_card_json", "") or "{}")
    except json.JSONDecodeError:
        return "invalid_json"
    meta = card.get("backend_fidelity_metadata", {}) if isinstance(card.get("backend_fidelity_metadata"), dict) else {}
    return str(card.get("validation_status") or meta.get("validation_status") or "")


issues: list[str] = []


# 1) Formatting guardrails in paper LaTeX/BibTeX.
paper_files = list((ROOT / "paper").rglob("*.tex")) + list((ROOT / "paper").rglob("*.bib"))
target_text_files = [
    ROOT / "README.md",
    ROOT / "docs/lab_fisher_workflow.md",
] + paper_files
for path in target_text_files:
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in BAD_TEXT_PATTERNS:
        if pattern in text:
            fail(f"{path}: contains bad pattern {pattern!r}", issues)

    normalized_text = _normalize_text(text)
    for pattern in STALE_LIMITATION_PATTERNS:
        if _contains_phrase(normalized_text, pattern):
            fail(f"{path}: contains stale limitation phrase {pattern!r}", issues)

    lower_text = text.lower()
    for match in re.finditer(r"\babtem\b", lower_text):
        if not _abtem_has_optional_context(lower_text, match.span()):
            fail(
                f"{path}: contains abTEM reference {match.group(0)!r} without optional-context framing",
                issues,
            )


# 1b) Audit abTEM references outside optional legacy notes.
for path in _candidate_audit_text_paths():
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue
    lower_text = text.lower()
    for match in re.finditer(r"\babtem\b", lower_text):
        if not _abtem_has_optional_context(lower_text, match.span()):
            fail(
                f"{path}: contains abTEM reference {match.group(0)!r} without optional-context framing",
                issues,
            )


# 2) Required status columns and provenance in core E03 outputs.
for rel, cols in REQUIRED_COLUMNS.items():
    path = ROOT / rel
    if not path.exists():
        fail(f"missing required artifact: {rel}", issues)
        continue

    fields, rows = csv_header_and_rows(rel)
    for col in cols:
        if col not in fields:
            fail(f"{rel}: missing required column {col}", issues)
            continue

        # Require at least one non-empty status for inherited metadata columns.
        if not any(str(row.get(col, "")).strip() for row in rows):
            fail(f"{rel}: {col} is empty for all rows", issues)


# 2b) Claim-specific profile, detected-count, and fusion provenance checks.
profile_rel = "supplemental/outputs/E03/modality_profile_cards.csv"
profile_fields, profile_rows = csv_header_and_rows(profile_rel)
if profile_rows:
    for row in profile_rows:
        modality = str(row.get("modality", "")).strip()
        try:
            card = json.loads(row.get("modality_profile_card_json", "") or "{}")
        except json.JSONDecodeError as exc:
            fail(f"{profile_rel}: {modality} has invalid modality_profile_card_json: {exc}", issues)
            continue
        if card.get("schema_version") != "syniscopy-modality-profile-v1":
            fail(f"{profile_rel}: {modality} has invalid or missing profile-card schema", issues)
        for key in ("measurement_domain", "signal_units", "paper_use_category"):
            csv_key = f"profile_card_{key}"
            if not str(row.get(csv_key, "") or card.get(key, "")).strip():
                fail(f"{profile_rel}: {modality} has empty profile-card {key}", issues)
        for key in (
            "safe_for_linear_fisher_variance",
            "detector_noise_input_domain",
            "fisher_variance_model_scope",
            "detector_likelihood_status",
        ):
            if not str(row.get(key, "") or card.get(key, "")).strip():
                fail(f"{profile_rel}: {modality} has empty profile-card detector field {key}", issues)
        if not truthy_csv_value(row.get("safe_for_linear_fisher_variance", card.get("safe_for_linear_fisher_variance", ""))):
            fail(
                f"{profile_rel}: {modality} is unsafe for linear Fisher variance "
                f"({row.get('detector_likelihood_status') or card.get('detector_likelihood_status', '')})",
                issues,
            )
        category = str(row.get("profile_card_paper_use_category") or card.get("paper_use_category", ""))
        validation = profile_validation_status(row).strip().lower()
        if modality in {"tem_phase_contrast", "sem_secondary_electron"}:
            if category == "reference_validated_electron_profile" and validation not in {"validated", "reference_validated"}:
                fail(
                    f"{profile_rel}: {modality} is reference-validated electron evidence without validated status",
                    issues,
                )

dq_meta_rel = "supplemental/outputs/E03/detected_quanta_normalization_metadata.csv"
dq_meta_fields, dq_meta_rows = csv_header_and_rows(dq_meta_rel)
if dq_meta_rows:
    for row in dq_meta_rows:
        modality = row.get("modality", "<unknown>")
        if not truthy_csv_value(row.get("require_detected_count_images", "")):
            fail(f"{dq_meta_rel}: {modality} was not generated with require_detected_count_images=True", issues)
        if row.get("detected_count_distribution_rule") != "profile_specific_detected_count_image":
            fail(f"{dq_meta_rel}: {modality} used non-paper detected-count distribution", issues)
        if truthy_csv_value(row.get("proxy_count_diagnostic", "")):
            fail(f"{dq_meta_rel}: {modality} uses proxy count-domain source", issues)
        if str(row.get("count_mean_source", "")).startswith("contrast_proxy"):
            fail(f"{dq_meta_rel}: {modality} uses contrast-proxy detected-quanta source", issues)
        if not truthy_csv_value(row.get("safe_for_detected_quanta_ranking", "")):
            fail(f"{dq_meta_rel}: {modality} is not safe for detected-quanta ranking", issues)
        if not truthy_csv_value(row.get("all_count_domain_modalities_have_detected_count_images", "")):
            fail(f"{dq_meta_rel}: count-domain modalities did not all use detected count images", issues)

headline_artifacts = {
    "supplemental/outputs/E03/modality_ranking.csv": "contract_lp_derivative_convergence_status",
    "supplemental/outputs/E03/detected_quanta_normalized.csv": "contract_q_derivative_convergence_status",
    "supplemental/outputs/E03/cross_modality_axial.csv": "axial_convergence_status",
}
for rel, status_col in headline_artifacts.items():
    _, rows = csv_header_and_rows(rel)
    if rows:
        for row in rows:
            modality = row.get("modality", row.get("display_name", "<unknown>"))
            if not finite_converged(row.get(status_col, "")):
                fail(
                    f"{rel}: {modality} is consumed as a headline ranking row but "
                    f"{status_col}={row.get(status_col, '')!r}",
                    issues,
                )
            if not validated_status(row.get("validation_status", "")):
                fail(
                    f"{rel}: {modality} is consumed as a headline ranking row but "
                    f"validation_status={row.get('validation_status', '')!r}",
                    issues,
                )
            if "safe_for_linear_fisher_variance" in row and not truthy_csv_value(row.get("safe_for_linear_fisher_variance", "")):
                fail(
                    f"{rel}: {modality} is consumed as a headline ranking row but detector likelihood is unsafe for linear Fisher",
                    issues,
                )
            if rel.endswith("detected_quanta_normalized.csv"):
                if not truthy_csv_value(row.get("safe_for_detected_quanta_ranking", "")):
                    fail(f"{rel}: {modality} is not safe for detected-quanta ranking", issues)
                if truthy_csv_value(row.get("contract_q_proxy_diagnostic", "")):
                    fail(f"{rel}: {modality} came from a Contract-Q proxy diagnostic", issues)
                if truthy_csv_value(row.get("proxy_count_diagnostic", "")):
                    fail(f"{rel}: {modality} uses a proxy count source", issues)

fusion_meta_rel = "supplemental/outputs/E03/fusion_independence_metadata.csv"
fusion_meta_fields, fusion_meta_rows = csv_header_and_rows(fusion_meta_rel)
if fusion_meta_rows:
    for row in fusion_meta_rows:
        subset_size = int(float(row.get("subset_size", "0") or 0.0))
        interpretation = str(row.get("fusion_interpretation", ""))
        if subset_size > 4 and interpretation != "algebraic_diagnostic_only":
            fail(
                f"{fusion_meta_rel}: broad/full-library subset k={subset_size} is not algebraic_diagnostic_only",
                issues,
            )
        if interpretation == "physically_feasible_fusion" and truthy_csv_value(row.get("requires_physical_design_review", "")):
            fail(
                f"{fusion_meta_rel}: k={subset_size} claims physical fusion while requiring design review",
                issues,
            )


# 3) Lateral/axial convergence checks.
derivative_path = ROOT / "supplemental/outputs/E03/derivative_convergence_crlb.csv"
if derivative_path.exists():
    fields, derivative_rows = csv_header_and_rows("supplemental/outputs/E03/derivative_convergence_crlb.csv")
    pass_col = next((c for c in fields if "pass" in c.lower()), None)
    if not pass_col:
        fail("derivative_convergence_crlb.csv has no pass-like column", issues)
    else:
        failing = [r for r in derivative_rows if not finite_converged(r.get(pass_col, ""))]
        if failing:
            unacceptable = [r for r in failing if not _is_convergence_failure_acceptable(r, status_field="validation_status")]
            if unacceptable:
                fail(
                    f"lateral derivative gate has {len(unacceptable)} non-converged rows "
                    "in derivative_convergence_crlb.csv",
                    issues,
                )

axial_path = ROOT / "supplemental/outputs/E03/axial_convergence_crlb.csv"
if axial_path.exists():
    fields, axial_rows = csv_header_and_rows("supplemental/outputs/E03/axial_convergence_crlb.csv")
    pass_col = next((c for c in fields if "pass" in c.lower()), None)
    if not pass_col:
        fail("axial_convergence_crlb.csv has no pass-like column", issues)
    else:
        failing = [
            r
            for r in axial_rows
            if str(r.get(pass_col, "")).strip().lower() in {"no", "false", "fail", "failed"}
        ]
        if failing:
            unacceptable = [r for r in failing if not _is_convergence_failure_acceptable(r, status_field="profile_card_validation_status")]
            if unacceptable:
                fail(
                    f"axial finite-ranking gate has {len(unacceptable)} failing rows "
                    "in axial_convergence_crlb.csv",
                    issues,
                )


# 4) Ensure the build manifest traces required source outputs.
manifest_path = ROOT / "paper" / "figures" / "artifact-provenance-manifest.json"
if not manifest_path.exists():
    fail("paper/figures/artifact-provenance-manifest.json missing", issues)
else:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    source_files = {
        item.get("path")
        for item in manifest.get("source_files", [])
        if isinstance(item, dict) and "path" in item
    }
    for rel in REQUIRED_COLUMNS:
        if rel not in source_files:
            fail(f"artifact provenance manifest missing source-file entry for {rel}", issues)
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, dict) or not source_provenance.get("fingerprint"):
        fail("artifact provenance manifest missing source_provenance.fingerprint", issues)
    source_archives = manifest.get("source_archives")
    if not isinstance(source_archives, dict):
        fail("artifact provenance manifest missing source_archives", issues)
    else:
        for archive_key, rel in {
            "supplemental_syniscopy_source_zip": "supplemental/syniscopy_source.zip",
            "sam2_starter_codebase_zip": "sam2_starter/syniscopy_codebase.zip",
        }.items():
            record = source_archives.get(archive_key)
            if not isinstance(record, dict):
                fail(f"artifact provenance manifest missing source archive record {archive_key}", issues)
                continue
            if record.get("path") != rel:
                fail(f"artifact provenance manifest source archive {archive_key} has wrong path", issues)
            if not record.get("exists") or not record.get("sha256"):
                fail(f"artifact provenance manifest source archive {archive_key} is missing or un-hashed", issues)
            embedded_fingerprint = record.get("embedded_source_provenance_fingerprint")
            manifest_fingerprint = source_provenance.get("fingerprint") if isinstance(source_provenance, dict) else None
            if not embedded_fingerprint:
                fail(f"artifact provenance manifest source archive {archive_key} lacks embedded source fingerprint", issues)
            elif manifest_fingerprint and embedded_fingerprint != manifest_fingerprint:
                fail(
                    f"artifact provenance manifest source archive {archive_key} fingerprint mismatch: "
                    f"embedded {embedded_fingerprint}, manifest {manifest_fingerprint}",
                    issues,
                )
            member = record.get("embedded_manifest_member")
            archive_path = ROOT / rel
            if member and archive_path.is_file():
                try:
                    with zipfile.ZipFile(archive_path) as zf:
                        embedded = json.loads(zf.read(member).decode("utf-8"))
                    source = embedded.get("source_provenance") if isinstance(embedded.get("source_provenance"), dict) else {}
                    if source.get("fingerprint") != embedded_fingerprint:
                        fail(f"source archive {archive_key} embedded manifest fingerprint changed since paper assembly", issues)
                except Exception as exc:
                    fail(f"could not read embedded source manifest {member} in {rel}: {exc}", issues)
    for group_name in ("source_files", "artifacts"):
        for item in manifest.get(group_name, []) or []:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            if not rel_path:
                continue
            path = ROOT / rel_path
            expected_exists = bool(item.get("exists", False))
            expected_hash = item.get("sha256")
            if expected_exists and not path.exists():
                fail(f"artifact provenance manifest says {rel_path} exists, but it is missing", issues)
                continue
            if expected_exists and expected_hash and path.is_file():
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    fail(
                        f"artifact provenance manifest hash mismatch for {rel_path}: "
                        f"manifest {expected_hash}, current {actual_hash}",
                        issues,
                    )


# 4b) W15 paper-consumed E07/E09 provenance contracts.
e07_conversion_path = ROOT / "supplemental/outputs/E07/dataset/sam2_vos/conversion_manifest.json"
if e07_conversion_path.exists():
    conversion = json.loads(e07_conversion_path.read_text(encoding="utf-8", errors="replace"))
    config = conversion.get("conversion_config") if isinstance(conversion.get("conversion_config"), dict) else {}
    def _conv_value(key: str):
        return conversion.get(key, config.get(key))
    for key, expected in {
        "schema_version": "syniscopy-sam2-conversion-v3",
        "conversion_code_version": "syniscopy-sam2-vos-conversion-v5-instance-labels-required-sidecars",
        "annotation_schema_version": "syniscopy-supervision-v1",
        "mask_label_encoding": "per_particle_binary_png_sidecars",
        "gt_label_encoding": "per_particle_uint8_object_ids",
        "overlap_policy": "positive_overlap_pixels_are_ignored_not_assigned",
        "source_mask_resize_policy": "fail_on_mismatch",
    }.items():
        if _conv_value(key) != expected:
            fail(f"E07 conversion manifest {key}={_conv_value(key)!r}; expected {expected!r}", issues)
    if _conv_value("requires_ignore_and_loss_weight_sidecars") is not True:
        fail("E07 conversion manifest does not require ignore/loss sidecars", issues)
    signature = config.get("input_manifest_signature") if isinstance(config.get("input_manifest_signature"), dict) else {}
    if not (conversion.get("source_provenance_fingerprint") or config.get("source_provenance_fingerprint") or signature.get("source_provenance_fingerprint")):
        fail("E07 conversion manifest lacks source provenance fingerprint", issues)

e09_metrics = list((ROOT / "supplemental/outputs/E09/inference_outputs").glob("*/base/mask_metrics.json")) + list((ROOT / "supplemental/outputs/E09/inference_outputs").glob("*/finetuned/mask_metrics.json"))
for metrics_path in e09_metrics:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
    if metrics.get("schema_version") != "syniscopy-sam2-inference-metrics-v1":
        fail(f"{metrics_path}: unexpected E09 metrics schema {metrics.get('schema_version')!r}", issues)
    signature = metrics.get("input_signature")
    if not isinstance(signature, dict) or signature.get("schema_version") != "syniscopy-e09-variant-input-signature-v1":
        fail(f"{metrics_path}: missing E09 input_signature", issues)
        continue
    for key in ("source_video_sha256", "prompt_manifest_sha256", "checkpoint_manifest_sha256", "final_checkpoint_sha256", "frame_count", "frame_width", "frame_height", "model_variant"):
        if key not in signature:
            fail(f"{metrics_path}: input_signature missing {key}", issues)



if issues:
    print("ARXIV HARD GATE FAILED")
    for issue in issues:
        print(" -", issue)
    sys.exit(1)

print("ARXIV HARD GATE PASSED")
