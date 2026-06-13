from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from syniscopy_verification.artifacts import (
    iter_fusion_rows,
    iter_packet_paths,
    load_supervision_audit,
    read_lab_ranking_fishers,
    sigma_xy_from_fisher,
    validate_mask_group,
)


pytestmark = pytest.mark.artifacts


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def test_lab_report_fusion_csv_matches_raw_fim_addition() -> None:
    lab_report = _path_from_env("SYNISCOPY_VERIFY_LAB_REPORT")
    if lab_report is None:
        pytest.skip("set SYNISCOPY_VERIFY_LAB_REPORT or pass --lab-report")

    fishers = read_lab_ranking_fishers(lab_report)
    checked = 0
    for row in iter_fusion_rows(lab_report):
        microscopes = [item for item in str(row.get("microscopes_used", "")).split(";") if item]
        if not microscopes:
            continue
        missing = [m for m in microscopes if m not in fishers]
        assert not missing, f"fusion row references missing ranking FIMs: {missing}"
        F_sum = np.zeros((2, 2), dtype=float)
        for microscope in microscopes:
            F_sum += fishers[microscope]
        expected_sigma = sigma_xy_from_fisher(F_sum)
        observed_sigma = float(row["fusion_sigma_xy_nm"])
        if np.isfinite(expected_sigma):
            assert np.isclose(observed_sigma, expected_sigma, rtol=1.0e-6, atol=1.0e-9)
        else:
            assert not np.isfinite(observed_sigma)
        checked += 1
    assert checked > 0


def test_mask_directory_semantics_and_ambiguity_audit() -> None:
    mask_root = _path_from_env("SYNISCOPY_VERIFY_MASK_ROOT")
    if mask_root is None:
        pytest.skip("set SYNISCOPY_VERIFY_MASK_ROOT or pass --mask-root")

    checked = validate_mask_group(mask_root)
    assert checked > 0

    audit = load_supervision_audit(mask_root)
    if os.environ.get("SYNISCOPY_VERIFY_EXPECT_AMBIGUITY_DROPS") == "1":
        drop_counts = audit.get("drop_reason_counts", {})
        assert any("ambiguous_assignment" in str(reason) for reason in drop_counts), drop_counts


def test_matched_microscope_packets_validate_and_fuse_raw_fims() -> None:
    packet_root = _path_from_env("SYNISCOPY_VERIFY_PACKET_ROOT")
    if packet_root is None:
        pytest.skip("set SYNISCOPY_VERIFY_PACKET_ROOT or pass --packet-root")

    from matched_microscope_packets import load_matched_microscope_packet
    from fisher import (
        FisherMatrixCandidate,
        compute_candidate_fusion_crlb_from_fisher_matrices,
    )

    paths = iter_packet_paths(packet_root)
    assert paths, f"No .npz packets found under {packet_root}"

    for path in paths:
        packet = load_matched_microscope_packet(str(path))
        metadata = packet["metadata"]
        extra = metadata.get("metadata", {})
        shared_frame = extra.get("shared_coordinate_frame", {})
        assert shared_frame.get("fisher_frame") or shared_frame.get("axes")
        fishers = {
            name: np.asarray(F, dtype=float)
            for name, F in packet["fisher_by_microscope"].items()
        }
        assert len(fishers) >= 2
        candidates = [
            FisherMatrixCandidate(key=name, fisher_matrix=matrix)
            for name, matrix in fishers.items()
        ]
        fused = compute_candidate_fusion_crlb_from_fisher_matrices(candidates)
        expected = np.sum(list(fishers.values()), axis=0)
        assert np.allclose(fused["fusion_fisher"], expected)
