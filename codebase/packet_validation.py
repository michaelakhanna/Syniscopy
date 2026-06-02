"""Validators for matched-modality packet provenance payloads."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import numpy as np
from param_utils import _coerce_contract_truthy_flag
from shared_constants import MATCHED_INFORMATION_MASK_ROLES


MATCHED_PACKET_SCHEMA_VERSION = "syniscopy-matched-modality-packet-v1"


def _finite_array(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric and non-object; got dtype {arr.dtype}.")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued.")
    arr = np.asarray(arr, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _mask(name: str, value: Any, shape: tuple[int, ...], *, binary: bool) -> np.ndarray:
    arr = _finite_array(name, value)
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}; expected {shape}.")
    if binary and not np.all(np.isin(np.unique(arr), [0.0, 1.0, 255.0])):
        raise ValueError(f"{name} must be binary with values 0/1 or 0/255.")
    if not binary and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    if binary:
        return np.where(arr > 0.0, 255.0, 0.0)
    return arr


def _validate_mask_semantics(modality: str, masks: Mapping[str, Any], shape: tuple[int, ...]) -> None:
    geometry = _mask(f"{modality}.mask_geometry", masks["mask_geometry"], shape, binary=True) > 0
    supported = _mask(f"{modality}.mask_supported", masks["mask_supported"], shape, binary=True) > 0
    ignored = _mask(f"{modality}.ignore_mask", masks["ignore_mask"], shape, binary=True) > 0
    loss_weight = _mask(f"{modality}.loss_weight", masks["loss_weight"], shape, binary=False)
    if np.any(supported & ~geometry):
        raise ValueError(f"Modality {modality!r} mask_supported must be a subset of mask_geometry.")
    if np.any(ignored & ~geometry):
        raise ValueError(f"Modality {modality!r} ignore_mask must be a subset of mask_geometry.")
    if not np.array_equal(ignored, geometry & ~supported):
        raise ValueError(
            f"Modality {modality!r} ignore_mask must equal mask_geometry & ~mask_supported."
        )
    if np.any(loss_weight[~supported] > 0.0):
        raise ValueError(
            f"Modality {modality!r} loss_weight must be zero outside mask_supported."
        )
    if np.any(supported) and not np.any(loss_weight[supported] > 0.0):
        raise ValueError(
            f"Modality {modality!r} has supported pixels but no positive loss_weight."
        )


def _validate_profile_card_contract(modality: str, card: Mapping[str, Any]) -> None:
    if card.get("schema_version") != "syniscopy-modality-profile-v1":
        raise ValueError(f"Modality {modality!r} has invalid modality profile card schema.")
    for key in ("canonical_modality_name", "measurement_domain", "signal_units"):
        if not str(card.get(key, "") or "").strip():
            raise ValueError(f"Modality {modality!r} profile card missing {key!r}.")
    for key in (
        "backend_contract",
        "backend_fidelity_level",
        "backend_fidelity_metadata",
        "comparison_contract_id",
        "detector_model",
        "safe_for_linear_fisher_variance",
        "fisher_variance_model_scope",
        "detector_likelihood_status",
        "model_card",
    ):
        if key not in card:
            raise ValueError(f"Modality {modality!r} profile card missing contract field {key!r}.")
    backend_contract = card.get("backend_contract")
    if not isinstance(backend_contract, Mapping):
        raise ValueError(f"Modality {modality!r} profile card backend_contract must be a mapping.")
    if "known_omissions" not in backend_contract and "known_omissions" not in card:
        raise ValueError(f"Modality {modality!r} profile card must explicitly declare known_omissions.")
    fidelity = card.get("backend_fidelity_metadata")
    if not isinstance(fidelity, Mapping):
        raise ValueError(f"Modality {modality!r} profile card backend_fidelity_metadata must be a mapping.")
    for key in ("validation_status", "convergence_status", "comparison_contract_id"):
        if not str(fidelity.get(key, card.get(key, "")) or "").strip():
            raise ValueError(f"Modality {modality!r} backend_fidelity_metadata missing {key!r}.")
    if not _coerce_contract_truthy_flag(card.get("safe_for_linear_fisher_variance")):
        raise ValueError(
            f"Modality {modality!r} profile card is not safe_for_linear_fisher_variance "
            f"({card.get('detector_likelihood_status', '')})."
        )


def validate_matched_modality_packet(packet: Mapping[str, Any]) -> None:
    """Validate the canonical matched-modality packet schema.

    The current public writer in ``counterfactual_packets`` emits packets with
    top-level image/fisher mappings and the same matched-packet schema version.
    Older development records used a nested ``modalities`` mapping. Accept both
    shapes while preserving the shared schema-version guard.
    """
    metadata = dict(packet.get("metadata", {}) or {})
    schema = packet.get("schema_version", metadata.get("schema_version"))
    if schema != MATCHED_PACKET_SCHEMA_VERSION:
        raise ValueError(
            f"Matched packet schema_version must be {MATCHED_PACKET_SCHEMA_VERSION!r}; got {schema!r}."
        )

    if "images_by_modality" in packet:
        packet_kind = metadata.get("packet_kind")
        if packet_kind not in {None, "matched_modality_information_packet"}:
            raise ValueError(
                "Matched packet metadata packet_kind must be "
                "'matched_modality_information_packet' when present; "
                f"got {packet_kind!r}."
            )
        validate_counterfactual_modality_packet = import_module(
            "counterfactual_packets"
        ).validate_counterfactual_modality_packet
        validate_counterfactual_modality_packet(packet)
        return

    for key in ("latent_scene_id", "frame_index", "particle_state", "sample_environment_config", "provenance"):
        if key not in packet:
            raise ValueError(f"Matched packet missing required field {key!r}.")

    records = packet.get("modalities", None)
    if not isinstance(records, Mapping) or len(records) < 2:
        raise ValueError("Matched packet must contain at least two modality records.")

    for modality, record_raw in records.items():
        record = dict(record_raw)
        for key in (
            "modality_profile_card",
            "rendered_signal_frame",
            "analysis_contrast_image",
            "noise_variance_map",
            "fisher_matrix",
            "crlb_summary",
            "fisher_metadata",
            "masks",
            "support_factors",
            "rejection_reasons",
            "provenance",
        ):
            if key not in record:
                raise ValueError(f"Modality {modality!r} missing required field {key!r}.")

        card = dict(record["modality_profile_card"])
        _validate_profile_card_contract(str(modality), card)

        signal = _finite_array(f"{modality}.rendered_signal_frame", record["rendered_signal_frame"], ndim=2)

        if "reference_frame" in record and record["reference_frame"] is not None:
            ref = _finite_array(f"{modality}.reference_frame", record["reference_frame"], ndim=2)
            if ref.shape != signal.shape:
                raise ValueError(f"Modality {modality!r} reference_frame shape does not match signal frame.")

        contrast = _finite_array(f"{modality}.analysis_contrast_image", record["analysis_contrast_image"], ndim=2)
        noise = _finite_array(f"{modality}.noise_variance_map", record["noise_variance_map"], ndim=2)

        if contrast.shape != signal.shape or noise.shape != signal.shape:
            raise ValueError(f"Modality {modality!r} contrast/noise shapes must match signal frame.")
        if np.any(noise <= 0.0):
            raise ValueError(f"Modality {modality!r} noise_variance_map must be positive.")

        fisher = _finite_array(f"{modality}.fisher_matrix", record["fisher_matrix"], ndim=2)
        if fisher.shape[0] != fisher.shape[1] or not np.allclose(fisher, fisher.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Modality {modality!r} fisher_matrix must be square symmetric.")

        masks = dict(record["masks"])
        for key in MATCHED_INFORMATION_MASK_ROLES:
            if key not in masks:
                raise ValueError(f"Modality {modality!r} masks missing {key!r}.")

        _validate_mask_semantics(str(modality), masks, signal.shape)


__all__ = ["MATCHED_PACKET_SCHEMA_VERSION", "validate_matched_modality_packet"]
