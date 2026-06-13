"""Validators for matched microscope packet provenance payloads."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from measurement_units import canonical_measurement_domain_and_signal_units_for_output
from modality_registry import SUPPORTED_MODALITIES, canonical_modality_name
from param_utils import _coerce_contract_truthy_flag


MATCHED_PACKET_SCHEMA_VERSION = "syniscopy-matched-microscope-packet-v2"


def _validate_profile_card_contract(modality: str, card: Mapping[str, Any]) -> None:
    modality_key = canonical_modality_name(modality)
    if modality_key not in set(SUPPORTED_MODALITIES):
        raise ValueError(f"Modality {modality!r} is not a supported registry modality.")
    if card.get("schema_version") != "syniscopy-modality-profile-v1":
        raise ValueError(f"Modality {modality!r} has invalid modality profile card schema.")
    for key in ("canonical_modality_name", "measurement_domain", "signal_units"):
        if not str(card.get(key, "") or "").strip():
            raise ValueError(f"Modality {modality!r} profile card missing {key!r}.")
    card_modality = canonical_modality_name(card["canonical_modality_name"])
    if card_modality != modality_key:
        raise ValueError(
            f"Modality {modality!r} profile card canonical_modality_name "
            f"{card['canonical_modality_name']!r} does not match record key."
        )
    response = dict(card.get("response_function", {}) or {})
    output_type = str(card.get("output_type", response.get("output_type", "intensity")))
    expected_domain, expected_units = canonical_measurement_domain_and_signal_units_for_output(
        modality_key,
        output_type,
        response_function=response,
    )
    if (str(card.get("measurement_domain")), str(card.get("signal_units"))) != (
        expected_domain,
        expected_units,
    ):
        raise ValueError(
            f"Modality {modality!r} profile card measurement domain/units "
            f"{(card.get('measurement_domain'), card.get('signal_units'))!r} do not match "
            f"canonical {(expected_domain, expected_units)!r}."
        )
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
    backend_modality = canonical_modality_name(
        backend_contract.get("modality_id", backend_contract.get("canonical_name", modality_key))
    )
    if backend_modality != modality_key:
        raise ValueError(
            f"Modality {modality!r} backend_contract modality "
            f"{backend_contract.get('modality_id', backend_contract.get('canonical_name'))!r} "
            "does not match record key."
        )
    if "known_omissions" not in backend_contract and "known_omissions" not in card:
        raise ValueError(f"Modality {modality!r} profile card must explicitly declare known_omissions.")
    fidelity = card.get("backend_fidelity_metadata")
    if not isinstance(fidelity, Mapping):
        raise ValueError(f"Modality {modality!r} profile card backend_fidelity_metadata must be a mapping.")
    for key in ("validation_status", "convergence_status", "comparison_contract_id"):
        if not str(fidelity.get(key, card.get(key, "")) or "").strip():
            raise ValueError(f"Modality {modality!r} backend_fidelity_metadata missing {key!r}.")
    _coerce_contract_truthy_flag(card.get("safe_for_linear_fisher_variance"))


def validate_matched_microscope_packet(packet: Mapping[str, Any]) -> None:
    """Validate the canonical matched microscope packet schema."""

    metadata = dict(packet.get("metadata", {}) or {})
    schema = packet.get("schema_version", metadata.get("schema_version"))
    if schema != MATCHED_PACKET_SCHEMA_VERSION:
        raise ValueError(
            f"Matched packet schema_version must be {MATCHED_PACKET_SCHEMA_VERSION!r}; got {schema!r}."
        )
    if metadata.get("packet_kind") != "matched_microscope_information_packet":
        raise ValueError(
            "Matched packet metadata packet_kind must be "
            "'matched_microscope_information_packet'."
        )
    if metadata.get("comparison_unit") != "microscope":
        raise ValueError("Matched packet comparison_unit must be 'microscope'.")
    for key in (
        "microscopes",
        "modality_by_microscope",
        "image_key_to_microscope",
        "fisher_key_to_microscope",
        "crlb_by_microscope",
    ):
        if key not in metadata:
            raise ValueError(f"Matched microscope packet metadata missing {key!r}.")

    validate_packet = import_module(
        "matched_microscope_packets"
    ).validate_matched_microscope_packet
    validate_packet(packet)


__all__ = [
    "MATCHED_PACKET_SCHEMA_VERSION",
    "_validate_profile_card_contract",
    "validate_matched_microscope_packet",
]
