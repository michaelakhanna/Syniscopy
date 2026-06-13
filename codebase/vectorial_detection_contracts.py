"""Shared vectorial-detection policy contracts.

The public ``vectorial_detection_mode`` parameter crosses schema validation,
PSF construction, concrete renderers, and metadata.  Keeping the policy in one
import-light module prevents modality-specific guards from drifting into
incompatible meanings for analyzer projections, full-vector fields, and
incoherent intensity reductions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


VECTORIAL_DETECTION_FULL_VECTOR = "full_vector"
VECTORIAL_DETECTION_ANALYZER_X = "analyzer_x"
VECTORIAL_DETECTION_ANALYZER_Y = "analyzer_y"
VECTORIAL_DETECTION_INCOHERENT_SUM = "incoherent_sum"
VECTORIAL_DETECTION_UNPOLARIZED = "unpolarized"

VALID_VECTORIAL_DETECTION_MODES = frozenset(
    {
        VECTORIAL_DETECTION_ANALYZER_X,
        VECTORIAL_DETECTION_ANALYZER_Y,
        VECTORIAL_DETECTION_INCOHERENT_SUM,
        VECTORIAL_DETECTION_UNPOLARIZED,
        VECTORIAL_DETECTION_FULL_VECTOR,
    }
)

DPC_VECTORIAL_CHANNEL_MODELS = frozenset(
    {
        "vectorial_debye_split_pupil_detection",
        "vectorial_debye_asymmetric_illumination",
        "two_axis_vectorial_debye_asymmetric_illumination",
        "vectorial",
    }
)

DPC_MODALITY_NAMES = frozenset({"dpc", "differential_phase_contrast"})


class VectorialDetectionReductionKind(str, Enum):
    """Canonical physical meaning of a vectorial detection mode."""

    COHERENT_ANALYZER_PROJECTION = "coherent_analyzer_projection"
    COHERENT_FULL_VECTOR_FIELD = "coherent_full_vector_field"
    INCOHERENT_INTENSITY_MIXTURE = "incoherent_intensity_mixture"


@dataclass(frozen=True)
class VectorialDetectionContract:
    """Resolved contract for one public vectorial detection mode."""

    mode: str
    reduction_kind: str
    requires_vector_field: bool
    coherent_reference_compatible: bool
    dpc_vectorial_channel_compatible: bool
    field_or_intensity_basis: str


def normalize_vectorial_detection_mode(value: Any) -> str:
    """Return a canonical vectorial detection mode or raise with all choices."""

    mode = str(value).strip().lower()
    if mode not in VALID_VECTORIAL_DETECTION_MODES:
        raise ValueError(
            "vectorial_detection_mode must be one of "
            f"{sorted(VALID_VECTORIAL_DETECTION_MODES)}; got {value!r}."
        )
    return mode


def is_vectorial_dpc_channel_model(value: Any) -> bool:
    """Return whether a DPC channel model consumes a vectorial Debye field."""

    return str(value).strip().lower() in DPC_VECTORIAL_CHANNEL_MODELS


def is_dpc_vectorial_field_path(modality: Any, dpc_channel_model: Any) -> bool:
    """Return whether the current modality/channel must transport Ex/Ey/Ez.

    This is intentionally independent of ``vectorial_detection_mode``: DPC
    analyzer and incoherent reductions still need the vector components so the
    DPC renderer, not the scalar PSF compatibility layer, owns the physical
    detection reduction.
    """

    return (
        str(modality).strip().lower() in DPC_MODALITY_NAMES
        and is_vectorial_dpc_channel_model(dpc_channel_model)
    )


def vectorial_detection_contract_for_mode(value: Any) -> VectorialDetectionContract:
    """Resolve the public mode string into a physical reduction contract."""

    mode = normalize_vectorial_detection_mode(value)
    if mode in {VECTORIAL_DETECTION_ANALYZER_X, VECTORIAL_DETECTION_ANALYZER_Y}:
        return VectorialDetectionContract(
            mode=mode,
            reduction_kind=VectorialDetectionReductionKind.COHERENT_ANALYZER_PROJECTION.value,
            requires_vector_field=False,
            coherent_reference_compatible=True,
            dpc_vectorial_channel_compatible=True,
            field_or_intensity_basis="coherent_scalar_field_projection",
        )
    if mode == VECTORIAL_DETECTION_FULL_VECTOR:
        return VectorialDetectionContract(
            mode=mode,
            reduction_kind=VectorialDetectionReductionKind.COHERENT_FULL_VECTOR_FIELD.value,
            requires_vector_field=True,
            coherent_reference_compatible=True,
            dpc_vectorial_channel_compatible=True,
            field_or_intensity_basis="coherent_vector_field",
        )
    return VectorialDetectionContract(
        mode=mode,
        reduction_kind=VectorialDetectionReductionKind.INCOHERENT_INTENSITY_MIXTURE.value,
        requires_vector_field=False,
        coherent_reference_compatible=False,
        dpc_vectorial_channel_compatible=True,
        field_or_intensity_basis="incoherent_intensity_mixture",
    )


def validate_vectorial_detection_policy(
    *,
    imaging_model: Any,
    optical_field_backend: Any,
    dpc_channel_model: Any,
    vectorial_detection_mode: Any,
    polarization_model: Any,
    is_coherent_reference_modality: bool,
) -> VectorialDetectionContract:
    """Validate cross-module vectorial detection semantics.

    The important invariant is that schema validation and renderers must agree
    on the physical reduction basis.  Analyzer modes are coherent scalar
    projections; full_vector transports Ex/Ey/Ez; incoherent modes are
    intensity mixtures and cannot be used where a coherent reference field is
    required.  Vectorial DPC is vector-aware even for analyzer modes, because
    the DPC renderer must own the reduction instead of receiving a scalar proxy
    from the PSF compatibility layer.
    """

    mode_contract = vectorial_detection_contract_for_mode(vectorial_detection_mode)
    backend = str(optical_field_backend).strip().lower()
    modality = str(imaging_model).strip().lower()
    polarization = str(polarization_model).strip().lower()
    vectorial_dpc = is_dpc_vectorial_field_path(modality, dpc_channel_model)

    if backend == "vectorial_debye" and is_coherent_reference_modality:
        if not mode_contract.coherent_reference_compatible:
            raise ValueError(
                "Incoherent vectorial detection reductions cannot be used as "
                f"coherent complex fields for imaging_model={modality!r}. Use "
                "analyzer_x, analyzer_y, full_vector, or "
                "optical_field_backend='scalar_paraxial'."
            )
        if mode_contract.mode == VECTORIAL_DETECTION_FULL_VECTOR and polarization == "unpolarized":
            raise ValueError(
                "polarization_model='unpolarized' is an incoherent average and "
                "cannot define the coherent reference field required by "
                "vectorial_detection_mode='full_vector' for "
                f"imaging_model={modality!r}. Use polarization_model='linear_x' "
                "or 'linear_y', an analyzer mode, or "
                "optical_field_backend='scalar_paraxial'."
            )

    if vectorial_dpc:
        if backend != "vectorial_debye":
            raise ValueError(
                "Vectorial DPC channel models require "
                "parameters['optical_field_backend']='vectorial_debye' for "
                "differential_phase_contrast; got "
                f"optical_field_backend={backend!r}."
            )
        if not mode_contract.dpc_vectorial_channel_compatible:
            raise ValueError(
                "Vectorial DPC channel models do not support "
                f"parameters['vectorial_detection_mode']={mode_contract.mode!r}."
            )
        if polarization == "unpolarized":
            raise ValueError(
                "parameters['polarization_model']='unpolarized' is an incoherent "
                "source average and cannot define the coherent vector field that "
                "vectorial DPC reduces. Use polarization_model='linear_x' or "
                "'linear_y', then choose full_vector, analyzer_x/analyzer_y, or "
                "an incoherent detection reduction explicitly."
            )

    return mode_contract


__all__ = [
    "DPC_MODALITY_NAMES",
    "DPC_VECTORIAL_CHANNEL_MODELS",
    "VALID_VECTORIAL_DETECTION_MODES",
    "VECTORIAL_DETECTION_ANALYZER_X",
    "VECTORIAL_DETECTION_ANALYZER_Y",
    "VECTORIAL_DETECTION_FULL_VECTOR",
    "VECTORIAL_DETECTION_INCOHERENT_SUM",
    "VECTORIAL_DETECTION_UNPOLARIZED",
    "VectorialDetectionContract",
    "VectorialDetectionReductionKind",
    "is_dpc_vectorial_field_path",
    "is_vectorial_dpc_channel_model",
    "normalize_vectorial_detection_mode",
    "validate_vectorial_detection_policy",
    "vectorial_detection_contract_for_mode",
]
