"""Typed contracts for rendered analysis-contrast products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from array_representation import ArrayRepresentation, UNKNOWN_ARRAY_REPRESENTATION


ANALYSIS_CONTRAST_PRODUCT_CONTRACT_ID = "syniscopy-analysis-contrast-product-v1"


@dataclass(frozen=True)
class AnalysisContrastProduct:
    """Validated quantitative analysis frames plus their array representation.

    Display/raw background-subtraction settings are user-facing rendering
    policy, not proof that a frame is in a physical analysis basis.  The
    composed ``ArrayRepresentation`` is the single representation owner for
    this product; frame/contrast labels in metadata are projections of
    that descriptor, not independent enums.
    """

    frames: Sequence[np.ndarray]
    source: str
    representation: ArrayRepresentation
    semantics: str
    quantitative: bool
    safe_for_fisher: bool
    background_subtraction_method: str | None
    display_background_subtraction_applied: bool = False
    provenance_warning: str = ""
    contract_id: str = ANALYSIS_CONTRAST_PRODUCT_CONTRACT_ID

    def __post_init__(self) -> None:
        representation = (
            self.representation
            if isinstance(self.representation, ArrayRepresentation)
            else UNKNOWN_ARRAY_REPRESENTATION
        )
        object.__setattr__(self, "representation", representation)

    @property
    def frame_basis(self) -> str | None:
        """Metadata projection of the composed representation stage."""

        if self.representation is UNKNOWN_ARRAY_REPRESENTATION:
            return None
        return self.representation.pipeline_stage

    @property
    def contrast_basis(self) -> str | None:
        """Metadata projection of the composed representation semantic label."""

        if self.representation is UNKNOWN_ARRAY_REPRESENTATION:
            return None
        return self.representation.semantic_label or self.representation.value_form

    @property
    def units(self) -> str | None:
        """Metadata projection of the composed representation units."""

        if self.representation is UNKNOWN_ARRAY_REPRESENTATION:
            return None
        return self.representation.units

    def metadata(self, *, prefix: str) -> dict[str, Any]:
        payload = {
            f"{prefix}_source": self.source,
            f"{prefix}_frame_basis": self.frame_basis,
            f"{prefix}_contrast_basis": self.contrast_basis,
            f"{prefix}_units": self.units,
            f"{prefix}_semantics": self.semantics,
            f"{prefix}_quantitative": bool(self.quantitative),
            f"{prefix}_safe_for_fisher": bool(self.safe_for_fisher),
            f"{prefix}_provenance_warning": self.provenance_warning,
            f"{prefix}_background_subtraction_method": self.background_subtraction_method,
            f"{prefix}_display_background_subtraction_applied": bool(
                self.display_background_subtraction_applied
            ),
            f"{prefix}_contract_id": self.contract_id,
        }
        payload.update(self.representation.metadata(prefix=f"{prefix}_array"))
        return payload


__all__ = [
    "ANALYSIS_CONTRAST_PRODUCT_CONTRACT_ID",
    "AnalysisContrastProduct",
]
