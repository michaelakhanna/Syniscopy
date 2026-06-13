"""Shared fluorescence-state kinetics helpers for vectorial photophysics backends."""

from __future__ import annotations
from configured_parameters import configured_value

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np



CANONICAL_BRIGHT_TO_DARK_KEY = "fluorescence_bright_to_dark_rate_per_frame"
CANONICAL_DARK_TO_BRIGHT_KEY = "fluorescence_dark_to_bright_rate_per_frame"
BLEACHING_RATE_KEY = "fluorescence_bleaching_rate_per_frame"

TRANSITION_RATE_SEMANTICS = {
    CANONICAL_BRIGHT_TO_DARK_KEY: "bright_to_dark",
    CANONICAL_DARK_TO_BRIGHT_KEY: "dark_to_bright",
}


def _finite_nonnegative(value: Any, *, key: str) -> float:
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{key} must be finite and non-negative; got {value!r}.")
    return rate


@dataclass(frozen=True)
class FluorescenceStateKinetics:
    """Resolved two-state fluorophore occupancy and bleaching rates."""

    bright_to_dark_rate_per_frame: float
    dark_to_bright_rate_per_frame: float
    bleaching_rate_per_frame: float

    def state_factor(self, frame_index: int) -> float:
        """Return the mean emitting fraction applied before detector-count scaling."""

        t = max(float(frame_index), 0.0)
        bleached_survival = np.exp(-self.bleaching_rate_per_frame * t)
        if self.bright_to_dark_rate_per_frame <= 0.0:
            emitting_fraction = 1.0
        else:
            total = max(
                self.bright_to_dark_rate_per_frame + self.dark_to_bright_rate_per_frame,
                1e-12,
            )
            bright_equilibrium = self.dark_to_bright_rate_per_frame / total
            emitting_fraction = bright_equilibrium + (1.0 - bright_equilibrium) * np.exp(-total * t)
        return float(np.clip(bleached_survival * emitting_fraction, 0.0, 1.0))

    def to_metadata(self) -> dict[str, Any]:
        """Expose the exact runtime kinetics contract used by the backend."""

        return {
            CANONICAL_BRIGHT_TO_DARK_KEY: self.bright_to_dark_rate_per_frame,
            CANONICAL_DARK_TO_BRIGHT_KEY: self.dark_to_bright_rate_per_frame,
            BLEACHING_RATE_KEY: self.bleaching_rate_per_frame,
            "fluorescence_transition_rate_semantics": dict(TRANSITION_RATE_SEMANTICS),
        }


def resolve_fluorescence_state_kinetics(params: Mapping[str, Any]) -> FluorescenceStateKinetics:
    """Resolve public fluorophore-state kinetics into one backend-owned contract.

    Kinetics contract: vectorial fluorescence consumers must not read the
    bright/dark/bleach keys independently. This resolver is the single authority
    that connects public transition-rate keys, runtime occupancy, and metadata
    provenance.
    """

    p = dict(params)
    canonical_bright_to_dark = _finite_nonnegative(
        configured_value(p, CANONICAL_BRIGHT_TO_DARK_KEY),
        key=CANONICAL_BRIGHT_TO_DARK_KEY,
    )
    canonical_dark_to_bright = _finite_nonnegative(
        configured_value(p, CANONICAL_DARK_TO_BRIGHT_KEY),
        key=CANONICAL_DARK_TO_BRIGHT_KEY,
    )
    bleaching_rate = _finite_nonnegative(
        configured_value(p, BLEACHING_RATE_KEY),
        key=BLEACHING_RATE_KEY,
    )

    return FluorescenceStateKinetics(
        bright_to_dark_rate_per_frame=canonical_bright_to_dark,
        dark_to_bright_rate_per_frame=canonical_dark_to_bright,
        bleaching_rate_per_frame=bleaching_rate,
    )


__all__ = [
    "TRANSITION_RATE_SEMANTICS",
    "FluorescenceStateKinetics",
    "resolve_fluorescence_state_kinetics",
]
