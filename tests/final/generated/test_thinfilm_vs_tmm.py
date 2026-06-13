from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

import tmm  # type: ignore
import thinfilm  # type: ignore

WAVELENGTH_NM = 550.0


def _layer_indices(layers: Iterable[dict[str, object]]) -> list[complex]:
    out: list[complex] = []
    for layer in layers:
        raw = layer["n_complex"]
        if isinstance(raw, dict):
            out.append(complex(float(raw["real"]), float(raw["imag"])))
        else:
            out.append(complex(raw))
    return out


def _convention_aligned_error(r_ref: complex, r_syn: complex) -> float:
    return abs(r_syn - r_ref)


def _tmm_reflection(materials: list[complex], thicknesses: list[float]) -> complex:
    return tmm.coh_tmm("s", materials, thicknesses, 0.0, WAVELENGTH_NM)["r"]


@pytest.mark.parametrize(
    "label,layers",
    [
        ("bare interface", []),
        (
            "one dielectric layer",
            [
                {"n_complex": {"real": 1.46, "imag": 0.0}, "thickness_nm": 100.0},
            ],
        ),
        (
            "three layer, one lossy",
            [
                {"n_complex": {"real": 1.46, "imag": 0.0}, "thickness_nm": 80.0},
                {"n_complex": {"real": 2.10, "imag": 0.0}, "thickness_nm": 40.0},
                {"n_complex": {"real": 1.60, "imag": 0.05}, "thickness_nm": 120.0},
            ],
        ),
    ],
)
def test_thinfilm_reflection_matches_tmm_under_convention(label: str, layers: list[dict[str, object]]) -> None:
    r_syn = thinfilm.normal_incidence_thinfilm_reflection(
        "water",
        "glass",
        layers,
        WAVELENGTH_NM,
    )

    n_top = thinfilm.WATER.n_complex(WAVELENGTH_NM)
    n_sub = thinfilm.GLASS.n_complex(WAVELENGTH_NM)
    layer_indices = _layer_indices(layers)

    material_stack = [n_top] + layer_indices + [n_sub]
    thickness_stack = [np.inf] + [float(layer["thickness_nm"]) for layer in layers] + [np.inf]
    r_tmm = _tmm_reflection(material_stack, thickness_stack)
    assert np.isfinite(r_syn.real)
    assert np.isfinite(r_syn.imag)
    assert _convention_aligned_error(r_tmm, r_syn) < 1e-9
