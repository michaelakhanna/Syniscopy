from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

import mie_scattering
import miepython


def _aligned_relative_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    good = np.abs(ref) > 1e-12
    if not np.any(good):
        return 0.0, 0.0

    scale = np.median(cand[good] / ref[good])
    residual = cand - scale * ref
    err = np.max(np.abs(residual[good]) / (np.abs(cand[good]) + 1e-30))
    spread = np.max(np.abs(cand[good] / ref[good] - scale) / (np.abs(scale) + 1e-30))
    return float(err), float(spread)


def _mp_S1_S2(m: complex, x: float, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return miepython.S1_S2(m, x, mu)


@pytest.mark.parametrize(
    "m,x",
    [
        (1.10 + 0.0j, 0.5),
        (1.10 + 0.0j, 3.0),
        (1.33 + 0.0j, 1.0),
        (1.33 + 0.0j, 8.0),
        (0.27 - 2.95j, 2.0),
    ],
)
def test_mie_amplitudes_match_miepython(m: complex, x: float) -> None:
    mu = np.cos(np.linspace(0.0, np.pi, 181))

    a_n, b_n = mie_scattering.mie_an_bn(m, x)
    s1_syn, s2_syn = mie_scattering.mie_S1_S2_from_coefficients(a_n, b_n, mu)
    s1_ref, s2_ref = _mp_S1_S2(m, x, mu)

    err1, spread1 = _aligned_relative_error(s1_ref, s1_syn)
    err2, spread2 = _aligned_relative_error(s2_ref, s2_syn)

    assert np.isfinite(s1_syn).all()
    assert np.isfinite(s2_syn).all()
    assert err1 < 1e-6
    assert err2 < 1e-6
    assert spread1 < 1e-6
    assert spread2 < 1e-6
