from __future__ import annotations

from pathlib import Path
import builtins
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBASE_DIR = PROJECT_ROOT / "codebase"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))


def _require_cv2_for_bootstrap() -> None:
    if hasattr(builtins, "require_cv2"):
        return

    class _MissingCV2:
        def __getattr__(self, name: str):
            raise ImportError(
                f"OpenCV (cv2) is required for substrate-dependent bootstrap; missing op {name!r}."
            )

    builtins.require_cv2 = lambda context: _MissingCV2()


def _analytic_relativistic_wavelength_m(acceleration_kV: float) -> float:
    h = 6.62607015e-34
    me = 9.1093837015e-31
    e = 1.602176634e-19
    c = 2.99792458e8

    V = float(acceleration_kV) * 1.0e3
    if V <= 0.0:
        raise ValueError("acceleration voltage must be positive")

    return h / np.sqrt(2 * me * e * V * (1.0 + e * V / (2 * me * c**2)))


def test_electron_wavelength_matches_relativistic_formula() -> None:
    from electron_optics import electron_wavelength_m

    for kV in (80.0, 100.0, 200.0, 300.0):
        actual = electron_wavelength_m(kV)
        expected = _analytic_relativistic_wavelength_m(kV)

        rel = abs(actual - expected) / expected
        assert rel < 1.0e-5
        assert actual > 0.0
        assert np.isfinite(actual)


def test_wavelength_monotonic_and_scherzer_defocus_contract() -> None:
    from electron_optics import electron_wavelength_m, scherzer_defocus_m

    volts = np.array([40.0, 80.0, 160.0, 300.0], dtype=float)
    lambdas = np.array([electron_wavelength_m(v) for v in volts], dtype=float)
    assert np.all(np.diff(lambdas) < 0.0)

    cs_mm = 1.5
    scherzer = np.array([scherzer_defocus_m(v, cs_mm) for v in volts], dtype=float)
    assert np.all(scherzer > 0.0)
    assert np.all(np.diff(scherzer) < 0.0)

    # Scherzer relation check: compare to formula definition.
    expected = np.sqrt(1.5 * lambdas * (1.0e-3 * cs_mm))
    np.testing.assert_allclose(scherzer, expected, rtol=1.0e-12, atol=1.0e-15)


def test_sem_transport_formulas_obey_expected_energy_trends() -> None:
    _require_cv2_for_bootstrap()
    from imaging_models.sem_backends.physical_transport import (
        joy_luo_stopping_power_keV_per_nm,
        kanaya_okayama_range_nm,
    )
    from material_optical_catalog import sem_transport_material

    mat = sem_transport_material("gold")
    energies_keV = np.array([5.0, 10.0, 20.0, 40.0], dtype=float)

    ranges = np.array([kanaya_okayama_range_nm(float(E), mat) for E in energies_keV], dtype=float)
    stopping = np.array([joy_luo_stopping_power_keV_per_nm(float(E), mat) for E in energies_keV], dtype=float)

    assert np.all(np.isfinite(ranges))
    assert np.all(np.isfinite(stopping))
    assert np.all(ranges > 0.0)
    assert np.all(stopping > 0.0)
    assert np.all(np.diff(ranges) > 0.0)
    assert np.all(np.diff(stopping) < 0.0)
