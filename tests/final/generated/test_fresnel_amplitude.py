from __future__ import annotations

import pytest

from substrate.materials import fresnel_reflection_amplitude, material_from_name


def _analytic_fresnel(top: str, bottom: str, wavelength_nm: float) -> complex:
    n_top = material_from_name(top).n_complex(wavelength_nm)
    n_bottom = material_from_name(bottom).n_complex(wavelength_nm)
    return (n_top - n_bottom) / (n_top + n_bottom)


@pytest.mark.parametrize(
    "top,bottom,wavelength_nm",
    [
        ("water", "glass", 550.0),
        ("water", "gold", 550.0),
        ("air", "glass", 620.0),
    ],
)
def test_fresnel_reflection_matches_closed_form(top: str, bottom: str, wavelength_nm: float) -> None:
    observed = fresnel_reflection_amplitude(top, bottom, wavelength_nm)
    expected = _analytic_fresnel(top, bottom, wavelength_nm)

    assert abs(observed - expected) < 1e-12
    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fresnel_reflection_is_symmetric_under_medium_swap() -> None:
    top = "water"
    bottom = "glass"
    w = 500.0
    r_wg = fresnel_reflection_amplitude(top, bottom, w)
    r_gw = fresnel_reflection_amplitude(bottom, top, w)

    assert r_wg == pytest.approx(-r_gw, rel=1e-12, abs=1e-12)
    assert abs(r_wg) < 1.0
