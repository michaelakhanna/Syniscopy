"""L08 [Level A] Verify the new absolute-normalization anchor.

codebase/mie_scattering.mie_scattering_cross_section_nm2(...) with the FULL cone
(collection_half_angle_rad=None) must equal the TOTAL Mie scattering cross-section
Qsca * pi * r^2, which PyMieScatt reports independently. This proves the constant
that the scattered-field normalization fix will use is physically correct BEFORE
it is wired into the renderer.

Run:  python L08_mie_cross_section_vs_pymiescatt.py
Needs: PyMieScatt (pip install PyMieScatt). numpy/scipy already required.
"""
from __future__ import annotations
import numpy as np
from common import add_paths, banner, verdict
add_paths()

# SciPy>=1.14 removed scipy.integrate.trapz; PyMieScatt 2.9 still imports it.
import scipy.integrate as _si
if not hasattr(_si, "trapz"):
    _si.trapz = _si.trapezoid

from mie_scattering import mie_scattering_cross_section_nm2
try:
    # Import the Mie submodule directly: the PyMieScatt package __init__ pulls in
    # matplotlib via PyMieScatt.Inverse, which we don't need and which can stall.
    from PyMieScatt.Mie import MieQ
except Exception as e:
    raise SystemExit(f"PyMieScatt import failed: {e}\nInstall: pip install PyMieScatt")

banner("L08  Mie scattering cross-section vs PyMieScatt (Qsca)")

# (n_particle, n_medium, diameter_nm, wavelength_nm)
cases = [
    (1.59, 1.33, 100.0, 532.0),   # polystyrene in water
    (1.59, 1.33, 200.0, 532.0),
    (1.45, 1.33, 150.0, 488.0),   # silica-ish
    (2.10, 1.33, 80.0, 532.0),    # high-index dielectric
]
all_ok = True
print(f"{'n_p':>5} {'n_med':>6} {'d(nm)':>6} {'lam':>5} | {'syn sigma(nm^2)':>16} {'pms sigma(nm^2)':>16} {'rel':>9}")
for n_p, n_med, d, lam in cases:
    m = (n_p / n_med) + 0.0j
    syn = mie_scattering_cross_section_nm2(m, d, lam, n_med, collection_half_angle_rad=None)
    # PyMieScatt: MieQ(m_particle_relative_to_medium? ) -- it expects ABSOLUTE m and
    # wavelength IN THE MEDIUM is handled by passing m=n_p and wavelength/n? Use the
    # standard form: MieQ(m_rel, wavelength_in_medium, diameter). wavelength_in_medium = lam/n_med.
    Qsca = MieQ(m, lam / n_med, d, asDict=True)["Qsca"]
    pms_sigma = Qsca * np.pi * (0.5 * d) ** 2
    rel = abs(syn - pms_sigma) / max(pms_sigma, 1e-30)
    ok = rel < 5e-3
    all_ok &= ok
    print(f"{n_p:>5.2f} {n_med:>6.2f} {d:>6.0f} {lam:>5.0f} | {syn:>16.6g} {pms_sigma:>16.6g} {rel:>9.2e} {'OK' if ok else 'FAIL'}")

print("\nIf PASS, the absolute scattered-power constant is correct and can be wired")
print("into optics.py / vectorial_optics.py per SCATTERED_FIELD_NORMALIZATION_FIX.md.")
raise SystemExit(verdict(all_ok, "(total Mie cross-section matches PyMieScatt Qsca*pi*r^2)"))
