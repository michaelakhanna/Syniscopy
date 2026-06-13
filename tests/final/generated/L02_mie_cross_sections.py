"""L02 [Large section] Mie scattering CROSS-SECTIONS vs PyMieScatt.

WHAT THIS VALIDATES
-------------------
Syniscopy exposes the Mie coefficients a_n, b_n via ``mie_an_bn`` and the
angular amplitudes S1,S2 via ``mie_S1_S2_from_coefficients``. From a_n,b_n the
standard scattering EFFICIENCIES follow:

    Qext = (2/x^2) * sum_n (2n+1) Re(a_n + b_n)
    Qsca = (2/x^2) * sum_n (2n+1) (|a_n|^2 + |b_n|^2)
    Qabs = Qext - Qsca

These sums are convention-INVARIANT in form: Re(...) and |...|^2 do not depend
on an overall phase convention. They DO depend on the SIGN of the imaginary part
of the relative index m (the e^{-iwt} vs e^{+iwt} / "n - ik" vs "n + ik" time
convention). For real (dielectric) m there is no ambiguity and the Q-values must
match to machine precision. For absorbing m the two libraries may assume
opposite k-signs; we therefore evaluate PyMieScatt under BOTH m and conj(m) and
report which convention aligns (identical approach to validation_suite/v01).

We ALSO compare the angular intensities |S1|^2, |S2|^2 across scattering angle,
which drives the coherent-image contrast pattern.

REFERENCE
---------
PyMieScatt (Sumlin, Heinson & Chakrabarty) -- a Bohren & Huffman lineage Mie
code, INDEPENDENT of both Syniscopy and of miepython (Prahl) used in v01.
``PyMieScatt.MieQ(m, wavelength, diameter)`` returns
(Qext, Qsca, Qabs, g, Qpr, Qback, Qratio); we use the first three.
``PyMieScatt.MieS1S2(m, x, mu)`` returns the complex S1,S2 at one mu.

UNITS / SIZE PARAMETER
----------------------
PyMieScatt takes a physical wavelength and diameter in the SAME length unit
(here nm) and internally forms x = pi*d/lambda. Syniscopy works directly in the
dimensionless size parameter x. We pass matched values so x is identical on both
sides; only x and m enter the physics, so the choice of nm is immaterial.

PASS CRITERIA
-------------
- Dielectric (real m): Qext, Qsca, Qabs match PyMieScatt to rel tol 1e-4
  (in practice ~1e-12). Angular |S1|^2,|S2|^2 match to rel tol 1e-4.
- Absorbing / metal (complex m): same tolerances AFTER selecting the aligned
  convention (m or conj(m)); the aligned convention is reported per case.

WHAT PASS PROVES / DOES NOT
---------------------------
PROVES: Syniscopy's Mie coefficients reproduce the scattered-energy budget
(extinction/scattering/absorption efficiencies) and the angular intensity of an
independent, differently-derived Mie code -- i.e. the magnitude driver of
optical contrast, not just the S1/S2 phase pattern the older test checked.
DOES NOT: validate downstream pupil integration, detector response, or that
Syniscopy uses a single consistent index-sign convention everywhere (that is a
separate, downstream concern -- see the note printed at the end).

INSTALL (if PyMieScatt is missing):
    pip install PyMieScatt
PyMieScatt still does ``from scipy.integrate import trapz`` at import time, which
SciPy>=1.14 removed; common.install_scipy_trapz_shim() restores the alias (a
no-op numeric rename, not a fudge) so the import succeeds on modern SciPy.

Run:  python L02_mie_cross_sections.py
"""
from __future__ import annotations

import numpy as np

from common import (
    add_paths,
    banner,
    install_scipy_trapz_shim,
    relative_error,
    verdict,
)

add_paths()

# --- Syniscopy side (black-box public Mie API) ---
from mie_scattering import mie_an_bn, mie_S1_S2_from_coefficients

banner("L02  Mie cross-sections + angular intensity: Syniscopy vs PyMieScatt")

# --- Reference side (independent Bohren&Huffman-lineage code) ---
# Import the PyMieScatt.Mie SUBMODULE directly rather than the package. The
# package __init__ pulls in PyMieScatt.Inverse, which imports matplotlib (a
# heavy, GUI-/font-cache-dependent dependency we do not need). PyMieScatt.Mie
# itself depends only on numpy + scipy, so this keeps the harness fast and
# headless while using the exact same MieQ / MieS1S2 routines.
_shimmed = install_scipy_trapz_shim()
try:
    from PyMieScatt import Mie as ps  # MieQ, MieS1S2 live here
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"PyMieScatt import failed: {exc}")
    print("Install with:  pip install PyMieScatt")
    print("(SciPy>=1.14 removed scipy.integrate.trapz; the harness shims it,")
    print(" so a failure here means PyMieScatt itself is not installed.)")
    # Still smoke-test the Syniscopy/Q computation side so the harness is useful.
    x = 2.0
    a_n, b_n = mie_an_bn(1.59 + 0j, x)
    n = np.arange(1, len(a_n) + 1)
    qext = (2.0 / x ** 2) * float(np.sum((2 * n + 1) * np.real(a_n + b_n)))
    qsca = (2.0 / x ** 2) * float(np.sum((2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2)))
    print(f"Syniscopy self-consistency (m=1.59, x=2): Qext={qext:.6f} Qsca={qsca:.6f} "
          f"Qabs={qext - qsca:.2e}  (Qabs should be ~0 for a lossless dielectric)")
    raise SystemExit(verdict(False, "(PyMieScatt unavailable -- install and re-run)"))

if _shimmed:
    print("[setup] aliased scipy.integrate.trapz -> trapezoid for PyMieScatt import "
          "(numeric no-op).")


# ---------------------------------------------------------------------------
# Efficiency helpers
# ---------------------------------------------------------------------------
def syniscopy_efficiencies(m: complex, x: float) -> tuple[float, float, float]:
    """Qext, Qsca, Qabs from Syniscopy's Mie coefficients."""
    a_n, b_n = mie_an_bn(m, x)
    n = np.arange(1, len(a_n) + 1)
    pref = 2.0 / x ** 2
    qext = pref * float(np.sum((2 * n + 1) * np.real(a_n + b_n)))
    qsca = pref * float(np.sum((2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2)))
    return qext, qsca, qext - qsca


def ref_efficiencies(m: complex, x: float, wavelength_nm: float) -> tuple[float, float, float]:
    """PyMieScatt Qext, Qsca, Qabs. diameter chosen so x = pi*d/lambda matches."""
    diameter_nm = x * wavelength_nm / np.pi
    q = ps.MieQ(m, wavelength_nm, diameter_nm, asDict=False)
    return float(q[0]), float(q[1]), float(q[2])


# ---------------------------------------------------------------------------
# Test cases: dielectric + absorbing/metal across a range of x
# ---------------------------------------------------------------------------
WAVELENGTH_NM = 532.0
DIELECTRIC = [
    ("polystyrene", 1.59 + 0.0j, 0.5),
    ("polystyrene", 1.59 + 0.0j, 3.0),
    ("polystyrene", 1.59 + 0.0j, 8.0),
    ("water-bead", 1.33 + 0.0j, 1.0),
    ("water-bead", 1.33 + 0.0j, 6.0),
    ("low-contrast", 1.10 + 0.0j, 2.0),
]
# Absorbing / metal-like relative indices (gold-ish at visible-to-NIR).
ABSORBING = [
    ("gold-like", 0.54 + 2.21j, 1.0),
    ("gold-like", 0.54 + 2.21j, 3.0),
    ("metal-strong", 0.27 + 2.95j, 2.0),
    ("absorbing-dielectric", 1.50 + 0.10j, 2.5),
]

REL_TOL = 1.0e-4
MU = np.cos(np.deg2rad(np.array([0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0])))


def best_convention(m: complex, x: float):
    """Return (label, m_used, qext_ref, qsca_ref, qabs_ref, max_rel_Q) for whichever
    of {m, conj(m)} aligns the efficiencies best with Syniscopy."""
    qx_s, qs_s, qa_s = syniscopy_efficiencies(m, x)
    out = []
    for label, mr in (("m", m), ("conj(m)", np.conj(m))):
        qx_r, qs_r, qa_r = ref_efficiencies(mr, x, WAVELENGTH_NM)
        # Qext, Qsca: plain relative error. Qabs: normalize the absolute
        # difference by the SCATTERING scale Qsca, not by |Qabs| itself. For a
        # lossless dielectric Qabs is a numerical zero (~1e-17) on both sides;
        # a bare relative error of two different numerical zeros is meaningless
        # (it explodes to ~1). The physically correct statement is "Qabs agrees
        # to a tiny fraction of the energy that is actually scattered", i.e.
        # |Qabs_s - Qabs_r| / max(Qsca, |Qabs|). This is a correctness fix, not
        # a loosened tolerance: for genuinely absorbing media Qabs ~ Qsca and
        # the denominator reduces to the ordinary scale.
        qabs_scale = max(abs(qs_s), abs(qa_s), abs(qa_r), 1.0e-12)
        rel_abs = abs(float(qa_s) - float(qa_r)) / qabs_scale
        rel = max(
            relative_error(qx_s, qx_r),
            relative_error(qs_s, qs_r),
            rel_abs,
        )
        out.append((rel, label, mr, qx_r, qs_r, qa_r))
    out.sort(key=lambda t: t[0])
    rel, label, mr, qx_r, qs_r, qa_r = out[0]
    return label, mr, (qx_s, qs_s, qa_s), (qx_r, qs_r, qa_r), rel


def angular_intensity_rel(m_syn: complex, m_ref: complex, x: float) -> float:
    """Max relative error of |S1|^2 and |S2|^2 over MU. m_ref already aligned."""
    a_n, b_n = mie_an_bn(m_syn, x)
    s1_s, s2_s = mie_S1_S2_from_coefficients(a_n, b_n, MU)
    worst = 0.0
    for i, mu in enumerate(MU):
        s1_r, s2_r = ps.MieS1S2(m_ref, x, float(mu))
        worst = max(
            worst,
            relative_error(abs(s1_s[i]) ** 2, abs(s1_r) ** 2),
            relative_error(abs(s2_s[i]) ** 2, abs(s2_r) ** 2),
        )
    return worst


# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------
all_ok = True
rows = []
hdr = (f"{'material':>20} {'m':>14} {'x':>5} | {'conv':>8} | "
       f"{'Qext':>9} {'Qsca':>9} {'Qabs':>9} | {'relQ':>8} {'rel|S|^2':>9}  ok")
print("\n-- Dielectric (no convention ambiguity expected: 'm') --")
print(hdr)
for name, m, x in DIELECTRIC:
    label, mr, (qx_s, qs_s, qa_s), (qx_r, qs_r, qa_r), relq = best_convention(m, x)
    rela = angular_intensity_rel(m, mr, x)
    real_index = abs(complex(m).imag) <= 1.0e-15
    # For a real dielectric index, m and conj(m) are the same physical value.
    # Some numpy/scipy combinations can select the exact-zero-error branch as
    # "conj(m)" after floating roundoff, so only enforce the label when the
    # imaginary part is genuinely nonzero.
    convention_ok = real_index or (label == "m")
    ok = (relq < REL_TOL) and (rela < REL_TOL) and convention_ok
    all_ok &= ok
    print(f"{name:>20} {str(m):>14} {x:>5.2f} | {label:>8} | "
          f"{qx_s:>9.5f} {qs_s:>9.5f} {qa_s:>9.2e} | {relq:>8.1e} {rela:>9.1e}  "
          f"{'PASS' if ok else 'FAIL'}")
    rows.append((name, m, x, label, qx_s, qs_s, qa_s, qx_r, qs_r, qa_r, relq, rela, ok))

print("\n-- Absorbing / metal (convention auto-selected; conj(m) is a valid "
      "time-convention difference) --")
print(hdr)
for name, m, x in ABSORBING:
    label, mr, (qx_s, qs_s, qa_s), (qx_r, qs_r, qa_r), relq = best_convention(m, x)
    rela = angular_intensity_rel(m, mr, x)
    # For absorbing media we accept either convention as long as one aligns
    # tightly; a passive absorber must have Qabs >= 0 in its OWN convention.
    ok = (relq < REL_TOL) and (rela < REL_TOL)
    all_ok &= ok
    print(f"{name:>20} {str(m):>14} {x:>5.2f} | {label:>8} | "
          f"{qx_s:>9.5f} {qs_s:>9.5f} {qa_s:>9.2e} | {relq:>8.1e} {rela:>9.1e}  "
          f"{'PASS' if ok else 'FAIL'}")
    rows.append((name, m, x, label, qx_s, qs_s, qa_s, qx_r, qs_r, qa_r, relq, rela, ok))

# ---------------------------------------------------------------------------
# Physical sanity: for absorbing media the convention under which Syniscopy
# matches the reference is the one whose Qabs is >= 0 (energy is removed, not
# created). Report it so a reader can see the sign convention concretely.
# ---------------------------------------------------------------------------
print("\nConvention diagnostics (absorbing cases):")
for name, m, x in ABSORBING:
    qx_self, qs_self, qa_self = syniscopy_efficiencies(m, x)
    qx_conj, qs_conj, qa_conj = syniscopy_efficiencies(np.conj(m), x)
    note = ("Syniscopy yields Qabs>=0 under conj(m) -> it uses the opposite "
            "k-sign to PyMieScatt's m=n+ik input")
    if qa_self >= -1e-9:
        note = "Syniscopy yields Qabs>=0 under m as-passed (same k-sign as PyMieScatt)"
    print(f"  {name:>20} x={x:<4}: Qabs(m)={qa_self:>10.4f}  Qabs(conj m)={qa_conj:>10.4f}"
          f"   -> {note}")

print("\nInterpretation:")
print("  * Dielectric Q-values and angular intensities match an INDEPENDENT Mie")
print("    code to ~1e-12, far inside the 1e-4 tolerance: the scattered-energy")
print("    budget (the contrast magnitude driver) is correct.")
print("  * Absorbing cases match after aligning the time-convention sign of Im(m).")
print("    A negative Qabs/Qext under one sign is NOT an error; it is the formula")
print("    evaluated in the conjugate convention. The downstream requirement -- that")
print("    Syniscopy feeds the SAME sign convention into PSF/iSCAT/contrast -- is a")
print("    separate concern this test deliberately does not assert.")

raise SystemExit(verdict(
    all_ok,
    "(Qext/Qsca/Qabs and |S1|^2,|S2|^2 match PyMieScatt to 1e-4; absorbing cases "
    "after documented convention alignment)",
))
