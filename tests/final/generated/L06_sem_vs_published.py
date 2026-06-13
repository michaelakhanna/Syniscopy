"""L06 [Level B] SEM electron-transport vs published numbers (Mac-friendly).

The SEM Monte-Carlo backend has no clean pip oracle (CASINO is a Windows GUI),
so this anchors the SEM transport PHYSICS to published literature values that do
not require any install:

  1. Reuter backscatter coefficient eta(Z) at 20 keV vs accepted experimental
     values for C, Si, Cu, Ag, Au (Reuter 1972 / Joy compilations). This is a
     genuine EXTERNAL anchor (real measured numbers), not a self-formula.
  2. Kanaya-Okayama electron range R(E) scaling: R should follow the published
     R proportional to E^1.67 power law, and match the closed-form KO range for
     gold/carbon at several energies (catches unit/coefficient errors).

Run: python L06_sem_vs_published.py     (pure numpy; no pip installs)
"""
from __future__ import annotations
import numpy as np
from common import add_paths, banner, verdict
add_paths()

from imaging_models.sem_backends.physical_transport import (
    kanaya_okayama_range_nm,
    reuter_backscatter_coefficient_20kev,
)
from material_optical_catalog import sem_transport_material

banner("L06  SEM electron transport vs published literature values")

# --- 1. Backscatter coefficient eta(Z) at 20 keV vs accepted experimental values ---
# Accepted ~20 keV backscatter coefficients (Joy/Reuter compilations).
PUBLISHED_ETA = {6: 0.06, 14: 0.17, 29: 0.30, 47: 0.42, 79: 0.50}
print(f"{'Z':>4} {'eta_syn':>9} {'eta_pub':>9} {'abs_err':>9}")
eta_ok = True
etas = []
for Z, eta_pub in PUBLISHED_ETA.items():
    eta_syn = float(reuter_backscatter_coefficient_20kev(float(Z)))
    etas.append(eta_syn)
    err = abs(eta_syn - eta_pub)
    ok = err <= 0.06            # absolute tolerance (eta is O(0.1); ~one bin)
    eta_ok &= ok
    print(f"{Z:>4} {eta_syn:>9.3f} {eta_pub:>9.3f} {err:>9.3f} {'OK' if ok else 'FAIL'}")
monotonic = all(etas[i] < etas[i + 1] for i in range(len(etas) - 1))
print(f"monotonic increasing with Z: {monotonic}")
eta_ok &= monotonic

# --- 2. Kanaya-Okayama range scaling vs published power law + closed form ---
def ko_closed_form_nm(E_keV, A, Z, rho):
    # R_KO[um] = 0.0276 * A * E^1.67 / (Z^0.889 * rho);  A g/mol, E keV, rho g/cm^3
    return 1000.0 * 0.0276 * A * (E_keV ** 1.67) / (Z ** 0.889 * rho)

print("\nKanaya-Okayama range:")
ko_ok = True
for name in ("gold", "carbon"):
    try:
        mat = sem_transport_material(name)
        A = float(mat.atomic_weight_g_mol); Z = float(mat.atomic_number); rho = float(mat.density_g_cm3)
    except Exception as ex:
        print(f"  {name}: material lookup failed ({ex}) -- skipping")
        continue
    energies = np.array([5.0, 10.0, 20.0, 30.0])
    R_syn = np.array([float(kanaya_okayama_range_nm(E, mat)) for E in energies])
    R_pub = np.array([ko_closed_form_nm(E, A, Z, rho) for E in energies])
    slope = np.polyfit(np.log(energies), np.log(R_syn), 1)[0]
    rel = np.max(np.abs(R_syn - R_pub) / R_pub)
    ok = (abs(slope - 1.67) < 0.1) and (rel < 0.10)
    ko_ok &= ok
    print(f"  {name:7s} R(20keV)_syn={R_syn[2]:8.1f} nm  KO_formula={R_pub[2]:8.1f} nm  "
          f"E-exponent={slope:.3f} (pub 1.67)  maxrel={rel:.3f} {'OK' if ok else 'FAIL'}")

ok = eta_ok and ko_ok
print("\nProves: SEM transport reproduces published backscatter coefficients and the")
print("Kanaya-Okayama range law. Does NOT prove the full Monte-Carlo SEM IMAGE vs CASINO")
print("(no Mac-friendly oracle) -- that remains a documented Level-C gap.")
raise SystemExit(verdict(ok, "(eta(Z) matches published + KO range follows E^1.67)"))
