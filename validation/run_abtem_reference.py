"""
Run abTEM multislice on a small test specimen and save its outputs to a .npz file.
The Syniscopy validator loads that file and compares PhysicalMultisliceTEMBackend
on the same projected potential, isolating the multislice algorithm from
atomic-potential parameterization differences.

REQUIREMENTS: Python 3.11+ (abtem 1.x imports typing.Self), abtem, ase.
    pip install abtem ase

RUN from the repo root:
    python validation/run_abtem_reference.py

OUTPUT:
    validation/abtem_reference.npz

If anything errors, paste the FULL stdout+stderr back — the abtem 1.x API can vary
slightly by version and the extraction calls below may need a one-line tweak. The
specimen is tiny on purpose so it runs fast and we can iterate.
"""
import sys, numpy as np

print("python:", sys.version.split()[0])
if sys.version_info < (3, 11):
    print("WARNING: abtem 1.x needs Python 3.11+. If import fails, that's why.")

import abtem
from ase import Atoms
print("abtem:", getattr(abtem, "__version__", "unknown"))

# ---------------- test specimen: a few heavy + light atoms in a box ----------------
A = 20.0  # cubic cell edge, Angstrom
atoms = Atoms(
    "Au3C",
    positions=[(8, 10, 5), (12, 10, 8), (10, 12, 11), (10, 8, 9)],
    cell=(A, A, A),
    pbc=True,
)

energy_eV        = 300e3     # 300 kV
sampling_A       = 0.10      # Angstrom / pixel
slice_thick_A    = 2.0       # Angstrom per multislice slice
defocus_A        = 540.0     # ~Scherzer at 300 kV, Cs=1 mm (54 nm)
Cs_A             = 1.0e7     # 1 mm in Angstrom

# ---------------- abTEM potential, exit wave, CTF image ----------------
potential = abtem.Potential(
    atoms, sampling=sampling_A, slice_thickness=slice_thick_A,
    parametrization="kirkland", projection="infinite",
)

# projected potential per slice (eV*Angstrom): the SHARED input for the comparison
pot_built = potential.build().compute()
pot_slices = np.asarray(pot_built.array)            # (n_slices, gy, gx)

wave = abtem.PlaneWave(energy=energy_eV, sampling=sampling_A)
exit_wave = wave.multislice(potential)
try:
    exit_wave = exit_wave.compute()
except Exception:
    pass
exit_arr = np.asarray(exit_wave.array)              # complex (gy, gx)

ctf = abtem.CTF(energy=energy_eV, defocus=defocus_A, Cs=Cs_A)
image = exit_wave.apply_ctf(ctf).intensity()
try:
    image = image.compute()
except Exception:
    pass
image_arr = np.asarray(image.array)                 # real (gy, gx)

# interaction parameter + wavelength in abTEM's own convention
try:
    from abtem.core.energy import energy2wavelength, energy2sigma
    wavelength_A = float(energy2wavelength(energy_eV))
    sigma_rad_per_eV_A = float(energy2sigma(energy_eV))
except Exception as e:
    print("note: could not import energy2sigma/energy2wavelength:", e)
    wavelength_A = float("nan"); sigma_rad_per_eV_A = float("nan")

out = "validation/abtem_reference.npz"
np.savez(
    out,
    pot_slices=pot_slices,
    exit_wave=exit_arr,
    image=image_arr,
    sigma_rad_per_eV_A=sigma_rad_per_eV_A,
    wavelength_A=wavelength_A,
    energy_eV=energy_eV,
    sampling_A=sampling_A,
    slice_thickness_A=slice_thick_A,
    defocus_A=defocus_A,
    Cs_A=Cs_A,
)
print("SAVED:", out)
print("  pot_slices:", pot_slices.shape, " exit_wave:", exit_arr.shape, " image:", image_arr.shape)
print("  sigma(rad/eV/A):", sigma_rad_per_eV_A, " wavelength(A):", wavelength_A)
print("Done. Run validation/tem_multislice_validation.py to compare against this fixture.")
