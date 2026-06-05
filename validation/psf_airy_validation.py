"""Level-B(-) check: in the Rayleigh/low-NA/no-aberration limit, Syniscopy's
scalar PSF |E|^2 should approximate the Airy pattern with first dark ring at
r = 0.61 * lambda / NA. Caveats: pupil carries Mie cos(theta) obliquity and the
ASF is force-radially-symmetrized, so an approximate (not exact) match is the
honest expectation. No production-code changes."""
import sys, numpy as np; sys.path.insert(0,"codebase")
from copy import deepcopy
from config import PARAMS
from optics import compute_complex_psf_stack

NA, n_med, lam = 0.30, 1.0, 550.0     # low NA -> minimal obliquity deviation
pix, osf, psamp = 100.0, 2, 256
p=deepcopy(PARAMS)
p.update({
    "numerical_aperture":NA, "refractive_index_medium":n_med, "wavelength_nm":lam,
    "pupil_samples":psamp, "psf_oversampling_factor":osf, "pixel_size_nm":pix,
    "apodization_factor":0.0, "spherical_aberration_strength":0.0,
    "random_aberration_strength":0.0, "optical_field_backend":"scalar_paraxial",
    "coverslip_correction_enabled":False,
})
interp=compute_complex_psf_stack(p, particle_diameter_nm=10.0,
                                 particle_refractive_index=complex(1.59,0.0),
                                 z_values_nm=np.array([0.0]))
E=np.asarray(interp(0.0)); I=np.abs(E)**2; I/=I.max()
canvas_pitch=pix/osf
c=psamp//2
# radial profile
yy,xx=np.indices(I.shape); r=np.sqrt((xx-c)**2+(yy-c)**2)
rb=np.arange(0, int(r.max()))
prof=np.array([I[(r>=k)&(r<k+1)].mean() for k in rb])
# first local minimum after the central peak
first_min=None
for k in range(2,len(prof)-1):
    if prof[k]<prof[k-1] and prof[k]<=prof[k+1] and prof[k]<0.2:
        first_min=k; break
r_first_zero_nm = first_min*canvas_pitch if first_min else float('nan')
airy = 0.61*lam/NA
print(f"NA={NA} n_med={n_med} lambda={lam}nm  canvas_pitch={canvas_pitch}nm")
print(f"  measured first dark ring  ~ {r_first_zero_nm:.1f} nm  (bin {first_min})")
print(f"  Airy prediction 0.61*lam/NA = {airy:.1f} nm")
print(f"  rel error = {abs(r_first_zero_nm/airy-1)*100:.1f}%")
print(f"  PSF radially symmetric? max-min over ring at r~5px: {np.ptp(I[(r>=5)&(r<6)]):.2e}")
