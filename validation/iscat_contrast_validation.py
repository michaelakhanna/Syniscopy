"""Level-B validation: InterferometricImagingModel.compute_per_particle_contrast
implements the published iSCAT contrast identity
    C = |E_ref + E_sca|^2 - |E_ref|^2 = 2*Re(conj(E_ref)*E_sca) + |E_sca|^2
and applies the reference amplitude/phase scale consistently. No code changes."""
import sys, numpy as np; sys.path.insert(0,"codebase")
from copy import deepcopy
from config import PARAMS
from imaging_models.interferometric import InterferometricImagingModel

rng=np.random.default_rng(0)
H=W=48
E_sca = (rng.standard_normal((H,W))+1j*rng.standard_normal((H,W)))*0.05
bg    = (rng.standard_normal((H,W))+1j*rng.standard_normal((H,W)))*1.0  # E_ref pre-scale

def params(ref_model="renderer", phase=0.0, amp=1.0, coll="scalar"):
    p=deepcopy(PARAMS)
    p.update({"imaging_model":"interferometric","reference_field_amplitude":1.0,
              "iscat_reference_model":ref_model,"iscat_reference_phase_rad":phase,
              "iscat_reference_amplitude_scale":amp,"iscat_collection_model":coll})
    return p

print("=== iSCAT contrast identity: C == 2Re(conj(E_ref)*E_sca) + |E_sca|^2 ===")
for (phase,amp) in [(0.0,1.0),(np.pi/3,1.0),(0.0,1.3),(np.pi/2,0.7)]:
    p=params(phase=phase,amp=amp)
    m=InterferometricImagingModel(p)
    C_model=m.compute_per_particle_contrast(E_sca, bg, p)
    # reconstruct E_ref exactly as the model does (renderer model: scale = amp*e^{i phase})
    E_ref = bg*(amp*np.exp(1j*phase))
    C_identity = 2.0*np.real(np.conj(E_ref)*E_sca) + np.abs(E_sca)**2
    err=np.max(np.abs(C_model-C_identity))
    print(f"  phase={phase:.3f} amp={amp}: max|C_model - identity| = {err:.3e}  "
          f"(contrast range {C_model.min():.3e}..{C_model.max():.3e})")

print("\n=== sanity: weak-scatterer limit dominated by 2Re cross-term ===")
p=params(); m=InterferometricImagingModel(p)
C=m.compute_per_particle_contrast(E_sca, bg, p)
cross=2.0*np.real(np.conj(bg)*E_sca); quad=np.abs(E_sca)**2
print(f"  ||2Re(cross)||={np.linalg.norm(cross):.3e}  |||E_sca|^2||={np.linalg.norm(quad):.3e}  "
      f"ratio quad/cross={np.linalg.norm(quad)/np.linalg.norm(cross):.3e} (should be <<1 for weak scatterer)")
