import sys, numpy as np; sys.path.insert(0,"codebase")
from thinfilm import normal_incidence_thinfilm_reflection
from substrate import material_from_name
from imaging_models.electron_constants import (
    electron_wavelength_m, scherzer_defocus_m, electron_interaction_parameter_rad_per_V_nm)

lam=550.0
n0=complex(material_from_name("water").n_complex(lam))
ns=complex(material_from_name("glass").n_complex(lam))
print("=== Thin-film / Fresnel ===")
# (1) no layers -> exact Fresnel
r_syn=normal_incidence_thinfilm_reflection("water","glass",None,lam)
r_fresnel=(n0-ns)/(n0+ns)
print(f"  no-layer: r_syn={r_syn:.8f}  Fresnel=(n0-ns)/(n0+ns)={r_fresnel:.8f}  |diff|={abs(r_syn-r_fresnel):.2e}")
print(f"           (n_water={n0.real:.4f}, n_glass={ns.real:.4f})")

# (2) single dielectric layer vs independent tmm package (same indices)
try:
    import tmm
    n_layer=complex(material_from_name("silica").n_complex(lam))
    d=120.0
    r_syn2=normal_incidence_thinfilm_reflection("water","glass",[{"n_complex":{"real":n_layer.real,"imag":n_layer.imag},"thickness_nm":d}],lam)
    r_tmm=tmm.coh_tmm('s',[n0,n_layer,ns],[np.inf,d,np.inf],0.0,lam)['r']
    print(f"  1-layer vs tmm: r_syn={r_syn2:.8f}  r_tmm={r_tmm:.8f}  |diff|={abs(r_syn2-r_tmm):.2e}")
    print(f"  |r|^2 reflectance: syn={abs(r_syn2)**2:.6f} tmm={abs(r_tmm)**2:.6f}")
except Exception as e:
    print("  tmm compare skipped:", e)

print("\n=== Electron constants vs published ===")
pub_wl_pm={100:3.7014, 200:2.5079, 300:1.9687}  # relativistic e- wavelength, picometres
for kV,wl_pm in pub_wl_pm.items():
    w=electron_wavelength_m(kV)*1e12
    print(f"  {kV} kV: lambda={w:.4f} pm  published={wl_pm} pm  rel_err={abs(w/wl_pm-1)*100:.3f}%")
# Scherzer vs sqrt(1.5 lam Cs)
for kV,Cs_mm in [(200,1.0),(300,2.0)]:
    s=scherzer_defocus_m(kV,Cs_mm)
    pred=np.sqrt(1.5*electron_wavelength_m(kV)*Cs_mm*1e-3)
    print(f"  Scherzer {kV}kV Cs={Cs_mm}mm: {s*1e9:.3f} nm  vs sqrt(1.5*lam*Cs)={pred*1e9:.3f} nm  |diff|={abs(s-pred):.2e}")
